"""Fake-robot only tests for the guarded one-frame recording boundary.

No robot, serial bus, camera, real LeRobotDataset, or torque I/O is accessed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from src.control.action_guard import ActionGuard, MOTOR_IDS
from src.control.safe_record_loop import (
    RecordFault,
    RecordState,
    RecordStateError,
    SafeRecordLoop,
    SendStatus,
)


def center_action() -> dict[str, float]:
    return {f"{name}.pos": (50.0 if name == "gripper" else 0.0) for name in MOTOR_IDS}


@pytest.fixture
def guard(tmp_path: Path) -> ActionGuard:
    ranges = {
        "shoulder_pan": (792, 3306),
        "shoulder_lift": (826, 3209),
        "elbow_flex": (846, 3047),
        "wrist_flex": (800, 3140),
        "wrist_roll": (0, 4095),
        "gripper": (2035, 3481),
    }
    cal = {
        name: {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": ranges[name][0],
            "range_max": ranges[name][1],
        }
        for name, motor_id in MOTOR_IDS.items()
    }
    path = tmp_path / "follower.json"
    path.write_text(json.dumps(cal), encoding="utf-8")
    return ActionGuard.from_calibration(path)


@dataclass
class FakeRobot:
    sent_calls: list[dict[str, float]] = field(default_factory=list)
    disconnect_calls: int = 0
    clip_gripper_to: float | None = None
    fail_during_send: bool = False
    return_out_of_bounds: bool = False

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.sent_calls.append(dict(action))
        if self.fail_during_send:
            raise OSError("simulated serial failure after send attempt")
        result = dict(action)
        if self.clip_gripper_to is not None:
            result["gripper.pos"] = self.clip_gripper_to
        if self.return_out_of_bounds:
            result["wrist_flex.pos"] = 999.0
        return result

    def disconnect(self) -> None:
        self.disconnect_calls += 1


def identity(payload: tuple[dict[str, float], dict[str, Any]]) -> dict[str, float]:
    action, _obs = payload
    return dict(action)


@pytest.fixture
def context(guard: ActionGuard):
    robot = FakeRobot()
    saved: list[dict[str, float]] = []
    controller = SafeRecordLoop(
        guard=guard,
        teleop_action_processor=identity,
        robot_action_processor=identity,
        robot=robot,
        frame_writer=lambda action: saved.append(dict(action)),
    )
    return controller, robot, saved


def test_not_started_cannot_send(context):
    controller, robot, saved = context
    with pytest.raises(RecordStateError):
        controller.process_frame(center_action(), {})
    assert controller.state is RecordState.READY
    assert robot.sent_calls == saved == []


def test_valid_frame_is_sent_and_saved(context):
    controller, robot, saved = context
    controller.start()
    result = controller.process_frame(center_action(), {})
    assert result.frame_index == 0
    assert result.sent_action == center_action()
    assert robot.sent_calls == saved == [center_action()]
    assert controller.frames_written == 1
    controller.finish()
    assert controller.state is RecordState.FINISHED


def test_clip_records_returned_command_not_requested(guard):
    robot = FakeRobot(clip_gripper_to=47.0)
    saved: list[dict[str, float]] = []
    loop = SafeRecordLoop(guard, identity, identity, robot, lambda action: saved.append(dict(action)))
    loop.start()
    result = loop.process_frame(center_action(), {})
    assert robot.sent_calls[0]["gripper.pos"] == 50.0
    assert saved[0]["gripper.pos"] == 47.0
    assert result.requested_action["gripper.pos"] == 50.0
    assert result.sent_action["gripper.pos"] == 47.0


@pytest.mark.parametrize("joint, bad", [
    ("wrist_flex.pos", -110.0),
    ("shoulder_lift.pos", 110.0),
    ("elbow_flex.pos", float("nan")),
])
def test_raw_violation_rejected_before_any_send(context, joint, bad):
    controller, robot, saved = context
    controller.start()
    action = center_action()
    action[joint] = bad
    with pytest.raises(RecordFault) as caught:
        controller.process_frame(action, {})
    assert caught.value.fault.stage == "raw_guard"
    assert caught.value.fault.send_status is SendStatus.NOT_ATTEMPTED
    assert controller.state is RecordState.FAULT
    assert controller.frames_written == 0
    assert not robot.sent_calls and not saved


def test_missing_joint_fails_closed(context):
    controller, robot, saved = context
    controller.start()
    action = center_action()
    del action["wrist_flex.pos"]
    with pytest.raises(RecordFault):
        controller.process_frame(action, {})
    assert not robot.sent_calls and not saved


@pytest.mark.parametrize("stage", ["teleop", "robot"])
def test_processor_modified_command_is_rechecked(guard, stage):
    robot = FakeRobot()
    saved: list[dict[str, float]] = []

    def invalid(payload):
        action, _obs = payload
        result = dict(action)
        result["wrist_flex.pos"] = -110.0
        return result

    loop = SafeRecordLoop(
        guard,
        invalid if stage == "teleop" else identity,
        invalid if stage == "robot" else identity,
        robot,
        lambda action: saved.append(dict(action)),
    )
    loop.start()
    with pytest.raises(RecordFault) as caught:
        loop.process_frame(center_action(), {})
    assert caught.value.fault.stage == ("teleop_guard" if stage == "teleop" else "final_command_guard")
    assert caught.value.fault.send_status is SendStatus.NOT_ATTEMPTED
    assert not robot.sent_calls and not saved


def test_send_failure_marks_uncertain_and_never_saves(guard):
    robot = FakeRobot(fail_during_send=True)
    saved: list[dict[str, float]] = []
    loop = SafeRecordLoop(guard, identity, identity, robot, lambda action: saved.append(dict(action)))
    loop.start()
    with pytest.raises(RecordFault) as caught:
        loop.process_frame(center_action(), {})
    assert caught.value.fault.stage == "send_action"
    assert caught.value.fault.send_status is SendStatus.UNCERTAIN
    assert len(robot.sent_calls) == 1  # it might have reached the motor bus
    assert saved == []
    assert loop.state is RecordState.FAULT
    assert robot.disconnect_calls == 0


def test_bad_send_return_latches_fault_after_send_without_saving(guard):
    robot = FakeRobot(return_out_of_bounds=True)
    saved: list[dict[str, float]] = []
    loop = SafeRecordLoop(guard, identity, identity, robot, lambda action: saved.append(dict(action)))
    loop.start()
    with pytest.raises(RecordFault) as caught:
        loop.process_frame(center_action(), {})
    assert caught.value.fault.stage == "sent_command_guard"
    assert caught.value.fault.send_status is SendStatus.RETURNED
    assert len(robot.sent_calls) == 1
    assert saved == []


def test_writer_failure_after_send_faults_and_does_not_retry(guard):
    robot = FakeRobot()

    def bad_writer(_action):
        raise OSError("simulated dataset error")

    loop = SafeRecordLoop(guard, identity, identity, robot, bad_writer)
    loop.start()
    with pytest.raises(RecordFault) as caught:
        loop.process_frame(center_action(), {})
    assert caught.value.fault.stage == "frame_writer"
    assert caught.value.fault.send_status is SendStatus.RETURNED
    assert loop.frames_written == 0
    assert len(robot.sent_calls) == 1
    with pytest.raises(RecordStateError):
        loop.process_frame(center_action(), {})
    assert len(robot.sent_calls) == 1


def test_fault_is_latched_and_never_disconnects_automatically(context):
    loop, robot, saved = context
    loop.start()
    action = center_action()
    action["wrist_flex.pos"] = -110.0
    with pytest.raises(RecordFault):
        loop.process_frame(action, {})
    for operation in [lambda: loop.process_frame(center_action(), {}), loop.start, loop.finish]:
        with pytest.raises(RecordStateError):
            operation()
    assert loop.state is RecordState.FAULT
    assert robot.sent_calls == saved == []
    assert robot.disconnect_calls == 0


def test_keyboard_interrupt_latches_fault_without_automatic_disconnect(guard):
    class InterruptedRobot(FakeRobot):
        def send_action(self, action):
            self.sent_calls.append(dict(action))
            raise KeyboardInterrupt()

    robot = InterruptedRobot()
    loop = SafeRecordLoop(guard, identity, identity, robot, lambda action: None)
    loop.start()
    with pytest.raises(KeyboardInterrupt):
        loop.process_frame(center_action(), {})
    assert loop.state is RecordState.FAULT
    assert loop.fault is not None
    assert loop.fault.send_status is SendStatus.UNCERTAIN
    assert robot.disconnect_calls == 0
    with pytest.raises(RecordStateError):
        loop.process_frame(center_action(), {})
    assert len(robot.sent_calls) == 1


def test_real_guarded_teleop_pipeline_compatible(guard, tmp_path: Path):
    """Exercise the existing LeRobot Processor in the user's LeRobot environment."""
    pytest.importorskip("lerobot")
    from src.control.guarded_record_processor import make_guarded_teleop_pipeline
    from lerobot.processor import make_default_robot_action_processor

    # Reuse the fully validated calibration bounds from this fixture.
    # LeRobot pipeline construction requires the JSON file, not the guard object.
    payload = {
        name: {
            "id": motor_id, "drive_mode": 0, "homing_offset": 0,
            "range_min": (800 if name == "wrist_flex" else 0),
            "range_max": (3140 if name == "wrist_flex" else 4095),
        }
        for name, motor_id in MOTOR_IDS.items()
    }
    path = tmp_path / "pipeline_calibration.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    robot = FakeRobot()
    saved: list[dict[str, float]] = []
    loop = SafeRecordLoop(
        guard=guard,
        teleop_action_processor=make_guarded_teleop_pipeline(path),
        robot_action_processor=make_default_robot_action_processor(),
        robot=robot,
        frame_writer=lambda action: saved.append(dict(action)),
    )
    loop.start()
    loop.process_frame(center_action(), {})
    assert saved == [center_action()]
