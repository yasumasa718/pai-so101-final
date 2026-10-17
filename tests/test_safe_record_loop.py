"""1フレームの検査→加工→送信→記録の順序と、FAULT後の停止を検証する。

FakeRobotとリスト保存先を使い、実機・カメラ・LeRobotDatasetは使わない。
送信前の拒否、送信結果不明、送信後の記録失敗を区別することが主な確認点。
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
    """腕0度・爪50ポイントの、正常な6軸指令を作る。"""
    return {f"{name}.pos": (50.0 if name == "gripper" else 0.0) for name in MOTOR_IDS}


@pytest.fixture
def guard(tmp_path: Path) -> ActionGuard:
    """一時的な校正JSONから、各テスト共通の範囲検査器を作る。"""
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
    """送信履歴を残し、爪の制限・送信例外・範囲外戻り値を再現する送信先。"""
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
    """(指令, 観測)から指令だけをコピーして返す、値を変えない加工処理。"""
    action, _obs = payload
    return dict(action)


@pytest.fixture
def context(guard: ActionGuard):
    """未開始のループ・模擬送信先・保存リストを一組で作る。"""
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
    """start前のフレーム処理は状態エラーになり、送信も保存もしないことを確認する。"""
    controller, robot, saved = context
    with pytest.raises(RecordStateError):
        controller.process_frame(center_action(), {})
    assert controller.state is RecordState.READY
    assert robot.sent_calls == saved == []


def test_valid_frame_is_sent_and_saved(context):
    """正常な1フレームを1回送信・保存し、0始まりの番号と正常終了状態が得られることを確認する。"""
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
    """要求した爪50ではなく、送信側が制限して返した47を保存し、処理結果でも両者を区別することを確認する。"""
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
    """元の入力指令の範囲外・NaNを最初の検査で拒否し、未送信としてFAULTを保持することを確認する。"""
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
    """欠けた軸を補完せず拒否し、送信・保存を行わないことを確認する。"""
    controller, robot, saved = context
    controller.start()
    action = center_action()
    del action["wrist_flex.pos"]
    with pytest.raises(RecordFault):
        controller.process_frame(action, {})
    assert not robot.sent_calls and not saved


@pytest.mark.parametrize("stage", ["teleop", "robot"])
def test_processor_modified_command_is_rechecked(guard, stage):
    """手動操作側・ロボット側のどちらの加工で範囲外になっても、送信前に再検査して拒否することを確認する。"""
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
    """送信中の例外は届かなかったと断定せずUNCERTAINとし、保存や自動切断を行わないことを確認する。"""
    robot = FakeRobot(fail_during_send=True)
    saved: list[dict[str, float]] = []
    loop = SafeRecordLoop(guard, identity, identity, robot, lambda action: saved.append(dict(action)))
    loop.start()
    with pytest.raises(RecordFault) as caught:
        loop.process_frame(center_action(), {})
    assert caught.value.fault.stage == "send_action"
    assert caught.value.fault.send_status is SendStatus.UNCERTAIN
    assert len(robot.sent_calls) == 1  # 例外が起きても、指令がモーターに届いた可能性はある。
    assert saved == []
    assert loop.state is RecordState.FAULT
    assert robot.disconnect_calls == 0


def test_bad_send_return_latches_fault_after_send_without_saving(guard):
    """送信戻り値が範囲外なら、送信済み1回という事実を保持しつつ保存を防ぐことを確認する。"""
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
    """送信後の保存失敗ではフレーム数を増やさず、次回の処理や再送を拒否することを確認する。"""
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
    """FAULT後は処理・再開始・正常終了を拒否し、ループ自身では切断しないことを確認する。"""
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
    """送信中のKeyboardInterruptも送信結果不明として保持し、割込みを再送せず呼出し側へ伝えることを確認する。"""
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
    """実際のLeRobot加工パイプラインを模擬送信先へつなぎ、正常な指令を1フレーム保存できることを確認する。"""
    pytest.importorskip("lerobot")
    from src.control.guarded_record_processor import make_guarded_teleop_pipeline
    from lerobot.processor import make_default_robot_action_processor

    # テスト用の共通準備で検証したキャリブレーションの範囲を再利用する。
    # LeRobot の処理を組み立てる際は、検査オブジェクトではなく JSON ファイルを渡す必要がある。
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
