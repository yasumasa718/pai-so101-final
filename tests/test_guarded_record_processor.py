"""指令を順番に加工・検査する処理をオフラインで試す。実機・シリアル通信・カメラは使わない。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from lerobot.processor import make_default_robot_action_processor

from src.control.action_guard import MOTOR_IDS
from src.control.guarded_record_processor import (
    GuardRejectedError,
    GuardedTeleopActionStep,
    make_guarded_teleop_pipeline,
)


@pytest.fixture
def calibration_path(tmp_path: Path) -> Path:
    ranges = {
        "shoulder_pan": (792, 3306),
        "shoulder_lift": (826, 3209),
        "elbow_flex": (846, 3047),
        "wrist_flex": (800, 3140),
        "wrist_roll": (0, 4095),
        "gripper": (2035, 3481),
    }
    data = {
        name: {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": ranges[name][0],
            "range_max": ranges[name][1],
        }
        for name, motor_id in MOTOR_IDS.items()
    }
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def center_action() -> dict[str, float]:
    return {
        f"{name}.pos": 50.0 if name == "gripper" else 0.0
        for name in MOTOR_IDS
    }


def test_complete_action_passes_teleop_pipeline(calibration_path: Path) -> None:
    processor = make_guarded_teleop_pipeline(calibration_path)
    input_action = center_action()
    processed = processor((input_action, {}))
    assert processed == input_action
    assert processed is not input_action


def test_f4_violation_rejected_before_output(calibration_path: Path) -> None:
    processor = make_guarded_teleop_pipeline(calibration_path)
    action = center_action()
    action["wrist_flex.pos"] = -110.0
    with pytest.raises(GuardRejectedError, match="wrist_flex.pos"):
        processor((action, {}))


def test_saved_action_and_robot_action_identity_path_match(calibration_path: Path) -> None:
    """record_loop 内の 2 段階の加工を再現する。実機への送信は行わない。"""
    teleop_action_processor = make_guarded_teleop_pipeline(calibration_path)
    robot_action_processor = make_default_robot_action_processor()
    original = center_action()

    action_values = teleop_action_processor((original, {}))
    robot_action_to_send = robot_action_processor((action_values, {}))

    assert action_values == robot_action_to_send == original


def test_step_does_not_mutate_input(calibration_path: Path) -> None:
    processor = make_guarded_teleop_pipeline(calibration_path)
    action = center_action()
    old = dict(action)
    _ = processor((action, {}))
    assert action == old
    assert isinstance(processor.steps[0], GuardedTeleopActionStep)
