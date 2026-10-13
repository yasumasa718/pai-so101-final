"""SO-101 の指令範囲をオフラインで検証する。シリアル通信やカメラは使わない。"""

import json
import math
from pathlib import Path

import pytest

from src.control.action_guard import ActionGuard, MOTOR_IDS


@pytest.fixture
def calibration_file(tmp_path: Path) -> Path:
    # F2/F4 は、2026-10-13 にユーザーが報告した範囲に合わせる。
    limits = {
        "shoulder_pan": (792, 3306),
        "shoulder_lift": (826, 3209),
        "elbow_flex": (846, 3047),
        "wrist_flex": (800, 3140),
        "wrist_roll": (0, 4095),
        "gripper": (2035, 3481),
    }
    data = {
        motor: {
            "id": motor_id,
            "drive_mode": 0,
            "homing_offset": 0,
            "range_min": limits[motor][0],
            "range_max": limits[motor][1],
        }
        for motor, motor_id in MOTOR_IDS.items()
    }
    path = tmp_path / "my_follower.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def center_action() -> dict[str, float]:
    return {f"{motor}.pos": 50.0 if motor == "gripper" else 0.0 for motor in MOTOR_IDS}


def test_accepts_complete_action(calibration_file: Path) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    result = guard.check(center_action())
    assert result.accepted
    assert result.checked_action == center_action()


def test_rejects_f4_outside_range(calibration_file: Path) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["wrist_flex.pos"] = -105.0
    result = guard.check(action)
    assert not result.accepted
    assert result.checked_action is None
    assert "wrist_flex.pos" in result.reasons[0]


def test_rejects_f2_outside_range(calibration_file: Path) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["shoulder_lift.pos"] = 110.0
    assert not guard.check(action).accepted


def test_rejects_gripper_over_100(calibration_file: Path) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["gripper.pos"] = 100.1
    assert not guard.check(action).accepted


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, "42"])
def test_rejects_invalid_number(calibration_file: Path, bad: object) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["elbow_flex.pos"] = bad
    assert not guard.check(action).accepted


def test_rejects_missing_key(calibration_file: Path) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    del action["shoulder_pan.pos"]
    assert not guard.check(action).accepted


def test_rejects_extra_key(calibration_file: Path) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["unknown.pos"] = 0.0
    assert not guard.check(action).accepted


def test_margin_narrows_range(calibration_file: Path) -> None:
    without = ActionGuard.from_calibration(calibration_file)
    with_margin = ActionGuard.from_calibration(calibration_file, margin_deg=2.0)
    original = next(b for b in without.bounds if b.action_key == "wrist_flex.pos")
    adjusted = next(b for b in with_margin.bounds if b.action_key == "wrist_flex.pos")
    assert adjusted.minimum == pytest.approx(original.minimum + 2.0)
    assert adjusted.maximum == pytest.approx(original.maximum - 2.0)
    action = center_action()
    action["wrist_flex.pos"] = original.minimum + 1.0
    assert without.check(action).accepted
    assert not with_margin.check(action).accepted


@pytest.mark.parametrize("margin", [-1.0, float("nan"), float("inf"), True])
def test_rejects_invalid_margin(calibration_file: Path, margin: float) -> None:
    with pytest.raises(ValueError):
        ActionGuard.from_calibration(calibration_file, margin_deg=margin)


def test_does_not_modify_action(calibration_file: Path) -> None:
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["wrist_flex.pos"] = 120.0
    snapshot = action.copy()
    guard.check(action)
    assert action == snapshot


def test_does_not_modify_calibration(calibration_file: Path) -> None:
    before = calibration_file.read_bytes()
    ActionGuard.from_calibration(calibration_file)
    assert calibration_file.read_bytes() == before
