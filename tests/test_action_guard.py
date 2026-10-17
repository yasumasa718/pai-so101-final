"""6軸の指令範囲・入力形式・内側余白を検証する。実機やカメラは使わない。

一時的な校正JSONからActionGuardを作り、正常な中央姿勢の1項目だけを変えて
拒否条件を切り分ける。検査で入力指令や校正を書き換えないことも確認する。
"""

import json
import math
from pathlib import Path

import pytest

from src.control.action_guard import ActionGuard, MOTOR_IDS


@pytest.fixture
def calibration_file(tmp_path: Path) -> Path:
    # F2/F4 は、2026-10-13 にユーザーが報告した範囲に合わせる。
    """軸ごとに異なる範囲を持つ6軸の校正JSONを一時フォルダーに作るfixture。"""
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
    """腕を0度、爪を50ポイントにした、校正範囲内の基準指令を毎回新しく返す。"""
    return {f"{motor}.pos": 50.0 if motor == "gripper" else 0.0 for motor in MOTOR_IDS}


def test_accepts_complete_action(calibration_file: Path) -> None:
    """6軸がそろった範囲内指令は受理され、検査済み指令として返ることを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    result = guard.check(center_action())
    assert result.accepted
    assert result.checked_action == center_action()


def test_rejects_f4_outside_range(calibration_file: Path) -> None:
    """F4の下限超過を拒否し、検査済み指令を返さず理由に軸名を含めることを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["wrist_flex.pos"] = -105.0
    result = guard.check(action)
    assert not result.accepted
    assert result.checked_action is None
    assert "wrist_flex.pos" in result.reasons[0]


def test_rejects_f2_outside_range(calibration_file: Path) -> None:
    """F2の上限を越える110度の指令を拒否することを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["shoulder_lift.pos"] = 110.0
    assert not guard.check(action).accepted


def test_rejects_gripper_over_100(calibration_file: Path) -> None:
    """腕と異なる0〜100の単位を使う爪で、100.1ポイントを拒否することを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["gripper.pos"] = 100.1
    assert not guard.check(action).accepted


@pytest.mark.parametrize("bad", [None, float("nan"), float("inf"), True, "42"])
def test_rejects_invalid_number(calibration_file: Path, bad: object) -> None:
    """欠損・NaN・無限大・bool・文字列を数値指令として受け入れないことを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["elbow_flex.pos"] = bad
    assert not guard.check(action).accepted


def test_rejects_missing_key(calibration_file: Path) -> None:
    """6軸のうち1軸でも欠けていれば、残りの値が正常でも拒否することを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    del action["shoulder_pan.pos"]
    assert not guard.check(action).accepted


def test_rejects_extra_key(calibration_file: Path) -> None:
    """未知の関節キーを黙って無視せず、指令全体を拒否することを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["unknown.pos"] = 0.0
    assert not guard.check(action).accepted


def test_margin_narrows_range(calibration_file: Path) -> None:
    """余白2度で上下限が内側へ狭まり、元は範囲内だった端点付近の指令を拒否することを確認する。"""
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
    """負数・非有限値・boolを範囲の余白として指定できないことを確認する。"""
    with pytest.raises(ValueError):
        ActionGuard.from_calibration(calibration_file, margin_deg=margin)


def test_does_not_modify_action(calibration_file: Path) -> None:
    """範囲外の指令を検査しても、自動で範囲内へ切り詰めず入力辞書を保持することを確認する。"""
    guard = ActionGuard.from_calibration(calibration_file)
    action = center_action()
    action["wrist_flex.pos"] = 120.0
    snapshot = action.copy()
    guard.check(action)
    assert action == snapshot


def test_does_not_modify_calibration(calibration_file: Path) -> None:
    """検査器の構築前後で校正JSONのバイト列が一致することを確認する。"""
    before = calibration_file.read_bytes()
    ActionGuard.from_calibration(calibration_file)
    assert calibration_file.read_bytes() == before
