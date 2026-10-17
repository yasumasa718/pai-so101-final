"""SO-101 の指令値が設定範囲内かを、実機に接続せず検査する。

6 関節分の指令の辞書を、Follower のキャリブレーションから求めた範囲と照合する。
モーターへの指令送信は行わず、衝突・負荷・再生時の安全を保証するものでもない。

LeRobot 0.6.1、SO-101 の use_degrees=True を前提にする。
F1～F5 は度、F6 は 0～100 に換算した値を使う。
キャリブレーションが現在の機器に対して有効であることは、別途確認が必要。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

MAX_TICKS = 4095
MOTOR_IDS: dict[str, int] = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


@dataclass(frozen=True)
class JointBound:
    """キャリブレーションから求めた設定上の範囲。実機の安全を保証する限界値ではない。"""

    action_key: str
    minimum: float
    maximum: float
    unit: str


@dataclass(frozen=True)
class ActionGuardResult:
    """6 関節分の指令を検査した結果。"""

    accepted: bool
    reasons: tuple[str, ...]
    # 範囲の検査に通った場合だけ、数値の形式をそろえてコピーした指令を返す。
    # この値が得られても、実機を動かしてよいという意味ではない。
    checked_action: dict[str, float] | None


@dataclass(frozen=True)
class ActionGuard:
    """LeRobot の SO-101 指令を、実機や設定を書き換えずに検査する。"""

    bounds: tuple[JointBound, ...]

    @classmethod
    def from_calibration(
        cls,
        path: Path,
        *,
        margin_deg: float = 0.0,
        margin_gripper: float = 0.0,
    ) -> "ActionGuard":
        """Follower の JSON から検査範囲を作る。元のファイルは変更しない。"""
        return cls.from_calibration_bytes(
            path.read_bytes(), margin_deg=margin_deg, margin_gripper=margin_gripper,
        )

    @classmethod
    def from_calibration_bytes(
        cls,
        payload: bytes,
        *,
        margin_deg: float = 0.0,
        margin_gripper: float = 0.0,
    ) -> "ActionGuard":
        """開始前の検査で確認したものと同じファイル内容から、検査範囲を作る。"""
        for name, margin in (
            ("margin_deg", margin_deg),
            ("margin_gripper", margin_gripper),
        ):
            if (
                type(margin) not in (int, float)
                or not math.isfinite(margin)
                or margin < 0
            ):
                raise ValueError(f"{name} must be finite and nonnegative")

        data = json.loads(payload)
        if not isinstance(data, dict) or set(data) != set(MOTOR_IDS):
            raise ValueError("Follower calibration must contain exactly six SO-101 motors")

        bounds: list[JointBound] = []
        for motor, motor_id in MOTOR_IDS.items():
            cal = data[motor]
            if not isinstance(cal, dict):
                raise ValueError(f"{motor}: invalid calibration entry")
            required = ("id", "drive_mode", "homing_offset", "range_min", "range_max")
            if any(type(cal.get(field)) is not int for field in required):
                raise ValueError(f"{motor}: missing/non-integer calibration field")
            if cal["id"] != motor_id or cal["drive_mode"] not in (0, 1):
                raise ValueError(f"{motor}: unexpected motor id/drive mode")
            low, high = cal["range_min"], cal["range_max"]
            if not 0 <= low < high <= MAX_TICKS:
                raise ValueError(f"{motor}: invalid calibration range")

            if motor == "gripper":
                # 爪だけは角度ではなく、校正区間の閉じ側〜開き側を0〜100で表す。
                minimum, maximum, margin = 0.0, 100.0, float(margin_gripper)
                unit = "normalized_0_100"
            else:
                # 腕の0度は校正区間の中点。生位置には機器側のhoming_offsetが反映済みなので、
                # ここでオフセットを加算し直すことはしない。
                midpoint = (low + high) / 2.0
                minimum = (low - midpoint) * 360.0 / MAX_TICKS
                maximum = (high - midpoint) * 360.0 / MAX_TICKS
                margin = float(margin_deg)
                unit = "degree"

            minimum += margin
            maximum -= margin
            if minimum >= maximum:
                raise ValueError(f"{motor}: margin leaves no valid interval")
            bounds.append(JointBound(f"{motor}.pos", minimum, maximum, unit))
        return cls(tuple(bounds))

    def check(self, action: Mapping[str, object]) -> ActionGuardResult:
        """関節の不足・余分な項目・不正な値・範囲外の値があれば拒否する。

        範囲内に収めるための自動補正はしない。目標値を変えると動きも変わるため。
        """
        if not isinstance(action, Mapping):
            return ActionGuardResult(False, ("action: expected mapping",), None)

        expected = {bound.action_key for bound in self.bounds}
        received = set(action)
        reasons: list[str] = []
        missing = expected - received
        unexpected = received - expected
        if missing:
            reasons.append(f"missing keys: {sorted(missing)}")
        if unexpected:
            reasons.append(f"unexpected keys: {sorted(unexpected, key=str)}")

        normalized: dict[str, float] = {}
        for bound in self.bounds:
            if bound.action_key not in action:
                continue
            raw = action[bound.action_key]
            # boolはintの派生型だが、True/Falseを位置指令として受け入れないため型を厳密に見る。
            if type(raw) not in (int, float) or not math.isfinite(raw):
                reasons.append(f"{bound.action_key}: invalid numeric value {raw!r}")
                continue
            value = float(raw)
            if not bound.minimum <= value <= bound.maximum:
                reasons.append(
                    f"{bound.action_key}: {value:.4f} outside configured interval "
                    f"[{bound.minimum:.4f}, {bound.maximum:.4f}] {bound.unit}"
                )
            else:
                normalized[bound.action_key] = value

        if reasons:
            return ActionGuardResult(False, tuple(reasons), None)
        return ActionGuardResult(True, (), normalized)
