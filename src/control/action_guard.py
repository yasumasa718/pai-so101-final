"""Offline SO-101 action-range gate (no hardware I/O).

This module only checks that a complete action dictionary lies inside
Follower calibration-derived *configured* position bounds. It neither
commands motors nor certifies collision safety, motor load, or replay safety.

LeRobot 0.6.1 with SO-101 use_degrees=True is assumed: F1-F5 use degrees,
F6 uses normalized 0..100. Calibration values are assumed current and valid.
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
    """Calibration-derived configured limits, not physical safety limits."""

    action_key: str
    minimum: float
    maximum: float
    unit: str


@dataclass(frozen=True)
class ActionGuardResult:
    """Result of a complete action dictionary check."""

    accepted: bool
    reasons: tuple[str, ...]
    # A copied, normalized action is returned only when the range checks pass.
    # Its presence does NOT authorize hardware execution.
    checked_action: dict[str, float] | None


@dataclass(frozen=True)
class ActionGuard:
    """Read-only offline gate for LeRobot SO-101 action dictionaries."""

    bounds: tuple[JointBound, ...]

    @classmethod
    def from_calibration(
        cls,
        path: Path,
        *,
        margin_deg: float = 0.0,
        margin_gripper: float = 0.0,
    ) -> "ActionGuard":
        """Build limits from Follower JSON. Never alters the source file."""
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

        with path.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
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
                minimum, maximum, margin = 0.0, 100.0, float(margin_gripper)
                unit = "normalized_0_100"
            else:
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
        """Fail closed on missing, extra, malformed or out-of-range entries.

        No clipping is performed: adjusting a target can change the motion.
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
