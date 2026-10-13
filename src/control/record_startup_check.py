"""Read-only SO-101 startup consistency check; no hardware or LeRobot imports.

The expected digest must be fixed from an independently reviewed Follower
snapshot BEFORE starting. Never learn/replace it from the current file during
verify(). Margins are explicit policy inputs, not certified safety distances.

This checks local bytes, loaded calibration and config, not motor registers,
bus normalization, connection identity, collisions, or physical stop readiness.
It neither discovers devices nor connects, disconnects, or changes torque.
The check applies only at start; do not mutate config/calibration/guard during
an episode. This is an offline integration component, not hardware clearance.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.control.action_guard import ActionGuard, JointBound, MOTOR_IDS

CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


class StartupCheckError(ValueError):
    """Local startup evidence is missing, invalid or inconsistent."""


@dataclass(frozen=True)
class StartupCheckResult:
    calibration_sha256: str
    margin_deg: float
    margin_gripper: float
    bounds: tuple[JointBound, ...]


@dataclass(frozen=True, kw_only=True)
class RecordStartupCheck:
    calibration_path: Path
    expected_sha256: str
    margin_deg: float
    margin_gripper: float

    def __post_init__(self) -> None:
        if not isinstance(self.calibration_path, Path):
            raise TypeError("calibration_path must be a Path")
        if not isinstance(self.expected_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", self.expected_sha256) is None:
            raise ValueError("expected_sha256 must be a fixed 64-character SHA-256 digest")

    def verify(self, guard: ActionGuard, robot: object) -> StartupCheckResult:
        """Validate an already-created sender's local metadata, without I/O to it.

        robot.config.use_degrees must be exactly True. robot.calibration entries
        may be JSON mappings or LeRobot-compatible MotorCalibration objects.
        Reading both guard and digest from one byte snapshot avoids accepting a
        guard derived from a different file revision between separate reads.
        """
        try:
            payload = self.calibration_path.read_bytes()
        except OSError as exc:
            raise StartupCheckError("Cannot read Follower calibration") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.expected_sha256.lower():
            raise StartupCheckError("Follower calibration SHA-256 mismatch")
        try:
            expected = ActionGuard.from_calibration_bytes(
                payload, margin_deg=self.margin_deg, margin_gripper=self.margin_gripper,
            )
            calibration = json.loads(payload)
        except (ValueError, TypeError) as exc:
            raise StartupCheckError(f"Invalid calibration or margins: {exc}") from exc

        config = getattr(robot, "config", None)
        if getattr(config, "use_degrees", None) is not True:
            raise StartupCheckError("Sender config.use_degrees must be exactly True")
        loaded = getattr(robot, "calibration", None)
        if not isinstance(loaded, Mapping) or set(loaded) != set(MOTOR_IDS):
            raise StartupCheckError("Sender must have six loaded Follower calibration entries")
        for motor in MOTOR_IDS:
            entry = loaded[motor]
            for name in CALIBRATION_FIELDS:
                value = entry.get(name) if isinstance(entry, Mapping) else getattr(entry, name, None)
                if type(value) is not int or value != calibration[motor][name]:
                    raise StartupCheckError(f"Loaded Follower calibration mismatch: {motor}.{name}")

        if not isinstance(guard, ActionGuard):
            raise StartupCheckError("Expected an ActionGuard")
        # Feature order is irrelevant; duplicate joints, changed units, expanded
        # ranges or a different inward margin must all be rejected.
        actual_bounds = {bound.action_key: bound for bound in guard.bounds}
        expected_bounds = {bound.action_key: bound for bound in expected.bounds}
        if len(guard.bounds) != len(expected.bounds) or actual_bounds != expected_bounds:
            raise StartupCheckError("ActionGuard ranges, units or margins do not match Follower calibration")
        return StartupCheckResult(digest, float(self.margin_deg), float(self.margin_gripper), expected.bounds)
