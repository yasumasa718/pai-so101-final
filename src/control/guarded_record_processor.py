"""LeRobot 0.6.1 teleoperation action guard processor (offline-tested adapter).

Do NOT enable this step in a live recording until handling a rejected action,
shutdown / torque release, and logging have been reviewed and tested safely.
The processor itself does not access any robot, serial port, or camera.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

from src.control.action_guard import ActionGuard


class GuardRejectedError(RuntimeError):
    """A position command violates the configured offline range gate."""


@dataclass
class GuardedTeleopActionStep(RobotActionProcessorStep):
    """Checks complete SO-101 teleop action dicts, never clips/repairs them."""

    guard: ActionGuard

    def action(self, action: RobotAction) -> RobotAction:
        checked = self.guard.check(action)
        if not checked.accepted or checked.checked_action is None:
            raise GuardRejectedError("; ".join(checked.reasons))
        return checked.checked_action

    def transform_features(self, features: dict[Any, Any]) -> dict[Any, Any]:
        """This gate does not change the type, shape, or names of features."""
        return features


def make_guarded_teleop_pipeline(
    calibration_path: Path,
    *,
    margin_deg: float = 0.0,
    margin_gripper: float = 0.0,
) -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    """Build a processor for offline tests; no hardware interaction occurs."""
    guard = ActionGuard.from_calibration(
        calibration_path,
        margin_deg=margin_deg,
        margin_gripper=margin_gripper,
    )
    return RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[GuardedTeleopActionStep(guard=guard)],
        name="so101_guarded_teleop_offline",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
