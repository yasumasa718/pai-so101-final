"""One guarded SO-101 recording *frame* (not a complete LeRobot recorder).

This module performs no hardware discovery, connect/disconnect, torque control,
image acquisition, or dataset finalization. Callers inject the action processors,
a robot-like sender, and a frame writer. Do not use with real hardware until the
supervising application has an independently reviewed emergency-stop, arm
support, shutdown and dataset-abort procedure.

The saved action comes from robot.send_action()'s returned *command*, not the
initial teleoperator request. This is still not measured motor motion.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

from src.control.action_guard import ActionGuard, MOTOR_IDS

logger = logging.getLogger(__name__)

# Deliberately avoid importing LeRobot here. Its compatible processor pipelines
# are callables; the interface can therefore be exercised with FakeRobot alone.
Action = dict[str, float]
Observation = dict[str, Any]
ActionProcessor = Callable[[tuple[Action, Observation]], Mapping[str, object]]
FrameWriter = Callable[[Action], None]


class ActionSender(Protocol):
    def send_action(self, action: Action) -> Mapping[str, object]:
        """Return the target command actually sent (may be clipped)."""
        ...


class RecordState(str, Enum):
    READY = "ready"
    RECORDING = "recording"
    FAULT = "fault"
    FINISHED = "finished"


class SendStatus(str, Enum):
    NOT_ATTEMPTED = "not_attempted"
    UNCERTAIN = "uncertain"  # call raised; command might already have reached bus
    RETURNED = "returned"  # method returned; this does NOT prove motor motion


@dataclass(frozen=True)
class RecordFaultInfo:
    frame_index: int
    stage: str
    send_status: SendStatus
    message: str


class RecordStateError(RuntimeError):
    """The requested operation is not allowed in the current latched state."""


class RecordFault(RuntimeError):
    """A frame failed. This object must never resume after a FAULT."""

    def __init__(self, fault: RecordFaultInfo) -> None:
        self.fault = fault
        super().__init__(
            f"frame={fault.frame_index}, stage={fault.stage}, "
            f"send_status={fault.send_status.value}: {fault.message}"
        )


@dataclass(frozen=True)
class RecordedFrameResult:
    frame_index: int
    requested_action: Action
    sent_action: Action


@dataclass
class SafeRecordLoop:
    """Guard one teleoperation frame, send once, and save the returned command.

    This is a *core frame boundary* to test before integration into LeRobot's
    dataset/episode/camera loop. start() is a logical transition only: it is
    NOT hardware clearance. A fault latches for this controller's lifetime.

    frame_writer must use the provided ``sent_action`` as the dataset's
    ``action`` field. Its caller is responsible for any associated observation,
    timestamp, episode handling, and atomic/abort policy. A writer failure after
    send cannot roll back a command already sent to the robot.
    """

    guard: ActionGuard
    teleop_action_processor: ActionProcessor
    robot_action_processor: ActionProcessor
    robot: ActionSender
    frame_writer: FrameWriter
    state: RecordState = field(default=RecordState.READY, init=False)
    fault: RecordFaultInfo | None = field(default=None, init=False)
    frames_written: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        expected = {f"{motor}.pos" for motor in MOTOR_IDS}
        keys = [item.action_key for item in self.guard.bounds]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Guard must contain each of the six SO-101 joints exactly once")
        if not callable(self.teleop_action_processor) or not callable(self.robot_action_processor):
            raise TypeError("Both action processors must be callable")
        if not callable(self.robot.send_action) or not callable(self.frame_writer):
            raise TypeError("Robot sender and frame writer must be callable")

    def start(self) -> None:
        """Start *logical* processing; does not connect, calibrate or move."""
        if self.state is not RecordState.READY:
            raise RecordStateError(f"Cannot start from {self.state.value}")
        self.state = RecordState.RECORDING

    def finish(self) -> None:
        """Finish normally. Does NOT disconnect motors or finalize a dataset."""
        if self.state is not RecordState.RECORDING:
            raise RecordStateError(f"Cannot finish from {self.state.value}")
        self.state = RecordState.FINISHED

    def _validated_copy(self, action: Mapping[str, object]) -> Action:
        result = self.guard.check(action)
        if not result.accepted or result.checked_action is None:
            raise ValueError("; ".join(result.reasons))
        return result.checked_action

    def process_frame(
        self,
        raw_action: Mapping[str, object],
        observation: Mapping[str, Any],
    ) -> RecordedFrameResult:
        """Process exactly one frame, or latch FAULT and raise RecordFault.

        Any exception after entering the guarded operation latches FAULT.
        In particular, exceptions raised *inside* send_action are ambiguous:
        the servo bus may or may not have received the goal. Do not retry.
        This function deliberately never releases torque or disconnects.
        """
        if self.state is not RecordState.RECORDING:
            raise RecordStateError(f"Cannot process frame from {self.state.value}")

        frame_index = self.frames_written
        stage = "raw_guard"
        send_status = SendStatus.NOT_ATTEMPTED
        try:
            # Raw commands are already SO-101-style joint action dictionaries.
            # Never silently repair a bad value by asking a processor to fix it.
            raw = self._validated_copy(raw_action)
            stage = "observation_copy"
            obs = dict(observation)

            stage = "teleop_processor"
            processed = self.teleop_action_processor((dict(raw), dict(obs)))
            stage = "teleop_guard"
            teleop = self._validated_copy(processed)

            stage = "robot_processor"
            candidate = self.robot_action_processor((dict(teleop), dict(obs)))
            stage = "final_command_guard"
            to_send = self._validated_copy(candidate)

            stage = "send_action"
            send_status = SendStatus.UNCERTAIN
            sent_response = self.robot.send_action(dict(to_send))
            send_status = SendStatus.RETURNED

            # A rejected returned command has ALREADY been sent. This check
            # serves dataset integrity / fault escalation, not motion prevention.
            stage = "sent_command_guard"
            sent = self._validated_copy(sent_response)

            # Only this verified return value may be recorded, never raw or
            # teleop action. The writer must copy the action if retaining it.
            stage = "frame_writer"
            self.frame_writer(dict(sent))

            self.frames_written += 1
            logger.debug("Recorded frame=%d (validated returned command)", frame_index)
            return RecordedFrameResult(frame_index, dict(raw), dict(sent))
        except BaseException as exc:
            # Also latch on KeyboardInterrupt/SystemExit, which are not Exception.
            # Do not call disconnect()/disable_torque() here. Arm fall risk.
            # No continuation or automatic retry once faulted.
            info = RecordFaultInfo(
                frame_index=frame_index,
                stage=stage,
                send_status=send_status,
                message=f"{type(exc).__name__}: {exc}",
            )
            self.fault = info
            self.state = RecordState.FAULT
            logger.error("SO-101 recording FAULT: %s", info, exc_info=True)
            if not isinstance(exc, Exception):
                # Preserve interrupts for the external supervising application.
                raise
            raise RecordFault(info) from exc
