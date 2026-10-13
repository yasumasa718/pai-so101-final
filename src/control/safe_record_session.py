"""Single-episode orchestration for already-created SO-101 recording components.

No hardware discovery, connect/disconnect, torque control, camera capture, or
LeRobot CLI invocation is provided here. This is an OFFLINE integration stage:
use FakeRobot / FakeDataset, never an unreviewed real SOFollower.

The existing SafeRecordLoop performs guarded send_action() once per frame and
passes the *returned target command* to DatasetRecordWriter. This coordinator
stages the corresponding observation *before* any send, ensures no incomplete
frame is committed, and makes episode commit/abort explicit.

IMPORTANT: A Python exception is NOT a physical emergency stop. Previously
issued goals may still move servos. Handle robot support, emergency stop, and
power/torque disconnect separately before any future real-hardware integration.

Dataset save_episode() is NOT transactional. If it fails, partial files may
exist. ``commit_uncertain`` is latched and abort is FORBIDDEN: quarantine that
dataset for manual inspection instead of claiming rollback.

One session represents ONE episode. To record another episode, construct a new
SafeRecordLoop and a new SafeRecordSession with the same READY writer after a
successful commit. Do not use concurrently or call the components independently.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from src.control.dataset_record_writer import DatasetRecordWriter, WriterState
from src.control.record_startup_check import RecordStartupCheck, StartupCheckResult
from src.control.safe_record_loop import (
    RecordFault,
    RecordFaultInfo,
    RecordedFrameResult,
    RecordState,
    SafeRecordLoop,
    SendStatus,
)

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    READY = "ready"
    RECORDING = "recording"
    FAULT = "fault"
    SAVED = "saved"
    ABORTED = "aborted"
    FINALIZED = "finalized"


class SessionStateError(RuntimeError):
    """Requested episode operation is invalid in this state."""


@dataclass(frozen=True)
class SessionFaultInfo:
    stage: str
    frame_index: int
    send_status: SendStatus
    commit_uncertain: bool
    message: str


@dataclass
class SafeRecordSession:
    """Fail-closed glue between SafeRecordLoop and DatasetRecordWriter.

    Preconditions:
        loop.frame_writer is writer (the SAME instance, not a lambda wrapper)
        loop has not started, writer has no uncommitted/staged frames
        all devices, if ever used in future, are independently managed

    ``start()`` grants no real-world safety clearance.
    ``record_frame()`` expects already-acquired observations and an action.
    ``save_episode()`` is only called after successful loop.finish().
    ``abort_episode()`` must be called explicitly; it never moves the robot.
    """

    loop: SafeRecordLoop
    writer: DatasetRecordWriter
    startup_check: RecordStartupCheck | None = field(default=None, kw_only=True)
    startup_result: StartupCheckResult | None = field(default=None, init=False)
    state: SessionState = field(default=SessionState.READY, init=False)
    fault: SessionFaultInfo | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.loop, SafeRecordLoop):
            raise TypeError("loop must be a SafeRecordLoop")
        if not isinstance(self.writer, DatasetRecordWriter):
            raise TypeError("writer must be a DatasetRecordWriter")
        if self.loop.frame_writer is not self.writer:
            raise ValueError("loop.frame_writer must be this exact DatasetRecordWriter")
        if self.loop.state is not RecordState.READY:
            raise ValueError("loop must be READY and unused")
        if self.writer.state is not WriterState.READY or self.writer.frames_in_episode != 0:
            raise ValueError("writer must be READY with an empty episode")
        if self.writer.commit_uncertain or self.writer._staged is not None:
            raise ValueError("writer has uncertain commit or pending observation")
        if self.startup_check is not None and not isinstance(self.startup_check, RecordStartupCheck):
            raise TypeError("startup_check must be a RecordStartupCheck")

    def _require(self, *states: SessionState) -> None:
        if self.state not in states:
            allowed = ", ".join(s.value for s in states)
            raise SessionStateError(f"Requires {allowed}, got {self.state.value}")

    def _latch_fault(
        self, stage: str, exc: BaseException,
        *, send_status_hint: SendStatus = SendStatus.NOT_ATTEMPTED,
    ) -> None:
        """Also latch the underlying loop if failure happened outside it.

        This prevents direct subsequent sends through the exposed loop when
        observation staging, episode saving or finalization fails.
        """
        if isinstance(exc, RecordFault):
            stage = exc.fault.stage
            send_status = exc.fault.send_status
            frame_index = exc.fault.frame_index
        elif stage == "process_frame" and self.loop.state is RecordState.FAULT and self.loop.fault is not None:
            # SafeRecordLoop re-raises KeyboardInterrupt/SystemExit unchanged.
            # Preserve its latched send outcome instead of assuming no send.
            stage = self.loop.fault.stage
            send_status = self.loop.fault.send_status
            frame_index = self.loop.fault.frame_index
        else:
            send_status = send_status_hint
            frame_index = self.loop.frames_written

        self.fault = SessionFaultInfo(
            stage=stage,
            frame_index=frame_index,
            send_status=send_status,
            commit_uncertain=self.writer.commit_uncertain,
            message=f"{type(exc).__name__}: {exc}",
        )
        if self.loop.state is not RecordState.FAULT:
            self.loop.fault = RecordFaultInfo(
                frame_index=frame_index,
                stage=stage,
                send_status=send_status,
                message=str(exc),
            )
            self.loop.state = RecordState.FAULT
        self.state = SessionState.FAULT
        logger.error("Episode FAULT: %s", self.fault)

    def start(self) -> None:
        """Check configured startup evidence, then start a logical episode.

        NEVER connects or moves real hardware. Omitting startup_check preserves
        legacy offline API behavior and leaves startup_result as None.
        """
        self._require(SessionState.READY)
        stage = "startup_check"
        try:
            if self.startup_check is not None:
                self.startup_result = self.startup_check.verify(self.loop.guard, self.loop.robot)
            stage = "start"
            self.loop.start()
            self.state = SessionState.RECORDING
        except BaseException as exc:
            self._latch_fault(stage, exc)
            raise

    def record_frame(
        self,
        raw_action: Mapping[str, object],
        observation: Mapping[str, Any],
        *,
        dataset_observation: Mapping[str, Any] | None = None,
    ) -> RecordedFrameResult:
        """Stage an observation, then guard/send exactly once and buffer its return.

        ``observation`` goes to LeRobot action processors; by default, this
        same snapshot is recorded. Provide ``dataset_observation`` only when a
        separately processed observation belongs to the SAME capture moment.
        Neither path acquires data or generates robot commands automatically.
        """
        self._require(SessionState.RECORDING)
        stage = "stage_observation"
        completed_send = False
        try:
            to_record = observation if dataset_observation is None else dataset_observation
            self.writer.stage_observation(to_record)  # must succeed BEFORE send
            stage = "process_frame"
            result = self.loop.process_frame(raw_action, observation)
            completed_send = True
            stage = "post_frame_consistency"
            if (
                self.writer.state is not WriterState.READY
                or self.writer.frames_in_episode != self.loop.frames_written
            ):
                raise RuntimeError("Frame count/state mismatch after writing; no more sends")
            return result
        except BaseException as exc:
            self._latch_fault(
                stage, exc,
                send_status_hint=(SendStatus.RETURNED if completed_send else SendStatus.NOT_ATTEMPTED),
            )
            raise

    def save_episode(self) -> None:
        """Finish a nonempty, fault-free episode, then commit it to the dataset.

        Save failure can leave files/metadata partially written. No retry.
        """
        self._require(SessionState.RECORDING)
        if self.loop.frames_written == 0:
            raise SessionStateError("Cannot save an empty episode")
        try:
            if (
                self.writer.state is not WriterState.READY
                or self.writer.frames_in_episode != self.loop.frames_written
            ):
                raise RuntimeError("Frame count/state mismatch before saving")
            self.loop.finish()
            self.writer.save_episode()
            self.state = SessionState.SAVED
        except BaseException as exc:
            self._latch_fault("save_episode", exc)
            raise

    def abort_episode(self) -> None:
        """Explicitly discard ONLY uncommitted buffers; does not stop motors.

        Call only after the external supervisor has independently addressed the
        robot's physical state. If save_episode partially committed, do NOT
        clear the buffer and do NOT treat the dataset as trustworthy.
        """
        self._require(SessionState.RECORDING, SessionState.FAULT)
        if self.writer.commit_uncertain:
            raise SessionStateError("Commit is uncertain: quarantine dataset; do not abort/rollback")
        try:
            self.writer.abort_episode()
            if self.loop.state is not RecordState.FAULT:
                self.loop.fault = RecordFaultInfo(
                    frame_index=self.loop.frames_written,
                    stage="manual_abort",
                    send_status=SendStatus.NOT_ATTEMPTED,
                    message="Episode aborted by caller (does not stop motors)",
                )
                self.loop.state = RecordState.FAULT
            self.state = SessionState.ABORTED
        except BaseException as exc:
            self._latch_fault("abort_episode", exc)
            raise

    def finalize(self) -> None:
        """Finalize dataset ONLY after a save or explicit buffer discard.

        On successful abort, finalize may leave previously saved episodes but
        never the aborted one. Does not disconnect hardware.
        """
        self._require(SessionState.SAVED, SessionState.ABORTED)
        try:
            self.writer.finalize()
            self.state = SessionState.FINALIZED
        except BaseException as exc:
            self._latch_fault("finalize", exc)
            raise
