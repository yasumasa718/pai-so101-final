"""Fail-closed LeRobotDataset v0.6.1 episode writer for the SO-101.

Offline-capable adapter for SafeRecordLoop.frame_writer. It is intentionally
NOT an executable robot recorder. It never opens a serial port, starts motors,
releases torque, acquires images, or invokes LeRobot's CLI record().

Typical SINGLE-threaded orchestration, after an independent safety review::

    writer = DatasetRecordWriter(dataset, task="Pick the crushed can")
    # For each frame, BEFORE SafeRecordLoop.process_frame():
    writer.stage_observation(obs_processed)  # already acquired observation
    loop.process_frame(raw_action, obs)        # sends once, calls writer(sent)
    # Only after an entire episode has finished without any fault:
    loop.finish()
    writer.save_episode()
    # When completely finished:
    writer.finalize()

On any fault: do NOT call save_episode(). Do not retry a send. After safely
handling the robot independently, call writer.abort_episode() to discard the
in-memory episode. A failed save_episode() may have partially written files;
this adapter cannot guarantee rollback, so the dataset must be quarantined.

No hardware or real-time safety guarantees are provided. build_dataset_frame
behavior mirrors LeRobot 0.6.1 for six-joint vector + optional image/video
observations. LeRobot owns timestamps, frame indices, and episode indices.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image

from src.control.action_guard import MOTOR_IDS

logger = logging.getLogger(__name__)
JOINT_KEYS = frozenset(f"{name}.pos" for name in MOTOR_IDS)


class DatasetLike(Protocol):
    """Small LeRobotDataset 0.6.1 contract, also implemented by test fakes."""

    features: dict[str, dict[str, Any]]

    def add_frame(self, frame: dict[str, Any]) -> None: ...

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None: ...

    def clear_episode_buffer(self, delete_images: bool = True) -> None: ...

    def finalize(self) -> None: ...


class WriterState(str, Enum):
    READY = "ready"       # active episode can be staged; may contain frames
    STAGED = "staged"     # one observation waiting for returned sent_action
    FAULT = "fault"       # write/staging/save failed; no further writes
    ABORTED = "aborted"   # in-memory episode discarded; no reuse
    FINALIZED = "finalized"


class DatasetWriterStateError(RuntimeError):
    """Operation not permitted with this episode-writer state."""


@dataclass
class DatasetRecordWriter:
    """Append synchronized observations and *returned sent commands*.

    - Caller must stage the observation for each frame BEFORE a possible send.
    - Only the action returned by SOFollower.send_action may be passed to __call__.
      Calling with a requested action is a caller contract violation, not
      something this class can independently detect.
    - Exactly one staged observation is consumed per successful add_frame.
    - The dataset buffer is committed *only* by explicit save_episode().
    - If any exception occurs, the writer latches FAULT. It never auto-saves.
    - Do not share this object between threads or simultaneous recording loops.
    """

    dataset: DatasetLike
    task: str
    state: WriterState = field(default=WriterState.READY, init=False)
    frames_in_episode: int = field(default=0, init=False)
    saved_episodes: int = field(default=0, init=False)
    commit_uncertain: bool = field(default=False, init=False)
    fault_message: str | None = field(default=None, init=False)
    _staged: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _action_names: tuple[str, ...] = field(default=(), init=False, repr=False)
    _state_names: tuple[str, ...] = field(default=(), init=False, repr=False)
    _image_features: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if not isinstance(getattr(self.dataset, "features", None), dict):
            raise ValueError("dataset.features must be a dict")
        for method in ("add_frame", "save_episode", "clear_episode_buffer", "finalize"):
            if not callable(getattr(self.dataset, method, None)):
                raise TypeError(f"dataset lacks callable {method}()")
        features = self.dataset.features
        native_writer = getattr(self.dataset, "writer", None)
        if native_writer is not None and getattr(native_writer, "_streaming_encoder", None) is not None:
            raise ValueError("Streaming encoding is not supported by the guarded image-file validation")
        self._action_names = self._validate_joint_feature(features, "action")
        self._state_names = self._validate_joint_feature(features, "observation.state")
        # Only explicitly supported SO-101 features. Fail before recording if
        # the feature schema would require a field we silently omit.
        defaults = {"index", "episode_index", "frame_index", "timestamp", "task_index"}
        for key, spec in features.items():
            if key in defaults or key in {"action", "observation.state"}:
                continue
            if (
                not key.startswith("observation.images.")
                or not isinstance(spec, dict)
                or spec.get("dtype") not in ("image", "video")
                or not isinstance(spec.get("shape"), (tuple, list))
                or len(spec["shape"]) != 3
                or any(type(v) is not int or v <= 0 for v in spec["shape"])
            ):
                raise ValueError(f"Unsupported dataset feature: {key}")
            self._image_features[key] = spec

    @staticmethod
    def _validate_joint_feature(features: dict, key: str) -> tuple[str, ...]:
        spec = features.get(key)
        if not isinstance(spec, dict) or spec.get("dtype") != "float32":
            raise ValueError(f"{key}: expected float32 feature")
        if tuple(spec.get("shape", ())) != (6,):
            raise ValueError(f"{key}: expected shape [6]")
        names = spec.get("names")
        if not isinstance(names, (list, tuple)) or len(names) != 6:
            raise ValueError(f"{key}: expected six joint names")
        if any(not isinstance(n, str) for n in names) or set(names) != JOINT_KEYS:
            raise ValueError(f"{key}: expected exactly six SO-101 <joint>.pos names")
        return tuple(names)

    @staticmethod
    def _joint_vector(action: Mapping[str, object], names: tuple[str, ...], source: str) -> np.ndarray:
        if not isinstance(action, Mapping) or set(action) != JOINT_KEYS:
            raise ValueError(f"{source}: expected exactly six joint keys")
        values: list[float] = []
        for key in names:
            raw = action[key]
            if type(raw) not in (float, int) or not math.isfinite(raw):
                raise ValueError(f"{source}.{key}: expected finite number")
            # Use the actual float32 values which LeRobot serializes.
            values.append(float(raw))
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{source}: float32 conversion overflow")
        return array

    def _make_observation_frame(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise ValueError("observation: expected mapping")
        expected = JOINT_KEYS | {key.removeprefix("observation.images.") for key in self._image_features}
        if set(observation) != expected:
            raise ValueError(f"observation: unexpected/missing keys: {sorted(set(observation) ^ expected)}")
        state = {key: observation[key] for key in self._state_names}
        frame: dict[str, Any] = {
            "observation.state": self._joint_vector(state, self._state_names, "observation.state")
        }
        for key, spec in self._image_features.items():
            camera_name = key.removeprefix("observation.images.")
            value = observation[camera_name]
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
                raise ValueError(f"{key}: expected uint8 ndarray")
            if value.shape != tuple(spec["shape"]):
                raise ValueError(f"{key}: expected image shape {spec['shape']}, got {value.shape}")
            frame[key] = value.copy()  # avoid caller mutating after capture
        return frame

    def _fault(self, stage: str, error: BaseException, *, uncertain_commit: bool = False) -> None:
        self.state = WriterState.FAULT
        self.commit_uncertain = self.commit_uncertain or uncertain_commit
        self.fault_message = f"{stage}: {type(error).__name__}: {error}"
        self._staged = None
        logger.error("SO-101 dataset writer FAULT: %s", self.fault_message)

    def stage_observation(self, observation: Mapping[str, Any]) -> None:
        """Stage exactly one *already acquired* observation before sending.

        Must be called once per frame, immediately before process_frame().
        Failure latches FAULT and must prevent any subsequent motor send.
        """
        if self.state is not WriterState.READY:
            raise DatasetWriterStateError(f"Cannot stage from {self.state.value}")
        try:
            self._staged = self._make_observation_frame(observation)
            self.state = WriterState.STAGED
        except BaseException as exc:
            self._fault("stage_observation", exc)
            raise

    def __call__(self, sent_action: Mapping[str, object]) -> None:
        """FrameWriter callback; append only command returned by send_action().

        LeRobot v0.6.1 computes timestamp/frame_index itself. No external
        caller should add them to this frame. This is a buffered append,
        NOT a durable disk commit.
        """
        if self.state is not WriterState.STAGED or self._staged is None:
            raise DatasetWriterStateError(f"Cannot write frame from {self.state.value}")
        try:
            action = self._joint_vector(sent_action, self._action_names, "sent_action")
            frame = {
                **self._staged,
                "action": action,
                "task": self.task,
            }
            self.dataset.add_frame(frame)
            self.frames_in_episode += 1
            self._staged = None
            self.state = WriterState.READY
        except BaseException as exc:
            self._fault("add_frame", exc)
            raise

    def _validate_pending_images(self) -> None:
        """Check EVERY staged RGB file after async writes, before native save.

        LeRobot 0.6.1 write_image() logs disk failures without propagating them.
        Its statistics sample frames and video encoding enumerates existing
        files: a missing unsampled PNG can otherwise silently shorten a video.

        The native writer/episode_buffer interface is specific to LeRobot 0.6.1.
        In-memory DatasetLike fakes have no native writer or backing files.
        Streaming encoding is rejected at construction (no image-file buffer).
        This check is not a transaction or a physical motion safety mechanism.
        """
        if not self._image_features:
            return
        native_writer = getattr(self.dataset, "writer", None)
        if native_writer is None:
            return
        async_writer = getattr(native_writer, "image_writer", None)
        if async_writer is not None:
            async_writer.wait_until_done()
        buffer = getattr(native_writer, "episode_buffer", None)
        if not isinstance(buffer, dict) or buffer.get("size") != self.frames_in_episode:
            raise ValueError("Image episode buffer/frame count mismatch")
        for key, spec in self._image_features.items():
            paths = buffer.get(key)
            if not isinstance(paths, list) or len(paths) != self.frames_in_episode:
                raise ValueError(f"{key}: image buffer/frame count mismatch")
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    image.load()  # decode all pixels; existence/header alone is insufficient
                    if (
                        image.format != "PNG" or image.mode != "RGB"
                        or (image.height, image.width, 3) != tuple(spec["shape"])
                    ):
                        raise ValueError(f"{key}: invalid saved RGB image at frame {index}")

    def save_episode(self) -> None:
        """Explicitly commit an episode after the entire run succeeded.

        In particular, do NOT call this when SafeRecordLoop.state == FAULT.
        First wait for and validate every backing RGB PNG (native LeRobot).
        Image validation failure latches FAULT before commit; explicit abort is
        still possible. Once underlying save is attempted, failure may leave
        partial files/metadata. No automatic rollback is attempted.
        """
        if self.state is not WriterState.READY:
            raise DatasetWriterStateError(f"Cannot save from {self.state.value}")
        if self.frames_in_episode <= 0:
            raise DatasetWriterStateError("Cannot save empty episode")
        stage = "validate_pending_images"
        commit_attempted = False
        try:
            self._validate_pending_images()
            stage = "save_episode"
            commit_attempted = True
            self.dataset.save_episode()
            self.saved_episodes += 1
            self.frames_in_episode = 0
        except BaseException as exc:
            self._fault(stage, exc, uncertain_commit=commit_attempted)
            raise

    def abort_episode(self) -> None:
        """Discard UNSAVED episode buffers after a fault; never resume writer.

        Must be called by the supervising application after independently
        managing robot state. If native dataset.save_episode() failed, rollback is unknown
        and user must quarantine dataset instead.
        """
        if self.state in (WriterState.ABORTED, WriterState.FINALIZED):
            raise DatasetWriterStateError(f"Cannot abort from {self.state.value}")
        if self.commit_uncertain:
            raise DatasetWriterStateError("save_episode may have partially committed: quarantine dataset")
        try:
            self.dataset.clear_episode_buffer(delete_images=True)
            self.frames_in_episode = 0
            self._staged = None
            self.state = WriterState.ABORTED
        except BaseException as exc:
            self._fault("clear_episode_buffer", exc)
            raise

    def finalize(self) -> None:
        """Close writer only after successful save or explicit buffer abort.

        Does not disconnect SO-101 and does not clear an unfinished episode.
        Also not a substitute for an exception-safe hardware shutdown plan.
        """
        if self.state not in (WriterState.READY, WriterState.ABORTED):
            raise DatasetWriterStateError(f"Cannot finalize from {self.state.value}")
        if self.frames_in_episode or self._staged is not None or self.commit_uncertain:
            raise DatasetWriterStateError("Uncommitted frames/pending observation or uncertain commit")
        try:
            self.dataset.finalize()
            self.state = WriterState.FINALIZED
        except BaseException as exc:
            self._fault("finalize", exc, uncertain_commit=True)
            raise
