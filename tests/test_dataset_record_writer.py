"""Offline tests for LeRobot v0.6.1 dataset action/observation writer.

FakeDataset contains no motors, camera connections, disks, or network access.
One optional integration test creates an ephemeral LeRobotDataset with no robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.control.dataset_record_writer import (
    DatasetRecordWriter,
    DatasetWriterStateError,
    WriterState,
)

JOINTS = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)


def features(*, camera: bool = False, swap: bool = False) -> dict[str, dict[str, Any]]:
    action_names = list(reversed(JOINTS)) if swap else list(JOINTS)
    state_names = list(JOINTS) if swap else list(JOINTS)
    result = {
        "action": {"dtype": "float32", "shape": (6,), "names": action_names},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": state_names},
    }
    if camera:
        result["observation.images.top"] = {
            "dtype": "video", "shape": (2, 3, 3), "names": ["height", "width", "channels"]
        }
    return result


def action() -> dict[str, float]:
    return dict(zip(JOINTS, (0., 5., 10., -15., 20., 50.)))


def observation(*, camera: bool = False) -> dict[str, Any]:
    obs: dict[str, Any] = {**action(), "gripper.pos": 42.0}
    if camera:
        obs["top"] = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    return obs


@dataclass
class FakeDataset:
    features: dict[str, dict[str, Any]] = field(default_factory=features)
    added: list[dict[str, Any]] = field(default_factory=list)
    saved: list[list[dict[str, Any]]] = field(default_factory=list)
    clears: int = 0
    finalizes: int = 0
    fail_add: bool = False
    fail_save: bool = False
    fail_clear: bool = False
    fail_finalize: bool = False

    def add_frame(self, frame: dict[str, Any]) -> None:
        if self.fail_add:
            # Simulate partial mutation before the I/O/API exception.
            self.added.append(frame)
            raise OSError("simulated add_frame error")
        self.added.append(frame)

    def save_episode(self, episode_data=None, parallel_encoding=True) -> None:
        if self.fail_save:
            self.saved.append(list(self.added))  # partial write on disk
            raise OSError("simulated half-written parquet")
        self.saved.append(list(self.added))
        self.added.clear()

    def clear_episode_buffer(self, delete_images=True) -> None:
        assert delete_images is True
        self.clears += 1
        if self.fail_clear:
            raise OSError("simulated image writer cleanup failure")
        self.added.clear()

    def finalize(self) -> None:
        if self.fail_finalize:
            raise OSError("simulated finalize failure")
        self.finalizes += 1


@pytest.fixture
def dataset() -> FakeDataset:
    return FakeDataset()


def test_one_frame_is_buffered_with_sent_action_and_task(dataset):
    writer = DatasetRecordWriter(dataset, "Pick the crushed can")
    writer.stage_observation(observation())
    sent = {**action(), "gripper.pos": 47.}
    writer(sent)
    assert writer.state is WriterState.READY
    assert writer.frames_in_episode == 1
    assert dataset.saved == []  # never commit per frame
    stored = dataset.added[0]
    assert set(stored) == {"action", "observation.state", "task"}
    assert stored["task"] == "Pick the crushed can"
    assert stored["action"].dtype == np.float32
    assert stored["action"].tolist() == [0., 5., 10., -15., 20., 47.]
    assert stored["observation.state"].tolist() == [0., 5., 10., -15., 20., 42.]
    assert "timestamp" not in stored and "frame_index" not in stored


def test_feature_name_order_used_instead_of_assuming_f1_to_f6():
    ds = FakeDataset(features=features(swap=True))
    writer = DatasetRecordWriter(ds, "task")
    writer.stage_observation(observation())
    writer(action())
    assert ds.added[0]["action"].tolist() == [50., 20., -15., 10., 5., 0.]
    assert ds.added[0]["observation.state"].tolist() == [0., 5., 10., -15., 20., 42.]


def test_image_copied_when_staged_and_matches_features():
    ds = FakeDataset(features=features(camera=True))
    writer = DatasetRecordWriter(ds, "task")
    obs = observation(camera=True)
    expected = obs["top"].copy()
    writer.stage_observation(obs)
    obs["top"][:] = 0
    writer(action())
    assert np.array_equal(ds.added[0]["observation.images.top"], expected)


def test_save_episode_explicit_and_finalize(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    with pytest.raises(DatasetWriterStateError, match="empty"):
        writer.save_episode()
    for _ in range(2):
        writer.stage_observation(observation())
        writer(action())
    writer.save_episode()
    assert len(dataset.saved) == 1 and len(dataset.saved[0]) == 2
    assert writer.frames_in_episode == 0 and writer.saved_episodes == 1
    writer.finalize()
    assert writer.state is WriterState.FINALIZED and dataset.finalizes == 1
    with pytest.raises(DatasetWriterStateError):
        writer.stage_observation(observation())


def test_can_save_separate_complete_episodes(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    for _ in range(2):
        writer.stage_observation(observation())
        writer(action())
        writer.save_episode()
    assert writer.saved_episodes == 2
    assert len(dataset.saved) == 2


def test_double_stage_rejected_and_no_commit(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    with pytest.raises(DatasetWriterStateError):
        writer.stage_observation(observation())
    with pytest.raises(DatasetWriterStateError):
        writer.save_episode()
    assert dataset.added == dataset.saved == []
    writer.abort_episode()
    assert dataset.clears == 1 and writer.state is WriterState.ABORTED


def test_call_without_stage_rejected(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    with pytest.raises(DatasetWriterStateError):
        writer(action())
    assert dataset.added == []


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None, True, "1"])
def test_bad_action_latches_writer_fault(dataset, value):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    target = action()
    target["wrist_flex.pos"] = value
    with pytest.raises(ValueError):
        writer(target)
    assert writer.state is WriterState.FAULT
    assert dataset.added == []
    with pytest.raises(DatasetWriterStateError):
        writer.stage_observation(observation())
    writer.abort_episode()


def test_bad_observation_latches_fault_before_send(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    bad = observation()
    del bad["wrist_flex.pos"]
    with pytest.raises(ValueError, match="unexpected/missing"):
        writer.stage_observation(bad)
    assert writer.state is WriterState.FAULT
    assert dataset.added == []
    writer.abort_episode()


def test_bad_image_rejected(dataset):
    dataset.features = features(camera=True)
    writer = DatasetRecordWriter(dataset, "task")
    obs = observation(camera=True)
    obs["top"] = np.zeros((2, 3, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="uint8"):
        writer.stage_observation(obs)
    assert writer.state is WriterState.FAULT


def test_buffer_discard_on_fault_and_no_retry(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    writer(action())
    dataset.fail_add = True
    writer.stage_observation(observation())
    with pytest.raises(OSError):
        writer(action())
    assert writer.state is WriterState.FAULT
    with pytest.raises(DatasetWriterStateError):
        writer.save_episode()
    writer.abort_episode()
    assert dataset.clears == 1 and dataset.added == [] and dataset.saved == []
    with pytest.raises(DatasetWriterStateError):
        writer.stage_observation(observation())


def test_save_failure_cannot_claim_rollback(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    writer(action())
    dataset.fail_save = True
    with pytest.raises(OSError):
        writer.save_episode()
    assert writer.state is WriterState.FAULT
    assert writer.commit_uncertain
    with pytest.raises(DatasetWriterStateError, match="quarantine"):
        writer.abort_episode()
    with pytest.raises(DatasetWriterStateError):
        writer.finalize()
    assert dataset.clears == 0


def test_abort_never_finalizes_or_disconnects(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    writer(action())
    writer.abort_episode()
    assert writer.state is WriterState.ABORTED
    assert dataset.added == dataset.saved == []
    assert dataset.finalizes == 0
    writer.finalize()  # explicit cleanup only after buffer discarded
    assert dataset.finalizes == 1


def test_cleanup_failure_latches_fault(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    dataset.fail_clear = True
    with pytest.raises(OSError):
        writer.abort_episode()
    assert writer.state is WriterState.FAULT
    assert dataset.finalizes == 0


def test_invalid_feature_schema_rejected_before_any_frames():
    bad = features()
    bad["action"]["names"] = ["wrist_flex.pos"] * 6
    with pytest.raises(ValueError):
        DatasetRecordWriter(FakeDataset(features=bad), "task")
    bad2 = features()
    bad2["observation.extra"] = {"dtype": "float32", "shape": (3,)}
    with pytest.raises(ValueError, match="Unsupported"):
        DatasetRecordWriter(FakeDataset(features=bad2), "task")


def test_finalize_fails_for_unsaved_episode(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    writer(action())
    with pytest.raises(DatasetWriterStateError, match="Uncommitted"):
        writer.finalize()
    assert dataset.finalizes == 0


def test_safe_record_loop_callback_contract(tmp_path: Path):
    """Pair the real one-frame guard/controller with fake writer and sender."""
    from src.control.action_guard import ActionGuard, MOTOR_IDS
    from src.control.safe_record_loop import RecordFault, SafeRecordLoop, RecordState
    import json

    ranges = {name: (0, 4095) for name in MOTOR_IDS}
    cal = {
        name: {"id": motor_id, "drive_mode": 0, "homing_offset": 0,
               "range_min": ranges[name][0], "range_max": ranges[name][1]}
        for name, motor_id in MOTOR_IDS.items()
    }
    path = tmp_path / "follower.json"
    path.write_text(json.dumps(cal), encoding="utf-8")
    guard = ActionGuard.from_calibration(path)

    class FakeRobot:
        calls = 0

        def send_action(self, requested):
            self.calls += 1
            return {**requested, "gripper.pos": 47.}

    def identity(payload):
        return dict(payload[0])

    ds = FakeDataset()
    writer = DatasetRecordWriter(ds, "task")
    robot = FakeRobot()
    loop = SafeRecordLoop(guard, identity, identity, robot, writer)
    loop.start()
    writer.stage_observation(observation())
    loop.process_frame(action(), observation())
    assert robot.calls == 1
    assert ds.added[0]["action"].tolist()[-1] == 47.
    loop.finish()
    writer.save_episode()
    writer.finalize()
    assert len(ds.saved) == 1 and len(ds.saved[0]) == 1


def test_safe_record_loop_fault_requires_explicit_dataset_abort(tmp_path: Path):
    from src.control.action_guard import ActionGuard, MOTOR_IDS
    from src.control.safe_record_loop import RecordFault, SafeRecordLoop, RecordState
    import json

    cal = {
        name: {"id": motor_id, "drive_mode": 0, "homing_offset": 0,
               "range_min": 0, "range_max": 4095}
        for name, motor_id in MOTOR_IDS.items()
    }
    path = tmp_path / "follower.json"
    path.write_text(json.dumps(cal), encoding="utf-8")
    guard = ActionGuard.from_calibration(path)

    class FakeRobot:
        def __init__(self):
            self.calls = 0

        def send_action(self, requested):
            self.calls += 1
            return dict(requested)

    ds = FakeDataset()
    writer = DatasetRecordWriter(ds, "task")
    robot = FakeRobot()
    loop = SafeRecordLoop(guard, lambda p: p[0], lambda p: p[0], robot, writer)
    loop.start()
    writer.stage_observation(observation())
    bad = action()
    bad["wrist_flex.pos"] = -300.0
    with pytest.raises(RecordFault):
        loop.process_frame(bad, observation())
    assert loop.state is RecordState.FAULT
    assert writer.state is WriterState.STAGED  # explicit abort by supervisor
    with pytest.raises(DatasetWriterStateError):
        writer.save_episode()
    writer.abort_episode()
    assert ds.saved == [] and robot.calls == 0


def test_real_lerobot_dataset_roundtrip_without_robot(tmp_path: Path):
    """Run only in user's WSL2 environment with LeRobot 0.6.1 installed."""
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import pyarrow.parquet as pq

    root = tmp_path / "real_dataset"
    ds = LeRobotDataset.create(
        repo_id="local/so101_offline_writer_test", fps=30, features=features(),
        root=root, robot_type="so_follower", use_videos=False,
    )
    writer = DatasetRecordWriter(ds, "Pick the crushed can")
    writer.stage_observation(observation())
    writer({**action(), "gripper.pos": 47.})
    writer.save_episode()
    writer.finalize()
    parquets = list((root / "data").rglob("*.parquet"))
    assert len(parquets) == 1
    columns = pq.read_table(parquets[0], columns=["action", "observation.state"])
    assert columns.num_rows == 1
    assert columns["action"].to_pylist()[0][-1] == pytest.approx(47.0)
    assert columns["observation.state"].to_pylist()[0][-1] == pytest.approx(42.0)


def test_nonfinite_observation_fails_before_append(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    obs = observation()
    obs["gripper.pos"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        writer.stage_observation(obs)
    assert writer.state is WriterState.FAULT
    assert dataset.added == []


def test_extra_observation_key_fails_before_append(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    obs = observation()
    obs["debug_value"] = 13
    with pytest.raises(ValueError, match="unexpected/missing"):
        writer.stage_observation(obs)
    assert writer.state is WriterState.FAULT
    assert dataset.added == []


def test_failed_finalize_never_claims_success(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    writer(action())
    writer.save_episode()
    dataset.fail_finalize = True
    with pytest.raises(OSError):
        writer.finalize()
    assert writer.state is WriterState.FAULT
    assert writer.commit_uncertain is True


def test_writer_failure_does_not_allow_save_even_after_partial_frame(dataset):
    writer = DatasetRecordWriter(dataset, "task")
    writer.stage_observation(observation())
    dataset.fail_add = True
    with pytest.raises(OSError):
        writer(action())
    assert len(dataset.added) == 1  # fake mimics a partial side effect
    assert writer.state is WriterState.FAULT
    with pytest.raises(DatasetWriterStateError):
        writer.save_episode()
    writer.abort_episode()
    assert dataset.added == []

