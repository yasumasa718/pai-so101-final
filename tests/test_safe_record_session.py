"""複数フレームのエピソード管理をオフラインで検証する。モーターやカメラは操作しない。"""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.control.action_guard import ActionGuard, JointBound, MOTOR_IDS
from src.control.dataset_record_writer import DatasetRecordWriter, WriterState
from src.control.safe_record_loop import (
    RecordFault, RecordState, RecordStateError, SafeRecordLoop, SendStatus,
)
from src.control.safe_record_session import (
    SafeRecordSession, SessionState, SessionStateError,
)

JOINTS = tuple(f"{name}.pos" for name in MOTOR_IDS)


def action(gripper: float = 50.0) -> dict[str, float]:
    return {name: (gripper if name == "gripper.pos" else 0.0) for name in JOINTS}


def observation(*, camera: bool = False) -> dict[str, Any]:
    result: dict[str, Any] = action(42.0)
    if camera:
        result["top"] = np.full((2, 3, 3), 32, dtype=np.uint8)
    return result


def features(*, camera: bool = False):
    result = {
        "action": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
        "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
    }
    if camera:
        result["observation.images.top"] = {
            "dtype": "video", "shape": (2, 3, 3),
            "names": ["height", "width", "channels"],
        }
    return result


@dataclass
class FakeRobot:
    sent: list[dict[str, float]] = field(default_factory=list)
    clip_to: float | None = None
    fail_send: bool = False
    disconnect_count: int = 0

    def send_action(self, requested):
        self.sent.append(dict(requested))
        if self.fail_send:
            raise OSError("serial send outcome unknown")
        reply = dict(requested)
        if self.clip_to is not None:
            reply["gripper.pos"] = self.clip_to
        return reply

    def disconnect(self):
        self.disconnect_count += 1


@dataclass
class FakeDataset:
    features: dict = field(default_factory=features)
    pending: list[dict] = field(default_factory=list)
    episodes: list[list[dict]] = field(default_factory=list)
    cleared: int = 0
    finalized: int = 0
    fail_save: bool = False
    fail_add: bool = False
    fail_finalize: bool = False

    def add_frame(self, frame: dict):
        self.pending.append(frame)
        if self.fail_add:
            raise OSError("partial frame buffer write")

    def save_episode(self, episode_data=None, parallel_encoding=True):
        if self.fail_save:
            self.episodes.append(list(self.pending))  # ディスクに一部だけ書き込まれた状況を再現する。
            raise OSError("half committed episode")
        self.episodes.append(list(self.pending))
        self.pending.clear()

    def clear_episode_buffer(self, delete_images=True):
        assert delete_images is True
        self.cleared += 1
        self.pending.clear()

    def finalize(self):
        if self.fail_finalize:
            raise OSError("failed close parquet writer")
        self.finalized += 1


def identity(value):
    cmd, _obs = value
    return dict(cmd)


def guard() -> ActionGuard:
    return ActionGuard(tuple(
        JointBound(k, 0.0 if k == "gripper.pos" else -100.0,
                   100.0, "normalized_0_100" if k == "gripper.pos" else "degree")
        for k in JOINTS
    ))


def build(*, robot=None, dataset=None, writer=None):
    bot = robot if robot is not None else FakeRobot()
    ds = dataset if dataset is not None else FakeDataset()
    out = writer if writer is not None else DatasetRecordWriter(ds, task="Pick crushed can")
    loop = SafeRecordLoop(guard(), identity, identity, bot, out)
    return SafeRecordSession(loop, out), bot, ds


def test_constructor_requires_exact_writer_instance():
    session, robot, ds = build()
    loop = SafeRecordLoop(guard(), identity, identity, robot, lambda sent: None)
    with pytest.raises(ValueError, match="exact DatasetRecordWriter"):
        SafeRecordSession(loop, session.writer)
    assert robot.sent == []


def test_cannot_record_before_start():
    session, robot, ds = build()
    with pytest.raises(SessionStateError):
        session.record_frame(action(), observation())
    assert robot.sent == [] and ds.pending == []


def test_two_frames_then_explicit_commit_and_finalize():
    session, robot, ds = build()
    session.start()
    a = session.record_frame(action(50), observation())
    b = session.record_frame(action(60), observation())
    assert (a.frame_index, b.frame_index) == (0, 1)
    assert ds.episodes == []
    assert len(ds.pending) == 2
    session.save_episode()
    assert session.state is SessionState.SAVED
    assert len(ds.episodes) == 1
    assert ds.episodes[0][0]["action"][-1] == pytest.approx(50)
    assert ds.episodes[0][1]["action"][-1] == pytest.approx(60)
    assert len(robot.sent) == 2
    session.finalize()
    assert session.state is SessionState.FINALIZED
    assert ds.finalized == 1
    assert robot.disconnect_count == 0


def test_returns_clipped_target_for_dataset_not_original():
    session, robot, ds = build(robot=FakeRobot(clip_to=47.0))
    session.start()
    session.record_frame(action(90), observation())
    session.save_episode()
    assert robot.sent[0]["gripper.pos"] == 90.0
    assert ds.episodes[0][0]["action"][-1] == pytest.approx(47.0)


def test_invalid_observation_prevents_any_send_and_latches_fault():
    session, robot, ds = build()
    session.start()
    obs = observation()
    obs["gripper.pos"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        session.record_frame(action(), obs)
    assert session.state is SessionState.FAULT
    assert session.loop.state is RecordState.FAULT
    assert session.fault.send_status is SendStatus.NOT_ATTEMPTED
    assert robot.sent == []
    with pytest.raises(SessionStateError):
        session.record_frame(action(), observation())
    session.abort_episode()
    session.finalize()
    assert ds.episodes == [] and ds.cleared == 1


def test_raw_f4_violation_aborts_entire_buffer_not_just_bad_frame():
    session, robot, ds = build()
    session.start()
    session.record_frame(action(), observation())
    invalid = action()
    invalid["wrist_flex.pos"] = -105.0
    with pytest.raises(RecordFault):
        session.record_frame(invalid, observation())
    assert len(robot.sent) == 1  # 2 フレーム目は送信されていないことを確認する。
    assert session.state is SessionState.FAULT
    assert session.fault.stage == "raw_guard"
    with pytest.raises(SessionStateError):
        session.save_episode()
    session.abort_episode()
    assert ds.pending == [] and ds.episodes == []
    session.finalize()
    assert robot.disconnect_count == 0


def test_serial_exception_is_uncertain_and_no_commit():
    session, robot, ds = build(robot=FakeRobot(fail_send=True))
    session.start()
    with pytest.raises(RecordFault):
        session.record_frame(action(), observation())
    assert session.fault.send_status is SendStatus.UNCERTAIN
    assert len(robot.sent) == 1
    assert ds.pending == []
    session.abort_episode()
    session.finalize()
    assert ds.episodes == [] and robot.disconnect_count == 0


def test_add_frame_error_latches_fault_after_send_then_discards_buffer():
    session, robot, ds = build(dataset=FakeDataset(fail_add=True))
    session.start()
    with pytest.raises(RecordFault):
        session.record_frame(action(), observation())
    assert len(robot.sent) == 1
    assert session.fault.send_status is SendStatus.RETURNED
    assert ds.pending
    session.abort_episode()
    assert ds.pending == [] and ds.episodes == []
    session.finalize()


def test_manual_abort_discards_all_and_blocks_reuse():
    session, robot, ds = build()
    session.start()
    session.record_frame(action(), observation())
    session.abort_episode()
    assert session.state is SessionState.ABORTED
    assert session.loop.state is RecordState.FAULT
    with pytest.raises(SessionStateError):
        session.record_frame(action(), observation())
    with pytest.raises(SessionStateError):
        session.save_episode()
    session.finalize()
    assert ds.episodes == [] and ds.cleared == 1


def test_empty_episode_not_committed_and_can_still_record():
    session, robot, ds = build()
    session.start()
    with pytest.raises(SessionStateError, match="empty"):
        session.save_episode()
    assert session.state is SessionState.RECORDING
    session.record_frame(action(), observation())
    session.save_episode()
    assert len(ds.episodes) == 1


def test_save_exception_quarantines_no_rollback():
    session, robot, ds = build(dataset=FakeDataset(fail_save=True))
    session.start()
    session.record_frame(action(), observation())
    with pytest.raises(OSError, match="half committed"):
        session.save_episode()
    assert session.state is SessionState.FAULT
    assert session.loop.state is RecordState.FAULT
    assert session.fault.commit_uncertain
    assert session.writer.commit_uncertain
    with pytest.raises(SessionStateError, match="quarantine"):
        session.abort_episode()
    with pytest.raises(SessionStateError):
        session.finalize()
    assert ds.cleared == 0 and len(ds.episodes) == 1


def test_second_episode_can_use_same_dataset_writer_with_fresh_loop():
    session, robot, ds = build()
    writer = session.writer
    session.start()
    session.record_frame(action(40), observation())
    session.save_episode()
    second_loop = SafeRecordLoop(guard(), identity, identity, robot, writer)
    second = SafeRecordSession(second_loop, writer)
    second.start()
    second.record_frame(action(70), observation())
    second.save_episode()
    assert len(ds.episodes) == 2
    assert writer.saved_episodes == 2
    second.finalize()
    assert ds.finalized == 1


def test_image_staged_and_copied_before_send():
    ds = FakeDataset(features=features(camera=True))
    session, robot, _ = build(dataset=ds)
    obs = observation(camera=True)
    session.start()
    session.record_frame(action(), obs)
    obs["top"][:] = 0  # フレーム処理後に、呼び出し側が持つ元のデータを書き換える。
    assert np.all(ds.pending[0]["observation.images.top"] == 32)
    session.save_episode()
    session.finalize()


def test_dataset_observation_can_be_supplied_separately():
    session, robot, ds = build()
    session.start()
    separate = observation()
    separate["gripper.pos"] = 37.0
    session.record_frame(action(), observation(), dataset_observation=separate)
    assert ds.pending[0]["observation.state"][-1] == pytest.approx(37.0)


def test_finalization_error_latches_fault_and_no_disconnect():
    ds = FakeDataset(fail_finalize=True)
    session, robot, _ = build(dataset=ds)
    session.start()
    session.record_frame(action(), observation())
    session.save_episode()
    with pytest.raises(OSError):
        session.finalize()
    assert session.state is SessionState.FAULT
    assert session.writer.commit_uncertain
    assert robot.disconnect_count == 0


def test_normal_session_cannot_restart_or_abort_after_save():
    session, robot, ds = build()
    session.start()
    session.record_frame(action(), observation())
    session.save_episode()
    with pytest.raises(SessionStateError):
        session.abort_episode()
    with pytest.raises(SessionStateError):
        session.start()
    assert ds.cleared == 0


@pytest.fixture
def follower_guard() -> ActionGuard:
    """リポジトリに保存した Follower 設定を読む。実機や実機用キャッシュにはアクセスしない。"""
    path = Path(__file__).resolve().parents[1] / "configs/calibration_snapshots/my_follower_for_004.json"
    before = path.read_bytes()
    result = ActionGuard.from_calibration(path)
    assert path.read_bytes() == before
    return result


@pytest.fixture
def real_dataset(tmp_path: Path) -> Iterator[Any]:
    pytest.importorskip("lerobot")
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    schema = features()
    # 実際の API で、指令と観測の関節の並び順が別々でも正しく扱えるか試す。
    schema["action"]["names"] = list(reversed(JOINTS))
    ds = LeRobotDataset.create(
        repo_id="local/so101_offline_session_test", fps=30, features=schema,
        root=tmp_path / "session_integration", robot_type="so_follower", use_videos=False,
    )
    # 保存をわざと失敗させたテストでも、開いたファイルを最後に閉じる。
    # これはテストの後片付けであり、異常復旧や保存前への復元ではない。一時データも消さない。
    close_dataset = ds.finalize
    try:
        yield ds
    finally:
        close_dataset()


def real_session(ds: Any, gate: ActionGuard, robot: FakeRobot, writer: DatasetRecordWriter | None = None) -> SafeRecordSession:
    output = writer if writer is not None else DatasetRecordWriter(ds, "Pick crushed can")
    return SafeRecordSession(SafeRecordLoop(gate, identity, identity, robot, output), output)


def read_frames(ds: Any) -> Any:
    import pyarrow as pa
    import pyarrow.parquet as pq

    files = sorted((ds.root / "data").rglob("*.parquet"))
    assert files
    return pa.concat_tables([pq.read_table(path) for path in files])


def test_true_lerobot_dataset_roundtrip_without_hardware(real_dataset: Any, follower_guard: ActionGuard) -> None:
    """実際のデータセットに 2 エピソードを保存し、全フレームの戻り値の指令と観測値を比べる。"""
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds = real_dataset
    robot = FakeRobot(clip_to=47.123456789)
    session = real_session(ds, follower_guard, robot)
    expected_actions: list[list[float]] = []
    expected_states: list[list[float]] = []
    for episode_index in range(2):
        if episode_index:
            session = real_session(ds, follower_guard, robot, session.writer)
        session.start()
        for frame_index in range(3):
            requested = dict(zip(JOINTS, (1.23456789 + frame_index, -2.34567891,
                                         3.45678912, -4.56789123, 5.67891234, 60.12345678)))
            obs = {key: value / 2 for key, value in requested.items()}
            result = session.record_frame(requested, obs)
            assert result.frame_index == frame_index
            assert result.sent_action["gripper.pos"] != requested["gripper.pos"]
            expected_actions.append([result.sent_action[key] for key in ds.features["action"]["names"]])
            expected_states.append([obs[key] for key in ds.features["observation.state"]["names"]])
        assert ds.meta.total_episodes == episode_index  # 自動ではエピソードを保存しないことを確認する。
        session.save_episode()
        assert session.state is SessionState.SAVED
        assert not ds.has_pending_frames()
    session.finalize()

    table = read_frames(ds)
    assert table.num_rows == 6
    assert table.schema.field("action").type.value_type == pa.float32()
    np.testing.assert_allclose(table["action"].to_pylist(), expected_actions, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(table["observation.state"].to_pylist(), expected_states, rtol=1e-7, atol=1e-6)
    assert table["episode_index"].to_pylist() == [0, 0, 0, 1, 1, 1]
    assert table["frame_index"].to_pylist() == [0, 1, 2, 0, 1, 2]
    assert table["index"].to_pylist() == list(range(6))
    np.testing.assert_allclose(table["timestamp"].to_pylist(), [0, 1 / 30, 2 / 30] * 2, atol=1e-7)
    assert table["task_index"].to_pylist() == [0] * 6
    assert ds.meta.total_episodes == session.writer.saved_episodes == 2
    assert ds.meta.total_frames == len(robot.sent) == 6
    info = json.loads((ds.root / "meta/info.json").read_text(encoding="utf-8"))
    assert info["total_episodes"] == 2 and info["total_frames"] == 6
    episodes = pa.concat_tables([
        pq.read_table(path) for path in sorted((ds.root / "meta/episodes").rglob("*.parquet"))
    ])
    assert episodes.num_rows == 2
    assert episodes["episode_index"].to_pylist() == [0, 1]
    assert episodes["length"].to_pylist() == [3, 3]
    assert session.state is SessionState.FINALIZED
    assert robot.disconnect_count == 0


@pytest.mark.parametrize("failure", ["raw_f2", "raw_f4", "final_f2", "final_f4", "send", "manual_abort"])
def test_real_dataset_discards_fault_episode_preserving_saved_episode(
    real_dataset: Any, follower_guard: ActionGuard, failure: str,
) -> None:
    ds = real_dataset
    robot = FakeRobot()
    saved = real_session(ds, follower_guard, robot)
    saved.start()
    saved.record_frame(action(31.25), observation())
    saved.save_episode()

    session = real_session(ds, follower_guard, robot, saved.writer)
    session.start()
    session.record_frame(action(55), observation())
    assert ds.has_pending_frames()
    calls_before = len(robot.sent)
    if failure != "manual_abort":
        invalid = action()
        if failure == "send":
            robot.fail_send = True
            expected_stage, expected_status = "send_action", SendStatus.UNCERTAIN
        else:
            key = "shoulder_lift.pos" if failure.endswith("f2") else "wrist_flex.pos"
            bound = next(bound for bound in follower_guard.bounds if bound.action_key == key)
            invalid_value = bound.maximum + 0.01
            invalid[key] = invalid_value
            expected_stage, expected_status = "raw_guard", SendStatus.NOT_ATTEMPTED
            if failure.startswith("final"):
                session.loop.robot_action_processor = lambda payload: {**payload[0], key: invalid_value}
                expected_stage = "final_command_guard"
                invalid = action()  # 入力時の指令は通るが、加工後の最終候補は拒否される条件にする。
        with pytest.raises(RecordFault):
            session.record_frame(invalid, observation())
        assert session.state is SessionState.FAULT
        assert session.loop.state is RecordState.FAULT
        assert session.fault is not None
        assert session.fault.stage == expected_stage
        assert session.fault.send_status is expected_status
        assert len(robot.sent) == calls_before + (failure == "send")
        calls_after = len(robot.sent)
        with pytest.raises(SessionStateError):
            session.record_frame(action(), observation())
        with pytest.raises(RecordStateError):
            session.loop.process_frame(action(), observation())
        with pytest.raises(SessionStateError):
            session.save_episode()
        assert len(robot.sent) == calls_after
    session.abort_episode()
    assert session.state is SessionState.ABORTED
    assert session.writer.state is WriterState.ABORTED
    assert not ds.has_pending_frames()
    assert session.writer.frames_in_episode == 0
    session.finalize()

    table = read_frames(ds)
    assert table.num_rows == ds.meta.total_frames == 1
    assert ds.meta.total_episodes == session.writer.saved_episodes == 1
    assert table["episode_index"].to_pylist() == [0]
    assert table["action"].to_pylist()[0][0] == pytest.approx(31.25)
    assert robot.disconnect_count == 0


def test_real_dataset_save_failure_keeps_uncertainty_and_forbids_retry(
    real_dataset: Any, follower_guard: ActionGuard, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """実際に保存した後で失敗した場合に、保存前の状態に戻せたと誤って扱わないことを確認する。"""
    ds = real_dataset
    robot = FakeRobot()
    session = real_session(ds, follower_guard, robot)
    session.start()
    session.record_frame(action(), observation())
    real_save = ds.save_episode
    save_calls = 0

    def fail_after_save(*args: Any, **kwargs: Any) -> None:
        nonlocal save_calls
        save_calls += 1
        real_save(*args, **kwargs)
        raise OSError("failure after real episode persistence")

    def forbidden_cleanup(*args: Any, **kwargs: Any) -> None:
        pytest.fail("Uncertain session must not discard buffers or finalize")

    monkeypatch.setattr(ds, "save_episode", fail_after_save)
    monkeypatch.setattr(ds, "clear_episode_buffer", forbidden_cleanup)
    monkeypatch.setattr(ds, "finalize", forbidden_cleanup)
    with pytest.raises(OSError, match="persistence"):
        session.save_episode()
    assert session.state is SessionState.FAULT
    assert session.loop.state is RecordState.FAULT
    assert session.writer.state is WriterState.FAULT
    assert session.fault is not None and session.fault.commit_uncertain
    assert session.writer.commit_uncertain
    assert session.writer.saved_episodes == 0
    assert session.writer.frames_in_episode == 1  # 保存が完了したか不明なので、保存処理側の件数を確定させない。
    assert ds.meta.total_episodes == 1  # 実際の保存先には書き込まれていることを確認する。
    assert list((ds.root / "data").rglob("*.parquet"))
    for operation in (session.save_episode, session.abort_episode, session.finalize):
        with pytest.raises(SessionStateError):
            operation()
    with pytest.raises(SessionStateError):
        session.record_frame(action(), observation())
    assert save_calls == len(robot.sent) == 1
    assert session.writer.commit_uncertain and session.state is SessionState.FAULT
    assert robot.disconnect_count == 0


def test_session_preserves_uncertain_send_on_keyboard_interrupt() -> None:
    class InterruptedRobot(FakeRobot):
        def send_action(self, requested: dict[str, float]) -> dict[str, float]:
            self.sent.append(dict(requested))
            raise KeyboardInterrupt()

    session, robot, ds = build(robot=InterruptedRobot())
    session.start()
    with pytest.raises(KeyboardInterrupt):
        session.record_frame(action(), observation())
    assert session.state is SessionState.FAULT
    assert session.loop.fault is not None and session.fault is not None
    assert session.fault.stage == session.loop.fault.stage == "send_action"
    assert session.fault.send_status is session.loop.fault.send_status is SendStatus.UNCERTAIN
    assert session.fault.frame_index == session.loop.fault.frame_index == 0
    with pytest.raises(SessionStateError):
        session.record_frame(action(), observation())
    with pytest.raises(RecordStateError):
        session.loop.process_frame(action(), observation())
    with pytest.raises(SessionStateError):
        session.save_episode()
    assert len(robot.sent) == 1
    session.abort_episode()
    session.finalize()
    assert ds.episodes == []
    assert robot.disconnect_count == 0
