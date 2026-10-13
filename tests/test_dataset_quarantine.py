"""一時ディレクトリで使用禁止の印を検証し、実際の画像保存処理に異常を起こして確認する。"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from lerobot.datasets import image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.configs.video import RGBEncoderConfig

from src.control.action_guard import ActionGuard, MOTOR_IDS
from src.control.bounded_image_writer import BoundedImageWriter, ImageWorkerError, ImageWorkerTimeout
from src.control.dataset_quarantine import (
    QUARANTINE_FILENAME, DatasetQuarantinedError, assert_dataset_not_quarantined, record_dataset_quarantine,
)
from src.control.dataset_record_writer import DatasetRecordWriter, DatasetWriterStateError, WriterState
from src.control.record_startup_check import RecordStartupCheck
from src.control.safe_record_loop import RecordState, RecordStateError, SafeRecordLoop, SendStatus
from src.control.safe_record_session import SafeRecordSession, SessionState, SessionStateError

JOINTS = tuple(f"{name}.pos" for name in MOTOR_IDS)
CAMERA = "observation.images.top"


def mark(root: Path) -> Path:
    return record_dataset_quarantine(root, stage="save_episode", error_type="OSError",
                                     commit_uncertain=True, cleanup_uncertain=False,
                                     frames_in_episode=2, saved_episodes=1)


def test_marker_is_exclusive_preserves_data_and_excludes_root(tmp_path: Path) -> None:
    data = tmp_path / "existing.parquet"
    data.write_bytes(b"existing partial dataset bytes")
    marker = mark(tmp_path)
    original = marker.read_bytes()
    info = json.loads(original)
    assert info["commit_uncertain"] and info["manual_review_required"]
    assert not info["rollback_confirmed"] and not info["workers_confirmed_stopped"]
    assert info["frames_in_episode"] == 2 and info["saved_episodes"] == 1
    assert "message" not in info and "root" not in info and str(tmp_path) not in original.decode()
    assert data.read_bytes() == b"existing partial dataset bytes"
    assert mark(tmp_path) == marker and marker.read_bytes() == original
    with pytest.raises(DatasetQuarantinedError):
        assert_dataset_not_quarantined(tmp_path)


@pytest.mark.parametrize("contents", [b"", b"broken JSON", b'{"status": "approved"}'])
def test_any_existing_marker_blocks_reuse(tmp_path: Path, contents: bytes) -> None:
    (tmp_path / QUARANTINE_FILENAME).write_bytes(contents)
    with pytest.raises(DatasetQuarantinedError):
        assert_dataset_not_quarantined(tmp_path)
    mark(tmp_path)
    assert (tmp_path / QUARANTINE_FILENAME).read_bytes() == contents


def test_marker_symlink_is_never_followed(tmp_path: Path) -> None:
    outside = tmp_path / "outside.json"
    outside.write_bytes(b"unchanged")
    marker = tmp_path / QUARANTINE_FILENAME
    marker.symlink_to(outside)
    with pytest.raises(DatasetQuarantinedError):
        assert_dataset_not_quarantined(tmp_path)
    assert mark(tmp_path) == marker
    assert outside.read_bytes() == b"unchanged"


def test_missing_root_is_not_created_by_quarantine(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError):
        mark(missing)
    assert not missing.exists()


def test_failed_marker_creation_still_excludes_root_in_this_process(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    open_path = Path.open

    def deny(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == QUARANTINE_FILENAME:
            raise PermissionError("injected read-only metadata directory")
        return open_path(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", deny)
    with pytest.raises(PermissionError):
        mark(tmp_path)
    assert not (tmp_path / QUARANTINE_FILENAME).exists()
    with pytest.raises(DatasetQuarantinedError):
        assert_dataset_not_quarantined(tmp_path)


@dataclass
class FakeRobot:
    calibration: dict[str, Any]
    config: Any = field(default_factory=lambda: type("Config", (), {"use_degrees": True})())
    calls: list[dict[str, float]] = field(default_factory=list)

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.calls.append(dict(action))
        return {**action, "gripper.pos": 49.0}

    def connect(self) -> None:
        pytest.fail("No real device connection")

    def disconnect(self) -> None:
        pytest.fail("FAULT must not disconnect or release torque")


@dataclass
class Recording:
    dataset: LeRobotDataset
    writer: DatasetRecordWriter
    session: SafeRecordSession
    robot: FakeRobot


@pytest.fixture
def make_recording(tmp_path: Path) -> Iterator[Callable[[str], Recording]]:
    resources: list[tuple[LeRobotDataset, Any]] = []

    def create(mode: str = "image") -> Recording:
        features = {key: {"dtype": "float32", "shape": (6,), "names": list(JOINTS)}
                    for key in ("action", "observation.state")}
        features[CAMERA] = {"dtype": mode, "shape": (64, 64, 3), "names": ["height", "width", "channels"]}
        dataset = LeRobotDataset.create(
            repo_id="local/so101_offline_quarantine", fps=30, features=features,
            root=tmp_path / f"dataset_{len(resources)}", robot_type="so_follower", use_videos=mode == "video",
            image_writer_threads=1, image_writer_processes=0,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="ultrafast"), encoder_threads=1,
        )
        resources.append((dataset, dataset.writer.image_writer))
        path = Path(__file__).resolve().parents[1] / "configs/calibration_snapshots/my_follower_for_004.json"
        payload = path.read_bytes()
        robot = FakeRobot(json.loads(payload))
        writer = DatasetRecordWriter(dataset, "Offline quarantine test", image_timeout_s=0.1)
        gate = ActionGuard.from_calibration_bytes(payload)
        check = RecordStartupCheck(calibration_path=path, expected_sha256=hashlib.sha256(payload).hexdigest(),
                                   margin_deg=0.0, margin_gripper=0.0)
        loop = SafeRecordLoop(gate, lambda p: p[0], lambda p: p[0], robot, writer)
        return Recording(dataset, writer, SafeRecordSession(loop, writer, startup_check=check), robot)

    try:
        yield create
    finally:
        for dataset, native in resources:
            # テスト専用：待機させた画像処理は、テストの finally で必ず再開させる。
            # 未保存フレーム、使用禁止の印、保存処理の FAULT は消さずに、開いたファイルを閉じる。
            # これはテストの後片付けであり、実際のアプリの復旧手順ではない。
            if not native._stopped:
                for _ in native.threads:
                    native.queue.put_nowait(None)
                for thread in native.threads:
                    thread.join(timeout=3.0)
                assert not any(thread.is_alive() for thread in native.threads)
                native._stopped = True
            dataset.writer.image_writer = None
            dataset.finalize()


def frame(recording: Recording) -> None:
    action = {key: 50.0 if key == "gripper.pos" else 0.0 for key in JOINTS}
    recording.session.record_frame(action, {**action, "top": np.full((64, 64, 3), 100, dtype=np.uint8)})


def assert_excluded(recording: Recording, *, commit_uncertain: bool) -> None:
    assert recording.session.state is SessionState.FAULT
    assert recording.session.loop.state is RecordState.FAULT
    assert recording.writer.state is WriterState.FAULT
    assert recording.writer.commit_uncertain is commit_uncertain
    assert recording.writer.cleanup_uncertain and recording.writer.quarantine_required
    assert recording.session.fault is not None and recording.session.fault.cleanup_uncertain
    sends = len(recording.robot.calls)
    for operation in (recording.session.save_episode, recording.session.abort_episode, recording.session.finalize):
        with pytest.raises(SessionStateError):
            operation()
    for operation in (recording.writer.save_episode, recording.writer.abort_episode, recording.writer.finalize):
        with pytest.raises(DatasetWriterStateError):
            operation()
    with pytest.raises(SessionStateError):
        frame(recording)
    with pytest.raises(RecordStateError):
        recording.session.loop.process_frame({}, {})
    assert len(recording.robot.calls) == sends
    with pytest.raises(DatasetQuarantinedError):
        DatasetRecordWriter(recording.dataset, "Cannot reopen")


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("operation", ["save", "abort", "finalize"])
def test_real_worker_hang_is_bounded_and_late_output_stays_quarantined(
    make_recording: Callable[[str], Recording], monkeypatch: pytest.MonkeyPatch, mode: str, operation: str,
) -> None:
    recording = make_recording(mode)
    recording.session.start()
    frame(recording)
    recording.session.save_episode()
    assert recording.writer.saved_episodes == 1
    saved_assets = {path: path.read_bytes() for path in (recording.dataset.root / "videos").rglob("*.mp4")}
    if operation != "finalize":
        old = recording.session
        loop = SafeRecordLoop(old.loop.guard, lambda p: p[0], lambda p: p[0], recording.robot, recording.writer)
        recording.session = SafeRecordSession(loop, recording.writer, startup_check=old.startup_check)
        recording.session.start()
    started, release = threading.Event(), threading.Event()
    real_write = image_writer.write_image

    def blocked(*args: Any, **kwargs: Any) -> None:
        started.set()
        if release.wait(3):
            real_write(*args, **kwargs)

    monkeypatch.setattr(image_writer, "write_image", blocked)
    native = recording.dataset.writer.image_writer.native
    try:
        if operation == "finalize":
            # 終了処理が異常な状況にも対処できるかを試すため、保存後に画像処理の待ちを作る。
            # 保存処理側のフレーム用一時データは空のままにする。
            path = recording.dataset.root / "late.png"
            native.save_image(np.zeros((64, 64, 3), dtype=np.uint8), path)
        else:
            frame(recording)
            path = Path(recording.dataset.writer.episode_buffer[CAMERA][0])
        assert started.wait(1)
        before = time.monotonic()
        with pytest.raises(ImageWorkerTimeout):
            getattr(recording.session, {"save": "save_episode", "abort": "abort_episode", "finalize": "finalize"}[operation])()
        assert time.monotonic() - before < 1.0
        assert_excluded(recording, commit_uncertain=operation == "finalize")
        marker = recording.writer.quarantine_record
        assert marker is not None and marker.is_file()
        content = marker.read_bytes()
        info = json.loads(content)
        assert info["cleanup_uncertain"] and info["manual_review_required"]
        assert not info["workers_confirmed_stopped"]
        assert recording.dataset.meta.total_episodes == 1
        if operation != "finalize":
            assert recording.dataset.has_pending_frames()
            assert recording.writer.frames_in_episode == 1
            assert path.parent.is_dir()  # 画像の書き込み中に、データを消す後片付けを行わないことを確認する。
    finally:
        release.set()
    # 時間切れの後に処理が完了しても、以前の FAULT を解除する理由にはならない。
    for thread in native.threads:
        # 遅れていた処理の後に終了用の目印を送り、応答できるテスト用スレッドの終了を待つ。
        native.queue.put_nowait(None)
        thread.join(timeout=2)
        assert not thread.is_alive()
    native._stopped = True
    assert path.is_file()
    assert marker.read_bytes() == content
    assert_excluded(recording, commit_uncertain=operation == "finalize")
    assert all(path.read_bytes() == data for path, data in saved_assets.items())


def test_stopped_worker_is_rejected_before_session_start(make_recording: Callable[[str], Recording]) -> None:
    recording = make_recording("image")
    recording.dataset.writer.image_writer.native.stop()
    with pytest.raises(ImageWorkerError, match="stopped"):
        recording.session.start()
    assert recording.robot.calls == []
    assert_excluded(recording, commit_uncertain=False)
    assert recording.session.fault.send_status is SendStatus.NOT_ATTEMPTED


def test_external_marker_added_after_construction_blocks_start(make_recording: Callable[[str], Recording]) -> None:
    recording = make_recording("image")
    mark(recording.dataset.root)
    with pytest.raises(DatasetQuarantinedError):
        recording.session.start()
    assert recording.robot.calls == []
    assert recording.session.state is SessionState.FAULT and recording.writer.quarantine_required


def test_external_marker_added_during_episode_blocks_next_send(make_recording: Callable[[str], Recording]) -> None:
    recording = make_recording("image")
    recording.session.start()
    mark(recording.dataset.root)
    with pytest.raises(DatasetQuarantinedError):
        frame(recording)
    assert recording.robot.calls == []
    assert recording.session.fault is not None
    assert recording.session.fault.send_status is SendStatus.NOT_ATTEMPTED
    assert recording.writer.quarantine_required
    with pytest.raises(SessionStateError):
        recording.session.abort_episode()


def test_real_worker_shutdown_timeout_keeps_saved_dataset_quarantined(
    make_recording: Callable[[str], Recording], monkeypatch: pytest.MonkeyPatch,
) -> None:
    exited_loop, release = threading.Event(), threading.Event()
    native_loop = image_writer.worker_thread_loop

    def delayed_exit(work_queue: Any) -> None:
        native_loop(work_queue)
        exited_loop.set()
        release.wait(3)

    monkeypatch.setattr(image_writer, "worker_thread_loop", delayed_exit)
    recording = make_recording("image")
    recording.session.start()
    frame(recording)
    recording.session.save_episode()
    try:
        before = time.monotonic()
        with pytest.raises(ImageWorkerTimeout, match="shutdown"):
            recording.session.finalize()
        assert time.monotonic() - before < 1.0
        assert exited_loop.is_set()
        assert_excluded(recording, commit_uncertain=True)
        assert recording.writer.saved_episodes == recording.dataset.meta.total_episodes == 1
    finally:
        release.set()


def test_partial_marker_write_blocks_reuse_without_claiming_record_success(
    make_recording: Callable[[str], Recording], monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = make_recording("image")
    recording.session.start()
    frame(recording)
    save = recording.dataset.save_episode
    dump = json.dump

    def partial_dump(value: Any, stream: Any, *args: Any, **kwargs: Any) -> None:
        if str(getattr(stream, "name", "")).endswith(QUARANTINE_FILENAME):
            stream.write("{")
            raise OSError("injected partial marker write")
        dump(value, stream, *args, **kwargs)

    def fail_save() -> None:
        save()
        raise OSError("original persistence failure")

    monkeypatch.setattr(recording.dataset, "save_episode", fail_save)
    monkeypatch.setattr(json, "dump", partial_dump)
    with pytest.raises(OSError, match="original persistence"):
        recording.session.save_episode()
    marker = recording.dataset.root / QUARANTINE_FILENAME
    assert marker.read_text(encoding="utf-8") == "{"
    assert recording.writer.quarantine_record is None
    assert recording.writer.quarantine_record_error == "OSError"
    assert recording.writer.commit_uncertain and recording.writer.quarantine_required
    with pytest.raises(DatasetQuarantinedError):
        DatasetRecordWriter(recording.dataset, "Cannot reopen damaged marker")


def test_quarantine_record_io_error_does_not_mask_original_save_fault(
    make_recording: Callable[[str], Recording], monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = make_recording("image")
    recording.session.start()
    frame(recording)
    save = recording.dataset.save_episode
    original_fault = OSError("original save failure with private device details")

    def fail_save() -> None:
        save()
        raise original_fault

    real_open = Path.open

    def deny_marker(path: Path, *args: Any, **kwargs: Any) -> Any:
        if path.name == QUARANTINE_FILENAME:
            raise PermissionError("marker cannot be created")
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(recording.dataset, "save_episode", fail_save)
    monkeypatch.setattr(Path, "open", deny_marker)
    with pytest.raises(OSError) as caught:
        recording.session.save_episode()
    assert caught.value is original_fault
    assert recording.writer.commit_uncertain and recording.writer.quarantine_required
    assert recording.writer.quarantine_record is None
    assert recording.writer.quarantine_record_error == "PermissionError"
    with pytest.raises(DatasetQuarantinedError):
        DatasetRecordWriter(recording.dataset, "Cannot reopen after failed marker")
    assert recording.dataset.meta.total_episodes == 1  # 実際に LeRobot 側では保存が行われたことを確認する。
    with pytest.raises(SessionStateError):
        recording.session.abort_episode()


def test_partial_native_save_creates_marker_without_raw_exception_details(
    make_recording: Callable[[str], Recording], monkeypatch: pytest.MonkeyPatch,
) -> None:
    recording = make_recording("image")
    recording.session.start()
    frame(recording)
    save = recording.dataset.save_episode

    def fail_save() -> None:
        save()
        raise OSError("private USB serial and credentials must not be copied")

    monkeypatch.setattr(recording.dataset, "save_episode", fail_save)
    with pytest.raises(OSError):
        recording.session.save_episode()
    assert recording.writer.quarantine_record is not None
    payload = recording.writer.quarantine_record.read_text(encoding="utf-8")
    assert "USB" not in payload and "credentials" not in payload and str(recording.dataset.root) not in payload
    assert json.loads(payload)["commit_uncertain"]
    assert not json.loads(payload)["rollback_confirmed"]
    assert recording.dataset.meta.total_episodes == 1
    assert recording.writer.saved_episodes == 0
    with pytest.raises(DatasetQuarantinedError):
        assert_dataset_not_quarantined(recording.dataset.root)
