"""実LeRobotで、画像・動画の保存から終了までを検証する結合テスト。

ロボットは模擬し、既知の合成画像を一時フォルダーへPNG・Parquet・H.264として保存する。
同期保存とスレッド保存、保存後の読戻し、欠損画像、部分保存、後片付け失敗を試す。
実機・カメラ・ネットワークは使わない。
"""

from __future__ import annotations

import hashlib
import io
import json
import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image
from lerobot.configs.video import RGBEncoderConfig
from lerobot.datasets import image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from src.control.action_guard import ActionGuard, MOTOR_IDS
from src.control.dataset_record_writer import DatasetRecordWriter, WriterState
from src.control.record_startup_check import RecordStartupCheck
from src.control.safe_record_loop import RecordFault, RecordState, RecordStateError, SafeRecordLoop, SendStatus
from src.control.safe_record_session import SafeRecordSession, SessionState, SessionStateError

JOINTS = tuple(f"{name}.pos" for name in MOTOR_IDS)
CAMERA = "observation.images.top"
SHAPE = (64, 64, 3)


@dataclass
class FakeConfig:
    """開始前検査に必要な度単位設定だけを持つ代役。"""
    use_degrees: bool = True


@dataclass
class FakeRobot:
    """要求値と違う爪位置を返し、元の要求ではなく送信戻り値が保存されるか確認できる送信先。"""
    calibration: dict[str, Any]
    config: FakeConfig = field(default_factory=FakeConfig)
    calls: list[dict[str, float]] = field(default_factory=list)
    fail_send: bool = False

    def send_action(self, candidate: dict[str, float]) -> dict[str, float]:
        self.calls.append(dict(candidate))
        if self.fail_send:
            raise OSError("simulated uncertain send")
        return {**candidate, "gripper.pos": candidate["gripper.pos"] - 1.123456789}

    def connect(self) -> None:
        pytest.fail("Offline test must not connect hardware")

    def disconnect(self) -> None:
        pytest.fail("Offline test must not release torque or disconnect hardware")


@dataclass
class Recording:
    """Dataset・セッション・送信先と画像／動画モードをまとめるテスト用の入れ物。"""
    dataset: LeRobotDataset
    session: SafeRecordSession
    robot: FakeRobot
    mode: str

    def next_episode(self) -> None:
        """writerと開始前検査を引き継ぎ、未使用のループとセッションで次エピソードを準備する。"""
        previous = self.session
        loop = SafeRecordLoop(previous.loop.guard, lambda p: p[0], lambda p: p[0], self.robot, previous.writer)
        self.session = SafeRecordSession(loop, previous.writer, startup_check=previous.startup_check)


@pytest.fixture
def make_recording(tmp_path: Path) -> Iterator[Callable[..., Recording]]:
    """保存方式・スレッド数を指定して実Datasetを作る関数を提供し、終了時にファイル資源を解放するfixture。"""
    closers: list[Callable[[], None]] = []

    def create(mode: str, *, threads: int = 2, streaming: bool = False) -> Recording:
        features = {
            "action": {"dtype": "float32", "shape": (6,), "names": list(reversed(JOINTS))},
            "observation.state": {"dtype": "float32", "shape": (6,), "names": list(JOINTS)},
            CAMERA: {"dtype": mode, "shape": SHAPE, "names": ["height", "width", "channels"]},
        }
        dataset = LeRobotDataset.create(
            repo_id="local/so101_offline_images", fps=30, features=features,
            root=tmp_path / f"dataset_{len(closers)}", robot_type="so_follower",
            use_videos=mode == "video", image_writer_processes=0, image_writer_threads=threads,
            rgb_encoder=RGBEncoderConfig(vcodec="h264", crf=18, preset="ultrafast"),
            encoder_threads=1, video_backend="pyav",
            streaming_encoding=streaming,
        )
        # 使用禁止になったケースも、テスト終了時には開いたファイルを閉じる。
        # 保存処理の FAULT や不確実な状態は解除せず、保存前に戻せたとも扱わない。
        closers.append(dataset.finalize)
        path = Path(__file__).resolve().parents[1] / "configs/calibration_snapshots/my_follower_for_004.json"
        payload = path.read_bytes()
        robot = FakeRobot(json.loads(payload))
        writer = DatasetRecordWriter(dataset, "Offline simulated camera")
        gate = ActionGuard.from_calibration_bytes(payload)
        check = RecordStartupCheck(
            calibration_path=path, expected_sha256=hashlib.sha256(payload).hexdigest(),
            margin_deg=0.0, margin_gripper=0.0,
        )
        loop = SafeRecordLoop(gate, lambda p: p[0], lambda p: p[0], robot, writer)
        return Recording(dataset, SafeRecordSession(loop, writer, startup_check=check), robot, mode)

    try:
        yield create
    finally:
        for close in reversed(closers):
            close()


def synthetic_image(index: int) -> np.ndarray:
    """フレームごとに色と領域の模様が変わるRGB画像を作り、順序違い・色順違いを検出できるようにする。"""
    pixels = np.empty(SHAPE, dtype=np.uint8)
    pixels[:] = ((40 + 20 * index) % 220, (80 + 10 * index) % 220, 140)
    pixels[:32, :32] = (180, (30 + 15 * index) % 220, 60)
    pixels[32:, 32:] = (60, 150, (50 + 20 * index) % 220)
    return pixels


def frame(recording: Recording, index: int) -> tuple[np.ndarray, list[float], list[float]]:
    """1フレームを記録し、比較用の画像・送信戻り値・観測配列を返す。元画像を書き換えてコピー保持も試す。"""
    requested = {key: (60.23456789 + index % 10 if key == "gripper.pos" else 1.23456789 + i)
                 for i, key in enumerate(JOINTS)}
    obs = {key: value / 2 for key, value in requested.items()}
    pixels = synthetic_image(index)
    expected_image = pixels.copy()
    result = recording.session.record_frame(requested, {**obs, "top": pixels})
    pixels[:] = 0  # 呼び出し側の画像を書き換えても、保存済み・預かり中の画素が変わらないことを確認する。
    features = recording.dataset.features
    return (expected_image, [result.sent_action[key] for key in features["action"]["names"]],
            [obs[key] for key in features["observation.state"]["names"]])


def data_table(recording: Recording) -> pa.Table:
    """保存された全Parquetをファイル順に読み、1つのテーブルへ連結する。"""
    paths = sorted((recording.dataset.root / "data").rglob("*.parquet"))
    assert paths
    return pa.concat_tables([pq.read_table(path) for path in paths])


def decode_images(recording: Recording, table: pa.Table) -> list[np.ndarray]:
    """画像はParquet内のPNG、動画はMP4から復号する。動画は30 fpsと各フレームの表示時刻も確認する。"""
    if recording.mode == "image":
        result = []
        for value in table[CAMERA].to_pylist():
            assert value["bytes"]  # PNG は Parquet 内に埋め込まれており、一時ファイルの場所に依存しないことを確認する。
            with Image.open(io.BytesIO(value["bytes"])) as image:
                result.append(np.array(image.convert("RGB")))
        return result
    paths = sorted((recording.dataset.root / "videos").rglob("*.mp4"))
    assert len(paths) == 1
    with av.open(str(paths[0])) as container:
        frames = list(container.decode(video=0))
        assert container.streams.video[0].average_rate == 30
        np.testing.assert_allclose([float(f.pts * f.time_base) for f in frames], np.arange(len(frames)) / 30, atol=1e-6)
        return [f.to_ndarray(format="rgb24") for f in frames]


def assert_images(recording: Recording, table: pa.Table, expected: list[np.ndarray]) -> None:
    """フレーム数・寸法・画素を比較する。PNGは完全一致、H.264は圧縮誤差を許容して模様の対応を確かめる。"""
    decoded = decode_images(recording, table)
    assert len(decoded) == len(expected) == table.num_rows
    for actual, original in zip(decoded, expected, strict=True):
        assert actual.shape == SHAPE
        if recording.mode == "image":
            np.testing.assert_array_equal(actual, original)
        else:
            error = np.abs(actual.astype(np.int16) - original.astype(np.int16))
            assert error.mean() < 3.0  # H.264/yuv420p は圧縮で画素値が少し変わるため、誤差を許しつつ各フレームの模様を確かめる。
            assert np.quantile(error, 0.95) <= 10


def assert_quarantined(recording: Recording) -> None:
    """保存結果不明を各層で保持し、再保存・破棄・終了・追加送信によって解除されないことを共通検査する。"""
    session = recording.session
    assert session.state is SessionState.FAULT
    assert session.loop.state is RecordState.FAULT
    assert session.writer.state is WriterState.FAULT
    assert session.writer.commit_uncertain
    assert session.fault is not None and session.fault.commit_uncertain
    calls = len(recording.robot.calls)
    for operation in (session.save_episode, session.abort_episode, session.finalize):
        with pytest.raises(SessionStateError):
            operation()
    with pytest.raises(SessionStateError):
        frame(recording, 0)
    with pytest.raises(RecordStateError):
        session.loop.process_frame({}, {})
    assert len(recording.robot.calls) == calls
    assert session.writer.commit_uncertain  # 操作を試みても、異常の記録が消えないことを確認する。


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("threads", [0, 2], ids=["synchronous", "async_threads"])
def test_two_image_episodes_roundtrip(make_recording: Callable[..., Recording], mode: str, threads: int) -> None:
    """画像／動画と同期／非同期保存の組合せで2エピソードを読み戻し、指令・観測・画像・時刻・番号の対応を確認する。"""
    recording = make_recording(mode, threads=threads)
    expected_images, expected_actions, expected_states = [], [], []
    for episode in range(2):
        if episode:
            recording.next_episode()
        recording.session.start()
        for index in range(3):
            image, action, obs = frame(recording, episode * 3 + index)
            expected_images.append(image)
            expected_actions.append(action)
            expected_states.append(obs)
        assert recording.dataset.meta.total_episodes == episode
        recording.session.save_episode()
        assert not recording.dataset.has_pending_frames()
    recording.session.finalize()
    table = data_table(recording)
    np.testing.assert_allclose(table["action"].to_pylist(), expected_actions, rtol=1e-7, atol=1e-6)
    np.testing.assert_allclose(table["observation.state"].to_pylist(), expected_states, rtol=1e-7, atol=1e-6)
    assert table["frame_index"].to_pylist() == [0, 1, 2] * 2
    assert table["episode_index"].to_pylist() == [0, 0, 0, 1, 1, 1]
    np.testing.assert_allclose(table["timestamp"].to_pylist(), [0, 1 / 30, 2 / 30] * 2, atol=1e-7)
    assert_images(recording, table, expected_images)
    assert list(recording.dataset.root.rglob("*.png")) == []
    assert recording.dataset.meta.total_episodes == 2
    assert recording.dataset.meta.total_frames == len(recording.robot.calls) == 6
    episodes = pa.concat_tables([pq.read_table(path) for path in sorted((recording.dataset.root / "meta/episodes").rglob("*.parquet"))])
    assert episodes["length"].to_pylist() == [3, 3]
    if mode == "video":
        np.testing.assert_allclose(episodes[f"videos/{CAMERA}/from_timestamp"].to_pylist(), [0, 0.1], atol=1e-6)
        np.testing.assert_allclose(episodes[f"videos/{CAMERA}/to_timestamp"].to_pylist(), [0.1, 0.2], atol=1e-6)


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("failure", ["manual", "range", "send"])
def test_abort_waits_for_inflight_images_and_preserves_saved_episode(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch, mode: str, failure: str,
) -> None:
    """画像保存をEventで待たせ、中断が保存完了を待ってから一時画像を消し、前の保存済みエピソードを残すことを確認する。

    手動中断・範囲拒否・送信失敗の各経路で、削除後に遅い書込みがディレクトリを再作成しないことも調べる。
    """
    recording = make_recording(mode, threads=1)
    recording.session.start()
    expected = [frame(recording, index)[0] for index in range(2)]
    recording.session.save_episode()
    recording.next_episode()
    recording.session.start()
    started, release, waiting, finished = (threading.Event() for _ in range(4))
    # 保存開始→中断の待機開始→保存再開→中断完了をEventで確認し、単なるsleepの推測を避ける。
    errors: list[BaseException] = []
    write = image_writer.write_image
    native = recording.dataset.writer.image_writer
    wait = native.wait_until_done

    def delayed_write(*args: Any, **kwargs: Any) -> None:
        started.set()
        if not release.wait(5):
            errors.append(TimeoutError("test image writer was not released"))
            return
        write(*args, **kwargs)

    def observed_wait() -> None:
        waiting.set()
        wait()

    def abort() -> None:
        try:
            recording.session.abort_episode()
        except BaseException as exc:
            errors.append(exc)
        finally:
            finished.set()

    monkeypatch.setattr(image_writer, "write_image", delayed_write)
    monkeypatch.setattr(native, "wait_until_done", observed_wait)
    worker = threading.Thread(target=abort)
    try:
        frame(recording, 3)
        assert started.wait(2)
        temporary = Path(recording.dataset.writer.episode_buffer[CAMERA][0]).parent
        assert temporary.is_dir()
        if failure != "manual":
            requested = {key: 50.0 if key == "gripper.pos" else 0.0 for key in JOINTS}
            if failure == "range":
                requested["wrist_flex.pos"] = 999.0
            else:
                recording.robot.fail_send = True
            with pytest.raises(RecordFault):
                recording.session.record_frame(requested, {**requested, "top": synthetic_image(4)})
            assert recording.session.fault is not None
            assert recording.session.fault.send_status is (SendStatus.UNCERTAIN if failure == "send" else SendStatus.NOT_ATTEMPTED)
            with pytest.raises(SessionStateError):
                recording.session.save_episode()
        worker.start()
        assert waiting.wait(2)
        assert not finished.is_set()  # 待機中の画像保存が終わるまでは、ファイルを削除してはいけない。
        assert temporary.is_dir()
    finally:
        release.set()
        if worker.ident is not None:
            worker.join(5)
    assert not worker.is_alive() and not errors
    assert recording.session.state is SessionState.ABORTED
    assert not recording.dataset.has_pending_frames()
    assert not temporary.exists()
    native.wait_until_done()  # 遅れた書き込みで、削除済みのディレクトリが再び作られないことを確認する。
    assert not temporary.exists()
    assert recording.session.writer.frames_in_episode == 0
    assert len(recording.robot.calls) == 3 + (failure == "send")
    recording.session.finalize()
    table = data_table(recording)
    assert table.num_rows == 2 and table["episode_index"].to_pylist() == [0, 0]
    assert_images(recording, table, expected)
    assert recording.dataset.meta.total_episodes == 1
    assert recording.dataset.meta.total_frames == 2
    assert list(recording.dataset.root.rglob("*.png")) == []


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("threads", [0, 2], ids=["synchronous", "async_threads"])
def test_png_io_failure_does_not_save_normal_episode(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture,
    mode: str, threads: int,
) -> None:
    """LeRobot側がPNG書込み失敗をログだけに残しても、全画像検査がエピソード確定前に検出し、正常データとして保存しないことを確認する。"""
    recording = make_recording(mode, threads=threads)
    recording.session.start()
    real_save = Image.Image.save

    def fail_png(image: Image.Image, path: Any, *args: Any, **kwargs: Any) -> None:
        if isinstance(path, Path) and path.is_relative_to(recording.dataset.root):
            raise OSError("injected PNG disk failure")
        real_save(image, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_png)
    frame(recording, 0)
    frame(recording, 1)
    with pytest.raises((OSError, ValueError)):
        recording.session.save_episode()
    assert "injected PNG disk failure" in caplog.text  # LeRobot の write_image は例外を伝えず、ログだけを残す。
    assert recording.session.state is SessionState.FAULT
    assert recording.session.writer.state is WriterState.FAULT
    assert not recording.session.writer.commit_uncertain  # LeRobot 側のエピソード保存は始まっていないことを確認する。
    assert recording.dataset.meta.total_episodes == 0
    assert recording.session.writer.saved_episodes == 0
    assert recording.dataset.has_pending_frames()
    assert list((recording.dataset.root / "data").rglob("*.parquet")) == []
    with pytest.raises(SessionStateError):
        recording.session.save_episode()
    recording.session.abort_episode()
    recording.session.finalize()
    assert not recording.dataset.has_pending_frames()
    assert not list(recording.dataset.root.rglob("*.png"))


def test_unsampled_missing_video_frame_is_rejected_before_commit(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """統計計算の抽出対象外になる1画像を欠落させても、全画像検査で検出して短い動画を保存しないことを確認する。"""
    from lerobot.datasets.compute_stats import sample_indices

    recording = make_recording("video")
    length = 101
    missing_index = next(iter(set(range(length)) - set(sample_indices(length))))
    # 統計の抜き取り検査だけなら見逃す画像を選び、全件検査が必要な理由を再現する。
    missing_path = recording.dataset.writer._get_image_file_path(0, CAMERA, missing_index)
    real_save = Image.Image.save

    def fail_one_png(image: Image.Image, path: Any, *args: Any, **kwargs: Any) -> None:
        if path == missing_path:
            raise OSError("injected unsampled frame loss")
        real_save(image, path, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", fail_one_png)
    recording.session.start()
    for index in range(length):
        frame(recording, index)
    recording.dataset.writer.image_writer.wait_until_done()
    assert not missing_path.exists()
    with pytest.raises((OSError, ValueError)):
        recording.session.save_episode()
    assert recording.dataset.meta.total_episodes == 0
    assert recording.session.state is SessionState.FAULT
    assert not recording.session.writer.commit_uncertain
    assert not list((recording.dataset.root / "videos").rglob("*.mp4"))
    assert not list((recording.dataset.root / "data").rglob("*.parquet"))
    recording.session.abort_episode()
    recording.session.finalize()
    assert not list(recording.dataset.root.rglob("*.png"))


@pytest.mark.parametrize("mode", ["image", "video"])
@pytest.mark.parametrize("corruption", ["unreadable", "wrong_size"])
def test_all_images_are_decoded_and_checked_before_native_save(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch, mode: str, corruption: str,
) -> None:
    """保存済みPNGの破損・寸法違いを実際に作り、LeRobotのエピソード保存を呼ぶ前に拒否することを確認する。"""
    recording = make_recording(mode)
    recording.session.start()
    frame(recording, 0)
    frame(recording, 1)
    recording.dataset.writer.image_writer.wait_until_done()
    path = Path(recording.dataset.writer.episode_buffer[CAMERA][1])
    if corruption == "unreadable":
        path.write_bytes(b"injected damaged PNG")
    else:
        Image.fromarray(np.zeros((32, 64, 3), dtype=np.uint8)).save(path)
    calls = 0
    save = recording.dataset.save_episode

    def observed_save(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        save(*args, **kwargs)

    monkeypatch.setattr(recording.dataset, "save_episode", observed_save)
    with pytest.raises((OSError, ValueError)):
        recording.session.save_episode()
    assert calls == 0 and len(recording.robot.calls) == 2
    assert recording.session.state is SessionState.FAULT
    assert not recording.session.writer.commit_uncertain
    assert recording.session.writer.fault_message is not None
    assert "validate_pending_images" in recording.session.writer.fault_message
    assert recording.dataset.meta.total_episodes == 0
    with pytest.raises(SessionStateError):
        frame(recording, 2)
    recording.session.abort_episode()
    recording.session.finalize()
    assert not list(recording.dataset.root.rglob("*.png"))


def test_streaming_encoding_is_rejected_before_recording(make_recording: Callable[..., Recording]) -> None:
    """PNG全件検査に対応しないストリーミング動画変換は、収録部品の構築時に拒否することを確認する。"""
    with pytest.raises(ValueError, match="Streaming encoding is not supported"):
        make_recording("video", streaming=True)


@pytest.mark.parametrize("mode", ["image", "video"])
def test_partial_parquet_failure_keeps_uncertainty(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """実Parquet書込み直後に例外を出し、部分ファイルと一時画像を残して保存結果不明を保持することを確認する。"""
    recording = make_recording(mode)
    recording.session.start()
    frame(recording, 0)
    frame(recording, 1)
    native = recording.dataset.writer
    save_data = native._save_episode_data
    calls = 0

    def fail_after_parquet(buffer: dict[str, Any]) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        save_data(buffer)
        raise OSError("injected failure after real Parquet write")

    monkeypatch.setattr(native, "_save_episode_data", fail_after_parquet)
    with pytest.raises(OSError, match="Parquet write"):
        recording.session.save_episode()
    assert_quarantined(recording)
    assert calls == 1
    assert recording.dataset.meta.total_episodes == 0
    assert list((recording.dataset.root / "data").rglob("*.parquet"))
    assert list(recording.dataset.root.rglob("*.png"))  # 異常なデータを勝手に削除せず、使用禁止にすることを確認する。


def test_video_encoding_failure_preserves_previous_video_and_partial_files(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """次エピソードの動画変換に失敗しても、前の動画と失敗途中のファイルを勝手に削除せず使用禁止にすることを確認する。"""
    recording = make_recording("video")
    recording.session.start()
    for index in range(2):
        frame(recording, index)
    recording.session.save_episode()
    previous = next((recording.dataset.root / "videos").rglob("*.mp4"))
    before = previous.read_bytes()
    recording.next_episode()
    recording.session.start()
    frame(recording, 3)
    partial = recording.dataset.root / "injected_partial.mp4"
    calls = 0

    def fail_encode(*args: Any, **kwargs: Any) -> Path:
        nonlocal calls
        calls += 1
        partial.write_bytes(b"incomplete encoded video")
        raise OSError("injected video encoding failure")

    monkeypatch.setattr(recording.dataset.writer, "_encode_temporary_episode_video", fail_encode)
    with pytest.raises(OSError, match="encoding failure"):
        recording.session.save_episode()
    assert_quarantined(recording)
    assert calls == 1 and partial.is_file()
    assert previous.read_bytes() == before
    assert recording.dataset.meta.total_episodes == recording.session.writer.saved_episodes == 1


@pytest.mark.parametrize("mode", ["image", "video"])
def test_cleanup_failure_never_reports_successful_abort(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """実際の一時画像削除後に例外を出し、保存先が片付いていても中断成功とは報告しないことを確認する。"""
    recording = make_recording(mode)
    recording.session.start()
    frame(recording, 0)
    clear = recording.dataset.clear_episode_buffer
    calls = 0

    def fail_after_clear(*args: Any, **kwargs: Any) -> None:
        nonlocal calls
        calls += 1
        clear(*args, **kwargs)
        raise OSError("injected exception after image cleanup")

    monkeypatch.setattr(recording.dataset, "clear_episode_buffer", fail_after_clear)
    with pytest.raises(OSError, match="image cleanup"):
        recording.session.abort_episode()
    assert recording.session.state is SessionState.FAULT
    assert recording.session.writer.state is WriterState.FAULT
    assert recording.session.writer.frames_in_episode == 1
    assert not recording.dataset.has_pending_frames()  # LeRobot 側で後片付けが進んでいても、保存処理側では成功したと保証できない。
    assert calls == 1 and recording.dataset.meta.total_episodes == 0
    with pytest.raises(SessionStateError):
        recording.session.finalize()
    with pytest.raises(SessionStateError):
        recording.session.save_episode()


@pytest.mark.parametrize("mode", ["image", "video"])
def test_finalization_failure_retains_fault_even_after_native_close(
    make_recording: Callable[..., Recording], monkeypatch: pytest.MonkeyPatch, mode: str,
) -> None:
    """実際の終了処理直後に例外を出し、保存画像が読めてもFAULTと保存結果不明を保持することを確認する。"""
    recording = make_recording(mode)
    recording.session.start()
    expected = [frame(recording, index)[0] for index in range(2)]
    recording.session.save_episode()
    close = recording.dataset.finalize
    calls = 0

    def fail_after_close() -> None:
        nonlocal calls
        calls += 1
        close()
        raise OSError("injected exception after native close")

    monkeypatch.setattr(recording.dataset, "finalize", fail_after_close)
    with pytest.raises(OSError, match="native close"):
        recording.session.finalize()
    assert_quarantined(recording)
    assert calls == 1
    assert_images(recording, data_table(recording), expected)
    assert recording.dataset.meta.total_episodes == 1
