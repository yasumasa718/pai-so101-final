"""OpenCVの符号化／読戻しは実物を使い、カメラとモーターI/Oは模擬する。"""

import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.control import diagnose_observation as diagnostic
from src.control.diagnostic_camera import CameraDiagnosticError, DiagnosticCamera
from test_diagnose_observation import ReadOnlyBus, hardware_args, records


class FakeCapture:
    def __init__(self):
        self.props = {}
        self.opens = []
        self.read_count = 0
        self.release_count = 0
        self.fail_open = False
        self.reject_prop = None
        self.buffer_accepted = True
        self.bad_read = None
        self.report_override = {}
        self.release_error = None

    def open(self, device, backend):
        self.opens.append((device, backend))
        return not self.fail_open

    def set(self, prop, value):
        if prop == cv2.CAP_PROP_BUFFERSIZE:
            return self.buffer_accepted
        if prop == self.reject_prop:
            return False
        self.props[prop] = value
        return True

    def get(self, prop):
        return self.report_override.get(prop, self.props[prop])

    def read(self):
        self.read_count += 1
        if self.bad_read is not None:
            return self.bad_read
        frame = np.zeros((int(self.props[cv2.CAP_PROP_FRAME_HEIGHT]),
                          int(self.props[cv2.CAP_PROP_FRAME_WIDTH]), 3), dtype=np.uint8)
        frame[:] = (self.read_count, 80, 160)  # BGR順序とフレーム対応も検証する。
        return True, frame

    def release(self):
        self.release_count += 1
        if self.release_error:
            raise self.release_error


def camera_for(tmp_path, monkeypatch, capture=None, **kwargs):
    cap = capture if capture is not None else FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda: cap)
    directory = tmp_path / "images"
    directory.mkdir()
    camera = DiagnosticCamera(device="/dev/video0", image_dir=directory, width=64, height=48,
                              fps=30, focus=230, warmup_frames=1, max_read_s=1, **kwargs)
    return camera, cap


def test_configure_manual_focus_and_verify_real_png_roundtrip(tmp_path, monkeypatch):
    camera, cap = camera_for(tmp_path, monkeypatch)
    config = camera.open()
    assert cap.opens == [("/dev/video0", cv2.CAP_V4L2)]
    assert config["reported"]["focus_automatic_continuous"] == 0
    assert config["reported"]["focus_absolute"] == 230
    assert config["reported"]["fps"] == 30
    assert config["sensor_sync_verified"] is False
    assert cap.read_count == 1
    image = camera.capture(0)
    path = camera.image_dir / image["file"]
    assert image["file"] == "frame_000000.png"
    restored = cv2.imread(str(path))
    assert (restored == np.array([2, 80, 160], dtype=np.uint8)).all()
    assert image["read_started_at"] <= image["read_completed_at"]
    camera.close()
    assert cap.release_count == 1


def test_buffer_request_failure_does_not_claim_freshness(tmp_path, monkeypatch):
    cap = FakeCapture()
    cap.buffer_accepted = False
    camera, _ = camera_for(tmp_path, monkeypatch, cap)
    config = camera.open()
    assert config["buffer_size_1_accepted"] is False
    assert config["sensor_timestamp_available"] is False
    camera.close()


@pytest.mark.parametrize("prop,value", [
    (cv2.CAP_PROP_FOCUS, 280), (cv2.CAP_PROP_AUTOFOCUS, 1),
    (cv2.CAP_PROP_FPS, 15), (cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV")),
])
def test_reported_settings_mismatch_rejected(tmp_path, monkeypatch, prop, value):
    cap = FakeCapture()
    cap.report_override[prop] = value
    camera, _ = camera_for(tmp_path, monkeypatch, cap)
    with pytest.raises(CameraDiagnosticError) as error:
        camera.open()
    assert error.value.stage == "camera_settings"
    assert error.value.details["reported"] == value
    assert "setting" in error.value.details
    camera.close()
    assert cap.release_count == 1
    assert not list(camera.image_dir.iterdir())


@pytest.mark.parametrize("failure,stage", [("open", "camera_open"), ("setting", "camera_settings")])
def test_setup_failure_does_not_retry(tmp_path, monkeypatch, failure, stage):
    cap = FakeCapture()
    cap.fail_open = failure == "open"
    cap.reject_prop = cv2.CAP_PROP_FOCUS if failure == "setting" else None
    camera, _ = camera_for(tmp_path, monkeypatch, cap)
    with pytest.raises(CameraDiagnosticError) as error:
        camera.open()
    assert error.value.stage == stage
    with pytest.raises(RuntimeError, match="reopened"):
        camera.open()
    camera.close()
    assert len(cap.opens) == 1 and cap.release_count == 1


@pytest.mark.parametrize("bad_read", [
    (False, None), (True, None), (True, np.zeros((1, 1, 3), dtype=np.uint8)),
    (True, np.zeros((48, 64, 3), dtype=np.float32)),
])
def test_bad_frames_latch_failure(tmp_path, monkeypatch, bad_read):
    camera, cap = camera_for(tmp_path, monkeypatch)
    camera.open()
    cap.bad_read = bad_read
    with pytest.raises(CameraDiagnosticError) as error:
        camera.capture(0)
    assert error.value.stage == "camera_read"
    count = cap.read_count
    with pytest.raises(RuntimeError, match="faulted"):
        camera.capture(1)
    assert cap.read_count == count
    assert not list(camera.image_dir.iterdir())
    camera.close()


@pytest.mark.parametrize("ticks", [[10, 12], [10, 9], [10, float("nan")], [True]])
def test_camera_time_faults(tmp_path, monkeypatch, ticks):
    camera, _ = camera_for(tmp_path, monkeypatch, clock=iter(ticks).__next__)
    with pytest.raises(CameraDiagnosticError) as error:
        camera.open()
    assert error.value.stage == "camera_read"
    camera.close()


@pytest.mark.parametrize("failure,stage", [("encode", "image_encode"), ("write", "image_write")])
def test_png_failures_latch_and_keep_evidence(tmp_path, monkeypatch, failure, stage):
    camera, cap = camera_for(tmp_path, monkeypatch)
    camera.open()
    if failure == "encode":
        monkeypatch.setattr(cv2, "imencode", lambda *a: (False, None))
    elif failure == "write":
        original_open = Path.open

        def fail_image_open(path, *args, **kwargs):
            if path.suffix == ".png":
                raise OSError("SECRET_PATH")
            return original_open(path, *args, **kwargs)

        monkeypatch.setattr(Path, "open", fail_image_open)
    with pytest.raises(CameraDiagnosticError) as error:
        camera.capture(0)
    assert error.value.stage == stage
    assert "SECRET_PATH" not in str(error.value)
    with pytest.raises(RuntimeError, match="faulted"):
        camera.capture(1)
    camera.close()


def test_existing_image_is_never_overwritten(tmp_path, monkeypatch):
    camera, _ = camera_for(tmp_path, monkeypatch)
    path = camera.image_dir / "frame_000000.png"
    path.write_bytes(b"previous evidence")
    camera.open()
    with pytest.raises(CameraDiagnosticError) as error:
        camera.capture(0)
    assert error.value.stage == "image_write"
    assert path.read_bytes() == b"previous evidence"
    camera.close()


def test_fake_camera_cli_saves_paired_frames_without_opening_video(tmp_path, monkeypatch):
    # 子プロセスでもfake分岐のためVideoCaptureを呼ばない。直接経路でもtrapを検証する。
    monkeypatch.setattr(cv2, "VideoCapture", lambda *a, **k: pytest.fail("must not open camera"))
    camera = DiagnosticCamera(device="fake", image_dir=tmp_path, width=64, height=48,
                              fps=30, focus=230, warmup_frames=1, max_read_s=1, fake=True)
    camera.open()
    camera.capture(0)
    camera.close()

    output = tmp_path / "paired.jsonl"
    assert diagnostic.main([
        "--output", str(output), "--camera", "fake", "--camera-focus", "230",
        "--camera-width", "64", "--camera-height", "48", "--camera-warmup-frames", "1",
        "--samples", "3", "--period-s", "0.001",
    ]) == 0
    events = records(output.read_text())
    observations = [row for row in events if row["event"] == "observation"]
    folder = diagnostic.image_directory(output)
    assert len(observations) == len(list(folder.glob("*.png"))) == 3
    for index, row in enumerate(observations):
        assert row["index"] == index
        image = row["image"]
        path = folder / image["file"]
        assert image["file"] == f"frame_{index:06d}.png"
        assert row["read_completed_at"] <= image["read_started_at"] <= image["read_completed_at"]
        assert row["joint_to_image_gap_s"] >= 0
        assert (cv2.imread(str(path)) == index + 1).all()
    assert events[-2]["event"] == "summary"
    assert events[-2]["timing_s"]["image_read_s"]["min"] >= 0
    assert events[-1]["images_saved"] == 3
    assert events[-1]["sensor_sync_verified"] is False


@pytest.mark.parametrize("failure,stage", [(None, None), ("read", "camera_read"),
                                          ("close", "camera_close"), ("open", "camera_open")])
def test_hardware_pair_path_never_writes_motors_and_closes_both(tmp_path, monkeypatch, failure, stage):
    bus = ReadOnlyBus()
    cap = FakeCapture()
    if failure == "read":
        cap.bad_read = (False, None)
    if failure == "close":
        cap.release_error = OSError()
    if failure == "open":
        cap.fail_open = True
    monkeypatch.setattr(cv2, "VideoCapture", lambda: cap)
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *a: bus)
    args = hardware_args(tmp_path)
    args.camera, args.camera_focus = "/dev/video0", 230
    args.camera_width, args.camera_height, args.camera_warmup_frames = 64, 48, 1
    diagnostic.image_directory(args.output).mkdir()
    stream = io.StringIO()
    assert diagnostic.run_worker(args, stream) == (0 if failure is None else 1)
    events = records(stream.getvalue())
    assert cap.release_count == 1
    assert bus.calls[-1] == "close_without_torque_change"
    if stage:
        assert any(row.get("stage") == stage and row["event"] == "fault" for row in events)
        assert all(row["event"] != "complete" for row in events)
    else:
        assert events[-1]["images_saved"] == args.samples
        assert bus.calls.count("positions") == args.samples


@pytest.mark.parametrize("options", [
    ["--camera", "/dev/video0", "--camera-focus", "230"],
    ["--camera", "fake"], ["--camera", "fake", "--camera-focus", "1024"],
    ["--camera-focus", "230"], ["--camera-width", "0"], ["--max-camera-read-s", "nan"],
])
def test_invalid_camera_arguments_never_launch(tmp_path, monkeypatch, options):
    monkeypatch.setattr(diagnostic, "supervise", lambda *a: pytest.fail("must not launch"))
    output = tmp_path / "absent.jsonl"
    with pytest.raises(SystemExit):
        diagnostic.main(["--output", str(output), *options])
    assert not output.exists()


def test_existing_image_folder_is_not_reused(tmp_path, monkeypatch):
    output = tmp_path / "test.jsonl"
    folder = diagnostic.image_directory(output)
    folder.mkdir()
    evidence = folder / "evidence.txt"
    evidence.write_text("keep")
    monkeypatch.setattr(diagnostic, "supervise", lambda *a: pytest.fail("must not launch"))
    assert diagnostic.main(["--output", str(output), "--camera", "fake", "--camera-focus", "230"]) == 1
    assert evidence.read_text() == "keep"


def test_pair_read_failure_keeps_previous_pair_and_does_not_retry(tmp_path, monkeypatch):
    bus = ReadOnlyBus()
    cap = FakeCapture()
    original_read = cap.read

    def fail_after_one_pair():
        if cap.read_count >= 2:  # warmup1枚 + 正常ペア1枚。
            cap.read_count += 1
            return False, None
        return original_read()

    cap.read = fail_after_one_pair
    monkeypatch.setattr(cv2, "VideoCapture", lambda: cap)
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *a: bus)
    args = hardware_args(tmp_path)
    args.samples = 3
    args.camera, args.camera_focus = "/dev/video0", 230
    args.camera_width, args.camera_height, args.camera_warmup_frames = 64, 48, 1
    folder = diagnostic.image_directory(args.output)
    folder.mkdir()
    stream = io.StringIO()
    assert diagnostic.run_worker(args, stream) == 1
    events = records(stream.getvalue())
    observations = [row for row in events if row["event"] == "observation"]
    assert [row["index"] for row in observations] == [0]
    assert len(list(folder.glob("*.png"))) == 1
    assert observations[0]["image"]["file"] == "frame_000000.png"
    assert cap.read_count == 3 and bus.calls.count("positions") == 2
    assert bus.calls[-1] == "close_without_torque_change"
    assert all(row["event"] not in ("summary", "complete") for row in events)


def test_hung_camera_read_is_stopped_by_supervisor(tmp_path):
    output = tmp_path / "hung_camera.jsonl"
    diagnostic.image_directory(output).mkdir()
    code = (
        "import sys,time\n"
        "from src.control import diagnose_observation as d\n"
        "from src.control.diagnostic_camera import DiagnosticCamera\n"
        "DiagnosticCamera._read = lambda self: time.sleep(60)\n"
        f"args=d.parser().parse_args(['--output',{str(output)!r},'--camera','fake','--camera-focus','230'])\n"
        "raise SystemExit(d.run_worker(args,sys.stdout))\n"
    )
    with output.open("w") as stream:
        assert diagnostic.supervise([sys.executable, "-c", code], stream, 2) == 124
    events = records(output.read_text())
    assert any(row.get("stage") == "camera_open" for row in events)
    assert events[-1]["worker_stopped"] is True
    assert not any(row["event"] == "complete" for row in events)
