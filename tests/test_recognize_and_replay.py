"""撮影・検出・モーターを模擬し、認識から固定再生への境界を確認する。"""

import io
import json

import pytest
from PIL import Image

from src.control import recognize_and_replay as collection
from src.control import guarded_replay as replay
from src.vision.validate_detection import ROI
from test_diagnose_observation import records
from test_guarded_replay import setup as replay_setup


CAN = {"label": "a soda can", "score": 0.71,
       "box": {"xmin": 269, "ymin": 219, "xmax": 338, "ymax": 335}}


@pytest.fixture
def setup(replay_setup, monkeypatch, tmp_path):
    args, bus, _, _, clock = replay_setup
    args.image = None
    args.camera = "/dev/video0"
    args.roi = ROI(260, 210, 350, 350)
    args.min_score = 0.1
    args.output_dir = tmp_path / "collection"
    args.output_dir.mkdir()
    cameras = []
    detections = [CAN]

    class Camera:
        def __init__(self, **kwargs):
            self.directory = kwargs["image_dir"]
            self.closed = False
            cameras.append(self)

        def open(self):
            return {"fake": True}

        def capture(self, index):
            timestamp = clock()
            Image.new("RGB", (640, 480)).save(self.directory / "frame_000000.png")
            return {"file": "frame_000000.png", "read_started_at": timestamp,
                    "read_completed_at": clock()}

        def close(self):
            self.closed = True

    def detector(image, **kwargs):
        assert all(c.closed for c in cameras)  # カメラを閉じてから推論・接続する。
        return list(detections)

    monkeypatch.setattr(collection, "DiagnosticCamera", Camera)
    monkeypatch.setattr(collection, "load_detector", lambda: detector)
    return args, bus, clock, cameras, detections


def run(args):
    stream = io.StringIO()
    code = collection.run_worker(args, stream)
    return code, records(stream.getvalue())


def test_fresh_pass_runs_existing_replay_and_releases_all_axes(setup):
    args, bus, _, cameras, _ = setup
    code, rows = run(args)
    assert code == 0
    assert len(cameras) == 1 and cameras[0].closed
    assert [r["name"] for r in rows if r["event"] == "checkpoint_complete"] == list(replay.PHASES)
    assert rows[-1]["replay_attempted"] is True and rows[-1]["replay_completed"] is True
    assert rows[-1]["recognition_gate"] == "PASS"
    assert not any(bus.torque.values())
    assert set(next(r for r in rows if r["event"] == "torque_off")["verified"]) == set(bus.torque)
    assert json.loads((args.output_dir / "recognition" / "detection_gate.json").read_text())["robot_execution_allowed"] is False


@pytest.mark.parametrize("count,gate", [(0, "FAIL"), (2, "REVIEW")])
def test_no_target_or_multiple_targets_never_connects_even_when_execute_requested(setup, count, gate):
    args, bus, _, _, detections = setup
    detections[:] = [CAN]*count
    code, rows = run(args)
    assert code == 0
    assert rows[-1]["recognition_gate"] == gate
    assert rows[-1]["replay_attempted"] is False and rows[-1]["replay_completed"] is False
    assert not bus.calls and not bus.targets and not bus.writes
    assert not any(r["event"] == "fault" for r in rows)


@pytest.mark.parametrize("saved_image", [False, True])
def test_default_recognition_only_never_connects_for_pass(setup, saved_image, tmp_path):
    args, bus, _, cameras, _ = setup
    args.execute = False
    if saved_image:
        args.image = tmp_path / "saved.png"
        Image.new("RGB", (640, 480)).save(args.image)
    code, rows = run(args)
    assert code == 0 and rows[-1]["replay_attempted"] is False
    assert rows[-1]["recognition_gate"] == "PASS"
    assert bool(cameras) is not saved_image
    assert not bus.calls and not bus.targets and not bus.writes


def test_inference_too_slow_rejects_before_motor_connection(setup, monkeypatch):
    args, bus, clock, _, _ = setup
    def detector(*a, **kw):
        clock.now += collection.MAX_IMAGE_AGE_S+1
        return [CAN]
    monkeypatch.setattr(collection, "load_detector", lambda: detector)
    code, rows = run(args)
    assert code == 1 and rows[-1]["stage"] == "image_age"
    assert not bus.calls and not bus.targets and not bus.writes


@pytest.mark.parametrize("stage", ["model_load", "capture", "recognition"])
def test_capture_and_model_failures_never_connect(setup, monkeypatch, stage):
    args, bus, _, cameras, _ = setup
    def failed(*a, **kw):
        raise RuntimeError("SECRET_PORT")
    if stage == "model_load":
        monkeypatch.setattr(collection, "load_detector", failed)
    elif stage == "capture":
        monkeypatch.setattr(collection.DiagnosticCamera, "capture", failed)
    else:
        monkeypatch.setattr(collection, "load_detector", lambda: failed)
    code, rows = run(args)
    assert code == 1 and rows[-1]["stage"] == stage
    assert all(c.closed for c in cameras)
    assert not bus.calls and not bus.targets and not bus.writes
    assert "SECRET_PORT" not in json.dumps(rows)


def test_pass_does_not_bypass_start_pose_check(setup):
    args, bus, _, _, _ = setup
    bus.raw["shoulder_pan"] += 100
    code, rows = run(args)
    assert code == 1 and rows[-1]["replay_attempted"] is True and rows[-1]["replay_completed"] is False
    assert next(r for r in rows if r["event"] == "fault")["stage"] == "initial_position"
    assert not bus.targets and not bus.writes


def test_replay_failure_still_releases_and_collection_cannot_claim_success(setup):
    args, bus, _, _, _ = setup
    bus.fail_send = True
    code, rows = run(args)
    assert code == 1 and rows[-1]["replay_attempted"] is True and rows[-1]["replay_completed"] is False
    assert not any(bus.torque.values())
    assert len(bus.targets) == 2


def argv(args, output):
    return ["--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
            "--calibration", str(args.calibration), "--expected-sha256", args.expected_sha256,
            "--roi", "260,210,350,350", "--output-dir", str(output)]


def test_cli_rejects_saved_image_for_execution_before_connect_or_output_creation(setup, tmp_path):
    args, bus, _, cameras, _ = setup
    output = tmp_path / "not_created"
    with pytest.raises(SystemExit) as error:
        collection.main(argv(args, output)+["--execute", "--port", args.port, "--image", "saved.png"])
    assert error.value.code == 2 and not output.exists()
    assert not bus.calls and not cameras


def test_supervised_cli_pins_inputs_and_preserves_existing_output(setup, tmp_path, monkeypatch):
    args, bus, _, _, _ = setup
    output = tmp_path / "parent"
    def supervise(command, stream, timeout_s, **kwargs):
        assert "--execute" not in command and "--_worker" in command
        assert "--_source-sha256" in command and "--_checkpoints-sha256" in command
        assert timeout_s == 135+replay.runtime_budget(replay.prepare(args)[2])
        return 0
    monkeypatch.setattr(collection, "supervise", supervise)
    assert collection.main(argv(args, output)) == 0
    log = output / "events.jsonl"
    log.write_text("preserve me\n")
    assert collection.main(argv(args, output)) == 1
    assert log.read_text() == "preserve me\n"
    assert not bus.calls


def test_worker_detects_input_changed_before_loading_model_or_camera(setup):
    args, bus, _, cameras, _ = setup
    args._source_sha256 = "0"*64
    code, rows = run(args)
    assert code == 1 and rows[-1]["stage"] == "preflight"
    assert not cameras and not bus.calls
