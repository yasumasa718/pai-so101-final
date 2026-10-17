"""撮影・検出・モーターを模擬し、認識から固定再生へ進む条件を検証する。

再生テストの模擬バス・時計を再利用し、画像だけは一時フォルダーへ実際に保存する。
候補数、画像の古さ、開始姿勢、途中失敗を変え、接続・送信の有無と終了ログを確認する。
"""

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
    """再生用fixtureへ模擬カメラと缶1件の検出器を追加し、変更可能な候補リストとともに返す。"""
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
        """一定のRGB画像を保存し、作成された個数と解放状態を追跡するカメラ代役。"""
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
        """撮影用カメラが閉じていることを検査してから、現在の候補リストのコピーを返す。"""
        assert all(c.closed for c in cameras)  # カメラを閉じてから推論・接続する。
        return list(detections)

    # バスと時計は再生テストのfixtureで差替え済み。ここでは撮影・モデル読込みだけを置き換える。
    monkeypatch.setattr(collection, "DiagnosticCamera", Camera)
    monkeypatch.setattr(collection, "load_detector", lambda: detector)
    return args, bus, clock, cameras, detections


def run(args):
    """統合処理の子側入口を直接呼び、終了コードと解析済みJSONLイベントを返す。"""
    stream = io.StringIO()
    code = collection.run_worker(args, stream)
    return code, records(stream.getvalue())


def test_fresh_pass_runs_existing_replay_and_releases_all_axes(setup):
    """新しい画像のPASSから5要所の再生完了まで進み、全軸OFFと統合完了ログが残ることを確認する。

    認識判定ファイル自体の実行許可フラグはFalseのままであることも確かめる。
    """
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
    """実行指定があっても候補0件・2件なら再生せず、接続も書込みもしないことを確認する。"""
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
    """認識専用モードでは撮影画像・保存画像のどちらがPASSでも接続せず、保存画像時は撮影もしないことを確認する。"""
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
    """模擬時計で推論時間を画像の期限より長くし、モーターへ接続する前に拒否することを確認する。"""
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
    """モデル読込み・撮影・推論の各例外で接続を防ぎ、カメラを解放し、例外内の識別情報をログへ漏らさないことを確認する。"""
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
    """認識PASSでも開始姿勢がずれていれば、位置指令・トルク書込み前に再生が失敗することを確認する。"""
    args, bus, _, _, _ = setup
    bus.raw["shoulder_pan"] += 100
    code, rows = run(args)
    assert code == 1 and rows[-1]["replay_attempted"] is True and rows[-1]["replay_completed"] is False
    assert next(r for r in rows if r["event"] == "fault")["stage"] == "initial_position"
    assert not bus.targets and not bus.writes


def test_replay_failure_still_releases_and_collection_cannot_claim_success(setup):
    """再生中の送信失敗でもトルクを解除し、統合ログが再生完了を報告しないことを確認する。"""
    args, bus, _, _, _ = setup
    bus.fail_send = True
    code, rows = run(args)
    assert code == 1 and rows[-1]["replay_attempted"] is True and rows[-1]["replay_completed"] is False
    assert not any(bus.torque.values())
    assert len(bus.targets) == 2


def argv(args, output):
    """親CLIの起動に必要な経路・校正・ROI・出力先の引数を組み立てる。"""
    return ["--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
            "--calibration", str(args.calibration), "--expected-sha256", args.expected_sha256,
            "--roi", "260,210,350,350", "--output-dir", str(output)]


def test_cli_rejects_saved_image_for_execution_before_connect_or_output_creation(setup, tmp_path):
    """保存画像と実行指定の併用を引数エラーにし、出力先作成・撮影・接続より前に拒否することを確認する。"""
    args, bus, _, cameras, _ = setup
    output = tmp_path / "not_created"
    with pytest.raises(SystemExit) as error:
        collection.main(argv(args, output)+["--execute", "--port", args.port, "--image", "saved.png"])
    assert error.value.code == 2 and not output.exists()
    assert not bus.calls and not cameras


def test_supervised_cli_pins_inputs_and_preserves_existing_output(setup, tmp_path, monkeypatch):
    """親が入力ハッシュと期限を子へ渡し、既存出力先への再実行ではログを上書きしないことを確認する。"""
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
    """親から渡された元ログのハッシュが異なると、撮影・接続より前の事前検査で失敗することを確認する。"""
    args, bus, _, cameras, _ = setup
    args._source_sha256 = "0"*64
    code, rows = run(args)
    assert code == 1 and rows[-1]["stage"] == "preflight"
    assert not cameras and not bus.calls
