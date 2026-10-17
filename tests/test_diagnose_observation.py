"""診断の読取り限定経路、設定照合、ログ、子プロセス監視を検証する。

実バスには接続せずReadOnlyBusへ差し替える。一部はインストール済みLeRobotの
接続処理をポートI/Oだけ模擬して検査し、監視テストは短命の実Python子プロセスを使う。
"""

import hashlib
import io
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.control import diagnose_observation as diagnostic
from src.control.action_guard import MOTOR_IDS


SNAPSHOT = Path("configs/calibration_snapshots/my_follower_for_004.json")


def records(text):
    """JSONLの空行を除き、イベント辞書のリストへ変換する。他の制御テストでも共用する。"""
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def hardware_args(tmp_path):
    """保存済み校正とテスト専用ポートを指定する引数を作る。実接続は呼出し側で模擬バスへ差し替える。"""
    return diagnostic.parser().parse_args([
        "--backend", "hardware", "--port", "/dev/serial/by-id/test-only",
        "--calibration", str(SNAPSHOT), "--expected-sha256", hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        "--samples", "2", "--period-s", "0.001", "--output", str(tmp_path / "unused.jsonl"),
    ])


class ReadOnlyBus:
    """読取り・接続・切断だけを許し、未知の操作は即テスト失敗にするバス代役。

    正常な校正レジスタを返し、overrideや各error属性で不一致・通信失敗を注入できる。
    Torque_Enableの既定値1は、診断が元のトルク状態を変えないことを確かめるための値。
    """
    model_number_table = {"sts3215": 777}

    def __init__(self):
        self.calibration = json.loads(SNAPSHOT.read_bytes())
        self.calls = []
        self.is_connected = False
        self.port_handler = SimpleNamespace(ser=SimpleNamespace())
        self.override = {}
        self.connect_error = None
        self.read_error = None
        self.close_error = None

    def connect(self, *, handshake):
        assert handshake is False
        self.calls.append("connect")
        self.is_connected = True
        if self.connect_error:
            raise self.connect_error

    def read(self, register, motor, *, normalize, num_retry):
        assert normalize is False and num_retry == 0
        self.calls.append((register, motor))
        cal = self.calibration[motor]
        values = {
            "ID": cal["id"], "Model_Number": 777, "Homing_Offset": cal["homing_offset"],
            "Min_Position_Limit": cal["range_min"], "Max_Position_Limit": cal["range_max"],
            "Phase": 0, "Operating_Mode": 0, "Torque_Enable": 1,
        }
        return self.override.get(register, values[register])

    def sync_read(self, register, *, normalize, num_retry):
        assert register == "Present_Position" and normalize is True and num_retry == 0
        self.calls.append("positions")
        if self.read_error:
            raise self.read_error
        return {name: 50.0 if name == "gripper" else 0.0 for name in MOTOR_IDS}

    def disconnect(self, *, disable_torque):
        assert disable_torque is False
        self.calls.append("close_without_torque_change")
        if self.close_error:
            raise self.close_error
        self.is_connected = False

    def __getattr__(self, name):
        pytest.fail(f"Unexpected operation on diagnostic bus: {name}")


def run_hardware_fake(monkeypatch, tmp_path, bus):
    """実機モードのバス生成を差し替え、子側処理の終了コードとイベント列を返す。"""
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *args: bus)
    stream = io.StringIO()
    code = diagnostic.run_worker(hardware_args(tmp_path), stream)
    return code, records(stream.getvalue())


def test_hardware_path_reads_checks_and_closes_without_writes(monkeypatch, tmp_path):
    """実機モードの経路が設定照合と2回の取得を行い、トルクを変更せず切断することを模擬バスで確認する。"""
    bus = ReadOnlyBus()
    code, events = run_hardware_fake(monkeypatch, tmp_path, bus)
    assert code == 0
    assert bus.calls[0] == "connect"
    assert bus.calls[-1] == "close_without_torque_change"
    assert bus.calls.count("positions") == 2
    checked = next(event for event in events if event["event"] == "register_check")
    assert checked["registers"]["shoulder_pan"]["Torque_Enable"] == 1
    assert events[-1] == {"event": "complete", "samples": 2, "motion_authorized": False}
    assert "test-only" not in json.dumps(events)
    observations = [event for event in events if event["event"] == "observation"]
    assert observations[1]["start_interval_s"] > 0


@pytest.mark.parametrize("register,value", [
    ("ID", 9), ("Model_Number", 42), ("Homing_Offset", 5000), ("Min_Position_Limit", -1),
    ("Max_Position_Limit", -1), ("Operating_Mode", 1), ("Phase", 0x10), ("Torque_Enable", 2),
])
def test_register_mismatch_prevents_samples_and_still_closes(monkeypatch, tmp_path, register, value):
    """各レジスタの不一致で位置取得へ進まず、失敗した項目を記録して切断することを確認する。"""
    bus = ReadOnlyBus()
    bus.override[register] = value
    code, events = run_hardware_fake(monkeypatch, tmp_path, bus)
    assert code == 1
    assert "positions" not in bus.calls
    assert bus.calls[-1] == "close_without_torque_change"
    fault = next(event for event in events if event["event"] == "fault")
    assert fault["stage"] == "register_check"
    assert fault["register"] == register
    assert all(event["event"] != "complete" for event in events)


@pytest.mark.parametrize("attribute,stage", [("connect_error", "bus_connect"), ("read_error", "joint_read")])
def test_failure_closes_once_and_does_not_retry(monkeypatch, tmp_path, attribute, stage):
    """接続・読取りの通信例外で再試行せず1回だけ切断し、秘密を含む例外本文をログへ出さないことを確認する。"""
    bus = ReadOnlyBus()
    setattr(bus, attribute, ConnectionError("SECRET_PORT"))
    code, events = run_hardware_fake(monkeypatch, tmp_path, bus)
    assert code == 1
    assert bus.calls.count("connect") == 1
    assert bus.calls.count("positions") <= 1
    assert bus.calls.count("close_without_torque_change") == 1
    assert next(event for event in events if event["event"] == "fault")["stage"] == stage
    assert "SECRET_PORT" not in json.dumps(events)


def test_close_failure_preserves_original_fault(monkeypatch, tmp_path):
    """読取り中断に加えて切断も失敗した場合、元の中断と切断失敗の両方がログに残ることを確認する。"""
    bus = ReadOnlyBus()
    bus.read_error = KeyboardInterrupt()
    bus.close_error = OSError()
    code, events = run_hardware_fake(monkeypatch, tmp_path, bus)
    assert code == 1
    assert [(e["stage"], e["error_type"]) for e in events if e["event"] == "fault"] == [
        ("joint_read", "KeyboardInterrupt"), ("bus_close", "OSError"),
    ]


def test_joint_range_failure_logs_reason_and_stops_without_retry(monkeypatch, tmp_path):
    """範囲外のF4観測について値・許容範囲を記録し、再読取りや正常完了報告を行わないことを確認する。"""
    bus = ReadOnlyBus()
    original_read = bus.sync_read

    def out_of_range(*args, **kwargs):
        values = original_read(*args, **kwargs)
        values["wrist_flex"] = -120.0
        return values

    bus.sync_read = out_of_range
    code, events = run_hardware_fake(monkeypatch, tmp_path, bus)
    assert code == 1
    fault = next(event for event in events if event["event"] == "fault")
    assert fault["stage"] == "joint_read"
    assert "wrist_flex.pos: -120.0000 outside configured interval" in fault["reason"]
    assert "[-102.8571, 102.8571] degree" in fault["reason"]
    assert bus.calls.count("positions") == 1
    assert bus.calls[-1] == "close_without_torque_change"
    assert not any(event["event"] == "complete" for event in events)


def test_hash_failure_never_constructs_bus(monkeypatch, tmp_path):
    """校正ハッシュが違えばバスを構築する前に失敗することを確認する。"""
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *args: pytest.fail("must not construct bus"))
    args = hardware_args(tmp_path)
    args.expected_sha256 = "0" * 64
    stream = io.StringIO()
    assert diagnostic.run_worker(args, stream) == 1
    assert records(stream.getvalue())[-1]["stage"] == "calibration"


def test_log_failure_still_closes_without_torque_change(monkeypatch, tmp_path):
    """観測もFAULTもログへ書けない場合でも、例外を伝える前にトルクを変えず切断することを確認する。"""
    bus = ReadOnlyBus()
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *args: bus)

    class BrokenLog(io.StringIO):
        """観測行とFAULT行の書込みだけを失敗させ、記録用ディスクの異常を再現する。"""
        def write(self, text):
            if '"event": "observation"' in text or '"event": "fault"' in text:
                raise OSError("disk full")
            return super().write(text)

    with pytest.raises(OSError):
        diagnostic.run_worker(hardware_args(tmp_path), BrokenLog())
    assert bus.calls[-1] == "close_without_torque_change"
    assert bus.calls.count("positions") == 1


def test_hardware_bus_uses_expected_units_and_never_opens_port():
    """実ライブラリのバス構築で腕は度・爪は0〜100に設定され、構築だけではポートを開かないことを確認する。"""
    from lerobot.motors import MotorNormMode
    calibration, _ = diagnostic.load_calibration(SNAPSHOT, hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest())
    bus = diagnostic.make_hardware_bus("/nonexistent/test-only", calibration)
    assert not bus.is_connected
    for name, motor in bus.motors.items():
        assert motor.id == MOTOR_IDS[name]
        assert motor.norm_mode == (MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES)


def test_installed_bus_connection_and_close_do_not_write_motor_registers(monkeypatch):
    """実ライブラリの接続・切断を模擬ポートで通し、書込み・トルク操作・handshakeを呼ばないことを確認する。"""
    calibration, _ = diagnostic.load_calibration(SNAPSHOT, hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest())
    bus = diagnostic.make_hardware_bus("/nonexistent/test-only", calibration)
    calls = []

    def open_port():
        calls.append("open")
        bus.port_handler.is_open = True
        return True

    def close_port():
        calls.append("close")
        bus.port_handler.is_open = False

    monkeypatch.setattr(bus.port_handler, "openPort", open_port)
    monkeypatch.setattr(bus.port_handler, "closePort", close_port)
    for method in ("write", "sync_write", "enable_torque", "disable_torque", "_handshake"):
        monkeypatch.setattr(bus, method, lambda *a, **k: pytest.fail("Unexpected write/handshake"))
    bus.connect(handshake=False)
    bus.disconnect(disable_torque=False)
    assert calls == ["open", "close"]


def test_fake_cli_completes_in_child_without_hardware(tmp_path):
    """模擬CLIを実際の子プロセスで実行して2観測を保存し、同名ログへの再実行は上書きせず失敗することを確認する。"""
    output = tmp_path / "observations.jsonl"
    assert diagnostic.main(["--output", str(output), "--samples", "2", "--period-s", "0.001"]) == 0
    events = records(output.read_text())
    assert events[0]["backend"] == "fake"
    assert len([e for e in events if e["event"] == "observation"]) == 2
    assert events[-1]["event"] == "complete"
    before = output.read_bytes()
    assert diagnostic.main(["--output", str(output)]) == 1
    assert output.read_bytes() == before


@pytest.mark.parametrize("arguments", [
    ["--backend", "hardware"], ["--port", "/dev/serial/by-id/a"],
    ["--samples", "0"], ["--timeout-s", "nan"], ["--period-s", "-1"], ["--max-read-s", "0"],
])
def test_invalid_cli_does_not_launch_or_create_log(monkeypatch, tmp_path, arguments):
    """不足した実機指定や不正な取得条件を、子の起動・ログ作成より前に引数エラーとすることを確認する。"""
    monkeypatch.setattr(diagnostic, "supervise", lambda *args: pytest.fail("must not launch"))
    output = tmp_path / "missing.jsonl"
    with pytest.raises(SystemExit):
        diagnostic.main(["--output", str(output), *arguments])
    assert not output.exists()


def test_supervisor_stops_a_real_hung_child(tmp_path):
    """長時間sleepする実子プロセスを0.2秒の期限で停止し、終了コード124と停止確認ログを返すことを確認する。"""
    path = tmp_path / "hung.jsonl"
    start = time.monotonic()
    with path.open("w") as stream:
        code = diagnostic.supervise([sys.executable, "-c", "import time; time.sleep(60)"], stream, 0.2)
    assert code == 124
    assert time.monotonic() - start < 4
    assert records(path.read_text())[-1] == {
        "event": "fault", "stage": "supervisor", "error_type": "TimeoutExpired", "worker_stopped": True,
    }


def test_supervisor_records_abnormal_exit(tmp_path):
    """子の終了コード7を親では失敗1として返し、元の終了コードをログへ残すことを確認する。"""
    path = tmp_path / "crash.jsonl"
    with path.open("w") as stream:
        assert diagnostic.supervise([sys.executable, "-c", "raise SystemExit(7)"], stream, 3) == 1
    assert records(path.read_text())[-1]["returncode"] == 7


def test_supervisor_interrupt_stops_child(monkeypatch, tmp_path):
    """親のwaitへ1回だけ割込みを注入し、実子プロセスを停止して終了コード130を返すことを確認する。"""
    real_popen = diagnostic.subprocess.Popen
    processes = []

    def interrupt_once(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        processes.append(process)
        real_wait = process.wait

        def wait(**kwargs):
            process.wait = real_wait
            raise KeyboardInterrupt()

        process.wait = wait
        return process

    monkeypatch.setattr(diagnostic.subprocess, "Popen", interrupt_once)
    path = tmp_path / "interrupt.jsonl"
    with path.open("w") as stream:
        assert diagnostic.supervise([sys.executable, "-c", "import time; time.sleep(60)"], stream, 2) == 130
    assert processes[0].poll() is not None
    assert records(path.read_text())[-1]["error_type"] == "KeyboardInterrupt"


def test_unstoppable_worker_is_not_reported_stopped():
    """terminate・kill後も終了を確認できない模擬プロセスについて、停止成功と報告しないことを確認する。"""
    class Unstoppable:
        """停止要求を無視し、常に生存中・待機期限切れを返すプロセス代役。"""
        def poll(self):
            return None

        def terminate(self):
            pass

        kill = terminate

        def wait(self, **kwargs):
            raise subprocess.TimeoutExpired("fake", 1)

    assert diagnostic.stop_worker(Unstoppable()) is False
