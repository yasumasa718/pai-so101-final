"""実バス・カメラには接続せず、診断の読取り限定経路と監視を検証する。"""

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
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def hardware_args(tmp_path):
    return diagnostic.parser().parse_args([
        "--backend", "hardware", "--port", "/dev/serial/by-id/test-only",
        "--calibration", str(SNAPSHOT), "--expected-sha256", hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
        "--samples", "2", "--period-s", "0.001", "--output", str(tmp_path / "unused.jsonl"),
    ])


class ReadOnlyBus:
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
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *args: bus)
    stream = io.StringIO()
    code = diagnostic.run_worker(hardware_args(tmp_path), stream)
    return code, records(stream.getvalue())


def test_hardware_path_reads_checks_and_closes_without_writes(monkeypatch, tmp_path):
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
    bus = ReadOnlyBus()
    bus.read_error = KeyboardInterrupt()
    bus.close_error = OSError()
    code, events = run_hardware_fake(monkeypatch, tmp_path, bus)
    assert code == 1
    assert [(e["stage"], e["error_type"]) for e in events if e["event"] == "fault"] == [
        ("joint_read", "KeyboardInterrupt"), ("bus_close", "OSError"),
    ]


def test_hash_failure_never_constructs_bus(monkeypatch, tmp_path):
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *args: pytest.fail("must not construct bus"))
    args = hardware_args(tmp_path)
    args.expected_sha256 = "0" * 64
    stream = io.StringIO()
    assert diagnostic.run_worker(args, stream) == 1
    assert records(stream.getvalue())[-1]["stage"] == "calibration"


def test_log_failure_still_closes_without_torque_change(monkeypatch, tmp_path):
    bus = ReadOnlyBus()
    monkeypatch.setattr(diagnostic, "make_hardware_bus", lambda *args: bus)

    class BrokenLog(io.StringIO):
        def write(self, text):
            if '"event": "observation"' in text or '"event": "fault"' in text:
                raise OSError("disk full")
            return super().write(text)

    with pytest.raises(OSError):
        diagnostic.run_worker(hardware_args(tmp_path), BrokenLog())
    assert bus.calls[-1] == "close_without_torque_change"
    assert bus.calls.count("positions") == 1


def test_hardware_bus_uses_expected_units_and_never_opens_port():
    from lerobot.motors import MotorNormMode
    calibration, _ = diagnostic.load_calibration(SNAPSHOT, hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest())
    bus = diagnostic.make_hardware_bus("/nonexistent/test-only", calibration)
    assert not bus.is_connected
    for name, motor in bus.motors.items():
        assert motor.id == MOTOR_IDS[name]
        assert motor.norm_mode == (MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES)


def test_installed_bus_connection_and_close_do_not_write_motor_registers(monkeypatch):
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
    monkeypatch.setattr(diagnostic, "supervise", lambda *args: pytest.fail("must not launch"))
    output = tmp_path / "missing.jsonl"
    with pytest.raises(SystemExit):
        diagnostic.main(["--output", str(output), *arguments])
    assert not output.exists()


def test_supervisor_stops_a_real_hung_child(tmp_path):
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
    path = tmp_path / "crash.jsonl"
    with path.open("w") as stream:
        assert diagnostic.supervise([sys.executable, "-c", "raise SystemExit(7)"], stream, 3) == 1
    assert records(path.read_text())[-1]["returncode"] == 7


def test_supervisor_interrupt_stops_child(monkeypatch, tmp_path):
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
    class Unstoppable:
        def poll(self):
            return None

        def terminate(self):
            pass

        kill = terminate

        def wait(self, **kwargs):
            raise subprocess.TimeoutExpired("fake", 1)

    assert diagnostic.stop_worker(Unstoppable()) is False
