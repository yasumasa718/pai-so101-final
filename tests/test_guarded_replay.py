"""実機なしで全経路の再生、把持待機、起動拒否、停止時の全軸解除を確認する。"""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.control import guarded_replay as replay
from src.control.action_guard import MOTOR_IDS
from src.control.guarded_teleop import positions_from_raw, raw_from_positions
from test_diagnose_observation import records
from test_guarded_teleop import Clock, TeleopBus


SNAPSHOT = Path("configs/calibration_snapshots/my_follower_for_004.json")


class ReplayBus(TeleopBus):
    arm_bias = None
    freeze_arm = False
    blocked_close = False
    blocked_open = False
    has_closed = False

    def sync_write(self, register, values, **kwargs):
        previous = dict(self.raw)
        super().sync_write(register, values, **kwargs)
        for motor, degrees in (self.arm_bias or {}).items():
            if self.torque[motor]:
                self.raw[motor] += round(degrees*4095/360)
        if self.torque["shoulder_pan"] and self.freeze_arm:
            self.raw["shoulder_pan"] = previous["shoulder_pan"]
        if self.torque["gripper"]:
            blocked = raw_from_positions({**positions_from_raw(values, self.calibration),
                                          "gripper.pos": 20.0}, self.calibration)["gripper"]
            if values["gripper"] < blocked:
                self.has_closed = True
            if self.blocked_close:
                self.raw["gripper"] = max(blocked, values["gripper"])
            if self.blocked_open and self.has_closed:
                self.raw["gripper"] = blocked


@pytest.fixture
def setup(tmp_path, monkeypatch):
    clock = Clock()
    monkeypatch.setattr(replay.time, "monotonic", clock)
    monkeypatch.setattr(replay.time, "sleep", clock.sleep)
    calibration = json.loads(SNAPSHOT.read_bytes())
    bus = ReplayBus(calibration, clock)
    bus.arm_bias = {"shoulder_lift": -7.0}  # 手本の指令と実測に残る一定の差を模擬。
    initial_raw = {**bus.raw, "gripper": calibration["gripper"]["range_min"]
                   + round(0.35*(calibration["gripper"]["range_max"]-calibration["gripper"]["range_min"]))}
    bus.raw = dict(initial_raw)
    initial = positions_from_raw(initial_raw, calibration)
    frames = []
    grippers = (35, 35, 20, 15, 15, 15, 25, 35, 35, 35)
    for i, grip in enumerate(grippers, 1):
        observed = positions_from_raw({**initial_raw, "shoulder_pan": initial_raw["shoulder_pan"]+20*i},
                                      calibration)
        sent = {**observed, "shoulder_lift.pos": observed["shoulder_lift.pos"]+7, "gripper.pos": grip}
        frames.append({"event": "frame", "index": i, "read_started_at": 10+i*0.11,
                       "leader": sent, "sent_target": sent, "observation": observed})
    rows = [{"event": "plan", "follower_initial": initial}, *frames,
            {"event": "torque_off", "verified": list(MOTOR_IDS), "unconfirmed": [], "failures": []},
            {"event": "complete", "executed": True}]
    source = tmp_path / "source.jsonl"
    source.write_text("".join(json.dumps(row)+"\n" for row in rows))
    checkpoints = tmp_path / "selected.json"
    checkpoints.write_text(json.dumps({"poses": [
        {"name": name, "frame_index": index, "observation": frames[index-1]["observation"],
         "sent_target": frames[index-1]["sent_target"]}
        for name, index in zip(replay.PHASES, (2, 4, 6, 8, 10))]}))
    args = SimpleNamespace(source_log=source, checkpoints=checkpoints, calibration=SNAPSHOT,
                           expected_sha256=hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest(),
                           execute=True, port="/dev/serial/by-id/fake-only")
    monkeypatch.setattr(replay, "make_hardware_bus", lambda port, cal: bus)
    return args, bus, frames, rows, clock


def run(args):
    stream = io.StringIO()
    code = replay.run_worker(args, stream)
    return code, records(stream.getvalue())


def test_compiles_every_recorded_command_and_keeps_observation_as_arrival_reference(setup):
    args, bus, frames, _, _ = setup
    _, guard, plan = replay.prepare(args)
    finished = [s for s in plan["steps"] if s["source_frame_complete"]]
    assert [s["source_frame"] for s in finished] == list(range(1, 11))
    for step, frame in zip(finished, frames):
        expected = frame["sent_target"]
        assert raw_from_positions(step["target"], bus.calibration) == raw_from_positions(expected, bus.calibration)
        assert step["target"]["shoulder_lift.pos"] != frame["observation"]["shoulder_lift.pos"]
        if step["checkpoint"]:
            assert step["arrival_target"]["shoulder_lift.pos"] == frame["observation"]["shoulder_lift.pos"]
    previous = plan["initial"]
    for step in plan["steps"]:
        assert guard.check(step["target"]).accepted
        assert all(abs(step["target"][key]-previous[key]) <= 2 for key in previous)
        assert all(abs(step["target"][key]-previous[key])/step["period_s"] <= 20 for key in previous)
        previous = step["target"]
    assert len(plan["steps"]) > len(frames)  # 大きいF6変化を制限内へ分割。
    assert not bus.calls and not bus.targets


def test_default_preview_has_no_hardware_io_and_preserves_existing_output(setup, tmp_path):
    args, bus, _, _, _ = setup
    output = tmp_path / "preview.json"
    argv = ["--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
            "--calibration", str(SNAPSHOT), "--expected-sha256", args.expected_sha256,
            "--output", str(output)]
    assert replay.main(argv) == 0
    original = output.read_bytes()
    assert json.loads(original)["hardware_verified"] is False
    assert replay.main(argv) == 1
    assert output.read_bytes() == original
    assert not bus.calls and not bus.writes and not bus.targets


@pytest.mark.parametrize("offset", [0, 15])
def test_success_keeps_all_frames_prime_goal_and_turns_all_axes_off(setup, offset):
    args, bus, _, _, _ = setup
    bus.raw["shoulder_pan"] -= offset  # 開始許容幅内のずれも段階追従する。
    initial = dict(bus.raw)
    code, events = run(args)
    assert code == 0
    assert events[-1] == {"event": "complete", "executed": True, "grasp_verified": False}
    assert bus.targets[0] == initial  # 古いGoalで有効化しない。
    assert bus.writes[:6] == [(m, 1) for m in MOTOR_IDS]
    assert bus.writes[6:] == [(m, 0) for m in reversed(MOTOR_IDS)]
    assert all(v == 0 for v in bus.torque.values())
    assert bus.calls[-1] == "close_without_torque_change"
    completed = [e["source_frame"] for e in events if e["event"] == "frame" and e["source_frame_complete"]]
    assert completed == list(range(1, 11))
    assert [e["name"] for e in events if e["event"] == "checkpoint_complete"] == list(replay.PHASES)
    previous = positions_from_raw(initial, bus.calibration)
    for raw in bus.targets[1:]:
        current = positions_from_raw(raw, bus.calibration)
        assert all(abs(current[k]-previous[k]) <= 2 for k in current)
        previous = current


@pytest.mark.parametrize("execute", [False, True])
def test_start_mismatch_never_sends_or_enables(setup, execute):
    args, bus, _, _, _ = setup
    args.execute = execute
    bus.raw["shoulder_pan"] += 40
    code, events = run(args)
    assert code == 1
    assert next(e for e in events if e["event"] == "start_check")["matched"] is False
    assert next(e for e in events if e["event"] == "fault")["stage"] == "initial_position"
    assert not bus.writes and not bus.targets


def test_check_start_only_reads(setup):
    args, bus, _, _, _ = setup
    args.execute = False
    code, events = run(args)
    assert code == 0 and events[-1]["executed"] is False
    assert not bus.targets and not bus.writes
    assert bus.calls[-1] == "close_without_torque_change"


def test_changed_physical_response_does_not_pass_just_because_command_was_reached(setup):
    args, bus, _, _, _ = setup
    bus.arm_bias = None  # 手本では指令より7度小さかった実測が、今回は指令そのものに到達。
    code, events = run(args)
    assert code == 1
    fault = next(e for e in events if e["event"] == "fault")
    assert fault["stage"] == "wait_approach" and fault["error_type"] == "TimeoutError"
    assert not any(e["event"] == "checkpoint_complete" for e in events)
    assert all(v == 0 for v in bus.torque.values())


def test_watch_start_continues_through_mismatch_and_exits_on_stable_match(setup, capsys):
    args, bus, _, _, _ = setup
    args.execute, args.watch_start, args.watch_duration_s = False, True, 10
    bus.raw["shoulder_pan"] -= 30
    bus.pan_step = 15
    code, events = run(args)
    checks = [e for e in events if e["event"] == "start_check"]
    assert code == 0 and not checks[0]["matched"]
    assert len(checks) == 4 and all(e["matched"] for e in checks[-3:])
    assert events[-1]["start_matched"] is True and events[-1]["executed"] is False
    assert not bus.targets and not bus.writes
    assert bus.calls[-1] == "close_without_torque_change"
    output = capsys.readouterr().err
    assert "土台の旋回" in output and "値を増やす" in output
    assert "校正時の姿勢ではありません" in output


def test_watch_start_duration_ends_without_false_match_or_writes(setup, capsys):
    args, bus, _, _, clock = setup
    args.execute, args.watch_start, args.watch_duration_s = False, True, 2
    bus.raw["shoulder_lift"] += 300
    code, events = run(args)
    assert code == 0 and events[-1]["start_matched"] is False
    assert not any(e["event"] == "fault" for e in events)
    assert bus.read_count >= 2 and clock.now < 14
    assert not bus.targets and not bus.writes
    assert "値を減らす" in capsys.readouterr().err


def test_watch_start_read_failure_stops_and_closes_without_writes(setup):
    args, bus, _, _, _ = setup
    args.execute, args.watch_start, args.watch_duration_s = False, True, 10
    bus.fail_read_at = 2
    code, events = run(args)
    assert code == 1 and bus.read_count == 2
    assert events[-1]["event"] == "fault" and events[-1]["stage"] == "watch_start"
    assert bus.calls[-1] == "close_without_torque_change"
    assert not bus.targets and not bus.writes


def test_watch_start_and_execute_cannot_be_combined(setup):
    args, bus, _, _, _ = setup
    with pytest.raises(SystemExit) as error:
        replay.main(["--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
                     "--calibration", str(SNAPSHOT), "--expected-sha256", args.expected_sha256,
                     "--port", args.port, "--watch-start", "--execute"])
    assert error.value.code == 2 and not bus.calls


def test_watch_parent_allows_display_duration_and_forwards_read_only_mode(setup, tmp_path, monkeypatch):
    args, bus, _, _, _ = setup
    def supervise(command, stream, timeout_s, **kwargs):
        assert "--watch-start" in command and "--execute" not in command
        assert command[command.index("--watch-duration-s")+1] == "120.0"
        assert timeout_s == 135
        return 0
    monkeypatch.setattr(replay, "supervise", supervise)
    assert replay.main(["--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
                        "--calibration", str(SNAPSHOT), "--expected-sha256", args.expected_sha256,
                        "--port", args.port, "--watch-start", "--watch-duration-s", "120",
                        "--output", str(tmp_path / "watch.jsonl")]) == 0
    assert not bus.calls


def test_grasp_allows_blocked_closing_position_but_waits_and_checks_opening(setup):
    args, bus, _, _, _ = setup
    bus.blocked_close = True
    code, events = run(args)
    assert code == 0
    for phase in ("grasp", "box"):
        arrived = next(e for e in events if e["event"] == "checkpoint_complete" and e["name"] == phase)
        assert arrived["observation"]["gripper.pos"]-arrived["target"]["gripper.pos"] > 4
        assert arrived["gripper_position_required"] is False
        assert arrived["grasp_verified"] is False
    waits = [e for e in events if e["event"] == "frame" and e["waiting"] and e["checkpoint"] == "grasp"]
    assert waits[-1]["read_started_at"]-waits[1]["read_started_at"] >= replay.GRASP_DWELL
    release = next(e for e in events if e["event"] == "checkpoint_complete" and e["name"] == "release")
    assert release["gripper_position_required"] is True
    assert abs(release["observation"]["gripper.pos"]-release["target"]["gripper.pos"]) <= 2


@pytest.mark.parametrize("blocked,phase,last_frame", [("freeze_arm", "approach", 2),
                                                       ("blocked_open", "release", 8)])
def test_arrival_timeout_stops_before_next_phase_and_releases(setup, blocked, phase, last_frame):
    args, bus, _, _, clock = setup
    setattr(bus, blocked, True)
    code, events = run(args)
    assert code == 1
    fault = next(e for e in events if e["event"] == "fault")
    assert fault["stage"] == "wait_"+phase and fault["error_type"] == "TimeoutError"
    assert max(e["source_frame"] for e in events if e["event"] == "frame") == last_frame
    assert not any(e["event"] == "complete" for e in events)
    assert all(v == 0 for v in bus.torque.values())
    assert clock.now < 40  # 待機は無期限にならない。


@pytest.mark.parametrize("fault", ["fail_send", "fail_read_at", "fail_enable", "interrupt"])
def test_communication_and_interrupt_stop_without_retry_and_release(setup, monkeypatch, fault):
    args, bus, _, _, _ = setup
    if fault == "interrupt":
        original = bus.sync_read
        def interrupted(*a, **kw):
            if bus.read_count == 1:
                raise KeyboardInterrupt()
            return original(*a, **kw)
        monkeypatch.setattr(bus, "sync_read", interrupted)
    else:
        setattr(bus, fault, {"fail_send": True, "fail_read_at": 2,
                             "fail_enable": "elbow_flex"}[fault])
    code, events = run(args)
    assert code == 1
    assert not any(e["event"] == "complete" for e in events)
    assert len(bus.targets) <= 2
    assert all(v == 0 for v in bus.torque.values())
    off = next(e for e in events if e["event"] == "torque_off")
    assert off["unconfirmed"] == []
    assert "SECRET_PORT" not in json.dumps(events)
    if fault == "fail_enable":
        assert set(off["verified"]) == {"shoulder_pan", "shoulder_lift", "elbow_flex"}


def test_one_off_failure_does_not_skip_other_axes_or_claim_complete(setup):
    args, bus, _, _, _ = setup
    bus.fail_off = "gripper"
    code, events = run(args)
    assert code == 1
    off = next(e for e in events if e["event"] == "torque_off")
    assert off["unconfirmed"] == ["gripper"]
    assert set(off["verified"]) == set(MOTOR_IDS)-{"gripper"}
    assert not any(e["event"] == "complete" for e in events)


@pytest.mark.parametrize("damage", ["fault_source", "read_only_source", "unconfirmed_off",
                                    "range", "checkpoint", "input_changed", "calibration_hash"])
def test_bad_input_is_rejected_before_hardware_connection(setup, damage):
    args, bus, _, rows, _ = setup
    if damage == "fault_source":
        rows.insert(-1, {"event": "fault"})
    elif damage == "read_only_source":
        rows[-1]["executed"] = False
    elif damage == "unconfirmed_off":
        rows[-2]["unconfirmed"] = ["gripper"]
    elif damage == "range":
        rows[1]["observation"]["shoulder_pan.pos"] = 999
    elif damage == "checkpoint":
        data = json.loads(args.checkpoints.read_bytes())
        data["poses"][0]["sent_target"]["gripper.pos"] = 99
        args.checkpoints.write_text(json.dumps(data))
    elif damage == "input_changed":
        args._source_sha256 = "0"*64
    elif damage == "calibration_hash":
        args.expected_sha256 = "0"*64
    args.source_log.write_text("".join(json.dumps(row)+"\n" for row in rows))
    code, events = run(args)
    assert code == 1 and events[-1]["stage"] == "preflight"
    assert not bus.calls and not bus.targets and not bus.writes


def test_initial_torque_on_is_rejected_without_writing(setup):
    args, bus, _, _, _ = setup
    bus.torque["wrist_roll"] = 1
    code, events = run(args)
    assert code == 1
    assert events[-1]["stage"] == "register_check"
    assert not bus.targets and not bus.writes


def test_initial_goal_readback_failure_never_enables(setup, monkeypatch):
    args, bus, _, _, _ = setup
    original = bus.read
    def read(register, motor, **kwargs):
        value = original(register, motor, **kwargs)
        return value+1 if register == "Goal_Position" else value
    monkeypatch.setattr(bus, "read", read)
    code, events = run(args)
    assert code == 1 and events[-1]["stage"] == "prepare_hold"
    assert len(bus.targets) == 1 and not bus.writes


@pytest.mark.parametrize("expire_during_readback", [False, True])
def test_recognition_deadline_blocks_stale_motor_startup(setup, monkeypatch, expire_during_readback):
    args, bus, _, _, clock = setup
    deadline = clock.now+10 if expire_during_readback else clock.now-1
    if expire_during_readback:
        original = bus.read
        def read(register, motor, **kwargs):
            if register == "Goal_Position":
                clock.now += 11
            return original(register, motor, **kwargs)
        monkeypatch.setattr(bus, "read", read)
    stream = io.StringIO()
    assert replay.run_worker(args, stream, activation_deadline=deadline) == 1
    events = records(stream.getvalue())
    assert events[-1]["error_type"] == "TimeoutError"
    assert "image expired" in events[-1]["reason"]
    assert not bus.writes and len(bus.targets) == int(expire_during_readback)


def test_slow_read_during_motion_sends_no_next_command(setup, monkeypatch):
    args, bus, _, _, clock = setup
    original = bus.sync_read
    def read(*a, **kwargs):
        if bus.read_count:
            clock.now += 0.3
        return original(*a, **kwargs)
    monkeypatch.setattr(bus, "sync_read", read)
    code, events = run(args)
    assert code == 1
    assert next(e for e in events if e["event"] == "fault")["error_type"] == "JointReadError"
    assert len(bus.targets) == 1
    assert all(v == 0 for v in bus.torque.values())


def test_parent_starts_supervised_worker_with_exact_input_hashes(setup, tmp_path, monkeypatch):
    args, bus, _, _, _ = setup
    output = tmp_path / "hardware.jsonl"
    def supervise(command, stream, timeout_s, **kwargs):
        assert "--_worker" in command and "--execute" in command
        assert command[command.index("--_source-sha256")+1] == hashlib.sha256(args.source_log.read_bytes()).hexdigest()
        assert command[command.index("--_checkpoints-sha256")+1] == hashlib.sha256(args.checkpoints.read_bytes()).hexdigest()
        assert timeout_s < 100
        return 124
    monkeypatch.setattr(replay, "supervise", supervise)
    assert replay.main(["--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
                        "--calibration", str(SNAPSHOT), "--expected-sha256", args.expected_sha256,
                        "--port", args.port, "--execute", "--output", str(output)]) == 124
    assert not bus.calls
