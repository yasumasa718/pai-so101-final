"""二つの模擬バスで、起動時保持・絶対目標への段階追従・全軸解除を確認する。"""

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

from src.control import guarded_teleop as teleop
from src.control.action_guard import MOTOR_IDS
from src.control.action_guard import ActionGuard
from test_diagnose_observation import ReadOnlyBus, records


class Clock:
    now = 10.0

    def __call__(self):
        self.now += 0.002
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class TeleopBus(ReadOnlyBus):
    def __init__(self, calibration, clock):
        super().__init__()
        self.calibration = calibration
        self.clock = clock
        self.raw = {name: round((cal["range_min"]+cal["range_max"])/2)
                    for name, cal in calibration.items()}
        self.torque = {name: 0 for name in MOTOR_IDS}
        self.goal = {name: 0 for name in MOTOR_IDS}  # 古い目標で起動しないこと。
        self.writes = []
        self.targets = []
        self.pan_step = 0
        self.read_count = 0
        self.fail_enable = None
        self.fail_off = None
        self.fail_send = False

    def read(self, register, motor, *, normalize, num_retry):
        self.clock.now += 0.002
        if register == "Torque_Enable":
            return self.torque[motor]
        if register == "Goal_Position":
            return self.goal[motor]
        return super().read(register, motor, normalize=normalize, num_retry=num_retry)

    def sync_read(self, register, *, normalize, num_retry):
        assert register == "Present_Position" and not normalize and num_retry == 0
        self.clock.now += 0.002
        if self.read_count:
            self.raw["shoulder_pan"] += self.pan_step
        self.read_count += 1
        return dict(self.raw)

    def sync_write(self, register, values, *, normalize, num_retry):
        assert register == "Goal_Position" and not normalize and num_retry == 0
        assert set(values) == set(MOTOR_IDS)
        self.clock.now += 0.002
        self.targets.append(dict(values))
        self.goal = dict(values)
        for name, value in values.items():
            if self.torque[name]:
                self.raw[name] = value
        if len(self.targets) > 1 and self.fail_send:
            raise ConnectionError("SECRET_PORT")

    def write(self, register, motor, value, *, normalize, num_retry):
        assert register == "Torque_Enable" and not normalize and num_retry == 0
        self.clock.now += 0.002
        self.writes.append((motor, value))
        if value == 0 and motor == self.fail_off:
            raise ConnectionError("SECRET_PORT")
        self.torque[motor] = value
        if value:
            self.raw[motor] = self.goal[motor]
        if value == 1 and motor == self.fail_enable:
            raise ConnectionError("SECRET_PORT")


def setup(tmp_path, monkeypatch, execute=True):
    clock = Clock()
    monkeypatch.setattr(teleop.time, "monotonic", clock)
    monkeypatch.setattr(teleop.time, "sleep", clock.sleep)
    args = SimpleNamespace(execute=execute, duration_s=0.4)
    buses = {}
    for role in ("leader", "follower"):
        path = Path(f"configs/calibration_snapshots/my_{role}_for_004.json")
        setattr(args, role+"_calibration", path)
        setattr(args, role+"_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
        setattr(args, role+"_port", role)
        buses[role] = TeleopBus(json.loads(path.read_bytes()), clock)
    buses["follower"].raw["shoulder_pan"] += 200
    buses["leader"].pan_step = 5
    monkeypatch.setattr(teleop, "make_hardware_bus", lambda port, cal: buses[port])
    return args, buses


def run(args):
    stream = io.StringIO()
    code = teleop.run_worker(args, stream)
    return code, records(stream.getvalue())


def test_preview_never_writes_or_enables(tmp_path, monkeypatch):
    args, buses = setup(tmp_path, monkeypatch, execute=False)
    code, events = run(args)
    assert code == 0 and events[-1] == {"event": "complete", "executed": False}
    assert all(not b.writes and not b.targets for b in buses.values())
    assert all(b.calls[-1] == "close_without_torque_change" for b in buses.values())


def test_absolute_teleop_primes_current_pose_and_only_writes_follower(tmp_path, monkeypatch):
    args, buses = setup(tmp_path, monkeypatch)
    follower, leader = buses["follower"], buses["leader"]
    initial = dict(follower.raw)
    code, events = run(args)
    assert code == 0
    assert follower.targets[0] == initial
    # Leaderとの姿勢差を解消する方向へ動く。固定オフセットは残さない。
    assert follower.targets[1]["shoulder_pan"] == initial["shoulder_pan"]-22
    for previous, current in zip(follower.targets, follower.targets[1:]):
        assert abs(current["shoulder_pan"]-previous["shoulder_pan"]) * 360/4095 <= 2
    assert events[0]["mode"] == "absolute"
    assert not leader.targets and not leader.writes
    assert not any(follower.torque.values())
    assert set(events[-2]["verified"]) == set(MOTOR_IDS)
    assert events[-2]["unconfirmed"] == []
    assert events[-1]["event"] == "complete"
    assert any(e["event"] == "frame" for e in events)


def test_large_leader_change_is_followed_gradually(tmp_path, monkeypatch):
    args, buses = setup(tmp_path, monkeypatch)
    buses["leader"].pan_step = 100
    code, events = run(args)
    assert code == 0 and len(buses["follower"].targets) > 1
    for previous, current in zip(buses["follower"].targets, buses["follower"].targets[1:]):
        assert abs(current["shoulder_pan"]-previous["shoulder_pan"]) * 360/4095 <= 2
    assert not any(e["event"] == "fault" for e in events)
    assert not any(buses["follower"].torque.values())


def test_tick_limiter_respects_step_and_rate_for_arm_and_gripper():
    calibration = json.loads(Path("configs/calibration_snapshots/my_follower_for_004.json").read_bytes())
    guard = ActionGuard.from_calibration(Path("configs/calibration_snapshots/my_follower_for_004.json"))
    previous = {f"{name}.pos": 30.0 if name == "gripper" else 0.0 for name in MOTOR_IDS}
    desired = {key: value+8.79121 for key, value in previous.items()}
    previous = teleop.positions_from_raw(teleop.raw_from_positions(previous, calibration), calibration)
    policy = teleop.motion_policy(guard)
    for elapsed in (0.013, 0.04, 0.1, 0.7):
        target = teleop.limited_target(desired, previous, calibration, elapsed, policy)
        for limit in policy.limits:
            delta = target[limit.action_key]-previous[limit.action_key]
            assert 0 <= delta <= min(limit.max_step, limit.max_speed*elapsed)


def test_out_of_follower_range_leader_pose_refused_before_enable(tmp_path, monkeypatch):
    args, buses = setup(tmp_path, monkeypatch)
    # LeaderのF4校正範囲はFollowerより広い。範囲外目標へは徐々にも移動しない。
    buses["leader"].raw["wrist_flex"] = buses["leader"].calibration["wrist_flex"]["range_min"]+1
    code, events = run(args)
    assert code == 1 and not buses["follower"].targets and not buses["follower"].writes
    fault = next(e for e in events if e["event"] == "fault")
    assert "outside configured interval" in fault["reason"]
    assert not any(buses["follower"].torque.values())


def test_partial_enable_failure_releases_every_attempted_motor(tmp_path, monkeypatch):
    args, buses = setup(tmp_path, monkeypatch)
    follower = buses["follower"]
    follower.fail_enable = "elbow_flex"
    code, events = run(args)
    assert code == 1
    assert follower.writes == [("shoulder_pan", 1), ("shoulder_lift", 1), ("elbow_flex", 1),
                               ("elbow_flex", 0), ("shoulder_lift", 0), ("shoulder_pan", 0)]
    assert not any(follower.torque.values())
    assert not any(e["event"] == "complete" for e in events)


def test_send_failure_never_retries_and_releases_all(tmp_path, monkeypatch):
    args, buses = setup(tmp_path, monkeypatch)
    follower = buses["follower"]
    follower.fail_send = True
    code, events = run(args)
    assert code == 1 and len(follower.targets) == 2
    fault = next(e for e in events if e["event"] == "fault")
    assert fault["send_status"] == "uncertain"
    assert "SECRET_PORT" not in str(events)
    assert not any(follower.torque.values())


def test_one_release_failure_does_not_skip_other_motors(tmp_path, monkeypatch):
    args, buses = setup(tmp_path, monkeypatch)
    follower = buses["follower"]
    follower.fail_off = "elbow_flex"
    code, events = run(args)
    assert code == 1
    assert follower.torque == {name: int(name == "elbow_flex") for name in MOTOR_IDS}
    assert events[-1]["unconfirmed"] == ["elbow_flex"]
    assert not any(e["event"] == "complete" for e in events)
