"""Leaderの姿勢へ、送信目標を段階的に近づける短時間テレオペ。

既定は読取りのみ。--executeでFollower全軸を有効化し、終了時に解除する。
アーム支持・落下防止を用意し、最初は缶なしで小さく操作する。
"""

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

from src.control.action_guard import MOTOR_IDS
from src.control.diagnose_observation import (
    RegisterMismatch, emit, load_calibration, make_hardware_bus, supervise, verify_registers,
)
from src.control.motion_guard import JointMotionLimit, MotionGuard
from src.control.safe_record_loop import RecordFault, SafeRecordLoop
from src.control.timed_observation import JointReadError, TimedJointReader


def positions_from_raw(raw, calibration):
    result = {}
    for motor in MOTOR_IDS:
        low, high = calibration[motor]["range_min"], calibration[motor]["range_max"]
        if motor == "gripper":
            if not low <= raw[motor] <= high:
                raise JointReadError("Raw gripper position is outside calibration")
            value = (raw[motor]-low) * 100 / (high-low)
        else:
            value = (raw[motor]-(low+high)/2) * 360 / 4095
        result[f"{motor}.pos"] = value
    return result


def raw_from_positions(action, calibration):
    raw = {}
    for motor in MOTOR_IDS:
        low, high = calibration[motor]["range_min"], calibration[motor]["range_max"]
        value = action[f"{motor}.pos"]
        raw[motor] = round(low + value * (high-low)/100 if motor == "gripper"
                           else (low+high)/2 + value * 4095/360)
    return raw


class RawReader:
    """校正外のグリッパー値をclampで隠さず、同じ読取りの生値も保持する。"""

    def __init__(self, bus, calibration):
        self.bus, self.calibration = bus, calibration
        self.raw = None

    def sync_read(self, register, *, normalize, num_retry):
        self.raw = self.bus.sync_read(register, normalize=False, num_retry=num_retry)
        positions = positions_from_raw(self.raw, self.calibration)
        return {key.removesuffix(".pos"): value for key, value in positions.items()}


class FollowerSender:
    def __init__(self, bus, calibration):
        self.bus, self.calibration = bus, calibration

    def quantize(self, pair):
        # 丸め後の実際の送信候補も、既存ループで送信前に検査する。
        return positions_from_raw(raw_from_positions(pair[0], self.calibration), self.calibration)

    def send_action(self, action):
        raw = raw_from_positions(action, self.calibration)
        self.bus.sync_write("Goal_Position", raw, normalize=False, num_retry=0)
        return positions_from_raw(raw, self.calibration)


def motion_policy(guard):
    return MotionGuard(
        limits=tuple(JointMotionLimit(b.action_key, b.unit, max_step=2.0, max_speed=20.0)
                     for b in guard.bounds),
        min_interval_s=0.001, max_interval_s=0.5,
        max_observation_age_s=0.25, max_send_duration_s=0.25,
    )


def limited_target(desired, previous, calibration, elapsed_s, policy):
    """絶対目標への追従速度を制限する。丸め後も上限を越えないtick幅で生成する。"""
    desired_raw = raw_from_positions(desired, calibration)
    previous_raw = raw_from_positions(previous, calibration)
    result = {}
    for limit in policy.limits:
        motor = limit.action_key.removesuffix(".pos")
        cal = calibration[motor]
        unit_per_tick = 100/(cal["range_max"]-cal["range_min"]) if motor == "gripper" else 360/4095
        budget = max(0, min(limit.max_step, limit.max_speed * elapsed_s))
        ticks = int(budget / unit_per_tick)
        low, high = previous_raw[motor]-ticks, previous_raw[motor]+ticks
        result[motor] = min(high, max(low, desired_raw[motor]))
    return positions_from_raw(result, calibration)


def run_worker(args, stream):
    buses = {}
    enabled = []
    failed = False
    stage = "calibration"
    try:
        readers, raw_readers, guards, calibrations = {}, {}, {}, {}
        for role in ("leader", "follower"):
            calibration, guard = load_calibration(Path(getattr(args, role+"_calibration")),
                                                  getattr(args, role+"_sha256"))
            calibrations[role], guards[role] = calibration, guard
            stage = role+"_connect"
            bus = buses[role] = make_hardware_bus(getattr(args, role+"_port"), calibration)
            bus.connect(handshake=False)
            bus.port_handler.ser.exclusive = True
            bus.port_handler.ser.write_timeout = 0.25
            stage = role+"_register_check"
            registers = verify_registers(bus, calibration)
            if any(item["Torque_Enable"] != 0 for item in registers.values()):
                raise ValueError(role+": all motors must initially have torque OFF")
            raw_readers[role] = RawReader(bus, calibration)
            readers[role] = TimedJointReader(raw_readers[role], guard, max_read_duration_s=0.25,
                                           clock=time.monotonic)

        if args.execute:
            print("3秒後にFollower全軸を有効化し、Leaderの姿勢へ徐々に動かします。終了時は全軸OFFです。",
                  file=sys.stderr, flush=True)
            time.sleep(3)
        stage = "initial_position"
        leader_initial = readers["leader"].read().as_observation()
        initial = readers["follower"].read()
        follower_initial = initial.as_observation()
        stage = "initial_target"
        desired_check = guards["follower"].check(leader_initial)
        if not desired_check.accepted:
            raise JointReadError("Invalid Leader target: " + "; ".join(desired_check.reasons))
        policy = motion_policy(guards["follower"])
        emit(stream, "plan", execute=args.execute, duration_s=args.duration_s,
             mode="absolute", leader_initial=leader_initial, follower_initial=follower_initial,
             max_step=2.0, max_target_rate=20.0, period_s=0.1)
        if args.execute:
            follower = buses["follower"]
            sender = FollowerSender(follower, calibrations["follower"])
            loop = SafeRecordLoop(guards["follower"], lambda pair: pair[0], sender.quantize,
                                  sender, lambda action: None,
                                  motion_guard=policy, clock=time.monotonic)
            loop.start()
            # トルクOFFのまま現在位置を目標にし、全軸の読戻し後に有効化する。
            stage = "prepare_hold"
            time.sleep(0.01)  # 初回もMotionGuardの最短検査間隔を満たす。
            hold = loop.process_frame(follower_initial, follower_initial,
                                      observation_time=initial.read_started_at,
                                      observation_end_time=initial.read_completed_at)
            previous, previous_at = hold.sent_action, time.monotonic()
            for motor, value in raw_readers["follower"].raw.items():
                if follower.read("Goal_Position", motor, normalize=False, num_retry=0) != value:
                    raise ValueError(motor+": initial goal readback mismatch")
            stage = "enable_follower"
            for motor in MOTOR_IDS:
                enabled.append(motor)  # 書込みの応答が失われても解除対象に含める。
                follower.write("Torque_Enable", motor, 1, normalize=False, num_retry=0)
                if follower.read("Torque_Enable", motor, normalize=False, num_retry=0) != 1:
                    raise ValueError(motor+": torque enable unconfirmed")

            print(f"テレオペ開始：{args.duration_s:g}秒間。FollowerがLeaderの姿勢へ徐々に動きます。",
                  file=sys.stderr, flush=True)
            stage = "teleop"
            deadline = time.monotonic() + args.duration_s
            warned = False
            while time.monotonic() < deadline:
                time.sleep(0.1)
                if time.monotonic() >= deadline:
                    break
                leader = readers["leader"].read()
                observed = readers["follower"].read()
                if time.monotonic()-leader.read_started_at > 0.25:
                    raise ValueError("Leader observation is stale")
                desired = leader.as_observation()
                desired_check = guards["follower"].check(desired)
                if not desired_check.accepted:
                    raise JointReadError("Invalid Leader target: " + "; ".join(desired_check.reasons))
                target = limited_target(desired, previous, calibrations["follower"],
                                        time.monotonic()-previous_at, policy)
                result = loop.process_frame(target, observed.as_observation(),
                                            observation_time=observed.read_started_at,
                                            observation_end_time=observed.read_completed_at)
                previous, previous_at = result.sent_action, time.monotonic()
                emit(stream, "frame", index=result.frame_index, leader=leader.as_observation(),
                     observation=observed.as_observation(), sent_target=result.sent_action,
                     read_started_at=observed.read_started_at,
                     read_completed_at=observed.read_completed_at)
                if not warned and deadline-time.monotonic() <= 2:
                    print("まもなく終了し、Follower全軸をOFFにします。支持を維持してください。",
                          file=sys.stderr, flush=True)
                    warned = True
            loop.finish()
    except BaseException as exc:
        failed = True
        details = exc.details if isinstance(exc, (JointReadError, RegisterMismatch)) else {}
        if isinstance(exc, RecordFault):
            details = {"loop_stage": exc.fault.stage, "send_status": exc.fault.send_status.value}
            if "guard" in exc.fault.stage:
                details["reason"] = exc.fault.message
        elif type(exc) is ValueError:
            details = {"reason": str(exc)}
        print(f"テレオペ停止：{stage} / {type(exc).__name__}", file=sys.stderr, flush=True)
        emit(stream, "fault", stage=stage, error_type=type(exc).__name__, **details)
    finally:
        try:
            if enabled:
                # 一軸の通信・ログ失敗で、残りの軸の解除を飛ばさない。
                off, uncertain = [], []
                for motor in reversed(enabled):
                    try:
                        buses["follower"].write("Torque_Enable", motor, 0, normalize=False, num_retry=0)
                        if buses["follower"].read("Torque_Enable", motor, normalize=False, num_retry=0) != 0:
                            raise RuntimeError("Torque OFF unconfirmed")
                        off.append(motor)
                    except BaseException:
                        uncertain.append(motor)
                failed |= bool(uncertain)
                emit(stream, "torque_off", verified=off, unconfirmed=uncertain)
        finally:
            close_failed = []
            for role, bus in buses.items():
                if bus.is_connected:
                    try:
                        bus.disconnect(disable_torque=False)
                    except BaseException:
                        failed = True
                        close_failed.append(role)
            if close_failed:
                emit(stream, "fault", stage="bus_close", roles=close_failed)
    if not failed:
        emit(stream, "complete", executed=args.execute)
    return 1 if failed else 0


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    for role in ("leader", "follower"):
        cli.add_argument(f"--{role}-port", required=True)
        cli.add_argument(f"--{role}-calibration", type=Path, required=True)
        cli.add_argument(f"--{role}-sha256", required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--duration-s", type=float, default=10)
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = cli.parse_args(arguments)
    if not 0 < args.duration_s <= 30:
        cli.error("duration-s must be in (0, 30]")
    for role in ("leader", "follower"):
        if getattr(args, role+"_port") == getattr(args, ("follower" if role == "leader" else "leader")+"_port"):
            cli.error("Leader and Follower must have different ports")
        if Path(getattr(args, role+"_port")).parent != Path("/dev/serial/by-id"):
            cli.error("Use explicit /dev/serial/by-id/... ports")
        load_calibration(getattr(args, role+"_calibration"), getattr(args, role+"_sha256"))
    if args._worker:
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        logging.disable(logging.CRITICAL)
        return run_worker(args, sys.stdout)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            code = supervise([sys.executable, "-m", "src.control.guarded_teleop", *arguments, "--_worker"],
                             stream, args.duration_s+15, stderr=None)
    except OSError as exc:
        print(f"Teleop could not run: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Teleop {'completed' if code == 0 else 'failed'}: {args.output}")
    if code and args.execute:
        print("OFF未確認の軸がある場合は、アームを支持して電源を切ってください。自動再開はしません。")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
