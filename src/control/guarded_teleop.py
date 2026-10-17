"""Leaderの姿勢へ、送信目標を段階的に近づける短時間テレオペ。

既定は指定時間の継続読取りのみ。--executeでFollower全軸を有効化し、終了時に解除する。
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


PICK_PLACE_GUIDE = (
    (0, "settle", "Leaderを静止し、Followerの追従を待ちます。"),
    (3, "approach", "爪を開き、缶があった位置の上へ近づけます。爪先を机から離してください。"),
    (9, "grasp", "その位置で爪を閉じます。Followerの到着を見てから操作してください。"),
    (14, "box", "爪を閉じたまま持ち上げ、白い箱の上へ移動します。箱の縁に触れないようにします。"),
    (20, "release", "箱の上で腕を止め、爪を開きます。"),
    (24, "retreat", "爪を開いたまま、箱から離して退避します。"),
    (28, "support", "Leaderを静止し、Followerの追従を待ちます。終了時の落下に備えて支持してください。"
                     "{duration_s:g}秒で全軸OFFになります。"),
)


def show_guide(stream, elapsed_s, previous_phase, next_frame_index, duration_s=30):
    """人への案内を記録するだけ。関節指令や工程の完了判定には使わない。"""
    _, phase, instruction = next(row for row in reversed(PICK_PLACE_GUIDE)
                                 if elapsed_s >= row[0]*duration_s/30)
    instruction = instruction.format(duration_s=duration_s)
    if phase != previous_phase:
        emit(stream, "guide", phase=phase, elapsed_s=elapsed_s, instruction=instruction,
             next_frame_index=next_frame_index, completion_verified=False)
        print(f"\n工程案内 [{phase}]：{instruction}", file=sys.stderr, flush=True)
    return phase


def positions_from_raw(raw, calibration):
    result = {}
    for motor in MOTOR_IDS:
        low, high = calibration[motor]["range_min"], calibration[motor]["range_max"]
        if motor == "gripper":
            if not low <= raw[motor] <= high:
                exceeded = low-raw[motor] if raw[motor] < low else raw[motor]-high
                error = JointReadError(
                    f"Raw gripper position {raw[motor]} is outside calibration [{low}, {high}] "
                    f"by {exceeded} ticks"
                )
                error.details.update(
                    motor=motor, raw_position=raw[motor], range_min=low, range_max=high,
                    normalized_position=(raw[motor]-low) * 100 / (high-low),
                    exceeded_boundary="lower" if raw[motor] < low else "upper",
                    exceeded_by_ticks=exceeded,
                )
                raise error
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
    """生値を保持し、Leader F6だけ明示した微小な端点超過を入力処理する。"""

    def __init__(self, bus, calibration, *, role, gripper_tolerance_ticks=0):
        self.bus, self.calibration = bus, calibration
        self.role = role
        if type(gripper_tolerance_ticks) is not int or not 0 <= gripper_tolerance_ticks <= 8:
            raise ValueError("Gripper input tolerance must be an integer in [0, 8] ticks")
        if role != "leader" and gripper_tolerance_ticks:
            raise ValueError("Only Leader input may have a gripper tolerance")
        self.gripper_tolerance_ticks = gripper_tolerance_ticks
        self.raw = None
        self.gripper_adjustment = None

    def sync_read(self, register, *, normalize, num_retry):
        self.raw = self.bus.sync_read(register, normalize=False, num_retry=num_retry)
        self.gripper_adjustment = None
        converted_raw = self.raw
        value = self.raw["gripper"]
        low, high = self.calibration["gripper"]["range_min"], self.calibration["gripper"]["range_max"]
        tolerance = self.gripper_tolerance_ticks
        if (self.role == "leader" and tolerance and type(value) is int
                and 0 <= value <= 4095 and low-tolerance <= value <= high+tolerance
                and not low <= value <= high):
            bounded = min(high, max(low, value))
            converted_raw = {**self.raw, "gripper": bounded}
            self.gripper_adjustment = {
                "role": "leader", "motor": "gripper", "raw_position": value,
                "range_min": low, "range_max": high, "bounded_raw_position": bounded,
                "exceeded_by_ticks": abs(value-bounded), "tolerance_ticks": tolerance,
                "normalized_input": (value-low)*100/(high-low),
                "normalized_target": (bounded-low)*100/(high-low),
            }
        try:
            if type(value) is not int or not 0 <= value <= 4095:
                raise JointReadError("Expected a raw gripper position integer in [0, 4095]")
            positions = positions_from_raw(converted_raw, self.calibration)
        except JointReadError as exc:
            exc.details.update(role=self.role, register=register)
            raise
        return {key.removesuffix(".pos"): value for key, value in positions.items()}


class TeleopDisplay:
    """端点警告は状態変化時、関節値は約1秒ごとにstderrへ表示する。制御値は変えない。"""

    def __init__(self, stream, leader_guard, follower_guard):
        leader_bounds = {b.action_key: b for b in leader_guard.bounds}
        self.bounds = [
            (b.action_key, max(b.minimum, leader_bounds[b.action_key].minimum),
             min(b.maximum, leader_bounds[b.action_key].maximum), b.unit)
            for b in follower_guard.bounds
        ]
        self.stream = stream
        self.previous_warnings = ()
        self.previous_endpoint = None
        self.next_status_at = 0.0

    def update(self, leader, follower, adjustment, now):
        warnings = []
        for key, low, high, unit in self.bounds:
            value = leader[key]
            remaining = min(value-low, high-value)
            if remaining <= 5:
                warnings.append({"motor": key.removesuffix(".pos"), "value": value,
                                 "minimum": low, "maximum": high, "unit": unit,
                                 "boundary": "lower" if value-low <= high-value else "upper"})
        signature = tuple((w["motor"], w["boundary"]) for w in warnings)
        if signature != self.previous_warnings:
            emit(self.stream, "leader_limit_warning", limits=warnings)
            for warning in warnings:
                motor = warning["motor"]
                print(f"端点注意：Leader F{MOTOR_IDS[motor]} {motor}={warning['value']:.2f} "
                      f"/ 共通範囲 [{warning['minimum']:.2f}, {warning['maximum']:.2f}] "
                      f"{warning['unit']}。端へ押し付けないでください。", file=sys.stderr, flush=True)
            self.previous_warnings = signature
        endpoint = adjustment["normalized_target"] if adjustment else None
        if endpoint != self.previous_endpoint:
            emit(self.stream, "leader_gripper_endpoint", active=adjustment is not None,
                 **(adjustment or {}))
            if adjustment:
                print(f"入力端点処理：Leader F6 raw={adjustment['raw_position']} "
                      f"→{adjustment['bounded_raw_position']} "
                      f"（超過{adjustment['exceeded_by_ticks']} ticks）。Followerの範囲は維持します。",
                      file=sys.stderr, flush=True)
            else:
                print("Leader F6は校正範囲内へ戻りました。", file=sys.stderr, flush=True)
            self.previous_endpoint = endpoint
        if now >= self.next_status_at:
            values = "  ".join(f"F{motor_id} {leader[name+'.pos']:.1f}/{follower[name+'.pos']:.1f}"
                               for name, motor_id in MOTOR_IDS.items())
            print("関節値 Leader/Follower（F1〜F5=度、F6=ポイント）：" + values,
                  file=sys.stderr, flush=True)
            self.next_status_at = now+1.0


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


def check_leader_target(desired, follower_guard):
    checked = follower_guard.check(desired)
    if not checked.accepted:
        error = JointReadError("Invalid Leader target: " + "; ".join(checked.reasons))
        error.details.update(role="leader", checked_against="follower", target=dict(desired))
        raise error


def release_follower(bus, enabled, stream):
    """一軸の解除失敗でも他軸を試みる。全軸のOFF読戻しに成功した場合だけTrue。"""
    off, uncertain, failures = [], [], []
    for motor in reversed(enabled):
        operation = "write"
        try:
            bus.write("Torque_Enable", motor, 0, normalize=False, num_retry=0)
            operation = "read"
            if bus.read("Torque_Enable", motor, normalize=False, num_retry=0) != 0:
                raise RuntimeError("Torque OFF unconfirmed")
            off.append(motor)
        except BaseException as exc:
            uncertain.append(motor)
            failures.append({"motor": motor, "operation": operation,
                             "error_type": type(exc).__name__})
    if uncertain:
        print("トルクOFF未確認：" + ", ".join(uncertain)
              + "。アームを支持し、Followerのコンセントを抜いてください。",
              file=sys.stderr, flush=True)
    emit(stream, "torque_off", verified=off, unconfirmed=uncertain, failures=failures)
    return not uncertain


def run_worker(args, stream):
    buses = {}
    enabled = []
    failed = False
    reading_role = None
    stage = "calibration"
    guided = getattr(args, "guide_pick_place", False)
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
            raw_readers[role] = RawReader(
                bus, calibration, role=role,
                gripper_tolerance_ticks=args.leader_gripper_tolerance_ticks if role == "leader" else 0,
            )
            readers[role] = TimedJointReader(raw_readers[role], guard, max_read_duration_s=0.25,
                                           clock=time.monotonic)

        if args.execute:
            print("3秒後にFollower全軸を有効化し、Leaderの姿勢へ徐々に動かします。終了時は全軸OFFです。",
                  file=sys.stderr, flush=True)
            time.sleep(3)
        stage = "initial_position"
        reading_role = "leader"
        leader_initial = readers["leader"].read().as_observation()
        reading_role = "follower"
        initial = readers["follower"].read()
        reading_role = None
        follower_initial = initial.as_observation()
        stage = "initial_target"
        check_leader_target(leader_initial, guards["follower"])
        policy = motion_policy(guards["follower"])
        emit(stream, "plan", execute=args.execute, duration_s=args.duration_s,
             mode="absolute", leader_initial=leader_initial, follower_initial=follower_initial,
             leader_initial_raw=raw_readers["leader"].raw,
             follower_initial_raw=raw_readers["follower"].raw,
             leader_gripper_tolerance_ticks=args.leader_gripper_tolerance_ticks,
             **({"guide": f"pick_place_{args.duration_s:g}s"} if guided else {}),
             max_step=2.0, max_target_rate=20.0, period_s=0.1)
        display = TeleopDisplay(stream, guards["leader"], guards["follower"])
        display.update(leader_initial, follower_initial,
                       raw_readers["leader"].gripper_adjustment, time.monotonic())
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

        if args.execute:
            print(f"テレオペ開始：{args.duration_s:g}秒間。FollowerがLeaderの姿勢へ徐々に動きます。",
                  file=sys.stderr, flush=True)
        else:
            print(f"読取り開始：{args.duration_s:g}秒間。位置指令・トルク操作は行いません。",
                  file=sys.stderr, flush=True)
        stage = "teleop" if args.execute else "observation"
        deadline = time.monotonic() + args.duration_s
        warned = False
        index = 0
        guide_phase = None
        if guided:
            print("工程案内です。動かすのはLeaderです。案内は工程完了の判定ではありません。\n"
                  "間に合わない場合は次の工程へ急がず、Ctrl+Cで終了してください。",
                  file=sys.stderr, flush=True)
            guide_phase = show_guide(stream, 0, guide_phase, 1, args.duration_s)
        while time.monotonic() < deadline:
            time.sleep(0.1)
            if time.monotonic() >= deadline:
                break
            if guided:
                guide_phase = show_guide(stream, time.monotonic()-(deadline-args.duration_s),
                                         guide_phase, index+1, args.duration_s)
            reading_role = "leader"
            leader = readers["leader"].read()
            reading_role = "follower"
            observed = readers["follower"].read()
            reading_role = None
            if time.monotonic()-leader.read_started_at > 0.25:
                raise ValueError("Leader observation is stale")
            desired = leader.as_observation()
            check_leader_target(desired, guards["follower"])
            index += 1
            if args.execute:
                target = limited_target(desired, previous, calibrations["follower"],
                                        time.monotonic()-previous_at, policy)
                result = loop.process_frame(target, observed.as_observation(),
                                            observation_time=observed.read_started_at,
                                            observation_end_time=observed.read_completed_at)
                previous, previous_at = result.sent_action, time.monotonic()
            emit(stream, "frame" if args.execute else "observation", index=index,
                 **({"guide_phase": guide_phase} if guided else {}),
                 leader=leader.as_observation(), observation=observed.as_observation(),
                 leader_raw=raw_readers["leader"].raw, follower_raw=raw_readers["follower"].raw,
                 leader_gripper_adjustment=raw_readers["leader"].gripper_adjustment,
                 **({"sent_target": result.sent_action} if args.execute else {}),
                 read_started_at=observed.read_started_at,
                 read_completed_at=observed.read_completed_at)
            display.update(leader.as_observation(), observed.as_observation(),
                           raw_readers["leader"].gripper_adjustment, time.monotonic())
            if args.execute and not warned and deadline-time.monotonic() <= 2:
                print("まもなく終了し、Follower全軸をOFFにします。支持を維持してください。",
                      file=sys.stderr, flush=True)
                warned = True
        if args.execute:
            loop.finish()
    except BaseException as exc:
        failed = True
        details = dict(exc.details) if isinstance(exc, (JointReadError, RegisterMismatch)) else {}
        if isinstance(exc, JointReadError) and reading_role is not None:
            # RawReaderの換算後にTimedJointReaderが拒否した軸にも機器名を残す。
            details.update(role=reading_role, register="Present_Position")
            if str(exc).startswith("Invalid joint observation:"):
                details["raw_positions"] = raw_readers[reading_role].raw
        if isinstance(exc, RecordFault):
            details = {"loop_stage": exc.fault.stage, "send_status": exc.fault.send_status.value}
            if exc.fault.stage == "send_action" and isinstance(exc.__cause__, ConnectionError):
                details.update(role="follower", operation="sync_write", register="Goal_Position",
                               motors=list(MOTOR_IDS))
            if "guard" in exc.fault.stage:
                details["reason"] = exc.fault.message
        elif isinstance(exc, ConnectionError) and reading_role is not None:
            details = {"role": reading_role, "operation": "sync_read",
                       "register": "Present_Position", "motors": list(MOTOR_IDS)}
        elif type(exc) is ValueError:
            details = {"reason": str(exc)}
        print(f"テレオペ停止：{stage} / {type(exc).__name__}", file=sys.stderr, flush=True)
        if "reason" in details:
            print(f"理由：{details.get('role', 'unknown')} / {details['reason']}",
                  file=sys.stderr, flush=True)
        if "operation" in details:
            print(f"通信失敗：{details['role']} / {details['operation']} / {details['register']}",
                  file=sys.stderr, flush=True)
        emit(stream, "fault", stage=stage, error_type=type(exc).__name__, **details)
    finally:
        try:
            if enabled:
                failed |= not release_follower(buses["follower"], enabled, stream)
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
    cli.add_argument("--leader-gripper-tolerance-ticks", type=int, default=8,
                     help="Leader F6だけの入力端点許容幅（0〜8 ticks、既定8、0で厳格拒否）")
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--guide-pick-place", action="store_true",
                     help="5工程を画面で案内し時刻を記録（--execute、--duration-s 30または60が必要）")
    cli.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = cli.parse_args(arguments)
    maximum_duration = 60 if args.guide_pick_place else 30
    if not 0 < args.duration_s <= maximum_duration:
        cli.error(f"duration-s must be in (0, {maximum_duration}]")
    if not 0 <= args.leader_gripper_tolerance_ticks <= 8:
        cli.error("leader-gripper-tolerance-ticks must be in [0, 8]")
    if args.guide_pick_place and (not args.execute or args.duration_s not in (30, 60)):
        cli.error("guide-pick-place requires --execute --duration-s 30 or 60")
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
