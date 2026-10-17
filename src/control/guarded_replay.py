"""成功テレオペの全経路を固定配置用に再生する。既定は実機接続なしのプレビュー。

6軸の記録済み送信目標を再生し、腕の到達確認には記録済みFollower観測を参照する。
把持の成否や衝突は検出しない。
--check-start/--watch-startは読取りのみ。--executeだけがFollowerを動かし、終了時に全軸OFFを試みる。
"""

import argparse
import hashlib
import json
import logging
import signal
import sys
import time
from datetime import datetime
from pathlib import Path

from src.control.action_guard import MOTOR_IDS
from src.control.diagnose_observation import (
    RegisterMismatch, emit, load_calibration, make_hardware_bus, supervise, verify_registers,
)
from src.control.guarded_teleop import (
    FollowerSender, RawReader, limited_target, motion_policy, positions_from_raw,
    raw_from_positions, release_follower,
)
from src.control.review_teleop import parse_frames
from src.control.safe_record_loop import RecordFault, SafeRecordLoop
from src.control.timed_observation import JointReadError, TimedJointReader


ROOT = Path(__file__).resolve().parents[2]
PHASES = ("approach", "grasp", "box", "release", "retreat")
PERIOD = 0.1
START_TOLERANCE = 2.0
ARM_TOLERANCE = 3.0
OPEN_TOLERANCE = 2.0
WAIT_TIMEOUT = 5.0
GRASP_DWELL = 0.5
JOINT_LABELS = {
    "shoulder_pan": "土台の旋回", "shoulder_lift": "肩の上下", "elbow_flex": "肘の曲げ伸ばし",
    "wrist_flex": "手首の曲げ伸ばし", "wrist_roll": "手首の回転", "gripper": "爪の開閉",
}


def checked(action, guard):
    result = guard.check(action)
    if not result.accepted:
        raise ValueError("Invalid replay position: " + "; ".join(result.reasons))
    return dict(result.checked_action)


def prepare(args):
    """同じ入力バイト列を照合・解析。接続前に全フレームを検査し、制限内の指令へ分割する。"""
    calibration, guard = load_calibration(args.calibration, args.expected_sha256)
    source = args.source_log.read_bytes()
    checkpoints = args.checkpoints.read_bytes()
    source_hash = hashlib.sha256(source).hexdigest()
    checkpoints_hash = hashlib.sha256(checkpoints).hexdigest()
    for name, actual in (("source", source_hash), ("checkpoints", checkpoints_hash)):
        expected = getattr(args, "_"+name+"_sha256", None)
        if expected is not None and expected != actual:
            raise ValueError(f"{name} changed after parent preflight")
    source_plan, frames = parse_frames(source.decode("utf-8"))
    candidates = json.loads(checkpoints)
    poses = candidates["poses"]
    if [p["name"] for p in poses] != list(PHASES):
        raise ValueError("Checkpoints must be approach, grasp, box, release, retreat in this order")
    indices = [frame["index"] for frame in frames]
    if indices != list(range(1, len(frames)+1)):
        raise ValueError("Source frame indices must be consecutive from 1")
    selected = [p["frame_index"] for p in poses]
    if (selected != sorted(set(selected)) or selected[-1] != indices[-1]
            or any(index not in indices for index in selected)):
        raise ValueError("Checkpoints must increase and retreat must be the final source frame")
    for pose in poses:
        frame = frames[pose["frame_index"]-1]
        if any(pose[field] != frame[field] for field in ("observation", "sent_target")):
            raise ValueError("Checkpoint values do not match the source log")
    initial = checked(source_plan["follower_initial"], guard)
    previous = checked(positions_from_raw(raw_from_positions(initial, calibration), calibration), guard)
    policy = motion_policy(guard)
    steps = []
    by_index = dict(zip(selected, PHASES))
    previous_time = frames[0]["read_started_at"]-PERIOD
    for frame in frames:
        observed = checked(frame["observation"], guard)
        sent = checked(frame["sent_target"], guard)
        # 実測値を位置指令へ置き換えると、実機の指令と実測の差が再び加わり得る。
        # 送信目標を再現し、実測値は要所の到達参照として別に保持する。
        desired = sent
        desired = checked(positions_from_raw(raw_from_positions(desired, calibration), calibration), guard)
        arrival = {**observed, "gripper.pos": desired["gripper.pos"]}
        interval = max(PERIOD, frame["read_started_at"]-previous_time)
        if interval >= policy.max_interval_s:
            raise ValueError("Source frame gap exceeds the replay motion interval")
        previous_time = frame["read_started_at"]
        while True:
            target = checked(limited_target(desired, previous, calibration, PERIOD, policy), guard)
            reached = raw_from_positions(target, calibration) == raw_from_positions(desired, calibration)
            steps.append({"source_frame": frame["index"], "period_s": interval,
                          "target": target, "source_frame_complete": reached,
                          "checkpoint": by_index.get(frame["index"]) if reached else None,
                          "arrival_target": arrival if reached and frame["index"] in by_index else None})
            previous = target
            interval = PERIOD
            if reached:
                break
    report = {
        "mode": "fixed_follower_path", "source_log": str(args.source_log),
        "source_sha256": source_hash, "checkpoints_sha256": checkpoints_hash,
        "calibration_sha256": args.expected_sha256.lower(),
        "hardware_verified": False, "grasp_verified": False,
        "source_frames": len(frames), "target_steps": len(steps),
        "nominal_motion_s": sum(s["period_s"] for s in steps),
        "initial": initial, "start_tolerance": START_TOLERANCE,
        "arm_tolerance_degree": ARM_TOLERANCE, "open_tolerance_points": OPEN_TOLERANCE,
        "checkpoint_timeout_s": WAIT_TIMEOUT, "grasp_dwell_s": GRASP_DWELL,
        "max_step": 2.0, "max_target_rate": 20.0,
        "target_rule": "F1-F6: source sent_target",
        "arrival_rule": "F1-F5: source Follower observation; F6: source sent_target",
        "checkpoints": [{"name": p["name"], "source_frame": p["frame_index"]} for p in poses],
        "steps": steps,
    }
    return calibration, guard, report


def check_start(current, plan, stream, *, require_match=True):
    differences = {key: current[key]-expected for key, expected in plan["initial"].items()}
    matches = all(abs(delta) <= START_TOLERANCE for delta in differences.values())
    emit(stream, "start_check", expected=plan["initial"], observation=current,
         differences=differences, tolerance=START_TOLERANCE, matched=matches)
    for motor, motor_id in MOTOR_IDS.items():
        key = motor+".pos"
        delta = differences[key]
        hint = "OK" if abs(delta) <= START_TOLERANCE else "値を減らす" if delta > 0 else "値を増やす"
        print(f"F{motor_id} {JOINT_LABELS[motor]}: 開始 {plan['initial'][key]:.2f} / 現在 {current[key]:.2f} "
              f"/ 差 {delta:+.2f} / {hint}", file=sys.stderr, flush=True)
    if require_match and not matches:
        raise ValueError("Start pose mismatch (allowed difference: 2 degrees / gripper 2 points)")
    return matches


def watch_start(reader, initial, plan, stream, duration_s):
    """トルクOFFのまま手で合わせるための約1 Hz表示。一致しても動作を開始しない。"""
    print("開始姿勢あわせ：読取りのみ・全軸トルクOFFを確認済み。手や支えで重さを受けてください。\n"
          "表示の『開始』は成功試行の開始姿勢です。校正時の姿勢ではありません。\n"
          "一つずつ少し動かして数値の変化を見てください。自動で保持・移動はしません。",
          file=sys.stderr, flush=True)
    deadline = time.monotonic()+duration_s
    observed = initial
    consecutive = 0
    while True:
        print("\n開始姿勢との差（F1〜F5=度、F6=ポイント）：", file=sys.stderr, flush=True)
        matched = check_start(observed.as_observation(), plan, stream, require_match=False)
        consecutive = consecutive+1 if matched else 0
        if consecutive >= 3:
            print("開始姿勢が3回連続で許容差内です。表示を終了します。位置指令・トルク操作は行っていません。",
                  file=sys.stderr, flush=True)
            return True
        remaining = deadline-time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(1.0, remaining))
        if time.monotonic() >= deadline:
            break
        observed = reader.read()
    print("表示時間が終了しました。開始姿勢の一致はまだ確定していません。支持を維持してください。",
          file=sys.stderr, flush=True)
    return False


class ReplayRunner:
    def __init__(self, reader, raw_reader, sender, guard, stream, deadline):
        self.reader, self.raw_reader, self.stream = reader, raw_reader, stream
        self.deadline = deadline
        self.calibration = raw_reader.calibration
        self.policy = motion_policy(guard)
        self.previous = None
        self.previous_at = None
        self.target_reached = False
        self.loop = SafeRecordLoop(guard, lambda pair: pair[0], sender.quantize, sender,
                                   lambda action: None, motion_guard=self.policy,
                                   clock=time.monotonic)
        self.loop.start()

    def send(self, target, observed):
        result = self.loop.process_frame(target, observed.as_observation(),
                                         observation_time=observed.read_started_at,
                                         observation_end_time=observed.read_completed_at)
        self.previous, self.previous_at = result.sent_action, time.monotonic()
        return result

    def tick(self, step, *, waiting=False, deadline=None):
        limit = min(self.deadline, deadline if deadline is not None else self.deadline)
        if time.monotonic()+step["period_s"] >= limit:
            raise TimeoutError("Replay / checkpoint time limit exceeded")
        time.sleep(step["period_s"])
        observed = self.reader.read()
        if time.monotonic() >= limit:
            raise TimeoutError("Replay / checkpoint time limit exceeded")
        target = limited_target(step["target"], self.previous, self.calibration,
                                time.monotonic()-self.previous_at, self.policy)
        result = self.send(target, observed)
        self.target_reached = (raw_from_positions(result.sent_action, self.calibration)
                               == raw_from_positions(step["target"], self.calibration))
        emit(self.stream, "frame", source_frame=step["source_frame"], waiting=waiting,
             source_frame_complete=step["source_frame_complete"] and self.target_reached and not waiting,
             checkpoint=step["checkpoint"], observation=observed.as_observation(),
             arrival_target=step["arrival_target"],
             follower_raw=self.raw_reader.raw, desired_target=step["target"], sent_target=result.sent_action,
             read_started_at=observed.read_started_at, read_completed_at=observed.read_completed_at)
        return observed.as_observation()

    def wait_checkpoint(self, step):
        phase = step["checkpoint"]
        print(f"到達待ち：{phase}（最大{WAIT_TIMEOUT:g}秒）。", file=sys.stderr, flush=True)
        emit(self.stream, "checkpoint_wait", name=phase, source_frame=step["source_frame"])
        deadline = time.monotonic()+WAIT_TIMEOUT
        hold = {**step, "period_s": PERIOD}
        reference = step["arrival_target"]
        consecutive = 0
        stable_since = None
        while True:
            current = self.tick(hold, waiting=True, deadline=deadline)
            arms_arrived = all(abs(current[m+".pos"]-reference[m+".pos"]) <= ARM_TOLERANCE
                               for m in MOTOR_IDS if m != "gripper")
            # 缶に接触した閉じ側では位置一致を要求しない。把持の検出とは区別する。
            grip_arrived = (phase in ("grasp", "box")
                            or abs(current["gripper.pos"]-reference["gripper.pos"]) <= OPEN_TOLERANCE)
            consecutive = consecutive+1 if arms_arrived and grip_arrived else 0
            if consecutive < 2:
                stable_since = None
                continue
            now = time.monotonic()
            if stable_since is None:
                stable_since = now
            if phase != "grasp" or now-stable_since >= GRASP_DWELL:
                emit(self.stream, "checkpoint_complete", name=phase, observation=current,
                     target=step["target"], arrival_target=reference,
                     gripper_position_required=phase not in ("grasp", "box"),
                     grasp_verified=False)
                return


def runtime_budget(plan):
    return plan["nominal_motion_s"]+len(PHASES)*WAIT_TIMEOUT+10


def run_worker(args, stream, *, activation_deadline=None):
    bus = None
    enabled = []
    failed = False
    watching = getattr(args, "watch_start", False)
    watch_matched = False
    stage = "preflight"
    try:
        calibration, guard, plan = prepare(args)
        emit(stream, "plan", **{k: v for k, v in plan.items() if k != "steps"}, execute=args.execute)
        stage = "follower_connect"
        bus = make_hardware_bus(args.port, calibration)
        bus.connect(handshake=False)
        bus.port_handler.ser.exclusive = True
        bus.port_handler.ser.write_timeout = 0.25
        stage = "register_check"
        registers = verify_registers(bus, calibration)
        if any(row["Torque_Enable"] != 0 for row in registers.values()):
            raise ValueError("Follower motors must initially have torque OFF")
        raw_reader = RawReader(bus, calibration, role="follower")
        reader = TimedJointReader(raw_reader, guard, max_read_duration_s=0.25, clock=time.monotonic)
        if args.execute:
            print("3秒後に開始姿勢を照合し、一致した場合だけ固定動作を開始します。終了時は全軸OFFです。",
                  file=sys.stderr, flush=True)
            time.sleep(3)
        stage = "initial_position"
        initial = reader.read()
        if watching:
            stage = "watch_start"
            watch_matched = watch_start(reader, initial, plan, stream, args.watch_duration_s)
        else:
            check_start(initial.as_observation(), plan, stream)
        if args.execute:
            if activation_deadline is not None and time.monotonic() >= activation_deadline:
                raise TimeoutError("Recognition image expired before motion startup")
            runner = ReplayRunner(reader, raw_reader, FollowerSender(bus, calibration), guard,
                                  stream, time.monotonic()+runtime_budget(plan))
            stage = "prepare_hold"
            time.sleep(0.01)
            runner.send(initial.as_observation(), initial)
            for motor, value in raw_reader.raw.items():
                if bus.read("Goal_Position", motor, normalize=False, num_retry=0) != value:
                    raise ValueError(motor+": initial goal readback mismatch")
            stage = "enable_follower"
            if activation_deadline is not None and time.monotonic() >= activation_deadline:
                raise TimeoutError("Recognition image expired before torque enable")
            for motor in MOTOR_IDS:
                enabled.append(motor)
                bus.write("Torque_Enable", motor, 1, normalize=False, num_retry=0)
                if bus.read("Torque_Enable", motor, normalize=False, num_retry=0) != 1:
                    raise ValueError(motor+": torque enable unconfirmed")
            print("固定動作を開始します。終了・異常停止時のアーム支持を維持してください。",
                  file=sys.stderr, flush=True)
            for step in plan["steps"]:
                stage = "replay"
                while True:
                    runner.tick(step)
                    if runner.target_reached:
                        break
                if step["checkpoint"]:
                    stage = "wait_"+step["checkpoint"]
                    runner.wait_checkpoint(step)
                    if step["checkpoint"] == "release":
                        print("この後の退避が終わるとFollower全軸をOFFにします。支持を維持してください。",
                              file=sys.stderr, flush=True)
            runner.loop.finish()
    except BaseException as exc:
        failed = True
        details = dict(exc.details) if isinstance(exc, (JointReadError, RegisterMismatch)) else {}
        if isinstance(exc, JointReadError):
            details.update(role="follower", register="Present_Position")
        elif isinstance(exc, RecordFault):
            details = {"loop_stage": exc.fault.stage, "send_status": exc.fault.send_status.value}
            if "guard" in exc.fault.stage:
                details["reason"] = exc.fault.message
            if exc.fault.stage == "send_action":
                details.update(role="follower", operation="sync_write", register="Goal_Position")
        elif type(exc) in (ValueError, TimeoutError):
            details = {"reason": str(exc)}
        elif isinstance(exc, ConnectionError):
            details = {"role": "follower", "operation": stage}
        print(f"固定動作停止：{stage} / {type(exc).__name__}", file=sys.stderr, flush=True)
        if "reason" in details:
            print("理由："+details["reason"], file=sys.stderr, flush=True)
        emit(stream, "fault", stage=stage, error_type=type(exc).__name__, **details)
    finally:
        try:
            if enabled:
                failed |= not release_follower(bus, enabled, stream)
        finally:
            if bus is not None and bus.is_connected:
                try:
                    bus.disconnect(disable_torque=False)
                except BaseException as exc:
                    failed = True
                    emit(stream, "fault", stage="bus_close", error_type=type(exc).__name__)
    if not failed:
        emit(stream, "complete", executed=args.execute, grasp_verified=False,
             **({"start_matched": watch_matched} if watching else {}))
    return int(failed)


def main(argv=None):
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-log", type=Path, required=True)
    cli.add_argument("--checkpoints", type=Path, required=True)
    cli.add_argument("--calibration", type=Path, required=True)
    cli.add_argument("--expected-sha256", required=True)
    cli.add_argument("--output", type=Path)
    cli.add_argument("--port", help="Followerの/dev/serial/by-id/...（実機モードだけ必須）")
    mode = cli.add_mutually_exclusive_group()
    mode.add_argument("--check-start", action="store_true", help="開始姿勢の照合のみ。書込みなし")
    mode.add_argument("--watch-start", action="store_true", help="現在値と調整方向を約1秒ごとに表示。書込みなし")
    mode.add_argument("--execute", action="store_true", help="Followerだけで固定動作を実行")
    cli.add_argument("--watch-duration-s", type=float, default=60,
                     help="--watch-startの表示時間（秒、既定60、最大300）")
    cli.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    for name in ("source", "checkpoints"):
        cli.add_argument("--_"+name+"-sha256", help=argparse.SUPPRESS)
    args = cli.parse_args(argv)
    hardware = args.check_start or args.watch_start or args.execute
    if not 0 < args.watch_duration_s <= 300:
        cli.error("watch-duration-s must be in (0, 300]")
    if hardware and (not args.port or Path(args.port).parent != Path("/dev/serial/by-id")):
        cli.error("Use --port /dev/serial/by-id/... for the Follower")
    if args._worker and not hardware:
        cli.error("Worker requires a hardware mode")
    if args._worker:
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        logging.disable(logging.CRITICAL)
        return run_worker(args, sys.stdout)
    try:
        _, _, plan = prepare(args)
        suffix = "jsonl" if hardware else "json"
        output = args.output or ROOT / "outputs" / f"replay_{datetime.now():%Y%m%d_%H%M%S_%f}.{suffix}"
        with output.open("x", encoding="utf-8") as stream:
            if not hardware:
                json.dump(plan, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.write("\n")
                print(f"プレビュー：元{plan['source_frames']}フレーム → {plan['target_steps']}指令。"
                      f"移動設定時間 {plan['nominal_motion_s']:.2f}秒＋要所の待機。")
                print("開始姿勢：", json.dumps(plan["initial"], ensure_ascii=False))
                print("実機接続・送信は行っていません。経路と要所の実機確認は未完了です。")
                code = 0
            else:
                command = [sys.executable, "-m", "src.control.guarded_replay",
                           "--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
                           "--calibration", str(args.calibration), "--expected-sha256", args.expected_sha256,
                           "--port", args.port,
                           "--execute" if args.execute else "--watch-start" if args.watch_start else "--check-start",
                           "--watch-duration-s", str(args.watch_duration_s),
                           "--_source-sha256", plan["source_sha256"],
                           "--_checkpoints-sha256", plan["checkpoints_sha256"], "--_worker"]
                timeout = runtime_budget(plan)+15 if args.execute else args.watch_duration_s+15 if args.watch_start else 15
                code = supervise(command, stream, timeout,
                                 stderr=None)
        print(f"Replay {'completed' if code == 0 else 'failed'}: {output}")
        if code and args.execute:
            print("OFF未確認の軸がある場合は、アームを支持して電源を切ってください。自動再開はしません。")
        return code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"固定動作の準備に失敗しました：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
