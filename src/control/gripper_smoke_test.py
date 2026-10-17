"""F6だけを現在位置から約2ポイント開く単発テスト。既定は読取りのみ。

F1〜F5の位置・トルクには書き込まない。アームを支持し、爪を空にして使用する。
正常終了時はF6のトルクもOFFに戻す。強制終了や通信障害時の解除は保証できない。
"""

import argparse
import signal
import sys
import time
from pathlib import Path

from src.control.diagnose_observation import (
    emit, load_calibration, make_hardware_bus, supervise, verify_registers,
)
from src.control.timed_observation import JointReadError, TimedJointReader


def run_worker(args, stream):
    """爪の現在値と小さな開き目標を確認し、実行指定時だけF6を動かす。

    腕の5軸は書き換えず、爪の校正幅の2%を整数tickへ切り捨てて開く。
    有効化を試みた場合はfinallyでF6のOFFと読戻しを試す。成功0、異常1。
    """
    bus = None
    activation_attempted = False
    failed = False
    stage = "calibration"
    try:
        calibration, guard = load_calibration(Path(args.calibration), args.expected_sha256)
        stage = "bus_connect"
        bus = make_hardware_bus(args.port, calibration)
        bus.connect(handshake=False)
        bus.port_handler.ser.exclusive = True
        bus.port_handler.ser.write_timeout = 0.25
        stage = "register_check"
        registers = verify_registers(bus, calibration)
        if any(item["Torque_Enable"] != 0 for item in registers.values()):
            raise ValueError("All six motors must initially have torque OFF")

        stage = "initial_position"
        reader = TimedJointReader(bus, guard, max_read_duration_s=0.25)
        positions = reader.read().as_observation()
        raw = bus.read("Present_Position", "gripper", normalize=False, num_retry=0)
        low, high = calibration["gripper"]["range_min"], calibration["gripper"]["range_max"]
        # RANGE_0_100の読取りはLeRobot内でclampされるため、生値も確認する。
        if type(raw) is not int or not low <= raw <= high:
            raise ValueError("Raw gripper position is outside calibration")
        initial = (raw - low) * 100 / (high - low)
        target_raw = raw + int((high - low) * 0.02)
        target = (target_raw - low) * 100 / (high - low)
        candidate = {**positions, "gripper.pos": target}
        checked = guard.check(candidate)
        if not checked.accepted:
            raise JointReadError("Invalid target: " + "; ".join(checked.reasons))
        emit(stream, "plan", execute=args.execute, positions=positions,
             gripper_initial=initial, gripper_target=target, delta=target-initial,
             motor_writes_only="gripper.Goal_Position,gripper.Torque_Enable")

        if args.execute:
            stage = "prepare_goal"
            # 古い目標で動き出さないよう、OFF中に現在位置を目標へ設定する。
            if bus.read("Torque_Enable", "gripper", normalize=False, num_retry=0) != 0:
                raise ValueError("Gripper torque changed before goal preparation")
            bus.write("Goal_Position", "gripper", raw, normalize=False, num_retry=0)
            if bus.read("Goal_Position", "gripper", normalize=False, num_retry=0) != raw:
                raise ValueError("Initial goal readback mismatch")
            stage = "enable_gripper"
            activation_attempted = True  # 応答が失われても解除を試みる。
            emit(stream, "activation_attempted", motor="gripper")
            bus.write("Torque_Enable", "gripper", 1, normalize=False, num_retry=0)
            if bus.read("Torque_Enable", "gripper", normalize=False, num_retry=0) != 1:
                raise ValueError("Gripper torque enable was not confirmed")

            stage = "send_target"
            # 有効化後にも現在位置を読み、手動移動などで大きな指令差が出るのを防ぐ。
            fresh = reader.read()
            current = fresh.as_observation()["gripper.pos"]
            if abs(current-initial) > 0.5 or not 0 < target-current <= 2.5:
                raise ValueError("Gripper moved before target send")
            if time.monotonic() - fresh.read_started_at > 0.25:
                raise ValueError("Gripper observation is stale")
            bus.write("Goal_Position", "gripper", target_raw, normalize=False, num_retry=0)
            emit(stream, "target_sent", gripper_target=target)
            time.sleep(0.5)
            stage = "result_read"
            observed_raw = bus.read("Present_Position", "gripper", normalize=False, num_retry=0)
            observed = (observed_raw-low) * 100 / (high-low)
            reached = low <= observed_raw <= high and abs(observed-target) <= 0.5
            emit(stream, "result", gripper_observed=observed, gripper_raw=observed_raw,
                 gripper_target=target, reached=reached)
            if not reached:
                raise ValueError("Gripper did not reach target; no retry")
    except BaseException as exc:
        failed = True
        reason = exc.details if isinstance(exc, JointReadError) else {}
        # ここで生成したValueErrorの理由だけ出す。通信例外の本文は出さない。
        if type(exc) is ValueError:
            reason = {"reason": str(exc)}
        emit(stream, "fault", stage=stage, error_type=type(exc).__name__, **reason)
    finally:
        try:
            if activation_attempted:
                try:
                    bus.write("Torque_Enable", "gripper", 0, normalize=False, num_retry=0)
                    if bus.read("Torque_Enable", "gripper", normalize=False, num_retry=0) != 0:
                        raise RuntimeError("Gripper torque OFF was not confirmed")
                    emit(stream, "torque_off_verified", motor="gripper")
                except BaseException as exc:
                    failed = True
                    emit(stream, "fault", stage="gripper_torque_off", error_type=type(exc).__name__)
        finally:
            if bus is not None and bus.is_connected:
                try:
                    bus.disconnect(disable_torque=False)
                except BaseException as exc:
                    failed = True
                    emit(stream, "fault", stage="bus_close", error_type=type(exc).__name__)
    if not failed:
        emit(stream, "complete", executed=args.execute)
    return 1 if failed else 0


def main(argv=None):
    """ポート・校正を検査して、上限15秒の子プロセスで単発テストを行うCLI。

    --executeなしでも実機を読み取るが、位置指令やトルク操作は行わない。
    """
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--port", required=True)
    cli.add_argument("--calibration", type=Path, required=True)
    cli.add_argument("--expected-sha256", required=True)
    cli.add_argument("--output", type=Path, required=True, help="New JSONL file")
    cli.add_argument("--execute", action="store_true", help="Actually enable and move F6; otherwise preview only")
    cli.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = cli.parse_args(arguments)
    if Path(args.port).parent != Path("/dev/serial/by-id"):
        cli.error("Use an explicit /dev/serial/by-id/... Follower port")
    load_calibration(args.calibration, args.expected_sha256)
    if args._worker:
        # 親の停止要求でも通常のfinallyを通す。ハング時は強制終了され得る。
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        return run_worker(args, sys.stdout)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            code = supervise([sys.executable, "-m", "src.control.gripper_smoke_test",
                              *arguments, "--_worker"], stream, 15.0)
    except OSError as exc:
        print(f"Gripper test could not run: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Gripper test {'completed' if code == 0 else 'failed'}: {args.output}")
    if code and args.execute:
        print("If F6 torque OFF is unconfirmed, support the arm and cut power. Do not retry automatically.")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
