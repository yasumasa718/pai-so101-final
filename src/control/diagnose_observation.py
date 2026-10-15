"""SO-101 の関節・画像診断。既定は模擬データ。位置指令・モーター設定書込みはしない。

実機モードは専用の子プロセスでバスを開き、校正レジスタ等を照合する。
通常の SOFollower.connect/configure/disconnect は使わない。
終了・タイムアウトでもモーターのトルク状態は変えない。物理停止機能ではない。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from importlib.metadata import version
from pathlib import Path
from statistics import mean
from typing import TextIO

from src.control.action_guard import ActionGuard, MOTOR_IDS
from src.control.diagnostic_camera import CameraDiagnosticError, DiagnosticCamera
from src.control.record_startup_check import CALIBRATION_FIELDS
from src.control.timed_observation import JointReadError, TimedJointReader, finite_time


def emit(stream: TextIO, event: str, **fields: object) -> None:
    stream.write(json.dumps({"event": event, **fields}, ensure_ascii=False, allow_nan=False) + "\n")
    stream.flush()


class FakeBus:
    """ハードウェアや LeRobot を読み込まない診断用の模擬読取り元。"""

    def sync_read(self, data_name: str, *, normalize: bool, num_retry: int) -> dict[str, float]:
        return {name: 50.0 if name == "gripper" else 0.0 for name in MOTOR_IDS}


class RegisterMismatch(ValueError):
    def __init__(self, motor: str, register: str, expected: object, actual: object):
        super().__init__(f"Motor register mismatch: {motor}.{register}")
        self.details = {"motor": motor, "register": register, "expected": expected, "actual": actual}


def load_calibration(path: Path, expected_sha256: str) -> tuple[dict, ActionGuard]:
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256.lower():
        raise ValueError("Follower calibration SHA-256 mismatch")
    guard = ActionGuard.from_calibration_bytes(payload)
    data = json.loads(payload)
    # LeRobot の read_calibration() は drive_mode を読まず 0 を返す。
    # この診断では未確認の反転設定を受け入れない。
    if any(entry["drive_mode"] != 0 for entry in data.values()):
        raise ValueError("Hardware diagnostic only supports drive_mode=0")
    return data, guard


def make_hardware_bus(port: str, calibration: dict):
    if version("lerobot") != "0.6.1":
        raise RuntimeError("Hardware diagnostic requires audited LeRobot 0.6.1")
    from lerobot.motors import Motor, MotorCalibration, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    return FeetechMotorsBus(
        port=port,
        motors={
            name: Motor(motor_id, "sts3215", MotorNormMode.RANGE_0_100 if name == "gripper" else MotorNormMode.DEGREES)
            for name, motor_id in MOTOR_IDS.items()
        },
        calibration={name: MotorCalibration(**{key: entry[key] for key in CALIBRATION_FIELDS})
                     for name, entry in calibration.items()},
    )


def verify_registers(bus, calibration: dict) -> dict:
    """設定を書かずに照合。失敗時に修復・再校正は行わない。"""
    result = {}
    for name, motor_id in MOTOR_IDS.items():
        registers = {
            register: bus.read(register, name, normalize=False, num_retry=0)
            for register in (
                "ID", "Model_Number", "Homing_Offset", "Min_Position_Limit",
                "Max_Position_Limit", "Phase", "Operating_Mode", "Torque_Enable",
            )
        }
        expected = {
            "ID": motor_id, "Model_Number": bus.model_number_table["sts3215"],
            "Homing_Offset": calibration[name]["homing_offset"],
            "Min_Position_Limit": calibration[name]["range_min"],
            "Max_Position_Limit": calibration[name]["range_max"], "Operating_Mode": 0,
        }
        if any(type(value) is not int for value in registers.values()):
            raise ValueError(f"Non-integer motor register: {name}")
        for key, value in expected.items():
            if registers[key] != value:
                raise RegisterMismatch(name, key, value, registers[key])
        if registers["Phase"] & 0x10:
            raise RegisterMismatch(name, "Phase", "bit 0x10 clear", registers["Phase"])
        if registers["Torque_Enable"] not in (0, 1):
            raise RegisterMismatch(name, "Torque_Enable", [0, 1], registers["Torque_Enable"])
        result[name] = registers
    return result


def image_directory(output: Path) -> Path:
    return output.with_name(output.stem + "_images")


def make_camera(args: argparse.Namespace) -> DiagnosticCamera:
    return DiagnosticCamera(
        device=args.camera, image_dir=image_directory(args.output), width=args.camera_width,
        height=args.camera_height, fps=args.camera_fps, focus=args.camera_focus,
        warmup_frames=args.camera_warmup_frames, max_read_s=args.max_camera_read_s,
        fake=args.backend == "fake",
    )


def collect_samples(
    reader: TimedJointReader, *, samples: int, period_s: float, stream: TextIO,
    camera: DiagnosticCamera | None = None,
) -> None:
    previous_start = None
    measurements = {key: [] for key in ("joint_read_s", "image_read_s", "joint_to_image_gap_s", "start_interval_s")}
    for index in range(samples):
        sample = reader.read()
        interval = None if previous_start is None else sample.read_started_at - previous_start
        extra = {}
        if camera is not None:
            image = camera.capture(index)
            gap = image["read_started_at"] - sample.read_completed_at
            if gap < 0:
                raise ValueError("Image read predates joint read completion")
            extra = {
                "image": image, "joint_to_image_gap_s": gap,
                "joint_start_to_image_end_s": image["read_completed_at"] - sample.read_started_at,
                "sensor_sync_verified": False,
            }
            measurements["joint_read_s"].append(sample.read_duration_s)
            measurements["image_read_s"].append(image["read_duration_s"])
            measurements["joint_to_image_gap_s"].append(gap)
            if interval is not None:
                measurements["start_interval_s"].append(interval)
        emit(
            stream, "observation", index=index, positions=sample.as_observation(),
            read_started_at=sample.read_started_at, read_completed_at=sample.read_completed_at,
            read_duration_s=sample.read_duration_s,
            start_interval_s=interval,
            timestamp_source="host_monotonic_sync_read_interval",
            **extra,
        )
        previous_start = sample.read_started_at
        if index + 1 < samples:
            # 遅れた場合は追いつくための連続読取りをせず、次の周期を測る。
            time.sleep(max(0.0, period_s - (time.monotonic() - sample.read_started_at)))
    if camera is not None:
        emit(stream, "summary", samples=samples, images_saved=samples,
             timing_s={key: {"min": min(values), "mean": mean(values), "max": max(values)}
                       for key, values in measurements.items() if values},
             sensor_sync_verified=False)


def run_worker(args: argparse.Namespace, stream: TextIO) -> int:
    bus = None
    camera = None
    stage = "calibration"
    failed = False
    emit(stream, "start", backend=args.backend, samples=args.samples, period_s=args.period_s,
         max_read_s=args.max_read_s, timeout_s=args.timeout_s, motor_writes=False,
         camera_enabled=args.camera is not None)
    try:
        if args.backend == "hardware":
            calibration, guard = load_calibration(Path(args.calibration), args.expected_sha256)
            stage = "bus_construct"
            bus = make_hardware_bus(args.port, calibration)
            stage = "bus_connect"
            emit(stream, "stage", stage=stage)
            bus.connect(handshake=False)
            # OS のポート設定だけを変更。SDK の read request 自体の書込み待ちを制限する。
            # SDK のパケット期限は別に存在し、子プロセス全体の上限は親で監視する。
            bus.port_handler.ser.exclusive = True
            bus.port_handler.ser.write_timeout = args.max_read_s
            stage = "register_check"
            emit(stream, "stage", stage=stage)
            registers = verify_registers(bus, calibration)
            emit(stream, "register_check", registers=registers, calibration_sha256=args.expected_sha256.lower(),
                 use_degrees=True, num_read_retries=0)
        else:
            payload = json.dumps({name: {
                "id": motor_id, "drive_mode": 0, "homing_offset": 0,
                "range_min": 0, "range_max": 4095,
            } for name, motor_id in MOTOR_IDS.items()}).encode()
            guard = ActionGuard.from_calibration_bytes(payload)
            bus = FakeBus()
        if args.camera is not None:
            stage = "camera_open"
            emit(stream, "stage", stage=stage)
            camera = make_camera(args)
            settings = camera.open()
            emit(stream, "camera_config", image_directory=image_directory(args.output).name, **settings)
        stage = "joint_read"
        emit(stream, "stage", stage=stage)
        reader = TimedJointReader(bus, guard, max_read_duration_s=args.max_read_s)
        collect_samples(reader, samples=args.samples, period_s=args.period_s, stream=stream, camera=camera)
    except BaseException as exc:
        failed = True
        # 通信例外の本文は伏せ、検査側で生成した理由・照合値だけ記録する。
        details = exc.details if isinstance(exc, (RegisterMismatch, JointReadError)) else {}
        if isinstance(exc, CameraDiagnosticError):
            stage = exc.stage
            details = exc.details
        emit(stream, "fault", stage=stage, error_type=type(exc).__name__, **details)
    finally:
        try:
            if camera is not None:
                try:
                    camera.close()
                except BaseException as exc:
                    failed = True
                    emit(stream, "fault", stage="camera_close", error_type=type(exc).__name__)
        finally:
            if args.backend == "hardware" and bus is not None and bus.is_connected:
                try:
                    bus.disconnect(disable_torque=False)
                except BaseException as exc:
                    failed = True
                    emit(stream, "fault", stage="bus_close", error_type=type(exc).__name__)
    if not failed:
        details = {} if camera is None else {"images_saved": args.samples, "sensor_sync_verified": False}
        emit(stream, "complete", samples=args.samples, motion_authorized=False, **details)
    return 1 if failed else 0


def stop_worker(process: subprocess.Popen) -> bool:
    """停止要求は期限付き。停止できなかったプロセスを停止済みと報告しない。"""
    for operation in (process.terminate, process.kill):
        if process.poll() is not None:
            return True
        try:
            operation()
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=1.0)
            return True
        except subprocess.TimeoutExpired:
            continue
    return process.poll() is not None


def supervise(command: list[str], stream: TextIO, timeout_s: float, *, stderr=subprocess.DEVNULL) -> int:
    """子のログを保存し、完了待ち上限・割込み・異常終了を明示する。再起動しない。"""
    process = subprocess.Popen(command, stdout=stream, stderr=stderr, start_new_session=True)
    try:
        code = process.wait(timeout=timeout_s)
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
        stopped = stop_worker(process)
        if stopped:
            # 子の途中出力があっても、親の fault は独立した行にする。
            stream.write("\n")
            emit(stream, "fault", stage="supervisor", error_type=type(exc).__name__, worker_stopped=True)
        else:
            # 生存中の子と同じファイルへ並行書込みしない。
            print("Diagnostic worker stop is unconfirmed; log remains incomplete.", file=sys.stderr)
        return 130 if isinstance(exc, KeyboardInterrupt) else 124
    if code != 0:
        stream.write("\n")
        emit(stream, "fault", stage="worker_exit", returncode=code, worker_stopped=True)
    return 0 if code == 0 else 1


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--backend", choices=("fake", "hardware"), default="fake")
    result.add_argument("--output", type=Path, required=True, help="New JSONL file; never overwritten")
    result.add_argument("--samples", type=int, default=30)
    result.add_argument("--period-s", type=float, default=0.1)
    result.add_argument("--max-read-s", type=float, default=0.25)
    result.add_argument("--timeout-s", type=float, default=15.0, help="Whole worker deadline, including setup/close")
    result.add_argument("--port", help="Explicit /dev/serial/by-id/... Follower path")
    result.add_argument("--calibration", help="Reviewed Follower snapshot")
    result.add_argument("--expected-sha256", help="Previously reviewed fixed SHA-256")
    result.add_argument("--camera", help="Optional /dev/videoN in hardware mode, or fake in fake mode")
    result.add_argument("--camera-focus", type=int, help="Required with --camera; manual focus 0..1023")
    result.add_argument("--camera-width", type=int, default=640)
    result.add_argument("--camera-height", type=int, default=480)
    result.add_argument("--camera-fps", type=float, default=30.0, help="Requested camera FPS; not measured FPS")
    result.add_argument("--camera-warmup-frames", type=int, default=60)
    result.add_argument("--max-camera-read-s", type=float, default=1.0, help="Post-return camera read limit")
    result.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    return result


def main(argv: list[str] | None = None) -> int:
    cli = parser()
    arguments = list(sys.argv[1:] if argv is None else argv)
    args = cli.parse_args(arguments)
    try:
        if args.samples <= 0:
            raise ValueError("samples must be positive")
        for value in (args.period_s, args.max_read_s, args.timeout_s, args.camera_fps, args.max_camera_read_s):
            if finite_time(value) <= 0:
                raise ValueError("Time limits and period must be positive")
        if min(args.camera_width, args.camera_height) <= 0 or args.camera_warmup_frames < 0:
            raise ValueError("Camera dimensions must be positive and warmup frames nonnegative")
        if args.camera is not None:
            if args.camera_focus is None or not 0 <= args.camera_focus <= 1023:
                raise ValueError("Camera requires an explicit manual focus in 0..1023")
            if args.backend == "fake" and args.camera != "fake":
                raise ValueError("Fake mode only accepts --camera fake; never opens a video device")
            if args.backend == "hardware" and re.fullmatch(r"/dev/video[0-9]+", args.camera) is None:
                raise ValueError("Hardware camera must be an explicit /dev/videoN device")
        elif args.camera_focus is not None:
            raise ValueError("Camera focus requires --camera")
        if args.backend == "hardware":
            if not args.port or Path(args.port).parent != Path("/dev/serial/by-id"):
                raise ValueError("Hardware mode requires an explicit /dev/serial/by-id/... port")
            if not args.calibration or not args.expected_sha256:
                raise ValueError("Hardware mode requires calibration and a fixed SHA-256")
            # 不正な設定は子プロセスやシリアルポートを作る前に拒否する。
            load_calibration(Path(args.calibration), args.expected_sha256)
        elif any((args.port, args.calibration, args.expected_sha256)):
            raise ValueError("Hardware options require --backend hardware")
    except (ValueError, OSError) as exc:
        cli.error(str(exc))
    if args._worker:
        return run_worker(args, sys.stdout)
    try:
        with args.output.open("x", encoding="utf-8") as stream:
            if args.camera is not None:
                image_directory(args.output).mkdir()  # 既存ディレクトリやsymlinkを再利用しない。
            code = supervise(
                [sys.executable, "-m", "src.control.diagnose_observation", *arguments, "--_worker"],
                stream, args.timeout_s,
            )
    except OSError as exc:
        print(f"Diagnostic could not run: {type(exc).__name__}", file=sys.stderr)
        return 1
    print(f"Diagnostic {'completed' if code == 0 else 'failed'}: {args.output}")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
