"""K20Pro撮影→OWLv2→確認済み固定経路を1回再生。既定は認識だけ。

--execute時もROI内候補が1件のPASS以外はロボットに接続しない。
缶・箱・カメラ・土台の配置を保ち、開始姿勢へ合わせて使用する。

処理順は入力経路の事前検査→モデル読込み→撮影→候補判定→固定経路再生。
画像から関節角度を計算するのではなく、認識結果を記録経路の開始条件に使う。
親プロセスはevents.jsonlへの出力と子プロセスの期限を管理し、
子プロセスが撮影・推論・実機処理を順に担当する。
"""

import argparse
import contextlib
import logging
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from PIL import Image

from src.control import guarded_replay as replay
from src.control.diagnose_observation import emit, supervise
from src.control.diagnostic_camera import CameraDiagnosticError, DiagnosticCamera
from src.vision.detect_objects import MODEL_ID, detect_image, load_detector
from src.vision.validate_detection import ROI


ROOT = Path(__file__).resolve().parents[2]
MAX_IMAGE_AGE_S = 10.0


def capture(args):
    """固定カメラ設定で1枚保存し、画像Pathと設定・読取り時刻のメタデータを返す。

    保存先はoutput_dir/camera。成功・失敗のどちらでもカメラを解放する。
    時刻は再生の開始期限と比較できるようtime.monotonicで記録する。
    """
    image_dir = args.output_dir / "camera"
    image_dir.mkdir()
    camera = DiagnosticCamera(device=args.camera, image_dir=image_dir, width=640, height=480,
                              fps=30, focus=230, warmup_frames=60, max_read_s=1,
                              clock=time.monotonic)
    try:
        settings = camera.open()
        observation = camera.capture(0)
    finally:
        camera.close()
    return image_dir / observation["file"], {**observation, "settings": settings}


def run_worker(args, stream):
    """認識から再生までを実行し、JSONLイベントをstreamへ出力する。

    実行指定なし、または認識がPASS以外なら接続せず0を返す。したがって0だけでは
    動作完了を示さず、collection_completeのreplay_attempted/replay_completedも見る。
    実機処理とトルク解除はreplay.run_workerへ委譲し、異常終了は非0とする。
    """
    stage = "preflight"
    try:
        _, _, plan = replay.prepare(args)
        emit(stream, "collection_plan", execute_requested=args.execute, model=MODEL_ID,
             roi=asdict(args.roi), min_score=args.min_score,
             source_sha256=plan["source_sha256"], checkpoints_sha256=plan["checkpoints_sha256"],
             calibration_sha256=plan["calibration_sha256"], max_image_age_s=MAX_IMAGE_AGE_S)
        # モデル読み込みにかかる時間を撮影後の待機に含めない。
        stage = "model_load"
        with contextlib.redirect_stdout(sys.stderr):
            detector = load_detector()
        stage = "capture"
        if args.image is None:
            image_path, metadata = capture(args)
        else:
            image_path, metadata = args.image, {"source": "saved_image", "read_started_at": None}
        emit(stream, "image", path=str(image_path), **metadata)
        stage = "recognition"
        with Image.open(image_path) as image:
            if image.size != (640, 480):
                raise ValueError("Fixed camera demo requires a 640x480 image")
        with contextlib.redirect_stdout(sys.stderr):
            report = detect_image(image_path, args.output_dir / "recognition", args.roi,
                                  args.min_score, detector=detector)
        gate = report["recognition_gate"]
        emit(stream, "recognition", **report)
        print(f"認識判定：{gate} / ROI内候補 {report['roi_candidates']}件", file=sys.stderr, flush=True)
        if not args.execute or gate != "PASS":
            print("認識のみで終了します。ロボットへの接続・指令・トルク操作は行っていません。",
                  file=sys.stderr, flush=True)
            emit(stream, "collection_complete", recognition_gate=gate,
                 replay_attempted=False, replay_completed=False,
                 reason="recognition_only" if not args.execute else "recognition_"+gate)
            return 0
        stage = "image_age"
        # 推論に費やした時間も期限に含める。これはホストのread開始からの経過時間で、
        # カメラ内部の露光時刻や、撮影後に物が動いていないことを保証するものではない。
        activation_deadline = metadata["read_started_at"]+MAX_IMAGE_AGE_S
        if time.monotonic() >= activation_deadline:
            raise TimeoutError("Recognition image expired before robot connection")
        print("認識PASSです。開始姿勢・校正・トルク状態を照合してから固定再生します。",
              file=sys.stderr, flush=True)
        stage = "replay"
        # 接続・開始姿勢照合にも時間がかかるため、同じ期限を再生側へ渡して再確認する。
        code = replay.run_worker(args, stream, activation_deadline=activation_deadline)
        emit(stream, "collection_complete", recognition_gate=gate,
             replay_attempted=True, replay_completed=code == 0,
             replay_returncode=code)
        return code
    except BaseException as exc:
        details = {}
        if isinstance(exc, CameraDiagnosticError):
            details = {"camera_stage": exc.stage, **exc.details}
        elif type(exc) in (ValueError, TimeoutError):
            details = {"reason": str(exc)}
        print(f"認識・回収停止：{stage} / {type(exc).__name__}", file=sys.stderr, flush=True)
        if "reason" in details:
            print("理由："+details["reason"], file=sys.stderr, flush=True)
        emit(stream, "fault", stage=stage, error_type=type(exc).__name__, **details)
        return 1


def main(argv=None):
    """入力条件を検査し、新規の出力先と監視付き子プロセスを用意するCLI。

    --imageは保存画像による認識確認専用。--executeとの併用は引数検査で拒否する。
    --_workerおよび入力ハッシュの引数は親子間の照合に使う内部用オプション。
    """
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source-log", type=Path, required=True)
    cli.add_argument("--checkpoints", type=Path, required=True)
    cli.add_argument("--calibration", type=Path, required=True)
    cli.add_argument("--expected-sha256", required=True)
    cli.add_argument("--roi", type=ROI.from_csv, required=True)
    cli.add_argument("--min-score", type=float, default=0.1)
    cli.add_argument("--camera", default="/dev/video0")
    cli.add_argument("--image", type=Path, help="保存画像を認識するだけ。--executeとの併用は禁止")
    cli.add_argument("--port", help="--execute時だけ必須のFollower by-idポート")
    cli.add_argument("--output-dir", type=Path, help="新規保存先（既存フォルダーは上書きしない）")
    cli.add_argument("--execute", action="store_true")
    cli.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    for name in ("source", "checkpoints"):
        cli.add_argument("--_"+name+"-sha256", help=argparse.SUPPRESS)
    args = cli.parse_args(argv)
    if not 0 <= args.min_score <= 1:
        cli.error("min-score must be in [0, 1]")
    try:
        args.roi.validate(640, 480)
    except ValueError as exc:
        cli.error(str(exc))
    if args.execute and args.image is not None:
        cli.error("--execute requires a fresh camera capture; saved images cannot trigger motion")
    if args.execute and (not args.port or Path(args.port).parent != Path("/dev/serial/by-id")):
        cli.error("Use --port /dev/serial/by-id/... for the Follower")
    if args._worker:
        # 親からのSIGTERMを例外に変え、再生側のfinallyによる解除を試みられるようにする。
        def interrupted(signum, frame):
            raise KeyboardInterrupt()
        signal.signal(signal.SIGTERM, interrupted)
        logging.disable(logging.CRITICAL)
        return run_worker(args, sys.stdout)
    try:
        _, _, plan = replay.prepare(args)
        output = args.output_dir or ROOT / "outputs" / f"collection_{datetime.now():%Y%m%d_%H%M%S_%f}"
        output.mkdir(parents=True, exist_ok=False)
        command = [sys.executable, "-m", "src.control.recognize_and_replay",
                   "--source-log", str(args.source_log), "--checkpoints", str(args.checkpoints),
                   "--calibration", str(args.calibration), "--expected-sha256", args.expected_sha256,
                   "--roi", ",".join(str(v) for v in asdict(args.roi).values()),
                   "--min-score", str(args.min_score), "--camera", args.camera,
                   "--output-dir", str(output), "--_worker",
                   "--_source-sha256", plan["source_sha256"],
                   "--_checkpoints-sha256", plan["checkpoints_sha256"]]
        if args.execute:
            command.extend(["--execute", "--port", args.port])
        if args.image is not None:
            command.extend(["--image", str(args.image)])
        with (output / "events.jsonl").open("x", encoding="utf-8") as stream:
            # 単一子プロセス内で推論と再生を順に実行し、既存の終了処理を使用する。
            code = supervise(command, stream, 120+replay.runtime_budget(plan)+15, stderr=None)
        print(f"Collection {'completed' if code == 0 else 'failed'}: {output}")
        if code and args.execute:
            print("OFF未確認の軸がある場合は、アームを支持して電源を切ってください。自動再開はしません。")
        return code
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"認識・回収の準備に失敗しました：{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
