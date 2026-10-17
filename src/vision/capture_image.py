"""K20Proから静止画を1枚保存する。既定はMJPG・640×480・30 fps・フォーカス230。"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

from src.control.diagnostic_camera import CameraDiagnosticError, DiagnosticCamera


ROOT = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="/dev/video0", help="カメラのデバイス")
    parser.add_argument("--focus", type=int, default=230, help="手動フォーカス（既定: 230）")
    parser.add_argument("--output-dir", type=Path, help="新規の保存先（省略時は日時付きフォルダー）")
    args = parser.parse_args(argv)
    image_dir = args.output_dir or (
        ROOT / "outputs" / "camera_check" / f"k20pro_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    camera = DiagnosticCamera(
        device=args.device, image_dir=image_dir,
        width=640, height=480, fps=30, focus=args.focus,
        warmup_frames=60, max_read_s=1,
    )
    try:
        image_dir.mkdir(parents=True, exist_ok=False)
        try:
            camera.open()
            image = camera.capture(0)
        finally:
            camera.close()
    except CameraDiagnosticError as exc:
        print(f"撮影失敗: {exc.stage} / {exc.cause_type} / {exc.details}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"保存先エラー: {exc}", file=sys.stderr)
        return 1
    print("保存先:", (image_dir / image["file"]).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
