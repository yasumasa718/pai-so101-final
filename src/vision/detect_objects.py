"""保存画像のOWLv2認識と、任意のROIによる対象候補の判定。"""

import argparse
import json
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw

from src.vision.validate_detection import ROI, validate_detections


MODEL_ID = "google/owlv2-base-patch16-ensemble"

TARGETS = [
    "a soda can",
]

THRESHOLD = 0.1

# プロジェクトルート
ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "outputs/ai_demo/input.jpg"


def load_detector():
    import torch
    from transformers import pipeline

    # GPU確認
    device = 0 if torch.cuda.is_available() else -1
    print("Device:", "CUDA" if device == 0 else "CPU")

    # OWLv2モデル読み込み
    return pipeline(
        task="zero-shot-object-detection",
        model=MODEL_ID,
        device=device,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=INPUT_PATH, help="入力画像")
    parser.add_argument("--output-dir", type=Path, help="保存先（省略時は日時付きフォルダー）")
    parser.add_argument("--roi", type=ROI.from_csv, help="xmin,ymin,xmax,ymax（画像座標）")
    parser.add_argument("--min-score", type=float, default=THRESHOLD, help="検出・ROI判定の共通閾値（既定: 0.1）")
    args = parser.parse_args(argv)
    if not 0 <= args.min_score <= 1:
        parser.error("--min-score must be between 0 and 1")

    with Image.open(args.image) as source:
        image = source.convert("RGB")
    if args.roi is not None:
        args.roi.validate(*image.size)
    output_dir = args.output_dir or (
        ROOT / "outputs" / f"owlv2_check_{datetime.now():%Y%m%d_%H%M%S_%f}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    detector = load_detector()

    # AI推論
    results = detector(
        image,
        candidate_labels=TARGETS,
        threshold=args.min_score,
    )

    print("Detection count:", len(results))

    # 描画処理
    draw = ImageDraw.Draw(image)
    detections = []

    for result in results:
        box = result["box"]
        label = result["label"]
        score = float(result["score"])

        # JSON保存用のデータ
        detection = {
            "label": label,
            "score": score,
            "box": {
                k: int(v) for k, v in box.items()
            },
        }

        detections.append(detection)

        # バウンディングボックス
        draw.rectangle(
            (
                box["xmin"],
                box["ymin"],
                box["xmax"],
                box["ymax"],
            ),
            outline="red",
            width=3,
        )

        draw.text(
            (box["xmin"], max(0, box["ymin"] - 15)),
            f"{label}: {score:.2f}",
            fill="red",
        )

        print(
            f"Detected: {label}, "
            f"score={score:.3f}, box={box}"
        )

    if not detections:
        print("No objects detected above threshold.")

    if args.roi is not None:
        report = validate_detections(
            detections, image_width=image.width, image_height=image.height,
            roi=args.roi, min_score=args.min_score,
        )
        report_path = output_dir / "detection_gate.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        draw.rectangle((args.roi.xmin, args.roi.ymin, args.roi.xmax, args.roi.ymax), outline="yellow", width=2)
        print("ROI candidates:", report["roi_candidates"])
        print("Recognition gate:", report["recognition_gate"])
        print("Report saved:", report_path.resolve())
    print("Robot execution allowed: NO")

    # 検出画像の保存
    image.save(output_dir / "detected.jpg")

    # JSONの保存
    with (output_dir / "detections.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            detections,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved:", (output_dir / "detected.jpg").resolve())
    print("Saved:", (output_dir / "detections.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
