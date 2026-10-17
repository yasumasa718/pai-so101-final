"""保存画像のOWLv2認識と、任意のROIによる対象候補の判定。

テキストで指定した対象を検出し、検出枠付き画像と検出一覧を保存する。
ROI（画像上の矩形領域）を指定した場合はvalidate_detectionsで候補数も判定する。
統合処理recognize_and_replayからも利用されるが、このモジュール自体は実機を動かさない。
"""

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
    """OWLv2推論用のpipelineを作る。CUDAが使えればGPU 0、それ以外はCPUを使う。

    重いライブラリのimportとモデル読込みは呼出し時まで遅らせる。
    初回はモデルの取得が発生し得るため、統合処理では撮影より先に呼ぶ。
    """
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


def detect_image(image_path, output_dir, roi=None, min_score=THRESHOLD, *, detector=None):
    """1枚を認識して結果を保存し、ROI指定時は判定辞書、省略時はNoneを返す。

    image_pathは入力画像、output_dirは保存先のPath。min_scoreは推論と
    ROI候補判定で共通に使うスコア閾値。detectorには読込み済みモデルや
    テスト用の呼出し可能オブジェクトを渡せ、省略するとここでモデルを作る。

    detected.jpgとdetections.jsonを保存し、ROIがあればdetection_gate.jsonも作る。
    同名の出力ファイルは上書きするため、実行ごとに保存先を分けるのは呼出し側の責務。
    画像読込み・推論・検査・保存の例外は呼出し側へ伝える。
    """
    with Image.open(image_path) as source:
        image = source.convert("RGB")
    if roi is not None:
        roi.validate(*image.size)
    output_dir.mkdir(parents=True, exist_ok=True)
    if detector is None:
        detector = load_detector()

    # 画像内を探索する対象はTARGETSのテキスト。ROIで画像を切り抜く処理ではなく、
    # 全画像の検出結果から後段でROI内の候補を数える。
    results = detector(
        image,
        candidate_labels=TARGETS,
        threshold=min_score,
    )

    print("Detection count:", len(results))

    # 描画処理
    draw = ImageDraw.Draw(image)
    detections = []

    for result in results:
        box = result["box"]
        label = result["label"]
        score = float(result["score"])

        # モデルの出力をJSONで扱える通常の数値にそろえる。
        # 後段の候補判定には、この整数化した検出枠を渡す。
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

    report = None
    if roi is not None:
        report = validate_detections(
            detections, image_width=image.width, image_height=image.height,
            roi=roi, min_score=min_score,
        )
        report_path = output_dir / "detection_gate.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                               encoding="utf-8")
        draw.rectangle((roi.xmin, roi.ymin, roi.xmax, roi.ymax), outline="yellow", width=2)
        print("ROI candidates:", report["roi_candidates"])
        print("Recognition gate:", report["recognition_gate"])
        print("Report saved:", report_path.resolve())

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
    return report


def main(argv=None) -> int:
    """保存画像を認識するCLI。正常終了の0は、ロボットの実行許可を意味しない。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=INPUT_PATH, help="入力画像")
    parser.add_argument("--output-dir", type=Path, help="保存先（省略時は日時付きフォルダー）")
    parser.add_argument("--roi", type=ROI.from_csv, help="xmin,ymin,xmax,ymax（画像座標）")
    parser.add_argument("--min-score", type=float, default=THRESHOLD, help="検出・ROI判定の共通閾値（既定: 0.1）")
    args = parser.parse_args(argv)
    if not 0 <= args.min_score <= 1:
        parser.error("--min-score must be between 0 and 1")
    output = args.output_dir or ROOT / "outputs" / f"owlv2_check_{datetime.now():%Y%m%d_%H%M%S_%f}"
    detect_image(args.image, output, args.roi, args.min_score)
    print("Robot execution allowed: NO")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
