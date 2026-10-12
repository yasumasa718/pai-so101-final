
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import pipeline


MODEL_ID = "google/owlv2-base-patch16-ensemble"

TARGETS = [
    "a soda can",
]

THRESHOLD = 0.1

# プロジェクトルート
ROOT = Path(__file__).resolve().parents[2]

INPUT_PATH = ROOT / "outputs/ai_demo/input.jpg"
OUTPUT_DIR = ROOT / "outputs/ai_demo"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(INPUT_PATH)

    # GPU確認
    device = 0 if torch.cuda.is_available() else -1
    print("Device:", "CUDA" if device == 0 else "CPU")

    # OWLv2モデル読み込み
    detector = pipeline(
        task="zero-shot-object-detection",
        model=MODEL_ID,
        device=device,
    )

    # 画像読み込み
    image = Image.open(INPUT_PATH).convert("RGB")

    # AI推論
    results = detector(
        image,
        candidate_labels=TARGETS,
        threshold=THRESHOLD,
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

    # 検出画像の保存
    image.save(OUTPUT_DIR / "detected.jpg")

    # JSONの保存
    with (OUTPUT_DIR / "detections.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            detections,
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("Saved: outputs/ai_demo/detected.jpg")
    print("Saved: outputs/ai_demo/detections.json")


if __name__ == "__main__":
    main()
