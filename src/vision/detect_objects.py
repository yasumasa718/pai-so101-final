import torch
from PIL import Image, ImageDraw
from transformers import pipeline

MODEL_ID = "google/owlv2-base-patch16-ensemble"

# 認識対象
TARGETS = [
    "a plastic bottle",
    "a drinking cup",
]

device = 0 if torch.cuda.is_available() else -1

# AIモデルの読み込み
detector = pipeline(
    task="zero-shot-object-detection",
    model=MODEL_ID,
    device=device,
)

# 撮影した画像を読み込む
image = Image.open(
    "outputs/ai_demo/input.jpg"
).convert("RGB")

# AIによる推論
results = detector(
    image,
    candidate_labels=TARGETS,
    threshold=0.2,
)

# 検出結果を描画
draw = ImageDraw.Draw(image)

for result in results:
    box = result["box"]
    label = result["label"]
    score = result["score"]

    draw.rectangle(
        (
            box["xmin"], box["ymin"],
            box["xmax"], box["ymax"]
        ),
        outline="red",
        width=3,
    )

    draw.text(
        (box["xmin"], box["ymin"]),
        f"{label}: {score:.2f}",
        fill="red",
    )

    print(label, score, box)

# 保存
image.save("outputs/ai_demo/detected.jpg")