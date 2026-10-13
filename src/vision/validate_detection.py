"""Validate a saved OWLv2 result without connecting to SO-101 hardware.

A recognition PASS is NOT an authorization to execute robot motion.
The ROI is a camera-image heuristic, not a calibrated robot workspace.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from PIL import Image


@dataclass(frozen=True)
class ROI:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @classmethod
    def from_csv(cls, text: str) -> ROI:
        try:
            values = [int(v.strip()) for v in text.split(",")]
        except ValueError as exc:
            raise ValueError("ROI must be four integers: xmin,ymin,xmax,ymax") from exc
        if len(values) != 4:
            raise ValueError("ROI must be four integers: xmin,ymin,xmax,ymax")
        return cls(*values)

    def validate(self, width: int, height: int) -> None:
        if not (0 <= self.xmin < self.xmax <= width):
            raise ValueError(f"ROI x bounds must be within image width {width}")
        if not (0 <= self.ymin < self.ymax <= height):
            raise ValueError(f"ROI y bounds must be within image height {height}")

    def contains_center(self, x: float, y: float) -> bool:
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax

    def contains_box(self, box: dict[str, float]) -> bool:
        return (
            self.xmin <= box["xmin"]
            and box["xmax"] <= self.xmax
            and self.ymin <= box["ymin"]
            and box["ymax"] <= self.ymax
        )


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def validate_detections(
    detections: object,
    *,
    image_width: int,
    image_height: int,
    roi: ROI,
    min_score: float = 0.50,
    target_label: str = "a soda can",
) -> dict[str, Any]:
    """Apply a conservative, offline recognition-only gate to OWLv2 JSON.

    Raises ValueError for malformed input instead of approving by omission.
    A single ROI-center candidate is only a *recognition* PASS.
    """
    if type(image_width) is not int or type(image_height) is not int:
        raise ValueError("Image dimensions must be integers")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    roi.validate(image_width, image_height)
    if not _finite_number(min_score) or not 0 <= min_score <= 1:
        raise ValueError("min_score must be finite and between 0 and 1")
    if not isinstance(target_label, str) or not target_label.strip():
        raise ValueError("target_label must be a non-empty string")
    if not isinstance(detections, list):
        raise ValueError("Detection JSON must contain an array")

    candidates: list[dict[str, Any]] = []
    for index, detection in enumerate(detections):
        if not isinstance(detection, dict):
            raise ValueError(f"Detection {index}: expected an object")
        label, score, box = (
            detection.get("label"),
            detection.get("score"),
            detection.get("box"),
        )
        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Detection {index}: invalid label")
        if not _finite_number(score) or not 0 <= score <= 1:
            raise ValueError(f"Detection {index}: invalid confidence score")
        if not isinstance(box, dict):
            raise ValueError(f"Detection {index}: invalid bounding box")
        keys = ("xmin", "ymin", "xmax", "ymax")
        if any(not _finite_number(box.get(key)) for key in keys):
            raise ValueError(f"Detection {index}: non-finite or missing box coordinate")
        xmin, ymin, xmax, ymax = (float(box[key]) for key in keys)
        if not (0 <= xmin < xmax <= image_width and 0 <= ymin < ymax <= image_height):
            raise ValueError(f"Detection {index}: box is outside image or inverted")

        center_x, center_y = (xmin + xmax) / 2, (ymin + ymax) / 2
        if label == target_label and score >= min_score and roi.contains_center(center_x, center_y):
            candidates.append(
                {
                    "detection_index": index,
                    "label": label,
                    "score": float(score),
                    "box": {key: float(box[key]) for key in keys},
                    "center": {"x": center_x, "y": center_y},
                    "full_box_inside_roi": roi.contains_box(box),
                }
            )

    count = len(candidates)
    if count == 1:
        gate, reason = "PASS", "Exactly one detection matched the image-space criteria"
    elif count == 0:
        gate, reason = "FAIL", "No detection matched the image-space criteria"
    else:
        gate, reason = "REVIEW", "Multiple detections matched; target is ambiguous"

    return {
        "schema_version": "1.0",
        "recognition_gate": gate,
        "reason": reason,
        "detections_total": len(detections),
        "roi_candidates": count,
        "roi": asdict(roi),
        "roi_rule": "bounding_box_center_inside",
        "min_score": float(min_score),
        "target_label": target_label,
        "image_size": {"width": image_width, "height": image_height},
        "candidates": candidates,
        "manual_visual_review_required": True,
        "robot_execution_allowed": False,
        "note": (
            "Recognition only; not a robot workspace, grasp pose, or collision-safety check. "
            "Box center inside ROI does not require full box containment."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline OWLv2 detection gate (no motor control)")
    parser.add_argument("--detections", type=Path, default=Path("outputs/ai_demo/detections.json"))
    parser.add_argument("--image", type=Path, default=Path("outputs/ai_demo/input.jpg"))
    parser.add_argument("--roi", default="210,340,350,455", help="xmin,ymin,xmax,ymax; image pixels")
    parser.add_argument("--min-score", type=float, default=0.50)
    parser.add_argument("--target-label", default="a soda can")
    parser.add_argument("--report-json", type=Path, default=Path("outputs/ai_demo/detection_gate.json"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        with Image.open(args.image) as image:
            width, height = image.size
        data = json.loads(args.detections.read_text(encoding="utf-8"))
        report = validate_detections(
            data,
            image_width=width,
            image_height=height,
            roi=ROI.from_csv(args.roi),
            min_score=args.min_score,
            target_label=args.target_label,
        )
        target = args.report_json.expanduser().resolve()
        if target in {args.detections.expanduser().resolve(), args.image.expanduser().resolve()}:
            raise ValueError("The report path must not overwrite the input image or detections")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        print(f"Detection gate: ERROR ({exc})")
        print("Robot execution allowed: NO")
        return 1

    print("Total detections:", report["detections_total"])
    print("ROI candidates:", report["roi_candidates"])
    for item in report["candidates"]:
        print(
            f"Candidate index={item['detection_index']}, score={item['score']:.3f}, "
            f"center=({item['center']['x']:.1f}, {item['center']['y']:.1f}), "
            f"full_box_inside_roi={item['full_box_inside_roi']}"
        )
    print("Recognition gate:", report["recognition_gate"])
    print("Robot execution allowed: NO")
    print("Report saved:", args.report_json)
    # Exit 0 is intentionally never used to imply clearance for robot motion.
    return 2 if report["recognition_gate"] == "PASS" else 3


if __name__ == "__main__":
    raise SystemExit(main())
