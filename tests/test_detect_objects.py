"""モデルを模擬し、CLIから画像・認識結果・ROI判定の保存までを確認する。"""

import json

import pytest
from PIL import Image

from src.vision import detect_objects


TARGET = {"label": "a soda can", "score": 0.104,
          "box": {"xmin": 209, "ymin": 267, "xmax": 288, "ymax": 360}}
BACKGROUND = {"label": "a soda can", "score": 0.146,
              "box": {"xmin": 195, "ymin": 160, "xmax": 233, "ymax": 219}}


@pytest.mark.parametrize("detections,gate,count", [
    ([BACKGROUND, TARGET], "PASS", 1),
    ([BACKGROUND], "FAIL", 0),
    ([TARGET, {**TARGET, "score": 0.2}], "REVIEW", 2),
])
def test_cli_detects_and_saves_roi_decision(tmp_path, monkeypatch, capsys, detections, gate, count):
    image = tmp_path / "input.png"
    Image.new("RGB", (640, 480), (20, 40, 60)).save(image)
    original = image.read_bytes()
    output = tmp_path / "result"

    def detector(frame, *, candidate_labels, threshold):
        assert frame.mode == "RGB" and frame.size == (640, 480)
        assert candidate_labels == ["a soda can"]
        assert threshold == 0.1
        return detections

    monkeypatch.setattr(detect_objects, "load_detector", lambda: detector)
    assert detect_objects.main([
        "--image", str(image), "--output-dir", str(output),
        "--roi", "190,240,320,390", "--min-score", "0.1",
    ]) == 0
    assert json.loads((output / "detections.json").read_text()) == detections
    report = json.loads((output / "detection_gate.json").read_text())
    assert report["recognition_gate"] == gate
    assert report["roi_candidates"] == count
    assert report["robot_execution_allowed"] is False
    assert report["min_score"] == 0.1
    with Image.open(output / "detected.jpg") as annotated:
        assert annotated.size == (640, 480)
    assert image.read_bytes() == original
    assert f"Recognition gate: {gate}" in capsys.readouterr().out


def test_without_roi_saves_detections_to_new_default_folder(tmp_path, monkeypatch):
    image = tmp_path / "input.png"
    Image.new("RGB", (640, 480)).save(image)
    monkeypatch.setattr(detect_objects, "ROOT", tmp_path)
    monkeypatch.setattr(detect_objects, "load_detector", lambda: lambda *args, **kwargs: [])
    assert detect_objects.main(["--image", str(image)]) == 0
    output, = (tmp_path / "outputs").glob("owlv2_check_*")
    assert json.loads((output / "detections.json").read_text()) == []
    assert (output / "detected.jpg").is_file()
    assert not (output / "detection_gate.json").exists()
