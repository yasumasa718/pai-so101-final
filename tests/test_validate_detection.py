"""Offline tests for the image-space recognition gate."""

import json

import pytest
from PIL import Image

from src.vision.validate_detection import ROI, main, validate_detections


ROI_TEST = ROI(210, 340, 350, 455)


def detection(xmin=242, ymin=303, xmax=312, ymax=424, score=0.5346342921257019, label="a soda can"):
    return {"label": label, "score": score, "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}


def assess(data):
    return validate_detections(data, image_width=640, image_height=480, roi=ROI_TEST)


def test_known_target_recognition_only():
    result = assess([detection(), detection(400, 100, 480, 210, 0.8)])
    assert result["recognition_gate"] == "PASS"
    assert result["roi_candidates"] == 1
    assert result["candidates"][0]["center"] == {"x": 277.0, "y": 363.5}
    assert result["candidates"][0]["full_box_inside_roi"] is False
    assert result["robot_execution_allowed"] is False


def test_low_confidence_is_rejected():
    assert assess([detection(score=0.49)])["recognition_gate"] == "FAIL"


def test_box_outside_roi_is_rejected():
    assert assess([detection(xmin=100, xmax=160)])["roi_candidates"] == 0


def test_two_candidates_require_review():
    result = assess([detection(), detection(xmin=218, ymin=350, xmax=250, ymax=420, score=0.9)])
    assert result["recognition_gate"] == "REVIEW"
    assert result["robot_execution_allowed"] is False


@pytest.mark.parametrize("bad", [
    detection(score=float("nan")),
    detection(xmin=330, xmax=250),
    detection(xmin=-4),
    {"label": "a soda can", "score": 0.8, "box": {"xmin": 250}},
])
def test_malformed_detection_fails_closed(bad):
    with pytest.raises(ValueError):
        assess([bad])


def test_wrong_label_not_selected():
    assert assess([detection(label="a plastic bottle")])["recognition_gate"] == "FAIL"


def test_invalid_roi_rejected():
    with pytest.raises(ValueError):
        validate_detections([detection()], image_width=640, image_height=480, roi=ROI(300, 340, 200, 455))


def test_cli_does_not_approve_motion(tmp_path, capsys):
    image_path = tmp_path / "input.jpg"
    input_path = tmp_path / "detections.json"
    output_path = tmp_path / "detection_gate.json"
    Image.new("RGB", (640, 480)).save(image_path)
    input_path.write_text(json.dumps([detection()]), encoding="utf-8")
    status = main(["--detections", str(input_path), "--image", str(image_path), "--report-json", str(output_path)])
    assert status == 2
    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["recognition_gate"] == "PASS"
    assert saved["robot_execution_allowed"] is False
    assert "Robot execution allowed: NO" in capsys.readouterr().out


def test_prevent_overwriting_input(tmp_path):
    image_path = tmp_path / "input.jpg"
    input_path = tmp_path / "detections.json"
    Image.new("RGB", (640, 480)).save(image_path)
    input_path.write_text(json.dumps([detection()]), encoding="utf-8")
    assert main(["--detections", str(input_path), "--image", str(image_path), "--report-json", str(input_path)]) == 1
    assert isinstance(json.loads(input_path.read_text(encoding="utf-8")), list)
