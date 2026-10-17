"""画像上の候補判定を検証する。AI推論やロボット接続は行わない。

検出JSONを手作りし、中心位置・スコア・ラベル・候補数を個別に変える。
PASSでも実行許可にならないことと、CLIが入力ファイルを上書きしないことを確認する。
"""

import json

import pytest
from PIL import Image

from src.vision.validate_detection import ROI, main, validate_detections


ROI_TEST = ROI(210, 340, 350, 455)


def detection(xmin=242, ymin=303, xmax=312, ymax=424, score=0.5346342921257019, label="a soda can"):
    """座標・スコア・ラベルを部分的に変更できる検出1件を作る。既定値は中心だけがROI内の缶。"""
    return {"label": label, "score": score, "box": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax}}


def assess(data):
    """640×480画像と共通ROIを使い、検出一覧へ既定閾値0.50の判定を適用する。"""
    return validate_detections(data, image_width=640, image_height=480, roi=ROI_TEST)


def test_known_target_recognition_only():
    """枠全体がROI内でなくても中心が内側の1件をPASSとし、実機実行許可はFalseのままにすることを確認する。"""
    result = assess([detection(), detection(400, 100, 480, 210, 0.8)])
    assert result["recognition_gate"] == "PASS"
    assert result["roi_candidates"] == 1
    assert result["candidates"][0]["center"] == {"x": 277.0, "y": 363.5}
    assert result["candidates"][0]["full_box_inside_roi"] is False
    assert result["robot_execution_allowed"] is False


def test_low_confidence_is_rejected():
    """ROI内にあってもスコア0.49の候補は既定閾値0.50に届かずFAILになることを確認する。"""
    assert assess([detection(score=0.49)])["recognition_gate"] == "FAIL"


def test_box_outside_roi_is_rejected():
    """検出枠の中心がROIの外にある場合、候補に数えないことを確認する。"""
    assert assess([detection(xmin=100, xmax=160)])["roi_candidates"] == 0


def test_two_candidates_require_review():
    """候補2件から最高スコアを自動選択せず、REVIEWとして実行を許可しないことを確認する。"""
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
    """NaN・逆転した枠・画像外の座標・欠けた座標を、候補から除外するだけでなく入力エラーにすることを確認する。"""
    with pytest.raises(ValueError):
        assess([bad])


def test_wrong_label_not_selected():
    """位置とスコアが条件内でも、対象と異なるラベルは候補にしないことを確認する。"""
    assert assess([detection(label="a plastic bottle")])["recognition_gate"] == "FAIL"


def test_invalid_roi_rejected():
    """左右の境界が逆転したROIはValueErrorになり、判定へ進まないことを確認する。"""
    with pytest.raises(ValueError):
        validate_detections([detection()], image_width=640, image_height=480, roi=ROI(300, 340, 200, 455))


def test_cli_does_not_approve_motion(tmp_path, capsys):
    """認識PASSの保存結果でも実行許可はFalse、終了コードは2となるCLIの契約を確認する。"""
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
    """判定JSONの出力先に入力検出JSONを指定すると失敗し、元の検出一覧が残ることを確認する。"""
    image_path = tmp_path / "input.jpg"
    input_path = tmp_path / "detections.json"
    Image.new("RGB", (640, 480)).save(image_path)
    input_path.write_text(json.dumps([detection()]), encoding="utf-8")
    assert main(["--detections", str(input_path), "--image", str(image_path), "--report-json", str(input_path)]) == 1
    assert isinstance(json.loads(input_path.read_text(encoding="utf-8")), list)
