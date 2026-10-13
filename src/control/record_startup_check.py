"""収録開始前に、保存済みの設定同士が一致するかを読み取りだけで確認する。

expected_sha256 は、事前に別途確認した Follower のキャリブレーションから固定する。
SHA-256 は、ファイルの内容が変わっていないかを比べるための値。
verify() の中で現在のファイルから正解の値を作り直したり、置き換えたりしない。
範囲の内側に取る余白は呼び出し側が指定する値で、実機の安全距離を保証しない。

検査するのはローカルのファイル内容、読み込み済みのキャリブレーション、設定のみ。
モーター内部の値、通信時の単位変換、接続した機器の同一性、衝突、非常停止の準備は確認しない。
機器の検索・接続・切断やトルクの変更は行わず、LeRobot も読み込まない。
開始時だけの検査なので、エピソード中に設定・キャリブレーション・検査条件を変更しないこと。
検査に通っても、実機を動かしてよいという判定にはならない。
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.control.action_guard import ActionGuard, JointBound, MOTOR_IDS

CALIBRATION_FIELDS = ("id", "drive_mode", "homing_offset", "range_min", "range_max")


class StartupCheckError(ValueError):
    """開始前の確認に必要な設定がない、不正、または一致しないことを表す例外。"""


@dataclass(frozen=True)
class StartupCheckResult:
    """開始前の検査に通ったファイルのハッシュ値、余白、関節範囲を残す。"""

    calibration_sha256: str
    margin_deg: float
    margin_gripper: float
    bounds: tuple[JointBound, ...]


@dataclass(frozen=True, kw_only=True)
class RecordStartupCheck:
    """事前に確認したファイルのハッシュ値と余白を使い、収録側の設定を照合する。"""

    calibration_path: Path
    expected_sha256: str
    margin_deg: float
    margin_gripper: float

    def __post_init__(self) -> None:
        """設定の作成直後に、ファイルパスの型と比較用ハッシュ値の形式を確認する。内容の照合は verify() で行う。"""
        if not isinstance(self.calibration_path, Path):
            raise TypeError("calibration_path must be a Path")
        if not isinstance(self.expected_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", self.expected_sha256) is None:
            raise ValueError("expected_sha256 must be a fixed 64-character SHA-256 digest")

    def verify(self, guard: ActionGuard, robot: object) -> StartupCheckResult:
        """作成済みの送信側オブジェクトの設定を調べる。機器とは通信しない。

        robot.config.use_degrees は True でなければならない。
        robot.calibration は JSON 由来の辞書でも、LeRobot の MotorCalibration 形式でもよい。
        ファイルを一度だけ読み、その同じ内容から比較用のハッシュ値と範囲を求める。
        これにより、読み取りの間にファイルが変わって別の版を比較してしまうことを防ぐ。
        """
        try:
            payload = self.calibration_path.read_bytes()
        except OSError as exc:
            raise StartupCheckError("Cannot read Follower calibration") from exc
        digest = hashlib.sha256(payload).hexdigest()
        if digest != self.expected_sha256.lower():
            raise StartupCheckError("Follower calibration SHA-256 mismatch")
        try:
            expected = ActionGuard.from_calibration_bytes(
                payload, margin_deg=self.margin_deg, margin_gripper=self.margin_gripper,
            )
            calibration = json.loads(payload)
        except (ValueError, TypeError) as exc:
            raise StartupCheckError(f"Invalid calibration or margins: {exc}") from exc

        config = getattr(robot, "config", None)
        if getattr(config, "use_degrees", None) is not True:
            raise StartupCheckError("Sender config.use_degrees must be exactly True")
        loaded = getattr(robot, "calibration", None)
        if not isinstance(loaded, Mapping) or set(loaded) != set(MOTOR_IDS):
            raise StartupCheckError("Sender must have six loaded Follower calibration entries")
        for motor in MOTOR_IDS:
            entry = loaded[motor]
            for name in CALIBRATION_FIELDS:
                value = entry.get(name) if isinstance(entry, Mapping) else getattr(entry, name, None)
                if type(value) is not int or value != calibration[motor][name]:
                    raise StartupCheckError(f"Loaded Follower calibration mismatch: {motor}.{name}")

        if not isinstance(guard, ActionGuard):
            raise StartupCheckError("Expected an ActionGuard")
        # 項目の並び順は問わないが、関節の重複、単位の変更、範囲の拡大、
        # 範囲の内側に取る余白の違いは、すべて拒否する。
        actual_bounds = {bound.action_key: bound for bound in guard.bounds}
        expected_bounds = {bound.action_key: bound for bound in expected.bounds}
        if len(guard.bounds) != len(expected.bounds) or actual_bounds != expected_bounds:
            raise StartupCheckError("ActionGuard ranges, units or margins do not match Follower calibration")
        return StartupCheckResult(digest, float(self.margin_deg), float(self.margin_gripper), expected.bounds)
