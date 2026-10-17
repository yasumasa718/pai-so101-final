"""LeRobot 0.6.1 の手動操作の指令を検査する、オフライン検証用の処理。

実機収録に組み込む前に、指令を拒否したときの対応、終了、トルク解除、
ログの扱いを確認し、安全な条件で試験すること。
この処理自体は、ロボット・シリアルポート・カメラにアクセスしない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lerobot.lerobot_types import RobotAction, RobotObservation
from lerobot.processor import (
    RobotActionProcessorStep,
    RobotProcessorPipeline,
    robot_action_observation_to_transition,
    transition_to_robot_action,
)

from src.control.action_guard import ActionGuard


class GuardRejectedError(RuntimeError):
    """位置の指令が、設定した範囲の検査に通らなかったことを表す例外。"""


@dataclass
class GuardedTeleopActionStep(RobotActionProcessorStep):
    """SO-101 の全関節の手動操作指令を検査する。値の切り詰めや補正はしない。"""

    guard: ActionGuard

    def action(self, action: RobotAction) -> RobotAction:
        """手動操作の指令を範囲検査し、通った指令を返す。違反があれば補正せず例外で知らせる。"""
        checked = self.guard.check(action)
        if not checked.accepted or checked.checked_action is None:
            raise GuardRejectedError("; ".join(checked.reasons))
        return checked.checked_action

    def transform_features(self, features: dict[Any, Any]) -> dict[Any, Any]:
        """この検査では、特徴量の型・形・名前を変更しない。"""
        return features


def make_guarded_teleop_pipeline(
    calibration_path: Path,
    *,
    margin_deg: float = 0.0,
    margin_gripper: float = 0.0,
) -> RobotProcessorPipeline[
    tuple[RobotAction, RobotObservation], RobotAction
]:
    """(指令, 観測)を受け、範囲検査した指令を返すLeRobotのパイプラインを作る。

    margin_degは腕の度、margin_gripperは爪の0〜100のポイントで、校正範囲の
    両端から内側へ取る余白。変換関数はLeRobotの内部形式との橋渡しを行う。
    この処理自体は時間・変化量の検査や送信を行わない。オフラインテスト用。
    """
    guard = ActionGuard.from_calibration(
        calibration_path,
        margin_deg=margin_deg,
        margin_gripper=margin_gripper,
    )
    return RobotProcessorPipeline[
        tuple[RobotAction, RobotObservation], RobotAction
    ](
        steps=[GuardedTeleopActionStep(guard=guard)],
        name="so101_guarded_teleop_offline",
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
