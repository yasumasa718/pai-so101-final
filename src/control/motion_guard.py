"""指令値の変化量と時間を、実機を使わずに検査する。値の補正やモーター操作はしない。

上限値はすべて呼び出し側が指定する。実機で安全と確認済みの数値は用意していない。
ここでいう速度は「目標位置の差 ÷ 経過時間」。実際に測ったモーターの速度ではなく、
モーターがその通りに動くことを保証するものでもない。

観測の取得と収録ループには、同じ単調時計（時刻合わせで戻らない時計）を使う。
観測時刻には、関節の取得時刻または同期読取りの開始・終了区間を渡す。
ホストで計測した区間は、センサー内部の測定時刻や同時性を保証しない。
通常の日時、FPS から作ったデータセット時刻、データが届いた時刻で代用しない。
正しい取得時刻を渡す責任は呼び出し側にある。
指令の履歴は各 SafeRecordLoop が持つ。この検査設定自体は変更せず共有できる。
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from src.control.action_guard import ActionGuard, MOTOR_IDS

JOINT_UNITS = {
    f"{name}.pos": "normalized_0_100" if name == "gripper" else "degree"
    for name in MOTOR_IDS
}


class MotionRejectedError(ValueError):
    """指令値または時刻が、指定した検査条件に違反したことを表す例外。"""


def _number(value: object, name: str) -> float:
    """値を有限なfloatへそろえる。boolや欠損を拒否し、nameで問題箇所を示す。"""
    if type(value) not in (int, float) or not math.isfinite(value):
        raise MotionRejectedError(f"{name}: expected finite number")
    return float(value)


@dataclass(frozen=True)
class JointMotionLimit:
    """1 関節分の上限。max_step は 1 回の変化量、max_speed は 1 秒あたりの変化量。"""

    action_key: str
    unit: str
    max_step: float
    max_speed: float  # 1 秒あたりの指令の変化量。実測した速度ではない。


@dataclass(frozen=True)
class MotionCheck:
    """送信前の検査で使った基準値と時刻。送信後の戻り値も同じ条件で調べるために残す。"""

    reference: tuple[tuple[str, float], ...]
    reference_source: str
    checked_at: float
    observed_at: float
    interval_s: float
    observation_age_s: float
    observation_completed_at: float


@dataclass(frozen=True, kw_only=True)
class MotionGuard:
    """6 関節の変化量と時間を検査する設定。作成後は書き換えない。"""

    limits: tuple[JointMotionLimit, ...]
    min_interval_s: float  # 基準時刻から検査までに必要な最短時間（秒）。
    max_interval_s: float  # 基準時刻から検査までに許す最長時間（秒）。
    max_observation_age_s: float  # 観測を取得してから使うまでに許す最長時間（秒）。
    max_send_duration_s: float  # 送信直前の検査から送信処理が戻るまでに許す最長時間（秒）。

    def __post_init__(self) -> None:
        """作成直後に、6 関節の設定がそろっているか、単位と上限値が適切か、時間条件に矛盾がないかを確認する。"""
        if not isinstance(self.limits, tuple) or any(not isinstance(item, JointMotionLimit) for item in self.limits):
            raise ValueError("limits must be a tuple of JointMotionLimit")
        keys = [item.action_key for item in self.limits]
        if len(keys) != 6 or set(keys) != set(JOINT_UNITS):
            raise ValueError("limits must contain each SO-101 joint exactly once")
        for item in self.limits:
            if item.unit != JOINT_UNITS[item.action_key]:
                raise ValueError(f"{item.action_key}: incompatible motion limit unit")
            for name in ("max_step", "max_speed"):
                if _number(getattr(item, name), f"{item.action_key}.{name}") <= 0:
                    raise ValueError(f"{item.action_key}.{name}: must be positive")
        for name in ("min_interval_s", "max_interval_s", "max_observation_age_s", "max_send_duration_s"):
            if _number(getattr(self, name), name) <= 0:
                raise ValueError(f"{name}: must be positive")
        if self.min_interval_s > min(self.max_interval_s, self.max_observation_age_s):
            raise ValueError("min_interval_s exceeds the interval/observation age limit")

    def validate_units(self, guard: ActionGuard) -> None:
        """位置範囲の検査と変化量の検査で、6 関節の名前と単位が一致しているかを確認する。"""
        if {item.action_key: item.unit for item in guard.bounds} != JOINT_UNITS or len(guard.bounds) != 6:
            raise ValueError("ActionGuard units must match the six motion limits")

    @staticmethod
    def _positions(action: Mapping[str, object], source: str) -> dict[str, float]:
        """6 関節分の位置がそろい、すべて有限の数値であることを確認して、float 型の辞書にそろえる。"""
        if not isinstance(action, Mapping) or set(action) != set(JOINT_UNITS):
            raise MotionRejectedError(f"{source}: expected six joint positions")
        return {key: _number(action[key], f"{source}.{key}") for key in JOINT_UNITS}

    def check(
        self,
        candidate: Mapping[str, object],
        observation: Mapping[str, object],
        *,
        observed_at: object,
        now: object,
        observation_end_time: object | None = None,
        previous_action: Mapping[str, object] | None = None,
        previous_sent_at: float | None = None,
        previous_observed_at: float | None = None,
    ) -> MotionCheck:
        """送信する直前に、最終候補の変化量と時間を検査する。

        初回は観測した関節位置を基準にし、観測の取得から今回の検査までの時間を使う。
        2 回目以降は前回の send_action() が返した指令値を基準にし、
        前回の戻り時刻から今回の検査までの時間を使う。
        時間が空きすぎた場合は拒否し、その分だけ大きな変化を許すことはしない。
        2 回目以降の観測は前回より新しく、前回の送信処理が戻った時刻以降のものが必要。
        区間を渡す場合、鮮度は observed_at（開始）、初回の速度は
        observation_end_time（終了）から評価する。終了省略時は従来の単一時刻。
        """
        checked_at = _number(now, "monotonic clock")
        capture_time = _number(observed_at, "observation_time")
        capture_end = capture_time if observation_end_time is None else _number(
            observation_end_time, "observation_end_time",
        )
        if not capture_time <= capture_end <= checked_at:
            raise MotionRejectedError("Invalid observation acquisition interval")
        age = checked_at - capture_time
        if not 0 <= age <= self.max_observation_age_s:
            raise MotionRejectedError("Observation is stale or its timestamp is in the future")
        measured = self._positions(observation, "observation")
        if previous_action is None:
            if previous_sent_at is not None or previous_observed_at is not None:
                raise MotionRejectedError("Incomplete motion history")
            # 鮮度は最も古い時刻、初回の速度許容量は最も新しい時刻で評価する。
            # 読取りの待機時間を、目標値を変化させてよい時間に含めない。
            reference, reference_time, source = measured, capture_end, "observation"
        else:
            reference = self._positions(previous_action, "previous returned target")
            reference_time = _number(previous_sent_at, "previous send return time")
            last_capture = _number(previous_observed_at, "previous observation_time")
            if capture_time <= last_capture or capture_time < reference_time:
                raise MotionRejectedError("Observation was reused or predates the previous send return")
            source = "previous_sent"
        interval = checked_at - reference_time
        if not self.min_interval_s <= interval <= self.max_interval_s:
            raise MotionRejectedError("Send interval is too short, too long or clock moved backwards")
        result = MotionCheck(tuple(reference.items()), source, checked_at, capture_time, interval, age, capture_end)
        # 送信後に時間を測り直して許容量を増やさないよう、この検査時点の条件を保存する。
        self.validate_target(candidate, result)
        return result

    def validate_target(self, target: Mapping[str, object], check: MotionCheck) -> None:
        """送信後に返された指令値も、送信前と同じ基準値・時間で検査する。

        戻り値を検査する時点では、すでに送信が行われている。
        異常な値の記録と次の送信は防げるが、送信済みの指令は取り消せない。
        """
        positions = self._positions(target, "target")
        reference = dict(check.reference)
        for item in self.limits:
            delta = abs(positions[item.action_key] - reference[item.action_key])
            if delta > item.max_step:
                raise MotionRejectedError(f"{item.action_key}: target step {delta:g} exceeds {item.max_step:g} {item.unit}")
            # 割り算を避け、ゼロ除算や計算結果が大きくなりすぎることを防ぐ。
            # 時間間隔にも上限があるので、古い観測を理由に大きな変化を許すことはない。
            if delta > item.max_speed * check.interval_s:
                raise MotionRejectedError(f"{item.action_key}: inferred target rate exceeds {item.max_speed:g} {item.unit}/s")

    def validate_send_completion(self, check: MotionCheck, returned_at: object) -> float:
        """送信処理が戻った後で、時間がかかりすぎていないかを調べ、戻り時刻を返す。"""
        returned = _number(returned_at, "send return clock")
        if not 0 <= returned - check.checked_at <= self.max_send_duration_s:
            raise MotionRejectedError("Send duration exceeded the limit or clock moved backwards")
        return returned
