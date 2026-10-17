"""関節の同期読取りをホストの単調時計で挟む。接続・指令・トルク操作はしない。

開始／終了は通信処理の境界であり、センサー内部の測定時刻ではない。
センサーの更新遅延・6 関節の同時性は、この計測では保証できない。
キャッシュを返さない読取り元を使い、カメラ取得はこの処理の後で別に行う。
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from src.control.action_guard import ActionGuard, MOTOR_IDS


class PositionReader(Protocol):
    """TimedJointReaderが必要とする最小の読取り口。実バスと模擬バスの双方で実装する。"""

    def sync_read(self, data_name: str, *, normalize: bool, num_retry: int) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class TimedJointObservation:
    """6軸の位置とホストの読取り区間を保持する。時刻の単位は単調時計の秒。

    positionsは<joint>.posと位置の組をtupleで保持し、加工処理へ渡すときだけ
    as_observation()で新しい辞書にする。位置の単位は読取り時のguardに従う。
    """

    positions: tuple[tuple[str, float], ...]
    read_started_at: float
    read_completed_at: float

    @property
    def read_duration_s(self) -> float:
        """ホストで測った読取り所要秒数。センサー内部の測定周期ではない。"""
        return self.read_completed_at - self.read_started_at

    def as_observation(self) -> dict[str, float]:
        """保持している観測を変更せず、制御処理に渡せる独立した辞書を返す。"""
        return dict(self.positions)


def finite_time(value: object) -> float:
    """有限かつ非負の数値だけを時刻・時間として受け付け、floatへそろえる。"""
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("Expected a finite nonnegative monotonic timestamp")
    return float(value)


class JointReadError(ValueError):
    """読取り側の検査で生成した理由。通信ライブラリの例外本文とは区別する。"""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.details = {"reason": reason}


class TimedJointReader:
    """再試行なしで読み、欠損・範囲違反・時間異常を保持して以後の読取りを拒否する。

    max_read_duration_s は戻り後の検査。ブロック中の I/O は中断しない。
    診断 CLI は別プロセスの監視を追加する。単一の所有者から逐次使用する。
    """

    def __init__(
        self, bus: PositionReader, guard: ActionGuard, *, max_read_duration_s: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if finite_time(max_read_duration_s) <= 0:
            raise ValueError("max_read_duration_s must be positive")
        expected = {f"{name}.pos" for name in MOTOR_IDS}
        if len(guard.bounds) != 6 or {b.action_key for b in guard.bounds} != expected:
            raise ValueError("Expected six joint bounds")
        self.bus = bus
        self.guard = guard
        self.max_read_duration_s = max_read_duration_s
        self.clock = clock
        self.fault: BaseException | None = None
        self._previous: TimedJointObservation | None = None

    def read(self) -> TimedJointObservation:
        """6軸を1回読み、範囲と読取り区間を検査した観測を返す。

        前回より古い時刻、欠損値、範囲外、長すぎる通信は拒否する。
        通信例外や割込みもfaultに保持し、このインスタンスでの再試行を禁止する。
        """
        if self.fault is not None:
            raise RuntimeError("Reader is faulted; no retry") from self.fault
        try:
            start = finite_time(self.clock())
            if self._previous is not None and (
                start <= self._previous.read_started_at or start < self._previous.read_completed_at
            ):
                raise JointReadError("Read clock moved backwards or was reused")
            raw = self.bus.sync_read("Present_Position", normalize=True, num_retry=0)
            end = finite_time(self.clock())
            if not 0 <= end - start <= self.max_read_duration_s:
                raise JointReadError("Joint read duration exceeded limit or clock moved backwards")
            if not isinstance(raw, Mapping) or set(raw) != set(MOTOR_IDS):
                raise JointReadError("Expected exactly six motor observations")
            checked = self.guard.check({f"{key}.pos": value for key, value in raw.items()})
            if not checked.accepted:
                raise JointReadError("Invalid joint observation: " + "; ".join(checked.reasons))
            result = TimedJointObservation(tuple(checked.checked_action.items()), start, end)
            self._previous = result
            return result
        except BaseException as exc:
            self.fault = exc
            raise
