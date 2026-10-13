"""作成済みの収録部品をまとめ、1 エピソード分の開始・記録・保存・中断を管理する。

エピソードとは、一まとまりの操作を収録したデータのこと。
機器の検索・接続・切断、トルク操作、カメラ撮影、LeRobot CLI の実行は行わない。
現在はオフライン検証用なので、FakeRobot / FakeDataset を使う。
実機での扱いを確認していない SOFollower を渡さないこと。

SafeRecordLoop が各フレームの指令を検査して 1 回だけ送信し、
戻り値の「送信目標指令」を DatasetRecordWriter に渡す。
このクラスは送信前に対応する観測を預け、不完全なフレームを保存しないようにする。
エピソードの保存と中断は、呼び出し側が明示的に選ぶ。

Python の例外や FAULT 状態は、物理的な非常停止ではない。
送信済みの目標に向かってモーターが動き続ける可能性がある。
将来実機を使う前に、アームの支持、非常停止、電源・トルクを切る手順を別途確認すること。

save_episode() は、失敗しても保存前の状態に戻るとは限らない。
一部のファイルが残る可能性があるため、commit_uncertain を保持して中断処理を禁止し、
データセットを使用禁止にして人による確認を待つ。
画像保存や後片付けの異常では cleanup_uncertain も保持し、中断と終了処理を禁止する。
画像保存の時間切れ後も、ファイルへの書き込みが続く可能性があるため。

一つのセッションは 1 エピソード専用。
保存に成功して次を収録する場合は、READY の writer を再利用し、
SafeRecordLoop と SafeRecordSession は新しく作る。
同時に操作したり、各部品をこのクラスとは別に操作したりしないこと。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping

from src.control.dataset_record_writer import DatasetRecordWriter, WriterState
from src.control.record_startup_check import RecordStartupCheck, StartupCheckResult
from src.control.safe_record_loop import (
    RecordFault,
    RecordFaultInfo,
    RecordedFrameResult,
    RecordState,
    SafeRecordLoop,
    SendStatus,
)

logger = logging.getLogger(__name__)


class SessionState(str, Enum):
    READY = "ready"  # 開始前。
    RECORDING = "recording"  # フレームを収録できる状態。
    FAULT = "fault"  # 異常を記憶した状態。収録は再開しない。
    SAVED = "saved"  # エピソードの保存に成功した状態。
    ABORTED = "aborted"  # 未保存のエピソードを明示的に捨てた状態。
    FINALIZED = "finalized"  # データセットの終了処理が完了した状態。


class SessionStateError(RuntimeError):
    """現在の状態では、要求されたエピソード操作を行えないことを表す例外。"""


@dataclass(frozen=True)
class SessionFaultInfo:
    """異常が起きた場所と、送信・保存・後片付けについて判明している状態を残す。"""

    stage: str
    frame_index: int
    send_status: SendStatus
    commit_uncertain: bool
    message: str
    cleanup_uncertain: bool = False


@dataclass
class SafeRecordSession:
    """収録ループと保存処理をつなぎ、異常時には処理を続けさせない。

    利用条件:
        loop.frame_writer と writer は同じオブジェクトにする。lambda などで包まない。
        loop は未開始、writer は未保存フレームも預かり中の観測もない状態にする。
        将来実機を使う場合も、機器そのものの管理は別途行う。

    start() はソフトウェア上の開始であり、実機の安全確認ではない。
    record_frame() には取得済みの観測と指令を渡す。
    save_episode() は loop.finish() の成功後だけ呼び出す。
    abort_episode() は明示的に呼ぶ必要があり、ロボットの動作は止めない。
    """

    loop: SafeRecordLoop
    writer: DatasetRecordWriter
    startup_check: RecordStartupCheck | None = field(default=None, kw_only=True)
    startup_result: StartupCheckResult | None = field(default=None, init=False)
    state: SessionState = field(default=SessionState.READY, init=False)
    fault: SessionFaultInfo | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        """作成直後に、ループと保存先の組み合わせ、未使用の状態、開始前検査の型を確認する。"""
        if not isinstance(self.loop, SafeRecordLoop):
            raise TypeError("loop must be a SafeRecordLoop")
        if not isinstance(self.writer, DatasetRecordWriter):
            raise TypeError("writer must be a DatasetRecordWriter")
        if self.loop.frame_writer is not self.writer:
            raise ValueError("loop.frame_writer must be this exact DatasetRecordWriter")
        if self.loop.state is not RecordState.READY:
            raise ValueError("loop must be READY and unused")
        if self.writer.state is not WriterState.READY or self.writer.frames_in_episode != 0:
            raise ValueError("writer must be READY with an empty episode")
        if self.writer.quarantine_required or self.writer._staged is not None:
            raise ValueError("writer has uncertain commit or pending observation")
        if self.startup_check is not None and not isinstance(self.startup_check, RecordStartupCheck):
            raise TypeError("startup_check must be a RecordStartupCheck")

    def _require(self, *states: SessionState) -> None:
        """現在の状態が、この操作で許された状態かを確認する。違う場合は例外で操作を拒否する。"""
        if self.state not in states:
            allowed = ", ".join(s.value for s in states)
            raise SessionStateError(f"Requires {allowed}, got {self.state.value}")

    def _latch_fault(
        self, stage: str, exc: BaseException,
        *, send_status_hint: SendStatus = SendStatus.NOT_ATTEMPTED,
    ) -> None:
        """ループの外で異常が起きた場合も、ループを FAULT 状態にする。

        観測の準備・エピソード保存・終了処理に失敗した後、
        loop を直接呼んで送信を続けてしまうことを防ぐ。
        """
        if isinstance(exc, RecordFault):
            stage = exc.fault.stage
            send_status = exc.fault.send_status
            frame_index = exc.fault.frame_index
        elif stage == "process_frame" and self.loop.state is RecordState.FAULT and self.loop.fault is not None:
            # ループは中断操作や終了要求の例外を、そのまま呼び出し側へ伝える。
            # その場合も未送信とは決めつけず、ループが保持した送信結果を残す。
            stage = self.loop.fault.stage
            send_status = self.loop.fault.send_status
            frame_index = self.loop.fault.frame_index
        else:
            send_status = send_status_hint
            frame_index = self.loop.frames_written

        self.fault = SessionFaultInfo(
            stage=stage,
            frame_index=frame_index,
            send_status=send_status,
            commit_uncertain=self.writer.commit_uncertain,
            message=f"{type(exc).__name__}: {exc}",
            cleanup_uncertain=self.writer.cleanup_uncertain,
        )
        if self.loop.state is not RecordState.FAULT:
            self.loop.fault = RecordFaultInfo(
                frame_index=frame_index,
                stage=stage,
                send_status=send_status,
                message=str(exc),
            )
            self.loop.state = RecordState.FAULT
        self.state = SessionState.FAULT
        logger.error("Episode FAULT: %s", self.fault)

    def start(self) -> None:
        """開始前の設定を確認し、ソフトウェア上のエピソードを開始する。

        実機への接続や動作は行わない。
        startup_check を省略した場合は従来のオフライン API と同じ動作になり、
        startup_result は None のままになる。
        """
        self._require(SessionState.READY)
        stage = "quarantine_check"
        try:
            self.writer.check_quarantine()
            stage = "startup_check"
            if self.startup_check is not None:
                self.startup_result = self.startup_check.verify(self.loop.guard, self.loop.robot)
            stage = "image_worker_health"
            self.writer.check_workers()
            stage = "start"
            self.loop.start()
            self.state = SessionState.RECORDING
        except BaseException as exc:
            self._latch_fault(stage, exc)
            raise

    def record_frame(
        self,
        raw_action: Mapping[str, object],
        observation: Mapping[str, Any],
        *,
        dataset_observation: Mapping[str, Any] | None = None,
        observation_time: float | None = None,
    ) -> RecordedFrameResult:
        """観測を預け、検査した指令を 1 回送信し、返された指令値を一時保存する。

        observation は LeRobot の指令加工処理に渡し、通常は同じ観測を記録する。
        dataset_observation を別に渡す場合も、必ず同じ取得時点の観測を加工したものにする。
        ここでは観測の取得や指令の自動生成は行わない。

        loop.motion_guard を使う場合、observation_time に関節観測の取得時刻を渡す。
        この時刻には loop.clock と同じ単調時計を使う。
        時刻は検査用として別に渡し、データセットの特徴量の辞書には追加しない。
        """
        self._require(SessionState.RECORDING)
        stage = "stage_observation"
        completed_send = False
        try:
            to_record = observation if dataset_observation is None else dataset_observation
            self.writer.stage_observation(to_record)  # 送信より先に、観測の準備が成功している必要がある。
            stage = "process_frame"
            result = self.loop.process_frame(raw_action, observation, observation_time=observation_time)
            completed_send = True
            stage = "post_frame_consistency"
            if (
                self.writer.state is not WriterState.READY
                or self.writer.frames_in_episode != self.loop.frames_written
            ):
                raise RuntimeError("Frame count/state mismatch after writing; no more sends")
            return result
        except BaseException as exc:
            self._latch_fault(
                stage, exc,
                send_status_hint=(SendStatus.RETURNED if completed_send else SendStatus.NOT_ATTEMPTED),
            )
            raise

    def save_episode(self) -> None:
        """1 フレーム以上あり、異常のないエピソードを終了して保存する。

        保存に失敗すると、ファイルや管理情報が一部だけ書き込まれている可能性がある。
        その場合は保存を再試行しない。
        """
        self._require(SessionState.RECORDING)
        if self.loop.frames_written == 0:
            raise SessionStateError("Cannot save an empty episode")
        try:
            if (
                self.writer.state is not WriterState.READY
                or self.writer.frames_in_episode != self.loop.frames_written
            ):
                raise RuntimeError("Frame count/state mismatch before saving")
            self.loop.finish()
            self.writer.save_episode()
            self.state = SessionState.SAVED
        except BaseException as exc:
            self._latch_fault("save_episode", exc)
            raise

    def abort_episode(self) -> None:
        """未保存の一時データだけを、明示的な指示で捨てる。モーターは停止しない。

        将来実機を使う際は、呼び出し側でロボットの状態に対処してから呼ぶこと。
        save_episode() が途中まで保存した可能性がある場合は、一時データを消してはならない。
        そのデータセットを正常なデータとして扱うこともできない。
        """
        self._require(SessionState.RECORDING, SessionState.FAULT)
        if self.writer.quarantine_required:
            raise SessionStateError("Commit/cleanup is uncertain: quarantine dataset; do not abort/rollback")
        try:
            self.writer.abort_episode()
            if self.loop.state is not RecordState.FAULT:
                self.loop.fault = RecordFaultInfo(
                    frame_index=self.loop.frames_written,
                    stage="manual_abort",
                    send_status=SendStatus.NOT_ATTEMPTED,
                    message="Episode aborted by caller (does not stop motors)",
                )
                self.loop.state = RecordState.FAULT
            self.state = SessionState.ABORTED
        except BaseException as exc:
            self._latch_fault("abort_episode", exc)
            raise

    def finalize(self) -> None:
        """保存成功または明示的な一時データの破棄の後に、データセットを閉じる。

        中断後には、それ以前に保存済みのエピソードは残ることがある。
        中断したエピソード自体は保存しない。機器の切断も行わない。
        """
        self._require(SessionState.SAVED, SessionState.ABORTED)
        try:
            self.writer.finalize()
            self.state = SessionState.FINALIZED
        except BaseException as exc:
            self._latch_fault("finalize", exc)
            raise
