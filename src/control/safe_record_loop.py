"""SO-101 の 1 フレーム分の指令を検査・送信・記録する。

収録全体を動かすプログラムではない。
機器の検索・接続・切断、トルク操作、画像取得、データセットの終了処理は行わない。
指令の加工処理、送信先、フレームの保存先は呼び出し側から渡す。
実機で使う前には、非常停止、アームの支持、終了、収録中断の手順を別途確認すること。

記録する action は、最初に要求された指令ではなく robot.send_action() の戻り値。
これは送信した目標値であり、実際に測ったモーターの動きではない。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Protocol

from src.control.action_guard import ActionGuard, MOTOR_IDS
from src.control.motion_guard import MotionCheck, MotionGuard

logger = logging.getLogger(__name__)

# ここでは LeRobot を読み込まない。指令の加工処理は呼び出し可能な関数として渡せるため、
# FakeRobot だけでも同じ呼び出し方を試せる。
Action = dict[str, float]
Observation = dict[str, Any]
ActionProcessor = Callable[[tuple[Action, Observation]], Mapping[str, object]]
FrameWriter = Callable[[Action], None]


class ActionSender(Protocol):
    def send_action(self, action: Action) -> Mapping[str, object]:
        """実際に送信した目標指令を返す。送信側で値が制限される場合もある。"""
        ...


class RecordState(str, Enum):
    """READY→RECORDING→FINISHEDの一方向の状態。処理中の異常はFAULTとして保持する。"""

    READY = "ready"  # まだ開始していない。
    RECORDING = "recording"  # 次のフレームを処理できる。
    FAULT = "fault"  # 異常を記憶しており、このループでは再開できない。
    FINISHED = "finished"  # ループが正常終了した。データセットの保存は別の処理で行う。


class SendStatus(str, Enum):
    """失敗したフレームの送信状況。次の対応で未送信と送信結果不明を区別するために使う。"""

    NOT_ATTEMPTED = "not_attempted"  # 送信処理を呼んでいない。
    UNCERTAIN = "uncertain"  # 例外が発生したが、指令はすでにモーターへ届いた可能性がある。
    RETURNED = "returned"  # 送信処理が戻った状態。実際にモーターが動いた証拠ではない。


@dataclass(frozen=True)
class RecordFaultInfo:
    """どのフレームのどの処理で失敗したかと、送信の状況を残す。"""

    frame_index: int
    stage: str
    send_status: SendStatus
    message: str


class RecordStateError(RuntimeError):
    """現在の状態では要求された操作ができないことを表す例外。"""


class RecordFault(RuntimeError):
    """1 フレームの処理に失敗したことを表す例外。FAULT 後はこのループを再開できない。"""

    def __init__(self, fault: RecordFaultInfo) -> None:
        """フレーム番号・失敗した処理・送信状況を持つ例外を作り、原因を読み取れるメッセージにまとめる。"""
        self.fault = fault
        super().__init__(
            f"frame={fault.frame_index}, stage={fault.stage}, "
            f"send_status={fault.send_status.value}: {fault.message}"
        )


@dataclass(frozen=True)
class RecordedFrameResult:
    """1 フレームの処理結果。送信前の要求値と、送信後に返された指令値を区別して保持する。"""

    frame_index: int
    requested_action: Action
    sent_action: Action


@dataclass
class SafeRecordLoop:
    """1 フレームを検査して 1 回送信し、戻り値の指令を保存する。

    LeRobot の収録全体に組み込む前に、1 フレームの境界で動作を検証するための部品。
    start() はソフトウェアの状態を変えるだけで、実機の安全を確認するものではない。
    一度 FAULT（異常を記憶した状態）になったオブジェクトは再開できない。

    frame_writer は、渡された sent_action をデータセットの action として記録すること。
    対応する観測・時刻・エピソードや、保存失敗時の扱いは呼び出し側が管理する。
    送信後に保存が失敗しても、送信済みの指令は取り消せない。

    motion_guard を指定すると、指令の変化量・変化速度・時間も検査する。
    その場合は、このループと同じ単調時計で測った観測取得時刻 observation_time が必要。
    省略時は従来の API と同じ動作になり、これらの追加検査は行わない。
    送信が遅かったことは処理が戻ってから検出するため、戻ってこない通信は中断できない。
    """

    guard: ActionGuard
    teleop_action_processor: ActionProcessor
    robot_action_processor: ActionProcessor
    robot: ActionSender
    frame_writer: FrameWriter
    motion_guard: MotionGuard | None = field(default=None, kw_only=True)
    clock: Callable[[], float] = field(default=time.monotonic, kw_only=True, repr=False)
    state: RecordState = field(default=RecordState.READY, init=False)
    fault: RecordFaultInfo | None = field(default=None, init=False)
    frames_written: int = field(default=0, init=False)
    _previous_sent_action: Action | None = field(default=None, init=False, repr=False)
    _previous_sent_at: float | None = field(default=None, init=False, repr=False)
    _previous_observed_at: float | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        """作成直後に、6 関節の検査設定と必要な呼び出し先を確認する。変化量の検査を使う場合は単位も照合する。"""
        expected = {f"{motor}.pos" for motor in MOTOR_IDS}
        keys = [item.action_key for item in self.guard.bounds]
        if len(keys) != len(expected) or set(keys) != expected:
            raise ValueError("Guard must contain each of the six SO-101 joints exactly once")
        if not callable(self.teleop_action_processor) or not callable(self.robot_action_processor):
            raise TypeError("Both action processors must be callable")
        if not callable(self.robot.send_action) or not callable(self.frame_writer):
            raise TypeError("Robot sender and frame writer must be callable")
        if self.motion_guard is not None:
            if not isinstance(self.motion_guard, MotionGuard) or not callable(self.clock):
                raise TypeError("motion_guard must be a MotionGuard and clock must be callable")
            self.motion_guard.validate_units(self.guard)

    def start(self) -> None:
        """ソフトウェア上の処理を開始する。接続・キャリブレーション・実機の動作は行わない。"""
        if self.state is not RecordState.READY:
            raise RecordStateError(f"Cannot start from {self.state.value}")
        self.state = RecordState.RECORDING

    def finish(self) -> None:
        """ループを正常終了する。モーターの切断やデータセットの終了処理は行わない。"""
        if self.state is not RecordState.RECORDING:
            raise RecordStateError(f"Cannot finish from {self.state.value}")
        self.state = RecordState.FINISHED

    def _validated_copy(self, action: Mapping[str, object]) -> Action:
        """指令の範囲を検査し、合格した指令のコピーを返す。不正な指令は例外で拒否する。"""
        result = self.guard.check(action)
        if not result.accepted or result.checked_action is None:
            raise ValueError("; ".join(result.reasons))
        return result.checked_action

    def process_frame(
        self,
        raw_action: Mapping[str, object],
        observation: Mapping[str, Any],
        *,
        observation_time: float | None = None,
        observation_end_time: float | None = None,
    ) -> RecordedFrameResult:
        """1 フレームを処理する。異常時は FAULT を保持し、RecordFault を発生させる。

        検査対象の処理に入った後の例外は、すべて FAULT として扱う。
        特に send_action() の中で失敗した場合、目標値がモーターへ届いたかは不明。
        自動で再送信してはいけない。
        このメソッドではトルク解除や切断は行わない。
        observation_time は関節の値を実際に取得した時刻であり、通常の日時や
        データセットが生成する時刻ではない。motion_guard を使う場合は必須。
        読取り区間を使う場合は observation_time に開始、observation_end_time に終了を渡す。
        """
        if self.state is not RecordState.RECORDING:
            raise RecordStateError(f"Cannot process frame from {self.state.value}")

        frame_index = self.frames_written
        stage = "raw_guard"
        send_status = SendStatus.NOT_ATTEMPTED
        try:
            # 入力は、すでに SO-101 の関節名と指令値を組にした辞書になっている。
            # 不正な値を後の加工処理で直す前提にせず、この段階で拒否する。
            raw = self._validated_copy(raw_action)
            stage = "observation_copy"
            obs = dict(observation)
            # 加工処理が観測を書き換える前に、測定した関節位置をコピーして残す。
            # この比較用の値には、カメラ画像などは含めない。
            measured: Action | None = None
            if self.motion_guard is not None:
                stage = "motion_observation_guard"
                measured = self._validated_copy({key: obs[key] for key in raw if key in obs})

            stage = "teleop_processor"
            processed = self.teleop_action_processor((dict(raw), dict(obs)))
            stage = "teleop_guard"
            teleop = self._validated_copy(processed)

            stage = "robot_processor"
            candidate = self.robot_action_processor((dict(teleop), dict(obs)))
            stage = "final_command_guard"
            to_send = self._validated_copy(candidate)

            motion_check: MotionCheck | None = None
            if self.motion_guard is not None:
                stage = "motion_guard"
                assert measured is not None
                motion_check = self.motion_guard.check(
                    to_send, measured, observed_at=observation_time, now=self.clock(),
                    observation_end_time=observation_end_time,
                    previous_action=self._previous_sent_action,
                    previous_sent_at=self._previous_sent_at,
                    previous_observed_at=self._previous_observed_at,
                )

            stage = "send_action"
            # 呼出し中の例外では送信済みか判断できないので、呼ぶ前にUNCERTAINへ切り替える。
            send_status = SendStatus.UNCERTAIN
            sent_response = self.robot.send_action(dict(to_send))
            send_status = SendStatus.RETURNED
            returned_at: float | None = None
            if self.motion_guard is not None:
                stage = "send_timing_guard"
                assert motion_check is not None
                returned_at = self.motion_guard.validate_send_completion(motion_check, self.clock())

            # 戻り値を拒否する時点では、すでに指令を送信している。
            # この検査は異常な記録や次の送信を防ぐためのもので、送信済みの動作は止められない。
            stage = "sent_command_guard"
            sent = self._validated_copy(sent_response)
            if self.motion_guard is not None:
                stage = "sent_motion_guard"
                assert motion_check is not None
                self.motion_guard.validate_target(sent, motion_check)
                # この後の保存が失敗しても、検査を通った送信目標値は履歴に残す。
                # 保存の失敗では、実際に行った送信を取り消せないため。
                self._previous_sent_action = dict(sent)
                self._previous_sent_at = returned_at
                self._previous_observed_at = motion_check.observed_at

            # 記録してよいのは、この検査済みの戻り値だけ。最初の指令や加工途中の値は記録しない。
            # 保存先が指令を後で使う場合は、書き換えの影響を受けないようコピーすること。
            stage = "frame_writer"
            self.frame_writer(dict(sent))

            self.frames_written += 1
            logger.debug("Recorded frame=%d (validated returned command)", frame_index)
            return RecordedFrameResult(frame_index, dict(raw), dict(sent))
        except BaseException as exc:
            # 通常の Exception には含まれない中断操作・終了要求でも FAULT を保持する。
            # 切断やトルク解除をここで行うとアームが落下するおそれがあるので、実行しない。
            # 異常後は処理を続けず、自動で再試行もしない。
            info = RecordFaultInfo(
                frame_index=frame_index,
                stage=stage,
                send_status=send_status,
                message=f"{type(exc).__name__}: {exc}",
            )
            self.fault = info
            self.state = RecordState.FAULT
            logger.error("SO-101 recording FAULT: %s", info, exc_info=True)
            if not isinstance(exc, Exception):
                # 中断や終了の要求は、呼び出し側が扱えるようそのまま伝える。
                raise
            raise RecordFault(info) from exc
