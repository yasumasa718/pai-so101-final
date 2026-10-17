"""SO-101 のエピソードを LeRobotDataset 0.6.1 に保存し、異常時には処理を止める。

SafeRecordLoop.frame_writer に渡す、オフライン検証が可能な部品。
単体で実機収録を行うプログラムではない。
シリアルポートの接続、モーターの始動、トルク解除、画像取得、LeRobot CLI は実行しない。

別途安全性を確認した上で、一つのスレッドから次の順に操作する::

    writer = DatasetRecordWriter(dataset, task="Pick the crushed can")
    # 各フレームで、送信より先に取得済みの観測を預ける。
    writer.stage_observation(obs_processed)
    loop.process_frame(raw_action, obs)  # 1 回送信し、戻り値を writer に渡す。
    # エピソード全体が異常なく終わった場合だけ保存する。
    loop.finish()
    writer.save_episode()
    # すべての収録が終わったら保存処理を閉じる。
    writer.finalize()

異常時には save_episode() を呼ばず、送信も再試行しない。
ロボットの状態に別途対処した後、quarantine_required が False の場合だけ
abort_episode() で未保存の一時データを捨てる。
画像保存や後片付けの異常では、処理が戻ってもファイルの変更が続く可能性があるため、
データを消す後片付けは禁止する。
LeRobot の save_episode() が失敗した場合も、一部だけ保存されている可能性がある。
元に戻せたとは保証できないため、データセットを使用禁止にする。

実機の安全や、決められた時間内での処理完了を保証するものではない。
フレームの組み立ては、LeRobot 0.6.1 の 6 関節の値と画像・動画の観測形式に合わせる。
時刻、フレーム番号、エピソード番号は LeRobot が管理する。
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Protocol

import numpy as np
from PIL import Image

from src.control.action_guard import MOTOR_IDS
from src.control.bounded_image_writer import BoundedImageWriter, ImageWorkerError
from src.control.dataset_quarantine import DatasetQuarantinedError, assert_dataset_not_quarantined, record_dataset_quarantine

logger = logging.getLogger(__name__)
JOINT_KEYS = frozenset(f"{name}.pos" for name in MOTOR_IDS)


class DatasetLike(Protocol):
    """この保存処理が使う LeRobotDataset 0.6.1 の最低限の機能。テスト用の偽物も同じ形にする。"""

    features: dict[str, dict[str, Any]]

    def add_frame(self, frame: dict[str, Any]) -> None:
        """観測と指令を含む 1 フレームを追加するための呼び出し形を定める。実際の処理は保存先が実装する。"""
        ...

    def save_episode(self, episode_data: dict | None = None, parallel_encoding: bool = True) -> None:
        """エピソードを保存するための呼び出し形を定める。ここでは保存処理そのものは実装しない。"""
        ...

    def clear_episode_buffer(self, delete_images: bool = True) -> None:
        """未保存の一時データを捨てるための呼び出し形を定める。delete_images で画像も削除するかを指定する。"""
        ...

    def finalize(self) -> None:
        """データセットの書き込みを終了するための呼び出し形を定める。実際の処理は保存先が実装する。"""
        ...


class WriterState(str, Enum):
    """1フレームごとにREADY→STAGED→READYと進み、異常時にはFAULTを保持する。"""

    READY = "ready"       # 観測を受け付けられる状態。未保存のフレームがある場合もある。
    STAGED = "staged"     # 観測を一つ預かり、送信後の指令値を待っている状態。
    FAULT = "fault"       # 観測の準備や保存で失敗した状態。追加の書き込みは禁止。
    ABORTED = "aborted"   # メモリ内のエピソードを捨てた状態。この writer は再利用しない。
    FINALIZED = "finalized"  # 保存処理を閉じた状態。追加の収録はできない。


class DatasetWriterStateError(RuntimeError):
    """保存処理の現在の状態では、要求された操作ができないことを表す例外。"""


@dataclass
class DatasetRecordWriter:
    """対応する観測と、送信後に返された指令値を組にして記録する。

    各フレームでは、送信前に stage_observation() で観測を預ける。
    __call__() に渡してよいのは SOFollower.send_action() の戻り値だけ。
    要求しただけの指令が渡されても、このクラス単独では区別できないので呼び出し側で守る。
    add_frame() が成功するたびに、預かった観測を一つ使う。
    データセットへの保存は、明示的に save_episode() を呼んだときだけ行う。
    異常時は FAULT を保持し、自動保存しない。
    複数のスレッドや収録ループから同時に使わないこと。
    """

    dataset: DatasetLike
    task: str
    image_timeout_s: float = field(default=5.0, kw_only=True)  # 画像処理を待つ上限秒数。収録全体の制限時間ではない。
    state: WriterState = field(default=WriterState.READY, init=False)
    frames_in_episode: int = field(default=0, init=False)
    saved_episodes: int = field(default=0, init=False)
    commit_uncertain: bool = field(default=False, init=False)  # 保存・終了処理がどこまで進んだか確定できない。
    cleanup_uncertain: bool = field(default=False, init=False)  # 画像処理や後片付けの完了を確認できない。
    quarantine_record: Path | None = field(default=None, init=False)  # データセットの使用禁止を示すファイルの場所。
    quarantine_record_error: str | None = field(default=None, init=False)  # 使用禁止の印を保存できなかった場合の例外名。
    fault_message: str | None = field(default=None, init=False)
    _staged: dict[str, Any] | None = field(default=None, init=False, repr=False)
    _action_names: tuple[str, ...] = field(default=(), init=False, repr=False)
    _state_names: tuple[str, ...] = field(default=(), init=False, repr=False)
    _image_features: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        """作成直後に、使用禁止の印、保存先の機能、記録項目の形式を確認し、画像保存の待ち時間制限を組み込む。"""
        root = getattr(self.dataset, "root", None)
        if root is not None:
            assert_dataset_not_quarantined(Path(root))
        if type(self.image_timeout_s) not in (int, float) or not math.isfinite(self.image_timeout_s) or not 0 < self.image_timeout_s <= threading.TIMEOUT_MAX:
            raise ValueError("image_timeout_s must be finite and positive")
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a non-empty string")
        if not isinstance(getattr(self.dataset, "features", None), dict):
            raise ValueError("dataset.features must be a dict")
        for method in ("add_frame", "save_episode", "clear_episode_buffer", "finalize"):
            if not callable(getattr(self.dataset, method, None)):
                raise TypeError(f"dataset lacks callable {method}()")
        features = self.dataset.features
        native_writer = getattr(self.dataset, "writer", None)
        if native_writer is not None and getattr(native_writer, "_streaming_encoder", None) is not None:
            raise ValueError("Streaming encoding is not supported by the guarded image-file validation")
        self._action_names = self._validate_joint_feature(features, "action")
        # actionとobservation.stateの並びは同じとは限らない。各メタデータの順序を保持する。
        self._state_names = self._validate_joint_feature(features, "observation.state")
        # 対応している SO-101 の項目だけを受け付ける。必要な項目を記録し忘れないよう、
        # 未対応の項目が必要な形式なら、収録前にエラーにする。
        defaults = {"index", "episode_index", "frame_index", "timestamp", "task_index"}
        for key, spec in features.items():
            if key in defaults or key in {"action", "observation.state"}:
                continue
            if (
                not key.startswith("observation.images.")
                or not isinstance(spec, dict)
                or spec.get("dtype") not in ("image", "video")
                or not isinstance(spec.get("shape"), (tuple, list))
                or len(spec["shape"]) != 3
                or any(type(v) is not int or v <= 0 for v in spec["shape"])
            ):
                raise ValueError(f"Unsupported dataset feature: {key}")
            self._image_features[key] = spec
        if native_writer is not None and native_writer.image_writer is not None:
            current = native_writer.image_writer
            if isinstance(current, BoundedImageWriter):
                if current.timeout_s != self.image_timeout_s:
                    raise ValueError("Existing bounded image writer uses a different timeout")
            else:
                native_writer.image_writer = BoundedImageWriter(current, timeout_s=self.image_timeout_s)

    @property
    def quarantine_required(self) -> bool:
        """保存や後片付けの結果が不明で、データセットを使用禁止にする必要があるかを返す。"""
        return self.commit_uncertain or self.cleanup_uncertain

    def check_quarantine(self) -> None:
        """開始・使用前に、ファイルやメモリに使用禁止の印が付いていないかを再確認する。"""
        root = getattr(self.dataset, "root", None)
        if root is not None:
            try:
                assert_dataset_not_quarantined(Path(root))
            except DatasetQuarantinedError as exc:
                self._fault("quarantine_check", exc, uncertain_cleanup=True)
                raise

    def _check_image_worker(self, *, wait: bool = False) -> None:
        """使用禁止の印と画像保存処理の状態を確認する。wait=True なら保存待ちの画像の処理完了も待つ。"""
        self.check_quarantine()
        native = getattr(self.dataset, "writer", None)
        worker = getattr(native, "image_writer", None)
        if worker is not None:
            if not isinstance(worker, BoundedImageWriter):
                raise ImageWorkerError("Image writer was replaced outside the guarded supervisor")
            if wait:
                worker.wait_until_done()
            else:
                worker.check_health()

    def check_workers(self) -> None:
        """セッションを開始する前に、画像保存用の処理が動いているかを確認する。"""
        try:
            self._check_image_worker()
        except BaseException as exc:
            self._fault("image_worker_health", exc)
            raise

    @staticmethod
    def _validate_joint_feature(features: dict, key: str) -> tuple[str, ...]:
        """保存形式が float32 の 6 関節分であることを確認し、データセットが指定する関節の並び順を返す。"""
        spec = features.get(key)
        if not isinstance(spec, dict) or spec.get("dtype") != "float32":
            raise ValueError(f"{key}: expected float32 feature")
        if tuple(spec.get("shape", ())) != (6,):
            raise ValueError(f"{key}: expected shape [6]")
        names = spec.get("names")
        if not isinstance(names, (list, tuple)) or len(names) != 6:
            raise ValueError(f"{key}: expected six joint names")
        if any(not isinstance(n, str) for n in names) or set(names) != JOINT_KEYS:
            raise ValueError(f"{key}: expected exactly six SO-101 <joint>.pos names")
        return tuple(names)

    @staticmethod
    def _joint_vector(action: Mapping[str, object], names: tuple[str, ...], source: str) -> np.ndarray:
        """関節値を指定の順番で float32 の配列にする。数値の不足・不正な値・変換時の桁あふれは拒否する。"""
        if not isinstance(action, Mapping) or set(action) != JOINT_KEYS:
            raise ValueError(f"{source}: expected exactly six joint keys")
        values: list[float] = []
        for key in names:
            raw = action[key]
            if type(raw) not in (float, int) or not math.isfinite(raw):
                raise ValueError(f"{source}.{key}: expected finite number")
            # LeRobot が保存する float32 形式にそろえた値を使う。
            values.append(float(raw))
        array = np.asarray(values, dtype=np.float32)
        if not np.isfinite(array).all():
            raise ValueError(f"{source}: float32 conversion overflow")
        return array

    def _make_observation_frame(self, observation: Mapping[str, Any]) -> dict[str, Any]:
        """観測の項目と画像形式を確認し、保存用の関節配列と画像コピーをまとめる。観測の取得自体は行わない。"""
        if not isinstance(observation, Mapping):
            raise ValueError("observation: expected mapping")
        expected = JOINT_KEYS | {key.removeprefix("observation.images.") for key in self._image_features}
        if set(observation) != expected:
            raise ValueError(f"observation: unexpected/missing keys: {sorted(set(observation) ^ expected)}")
        state = {key: observation[key] for key in self._state_names}
        frame: dict[str, Any] = {
            "observation.state": self._joint_vector(state, self._state_names, "observation.state")
        }
        for key, spec in self._image_features.items():
            camera_name = key.removeprefix("observation.images.")
            value = observation[camera_name]
            if not isinstance(value, np.ndarray) or value.dtype != np.uint8:
                raise ValueError(f"{key}: expected uint8 ndarray")
            if value.shape != tuple(spec["shape"]):
                raise ValueError(f"{key}: expected image shape {spec['shape']}, got {value.shape}")
            frame[key] = value.copy()  # 呼び出し側が後で値を書き換えても、預かった観測が変わらないようコピーする。
        return frame

    def _fault(self, stage: str, error: BaseException, *, uncertain_commit: bool = False, uncertain_cleanup: bool = False) -> None:
        """FAULT と異常の内容を記録し、預かり中の観測を解除する。保存や後片付けの結果が不明なら使用禁止の印も残す。"""
        self.state = WriterState.FAULT
        self.commit_uncertain = self.commit_uncertain or uncertain_commit
        self.cleanup_uncertain = self.cleanup_uncertain or uncertain_cleanup or isinstance(error, ImageWorkerError)
        self.fault_message = f"{stage}: {type(error).__name__}: {error}"
        self._staged = None
        logger.error("SO-101 dataset writer FAULT: %s", self.fault_message)
        if self.quarantine_required:
            try:
                root = getattr(self.dataset, "root", None)
                if root is None:
                    raise ValueError("No local dataset root for quarantine record")
                self.quarantine_record = record_dataset_quarantine(
                    Path(root), stage=stage, error_type=type(error).__name__,
                    commit_uncertain=self.commit_uncertain, cleanup_uncertain=self.cleanup_uncertain,
                    frames_in_episode=self.frames_in_episode, saved_episodes=self.saved_episodes,
                )
            except Exception as record_error:
                # 空き容量不足や書き込み禁止で印を保存できなくても、元の異常を失わないようにする。
                self.quarantine_record_error = type(record_error).__name__
                logger.error("Quarantine record failed: %s; dataset remains excluded", self.quarantine_record_error)

    def stage_observation(self, observation: Mapping[str, Any]) -> None:
        """送信前に、取得済みの観測を一つだけ預かる。

        各フレームの process_frame() の直前に 1 回呼ぶ。
        失敗したら FAULT を保持する。呼び出し側は、その後の指令送信を行ってはならない。
        """
        if self.state is not WriterState.READY:
            raise DatasetWriterStateError(f"Cannot stage from {self.state.value}")
        try:
            self._check_image_worker()
            self._staged = self._make_observation_frame(observation)
            self.state = WriterState.STAGED
        except BaseException as exc:
            self._fault("stage_observation", exc)
            raise

    def __call__(self, sent_action: Mapping[str, object]) -> None:
        """send_action() の戻り値の指令だけを、預かり中の観測と一緒に一時保存する。

        SafeRecordLoop が保存先として呼び出すメソッド。
        LeRobot 0.6.1 が timestamp と frame_index を計算するため、外部で追加しない。
        この段階は一時データへの追加であり、ディスクへの保存完了ではない。
        """
        if self.state is not WriterState.STAGED or self._staged is None:
            raise DatasetWriterStateError(f"Cannot write frame from {self.state.value}")
        try:
            self._check_image_worker()
            action = self._joint_vector(sent_action, self._action_names, "sent_action")
            frame = {
                **self._staged,
                "action": action,
                "task": self.task,
            }
            self.dataset.add_frame(frame)
            self.frames_in_episode += 1
            self._staged = None
            self.state = WriterState.READY
        except BaseException as exc:
            self._fault("add_frame", exc)
            raise

    def _validate_pending_images(self) -> None:
        """画像の書き込みを待ち、LeRobot の保存前に全 RGB 画像を確認する。

        LeRobot 0.6.1 の write_image() は、ディスク書き込みの失敗をログに残すだけで、
        呼び出し側へ例外を伝えない。また、統計計算では画像の一部しか調べず、
        動画変換では存在するファイルだけを使う。そのため、検査しなかった PNG が
        欠けていると、エラーにならず短い動画ができてしまう。ここでは全画像を調べて防ぐ。

        画像保存処理と episode_buffer の扱いは LeRobot 0.6.1 専用。
        メモリだけを使うテスト用 DatasetLike には、これらの処理や画像ファイルはない。
        画像ファイルを一時保存しないストリーミング変換は、作成時に受け付けない。
        この検査は、失敗時に元に戻ることや、実機の動作の安全を保証するものではない。
        """
        if not self._image_features:
            return
        native_writer = getattr(self.dataset, "writer", None)
        if native_writer is None:
            return
        self._check_image_worker(wait=True)
        buffer = getattr(native_writer, "episode_buffer", None)
        if not isinstance(buffer, dict) or buffer.get("size") != self.frames_in_episode:
            raise ValueError("Image episode buffer/frame count mismatch")
        for key, spec in self._image_features.items():
            paths = buffer.get(key)
            if not isinstance(paths, list) or len(paths) != self.frames_in_episode:
                raise ValueError(f"{key}: image buffer/frame count mismatch")
            for index, path in enumerate(paths):
                with Image.open(path) as image:
                    image.load()  # 存在やヘッダーだけでなく、全画素を読み出せるか確認する。
                    if (
                        image.format != "PNG" or image.mode != "RGB"
                        or (image.height, image.width, 3) != tuple(spec["shape"])
                    ):
                        raise ValueError(f"{key}: invalid saved RGB image at frame {index}")

    def save_episode(self) -> None:
        """エピソード全体が成功した後に、明示的に保存する。

        SafeRecordLoop.state が FAULT のときには呼ばないこと。
        実際の LeRobot を使う場合は、先に全 RGB PNG の書き込みと内容を確認する。
        画像内容の検査で失敗しただけなら、保存前なので明示的な中断は可能。
        ただし、画像保存処理の異常などで使用禁止になっていれば中断も禁止する。
        LeRobot の保存を一度でも試みた後の失敗では、ファイルや管理情報が一部残り得る。
        自動で保存前の状態へ戻す処理は行わない。
        """
        if self.state is not WriterState.READY:
            raise DatasetWriterStateError(f"Cannot save from {self.state.value}")
        if self.frames_in_episode <= 0:
            raise DatasetWriterStateError("Cannot save empty episode")
        stage = "validate_pending_images"
        commit_attempted = False
        try:
            self._check_image_worker()
            self._validate_pending_images()
            stage = "save_episode"
            # 呼出し後の例外では部分保存を否定できないため、試行前にフラグを立てる。
            commit_attempted = True
            self.dataset.save_episode()
            self.saved_episodes += 1
            self.frames_in_episode = 0
        except BaseException as exc:
            self._fault(stage, exc, uncertain_commit=commit_attempted)
            raise

    def abort_episode(self) -> None:
        """未保存の一時データを捨てる。異常が起きた writer は再利用しない。

        呼び出し側でロボットの状態に別途対処してから呼ぶこと。
        LeRobot の dataset.save_episode() が失敗した場合は、元に戻せるか不明なので、
        この方法で片付けずデータセットを使用禁止にする。
        画像保存や後片付けが完了したか不明な場合も、保存をまだ試みていなくても
        後片付けの再試行は禁止する。
        """
        if self.state in (WriterState.ABORTED, WriterState.FINALIZED):
            raise DatasetWriterStateError(f"Cannot abort from {self.state.value}")
        if self.quarantine_required:
            raise DatasetWriterStateError("Commit/cleanup may be incomplete: quarantine dataset")
        try:
            self._check_image_worker(wait=True)
            self.dataset.clear_episode_buffer(delete_images=True)
            self.frames_in_episode = 0
            self._staged = None
            self.state = WriterState.ABORTED
        except BaseException as exc:
            self._fault("clear_episode_buffer", exc, uncertain_cleanup=True)
            raise

    def finalize(self) -> None:
        """保存成功または明示的な一時データの破棄の後に、保存処理を閉じる。

        SO-101 の切断や、未完了エピソードの破棄は行わない。
        実機の異常時に安全に終了するための手順は、別途用意する必要がある。
        """
        if self.state not in (WriterState.READY, WriterState.ABORTED):
            raise DatasetWriterStateError(f"Cannot finalize from {self.state.value}")
        if self.frames_in_episode or self._staged is not None or self.quarantine_required:
            raise DatasetWriterStateError("Uncommitted frames/pending observation or uncertain commit")
        try:
            self._check_image_worker(wait=True)
            self.dataset.finalize()
            self.state = WriterState.FINALIZED
        except BaseException as exc:
            self._fault("finalize", exc, uncertain_commit=True)
            raise
