"""診断用の画像取得とPNG保存。ロボットやLeRobotDatasetは操作しない。

時刻はOpenCVのread()を挟んだホスト単調時計。露光時刻ではなく、
ドライバーのバッファに残った画像の鮮度や関節との同期を保証しない。
ブロック中の処理の停止は、診断CLIの親プロセスが担当する。
"""

from __future__ import annotations

import math
import time
from pathlib import Path

from src.control.timed_observation import finite_time


class CameraSettingMismatch(ValueError):
    """要求した撮影設定とドライバーの報告値の不一致。JSON用の比較情報を保持する。"""

    def __init__(self, setting: str, expected: object, reported: object):
        super().__init__(f"Camera setting mismatch: {setting}")
        # ドライバーがNaN等を返しても、元のFAULTをJSON変換エラーで隠さない。
        if isinstance(reported, (int, float)) and not math.isfinite(reported):
            reported = None
        self.details = {"setting": setting, "expected": expected, "reported": reported}


class CameraDiagnosticError(RuntimeError):
    """失敗段階と原因の型をまとめる例外。カメラ設定の不一致なら比較値も引き継ぐ。"""

    def __init__(self, stage: str, cause: BaseException):
        super().__init__(f"Camera diagnostic failed at {stage}")
        self.stage = stage
        self.cause_type = type(cause).__name__
        self.details = {"cause_type": self.cause_type}
        if isinstance(cause, CameraSettingMismatch):
            self.details.update(cause.details)


class DiagnosticCamera:
    """設定照合・時刻付き読取り・PNG保存をまとめる、単一所有者用のカメラ。

    呼出し側でimage_dirを作り、open()→capture(index)→close()の順に使う。
    close()や異常の後は同じインスタンスを再使用しない。fake=Trueではデバイスへ
    接続せず、画素値が順に変わる模擬画像を生成し、実際のPNG保存まで検証できる。
    """

    def __init__(
        self, *, device: str, image_dir: Path, width: int, height: int, fps: float,
        focus: int, warmup_frames: int, max_read_s: float, fake: bool = False,
        clock=time.monotonic,
    ) -> None:
        self.device = device
        self.image_dir = image_dir
        self.width, self.height, self.fps = width, height, fps
        self.focus = focus
        self.warmup_frames = warmup_frames
        self.max_read_s = max_read_s
        self.fake = fake
        self.clock = clock
        self.cap = None
        self.stage = "camera_open"
        self.faulted = False
        self._ready = False
        self._previous_end = None
        self._fake_index = 0

    def _read(self):
        """BGR画像とホストでの読取り開始・終了時刻を返し、寸法・型・所要時間を検査する。

        max_read_sの検査はread()が戻った後に行う。停止しないI/Oを中断する機能ではない。
        """
        import numpy as np

        self.stage = "camera_read"
        started = finite_time(self.clock())
        if self._previous_end is not None and started < self._previous_end:
            raise ValueError("Camera clock moved backwards")
        if self.fake:
            frame = np.full((self.height, self.width, 3), self._fake_index % 256, dtype=np.uint8)
            self._fake_index += 1
            ok = True
        else:
            ok, frame = self.cap.read()
        completed = finite_time(self.clock())
        if not 0 <= completed - started <= self.max_read_s:
            raise ValueError("Camera read duration exceeded limit or clock moved backwards")
        if not ok or not isinstance(frame, np.ndarray) or frame.dtype != np.uint8:
            raise ValueError("Camera did not return a uint8 image")
        if frame.shape != (self.height, self.width, 3):
            raise ValueError("Camera image dimensions do not match requested dimensions")
        self._previous_end = completed
        return frame, started, completed

    def _settings(self) -> dict:
        """要求値と読戻し値を比較し、記録用の設定辞書を返す。不一致は例外で知らせる。"""
        import cv2

        requested = {"width": self.width, "height": self.height, "fps": self.fps,
                     "focus_automatic_continuous": 0, "focus_absolute": self.focus, "fourcc": "MJPG"}
        if self.fake:
            return {"requested": requested, "reported": dict(requested), "buffer_size_1_accepted": None}
        reported = {
            "width": self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
            "height": self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
            "fps": self.cap.get(cv2.CAP_PROP_FPS),
            "focus_automatic_continuous": self.cap.get(cv2.CAP_PROP_AUTOFOCUS),
            "focus_absolute": self.cap.get(cv2.CAP_PROP_FOCUS),
        }
        fourcc = self.cap.get(cv2.CAP_PROP_FOURCC)
        if fourcc != cv2.VideoWriter_fourcc(*"MJPG"):
            raise CameraSettingMismatch("fourcc", "MJPG", fourcc)
        reported["fourcc"] = "MJPG"
        for key, value in requested.items():
            if reported[key] != value:
                raise CameraSettingMismatch(key, value, reported[key])
        return {"requested": requested, "reported": reported,
                "buffer_size_1_accepted": self.buffer_size_accepted}

    def open(self) -> dict:
        """カメラを設定し、起動直後の画像を読み捨てて設定を照合し、設定情報を返す。

        MJPG・画像サイズ・fps・手動フォーカスを設定する。warmup_framesは起動直後の
        画像を使わないための読取り回数で、露光や鮮度の保証ではない。
        途中で失敗した場合も、呼出し側のfinallyでclose()すること。
        """
        import cv2

        if self.faulted or self._ready or self.cap is not None:
            raise RuntimeError("Camera cannot be reopened")
        try:
            self.stage = "camera_open"
            if not self.image_dir.is_dir() or self.image_dir.is_symlink():
                raise ValueError("Expected a new diagnostic image directory")
            if not self.fake:
                self.cap = cv2.VideoCapture()
                if not self.cap.open(self.device, cv2.CAP_V4L2):
                    raise RuntimeError("Cannot open camera")
                self.stage = "camera_settings"
                for name, prop, value in (
                    ("fourcc", cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG")),
                    ("width", cv2.CAP_PROP_FRAME_WIDTH, self.width),
                    ("height", cv2.CAP_PROP_FRAME_HEIGHT, self.height),
                    ("fps", cv2.CAP_PROP_FPS, self.fps),
                    ("focus_automatic_continuous", cv2.CAP_PROP_AUTOFOCUS, 0),
                    ("focus_absolute", cv2.CAP_PROP_FOCUS, self.focus),
                ):
                    if not self.cap.set(prop, value):
                        raise CameraSettingMismatch(name, value, None)
                # バッファを小さくする要求は最善努力。成功しても最新露光の保証ではない。
                self.buffer_size_accepted = bool(self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1))
            for _ in range(self.warmup_frames):
                self._read()
            self.stage = "camera_settings"
            settings = self._settings()
            self._ready = True
            return {**settings, "warmup_frames": self.warmup_frames, "fake": self.fake,
                    "timestamp_source": "host_monotonic_opencv_read_interval",
                    "sensor_timestamp_available": False, "sensor_sync_verified": False}
        except BaseException as exc:
            self.faulted = True
            raise CameraDiagnosticError(self.stage, exc) from exc

    def capture(self, index: int) -> dict:
        """1枚をframe_XXXXXX.pngとして新規保存し、ファイル名・寸法・読取り時刻を返す。

        indexは呼出し側が与える連番。同名ファイルは上書きせず、保存失敗もFAULTにする。
        時刻は画像取得だけを挟み、PNGへの圧縮・ディスク書込みの時間は含まない。
        """
        import cv2

        if self.faulted or not self._ready:
            raise RuntimeError("Camera is not ready or faulted; no retry")
        try:
            frame, started, completed = self._read()
            path = self.image_dir / f"frame_{index:06d}.png"
            self.stage = "image_encode"
            ok, encoded = cv2.imencode(".png", frame)
            if not ok:
                raise RuntimeError("Cannot encode PNG")
            payload = encoded.tobytes()
            self.stage = "image_write"
            with path.open("xb") as stream:
                stream.write(payload)
            return {
                "file": path.name,
                "width": self.width, "height": self.height, "opencv_array_color_order": "BGR",
                "read_started_at": started, "read_completed_at": completed,
                "read_duration_s": completed - started,
                "timestamp_source": "host_monotonic_opencv_read_interval",
                "sensor_timestamp_available": False,
            }
        except BaseException as exc:
            self.faulted = True
            raise CameraDiagnosticError(self.stage, exc) from exc

    def close(self) -> None:
        """カメラを解放し、以後の取得・再オープンを禁止する。保存済みの画像は残す。"""
        self._ready = False
        self.faulted = True
        if self.cap is not None:
            self.cap.release()
