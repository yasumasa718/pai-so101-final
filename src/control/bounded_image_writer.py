"""LeRobot 0.6.1 の画像保存処理を、時間制限を付けて待つ。

ライブラリ本体は変更せず、CPython の queue.Queue とスレッドを使う。
スレッドとは、同じプログラム内で並行して動く処理のこと。
別プロセスで画像を保存する方式には対応していないため、受け付けない。
時間切れや予期しない停止を検出したら異常を記憶し、自動で解除しない。
作業の取り消し、再試行、スレッドの強制終了は行わない。
時間切れの後も、保存中のファイルが変更される可能性がある。

時間制限の対象は、画像の処理待ちとスレッドの終了待ち。
ディスク操作、動画変換、モーター通信、このクラスの外で行われる後片付けは対象外。
収録を統括する側からは、複数のスレッドで同時に呼び出さないこと。
"""

from __future__ import annotations

import math
import queue
import threading
import time
from typing import Any


class ImageWorkerError(RuntimeError):
    """画像保存処理に異常があり、収録・保存・後片付けを続けられないことを表す例外。"""


class ImageWorkerTimeout(ImageWorkerError):
    """画像の処理待ちや終了待ちが、指定した時間内に終わらなかったことを表す例外。"""


class BoundedImageWriter:
    """LeRobot の画像保存処理を包み、待ち時間の上限と異常の記憶を追加する。"""

    def __init__(self, native: Any, *, timeout_s: float) -> None:
        """待ち時間の上限と対応する画像保存方式を確認し、監視対象を登録する。ここでは画像保存を開始しない。"""
        if type(timeout_s) not in (int, float) or not math.isfinite(timeout_s) or not 0 < timeout_s <= threading.TIMEOUT_MAX:
            raise ValueError("image worker timeout must be finite and positive")
        if getattr(native, "num_processes", None) != 0 or type(getattr(native, "queue", None)) is not queue.Queue:
            raise ValueError("Bounded image waiting supports thread-mode queue.Queue only")
        if not getattr(native, "threads", None) or native.queue.maxsize != 0:
            raise ValueError("Expected native image threads and an unbounded submission queue")
        self.native = native
        self.timeout_s = float(timeout_s)
        self.failure: str | None = None

    def _fail(self, reason: str, *, timeout: bool = False) -> None:
        """最初の異常の理由を記憶し、時間切れかそれ以外かに応じた例外を発生させる。"""
        self.failure = self.failure or reason
        error = ImageWorkerTimeout if timeout else ImageWorkerError
        raise error(self.failure)

    def check_health(self) -> None:
        """記憶した異常や、画像保存スレッドの停止があれば例外で知らせる。"""
        if self.failure is not None:
            raise ImageWorkerError(f"Image worker fault is latched: {self.failure}")
        if self.native._stopped:
            self._fail("Image writer has already stopped")
        if any(not thread.is_alive() for thread in self.native.threads):
            self._fail("Image worker thread exited unexpectedly")

    def save_image(self, image: Any, fpath: Any, compress_level: int = 1) -> None:
        """変換済みの画像を保存待ちの列に追加する。列の操作権を待つ時間にも上限を設ける。

        DatasetRecordWriter がコピー済みの NumPy 配列を渡すため、ここでは Tensor を変換しない。
        通常の Queue.put() や AsyncImageWriter.save_image() は、列に空きがあっても
        ロック（同時操作を防ぐ仕組み）の取得を無期限に待つ可能性がある。
        そのため、ここではロックを時間制限付きで取得してから画像を追加する。
        """
        self.check_health()
        condition = self.native.queue.all_tasks_done
        if not condition.acquire(timeout=self.timeout_s):
            self._fail("Image submission queue lock exceeded the timeout", timeout=True)
        try:
            self.check_health()
            self._enqueue_locked((image, fpath, compress_level))
        finally:
            condition.release()

    def _enqueue_locked(self, item: Any) -> None:
        # CPython の待ち行列の内部処理。呼び出し元で共通のロックを取得してから操作する。
        """ロックを取得済みの待ち行列に作業を追加し、処理待ち件数を増やして画像保存スレッドへ知らせる。"""
        self.native.queue._put(item)
        self.native.queue.unfinished_tasks += 1
        self.native.queue.not_empty.notify()

    def wait_until_done(self) -> None:
        """画像保存の完了を時間制限付きで待ち、待っている間も処理の停止を確認する。"""
        self.check_health()
        deadline = time.monotonic() + self.timeout_s
        condition = self.native.queue.all_tasks_done
        if not condition.acquire(timeout=self.timeout_s):
            self._fail("Image queue lock wait exceeded the timeout", timeout=True)
        try:
            while self.native.queue.unfinished_tasks:
                self.check_health()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._fail("Image queue drain exceeded the timeout", timeout=True)
                condition.wait(timeout=min(remaining, 0.05))
            self.check_health()
        finally:
            condition.release()

    def stop(self) -> None:
        """全画像スレッドの終了を、一つの制限時間内で待つ。

        無期限に待つ可能性のある native.stop() は呼び出さない。
        LeRobot の終了処理は、先に保存待ちの画像を処理し終える。
        このメソッドを直接呼ぶ場合も、先に wait_until_done() を呼ぶこと。
        一部のスレッドしか終了しなかった場合は、データセットの使用禁止を続ける。
        """
        if self.failure is not None:
            raise ImageWorkerError(f"Image worker fault is latched: {self.failure}")
        if self.native._stopped:
            return
        self.check_health()
        deadline = time.monotonic() + self.timeout_s
        condition = self.native.queue.all_tasks_done
        if not condition.acquire(timeout=self.timeout_s):
            self._fail("Image shutdown queue lock exceeded the timeout", timeout=True)
        try:
            if self.native.queue.unfinished_tasks:
                self._fail("Image queue must be drained before shutdown")
            for _ in self.native.threads:
                self._enqueue_locked(None)
        finally:
            condition.release()
        for thread in self.native.threads:
            thread.join(timeout=max(0.0, deadline - time.monotonic()))
        if any(thread.is_alive() for thread in self.native.threads):
            self._fail("Image worker shutdown exceeded the timeout", timeout=True)
        self.native._stopped = True
