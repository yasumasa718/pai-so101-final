"""実際の待ち行列と制御可能な模擬スレッドで、待ち時間の上限を検証する。機器は使わない。"""

from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import pytest

from src.control.bounded_image_writer import BoundedImageWriter, ImageWorkerError, ImageWorkerTimeout


@dataclass
class ThreadHandle:
    alive: bool = True
    hang_on_join: bool = False
    joins: list[float] = field(default_factory=list)

    def is_alive(self) -> bool:
        return self.alive

    def join(self, timeout: float) -> None:
        self.joins.append(timeout)
        if self.hang_on_join:
            time.sleep(timeout)
        else:
            self.alive = False


@dataclass
class Native:
    queue: queue.Queue = field(default_factory=queue.Queue)
    threads: list[ThreadHandle] = field(default_factory=lambda: [ThreadHandle(), ThreadHandle()])
    num_processes: int = 0
    _stopped: bool = False
    submissions: int = 0

    def save_image(self, *args: Any, **kwargs: Any) -> None:
        self.submissions += 1
        self.queue.put((args, kwargs))

    def stop(self) -> None:
        pytest.fail("Must not invoke the native unbounded stop")


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), threading.TIMEOUT_MAX * 2])
def test_invalid_wait_limit_is_rejected(timeout: Any) -> None:
    with pytest.raises(ValueError):
        BoundedImageWriter(Native(), timeout_s=timeout)


@pytest.mark.parametrize("change", ["processes", "non_queue", "bounded_queue", "no_threads"])
def test_unsupported_native_writer_is_rejected(change: str) -> None:
    native = Native()
    if change == "processes":
        native.num_processes = 1
    elif change == "non_queue":
        native.queue = object()
    elif change == "bounded_queue":
        native.queue = queue.Queue(maxsize=1)
    else:
        native.threads = []
    with pytest.raises(ValueError):
        BoundedImageWriter(native, timeout_s=0.1)


def test_empty_healthy_queue_does_not_wait() -> None:
    worker = BoundedImageWriter(Native(), timeout_s=1.0)
    before = time.monotonic()
    worker.wait_until_done()
    assert time.monotonic() - before < 0.5
    assert worker.failure is None


def test_pending_queue_timeout_latches_and_forbids_submission_and_cleanup() -> None:
    native = Native()
    worker = BoundedImageWriter(native, timeout_s=0.03)
    worker.save_image("mock pixels", "mock path")
    before = time.monotonic()
    with pytest.raises(ImageWorkerTimeout, match="queue drain"):
        worker.wait_until_done()
    assert time.monotonic() - before < 0.5
    # 後から待ち行列が空になっても、記憶した異常を自動で解除してはいけない。
    native.queue.get_nowait()
    native.queue.task_done()
    for operation in (worker.wait_until_done, worker.stop, lambda: worker.save_image("again", "path")):
        with pytest.raises(ImageWorkerError, match="latched"):
            operation()
    assert native.submissions == 0 and native.queue.qsize() == 0 and not native._stopped


@pytest.mark.parametrize("state", ["dead", "stopped"])
def test_dead_or_stopped_worker_rejects_even_empty_queue(state: str) -> None:
    native = Native()
    worker = BoundedImageWriter(native, timeout_s=0.1)
    if state == "dead":
        native.threads[0].alive = False
    else:
        native._stopped = True
    with pytest.raises(ImageWorkerError):
        worker.wait_until_done()
    with pytest.raises(ImageWorkerError):
        worker.save_image("pixels", "path")
    assert native.submissions == 0


@pytest.mark.parametrize("operation", ["wait", "stop", "submit"])
def test_queue_lock_acquisition_is_bounded(operation: str) -> None:
    native = Native()
    worker = BoundedImageWriter(native, timeout_s=0.03)
    native.queue.mutex.acquire()
    before = time.monotonic()
    try:
        with pytest.raises(ImageWorkerTimeout, match="lock"):
            {"wait": worker.wait_until_done, "stop": worker.stop,
             "submit": lambda: worker.save_image("pixels", "path")}[operation]()
    finally:
        native.queue.mutex.release()
    assert time.monotonic() - before < 0.5


def test_shutdown_uses_one_deadline_for_all_threads() -> None:
    native = Native(threads=[ThreadHandle(hang_on_join=True), ThreadHandle(hang_on_join=True)])
    worker = BoundedImageWriter(native, timeout_s=0.03)
    before = time.monotonic()
    with pytest.raises(ImageWorkerTimeout, match="shutdown"):
        worker.stop()
    assert time.monotonic() - before < 0.5
    assert sum(sum(handle.joins) for handle in native.threads) <= 0.04
    assert not native._stopped
    with pytest.raises(ImageWorkerError, match="latched"):
        worker.stop()


def test_normal_shutdown_is_idempotent_and_never_calls_native_stop() -> None:
    native = Native()
    worker = BoundedImageWriter(native, timeout_s=1.0)
    worker.wait_until_done()
    worker.stop()
    assert native._stopped and all(not handle.alive for handle in native.threads)
    assert native.queue.qsize() == len(native.threads)
    worker.stop()
    assert all(len(handle.joins) == 1 for handle in native.threads)


def test_shutdown_refuses_pending_work_without_cancellation() -> None:
    native = Native()
    worker = BoundedImageWriter(native, timeout_s=1.0)
    worker.save_image("pixels", "path")
    with pytest.raises(ImageWorkerError, match="drained"):
        worker.stop()
    assert native.queue.unfinished_tasks == 1 and native.queue.qsize() == 1
    assert all(not handle.joins for handle in native.threads)
