"""Background worker: one thread, one queue, sole writer of LTM files.

Every LTM-touching job (extraction of an archived STM segment, budget
consolidation, final flush extraction) is enqueued here so the user's agent
never waits on memory maintenance — answering speed is unaffected. The worker
thread is the ONLY place ltm/ files are written during normal operation,
which is what keeps the tree consistent without file locks.

Queue overflow: if the queue is saturated the job runs INLINE on the caller
thread (slower) rather than dropping memories silently.

Shutdown: ``close()``/``flush()`` drain deterministically; ``atexit`` covers
users who forget to close (joined with ``atexit_timeout``). A hard SIGKILL can
still lose queued-but-unprocessed extractions — documented in the README; STM
archives survive because they are written synchronously before summarization.
"""

from __future__ import annotations

import atexit
import logging
import queue
import threading
from typing import Any, Callable

SENTINEL = object()
_MAX_QUEUED = 1000


class MemoryWorker:
    """A single daemon thread that processes jobs via ``handler(job)``.

    ``handler`` is supplied by the facade (it knows how to map a job payload
    to LTM calls) and must be thread-safe — it only ever runs on this thread.
    """

    def __init__(self, handler: Callable[[Any], None], logger: logging.Logger,
                 atexit_timeout: float = 30.0) -> None:
        self._handler = handler
        self.logger = logger
        self._queue: queue.Queue = queue.Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="memkit-worker", daemon=True)
        self._thread.start()
        self._atexit_timeout = atexit_timeout
        self._atexit_ref = atexit.register(self._atexit_shutdown)

    # -- submitting --------------------------------------------------------------

    def submit(self, job: Any, wait: bool = False,
               timeout: float | None = None) -> bool:
        """Enqueue a job. If ``wait``, block until it ran; returns whether it
        completed within ``timeout``. Never silently drops work."""
        done = threading.Event() if wait else None
        # NB: qsize() locks the queue's own mutex internally — do NOT wrap it
        # in `with self._queue.mutex` (non-reentrant lock -> self-deadlock).
        # An approximate size is fine for a saturation check.
        saturated = self._queue.qsize() >= _MAX_QUEUED
        if self._closed or saturated:
            if saturated and not self._closed:
                self.logger.warning(
                    "memkit worker: queue saturated (%d jobs); running a job "
                    "inline instead of dropping it.", _MAX_QUEUED)
            # closed/overflow: run inline now, never vanish
            try:
                self._handler(job)
            except Exception:
                self.logger.exception("memkit worker: inline job failed")
            if done:
                done.set()
            return True
        self._queue.put((job, done))
        if done is not None:
            return done.wait(timeout)
        return True

    # -- draining ------------------------------------------------------------------

    def flush(self, timeout: float | None = None) -> bool:
        """Block until every job queued *before this call* is processed.

        Returns True if drained within ``timeout`` (None = wait forever)."""
        if self._closed:
            return True  # close() already drained everything processable
        barrier = threading.Event()
        self._queue.put(("__barrier__", barrier))
        return barrier.wait(timeout) if timeout is not None else bool(barrier.wait())

    def close(self, timeout: float | None = None) -> None:
        """Flush, then stop the thread. Idempotent.

        Without an explicit timeout, drain for ``atexit_timeout`` — the same
        budget the atexit shutdown gets — not a hidden second default."""
        if self._closed:
            return
        drained = self.flush(timeout if timeout is not None else self._atexit_timeout)
        if not drained:
            self.logger.warning(
                "memkit worker: flush timed out; closing anyway — remaining "
                "queued jobs are lost (STM archives are already on disk).")
        self._closed = True
        self._queue.put((SENTINEL, None))
        self._thread.join(timeout=5.0)
        try:
            atexit.unregister(self._atexit_ref)
        except Exception:
            pass

    def _atexit_shutdown(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._queue.put((SENTINEL, None))
        self._thread.join(timeout=self._atexit_timeout)

    # -- the thread -------------------------------------------------------------------

    def _run(self) -> None:
        while True:
            job, done = self._queue.get()
            try:
                if job is SENTINEL:
                    return
                if job == "__barrier__":
                    continue
                self._handler(job)
            except Exception:
                self.logger.exception("memkit worker: job %r failed", job)
            finally:
                if done is not None:
                    done.set()
