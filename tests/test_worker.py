import threading
import time

from memkit.worker import MemoryWorker


def make_worker():
    log = []
    jobs = []
    lock = threading.Lock()

    def handler(job):
        with lock:
            jobs.append(job)

    w = MemoryWorker(handler, logger=_NullLogger(), atexit_timeout=1.0)
    return w, jobs, lock


class _NullLogger:
    def warning(self, *a): pass
    def error(self, *a): pass
    def exception(self, *a): pass


def test_jobs_run_off_the_caller_thread():
    w, jobs, lock = make_worker()
    ran_here = threading.current_thread()
    for i in range(5):
        w.submit(("extract", i))
    assert w.flush(timeout=5)
    assert [j[1] for j in jobs] == [0, 1, 2, 3, 4]
    assert all(j[1] != ran_here for j in jobs)  # ran on worker thread
    w.close()


def test_submit_wait_blocks_until_done():
    w, jobs, lock = make_worker()
    done = w.submit(("x", 1), wait=True, timeout=5)
    assert done
    assert ("x", 1) in jobs
    w.close()


def test_flush_barrier_only_waits_for_queued_prefix():
    gate = threading.Event()
    release = threading.Event()

    def handler(job):
        if job[0] == "slow":
            release.wait(5)      # actually block the worker thread
        else:
            gate.set()

    w = MemoryWorker(handler, logger=_NullLogger(), atexit_timeout=1.0)
    w.submit(("slow", 1))
    assert w.flush(timeout=0.2) is False   # must time out: slow job pending
    release.set()
    assert w.flush(timeout=5)
    w.close()


def test_close_is_idempotent_and_drains():
    w, jobs, lock = make_worker()
    w.submit(("a", 1))
    w.close()
    w.close()
    assert ("a", 1) in jobs


def test_late_submit_after_close_runs_inline():
    w, jobs, lock = make_worker()
    w.close()
    w.submit(("late", 9))             # must NOT vanish
    assert ("late", 9) in jobs


def test_flush_barrier_is_released_after_timeout():
    # If a flush times out, the barrier event must STILL be set once the
    # worker reaches it — a later flush()/close() has to drain cleanly and a
    # stale barrier must not leave anybody waiting forever.
    release = threading.Event()

    def handler(job):
        if job == ("slow",):
            release.wait(5)

    w = MemoryWorker(handler, logger=_NullLogger(), atexit_timeout=5.0)
    w.submit(("slow",))
    assert w.flush(timeout=0.2) is False      # times out as expected
    release.set()
    assert w.flush(timeout=5)                 # second flush drains cleanly
    w.close()


def test_close_uses_atexit_timeout_default():
    # close() with no explicit timeout must give the drain the configured
    # atexit_timeout, not a hidden hardcoded 30s default. A job that takes
    # ~1s with atexit_timeout=0.3 must surface "flush timed out" — with the
    # old hardcoded default the drain would have silently waited it out.
    class _RecLogger(_NullLogger):
        def __init__(self):
            self.msgs = []
        def warning(self, fmt, *a):
            self.msgs.append(fmt % a if a else fmt)

    release = threading.Event()

    def handler(job):
        release.wait(1.0)

    rec = _RecLogger()
    w = MemoryWorker(handler, logger=rec, atexit_timeout=0.3)
    w.submit(("slow",))
    w.close()                                  # no timeout arg
    assert any("flush timed out" in m for m in rec.msgs), (
        f"close did not drain with atexit_timeout: {rec.msgs}")
    release.set()


def test_inline_job_does_not_overlap_the_worker_thread():
    """The bug: an inline job (queue saturated, or closed) ran the handler
    directly on the caller thread with no lock, so it could execute
    CONCURRENTLY with the job in flight on the worker thread — two writers
    interleaving on the same ltm/ files. Both paths must take the handler
    lock, so an inline run waits for the in-flight job to finish."""
    inside = threading.Event()      # worker is inside the handler
    release = threading.Event()     # ...and stays there until we let it go
    overlap = []

    def handler(job):
        if job == ("slow",):
            inside.set()
            release.wait(5)
            return
        if inside.is_set() and not release.is_set():
            overlap.append(job)     # ran while ("slow",) was still executing

    w = MemoryWorker(handler, logger=_NullLogger(), atexit_timeout=5.0)
    w.submit(("slow",))
    assert inside.wait(5), "the worker never entered the slow job"

    # Saturate the queue so this submit takes the inline path while the worker
    # is still inside the slow job.
    for i in range(1001):
        w._queue.put((("filler", i), None))

    ran = threading.Thread(target=lambda: w.submit(("inline", 9)))
    ran.start()
    time.sleep(0.2)                 # plenty of time to run, if it were unlocked
    assert ran.is_alive(), "inline job did not wait for the in-flight job"
    assert overlap == [], "inline job ran concurrently with the worker's job"

    # Take the filler jobs back out: the worker is blocked inside the slow job
    # (never in get()), so nothing else can consume them. Without this the
    # worker would chew through 1001 fillers after release and close() would
    # join a busy thread forever.
    for _ in range(1001):
        w._queue.get_nowait()
    release.set()
    ran.join(5)
    w.close(timeout=5.0)


def test_atexit_shutdown_drains_queued_jobs_before_stopping():
    """The bug: atexit put the SENTINEL first, so the worker thread exited the
    moment it saw it and every queued extraction was silently discarded — at
    exactly the moment (the user forgot close()) they were the only thing left
    to save. atexit must flush first, then stop."""
    release = threading.Event()
    jobs = []
    lock = threading.Lock()

    def handler(job):
        if job == ("slow",):
            release.wait(5)
        with lock:
            jobs.append(job)

    w = MemoryWorker(handler, logger=_NullLogger(), atexit_timeout=5.0)
    w.submit(("slow",))
    time.sleep(0.1)                 # let the worker pick the slow job up
    w.submit(("queued-during-shutdown",))   # must still be processed

    threading.Thread(target=w._atexit_shutdown).start()
    time.sleep(0.2)                 # atexit is now blocked in flush()
    release.set()                   # let the worker reach the queued job

    deadline = time.time() + 5
    while ("queued-during-shutdown",) not in jobs and time.time() < deadline:
        time.sleep(0.02)
    assert ("queued-during-shutdown",) in jobs, (
        f"atexit discarded a queued job: {jobs}")


def test_handler_exception_does_not_kill_worker():
    boom_seen = []

    def handler(job):
        if job == ("boom",):
            raise ValueError("nope")
        boom_seen.append(job)

    w = MemoryWorker(handler, logger=_NullLogger(), atexit_timeout=1.0)
    w.submit(("boom",))
    w.submit(("after",))
    assert w.flush(timeout=5)
    assert boom_seen == [("after",)]   # worker survived the raising job
    w.close()
