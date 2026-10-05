# Async STM summarization — design spec

Date: 2026-10-04
Status: approved by partner in chat (approach A, "return without wait" semantics)

## Problem

`ShortTermMemory.add()` compresses the window **synchronously**: `_ensure_budget()`
(`stm.py`) loops up to 10 sequential summarizer LLM calls while holding the
STM's `RLock`, so every chat turn that crosses `max_tokens_stm` pays full LLM
latency (times up to 10) before `add()` returns. LTM extraction was already
off the hot path via `MemoryWorker`; STM summarization is the remaining
foreground blocker in the demo chatbot (`main.py`) and any library user.

## Decision (agreed)

Summarization moves to a dedicated background thread. `add()` only appends and
signals. `context()` **never waits**: while a summary is in flight the window
may temporarily exceed `max_tokens_stm` by one batch. The user-visible cost is
at most one oversized prompt; in practice human typing is slower than one
summary call, so it is invisible. A public `wait_until_settled()` is the
escape hatch for tests and for the final extraction at session end.

Rejected: reusing `MemoryWorker` (its queue can hold a multi-minute
consolidation agent loop — the summary would block behind it);
future-with-await-in-`context()` (puts the latency back exactly where the
chatbot reads context).

## Design

### Threading model

`ShortTermMemory` gains one daemon `threading.Thread` (name `memkit-stm`),
created in `__init__`, plus a `threading.Event` (`_work`) the producer sets and
the worker clears.

- **Producer (`add()` / `add_message()`):** append the message under
  `self._lock`, then `if self.token_count() > self.config.max_tokens_stm:
  self._work.set()`. No LLM call, no waiting. `_ensure_budget()` keeps its
  name/shape but its loop of summarize calls moves to the worker.
- **Worker thread:** loop `self._work.wait()` → clear → drain:
  `while self.token_count() > budget and self._summarize_once(): pass`.
  A failed `_summarize_once()` (split impossible / LLM failed) breaks the
  drain; the next `add()` re-arms `_work` and retries (same retry semantics
  as today — raw text is already durable before the LLM call).
  After the drain, if the window is under budget, mark the thread idle
  (`_idle.set()`); otherwise leave `_idle` clear and let the next signal
  re-trigger a drain attempt.

### The invariant that keeps it safe

All window surgery happens on the worker thread. The main thread's only
window mutation is `turns.append(msg)` under `_lock`. Therefore:

- `turns` list identity only changes in the commit step (worker thread, under
  `_lock`); `add()` appends to whatever list object is current. Appends that
  land mid-summary are preserved: the commit re-slices `turns[t:]` **under the
  lock at commit time**, so `t` messages are removed from the front and every
  newer message (including ones added during the LLM call) stays verbatim.
- `_appended` / `_lines` are touched only by the worker thread
  (`_append_transcript`) and the commit — serialized by construction.
- `_call_summarizer` and the prior-summary fold read `self.summary` on the
  worker thread itself; the sole writer is the same thread. No race.
- `context()` / `token_count()` keep reading under `_lock` (already true);
  `has_summary()` stays a GIL-atomic bool of a reference swap.
- `self.counter` (`TokenCounter`) is stateless per call and safe from both
  threads.

### `_summarize_once()` splits into three steps

1. **Snapshot (under `_lock`):** `t = self._choose_split()`; if `t <= 0`
   (warn-once path unchanged) return False; `to_compress = self.turns[:t]`;
   capture `prior_summary = self.summary`.
2. **Slow section (no lock):** `_append_transcript(to_compress)` — raw text
   fsynced first, abort-on-write-failure unchanged; then
   `_call_summarizer(appended, ...)`. Returns False if either fails; state is
   untouched so the next signal retries with a fresh snapshot.
   Note: `_appended` is advanced inside `_append_transcript` (worker-thread
   only — safe, see invariant); an abort after the write leaves the window
   intact exactly as today.
3. **Commit (under `_lock`):** `self.summary = new_summary`;
   `self.turns = self.turns[t:]`; `self._appended = max(0, self._appended - t)`;
   `self._fit_summary(new_summary)`.

`on_archive` (which submits the LTM extraction job to `MemoryWorker`) fires
**after commit, outside `_lock`** — if the worker queue were saturated,
`submit()` runs the job inline, and holding the STM lock there would block
`add()`/`context()` again (a lock held across a submit on a
queue-saturated path reproduces the bug we are fixing).

### Public API additions (on `ShortTermMemory`)

```python
def wait_until_settled(self, timeout: float | None = None) -> bool:
    """Block until the worker has no pending/outstanding compression work
    (window under budget after a full drain). True if settled in time."""

def close(self, timeout: float | None = None) -> bool:
    """wait_until_settled, then stop the worker thread. Idempotent."""
```

**Idle protocol (the race this pins down):** `_idle` flips and every
over-budget check happen under `self._lock`.

- Producer, under `_lock`: append → if `token_count() > budget`, then
  `_idle.clear()` and `_work.set()`.
- Worker, under `_lock`, at the end of a drain pass: if
  `token_count() <= budget`, `_idle.set()`.

Because both transitions hold the same lock, `wait_until_settled()` can never
observe "idle" while an over-budget append is unserviced (the append either
happens before the worker's final check — worker stays busy — or after the
worker released the lock with `_idle` set — the producer clears it again).
`_work` itself is a plain event: worker clears it at the top of each pass; a
set that lands mid-pass triggers one more drain immediately after.

`Memory.flush()` keeps meaning "drain the LTM worker"; STM settling is
`mem.stm.wait_until_settled()` (documented in README + facade docstring).

### Shutdown path (`facade._finalize`)

Order becomes:

1. `self.stm.wait_until_settled(timeout)` (bounded by `atexit_timeout`)
   — the final queued summary folds into `self.stm.summary` before anything
   reads it.
2. Build `remaining` from `summary + _render_transcript(turns)` and queue the
   end-of-session extract (unchanged) — now with the lock-free reads already
   safe because the worker is idle.
3. `self.stm.close()` stops the thread; `self.worker.close(...)` unchanged.

A hard SIGKILL mid-summary loses nothing new: the raw text is fsynced to the
transcript before the LLM call, and the next session re-summarizes from the
durable tail. Same durability story the README already documents.

### `Memory.close()` after `add()` races

`_finalized` guard already prevents double finalize. If `add()` runs
concurrently with `close()`, the new message may or may not make the final
extract; this is the documented pre-existing behavior, unchanged.

## Files touched

| File | Change |
|---|---|
| `src/memkit/stm.py` | worker thread + events; `add()`/`add_message()` signal-only; `_summarize_once` split snapshot/slow/commit; `on_archive` outside lock; new `wait_until_settled()` / `close()`; module docstring contract rewritten (lines 10-26: no longer "synchronous"). |
| `src/memkit/facade.py` | `_finalize()` gains `stm.wait_until_settled()` before reading the window and `stm.close()` at the end; class docstring note. |
| `tests/test_stm.py` | `stm.wait_until_settled()` after add-loops, before asserting `has_summary()`/`summary`/`turns`/`archives` (9 tests: transcript-file, append-twice, only-grows, resume, written-before-call, real-text, line-numbers, failed-retry, on-archive, pending-verbatim, abandoned-compressed, tool-group, prior-fold, under-cap). |
| `tests/test_integration.py` | `mem.stm.wait_until_settled()` before `has_summary()` assertions. |
| `tests/` (new cases) | 1. `test_add_returns_fast_while_summary_runs`: scripted LLM sleeps 200ms; assert `add()` returns < 50ms and the summary lands after `wait_until_settled`. 2. `test_adds_during_summary_stay_verbatim`: hold the LLM on an Event, add messages, release, commit must keep the new tail. 3. `test_wait_until_settled_timeout`: dead-slow LLM returns False without hanging. |
| `README.md` | STM section: summarize is backgrounded; the temporary-one-batch-over-budget note; API table if it lists behavior. |

Not touched: `worker.py`, `llm.py`, `ltm.py`, `agent_loop.py`, `tokens.py`,
`tools.py`, `config.py` (no new knobs), `main.py` (demo benefits unchanged —
same calls, now non-blocking).

## Out of scope (deliberate)

- Hard-ceiling inline fallback (approach C safety valve) — add later only if a
  real fast-streaming caller shows the oversized-window problem.
- `inline=True` deterministic test mode — the wait-helper covers tests with
  less API surface.
- Making `Memory.flush()` also drain STM (name collision with LTM-only
  semantics; keep the two knobs separate).

## Testing strategy

- All existing STM tests keep asserting the same end state (window under cap,
  one transcript, numbering, fold-in) — they only add
  `wait_until_settled()`; no test's *semantics* change, proving the
  compression contract survives.
- New tests above cover: non-blocking add, tail preservation across a commit,
  timeout behavior.
- Full suite must pass with the fake LLMs (no network); timing test uses
  generous margins (200ms sleep vs 50ms assert) to stay CI-stable.
