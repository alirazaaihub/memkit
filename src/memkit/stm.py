"""Short-term memory: a live message window that summarizes itself when it hits
the user's token cap.

Fresh per conversation. The one thing persisted is the conversation's raw
transcript: ``stm/<session_id>.md``. Exactly ONE file per conversation, named
by the session id, written *before* each summarization and only ever appended
to — so an agent in that conversation can always find the exact original text
again (see ``prompts._STM_FRAGMENT``).

Summarization runs on a background thread, not inside ``add``: an over-cap
``add`` only appends the message and signals the worker, so no chat turn ever
waits on an LLM. The trade-off is that ``context`` never waits either — while a
summary is in flight the window can sit one batch over the cap (in a real chat
a turn arrives far slower than one summary call, so this is invisible).
``wait_until_settled`` is the deterministic escape hatch for tests and for the
final extraction at session end. The contract:

1. Append the raw real text that is about to be compressed to the transcript
   first (``with open`` + fsync). If that write fails, DO NOT summarize — never
   lose the original. Only the not-yet-written delta is appended, so a retry
   after a failed summarization never duplicates a span.
2. Ask the user's LLM to compress the archived text (plus any prior summary)
   into structured sections that keep concrete values. This — the only slow
   step — happens on the worker thread with no lock held, so ``add`` and
   ``context`` keep working while a summary is being written.
3. New window = one summary message + the verbatim recent tail + the most
   recent *pending* tool-call group (an assistant tool_calls whose results
   haven't all arrived), which stays verbatim while it is still the newest
   thing in the window. Once the user continues past an unresolved call the
   call is treated as abandoned (no provider accepts new turns before its
   results), so it may be archived and summarized like any other text — the
   summary prompt keeps a "Pending Tool Calls" section for it.
"""

from __future__ import annotations

import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from memkit.config import MemoryConfig
from memkit.llm import call_with_retries
from memkit.prompts import STM_SUMMARY_SYSTEM, STM_SUMMARY_USER_HEADER
from memkit.tokens import TokenCounter


class ShortTermMemory:
    def __init__(
        self,
        config: MemoryConfig,
        counter: TokenCounter,
        on_archive: Callable[[str, str], None] | None = None,
    ) -> None:
        self.config = config
        self.counter = counter
        self.on_archive = on_archive
        self._lock = threading.RLock()
        self.turns: list[dict[str, Any]] = []       # verbatim messages only
        self.summary: str | None = None
        # Running token total of `turns`, maintained incrementally under
        # `_lock`. Without it, every add() re-tokenized the WHOLE window just to
        # answer "are we over cap?" — an O(window) cost on the hot path that
        # grows with the conversation. token_count() still measures the truth
        # and re-syncs this whenever the two could disagree.
        self._turns_tokens = 0
        # The user owns this cap when they set it; otherwise derive it from the
        # STM budget (35%, min 300) as a sane default.
        self._max_summary_tokens = (
            config.max_summary_tokens
            if config.max_summary_tokens is not None
            else max(300, int(config.max_tokens_stm * 0.35))
        )
        self._transcript = Path(config.session_file)
        # Relative to the memory root (storage_dir / user_id when a user_id is
        # set), so the path the agent sees is exactly what the memory tools
        # accept, whatever the on-disk layout looks like from the outside.
        self._rel = self._transcript.relative_to(config.root_dir).as_posix()
        # The one transcript, listed as soon as it exists (a resumed session
        # with a pinned id already has one before this object summarises).
        self.archives: list[str] = [self._rel] if self._transcript.exists() else []
        # How much of self.turns is already in the transcript, and how many
        # physical lines the file holds. Together these keep every [n] marker
        # equal to the file's true 1-based line number across appends.
        self._appended = 0
        self._lines = self._existing_line_count()
        # --- background summarizer ------------------------------------------
        # One dedicated worker thread, NOT the LTM MemoryWorker: that queue can
        # hold a multi-minute consolidation agent loop, and a summary must
        # never wait behind it. `_work` is the arm signal; `_idle` means "no
        # compression is in flight" (see wait_until_settled).
        self._work = threading.Event()
        self._idle = threading.Event()
        self._idle.set()          # nothing to compress yet
        # One-shot "compress now, cap or no cap" request (force_compact). It
        # carries *what* the pass should do, while `_work` only says "wake up",
        # so a forced pass and an ordinary over-cap pass cannot be confused.
        self._force = threading.Event()
        self._stopped = threading.Event()
        self._thread = threading.Thread(
            target=self._worker_loop, name="memkit-stm", daemon=True)
        self._thread.start()

    # -- public ---------------------------------------------------------------

    def add(self, role: str, content: Any,
            tool_calls: list[dict[str, Any]] | None = None,
            name: str | None = None, tool_call_id: str | None = None) -> None:
        """Append one turn. Never blocks on an LLM: if the window is over cap
        the summarizer is armed and runs on the background thread."""
        msg: dict[str, Any] = {"role": role}
        if content is not None:
            msg["content"] = content
        if tool_calls:
            msg["tool_calls"] = tool_calls
        if name:
            msg["name"] = name
        if tool_call_id:
            msg["tool_call_id"] = tool_call_id
        with self._lock:
            self.turns.append(msg)
            self._turns_tokens += self.counter.count_message(msg)
            self._ensure_budget()

    def add_message(self, msg: dict[str, Any]) -> None:
        if "role" not in msg:
            raise ValueError("message dict must have a 'role'")
        with self._lock:
            stored = dict(msg)
            self.turns.append(stored)
            self._turns_tokens += self.counter.count_message(stored)
            self._ensure_budget()

    def context(self) -> list[dict[str, Any]]:
        with self._lock:
            out: list[dict[str, Any]] = []
            if self.summary:
                out.append({"role": "user", "content": self._summary_text()})
            out.extend(dict(m) for m in self.turns)
            return out

    def token_count(self) -> int:
        with self._lock:
            measured = self.counter.count_messages(self.turns)
            if measured != self._turns_tokens:
                # Only reachable if a mutation bypassed the incremental path
                # (a direct `turns` edit by a caller): re-sync so the hot path
                # keeps its cheap running total.
                self._turns_tokens = measured
            if self.summary:
                measured += self.counter.count_message(
                    {"role": "user", "content": self._summary_text()})
            return measured

    def has_summary(self) -> bool:
        return bool(self.summary)

    def unarchived_transcript(self) -> str:
        """Numbered text of the turns no summarization has archived yet.

        This is the only part of the window LTM extraction has never seen: every
        archived span was already fed to extraction by the ``on_archive`` hook.
        Session end extracts exactly this delta, so it can never duplicate work
        that was queued (or done) during the conversation. Returns "" when
        everything in the window is already on disk.
        """
        with self._lock:
            if self._appended >= len(self.turns):
                return ""
            return _render_transcript(self.turns[self._appended:],
                                      first_lineno=self._lines + 1)

    @property
    def transcript_rel(self) -> str:
        """Storage-relative path of this conversation's transcript file."""
        return self._rel

    def force_compact(self) -> None:
        """Ask the summarizer to compress NOW, whether or not the window is
        over cap — this is what a user's ``/compact`` command calls.

        One forced pass ignores ``keep_recent``: it keeps only the newest span
        verbatim (the last message, or a still-pending tool-call group) and
        summarizes everything before it, so a summary is produced even from a
        short conversation. A window with nothing to fold (0 or 1 span) makes
        it a no-op.

        Returns immediately — the LLM call stays on the background thread, so a
        caller's UI never blocks; ``wait_until_settled`` is the explicit way to
        wait for the summary to appear. A call after ``close`` is ignored.
        """
        if self._stopped.is_set():
            return
        # Take the lock exactly as _ensure_budget does, so wait_until_settled
        # stays race-free: the worker can never look idle while a forced pass
        # is armed but unserviced.
        with self._lock:
            self._force.set()
            self._idle.clear()
            self._work.set()

    # -- internals --------------------------------------------------------------

    def _summary_text(self) -> str:
        return (
            "[memkit short-term memory] Earlier turns were compressed below. "
            "The originals are in your conversation transcript "
            f"({self._rel}) — recover exact details with recall_transcript. "
            "Followed by recent messages verbatim.\n\n" + self.summary
        )

    def _ensure_budget(self) -> None:
        """Arm the background summarizer when the window is over cap.

        Called by ``add``/``add_message`` with ``self._lock`` held. It must NOT
        compress here: the LLM call belongs to the worker thread, so no chat
        turn ever waits on it. The over-budget check and the ``_idle``
        transition share this lock, which is what makes ``wait_until_settled``
        race-free — the worker can never declare itself idle while an
        over-budget append is still unserviced.

        The check costs O(1), not O(window): it uses the running tail total plus
        one token count of the summary, both of which the append already paid
        for. Re-counting every message here made each add() cost grow with the
        conversation.
        """
        if not self.summary:
            if self._turns_tokens <= self.config.max_tokens_stm:
                return
        elif (self._turns_tokens + self.counter.count_message(
                {"role": "user", "content": self._summary_text()})
                <= self.config.max_tokens_stm):
            return
        self._idle.clear()
        self._work.set()

    def wait_until_settled(self, timeout: float | None = None) -> bool:
        """Block until no compression is pending or in flight.

        "Settled" means the summarizer is not mid-pass and nothing new is
        armed — not necessarily "window is under cap". A pass that gave up (no
        progress possible, transcript write error, dead provider) also settles,
        because nothing is running any more and the next ``add()`` re-arms the
        retry. That keeps ``_finalize`` from hanging on an unreachable LLM.
        Returns True if settled within ``timeout`` (None waits indefinitely).
        """
        return self._idle.wait(timeout)

    def close(self, timeout: float | None = None) -> bool:
        """Settle the window, then stop the worker thread. Idempotent.

        After ``close`` the window is frozen: no further compression runs, and
        ``add`` only appends (text that was never summarized is still durable
        in the transcript). Safe to call twice and from ``atexit``.

        Returns True only when the summarizer actually settled first. A pass
        that outlives ``timeout`` keeps the thread alive to finish its commit:
        killing it mid-pass is not an option (its LLM call is uninterruptible)
        and reporting "settled" would let the caller read and then clear a
        window that a late commit then rewrites — dropping the newest turns
        from the live context. The timeout only contains the *garbage
        collection* — the thread stops itself when the pass ends — it never
        contains liveness.
        """
        if self._stopped.is_set():
            return True
        settled = self.wait_until_settled(timeout)
        self._stopped.set()
        self._work.set()          # wake the worker so it sees the stop flag
        thread = self._thread
        if (thread is not None and thread.is_alive()
                and thread is not threading.current_thread()):
            # Unbounded: the stop flag is already set and `_worker_loop` checks
            # it before every pass, so the thread cannot start new work — it
            # only finishes the commit it is inside. See the docstring.
            thread.join()
        return settled

    def _worker_loop(self) -> None:
        """The single thread that ever performs window surgery.

        Producer threads only append under ``_lock``; every read-modify-write of
        ``turns``/``summary``/``_appended`` happens here, so those fields need
        no further synchronization beyond serializing the main thread's appends.
        """
        while True:
            self._work.wait()
            if self._stopped.is_set():
                return
            self._work.clear()
            # Drain: the window may need several compressions to fit, and each
            # is a slow LLM call, so do them all here and only then decide.
            guard = 0
            while guard < 10:
                guard += 1
                with self._lock:
                    # Consume the one-shot force request (one pass per call), so
                    # a /compact cannot be confused with an ordinary over-cap
                    # pass: while it is set, "over cap" is not what decides.
                    forced = self._force.is_set()
                    self._force.clear()
                    if not forced and self.token_count() <= self.config.max_tokens_stm:
                        break
                if not self._summarize_once(force=forced):
                    break  # nothing to compress, or it failed; next add retries
            with self._lock:
                # An append that arrived during the pass cleared `_idle` and set
                # `_work`; leave `_idle` clear so the waiters keep waiting and
                # loop around for another drain instead of reporting settled.
                # Once `close` has stopped the window, `_idle` is the closer's
                # to manage: waking a waiter now would report "settled" for a
                # window that is frozen mid-pass.
                if not self._work.is_set() and not self._stopped.is_set():
                    self._idle.set()

    def _summarize_once(self, force: bool = False) -> bool:
        """Compress one batch. Returns False if no progress was possible.

        Runs on the worker thread only, in three steps: a short locked
        snapshot, a slow lock-free LLM section, then a short locked commit.
        Every field the live window can observe is mutated under ``_lock``;
        the middle step touches nothing the main thread reads.

        ``force`` (a /compact request) is passed through to ``_choose_split``
        so this one pass keeps only the newest span verbatim. Everything else —
        archive-before-summarize, the mid-flight re-slice, feeding LTM — is
        the ordinary path.
        """
        # 1. Snapshot (locked): only what this compression will consume.
        with self._lock:
            t = self._choose_split(force)
            if t <= 0:
                if self.token_count() > self.config.max_tokens_stm:
                    self.config.logger.warning(
                        "memkit STM: single recent group exceeds the token cap; "
                        "cannot compress further without dropping it. Consider a "
                        "larger max_tokens_stm.")
                return False
            to_compress = self.turns[:t]
            prior_summary = self.summary

        # 2. Slow section (no lock): append the raw real text BEFORE
        #    summarizing, then compress. Only the not-yet-written delta is
        #    appended, so a retry after a failed summarization extends the
        #    transcript instead of duplicating the span (no duplicate text, no
        #    duplicate LTM extraction). The window is untouched on either
        #    failure, so the next signal retries from a fresh snapshot.
        try:
            appended = self._append_transcript(to_compress)
        except OSError as e:
            self.config.logger.error(
                "memkit STM: transcript write failed (%s); NOT summarizing to "
                "avoid losing original text.", e)
            return False  # window intact; the next add() retries

        new_summary = self._call_summarizer(appended, self._rel, prior_summary)
        if new_summary is None:
            # Summarizer failed: keep the un-summarized window; the raw text is
            # already durable, so the retry appends only what's still missing.
            return False

        # 3. Commit (locked). Re-slice here, not at snapshot time: messages
        #    added while the LLM was thinking sit past t and stay verbatim.
        with self._lock:
            self.summary = new_summary
            self.turns = self.turns[t:]
            self._appended = max(0, self._appended - t)
            # Re-derive the running total from the live tail rather than
            # subtracting: the compressed prefix and the retained tail are
            # disjoint, so this is exact and cannot drift.
            self._turns_tokens = self.counter.count_messages(self.turns)
            # Overflow guard: if the summary alone is huge, trim its Chronology.
            self._fit_summary(new_summary)

        # Outside the lock: on_archive submits to MemoryWorker, and a saturated
        # queue runs the job inline — holding the STM lock across that would put
        # the original blocking bug right back into add()/context().
        if appended and self.on_archive:
            try:
                self.on_archive(self._rel, appended)
            except Exception:  # feed failure must never break STM
                self.config.logger.exception("memkit STM: on_archive hook failed")
        return True

    # -- transcript helpers -------------------------------------------------------

    def _existing_line_count(self) -> int:
        """Physical lines already in the transcript (0 when there is none).

        A resumed session with a pinned ``session_id`` continues writing at the
        end of this file, so numbering must resume where the file left off.
        """
        if not self._transcript.exists():
            return 0
        text = self._transcript.read_text(encoding="utf-8")
        return len(text.splitlines()) if text else 0

    def _header_block(self) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        header = (
            f"# memkit STM transcript — session {self.config.session_id}\n"
            f"opened {ts}. Lines are numbered [n] and those numbers equal this "
            f"file's true line numbers, so a search_memory_lines hit at line n "
            f"maps to read_memory_file(offset=n-1), and recall_transcript "
            f"returns these lines verbatim.")
        return header + "\n\n"

    def _append_transcript(self, msgs: list[dict[str, Any]]) -> str:
        """Append the not-yet-written tail of ``msgs`` to the one transcript.

        Returns the numbered text that was appended (empty when everything in
        ``msgs`` is already on disk). The header is written only when the file
        is created, and numbering continues from the file's current last line,
        so ``[n]`` always equals the true 1-based line number.
        """
        delta = msgs[self._appended:]
        if not delta:
            return ""
        creating = not self._transcript.exists()
        header = self._header_block() if creating else ""
        first_lineno = self._lines + len(header.splitlines()) + 1
        body = _render_transcript(delta, first_lineno=first_lineno)
        self._transcript.parent.mkdir(parents=True, exist_ok=True)
        # Raw real text saved to the file FIRST, durably, with `with open`.
        with open(self._transcript, "a", encoding="utf-8", newline="\n") as fh:
            if header:
                fh.write(header)
            fh.write(body)
            fh.flush()
            os.fsync(fh.fileno())
        self._lines += len(header.splitlines()) + len(body.splitlines())
        self._appended = len(msgs)
        if creating and self._rel not in self.archives:
            self.archives.append(self._rel)
        return body

    def _call_summarizer(self, transcript_text: str, transcript_rel: str,
                         prior_summary: str | None = None) -> str | None:
        """Compress one archived span. ``prior_summary`` is passed in rather
        than read from ``self.summary`` so the fold always uses the summary
        that was current at snapshot time."""
        sys_prompt = STM_SUMMARY_SYSTEM.format(max_summary_tokens=self._max_summary_tokens)
        parts = [sys_prompt.strip()]
        if prior_summary:
            parts.append(
                "\nPRIOR SUMMARY (merge these facts into your new summary; "
                "do not lose any):\n" + prior_summary)
        parts.append(
            STM_SUMMARY_USER_HEADER.format(archive_path=transcript_rel)
            + "\n" + transcript_text)
        messages = [
            {"role": "system", "content": parts[0]},
            {"role": "user", "content": "\n".join(parts[1:])},
        ]
        try:
            return call_with_retries(
                lambda: self.config.llm(messages), self.config.retries,
                self.config.logger)
        except Exception:
            self.config.logger.exception("memkit STM: summarizer LLM failed")
            return None

    # -- summarization helpers ---------------------------------------------------

    def _fit_summary(self, summary: str) -> None:
        window = (self.counter.count_text(self._summary_text())
                  + self._turns_tokens)
        if window <= self.config.max_tokens_stm:
            return
        self.config.logger.warning(
            "memkit STM: summary still over budget after compression; "
            "hard-truncating oldest chronology (never silently aborting).")
        lines = summary.splitlines()
        chrono_at = next((i for i, l in enumerate(lines)
                          if l.strip().lower().startswith("## chronology")), None)
        if chrono_at is not None and len(lines) - chrono_at > 4:
            # Keep the heading plus the most recent few chronology lines. When
            # the body is already short enough that keep_from lands ON the
            # heading, slicing from it is the right answer — but do not then
            # prepend lines[:chrono_at + 1], which would duplicate the heading.
            keep_from = max(chrono_at, len(lines) - 5)
            if keep_from == chrono_at:
                trimmed = "\n".join(lines[chrono_at:])
            else:
                trimmed = "\n".join(lines[:chrono_at + 1] + lines[keep_from:])
            if self.counter.count_text(trimmed) < self.counter.count_text(summary):
                self.summary = trimmed

        # Still over budget? Cut the summary itself to the largest prefix that
        # fits the remaining room (wrapper + tail reserved). The full summary
        # is not lost forever: the raw originals are in the transcript.
        wrapped = self.counter.count_message(
            {"role": "user", "content": self._summary_text()})
        tail_cost = self._turns_tokens
        overhead = wrapped - self.counter.count_text(self.summary)
        available = self.config.max_tokens_stm - tail_cost - overhead
        if available > 0:
            self.summary = self.counter.truncate(self.summary, available)

    def _choose_split(self, force: bool = False) -> int:
        """Index in self.turns where the verbatim tail begins.

        Never splits a tool-call group; always keeps the most recent group
        verbatim; and NEVER compresses a group whose tool results have not
        all arrived (pending tool calls) — those stay in the window no
        matter the token cap, because dropping a still-needed call breaks
        the agent's task.

        ``force`` is the /compact path: the cap is irrelevant, so the
        ``keep_recent`` tail is ignored and only the newest span stays
        verbatim. That guarantees a summary out of any window holding more
        than one span — the point of an on-demand compaction.
        """
        turns = self.turns
        spans = _tool_spans(turns)
        if force:
            return spans[-1][0] if spans else 0
        keep_budget = self.config.keep_recent * self.config.max_tokens_stm
        acc = 0
        t = 0  # default: compress everything but we may not; see below
        for k in range(len(spans) - 1, -1, -1):
            s, e = spans[k]
            cost = self.counter.count_messages(turns[s:e])
            # Always absorb the newest span; add more only while under budget.
            if k < len(spans) - 1 and acc > 0 and acc + cost > keep_budget:
                t = spans[k + 1][0]
                break
            acc += cost
        else:
            t = 0  # everything fits in the tail → nothing to compress
        # A still-pending tool-call group at the END of the window is already
        # covered: the loop above always absorbs the newest span, so it stays
        # verbatim. A pending group followed by anything else was abandoned
        # (providers reject new turns before the results arrive — a result for
        # it can no longer legitimately follow the newer message), so older
        # pending groups lose compression immunity: keeping them verbatim
        # forever made the window grow without bound (its exact text is in the
        # transcript, and the summary prompt carries a "Pending Tool Calls"
        # section for it).
        return t


# -- module-level helpers -----------------------------------------------------

def _tool_spans(turns: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """Group assistant-with-tool_calls messages together with their results so
    a split can never fall inside a call/result pair. Returns [start, end) spans.
    """
    spans: list[tuple[int, int]] = []
    i, n = 0, len(turns)
    while i < n:
        m = turns[i]
        tcs = m.get("tool_calls") or []
        if m.get("role") == "assistant" and tcs:
            ids = {c.get("id") for c in tcs if c.get("id") is not None}
            j = i + 1
            while (j < n and turns[j].get("role") == "tool"
                   and turns[j].get("tool_call_id") in ids):
                j += 1
            spans.append((i, j))
            i = j
        else:
            spans.append((i, i + 1))
            i += 1
    return spans


def _render_transcript(msgs: list[dict[str, Any]],
                       first_lineno: int = 1) -> str:
    """Render messages to numbered physical lines: ``[lineno] role: text``.

    ``lineno`` starts at ``first_lineno`` and equals the true 1-based line
    number in the transcript file, so a search_memory_lines hit maps straight
    to read_memory_file(offset=lineno-1) and recall_transcript returns these
    same numbers. Standalone callers (tests, recovery dumps) keep the default
    of 1.
    """
    physical: list[tuple[str, str]] = []
    for m in msgs:
        role = m.get("role", "?")
        content = m.get("content")
        if isinstance(content, str) and content:
            for line in content.splitlines():
                physical.append((role, line))
        elif content is not None:
            physical.append((role, str(content)))
        for tc in m.get("tool_calls") or []:
            args = tc.get("arguments")
            if not isinstance(args, str):
                import json as _json
                args = _json.dumps(args, ensure_ascii=False)
            physical.append((role, f"-> tool_call id={tc.get('id')} {tc.get('name')}({args})"))
        if role == "tool":
            physical.append((role, f"<- result for id={m.get('tool_call_id')}"))
    out = []
    for idx, (role, text) in enumerate(physical, start=first_lineno):
        out.append(f"[{idx}] {role}: {text}")
    return "\n".join(out) + "\n"