"""The ``Memory`` facade — the whole public surface in a few lines.

    from memkit import Memory

    mem = Memory(
        storage_dir="./agent_memory",
        llm=my_llm,                # llm(messages) -> str, or ANY LLM client
                                   # object you initialized yourself
        max_tokens_stm=4000,       # YOUR cap, the framework fixes nothing
        max_tokens_ltm=8000,       # YOUR cap
        llm_tools=my_llm_tools,    # llm_tools(messages, tools) -> assistant msg
        user_id="alice",           # optional: one memory folder per user
    )

    mem.add("user", "hi")
    msgs = [{"role": "system", "content": mem.system_prompt_fragment()}] + mem.context()
    reply = call_my_model(msgs)
    mem.add("assistant", reply)
    mem.close()   # or rely on atexit

See module docstrings of stm/ltm/worker for the internals.
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Any, Callable

from memkit.config import MemoryConfig
from memkit.llm import adapt_llm
from memkit.ltm import LongTermMemory
from memkit.prompts import system_prompt_fragment as _build_fragment
from memkit.stm import ShortTermMemory
from memkit.tokens import TokenCounter
from memkit.tools import MemoryTools, read_specs
from memkit.worker import MemoryWorker

# How often (in add()s) to check the LTM budget from the foreground path;
# between checks the worker re-checks after every extraction anyway.
_LTM_CHECK_EVERY = 16


class Memory:
    def __init__(
        self,
        storage_dir,
        *,
        llm: Callable | None = None,
        max_tokens_stm: int | None = None,
        max_tokens_ltm: int | None = None,
        llm_tools: Callable | None = None,
        encoding: str = "o200k_base",
        msg_overhead: int = 4,
        token_counter: Callable[[str], int] | None = None,
        keep_recent: float = 0.25,
        max_summary_tokens: int | None = None,
        agent_loop_max_steps: int = 12,
        atexit_timeout: float = 30.0,
        session_id: str | None = None,
        user_id: str | int | None = None,
        retries: int = 2,
        logger=None,
    ) -> None:
        # --- positional-or-keyword sugar: Memory(dir, llm, stm, ltm) also works
        if llm is None or max_tokens_stm is None or max_tokens_ltm is None:
            raise TypeError(
                "Memory requires llm, max_tokens_stm and max_tokens_ltm — "
                "the framework deliberately sets no defaults for them.")

        # Accept ANY LLM the user initialized themselves: a plain callable
        # (used as-is), one of memkit's adapters, or a provider SDK *client*
        # object (openai.OpenAI(...), anthropic.Anthropic(), ...), which
        # adapt_llm recognizes by its API surface and wraps for them.
        # A logger was passed: use it for the adapter's own messages too, rather
        # than the package logger, so an adapt notice goes where the user looks.
        early_logger = logger or logging.getLogger("memkit")
        llm = adapt_llm(llm, early_logger)
        if llm_tools is not None:
            llm_tools = adapt_llm(llm_tools, early_logger)
        else:
            # adapters (OpenAILLM/AnthropicLLM) carry their own tool-capable
            # callable; anything else that has one is honored too.
            llm_tools = getattr(llm, "with_tools", None)

        self.config = MemoryConfig(
            storage_dir=storage_dir,
            llm=llm,
            llm_tools=llm_tools,
            max_tokens_stm=max_tokens_stm,
            max_tokens_ltm=max_tokens_ltm,
            encoding=encoding,
            msg_overhead=msg_overhead,
            token_counter=token_counter,
            keep_recent=keep_recent,
            max_summary_tokens=max_summary_tokens,
            agent_loop_max_steps=agent_loop_max_steps,
            atexit_timeout=atexit_timeout,
            session_id=session_id,
            user_id=user_id,
            retries=retries,
            logger=logger,
        )
        # No tool caller is not fatal: extraction and inbox persistence still
        # run, and ltm.consolidate() degrades to logged inbox-only mode.
        # Semantic merging/pruning needs one, so say so once — through the
        # config's resolved logger, which exists whether or not the user
        # passed one.
        if llm_tools is None:
            self.config.logger.info(
                "memkit: no tool-calling llm detected — LTM consolidation "
                "will not run (facts are still saved to topic files).")
        cfg = self.config
        cfg.storage_dir.mkdir(parents=True, exist_ok=True)

        self.counter = TokenCounter(cfg.encoding, cfg.msg_overhead, cfg.token_counter)
        # The memory tools see ONLY this instance's memory root: with a
        # user_id, that is storage_dir/<user_id>/, so one Memory for user 1
        # cannot read or write user 2's memories even through the tool layer.
        self.toolset = MemoryTools(cfg.root_dir, cfg.ltm_dir,
                                  session_id=cfg.session_id)
        self.stm = ShortTermMemory(cfg, self.counter, on_archive=self._on_archive)
        self.ltm = LongTermMemory(cfg, self.counter, self.toolset)
        self.worker = MemoryWorker(self._handle_job, cfg.logger, cfg.atexit_timeout)

        # Guards _adds_since_check / _finalized against a concurrent close()
        # (e.g. atexit firing while another live thread is still calling add()).
        self._state_lock = threading.Lock()
        self._adds_since_check = 0
        self._finalized = False
        # If the user never calls close(), atexit still performs the final
        # extraction so short sessions leave memories behind.
        atexit.register(self._atexit_finalize)

    # -- public API -------------------------------------------------------------

    def add(self, role: str, content: Any = None,
            tool_calls: list[dict[str, Any]] | None = None,
            name: str | None = None, tool_call_id: str | None = None) -> None:
        """Append one turn to short-term memory.

        Never blocks on an LLM: when the window is over cap the summarizer is
        armed and compresses on memkit's background thread, so your reply
        latency is unaffected. Use ``mem.stm.wait_until_settled()`` when you
        need the window deterministically compacted (tests, diagnostics)."""
        self.stm.add(role, content, tool_calls=tool_calls, name=name,
                     tool_call_id=tool_call_id)
        self._maybe_check_ltm_budget()

    def add_many(self, msgs: list[dict[str, Any]]) -> None:
        """Append several OpenAI-style message dicts at once."""
        for m in msgs:
            self.stm.add_message(m)
        self._maybe_check_ltm_budget()

    def context(self) -> list[dict[str, Any]]:
        """The current conversation window: summary + verbatim tail.

        Prepend your own system prompt (or ``system_prompt_fragment()``).
        """
        return self.stm.context()

    def system_prompt_fragment(self) -> str:
        """Ready-to-use system-prompt block: memory tools + transcript recovery
        instructions + the current MEMORY.md index."""
        return _build_fragment(
            memory_md=self.ltm.read_index(),
            transcript_path=self.stm.transcript_rel,
            has_summary=self.stm.has_summary(),
        )

    def tools(self) -> list[dict[str, Any]]:
        """OpenAI-format tool specs for the user's agent: the four MEMORY read
        tools (recall_transcript, read_memory_file, search_memory_lines,
        list_memory_files). Execute them with ``execute_tool``. The
        write/edit/delete memory tools belong to memkit's consolidation loop
        and are not exposed here."""
        return read_specs(["read_memory_file", "search_memory_lines",
                           "recall_transcript", "list_memory_files"])

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> str:
        """Run one of ``tools()``; returns the tool result text. Never raises."""
        if name in ("write_memory_file", "edit_memory_file",
                    "delete_memory_file"):
            return (f"Error: {name} is reserved for memkit's background memory "
                    "consolidation. Ask the user explicitly before managing "
                    "ltm/ memory files from the conversation.")
        return self.toolset.execute(name, arguments)

    def compact(self, timeout: float | None = None) -> bool:
        """Summarize short-term memory right now, ignoring the STM token cap.

        What a user-facing ``/compact`` command calls: it produces a summary
        on demand instead of waiting for the window to fill, keeping only the
        newest turn (or a still-pending tool-call group) verbatim. The folded
        text stays verbatim-recoverable from the transcript, and the archive
        feeds LTM extraction exactly as a cap-driven summary does.

        Blocks until the summarizer settles (or ``timeout`` elapses); returns
        True if it settled. A window with a single span already has nothing to
        fold, so this returns True without an LLM call."""
        self.stm.force_compact()
        return self.stm.wait_until_settled(timeout)

    def flush(self, timeout: float | None = None) -> bool:
        """Wait for the background **LTM** worker to finish everything queued.

        This is LTM-only by design; STM summarization is waited on separately
        with ``mem.stm.wait_until_settled()`` (``close()`` does both)."""
        return self.worker.flush(timeout)

    def close(self, timeout: float | None = None) -> bool:
        """Final extraction of the remaining conversation, drain, stop workers.

        Settles the STM summarizer first so a summary still in flight is folded
        into the end-of-session extraction rather than missed, then stops both
        background threads. Call this at session end (atexit does it too, with
        a bounded wait)."""
        return self._finalize(timeout=timeout)

    # -- internals ------------------------------------------------------------

    def _maybe_check_ltm_budget(self) -> None:
        """Every ``_LTM_CHECK_EVERY`` adds, ask the worker to look at the budget.

        The counter is read and reset under ``_state_lock`` so concurrent
        ``add()``s cannot both decide they are the "due" one; the budget scan
        itself (a whole-tree token count) stays outside the lock — it is the
        expensive part, and a duplicate consolidate job is a no-op when the
        tree is already under budget.
        """
        with self._state_lock:
            if self._finalized:
                return
            self._adds_since_check += 1
            due = self._adds_since_check >= _LTM_CHECK_EVERY
            if due:
                self._adds_since_check = 0
        if due and self.ltm.over_budget():
            self.worker.submit(("consolidate", None))

    def _on_archive(self, archive_rel: str, archive_text: str) -> None:
        # Every STM summarization automatically feeds LTM extraction,
        # asynchronously — the foreground add() does not wait.
        self.worker.submit(("extract", (f"Archived transcript {archive_rel}",
                                       archive_text)))

    def _handle_job(self, job: tuple) -> None:
        kind, payload = job
        if kind == "extract":
            label, text = payload
            self.ltm.extract(text, label)
            # Deterministic persistence first: facts reach topic files even
            # if no further consolidation ever runs. Semantic merging/pruning
            # happens later, once the budget is hit.
            self.ltm.flush_inbox_to_files()
            if self.ltm.over_budget():
                self.ltm.consolidate()
        elif kind == "consolidate":
            if self.ltm.over_budget():
                self.ltm.consolidate()
        else:
            self.config.logger.error("memkit: unknown worker job %r", kind)

    def _finalize(self, timeout: float | None = None) -> bool:
        with self._state_lock:
            if self._finalized:
                return True
            self._finalized = True
        try:
            atexit.unregister(self._atexit_finalize)
        except Exception:
            pass
        # Let the STM summarizer finish first: a summary still in flight would
        # otherwise be lost from the window we are about to read, and both the
        # reads below are lock-free (safe only once the worker is idle).
        settle_timeout = timeout if timeout is not None else self.config.atexit_timeout
        self.stm.wait_until_settled(settle_timeout)
        # Extract only the turns no summarization has archived (covers sessions
        # that never hit max_tokens_ltm) — queued, then drained by close().
        # Archived spans were already handed to extraction by on_archive, so
        # re-rendering the whole window here would duplicate that LLM work.
        remaining = self.stm.unarchived_transcript()
        if remaining.strip():
            self.worker.submit(("extract", ("End-of-session conversation", remaining)))
        if self.ltm.inbox_pending():
            self.worker.submit(("consolidate", None))
        # Freeze and stop the STM worker before the LTM worker closes, so no
        # late summary can submit an extraction job after the queue is shut.
        self.stm.close(settle_timeout)
        self.worker.close(timeout if timeout is not None else self.config.atexit_timeout)
        return True

    def _atexit_finalize(self) -> None:
        self._finalize()

    # convenience accessors for power users / tests
    @property
    def session_id(self) -> str:
        return self.config.session_id

    @property
    def user_id(self) -> str | None:
        return self.config.user_id

    @property
    def storage_dir(self):
        return self.config.storage_dir

    @property
    def memory_root(self):
        """The directory holding THIS user's memory set: ``storage_dir`` or
        ``storage_dir/<user_id>`` when a user_id is set."""
        return self.config.root_dir
