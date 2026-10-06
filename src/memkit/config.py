"""Configuration for a :class:`memkit.Memory` instance.

Every knob the user should own is here; nothing important is fixed by the
framework. Only ``llm``, ``max_tokens_stm`` and ``max_tokens_ltm`` are
required — the rest have documented defaults.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

# Duck-typed LLM protocols (see llm.py for adapters):
#   llm(messages) -> str
#   llm_tools(messages, tools) -> normalized assistant message dict
LLMCall = Callable[[list[dict[str, Any]]], str]
LLMToolsCall = Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]]


def _generate_session_id() -> str:
    """A fresh random id for one conversation.

    Random (not time-ordered) so two chats started in the same second still
    get distinct transcript files, and so nobody can guess another session's
    file name. Users who want a fixed, meaningful id can pass ``session_id``.
    """
    return uuid.uuid4().hex


def _validate_leaf(value: Any, name: str) -> None:
    """Reject anything that could escape a directory when used as a name."""
    if (not isinstance(value, str) or not value or value in (".", "..")
            or "/" in value or "\\" in value or "\x00" in value
            or any(ord(c) < 32 for c in value)):
        raise ValueError(
            f"{name} must be a simple directory/file name (no separators, "
            f"no '..'), got {value!r}")


@dataclass
class MemoryConfig:
    # --- required -----------------------------------------------------------
    storage_dir: Path
    llm: LLMCall
    max_tokens_stm: int
    max_tokens_ltm: int
    # --- LTM consolidation ---------------------------------------------------
    llm_tools: LLMToolsCall | None = None
    # --- token counting -------------------------------------------------------
    encoding: str = "o200k_base"
    msg_overhead: int = 4
    token_counter: Callable[[str], int] | None = None
    # --- STM behavior ----------------------------------------------------------
    keep_recent: float = 0.25
    # Token cap the STM summarizer is asked to keep its rolling summary under.
    # None = derive it from max_tokens_stm (35% of it, min 300), the historical
    # default. Set it to take charge of the summary budget yourself.
    max_summary_tokens: int | None = None
    # --- LTM/worker behavior -----------------------------------------------------
    agent_loop_max_steps: int = 12
    atexit_timeout: float = 30.0
    # --- misc ----------------------------------------------------------------------
    # Derived in __post_init__: the memory set this instance owns (storage_dir
    # / user_id, when a user_id was given). Every directory below hangs off it,
    # so a different user_id transparently switches to a separate memory set.
    root_dir: Path = field(init=False)
    # None means "generate one": the facade forwards its own ``session_id=None``
    # default straight through, and a dataclass default_factory only fires when
    # the argument is omitted — so an explicit None would otherwise reach
    # validation and be rejected as a non-leaf name.
    session_id: str = field(default_factory=_generate_session_id)
    # Names the per-user memory root: Memory(dir, user_id="alice") keeps this
    # user's whole memory set (ltm/ + stm/) under dir/alice/, so one storage
    # dir can hold many users' memories side by side. None keeps the flat
    # layout (everything directly under storage_dir).
    user_id: str | None = None
    retries: int = 2
    logger: logging.Logger | None = None

    def __post_init__(self) -> None:
        self.storage_dir = Path(self.storage_dir)
        # Treat an explicit None as "not given" — that is what the type says it
        # is, and the facade always passes the argument along.
        if self.session_id is None:
            self.session_id = _generate_session_id()
        # session_id becomes the transcript's file name (stm/<session_id>.md): a
        # value with separators or traversal could escape storage_dir via our
        # own writes, bypassing the MemoryTools path guard. Keep it a leaf name.
        _validate_leaf(self.session_id, "session_id")
        # user_id becomes a real directory name, so it needs the same guard.
        # int is accepted for convenience (user_id=1) and stored as str.
        if self.user_id is not None:
            if isinstance(self.user_id, bool):
                raise ValueError(f"user_id must be a string or int, got {self.user_id!r}")
            if isinstance(self.user_id, int):
                self.user_id = str(self.user_id)
            _validate_leaf(self.user_id, "user_id")
        # The memory root everything else hangs off: per-user subdir when a
        # user_id was given, otherwise storage_dir itself. It is *created* only
        # after every check below passes, so a rejected config cannot leave an
        # empty directory behind on disk.
        self.root_dir = (self.storage_dir / self.user_id
                         if self.user_id else self.storage_dir)
        if not callable(self.llm):
            raise ValueError("llm must be callable: llm(messages) -> str")
        if self.llm_tools is not None and not callable(self.llm_tools):
            raise ValueError("llm_tools must be callable: llm_tools(messages, tools) -> assistant msg")
        for name in ("max_tokens_stm", "max_tokens_ltm"):
            value = getattr(self, name)
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if not isinstance(self.msg_overhead, int) or self.msg_overhead < 0:
            raise ValueError(f"msg_overhead must be a non-negative int, got {self.msg_overhead!r}")
        if self.token_counter is not None and not callable(self.token_counter):
            raise ValueError("token_counter must be callable: token_counter(text) -> int")
        if self.max_summary_tokens is not None and (
                not isinstance(self.max_summary_tokens, int)
                or isinstance(self.max_summary_tokens, bool)
                or self.max_summary_tokens <= 0):
            raise ValueError(
                f"max_summary_tokens must be a positive int or None, "
                f"got {self.max_summary_tokens!r}")
        if not 0.0 < self.keep_recent < 1.0:
            raise ValueError(f"keep_recent must be in (0, 1), got {self.keep_recent!r}")
        if not isinstance(self.retries, int) or self.retries < 0:
            raise ValueError(f"retries must be a non-negative int, got {self.retries!r}")
        if self.atexit_timeout <= 0:
            raise ValueError(f"atexit_timeout must be > 0, got {self.atexit_timeout!r}")
        if not isinstance(self.agent_loop_max_steps, int) or self.agent_loop_max_steps < 1:
            raise ValueError("agent_loop_max_steps must be a positive int")
        if self.logger is None:
            self.logger = logging.getLogger("memkit")
        self.root_dir.mkdir(parents=True, exist_ok=True)

    @property
    def ltm_dir(self) -> Path:
        return self.root_dir / "ltm"

    @property
    def stm_dir(self) -> Path:
        return self.root_dir / "stm"

    @property
    def session_file(self) -> Path:
        """The ONE transcript file for this conversation: ``stm/<session_id>.md``.

        Every summarization appends to this file, so a session has a single,
        stable, predictable place its exact past text can be recovered from.
        """
        return self.stm_dir / f"{self.session_id}.md"
