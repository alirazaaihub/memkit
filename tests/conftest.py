"""Shared fixtures: scripted fake LLMs (no network, no keys).

Thread-safe: memkit calls the plain llm from the main thread (STM
summarization) and from the worker thread (LTM extraction) concurrently.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest


class ScriptedLLM:
    """Plain llm(messages)->str that pops canned replies in order.

    Records every call so tests can assert ordering (e.g. archive written
    BEFORE the first summarizer call). If replies run out it raises — unless
    repeat_last=True, in which case the final reply is reused forever.
    """

    def __init__(self, replies: list[str], repeat_last: bool = False) -> None:
        self.replies = list(replies)
        self.repeat_last = repeat_last
        self._last = replies[-1] if replies else ""
        self.calls: list[list[dict[str, Any]]] = []
        self._lock = threading.Lock()

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        with self._lock:
            self.calls.append(messages)
            if not self.replies:
                if self.repeat_last:
                    return self._last
                raise AssertionError("ScriptedLLM ran out of replies")
            self._last = self.replies.pop(0)
            return self._last

    @property
    def call_count(self) -> int:
        return len(self.calls)


class ScriptedToolsLLM:
    """llm_tools(messages, tools) -> normalized assistant message.

    ``script`` is a list of assistant turns::

        {"content": str|None, "tool_calls": [{"name","arguments"}] | None}

    ids are auto-assigned. Raises if the script runs dry — unless
    repeat_last=True, in which case the final turn is reused forever
    (that turn must end the loop: no tool_calls).
    """

    def __init__(self, script: list[dict[str, Any]],
                 repeat_last: bool = False) -> None:
        self.script = list(script)
        self.repeat_last = repeat_last
        self._last = script[-1] if script else None
        if repeat_last and self._last is not None and self._last.get("tool_calls"):
            raise ValueError("repeat_last turn must have no tool_calls "
                             "(else it would loop forever)")
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []
        self._id_seq = 0
        self._lock = threading.Lock()

    def __call__(self, messages, tools):
        with self._lock:
            self.calls.append((messages, tools))
            if not self.script:
                if self.repeat_last and self._last is not None:
                    turn = self._last
                else:
                    raise AssertionError("ScriptedToolsLLM ran out of script turns")
            else:
                turn = self.script.pop(0)
            calls = []
            for tc in turn.get("tool_calls") or []:
                self._id_seq += 1
                calls.append({"id": f"t{self._id_seq}", "name": tc["name"],
                              "arguments": tc.get("arguments", {})})
        return {"role": "assistant", "content": turn.get("content"),
                "tool_calls": calls}


@pytest.fixture
def fake_llm():
    return ScriptedLLM([])
