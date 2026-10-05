"""Mini tool-calling loop used by LTM consolidation (and available to users).

Speaks the normalized message format from llm.py: the loop sends OpenAI-style
messages to ``llm_tools(messages, tools)``, receives
``{"role": "assistant", "content": ..., "tool_calls": [...]}``, executes each
call through :class:`memkit.tools.MemoryTools`, appends tool results, and
repeats until the model returns no tool calls or the step cap is hit.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from memkit.llm import call_with_retries
from memkit.tools import MemoryTools, read_specs

CONSOLIDATION_TOOLS = ["list_memory_files", "read_memory_file",
                       "search_memory_lines", "edit_memory_file",
                       "write_memory_file", "delete_memory_file"]


def run_agent_loop(
    *,
    llm_tools: Callable[[list[dict[str, Any]], list[dict[str, Any]]], dict[str, Any]],
    tools: MemoryTools,
    system: str,
    user: str,
    max_steps: int,
    logger: logging.Logger,
    tool_names: list[str] | None = None,
    retries: int = 0,
) -> str:
    """Run the loop to completion; returns the final assistant text (if any).

    Raises if the LLM call itself fails after retries — callers decide how to
    recover (memkit's consolidation returns facts to the inbox).
    """
    specs = read_specs(tool_names if tool_names is not None else CONSOLIDATION_TOOLS)
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    final_text = ""
    for step in range(max_steps):
        assistant = call_with_retries(
            lambda: llm_tools(messages, specs), retries, logger)
        if not isinstance(assistant, dict):
            raise TypeError(
                "llm_tools must return a dict like "
                '{"role":"assistant","content":str|None,"tool_calls":[...]}')
        tool_calls = assistant.get("tool_calls") or []
        messages.append({"role": "assistant",
                         "content": assistant.get("content"),
                         "tool_calls": tool_calls})
        if assistant.get("content"):
            final_text = assistant["content"]
        if not tool_calls:
            return final_text
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("arguments") or {}
            if not isinstance(args, dict):
                args = {"_raw": args}
            result = tools.execute(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": result,
            })
    logger.warning(
        "memkit: agent loop hit max_steps=%d before finishing; state on disk "
        "is whatever the last completed tool call left (writes are atomic).",
        max_steps)
    return final_text
