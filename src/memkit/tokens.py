"""Token counting built on tiktoken.

Counts OpenAI-style message dicts, including tool_calls payloads, plus a
per-message msg_overhead to approximate chat-format framing. tiktoken counts
are approximate for non-OpenAI models; pass token_counter=your_fn to the
Memory constructor to use the exact tokenizer of another provider.
"""

from __future__ import annotations

import json
from typing import Any, Callable

import tiktoken


class TokenCounter:
    def __init__(
        self,
        encoding: str = "o200k_base",
        msg_overhead: int = 4,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        if token_counter is not None:
            self._count_text: Callable[[str], int] = token_counter
        else:
            enc = tiktoken.get_encoding(encoding)

            def _count(text: str, _enc=enc) -> int:
                # disallowed_special=() so a stray special-token string in user
                # text counts as plain text instead of raising.
                return len(_enc.encode(text, disallowed_special=()))

            self._count_text = _count
        self.msg_overhead = msg_overhead

    def count_text(self, text: str) -> int:
        return self._count_text(text or "")

    def count_message(self, msg: dict[str, Any]) -> int:
        """Tokens for one OpenAI-style message dict, framing overhead included."""
        total = self.msg_overhead
        content = msg.get("content")
        if isinstance(content, str):
            total += self._count_text(content)
        elif content is not None:
            total += self._count_text(str(content))
        # tool_calls: name + JSON-serialized arguments are what the model sees.
        for call in msg.get("tool_calls") or []:
            total += self._count_text(str(call.get("name", "")))
            args = call.get("arguments")
            if isinstance(args, dict):
                args = json.dumps(args, ensure_ascii=False, sort_keys=True)
            elif args is not None:
                args = str(args)
            total += self._count_text(args or "")
        # a tool result message's own name/role adds a little; already covered
        # by content above.
        return total

    def count_messages(self, msgs: list[dict[str, Any]]) -> int:
        return sum(self.count_message(m) for m in msgs)

    def truncate(self, text: str, budget: int) -> str:
        """Cut `text` to at most `budget` tokens, marker included (from the start).

        Used only as a last resort; the un-truncated original always remains
        in the STM transcript file for recall_transcript recovery.
        """
        if budget <= 0:
            return ""
        if self._count_text(text) <= budget:
            return text
        marker = "\n[... truncated; full text in archive ...]"
        # Reserve the marker so the RESULT stays within budget, not text+marker.
        room = budget - self._count_text(marker)
        if room <= 0:
            return ""  # no room for any real text beside the marker
        # Binary search on character cut points (works for any tokenizer).
        lo, hi = 0, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._count_text(text[:mid]) <= room:
                lo = mid
            else:
                hi = mid - 1
        return text[:lo] + marker
