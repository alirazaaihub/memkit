"""Adapter tests: memkit's normalized shapes <-> provider wire formats.

No SDKs, no network: adapters accept an injected ``client``, so a tiny fake
records exactly what would have been sent.
"""
from __future__ import annotations

import json

from memkit.llm import OpenAILLM


# -- fakes ---------------------------------------------------------------------

class _FakeCompletions:
    def __init__(self, message):
        self._message = message
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Resp", (), {"choices": [type("Ch", (), {"message": self._message})]})()


class _FakeClient:
    """Mimics client.chat.completions.create(**kwargs)."""

    def __init__(self, message):
        self.completions = _FakeCompletions(message)
        self.chat = type("Chat", (), {"completions": self.completions})()


def _assistant_message(content=None, tool_calls=None):
    return type("Msg", (), {"content": content, "tool_calls": tool_calls})()


# -- outbound conversion: memkit-normalized history -> OpenAI wire -------------

def test_with_tools_serializes_normalized_history_as_wire_format():
    """The bug: memkit hands the model history in ITS normalized shape
    ({"id","name","arguments":dict}), which the provider rejects as
    `messages.N.tool_calls.0.type was rejected as invalid`. Outbound history
    must be converted back to OpenAI's wire shape."""
    client = _FakeClient(_assistant_message(content="done"))
    llm = OpenAILLM(model="m", client=client)

    history = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "find my poem"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "name": "search_lines",
             "arguments": {"query": "Roses are red"}}]},
        {"role": "tool", "tool_call_id": "c1", "name": "search_lines",
         "content": "stm/s.md:4: Roses are red"},
    ]
    llm.with_tools(history, tools=[{"type": "function", "function": {"name": "x"}}])

    sent = client.completions.calls[0]["messages"]
    assistant = next(m for m in sent if m.get("tool_calls"))
    call = assistant["tool_calls"][0]

    assert call["type"] == "function", f"missing wire 'type': {call!r}"
    assert call["id"] == "c1"
    assert call["function"]["name"] == "search_lines"
    assert json.loads(call["function"]["arguments"]) == {"query": "Roses are red"}
    assert isinstance(call["function"]["arguments"], str), \
        "OpenAI wants arguments as a JSON string, not a dict"
    assert "name" not in call, "normalized 'name' leaked into wire shape"


def test_tool_role_message_drops_the_internal_name_field():
    """OpenAI's `tool` message schema is {role, tool_call_id, content} only;
    memkit's internal `name` key is not part of it and must not be sent."""
    client = _FakeClient(_assistant_message(content="ok"))
    llm = OpenAILLM(model="m", client=client)

    llm([{"role": "tool", "tool_call_id": "c1", "name": "search_lines",
          "content": "hit"}])

    assert client.completions.calls[0]["messages"][0] == {
        "role": "tool", "tool_call_id": "c1", "content": "hit"}


def test_already_wire_shaped_history_is_left_intact():
    """A caller who hand-built wire-shaped history must not be double-wrapped."""
    client = _FakeClient(_assistant_message(content="ok"))
    llm = OpenAILLM(model="m", client=client)

    wire = {"id": "c1", "type": "function",
            "function": {"name": "read_file", "arguments": '{"file_path": "a.md"}'}}
    llm([{"role": "assistant", "content": None, "tool_calls": [wire]}])

    call = client.completions.calls[0]["messages"][0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] == "read_file"
    assert json.loads(call["function"]["arguments"]) == {"file_path": "a.md"}


def test_plain_call_also_serializes_normalized_history_as_wire_format():
    """Same defect on the plain-completion path: history a caller replays
    through ``llm(messages)`` must be converted too."""
    client = _FakeClient(_assistant_message(content="hi"))
    llm = OpenAILLM(model="m", client=client)

    history = [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c9", "name": "read_file",
             "arguments": {"file_path": "stm/s.md"}}]},
        {"role": "tool", "tool_call_id": "c9", "name": "read_file",
         "content": "body"},
    ]
    llm(history)

    sent = client.completions.calls[0]["messages"]
    call = sent[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert json.loads(call["function"]["arguments"]) == {"file_path": "stm/s.md"}