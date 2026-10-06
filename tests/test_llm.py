"""Adapter tests: memkit's normalized shapes <-> provider wire formats.

No SDKs, no network: adapters accept an injected ``client``, so a tiny fake
records exactly what would have been sent.
"""
from __future__ import annotations

import json

from memkit.llm import AnthropicLLM, OpenAILLM


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


# -- inbound conversion: memkit-normalized history -> Anthropic ----------------

def test_anthropic_tool_result_after_a_plain_user_message():
    """The bug: a tool result whose preceding turn is an ordinary user TEXT
    message crashed the conversion with `'str' object has no attribute 'get'`
    — the merge branch assumed the previous turn's content was a block list.
    It must start a new user turn instead of merging into (or crashing on) a
    string-content message."""
    _, converted = AnthropicLLM._to_anthropic(None, [
        {"role": "user", "content": "what is in the file?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "name": "read_memory_file",
             "arguments": {"file_path": "ltm/user.md"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "line 1"},
        {"role": "tool", "tool_call_id": "c2", "content": "line 2"},
    ])

    assert converted[0] == {"role": "user", "content": "what is in the file?"}
    # both results land in ONE tool_result turn, never appended to the text turn
    results = converted[-1]
    assert results["role"] == "user"
    assert [b["type"] for b in results["content"]] == ["tool_result", "tool_result"]
    assert [b["tool_use_id"] for b in results["content"]] == ["c1", "c2"]
    assert all(isinstance(b["content"], str) for b in results["content"])


def test_anthropic_tool_results_still_merge_into_one_turn():
    """The merge the crash above was guarding: consecutive tool results with NO
    intervening user text must still collapse into a single user message, which
    is what the Messages API expects."""
    _, converted = AnthropicLLM._to_anthropic(None, [
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "name": "a", "arguments": {}},
            {"id": "c2", "name": "b", "arguments": {}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "r1"},
        {"role": "tool", "tool_call_id": "c2", "content": "r2"},
    ])

    assert len(converted) == 2, converted
    assert [b["tool_use_id"] for b in converted[1]["content"]] == ["c1", "c2"]


def test_anthropic_error_tool_result_is_flagged():
    """A failed tool result keeps its is_error flag through the merge."""
    _, converted = AnthropicLLM._to_anthropic(None, [
        {"role": "tool", "tool_call_id": "c1", "content": "Error: no such file"},
    ])

    assert converted[0]["content"][0]["is_error"] is True