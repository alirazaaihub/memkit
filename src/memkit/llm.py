"""Provider-agnostic LLM callables and ready-made adapters.

memkit talks to models through two duck-typed callables (the shapes the user
supplies at construction time):

* ``llm(messages) -> str`` — plain completion. ``messages`` are OpenAI-style
  dicts: ``{"role": "system"|"user"|"assistant"|"tool", "content": str, ...}``.
* ``llm_tools(messages, tools) -> dict`` — one turn with function tools. Must
  return the *normalized assistant message*::

      {"role": "assistant",
       "content": str | None,
       "tool_calls": [{"id": str, "name": str, "arguments": dict}]}

  A message with an empty ``tool_calls`` list means "the agent is done".

``OpenAILLM`` and ``AnthropicLLM`` adapt those SDKs to both shapes. The SDKs
are imported lazily, so ``memkit`` core needs neither installed. Each adapter
is itself the ``llm`` callable (``__call__``) and exposes ``.with_tools`` as
the ``llm_tools`` callable::

    from memkit import Memory, OpenAILLM
    llm = OpenAILLM(model="gpt-4o-mini")
    mem = Memory("./mem", llm=llm, max_tokens_stm=4000, max_tokens_ltm=8000)
    # llm_tools is picked up automatically from llm.with_tools

You are NOT limited to memkit's adapters — initialize any LLM yourself and
pass that variable in:

* any callable ``llm(messages) -> str`` (or ``llm_tools(messages, tools)``)
  works as-is — wrap your provider however you like;
* a ready-made SDK *client* object also works: ``Memory(dir, llm=client)``
  where ``client`` is e.g. ``OpenAI(model="gpt-4o-mini", api_key=...)`` or
  ``Anthropic()``. :func:`adapt_llm` recognizes it by its API surface
  (``client.chat.completions`` / ``client.messages``), reads the model name
  off the client when possible, and wraps it in the matching adapter.
"""

from __future__ import annotations

import json
import uuid
from typing import Any, Callable

__all__ = ["OpenAILLM", "AnthropicLLM", "call_with_retries",
           "messages_to_openai", "adapt_llm"]


def _new_call_id() -> str:
    return f"memkit-{uuid.uuid4().hex[:12]}"


def call_with_retries(fn: Callable[[], Any], retries: int, logger) -> Any:
    """Call ``fn``, retrying transient failures. Raises the last error."""
    last: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as e:  # provider errors are wildly typed; catch all
            last = e
            logger.warning("memkit: LLM call failed (attempt %d/%d): %s",
                           attempt + 1, retries + 1, e)
    raise last  # type: ignore[misc]


def _normalize_tool_arguments(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {}
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {"_raw": raw}
        if isinstance(parsed, dict):
            return parsed
        return {"_raw": parsed}
    return {"_raw": str(raw)}


def _wire_tool_call(tc: dict[str, Any]) -> dict[str, Any]:
    """Normalized tool call -> OpenAI wire tool call.

    memkit's canonical shape is ``{"id", "name", "arguments": dict}``; the
    provider wants ``{"id", "type": "function", "function": {"name",
    "arguments": "<json string>"}}``. Without this, replaying any history that
    contains a tool call fails with
    ``messages.N.tool_calls.0.type was rejected as invalid``.
    """
    if "function" in tc:  # already wire-shaped (idempotent)
        call = dict(tc)
        call.setdefault("type", "function")
        fn = dict(call["function"])
        if not isinstance(fn.get("arguments"), str):
            fn["arguments"] = json.dumps(fn.get("arguments") or {})
        call["function"] = fn
        return call
    return {
        "id": tc.get("id") or _new_call_id(),
        "type": "function",
        "function": {
            "name": tc.get("name", ""),
            "arguments": json.dumps(tc.get("arguments") or {}),
        },
    }


def messages_to_openai(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert memkit's normalized history to OpenAI wire messages.

    memkit normalizes provider tool calls on the way IN (``arguments`` as a
    dict, ``name`` at the top level); the chat-completions API needs the wire
    shape back on the way OUT. Applied to whatever a caller replays from
    ``context()`` or builds in its own agent loop. Messages without tool calls
    pass through untouched.
    """
    out: list[dict[str, Any]] = []
    for m in messages:
        tc = m.get("tool_calls")
        if not tc:
            # memkit keeps ``name`` on tool results for its own transcript; the
            # API's tool message is {role, tool_call_id, content} only.
            if m.get("role") == "tool" and "name" in m:
                out.append({k: v for k, v in m.items() if k != "name"})
            else:
                out.append(m)
            continue
        converted = dict(m)
        converted["tool_calls"] = [_wire_tool_call(c) for c in tc]
        out.append(converted)
    return out


class OpenAILLM:
    """Adapter for OpenAI (and OpenAI-compatible) chat-completions APIs."""

    def __init__(self, model: str, client: Any | None = None, **create_kwargs: Any) -> None:
        if client is None:
            from openai import OpenAI  # lazy: core does not depend on the SDK
            client = OpenAI()
        self.client = client
        self.model = model
        self.create_kwargs = create_kwargs

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages_to_openai(messages),
            **self.create_kwargs
        )
        return resp.choices[0].message.content or ""

    def with_tools(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]]) -> dict[str, Any]:
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages_to_openai(messages), tools=tools,
            **self.create_kwargs
        )
        msg = resp.choices[0].message
        tool_calls = []
        for tc in msg.tool_calls or []:
            tool_calls.append({
                "id": tc.id or _new_call_id(),
                "name": tc.function.name,
                "arguments": _normalize_tool_arguments(tc.function.arguments),
            })
        return {"role": "assistant", "content": msg.content or None, "tool_calls": tool_calls}


class AnthropicLLM:
    """Adapter for the Anthropic Messages API using the official ``anthropic`` SDK."""

    def __init__(self, model: str, client: Any | None = None,
                 max_tokens: int = 4096, **create_kwargs: Any) -> None:
        if client is None:
            import anthropic  # lazy
            client = anthropic.Anthropic()
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.create_kwargs = create_kwargs

    # -- internal (OpenAI-style) -> Anthropic conversion ---------------------

    @staticmethod
    def _specs_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out = []
        for t in tools:
            f = t.get("function", t)
            out.append({
                "name": f["name"],
                "description": f.get("description", ""),
                "input_schema": f.get("parameters")
                or {"type": "object", "properties": {}},
            })
        return out

    def _to_anthropic(self, messages: list[dict[str, Any]]):
        system_parts: list[str] = []
        converted: list[dict[str, Any]] = []
        for m in messages:
            role = m.get("role")
            content = m.get("content")
            if role == "system":
                system_parts.append(content if isinstance(content, str) else str(content or ""))
            elif role == "assistant":
                blocks: list[dict[str, Any]] = []
                if isinstance(content, str) and content:
                    blocks.append({"type": "text", "text": content})
                for tc in m.get("tool_calls") or []:
                    blocks.append({
                        "type": "tool_use",
                        "id": tc.get("id") or _new_call_id(),
                        "name": tc.get("name", ""),
                        "input": _normalize_tool_arguments(tc.get("arguments")),
                    })
                if not blocks:
                    blocks.append({"type": "text", "text": "(empty)"})
                converted.append({"role": "assistant", "content": blocks})
            elif role == "tool":
                text = content if isinstance(content, str) else str(content or "")
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": m.get("tool_call_id") or "",
                    "content": text,
                }
                if text.startswith("Error:"):
                    block["is_error"] = True
                # Anthropic wants all tool_results for a turn in ONE user message.
                if converted and converted[-1]["role"] == "user" and \
                        converted[-1]["content"][0].get("type") == "tool_result":
                    converted[-1]["content"].append(block)
                else:
                    converted.append({"role": "user", "content": [block]})
            else:  # user (or anything the provider can still read as user)
                converted.append({
                    "role": "user",
                    "content": content if isinstance(content, str) else str(content or ""),
                })
        return ("\n\n".join(system_parts) or None), converted

    def _create(self, messages: list[dict[str, Any]],
                tools: list[dict[str, Any]] | None):
        system, converted = self._to_anthropic(messages)
        if not converted:
            raise ValueError("Anthropic needs at least one non-system message")
        kwargs: dict[str, Any] = dict(self.create_kwargs)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._specs_to_anthropic(tools)
        return self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens,
            messages=converted, **kwargs,
        )

    # -- public callable surface ----------------------------------------------

    def __call__(self, messages: list[dict[str, Any]]) -> str:
        resp = self._create(messages, tools=None)
        return "".join(b.text for b in resp.content if b.type == "text")

    def with_tools(self, messages: list[dict[str, Any]],
                   tools: list[dict[str, Any]]) -> dict[str, Any]:
        resp = self._create(messages, tools=tools)
        text = "".join(b.text for b in resp.content if b.type == "text") or None
        tool_calls = [
            {"id": b.id or _new_call_id(), "name": b.name,
             "arguments": _normalize_tool_arguments(b.input)}
            for b in resp.content if b.type == "tool_use"
        ]
        return {"role": "assistant", "content": text, "tool_calls": tool_calls}


# -- accept any LLM the user initialized -------------------------------------

# Common attribute names SDK clients use to remember the model they were built
# with; the first one present wins.
_MODEL_ATTRS = ("model", "model_name", "default_model", "deployment_name")


def _client_model(client: Any) -> str | None:
    for attr in _MODEL_ATTRS:
        value = getattr(client, attr, None)
        if isinstance(value, str) and value:
            return value
    return None


def _client_kwargs(client: Any, adapter: type) -> dict[str, Any]:
    """Extra create()-kwargs the client may carry (a client set with
    ``temperature=...`` should keep using it). Unknown keys are dropped."""
    out: dict[str, Any] = {}
    import inspect
    valid = set(inspect.signature(adapter.__init__).parameters)
    for key in ("max_tokens", "temperature", "top_p", "timeout", "stop"):
        value = getattr(client, key, None)
        if value is not None and key in valid:
            out[key] = value
    return out


def adapt_llm(obj: Any, logger: Any = None) -> Any:
    """Turn whatever the user passed into something memkit can call.

    Callables (memkit adapters, the user's own wrapper functions, model
    objects with ``__call__``) are returned untouched — memkit just calls them.

    A non-callable *client* object — the instance a provider SDK gives back,
    e.g. ``OpenAI(model="gpt-4o-mini", api_key=...)`` or ``Anthropic()`` — is
    wrapped in the matching adapter, with the model name and any create kwargs
    read off the client. So a user can initialize their LLM exactly the way
    their provider documents and hand memkit that one variable.

    Anything else is returned as-is; the config's callable check produces the
    error, so the message stays in one place.
    """
    if callable(obj):
        return obj
    if isinstance(obj, str) or obj is None:
        return obj
    # OpenAI-style client: exposes .chat.completions.create
    chat = getattr(obj, "chat", None)
    if chat is not None and hasattr(getattr(chat, "completions", None), "create"):
        model = _client_model(obj)
        if not model:
            raise ValueError(
                "the client passed as llm has no readable model name — either "
                "set it on the client (OpenAI(model=...)) or wrap it: "
                "memkit.OpenAILLM(model='...', client=that_client)")
        if logger is not None:
            logger.info("memkit: adapting OpenAI-style client (model=%s)", model)
        return OpenAILLM(model=model, client=obj,
                         **_client_kwargs(obj, OpenAILLM))
    # Anthropic-style client: exposes .messages.create
    if hasattr(getattr(obj, "messages", None), "create"):
        model = _client_model(obj)
        if not model:
            raise ValueError(
                "the client passed as llm has no readable model name — either "
                "set it on the client (Anthropic(model=...)) or wrap it: "
                "memkit.AnthropicLLM(model='...', client=that_client)")
        if logger is not None:
            logger.info("memkit: adapting Anthropic-style client (model=%s)", model)
        return AnthropicLLM(model=model, client=obj,
                            **_client_kwargs(obj, AnthropicLLM))
    return obj
