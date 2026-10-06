# memkit

A reusable **memory framework** for chatbots and agents — short-term memory
(STM) with lossless archiving, and long-term memory (LTM) as a file-based,
Claude-Code-style markdown tree. Add both to any agent in a few lines.

```python
from memkit import Memory

mem = Memory(
    storage_dir="./agent_memory",
    llm=my_llm,                  # ANY LLM you initialized: a callable, or a
                                 #   provider client object (see LLM adapters)
    max_tokens_stm=4000,         # REQUIRED — your choice, no default
    max_tokens_ltm=8000,         # REQUIRED — your choice, no default
    user_id="alice",             # optional — one separate memory set per user
)

mem.add("user", "hello")
messages = [{"role": "system", "content": mem.system_prompt_fragment()}] \
           + mem.context()
# ... run your own agent loop; record turns with mem.add(...) / mem.add_many(...)
mem.close()                      # deterministic flush (atexit is wired too)
```

## How it works

### Short-term memory (per conversation)

* `context()` returns the working window: rolling summary + recent turns
  verbatim + any **pending tool call** at the end of the window (an assistant
  `tool_calls` message is never separated from its results, and an unresolved
  call stays verbatim while it is the newest thing in the window; once the
  conversation moves past it, the abandoned call is archived and summarized
  like any other text — its exact form stays recoverable from the archive).
* **Summarization is asynchronous.** When the window exceeds
  `max_tokens_stm`, `add()` only appends and signals a background summarizer
  thread — it never waits on an LLM, so your reply latency is unaffected. The
  cost is that `context()` never waits either: while a summary is in flight the
  window can sit one batch over the cap (in a real chat a turn arrives far
  slower than one summary call, so this is invisible). Need the window
  deterministically compacted? `mem.stm.wait_until_settled()` blocks until no
  compression is pending (used by tests and by `close()`).
* **On-demand compaction: `compact()`.** The cap is a background policy, not a
  restriction on you — `mem.compact(timeout)` summarizes right now, whether or
  not the window is full, and returns once it settled. It is what a
  user-facing `/compact` command calls (see `main.py`). Unlike a cap-driven
  pass it ignores `keep_recent` and keeps only the newest turn verbatim, so a
  summary always appears; a window already down to one turn is a no-op. The
  folded text is still recoverable verbatim through `recall_transcript`, and
  the archive feeds LTM extraction exactly as a cap-driven summary does.
* The summarizer **first** appends the raw outgoing turns to the
  conversation's one transcript file, `stm/<session_id>.md` (`with open` +
  fsync — if the write fails, summarization is skipped, so text is never
  summarized away without a copy on disk), then asks your summarizer LLM to
  compress that span.
* **One file per conversation, append-only.** Each conversation gets a fresh
  random `session_id` (or the one you pin); every summarization *appends* to
  that same file — never a new randomly-named archive — so there is one
  predictable place the conversation's exact text lives. Line numbers `[n]`
  are the file's true line numbers across appends.
* **One dedicated recovery tool: `recall_transcript`.** Its only job is to
  view *this* conversation's transcript (the file named by its session UUID)
  and return the exact original text summarization dropped — verbatim line
  windows around a case-sensitive match, or the recent tail when asked with an
  empty query. It reads no other file, and the system prompt requires the
  agent to answer any word-for-word request from it, never from the lossy
  summary. No extra plumbing: the enforcement is the tool description plus the
  prompt, exactly as designed.
* **Reads are scoped to your own conversation.** The agent may read the shared
  `ltm/` tree and *its own* transcript; other sessions' transcripts are
  refused, unlisted, and never searchable — `recall_transcript` has no
  parameter that could point it at another conversation. Running several
  chatbots over one memory root therefore cannot leak one user's chat into
  another's context.
* The agent is told (via `system_prompt_fragment()`) that when the user asks
  for something word-for-word (the poem from earlier, an exact quote, a config
  value), it MUST call `recall_transcript` and reproduce what comes back —
  never paraphrase the summary, never rewrite from memory. Nothing is thrown
  away; only moved to disk.

### Long-term memory (shared across sessions)

* `ltm/MEMORY.md` is the index — one line per topic file, injected into the
  system prompt every turn, exactly like Claude Code's memory:
  `- [user.md] — likes terse answers`.
* After every STM archive, a **background worker thread** runs an extraction
  LLM over the newly archived text; accepted facts (`USER_FACTS /
  PROJECT_FACTS / FEEDBACK / REFERENCES`) are appended to topic files without
  ever blocking your agent.
* When the whole tree exceeds `max_tokens_ltm`, a **tool-calling agent loop**
  autonomously merges, prunes and rewrites topic files and the index.
* Session end (explicit `close()` or process exit) triggers a final
  extraction, so even short sessions that never hit a cap still leave memories.

## Multiple users in one storage dir

`user_id` names the folder that holds one person's **whole memory set**
(`ltm/` + `stm/`). With it, the layout is `storage_dir/<user_id>/...`; without
it, everything sits directly under `storage_dir` (the flat layout):

```
./agent_memory/
├── alice/
│   ├── ltm/      MEMORY.md, user.md, ...      # alice's memories
│   └── stm/      <session_id>.md              # alice's transcripts
└── bob/
    ├── ltm/      ...                          # bob's memories, fully separate
    └── stm/
```

```python
mem = Memory("./agent_memory", llm=llm, user_id="alice", ...)   # alice's set
mem = Memory("./agent_memory", llm=llm, user_id="bob",   ...)   # bob's set
```

Switching is only a matter of changing `user_id` (an int like `user_id=1`
works too). The memory tools are confined to the active user's root, so one
user's agent cannot list, search, read, or write another user's memories —
different `user_id`s are different worlds, sharing nothing but a parent
directory. `mem.memory_root` returns the active set's directory.

## The memory tools

**These are memory tools, not file tools.** Every one of them can only see
inside the active memory root — never project files, never anything else on
the machine — and all of them are named `*_memory_*` (or `recall_transcript`)
precisely so they cannot be confused with the generic file tools a host
project may already offer. The LLM is told this both in every tool
description and at the top of the prompt fragment.

`mem.tools()` returns OpenAI-format specs for the four read tools — wire them
into your agent and route calls to `mem.execute_tool(name, arguments)`:

```python
recall_transcript(query="", window=20, max_matches=5)
    # the ONE verbatim-recovery tool: this conversation's transcript only;
    # case-sensitive exact text around each match (empty query = recent tail)
read_memory_file(file_path, offset=0, limit=100)
    # paginated read of one memory file; lines numbered from offset+1
search_memory_lines(query, file_path=None, max_matches=20)
    # -> "path:lineno: text" across the ltm/ tree + own transcript
list_memory_files()
    # every memory file this agent may read, with sizes
```

Writes go through three more memory tools that belong to memkit's own
background consolidation loop and are refused to user agents
(`edit_memory_file`, `write_memory_file`, `delete_memory_file`).
`edit_memory_file(file_path, old_text, new_text, replace_all=False)` is the
surgical one: exact-string replacement inside an existing `ltm/*.md` memory —
it must match exactly once unless `replace_all` — while
`write_memory_file` creates files or rewrites them whole and
`delete_memory_file` retires a topic file (never `MEMORY.md`, the index).
`user_id` is validated as a simple directory name and `session_id` as a file
name, so neither can be used to escape the memory folder.

## Install (uv)

```bash
git clone <this repo> && cd claudee
uv sync --extra openai --extra anthropic   # core needs only tiktoken; extras are for adapters
uv run pytest -q                           # offline test suite, no keys needed
```

Then `uv add memkit` equivalent: `uv pip install -e .` in your own project, or
add it as a path/git dependency.

## Bring ANY LLM you initialized yourself

You are not locked to memkit's adapters. Initialize your model exactly the way
your provider documents, and pass that one variable in — memkit's `adapt_llm`
accepts three shapes:

```python
# 1. Your own callable — any function or class instance: llm(messages) -> str
def my_llm(messages):
    return my_provider.complete(messages)
mem = Memory("./memory", llm=my_llm, max_tokens_stm=4000, max_tokens_ltm=8000)

# 2. A provider client OBJECT you constructed yourself — recognized by its
#    API surface (.chat.completions / .messages) and wrapped automatically.
#    The model name is read off the client (model / model_name /
#    deployment_name); if your client holds no default model, memkit says so
#    and you pass the wrapped form — still YOUR client object:
#        memkit.OpenAILLM(model="gpt-4o-mini", client=my_client)
from openai import OpenAI
llm = OpenAI(model="gpt-4o-mini", api_key="...")   # your init, not memkit's
mem = Memory("./memory", llm=llm, max_tokens_stm=4000, max_tokens_ltm=8000)

# 3. Any OTHER provider (Gemini, Ollama, vLLM, one you wrote): wrap its
#    client in one small function — or pass the client if it is callable —
#    and memkit calls it like any llm.
```

Without a tool-calling callable, memkit does **not** fail: extraction and
memory persistence keep working, and only semantic merging/pruning
(consolidation) is skipped with a warning — pass `llm_tools=...` (an explicit
`llm_tools(messages, tools) -> assistant dict`) whenever your tool-capable
caller differs from the plain one, and consolidation turns back on.

Built-in adapters, if you prefer them, add tool-calling on top:

```python
# OpenAI (also works with any OpenAI-compatible endpoint, e.g. a proxy)
from openai import OpenAI
from memkit import Memory, OpenAILLM

client = OpenAI(base_url="https://tokenharbor.ai/v1", api_key=os.environ["KEY"])
llm = OpenAILLM(model="qwen3.8-flash:free", client=client)

mem = Memory("./memory", llm=llm,              # llm.with_tools is auto-detected
             max_tokens_stm=4000, max_tokens_ltm=8000)

# Anthropic
from memkit import AnthropicLLM
llm = AnthropicLLM(model="claude-sonnet-4-6")
```

Both adapters accept history in memkit's **normalized** shape (`tool_calls`
entries carry `name` plus `arguments` as a dict — the same shape `context()`
returns), so you can replay a window containing past tool calls straight back
into the model. `OpenAILLM` converts it to the wire shape for you; if you build
your own agent loop against a non-standard client, call
`messages_to_openai(messages)` yourself before sending. (Anthropic's adapter
converts history internally — no call needed.)

## Configuration knobs

| Knob | Default | Meaning |
|---|---|---|
| `storage_dir` | — | root of everything; `ltm/` + `stm/` live in the memory root below it |
| `user_id` | — | when set, the memory root becomes `storage_dir/<user_id>/` — a fully separate memory set (memories + transcripts) per user; change it to switch |
| `llm` / `llm_tools` | — | summarizer + extractor; tool caller for consolidation. `llm` accepts any callable OR a provider client object you initialized (auto-adapted). No tool caller = inbox-only mode, no crash |
| `max_tokens_stm` / `max_tokens_ltm` | — | hard caps, yours to choose |
| `max_summary_tokens` | 35% of `max_tokens_stm` (min 300) | cap the rolling STM summary itself; pass your own number to own that budget |
| `encoding` | `o200k_base` | tiktoken encoding |
| `token_counter` | — | supply your model's exact counter instead of tiktoken |
| `msg_overhead` | 4 | per-message token overhead in counting |
| `keep_recent` | 0.25 | fraction of budget kept verbatim after a summarize |
| `agent_loop_max_steps` | 12 | consolidation agent's tool-loop cap |
| `retries` | 2 | LLM call retries |
| `atexit_timeout` | 30.0 | seconds to let background work finish at exit |
| `session_id` | random UUID | names this conversation's transcript (`stm/<session_id>.md`); pin it to resume a session or share the id across two chatbots |

## Notes & caveats

* **Token counts are approximate** for non-OpenAI models unless you pass
  `token_counter=`. tiktoken (`o200k_base`) is used otherwise.
* **SIGKILL / power loss**: an in-flight summary or the final end-of-session
  extraction can be lost, but neither loses text — the transcript is fsynced to
  disk *before* each summarization (a killed summary just means the window is
  re-compressed next session from the durable tail), and the inbox keeps facts
  that haven't been consolidated yet, so a later session (or `close()`)
  recovers them.
* **Resuming a conversation**: pass the same `session_id` and the transcript is
  appended to, with line numbering continuing from the existing file, so
  `[n]` markers stay true across runs.
* **Never commit your API key.** Keep it in `.env` (already gitignored here)
  and load it with your own reader or `python-dotenv`. memkit itself never
  sees or stores keys — they live only in the client you hand to the adapter.
* **STM summarization runs on its own background thread** (separate from the
  LTM worker, so a summary never queues behind a multi-minute consolidation
  loop). Two deterministic wait points: `mem.stm.wait_until_settled(timeout)`
  for the summarizer, `mem.flush(timeout)` for the LTM worker — `close()` does
  both.
* LTM writes happen on one worker thread (the sole writer); `flush(timeout)`
  lets tests/diagnostics wait deterministically for queued work.
