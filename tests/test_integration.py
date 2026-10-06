"""End-to-end: a long fake conversation through the real Memory facade."""
import time

from conftest import ScriptedLLM, ScriptedToolsLLM

from memkit import Memory

CHARS = lambda text: len(text)

SUMMARY_1 = (
    "## Decisions\n- chose uv\n## Facts & Figures\n- budget value is 42178-beta\n"
    "## Files & Paths\n- ./src/memkit/tools.py\n## User Preferences\n- likes terseness\n"
    "## Open Tasks\n- write README\n## Pending Tool Calls\n- none\n"
    "## Chronology\n- long discussion, compressed"
)
FACT_LINE = "USER_FACTS | user.md | new | (as of 2026-10-01) likes terseness"


def make_memory(tmp_path):
    """Plain llm: first reply = STM summary, then LTM extraction replies."""
    llm = ScriptedLLM([SUMMARY_1, FACT_LINE], repeat_last=True)
    tools_llm = ScriptedToolsLLM([
        {"content": "consolidated", "tool_calls": []},
    ] * 4)
    return Memory(
        tmp_path / "mem",
        llm=llm,
        llm_tools=tools_llm,
        max_tokens_stm=600,
        max_tokens_ltm=2000,
        token_counter=CHARS,
        keep_recent=0.3,
        session_id="s-int",
    )


def test_full_lifecycle(tmp_path):
    mem = make_memory(tmp_path)
    marker = "The magic value is 42178-beta"
    mem.add("user", marker)
    for i in range(60):
        mem.add("assistant", f"chatter {i} " + "z" * 25)
        mem.add("user", f"reply {i} " + "q" * 25)

    # 1. summarization happened: context() is compact and includes a summary
    assert mem.stm.wait_until_settled(10.0), "STM summarizer never settled"
    ctx = mem.context()
    assert mem.stm.has_summary()
    assert mem.stm.token_count() <= 1500  # compressed vs raw transcript size

    # 2. archive-before-summarize: the real raw text is on disk, numbered.
    #    Exactly ONE transcript per conversation, named by the session id.
    transcripts = sorted((mem.storage_dir / "stm").glob("*.md"))
    assert [p.name for p in transcripts] == ["s-int.md"], transcripts
    body = transcripts[0].read_text(encoding="utf-8")
    assert marker in body and "[4]" in body    # markers = true file line numbers

    # 3. the recovery loop works exactly as the prompt instructs:
    #    search_memory_lines finds the line number, read_memory_file reads
    #    only that window
    hits = mem.execute_tool("search_memory_lines", {"query": "42178-beta"})
    assert "stm/s-int.md" in hits
    line_no = int(hits.splitlines()[1].split(":", 2)[1])
    window = mem.execute_tool("read_memory_file", {
        "file_path": "stm/s-int.md", "offset": line_no - 1, "limit": 1})
    assert marker in window

    # 4. system prompt fragment names the transcript + MEMORY tools + MEMORY.md
    frag = mem.system_prompt_fragment()
    assert "recall_transcript" in frag and "stm/s-int.md" in frag and "MEMORY.md" in frag
    assert "memory tools" in frag.lower()
    # the tools are described as memory tools, not generic file tools
    assert "not file tools" in frag.lower() or "not general filesystem" in frag.lower()

    # 5. background LTM extraction ran without the foreground waiting, and
    #    wrote the memory file + index entry
    assert mem.flush(timeout=10)
    mem.close()
    user_md = mem.storage_dir / "ltm" / "user.md"
    assert user_md.exists(), "background extraction never wrote topic file"
    assert "likes terseness" in user_md.read_text(encoding="utf-8")
    assert "[user.md]" in (mem.storage_dir / "ltm" / "MEMORY.md").read_text(encoding="utf-8")


def test_exact_poem_is_recoverable_after_summarization(tmp_path):
    """The scenario this design exists for: the user asks for a poem, many turns
    follow, a summary compresses the poem away — then the user asks for that
    poem again, word for word. The transcript must still hold it, the prompt
    must tell the agent to go read it, and the recovery tools must find it."""
    mem = make_memory(tmp_path)
    poem = "Roses are red, violets are blue, session_id keeps this poem safe for you."
    mem.add("user", "write me a poem")
    mem.add("assistant", poem)
    for i in range(60):
        mem.add("user", f"unrelated question {i} " + "p" * 25)
        mem.add("assistant", f"unrelated answer {i} " + "a" * 25)
    assert mem.stm.wait_until_settled(10.0), "STM summarizer never settled"
    assert mem.stm.has_summary(), "expected the poem to be compressed away"
    assert poem not in str(mem.stm.summary), "poem survived summarization; test is void"

    frag = mem.system_prompt_fragment()
    assert "verbatim" in frag and "transcript" in frag
    # the prompt makes recall_transcript mandatory for word-for-word asks
    assert "MUST call recall_transcript" in frag

    # the one dedicated recovery tool: exact original text, verbatim
    out = mem.execute_tool("recall_transcript", {"query": "Roses are red"})
    assert poem in out          # not paraphrased — byte-for-byte the original
    assert "verbatim" in out
    # empty query gives the recent tail without needing to know any phrase
    tail = mem.execute_tool("recall_transcript", {})
    assert tail.startswith("Verbatim tail")


def test_agent_cannot_read_another_conversations_transcript(tmp_path):
    """Two chatbots sharing one storage_dir: a session must only see its own
    transcript (plus ltm/). The other session's file is written directly, the
    way a second running chatbot would have left it."""
    mem = make_memory(tmp_path)
    stm_dir = mem.storage_dir / "stm"
    stm_dir.mkdir(parents=True, exist_ok=True)
    (stm_dir / "chat-b.md").write_text(
        "[1] user: my private code word is MARMOT-3\n", encoding="utf-8")
    (mem.storage_dir / "ltm" / "shared.md").write_text(
        "# Shared\n- (as of 2026-10-01) shared LTM fact\n", encoding="utf-8")

    mem.add("user", "my private code word is OTTER-7")
    for i in range(80):
        mem.add("user", f"a chatter {i} " + "x" * 30)
    assert mem.stm.wait_until_settled(10.0), "STM summarizer never settled"
    assert (stm_dir / "s-int.md").exists()

    assert "chat-b" not in mem.execute_tool("list_memory_files", {})
    assert mem.execute_tool("read_memory_file", {"file_path": "stm/chat-b.md"}).startswith("Error:")
    assert "No matches" in mem.execute_tool("search_memory_lines", {"query": "MARMOT-3"})
    # ...while it still reaches its own transcript and the shared ltm/ tree
    assert "OTTER-7" in mem.execute_tool("search_memory_lines", {"query": "OTTER-7"})
    assert "shared.md" in mem.execute_tool("search_memory_lines", {"query": "shared LTM fact"})
    # recall_transcript is hard-wired to THIS session's file: no parameter can
    # point it at another conversation, so chat-b's code word stays unreachable
    assert "No verbatim match" in mem.execute_tool("recall_transcript", {"query": "MARMOT-3"})
    mem.close()


def test_two_users_one_storage_dir_get_separate_memory_roots(tmp_path):
    """Change 4: user_id switches the whole memory set — one user's agent can
    never list, search or read another user's memories or transcripts."""
    def make(user):
        return Memory(
            tmp_path / "mem", llm=ScriptedLLM([]),
            llm_tools=ScriptedToolsLLM([{"content": "ok", "tool_calls": []}]),
            max_tokens_stm=50_000, max_tokens_ltm=50_000,
            token_counter=CHARS, session_id=f"chat-{user}", user_id=user)
    mem1, mem2 = make("alice"), make("bob")
    assert mem1.memory_root == tmp_path / "mem" / "alice"
    assert mem2.memory_root == tmp_path / "mem" / "bob"

    (mem1.memory_root / "ltm" / "secrets.md").write_text(
        "# Alice\n- (as of 2026-10-01) alice favourite number is 71\n",
        encoding="utf-8")
    (mem1.memory_root / "stm").mkdir(exist_ok=True)
    (mem1.memory_root / "stm" / "chat-alice.md").write_text(
        "[1] user: alice private code word is OTTER-7\n", encoding="utf-8")

    assert "secrets.md" in mem1.execute_tool("list_memory_files", {})
    listing = mem2.execute_tool("list_memory_files", {})
    assert "secrets.md" not in listing and "alice" not in listing
    assert "No matches" in mem2.execute_tool(
        "search_memory_lines", {"query": "favourite number"})
    assert mem2.execute_tool(
        "read_memory_file", {"file_path": "stm/chat-alice.md"}).startswith("Error:")
    # even with alice's user_id embedded in the path, the confine root refuses
    assert mem2.execute_tool("read_memory_file",
                             {"file_path": "../alice/ltm/secrets.md"}).startswith("Error:")
    mem1.close()
    mem2.close()


def test_pending_tool_call_survives_full_flow(tmp_path):
    mem = make_memory(tmp_path)
    for i in range(40):
        mem.add("user", f"warm {i} " + "x" * 30)
    mem.add("assistant", None, tool_calls=[
        {"id": "keepme", "name": "search_memory_lines",
         "arguments": {"query": "budget"}}])
    mem.add("user", "keep going")   # triggers summarization under pressure
    assert mem.stm.wait_until_settled(10.0), "STM summarizer never settled"
    ctx = mem.context()
    assert any((m.get("tool_calls") or [{}])[0].get("id") == "keepme"
               for m in ctx), "pending tool call was lost by the facade"


def test_context_plus_fragment_is_ready_messages(tmp_path):
    mem = make_memory(tmp_path)
    mem.add("user", "hi")
    msgs = [{"role": "system", "content": mem.system_prompt_fragment()}] + mem.context()
    assert msgs[0]["role"] == "system" and "memory" in msgs[0]["content"].lower()
    assert msgs[1] == {"role": "user", "content": "hi"}


def test_tools_specs_exposed(tmp_path):
    mem = make_memory(tmp_path)
    names = [t["function"]["name"] for t in mem.tools()]
    assert names == ["read_memory_file", "search_memory_lines",
                    "recall_transcript", "list_memory_files"]
    # every exposed spec says it works on memories, inside the memory folder
    for spec in mem.tools():
        text = spec["function"]["description"].lower()
        assert "memory" in text or "memories" in text, spec["function"]["name"]
    # write/edit/delete tools are refused to the user's agent...
    for internal in ("write_memory_file", "edit_memory_file", "delete_memory_file"):
        assert mem.execute_tool(internal, {"file_path": "ltm/x.md"}).startswith("Error:")
    # ...and the old generic file-tool names do not exist at all
    assert mem.execute_tool("read_file", {"file_path": "ltm/x.md"}).startswith("Error: unknown tool")


def test_memory_accepts_any_llm_the_user_initialized(tmp_path):
    """Change 2: a plain user-initialized callable — no memkit adapter — works,
    including one that is callable but not a function (a class instance)."""
    class MyModel:                       # e.g. the user's own wrapper
        def __call__(self, messages):
            return "NONE"
    mem = Memory(tmp_path / "any", llm=MyModel(), max_tokens_stm=1000,
                 max_tokens_ltm=1000, token_counter=CHARS)
    assert mem.user_id is None
    mem.add("user", "remember that I like tea")
    mem.close()
    assert mem.stm is not None           # full lifecycle ran on the user's llm


def test_memory_adapts_a_provider_client_object(tmp_path):
    """A non-callable client object (openai.OpenAI(...) style) is wrapped."""
    class FakeCompletions:
        def __init__(self): self.seen = []
        def create(self, **kw):
            self.seen.append(kw)
            class R:
                choices = [type("C", (), {"message": type("M", (), {"content": "NONE", "tool_calls": None})()})]
            return R()
    class FakeChat:
        completions = None
    class FakeOpenAIClient:              # mimics openai.OpenAI(model=...)
        def __init__(self):
            self.model = "some-model"
            self.chat = FakeChat()
            self.chat.completions = FakeCompletions()
    client = FakeOpenAIClient()
    assert not callable(client)          # a real client object is NOT callable

    mem = Memory(tmp_path / "client", llm=client, max_tokens_stm=1000,
                 max_tokens_ltm=1000, token_counter=CHARS)
    from memkit.llm import OpenAILLM
    assert isinstance(mem.config.llm, OpenAILLM)
    assert mem.config.llm.model == "some-model"
    # the adapter's own tool-capable callable is auto-detected as llm_tools
    assert mem.config.llm_tools == mem.config.llm.with_tools
    mem.close()


def test_close_is_idempotent(tmp_path):
    mem = make_memory(tmp_path)
    mem.add("user", "by the way, the capstone deadline is Nov 12")
    assert mem.close() is True
    assert mem.close() is True  # second close must not raise


def test_short_session_still_produces_memories(tmp_path):
    """Session ends WITHOUT hitting either cap — atexit/close extraction
    still runs (user's explicit requirement)."""
    # (as of ...) is added by the parser; the reply is a bare fact line
    llm = ScriptedLLM(["REFERENCES | links.md | new | capstone deadline Nov 12"],
                      repeat_last=True)
    mem = Memory(
        tmp_path / "mem2", llm=llm,
        llm_tools=ScriptedToolsLLM([{"content": "ok", "tool_calls": []}] * 2),
        max_tokens_stm=50_000, max_tokens_ltm=50_000,
        token_counter=CHARS, session_id="tiny")
    mem.add("user", "The capstone deadline is Nov 12")
    mem.close()
    ltm = mem.storage_dir / "ltm"
    topic = ltm / "links.md"
    assert topic.exists()
    assert "Nov 12" in topic.read_text(encoding="utf-8")


# -- on-demand compaction (/compact) --------------------------------------------
# The host app exposes "/compact"; it calls Memory.compact() so the user can get
# a summary whenever they want one, not only when the cap happens to be hit.

def test_compact_summarizes_long_before_the_cap(tmp_path):
    mem = make_memory(tmp_path)
    for i in range(6):
        mem.add("user", f"tiny {i}")
    assert mem.stm.has_summary() is False, "test is void — already over cap"

    assert mem.compact(timeout=10.0) is True
    assert mem.stm.has_summary(), "compact did not summarize an under-cap window"
    assert "compressed" in mem.stm.summary
    mem.close()


def test_compact_folds_older_turns_into_the_summary(tmp_path):
    """What the /compact command relies on: the window shrinks to the summary
    plus the newest turn, and the folded-away text stays verbatim-recoverable."""
    mem = make_memory(tmp_path)
    mem.add("user", "the code word is OTTER-7")
    for i in range(5):
        mem.add("assistant", f"ack {i}")
    before = len(mem.stm.turns)

    assert mem.compact(timeout=10.0) is True
    after = len(mem.stm.turns)
    assert after < before, f"compact kept every turn ({before} -> {after})"
    assert after == 1, "only the newest turn should stay verbatim"
    # the dropped text is not gone: it is in the transcript, word for word
    assert "OTTER-7" in mem.execute_tool("recall_transcript", {"query": "OTTER-7"})
    mem.close()


def test_compact_feeds_ltm_extraction(tmp_path):
    """A forced summary archives the same text a cap-driven one would, so the
    LTM extraction job is queued automatically — no extra wiring."""
    mem = make_memory(tmp_path)
    mem.add("user", "I always want short answers")
    for i in range(5):
        mem.add("assistant", f"ok {i}")

    mem.compact(timeout=10.0)
    assert mem.flush(timeout=10), "background LTM worker never drained"
    mem.close()
    user_md = mem.storage_dir / "ltm" / "user.md"
    assert user_md.exists(), "forced compaction did not feed LTM extraction"
