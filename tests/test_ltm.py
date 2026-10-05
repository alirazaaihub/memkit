from memkit.config import MemoryConfig
from memkit.ltm import LongTermMemory, INBOX_NAME
from memkit.tokens import TokenCounter
from memkit.tools import MemoryTools

from conftest import ScriptedLLM, ScriptedToolsLLM

CHARS = lambda text: len(text)


def make_ltm(tmp_path, llm=None, llm_tools=None, max_ltm=10_000):
    cfg = MemoryConfig(
        storage_dir=tmp_path / "mem", llm=llm or ScriptedLLM([]),
        llm_tools=llm_tools, max_tokens_stm=2000, max_tokens_ltm=max_ltm,
        token_counter=CHARS, session_id="s",
    )
    tools = MemoryTools(cfg.root_dir, cfg.ltm_dir)
    return LongTermMemory(cfg, TokenCounter(token_counter=CHARS), tools), cfg


# -- initialization -------------------------------------------------------------

def test_creates_memory_md_index(tmp_path):
    ltm, cfg = make_ltm(tmp_path)
    assert (cfg.ltm_dir / "MEMORY.md").read_text(encoding="utf-8").startswith("# Memory Index")
    assert ltm.read_index() != ""


# -- extraction ---------------------------------------------------------------

def test_extract_parses_valid_fact_lines(tmp_path):
    llm = ScriptedLLM([
        "USER_FACTS | user.md | new | prefers Python\n"
        "PROJECT_FACTS | stack.md | update | uses uv for deps\n"
        "GARBAGE_LINE_WITHOUT_PIPES\n"
        "WRONG_CAT | x.md | new | ignored\n",
    ])
    ltm, cfg = make_ltm(tmp_path, llm=llm)
    facts = ltm.extract("conversation text here", "Archived transcript")
    assert len(facts) == 2
    assert all("| user.md |" in f or "| stack.md |" in f for f in facts)
    assert all("(as of 20" in f for f in facts)  # dated entries
    inbox = cfg.ltm_dir / INBOX_NAME
    assert inbox.exists() and "prefers Python" in inbox.read_text(encoding="utf-8")


def test_extract_none_reply_writes_nothing(tmp_path):
    ltm, cfg = make_ltm(tmp_path, llm=ScriptedLLM(["NONE"]))
    assert ltm.extract("x", "y") == []
    assert not (cfg.ltm_dir / INBOX_NAME).exists()


def test_extract_llm_failure_is_swallowed(tmp_path):
    ltm, cfg = make_ltm(tmp_path, llm=ScriptedLLM([]))  # runs dry -> raises
    assert ltm.extract("x", "y") == []


def test_extract_sanitizes_topic_paths(tmp_path):
    llm = ScriptedLLM(["USER_FACTS | ../evil/thing.md | new | sneaky fact"])
    ltm, cfg = make_ltm(tmp_path, llm=llm)
    facts = ltm.extract("x", "y")
    assert facts
    topic_field = facts[0].split("|")[1].strip()
    assert "/" not in topic_field and "\\" not in topic_field  # basename only
    assert topic_field.endswith(".md")
    # after take+flush it must land inside ltm/, never above it
    ltm.flush_inbox_to_files()
    assert (cfg.ltm_dir / "thing.md").exists()
    assert not (cfg.storage_dir.parent / "evil").exists()


# -- deterministic inbox flush ------------------------------------------------------

def test_flush_inbox_creates_topic_files_and_index(tmp_path):
    ltm, cfg = make_ltm(tmp_path, llm=ScriptedLLM([
        "USER_FACTS | user.md | new | name is Ada",
    ]))
    ltm.extract("...", "...")
    n = ltm.flush_inbox_to_files()
    assert n == 1
    user_md = (cfg.ltm_dir / "user.md").read_text(encoding="utf-8")
    assert "name is Ada" in user_md
    index = (cfg.ltm_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert "[user.md]" in index   # repair_index picked up the new file


def test_flush_dedupes_verbatim_facts(tmp_path):
    ltm, cfg = make_ltm(tmp_path)
    fact = "USER_FACTS | user.md | update | (as of 2026-10-01) likes tea"
    ltm.restore_inbox([fact, fact])
    ltm.flush_inbox_to_files()
    body = (cfg.ltm_dir / "user.md").read_text(encoding="utf-8")
    assert body.count("likes tea") == 1


# -- consolidation (tool-calling agent loop) -------------------------------------

def test_consolidate_runs_agent_and_writes(tmp_path):
    script = ScriptedToolsLLM([
        {"content": None, "tool_calls": [
            {"name": "list_memory_files", "arguments": {}}]},
        {"content": None, "tool_calls": [
            {"name": "read_memory_file", "arguments": {"file_path": "ltm/MEMORY.md"}}]},
        {"content": None, "tool_calls": [
            {"name": "write_memory_file", "arguments": {
                "file_path": "ltm/user.md",
                "content": "# User\n- (as of 2026-10-01) prefers Rust\n"}}]},
        {"content": None, "tool_calls": [
            {"name": "edit_memory_file", "arguments": {
                "file_path": "ltm/user.md",
                "old_text": "prefers Rust", "new_text": "prefers Rust and Python"}}]},
        {"content": "done", "tool_calls": []},
    ])
    ltm, cfg = make_ltm(tmp_path, llm_tools=script)
    ltm.restore_inbox(["USER_FACTS | user.md | new | prefers Rust"])
    assert ltm.consolidate() is True
    body = (cfg.ltm_dir / "user.md").read_text(encoding="utf-8")
    assert "prefers Rust and Python" in body
    assert "[user.md]" in ltm.read_index()


def test_consolidate_without_llm_tools_degrades_to_inbox_only(tmp_path):
    """Change 2: no tool-calling LLM must not crash anything. Extraction and
    deterministic persistence keep working; only the agent loop is skipped."""
    ltm, cfg = make_ltm(tmp_path, llm_tools=None)
    ltm.restore_inbox(["USER_FACTS | user.md | new | likes tea"])
    assert ltm.consolidate() is False        # no raise, no agent loop
    assert ltm.inbox_pending()               # facts were not lost
    assert ltm.flush_inbox_to_files() == 1   # ...and still reach topic files
    assert "likes tea" in (cfg.ltm_dir / "user.md").read_text(encoding="utf-8")


def test_consolidate_failure_returns_facts_to_inbox(tmp_path):
    script = ScriptedToolsLLM([])  # immediately raises
    ltm, cfg = make_ltm(tmp_path, llm_tools=script)
    ltm.restore_inbox(["USER_FACTS | user.md | new | survives"])
    assert ltm.consolidate() is False
    assert ltm.inbox_pending()
    assert "survives" in (cfg.ltm_dir / INBOX_NAME).read_text(encoding="utf-8")


def test_over_budget(tmp_path):
    ltm, cfg = make_ltm(tmp_path, max_ltm=1000)
    assert ltm.over_budget() is False
    (cfg.ltm_dir / "big.md").write_text("x" * 5000, encoding="utf-8")
    assert ltm.over_budget() is True


# -- index repair safety net ----------------------------------------------------------

def test_restore_inbox_appends_and_dedupes(tmp_path):
    # restore must not clobber facts another path added in the meantime
    ltm, cfg = make_ltm(tmp_path)
    ltm.restore_inbox(["USER_FACTS | a.md | new | first"])
    ltm.restore_inbox(["USER_FACTS | b.md | new | second",
                       "USER_FACTS | a.md | new | first"])  # dup ignored
    body = (cfg.ltm_dir / INBOX_NAME).read_text(encoding="utf-8").splitlines()
    assert "first" in "\n".join(body) and "second" in "\n".join(body)
    assert sum(1 for l in body if "first" in l) == 1


def test_repair_index_no_duplicate_for_restyled_line(tmp_path):
    ltm, cfg = make_ltm(tmp_path)
    (cfg.ltm_dir / "user.md").write_text("- (as of 2026-10-01) x\n", encoding="utf-8")
    # index mentions the file but with spaces inside the brackets
    (cfg.ltm_dir / "MEMORY.md").write_text(
        "# Memory Index\n- [ user.md ] — hook\n", encoding="utf-8")
    ltm._repair_index()
    index = (cfg.ltm_dir / "MEMORY.md").read_text(encoding="utf-8")
    assert index.count("user.md") == 1, f"repair duplicated the entry:\n{index}"


def test_repair_index_adds_missing_and_drops_stale(tmp_path):
    ltm, cfg = make_ltm(tmp_path)
    (cfg.ltm_dir / "orphan.md").write_text("- (as of 2026-10-01) loose end\n",
                                           encoding="utf-8")
    ltm._repair_index()
    assert "[orphan.md]" in ltm.read_index()
    (cfg.ltm_dir / "orphan.md").unlink()
    ltm._repair_index()
    assert "[orphan.md]" not in ltm.read_index()
