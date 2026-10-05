import threading
import time

import pytest

from memkit.config import MemoryConfig
from memkit.stm import ShortTermMemory, _tool_spans
from memkit.tokens import TokenCounter

# chars-as-tokens counter keeps budgets predictable in tests
CHARS = lambda text: len(text)

# Summarization is a background job now, so a test that asserts on the
# compressed window must first wait for the worker to finish. Generous by
# design: the fake LLMs are instant, so this only ever waits microseconds.
SETTLE = 10.0


def make_stm(tmp_path, replies, *, max_tokens=200, keep_recent=0.25,
             on_archive=None, retries=0, max_summary_tokens=None):
    llm_calls = []
    holder = {}

    def llm(messages):
        # record whether the transcript file already existed at call time, so a
        # test can prove the raw text hits disk BEFORE the summarizer runs
        llm_calls.append((messages, holder["cfg"].session_file.exists()))
        return replies[min(len(llm_calls) - 1, len(replies) - 1)]

    cfg = MemoryConfig(
        storage_dir=tmp_path / "mem", llm=llm,
        max_tokens_stm=max_tokens, max_tokens_ltm=10_000,
        token_counter=CHARS, keep_recent=keep_recent, retries=retries,
        session_id="s-test", max_summary_tokens=max_summary_tokens,
    )
    holder["cfg"] = cfg
    stm = ShortTermMemory(cfg, TokenCounter(token_counter=CHARS), on_archive=on_archive)
    stm._llm_calls = llm_calls  # test introspection
    return stm, cfg, llm_calls


def settle(stm):
    """Block until the background summarizer is idle (or fail loudly)."""
    assert stm.wait_until_settled(SETTLE), "STM never settled — summarizer wedged"


def test_under_cap_does_not_summarize(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["SHOULD NOT BE CALLED"])
    stm.add("user", "hello")
    stm.add("assistant", "hi there")
    settle(stm)
    assert stm.has_summary() is False
    assert stm._llm_calls == []
    assert len(stm.context()) == 2


# -- one transcript file per conversation --------------------------------------

def test_transcript_file_is_named_by_session_id(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nx"])
    for i in range(40):
        stm.add("user", f"msg {i} " + "v" * 35)
    settle(stm)
    assert cfg.session_file.exists()
    assert cfg.session_file.name == "s-test.md"
    assert stm.archives == ["stm/s-test.md"]


def test_summarizing_twice_appends_to_one_file(tmp_path):
    """The core requirement: a new summary must NOT create a new random file —
    it appends to the same conversation transcript."""
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nfirst",
                                          "## Chronology\nsecond"])
    first = "the wifi password hint is North-Star"
    stm.add("user", first)
    for i in range(40):
        stm.add("assistant", f"filler {i} " + "x" * 45)
    settle(stm)
    assert stm.has_summary()

    second = "the backup door code is 9988-Tango"
    stm.add("user", second)
    for i in range(40):
        stm.add("assistant", f"more filler {i} " + "y" * 45)
    settle(stm)

    files = sorted(p.name for p in cfg.stm_dir.glob("*.md"))
    assert files == ["s-test.md"], f"expected exactly one transcript, got {files}"
    text = cfg.session_file.read_text(encoding="utf-8")
    assert first in text and second in text


def test_transcript_is_created_once_and_only_grows(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nx"])
    for i in range(30):
        stm.add("user", f"early {i} " + "x" * 40)
    settle(stm)
    size_after_first = cfg.session_file.stat().st_size
    for i in range(60):
        stm.add("user", f"late {i} " + "x" * 40)
    settle(stm)
    assert cfg.session_file.stat().st_size > size_after_first


def test_resuming_a_pinned_session_id_continues_the_same_file(tmp_path):
    """A user who pins session_id (two chatbots, or an app restart) must land
    in the same transcript — with numbering that continues, not restarts."""
    replies = ["## Chronology\nx"]

    def build():
        return make_stm(tmp_path, replies)

    stm, cfg, calls = build()
    for i in range(40):
        stm.add("user", f"first run {i} " + "z" * 40)
    settle(stm)
    first_text = cfg.session_file.read_text(encoding="utf-8")
    assert "first run 0" in first_text

    stm2, cfg2, calls2 = build()   # an app restart, same session_id
    for i in range(40):
        stm2.add("user", f"second run {i} " + "z" * 40)
    settle(stm2)
    text = cfg2.session_file.read_text(encoding="utf-8")
    assert "first run 0" in text, "restart wiped the earlier transcript"
    assert "second run 0" in text, "restart did not append to the transcript"
    # markers still equal the file's true 1-based line numbers
    for idx, line in enumerate(text.splitlines(), start=1):
        if line.startswith("["):
            assert line.startswith(f"[{idx}] "), f"marker off in {line[:12]}"


# -- archive-before-summarize contract -----------------------------------------

def test_transcript_written_before_first_llm_call(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Decisions\nmade stuff"])
    for i in range(40):
        stm.add("user", f"turn {i} " + "y" * 30)
    settle(stm)
    assert stm.has_summary()
    assert calls, "summarizer was never called"
    for _, transcript_existed in calls:
        assert transcript_existed, "summarize ran before raw text hit the file"


def test_transcript_contains_real_text_and_numbered_lines(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\n- compressed"])
    marker = "The magic value is 42178-beta"
    stm.add("user", marker)
    for i in range(40):
        stm.add("assistant", f"padding line number {i} " + "z" * 40)
    settle(stm)
    text = cfg.session_file.read_text(encoding="utf-8")
    assert marker in text                      # real raw text, verbatim
    # [n] markers equal true file line numbers: header block is 3 lines, so
    # the first body line is [4]
    assert "[4] user: The magic value is 42178-beta" in text


def test_line_numbers_are_still_true_after_appending(tmp_path):
    """Appending must continue the numbering — a second summary's lines must
    not restart at [4], or every search_lines hit after it points at the
    wrong place."""
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nfirst",
                                          "## Chronology\nsecond"])
    stm.add("user", "alpha marker 11111")
    for i in range(40):
        stm.add("assistant", f"pad {i} " + "x" * 45)
    settle(stm)
    stm.add("user", "omega marker 22222")
    for i in range(40):
        stm.add("assistant", f"pad2 {i} " + "y" * 45)
    settle(stm)

    lines = cfg.session_file.read_text(encoding="utf-8").splitlines()
    for idx, line in enumerate(lines, start=1):
        if line.startswith("["):
            assert line.startswith(f"[{idx}] "), f"marker off in {line[:12]}"
    # the recovery walk the prompt instructs: marker -> read_file offset
    for marker in ("11111", "22222"):
        hits = [i for i, l in enumerate(lines) if marker in l and l.startswith("[")]
        assert hits, f"{marker} vanished from the transcript"
        assert lines[hits[0]].startswith(f"[{hits[0] + 1}]")


def test_failed_summarizer_appends_each_line_exactly_once(tmp_path):
    """Regression: after a failed summarization the retry must append only the
    not-yet-written delta — never duplicate the same span."""
    def boom(messages):
        raise RuntimeError("provider down")
    cfg = MemoryConfig(storage_dir=tmp_path / "mem", llm=boom,
                       max_tokens_stm=50, max_tokens_ltm=10_000,
                       token_counter=CHARS, retries=0, session_id="s")
    stm = ShortTermMemory(cfg, TokenCounter(token_counter=CHARS))
    for i in range(10):
        stm.add("user", f"line {i} " + "x" * 20)
    settle(stm)                              # every call failed; window intact
    assert cfg.session_file.exists(), "raw text was not saved despite failure"
    assert len(stm.context()) == 10          # window intact
    # recovery: once the provider works again, everything still compresses
    stm.config.llm = lambda m: "## Chronology\nall lines"
    stm.add("user", "last one " + "x" * 20)
    settle(stm)
    assert stm.has_summary()
    text = cfg.session_file.read_text(encoding="utf-8")
    for i in range(11):
        assert text.count(f"line {i} ") <= 1, f"line {i} was appended twice"
    for idx, line in enumerate(text.splitlines(), start=1):
        if line.startswith("["):
            assert line.startswith(f"[{idx}] "), f"marker off in {line[:12]}"


def test_on_archive_hook_fires_with_text(tmp_path):
    seen = []
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nx"],
                               on_archive=lambda rel, text: seen.append((rel, text)))
    for i in range(40):
        stm.add("user", f"msg {i} " + "v" * 35)
    settle(stm)
    assert seen, "on_archive was not called after summarization"
    rel, text = seen[0]
    assert rel == "stm/s-test.md"
    assert text.startswith("[4]"), f"expected numbered transcript text, got {text[:20]!r}"


# -- window behavior (unchanged semantics) -------------------------------------

def test_trailing_pending_tool_call_stays_verbatim(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Pending Tool Calls\ncall pending"])
    for i in range(25):
        stm.add("user", f"filler {i} " + "x" * 30)
    pending = {
        "role": "assistant", "content": None,
        "tool_calls": [{"id": "c1", "name": "search_lines",
                        "arguments": {"query": "42178"}}],
    }
    stm.add_message(pending)
    settle(stm)
    assert stm.token_count() > cfg.max_tokens_stm or stm.has_summary()
    ctx = stm.context()
    assert any(m.get("tool_calls") and m["tool_calls"][0]["id"] == "c1"
               for m in ctx), "trailing pending tool call was dropped"


def test_abandoned_pending_tool_call_is_compressed(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Pending Tool Calls\nabandoned c1"])
    for i in range(25):
        stm.add("user", f"filler {i} " + "x" * 30)
    stm.add_message({"role": "assistant", "content": None,
                     "tool_calls": [{"id": "c1", "name": "search_lines",
                                     "arguments": {"query": "42178"}}]})
    for i in range(60):
        stm.add("user", f"later {i} " + "x" * 30)
    settle(stm)
    assert stm.has_summary()
    # the abandoned call no longer sits verbatim in the window (it stopped
    # pinning it), ...
    assert not any(m.get("tool_calls") for m in stm.turns), \
        "abandoned tool call is still pinned verbatim in the window"
    # ... but its exact text is not lost: it is in the transcript.
    text = cfg.session_file.read_text(encoding="utf-8")
    assert "c1" in text and "search_lines" in text, \
        "abandoned tool call text missing from the transcript"


def test_completed_tool_group_survives_verbatim(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Decisions\nok"])
    for i in range(20):
        stm.add("user", f"warmup {i} " + "x" * 30)
    stm.add_message({"role": "assistant", "content": None,
                     "tool_calls": [{"id": "z9", "name": "read_file",
                                     "arguments": {"file_path": "a.md"}}]})
    stm.add("tool", "result body", tool_call_id="z9")
    settle(stm)
    ctx = stm.context()
    idx = next(i for i, m in enumerate(ctx)
               if m.get("tool_calls") and m["tool_calls"][0]["id"] == "z9")
    assert ctx[idx + 1]["role"] == "tool"
    assert ctx[idx + 1]["tool_call_id"] == "z9"


def test_split_never_orphans_tool_pairs(tmp_path):
    turns = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "tool_calls": [{"id": "1"}, {"id": "2"}]},
        {"role": "tool", "tool_call_id": "1", "content": "r1"},
        {"role": "tool", "tool_call_id": "2", "content": "r2"},
        {"role": "user", "content": "b"},
    ]
    spans = _tool_spans(turns)
    assert spans == [(0, 1), (1, 4), (4, 5)]


def test_prior_summary_folded_in(tmp_path):
    """The second compression must carry the first summary forward, not lose it.

    Two explicit phases (each settled) rather than one long add-loop: the
    summarizer now runs in the background and compresses a whole batch in one
    pass, so compressing twice must be forced deliberately."""
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nfirst summary",
                                          "## Chronology\nsecond summary"])
    for i in range(20):
        stm.add("user", f"phase one {i} " + "x" * 40)
    settle(stm)
    assert stm.has_summary()
    assert len(calls) >= 1, "first compression never ran"

    for i in range(20):
        stm.add("user", f"phase two {i} " + "x" * 40)
    settle(stm)
    assert stm.summary == "## Chronology\nsecond summary"
    second = calls[1][0]
    assert "first summary" in str(second), \
        "prior summary was not folded into re-compression"


def test_user_set_max_summary_tokens_reaches_the_prompt(tmp_path):
    """The user owns the summary budget: the number they pass must be the one
    the summarizer is told to stay under, not the derived 35%-of-STM default."""
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nshort"],
                               max_summary_tokens=777)
    assert stm._max_summary_tokens == 777
    for i in range(40):
        stm.add("user", f"msg {i} " + "v" * 35)
    settle(stm)
    assert stm.has_summary()
    prompt = str(calls[0][0])
    assert "777 tokens" in prompt
    assert str(max(300, int(cfg.max_tokens_stm * 0.35))) not in prompt


def test_default_summary_budget_derives_from_stm_cap(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Chronology\nshort"], max_tokens=1000)
    assert stm._max_summary_tokens == max(300, int(1000 * 0.35))


def test_window_stays_under_cap_after_summarize(tmp_path):
    stm, cfg, calls = make_stm(tmp_path, ["## Decisions\n" + "d" * 60],
                               max_tokens=2000)
    for i in range(80):
        stm.add("user", f"long message number {i} " + "w" * 50)
    settle(stm)
    assert stm.has_summary()
    assert stm.token_count() <= cfg.max_tokens_stm, (
        f"window {stm.token_count()} still over cap {cfg.max_tokens_stm}")


# -- the async contract itself --------------------------------------------------
# The point of the change: add() must not wait on the summarizer LLM, and a
# message that arrives mid-summary must survive the commit verbatim.

def test_add_returns_fast_while_summary_runs(tmp_path):
    """add() must return immediately even when the summarizer is slow."""
    release = threading.Event()
    entered = threading.Event()

    def slow(messages):
        entered.set()
        release.wait(5.0)            # hold the summarizer inside its LLM call
        return "## Chronology\ncompressed"

    cfg = MemoryConfig(storage_dir=tmp_path / "mem", llm=slow,
                       max_tokens_stm=200, max_tokens_ltm=10_000,
                       token_counter=CHARS, keep_recent=0.25, session_id="s")
    stm = ShortTermMemory(cfg, TokenCounter(token_counter=CHARS))
    for i in range(20):
        stm.add("user", f"msg {i} " + "x" * 30)
    assert entered.wait(5.0), "summarizer was never reached"

    # Still over cap and mid-LLM: the next add must not block on it.
    start = time.perf_counter()
    for i in range(5):
        stm.add("user", f"more {i} " + "y" * 30)
    elapsed = time.perf_counter() - start
    assert elapsed < 0.05, f"add() blocked {elapsed:.3f}s on the summarizer"

    release.set()
    settle(stm)
    assert stm.has_summary()


def test_adds_during_summary_stay_verbatim(tmp_path):
    """Messages appended while the summarizer is mid-flight must not be swept
    into the compression the worker is about to commit (it re-slices at commit
    time, so anything past the snapshot index stays verbatim)."""
    release = threading.Event()
    entered = threading.Event()

    def slow(messages):
        entered.set()
        release.wait(5.0)
        return "## Chronology\nold stuff"

    cfg = MemoryConfig(storage_dir=tmp_path / "mem", llm=slow,
                       max_tokens_stm=200, max_tokens_ltm=10_000,
                       token_counter=CHARS, keep_recent=0.25, session_id="s")
    stm = ShortTermMemory(cfg, TokenCounter(token_counter=CHARS))
    for i in range(20):
        stm.add("user", f"old {i} " + "x" * 30)
    assert entered.wait(5.0), "summarizer was never reached"

    # These land after the snapshot was taken; the in-flight commit must keep
    # them, even though it removes the older messages around them.
    stm.add("user", "FRESH-MARKER-MUST-SURVIVE")
    release.set()
    settle(stm)

    ctx = stm.context()
    assert any(m.get("content") == "FRESH-MARKER-MUST-SURVIVE" for m in ctx), \
        "a message added during summarization was lost by the commit"
    assert stm.has_summary()


def test_wait_until_settled_timeout(tmp_path):
    """A wedged summarizer must not hang the caller: the bounded wait returns
    False instead of blocking forever."""
    release = threading.Event()
    entered = threading.Event()

    def stuck(messages):
        entered.set()
        release.wait(10.0)
        return "## Chronology\nlate"

    cfg = MemoryConfig(storage_dir=tmp_path / "mem", llm=stuck,
                       max_tokens_stm=200, max_tokens_ltm=10_000,
                       token_counter=CHARS, keep_recent=0.25, session_id="s")
    stm = ShortTermMemory(cfg, TokenCounter(token_counter=CHARS))
    for i in range(20):
        stm.add("user", f"msg {i} " + "x" * 30)
    assert entered.wait(5.0), "summarizer was never reached"

    start = time.perf_counter()
    settled = stm.wait_until_settled(0.3)      # bounded: must give up quickly
    elapsed = time.perf_counter() - start
    assert settled is False, "reported settled while the summarizer was stuck"
    assert elapsed < 2.0, f"wait_until_settled hung for {elapsed:.2f}s"

    release.set()
    settle(stm)