import pytest

from memkit.tools import MemoryTools


@pytest.fixture
def tools(tmp_path):
    ltm = tmp_path / "ltm"
    ltm.mkdir()
    t = MemoryTools(tmp_path, ltm, session_id="s1")
    (ltm / "user.md").write_text(
        "# User\n- (as of 2026-01-01) prefers Python\n- (as of 2026-01-02) timezone is Europe/Berlin\n",
        encoding="utf-8")
    stm = tmp_path / "stm"
    stm.mkdir(parents=True)
    (stm / "s1.md").write_text(
        "\n".join(f"[{i}] user: line number {i} of the transcript" for i in range(1, 6)),
        encoding="utf-8")
    (stm / "other-session.md").write_text(
        "[1] user: a completely different conversation's secret\n", encoding="utf-8")
    return t


# -- read_memory_file: spec compliance ----------------------------------------

def test_read_memory_file_returns_numbered_lines_from_offset_plus_one(tools):
    out = tools.read_memory_file("stm/s1.md")
    lines = out.splitlines()
    assert lines[0] == "1  [1] user: line number 1 of the transcript"
    assert len(lines) == 5


def test_read_memory_file_pagination(tools):
    out = tools.read_memory_file("stm/s1.md", offset=2, limit=2)
    assert out.splitlines() == [
        "3  [3] user: line number 3 of the transcript",
        "4  [4] user: line number 4 of the transcript",
    ]


def test_read_memory_file_missing_file(tools):
    assert (tools.read_memory_file("ltm/nope.md")
            == "Error: memory file 'ltm/nope.md' does not exist.")


def test_read_memory_file_empty_file(tools, tmp_path):
    (tmp_path / "ltm" / "empty.md").write_text("", encoding="utf-8")
    assert tools.read_memory_file("ltm/empty.md") == "(memory file exists but is empty)"


def test_read_memory_file_beyond_end(tools):
    out = tools.read_memory_file("stm/s1.md", offset=99)
    assert "file has 5 lines" in out


def test_read_memory_file_invalid_paths(tools):
    assert tools.read_memory_file("").startswith("Error:")
    assert tools.read_memory_file("../outside.md").startswith("Error:")
    assert tools.read_memory_file("ltm/../secrets.env").startswith("Error:")


def test_read_memory_file_absolute_path_confined(tools, tmp_path):
    outside = tmp_path.parent / "sibling.txt"
    outside.write_text("nope", encoding="utf-8")
    assert tools.read_memory_file(str(outside)).startswith("Error:")


# -- the agent may only read its OWN conversation's transcript -------------------

def test_read_memory_file_cannot_open_another_conversations_transcript(tools):
    """An agent in session s1 must never pull text out of another chat's history."""
    out = tools.read_memory_file("stm/other-session.md")
    assert out.startswith("Error:"), f"other session's transcript was readable: {out!r}"
    assert "secret" not in out


def test_search_memory_lines_never_crosses_conversations(tools):
    out = tools.search_memory_lines("secret")
    assert "No matches" in out
    assert "other-session" not in out


def test_list_memory_files_hides_other_conversations(tools):
    out = tools.list_memory_files()
    assert "stm/s1.md" in out
    assert "other-session" not in out


def test_search_without_file_path_covers_own_transcript_and_ltm(tools):
    """The default (no file_path) search must reach both the LTM tree and this
    conversation's own transcript — that is the recovery path for exact text."""
    assert "stm/s1.md:3:" in tools.search_memory_lines("line number 3")
    assert "ltm/user.md:2:" in tools.search_memory_lines("prefers Python")


# -- search_memory_lines ----------------------------------------------------------

def test_search_memory_lines_returns_citable_line_numbers(tools):
    out = tools.search_memory_lines("line number 3")
    assert "stm/s1.md:3:" in out
    # the hit line number is directly usable as read_memory_file offset = n - 1
    window = tools.read_memory_file("stm/s1.md", offset=2, limit=1)
    assert "line number 3" in window


def test_search_memory_lines_case_insensitive_and_scoped(tools):
    out = tools.search_memory_lines("PREFERS python", file_path="ltm/user.md")
    assert "ltm/user.md:2:" in out
    assert "Berlin" not in out


def test_search_memory_lines_no_match(tools):
    assert "No matches" in tools.search_memory_lines("zzz-not-here")


def test_search_memory_lines_survives_a_file_vanishing_mid_scan(tools, tmp_path, monkeypatch):
    """A file the scan listed can be gone by the time it is read.

    The LTM worker deletes ``ltm/_inbox.md`` as soon as its facts are folded
    into topic files, so a search running at that moment sees the inbox in its
    file list but finds nothing on disk. A vanished file is simply absent, not
    an error: the search must still say "No matches" rather than emitting an
    "Error reading ..." line the agent cannot use.
    """
    ghost = tmp_path / "ltm" / "_inbox.md"   # listed by the scan, never on disk
    monkeypatch.setattr(tools, "_readable_markdown", lambda: [ghost])

    out = tools.search_memory_lines("anything")

    assert "No matches" in out
    assert "Error reading" not in out


def test_search_memory_lines_respects_max_matches(tools):
    out = tools.search_memory_lines("line number", max_matches=2)
    hits = [l for l in out.splitlines() if ": line number" in l]
    assert len(hits) == 2
    assert "5 match(es)" in out  # total reported even when truncated


def test_search_memory_lines_rejects_bad_input(tools):
    assert tools.search_memory_lines("").startswith("Error:")
    assert tools.search_memory_lines("x", file_path="ghost.md").startswith("Error:")


# -- list_memory_files -------------------------------------------------------------

def test_list_memory_files_shows_relative_paths(tools):
    out = tools.list_memory_files()
    assert "ltm/user.md" in out
    assert "stm/s1.md" in out


def test_inbox_is_invisible_to_listing_and_search(tools, tmp_path):
    """The bug: ``ltm/_inbox.md`` is the worker's staging area — raw, unvetted
    facts that are deleted the moment they are folded into topic files. It was
    listed and searched like any memory file, so the agent could read (and cite)
    lines that were about to vanish, and hits on it could 404 on the very next
    call. It must be invisible to both scans."""
    inbox = tmp_path / "ltm" / "_inbox.md"
    inbox.write_text("FACT | x.md | new | STAGED_INBOX_FACT\n", encoding="utf-8")

    assert "_inbox" not in tools.list_memory_files()
    assert "No matches" in tools.search_memory_lines("STAGED_INBOX_FACT")
    # ...while the real memory files are still scanned as before
    assert "ltm/user.md" in tools.list_memory_files()
    assert "ltm/user.md:2:" in tools.search_memory_lines("prefers Python")


# -- recall_transcript: THE verbatim-recovery tool ----------------------------------

def test_recall_returns_verbatim_window_around_match(tools):
    out = tools.recall_transcript("line number 3", window=3)
    # true file line numbers, exact original text, a few lines of context
    assert "[2] [2] user: line number 2 of the transcript" in out
    assert "[3] [3] user: line number 3 of the transcript" in out
    assert "[4] [4] user: line number 4 of the transcript" in out
    # window=3 centered on the hit: one before, one after — not the whole file
    assert "line number 1" not in out
    assert "line number 5" not in out


def test_recall_is_case_sensitive_for_word_for_word_recovery(tools):
    out = tools.recall_transcript("LINE NUMBER 3")
    assert out.startswith("No verbatim match")
    assert "case-sensitive" in out


def test_recall_empty_query_returns_recent_tail(tools):
    out = tools.recall_transcript("", window=2)
    assert "tail" in out
    assert "[5] [5] user: line number 5 of the transcript" in out
    assert "[4] [4] user: line number 4 of the transcript" in out
    assert "line number 3" not in out


def test_recall_coalesces_nearby_matches_into_one_window(tools):
    out = tools.recall_transcript("line number", window=4, max_matches=5)
    assert "5 verbatim match(es)" in out
    # overlapping windows merge; every line appears at most once
    body = out.split("\n\n...\n\n")
    all_lines = [l for block in body for l in block.splitlines() if l.startswith("[")]
    assert len(all_lines) == len(set(all_lines))


def test_recall_respects_max_matches(tools):
    out = tools.recall_transcript("line number", window=1, max_matches=2)
    assert "showing" not in out  # total is still reported...
    assert "5 verbatim match(es)" in out
    blocks = [b for b in out.splitlines() if b.startswith("[")]
    assert len(blocks) == 2  # ...but only the requested number of windows


def test_recall_only_sees_this_conversations_transcript(tools):
    """The other session's transcript holds 'secret'; recall must not surface it."""
    out = tools.recall_transcript("secret")
    assert out.startswith("No verbatim match")


def test_recall_without_bound_session_is_refused(tmp_path):
    t = MemoryTools(tmp_path, tmp_path / "ltm")  # no session_id
    assert "no session transcript" in t.recall_transcript("anything").lower()


def test_recall_before_first_summarization_says_so(tmp_path):
    ltm = tmp_path / "ltm"
    ltm.mkdir()
    t = MemoryTools(tmp_path, ltm, session_id="fresh")
    out = t.recall_transcript("anything")
    assert out.startswith("Error:")
    assert "no transcript yet" in out


# -- guarded LTM memory-write tools --------------------------------------------------

def test_write_memory_file_ltm_markdown_ok(tools, tmp_path):
    res = tools.write_memory_file("ltm/project.md", "# Project\n- (as of 2026-10-01) uses uv\n")
    assert "Wrote" in res
    assert (tmp_path / "ltm" / "project.md").exists()


def test_write_memory_file_outside_ltm_rejected(tools):
    assert "Error" in tools.write_memory_file("stm/s1.md", "nope")


def test_write_memory_file_non_markdown_rejected(tools):
    assert "Error" in tools.write_memory_file("ltm/data.txt", "nope")


def test_write_memory_file_traversal_rejected(tools, tmp_path):
    assert "Error" in tools.write_memory_file("ltm/../../evil.md", "nope")
    assert not (tmp_path.parent / "evil.md").exists()


def test_delete_memory_file_protects_memory_md(tools):
    (tools.ltm_dir / "MEMORY.md").write_text("# Memory Index\n", encoding="utf-8")
    assert "cannot be deleted" in tools.delete_memory_file("ltm/MEMORY.md")


def test_delete_memory_file_ok(tools):
    (tools.ltm_dir / "gone.md").write_text("x", encoding="utf-8")
    assert "Deleted" in tools.delete_memory_file("ltm/gone.md")
    assert not (tools.ltm_dir / "gone.md").exists()


# -- edit_memory_file -----------------------------------------------------------------

def test_edit_memory_file_replaces_exact_text(tools, tmp_path):
    res = tools.edit_memory_file("ltm/user.md", "prefers Python", "prefers Rust")
    assert "Edited ltm/user.md" in res
    text = (tmp_path / "ltm" / "user.md").read_text(encoding="utf-8")
    assert "prefers Rust" in text
    assert "prefers Python" not in text
    assert "Europe/Berlin" in text  # untouched rest of the memory file


def test_edit_memory_file_empty_new_text_deletes(tools, tmp_path):
    tools.edit_memory_file("ltm/user.md",
                           "- (as of 2026-01-02) timezone is Europe/Berlin\n", "")
    text = (tmp_path / "ltm" / "user.md").read_text(encoding="utf-8")
    assert "Berlin" not in text
    assert "prefers Python" in text


def test_edit_memory_file_requires_unique_match(tools):
    out = tools.edit_memory_file("ltm/user.md", "- ", "* ")
    assert "matches" in out and "replace_all" in out


def test_edit_memory_file_replace_all(tools, tmp_path):
    res = tools.edit_memory_file("ltm/user.md", "- ", "* ", replace_all=True)
    assert "replaced 2 occurrence(s)" in res
    text = (tmp_path / "ltm" / "user.md").read_text(encoding="utf-8")
    assert text.count("* (as of") == 2


def test_edit_memory_file_old_text_must_match_exactly(tools):
    assert "not found" in tools.edit_memory_file("ltm/user.md", "prefers  python", "x")


def test_edit_memory_file_requires_existing_file(tools):
    out = tools.edit_memory_file("ltm/new.md", "a", "b")
    assert "does not exist" in out
    assert "write_memory_file" in out  # and it points at the right tool


def test_edit_memory_file_rejects_bad_args(tools):
    assert "non-empty" in tools.edit_memory_file("ltm/user.md", "", "x")
    assert "string" in tools.edit_memory_file("ltm/user.md", "prefers", None)


def test_edit_memory_file_confined_to_ltm_markdown(tools, tmp_path):
    assert "Error" in tools.edit_memory_file("stm/s1.md", "line number 1", "x")
    assert "Error" in tools.edit_memory_file("ltm/notes.txt", "a", "b")
    assert "Error" in tools.edit_memory_file("ltm/../../evil.md", "a", "b")
    assert not (tmp_path.parent / "evil.md").exists()
    # the transcript is byte-for-byte unchanged after all those attempts
    assert "line number 1" in (tmp_path / "stm" / "s1.md").read_text(encoding="utf-8")


def test_edit_memory_file_can_edit_the_index(tools):
    """MEMORY.md cannot be deleted but may be edited — the index needs upkeep."""
    (tools.ltm_dir / "MEMORY.md").write_text("# Memory Index\n", encoding="utf-8")
    res = tools.edit_memory_file("ltm/MEMORY.md", "# Memory Index\n",
                                 "# Memory Index\n- [user.md] — user facts\n")
    assert "Edited" in res


# -- execute dispatch -----------------------------------------------------------------

def test_execute_dispatches_and_reports_errors_as_text(tools):
    out = tools.execute("read_memory_file", {"file_path": "ltm/user.md", "limit": 1})
    assert out.startswith("1  # User")
    assert tools.execute("bogus", {}).startswith("Error: unknown tool")
    assert tools.execute("read_memory_file", {"nope": 1}).startswith("Error: unexpected")


def test_execute_rejects_old_generic_file_tool_names(tools):
    """The rename is the point: generic file-tool names must not resolve here."""
    for old in ("read_file", "write_file", "delete_file", "search_lines",
                "list_files", "edit_file"):
        assert tools.execute(old, {}).startswith("Error: unknown tool")


def test_execute_recall_and_edit_round_trip(tools, tmp_path):
    res = tools.execute("edit_memory_file", {
        "file_path": "ltm/user.md",
        "old_text": "prefers Python",
        "new_text": "prefers Python and Rust",
    })
    assert "Edited" in res
    out = tools.execute("recall_transcript", {"query": "line number", "window": 1})
    assert "verbatim match" in out
