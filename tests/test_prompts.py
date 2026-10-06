"""The system-prompt fragment must describe the window the model actually has."""
from memkit.prompts import system_prompt_fragment


def frag(has_summary, transcript_path="stm/s.md"):
    return system_prompt_fragment(
        memory_md="- [user.md] — hook",
        transcript_path=transcript_path,
        has_summary=has_summary,
    )


def test_unsummarized_window_is_not_described_as_compressed():
    """The bug: the fragment claimed "your earlier conversation was compressed
    into the summary below" before any summarization had run, so the model went
    looking for a summary that did not exist and was invited to treat the
    verbatim window it DID have as a lossy compression."""
    text = frag(has_summary=False)

    assert "compressed into the summary below" not in text
    assert "summary below" not in text


def test_summarized_window_does_say_it_was_compressed():
    text = frag(has_summary=True)

    assert "compressed into the summary below" in text


def test_transcript_recovery_survives_both_states():
    """The transcript path and the recall_transcript instructions are needed
    whether or not a summary exists yet — the transcript is written from the
    first turn, and losing its instructions would strand the model when the
    summary does appear."""
    for has_summary in (False, True):
        text = frag(has_summary)
        assert "stm/s.md" in text
        assert "MUST call recall_transcript" in text
        assert "verbatim" in text and "transcript" in text


def test_ltm_index_is_always_included():
    assert "MEMORY.md" in frag(has_summary=False)
    assert "[user.md] — hook" in frag(has_summary=True)


def test_no_transcript_omits_the_short_term_section():
    """A window with no transcript yet has nothing to point at."""
    text = frag(has_summary=False, transcript_path="")

    assert "## Short-term memory" not in text
    assert "## Long-term memory" in text