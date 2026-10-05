from memkit.tokens import TokenCounter

CHARS = lambda text: len(text)


def test_counts_plain_text():
    c = TokenCounter()
    assert c.count_text("hello world") == 2  # o200k_base: two tokens
    assert c.count_text("") == 0


def test_message_overhead_applied():
    c = TokenCounter(msg_overhead=4)
    plain = c.count_text("hi")
    assert c.count_message({"role": "user", "content": "hi"}) == plain + 4


def test_tool_calls_are_counted():
    c = TokenCounter()
    bare = c.count_message({"role": "assistant", "content": "ok"})
    with_call = c.count_message({
        "role": "assistant", "content": "ok",
        "tool_calls": [{"id": "x", "name": "read_file",
                        "arguments": {"file_path": "ltm/a.md", "offset": 0}}],
    })
    assert with_call > bare + 5  # name + serialized arguments counted


def test_custom_token_counter_override():
    c = TokenCounter(token_counter=lambda text: len(text))  # chars as tokens
    assert c.count_text("abcd") == 4
    assert c.count_message({"role": "user", "content": "abcd"}) == 4 + c.msg_overhead


def test_special_token_text_does_not_raise():
    c = TokenCounter()
    weird = "".join(["<", "|", "im_start", "|", "> hello"])  # looks like a special token
    assert c.count_text(weird) >= 2


# -- truncate: the returned string (marker included) fits the budget ----------

def test_truncate_result_fits_budget_marker_included():
    c = TokenCounter(token_counter=CHARS)
    out = c.truncate("a" * 500, 100)
    assert len(out) <= 100, "truncate returned more than budget chars"
    assert "truncated" in out and out.startswith("a" * 10)


def test_truncate_returns_input_when_it_fits():
    c = TokenCounter(token_counter=CHARS)
    assert c.truncate("hello", 100) == "hello"


def test_truncate_zero_budget_is_empty():
    c = TokenCounter(token_counter=CHARS)
    assert c.truncate("hello", 0) == ""


def test_truncate_tiny_budget_still_bounded():
    # budget smaller than the marker must not return more than asked
    c = TokenCounter(token_counter=CHARS)
    assert len(c.truncate("hello world " * 50, 5)) <= 5
