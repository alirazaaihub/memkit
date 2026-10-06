"""The session identity: one random uuid per conversation, overridable by the
user, and the single transcript file it names. Plus user_id: one storage dir,
a separate memory root per user."""
import uuid

import pytest

from memkit.config import MemoryConfig, _generate_session_id


def dummy_llm(messages):
    return ""


def cfg(tmp_path, **kw):
    return MemoryConfig(storage_dir=tmp_path / "mem", llm=dummy_llm,
                        max_tokens_stm=100, max_tokens_ltm=100, **kw)


def test_generated_session_id_is_a_random_uuid():
    ids = {_generate_session_id() for _ in range(50)}
    assert len(ids) == 50, "generated session ids collided / are not random"
    for sid in ids:
        assert len(sid) == 32
        uuid.UUID(hex=sid)  # raises if it is not a real uuid


def test_each_conversation_gets_its_own_session_id(tmp_path):
    assert cfg(tmp_path).session_id != cfg(tmp_path).session_id


def test_session_id_is_user_settable_for_reuse_across_apps(tmp_path):
    # a user running two chatbots can pin their own id
    c = cfg(tmp_path, session_id="my-chatbot-1")
    assert c.session_id == "my-chatbot-1"


def test_explicit_session_id_none_means_generate_one(tmp_path):
    """The bug: the facade forwards its own ``session_id=None`` default straight
    through, and a dataclass default_factory only fires when the argument is
    OMITTED — so an explicit None reached validation and raised
    `ValueError: session_id must be a simple directory/file name ... got None`.
    An explicit None must mean "generate one", exactly as the type says."""
    c = cfg(tmp_path, session_id=None)
    assert c.session_id
    _validate = uuid.UUID(hex=c.session_id)   # a real, generated uuid
    assert c.session_id != cfg(tmp_path, session_id=None).session_id


def test_session_file_is_one_file_named_by_session_id(tmp_path):
    c = cfg(tmp_path, session_id="chat-abc")
    assert c.session_file == tmp_path / "mem" / "stm" / "chat-abc.md"


def test_session_file_lives_under_the_stm_dir(tmp_path):
    c = cfg(tmp_path, session_id="chat-abc")
    assert c.session_file.parent == c.stm_dir


# -- user_id: one storage dir, separate memory roots ------------------------------

def test_no_user_id_keeps_flat_layout(tmp_path):
    c = cfg(tmp_path)
    assert c.user_id is None
    assert c.root_dir == tmp_path / "mem"
    assert c.ltm_dir == tmp_path / "mem" / "ltm"


def test_user_id_maps_to_its_own_memory_root(tmp_path):
    a, b = cfg(tmp_path, user_id="alice"), cfg(tmp_path, user_id="bob")
    assert a.root_dir == tmp_path / "mem" / "alice"
    assert b.root_dir == tmp_path / "mem" / "bob"
    # ...and every derived directory follows, so switching user_id switches
    # the whole memory set
    assert a.ltm_dir == tmp_path / "mem" / "alice" / "ltm"
    assert a.stm_dir == tmp_path / "mem" / "alice" / "stm"
    assert a.session_file == a.stm_dir / f"{a.session_id}.md"
    assert a.stm_dir != b.stm_dir and a.ltm_dir != b.ltm_dir


def test_int_user_id_is_accepted_and_stringified(tmp_path):
    c = cfg(tmp_path, user_id=1)
    assert c.user_id == "1"
    assert c.root_dir == tmp_path / "mem" / "1"


def test_user_id_must_be_a_safe_leaf_name(tmp_path):
    for bad in ("../elsewhere", "a/b", "a\\b", "..", ".", "a\x00b"):
        try:
            cfg(tmp_path, user_id=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"user_id={bad!r} should have been refused")
    with pytest.raises(ValueError):
        cfg(tmp_path, user_id=True)   # bool is not a user id


def test_user_id_roots_are_created_on_disk(tmp_path):
    c = cfg(tmp_path, user_id="carol")
    assert c.root_dir.is_dir()


# -- max_summary_tokens: the summary budget is the user's to set ----------------

def test_max_summary_tokens_defaults_to_none(tmp_path):
    assert cfg(tmp_path).max_summary_tokens is None


def test_max_summary_tokens_is_user_settable(tmp_path):
    c = cfg(tmp_path, max_summary_tokens=1234)
    assert c.max_summary_tokens == 1234


def test_max_summary_tokens_must_be_a_positive_int_or_none(tmp_path):
    for bad in (0, -5, "300", 12.5, True):
        try:
            cfg(tmp_path, max_summary_tokens=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"max_summary_tokens={bad!r} should have been refused")