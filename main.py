"""memkit demo chatbot — a full agent loop with STM + LTM in ~50 lines.

Run:  uv run python main.py
Type in the console; the bot remembers the conversation (short term) and
remembers *you* across restarts (long term, under ./demo_memory/<USER_ID>/ltm/).

The bot is given memkit's MEMORY tools (see mem.tools()): recall_transcript
for the exact words summarization dropped, read_memory_file /
search_memory_lines / list_memory_files for its stored memories. They only
ever see inside the memory folder — they are not generic file tools.

Commands:  exit / quit  ends the session (final memories are extracted),
           reset       starts a fresh short-term window (LTM is kept),
           /compact    summarize now, whether or not the window is full.
"""
import sys
from pathlib import Path

from openai import OpenAI

from memkit import Memory, OpenAILLM

# Whose memories these are. Change it (or run two instances with different
# ids) and you get a completely separate memory set under ./demo_memory/.
USER_ID = "demo-user"

# ---- 1. credentials from .env ----------------------------------------------
env_path = Path(__file__).parent / ".env"
if not env_path.exists():
    sys.exit("no .env found — create one with TOKENHARBOR_BASE_URL, "
             "TOKENHARBOR_API_KEY, TOKENHARBOR_MODEL (see README)")
env = {}
for line in env_path.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
for k in ("TOKENHARBOR_BASE_URL", "TOKENHARBOR_API_KEY", "TOKENHARBOR_MODEL"):
    if not env.get(k):
        sys.exit(f".env is missing {k}")

client = OpenAI(base_url=env["TOKENHARBOR_BASE_URL"], api_key=env["TOKENHARBOR_API_KEY"])
llm = OpenAILLM(model=env["TOKENHARBOR_MODEL"], client=client, max_tokens=1024)

# ---- 2. memory: one object, both kinds -------------------------------------
# The caps live here, once, so the startup Memory and the 'reset' Memory can
# never drift apart. Tune them in one place.
MEMORY_DIR = "./demo_memory"
MAX_TOKENS_STM = 3000      # when STM fills, it archives + summarizes
MAX_TOKENS_LTM = 6000      # when LTM fills, the consolidation agent tidies the tree
MAX_SUMMARY_TOKENS = 1000  # cap the rolling summary itself


def build_memory() -> Memory:
    """A fresh Memory over the same user's memory set, with the same caps."""
    return Memory(
        MEMORY_DIR,                # root; each user's memory set lives in demo_memory/<user_id>/
        llm=llm,                   # used for summarizing + extracting (llm.with_tools auto-detected)
        max_tokens_stm=MAX_TOKENS_STM,
        max_tokens_ltm=MAX_TOKENS_LTM,
        max_summary_tokens=MAX_SUMMARY_TOKENS,
        user_id=USER_ID,           # per-user memory folder: change it to switch memories
        atexit_timeout=120,        # let slow end-of-session extraction finish
    )


mem = build_memory()
try:  # Windows console sometimes can't print em-dashes / unicode replies
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass


def agent_turn(user_text: str) -> str:
    """One user turn -> final assistant text, running memkit's tools if asked."""
    messages = [{"role": "system", "content": mem.system_prompt_fragment()}] \
        + mem.context()
    messages.append({"role": "user", "content": user_text})
    mem.add("user", user_text)

    for _ in range(8):  # safety cap on tool-loop steps
        reply = llm.with_tools(messages, mem.tools())
        mem.add("assistant", reply.get("content"), tool_calls=reply.get("tool_calls") or None)
        messages.append({"role": "assistant", "content": reply.get("content"),
                         "tool_calls": reply.get("tool_calls") or []})
        calls = reply.get("tool_calls") or []
        if not calls:
            return reply.get("content") or "(empty reply)"
        for call in calls:  # the agent recovers archived text / reads its own memories
            result = mem.execute_tool(call["name"], call["arguments"])
            messages.append({"role": "tool", "tool_call_id": call["id"],
                             "name": call["name"], "content": result})
            mem.add("tool", result, name=call["name"],
                    tool_call_id=call["id"])  # keep STM's view complete
    return "(stopped after 8 tool steps)"


# ---- 3. chat loop ------------------------------------------------------------
print(f"memkit demo — talking to {env['TOKENHARBOR_MODEL']} as user {USER_ID!r}. "
      f"memories in {mem.memory_root}/  ('exit' to quit, 'reset' for a new STM "
      f"window, '/compact' to summarize now)")
_index_path = mem.memory_root / "ltm" / "MEMORY.md"
if _index_path.exists():
    print("[ltm index from previous runs]\n" + _index_path.read_text(encoding="utf-8").strip() + "\n")

while True:
    try:
        text = input("you> ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not text:
        continue
    if text.lower() in ("exit", "quit"):
        break
    if text.lower() == "reset":
        mem.close()  # flush this session's memories
        mem = build_memory()  # same caps — a fresh STM window, LTM kept
        print("bot> (fresh short-term window; long-term memory kept)\n")
        continue
    if text.lower() in ("/compact", "compact"):
        before = mem.stm.token_count()
        mem.compact()  # summarize now, cap or no cap
        after = mem.stm.token_count()
        if after < before:
            print(f"bot> (memory compacted: {before} -> {after} tokens; the older "
                  f"turns live on in the summary and, verbatim, in the transcript)\n")
        else:
            print("bot> (nothing to compact — the window is already just one turn)\n")
        continue
    print("bot> " + agent_turn(text) + "\n")

mem.close()  # final extraction runs here too (atexit is only the safety net)
print(f"session closed — long-term memories saved in {mem.memory_root / 'ltm'}/")
