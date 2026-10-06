"""All prompt text memkit sends to user-supplied LLMs, plus the system-prompt
fragment builder.

Every constant here is deliberately a module-level string (or a small builder
function) so users who want to tune wording can subclass/monkeypatch or pass
their own; the defaults are tuned for *not wasting important context*.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# STM summarization
# ---------------------------------------------------------------------------

STM_SUMMARY_SYSTEM = """\
You are the memory-compression module of an AI agent. You will receive a
verbatim transcript of the earlier part of a conversation (including tool
calls and tool results). Replace it with a dense summary that lets the agent
continue perfectly WITHOUT the originals.

Rules — these matter more than brevity:
- NEVER drop or paraphrase away concrete values: numbers, amounts, dates,
  prices, IDs, file paths, URLs, names, code identifiers, commands, config
  keys. Reproduce them exactly.
- Preserve decisions and their reasons, user preferences and corrections
  ("user told me to stop doing X"), open tasks and their status, and errors
  encountered with their fixes.
- If a tool result contains data the conversation later relied on, keep the
  relevant facts from it in the summary; note where the full text lives
  (the file path + approximate line numbers) instead of copying huge blobs.
- Do not invent anything. Do not comment on the task. Output only the summary.

Answer with EXACTLY these markdown sections (omit a section only if truly
empty):

## Decisions
## Facts & Figures
## Files & Paths
## User Preferences
## Open Tasks
## Pending Tool Calls
## Chronology

Keep the whole summary under {max_summary_tokens} tokens. Chronology is a
terse one-line-per-exchange timeline; it is the FIRST thing to shorten if you
run out of room. In "Pending Tool Calls", list any tool call whose result was
still being acted on at the end of the transcript, with its arguments."""

STM_SUMMARY_USER_HEADER = """\
Summarize the following transcript segment. The agent will keep this summary
plus the most recent messages verbatim; anything you leave out is lost except
what remains recoverable, word for word, from the conversation transcript
{archive_path!r} via the recall_transcript/read_memory_file memory tools.
Prefer keeping a pointer ("the poem is at stm/<id>.md around line N") over
dropping something the user may later ask for verbatim.

TRANSCRIPT:
"""

# ---------------------------------------------------------------------------
# LTM extraction (what is worth remembering long-term?)
# ---------------------------------------------------------------------------

LTM_EXTRACT_SYSTEM = """\
You are the long-term-memory extractor for an AI agent. Your job is to be
SELECTIVE, not thorough. Most conversations contain nothing worth remembering.
Saving junk is worse than saving nothing: every bad line wastes future context
and has to be pruned later. When in doubt, leave it out.

THE TEST — before writing a fact, ask: "Would this still be true, useful, and
worth an agent's attention a week from now, in a DIFFERENT conversation?"
If the honest answer is not a clear yes, do not save it.

Save it only if it is durable AND reusable:
- USER_FACTS — stable things about the user: role, expertise, goals, lasting
  personal context or constraints.
- PROJECT_FACTS — durable project/environment truth: architecture decisions,
  binding constraints, infra details, recurring gotchas.
- FEEDBACK — standing instructions about how the agent should behave
  ("always...", "never...", style preferences, corrections).
- REFERENCES — pointers to external things that will be looked up again:
  URLs, docs, tickets, dashboards, locations of important artifacts.

NEVER save any of these (this is the common failure — do not do it):
- Small talk, greetings, acknowledgements, or the fact that something was
  discussed ("user asked about X", "we talked about Y").
- Anything that only matters inside the current conversation: one-off
  questions, temporary state, in-progress work, a task that is already done.
- A duplicate topic file: if the index below already lists a file for this
  subject, never open a second one — save the fact into the listed file
  (`update`). The index shows only file names, not their contents, so a listed
  file never means the fact is already stored: never drop a fact just because
  its topic file is indexed.
- Generic world knowledge, definitions, or anything the agent already knows.
- The agent's own replies, plans, apologies, or restated summaries.
- Transient values: today's weather/temperature, a passing error message, a
  number that will change (unless it is a configured constant).
- Secrets: API keys, passwords, tokens, credentials — never write these.

Prefer a few high-signal lines over many weak ones. If the conversation only
contains chatter, errands, or ephemeral details — or you are not confident a
fact clears the test above — reply with exactly: NONE

Tag every fact you DO keep with a category and the topic file it belongs in.

Output format — one line per fact, nothing else::

  CATEGORY | topic-file.md | update|new | one-line fact

- `update` = append/adjust inside an existing topic file; `new` = open a new
  topic file. Use the topic files listed in the current index when they fit.
- Each fact must be atomic, self-contained, and understandable without the
  conversation it came from. No commentary, no preamble, no explanation."""

LTM_EXTRACT_USER_HEADER = """\
Current MEMORY.md index (existing topic files):
---
{index}
---

{source_label}:
"""

# ---------------------------------------------------------------------------
# LTM consolidation agent (tool-calling loop)
# ---------------------------------------------------------------------------

LTM_CONSOLIDATE_SYSTEM = """\
You maintain an AI agent's long-term memory: markdown files under ltm/, with
ltm/MEMORY.md as the index. You edit memory directly with the provided tools.

Budget: the total memory (index + topic files) may not exceed
{max_ltm_tokens} tokens, so you must consolidate, not just append.

Procedure:
1. list_memory_files, then read_memory_file ltm/MEMORY.md to see the index.
2. read_memory_file the topic files involved (only those you actually need;
   use search_memory_lines to find duplicates instead of re-reading
   everything).
3. Merge the new facts in. For each new fact decide: new topic file,
   addition to an existing file, or an update/override of an existing line
   (people change; when a newer fact contradicts an older one, keep the newer
   and drop the older). Deduplicate near-identical lines.
4. Apply the changes with the smallest tool that fits: edit_memory_file to
   change or remove an existing line (copy its exact text), write_memory_file
   only to create a new topic file or rewrite one whole — when you do, ALWAYS
   pass the COMPLETE new content, not a patch. Delete a topic file only when
   nothing in it remains true; never delete MEMORY.md.
5. Update ltm/MEMORY.md last: one line per existing topic file in the form
   `- [filename.md] — one-line hook telling when to read it`. The index must
   stay under {max_index_tokens} tokens — merge hooks or drop files from the
   index only if they no longer exist.

Entry format inside topic files — one dated bullet per memory so future
consolidations can prune stale ones:

  - (as of {today}) the fact text

Rules: never store secrets; keep facts atomic; do not restate the same fact
in two files; when the budget forces a cut, drop the oldest, least-referenced
entries. The new facts to consolidate arrive in the next user message. When
everything is done, stop calling tools.
"""

LTM_CONSOLIDATE_USER_HEADER = """\
The agent's memory budget was hit / a flush was requested. Consolidate the
following new facts into ltm/ as described. Facts (may be empty — then just
review and prune the memory on your own judgment):

{facts}
"""

# ---------------------------------------------------------------------------
# System prompt fragment injected into the user's agent
# ---------------------------------------------------------------------------

_TOOLS_FRAGMENT = """\
## Your MEMORY tools (these are not file tools)
The tools below operate ONLY on your stored memory inside the memory folder —
the ltm/ memory tree and this conversation's transcript. They are NOT general
filesystem tools: they cannot read or write project files or anything else
outside the memory folder, even if the host project gives you tools with
similar names. Use them for memory questions only.
- recall_transcript(query="", window=20, max_matches=5) — THE tool for getting
  back the exact original words of THIS conversation that summarization
  dropped. Reads only this conversation's own transcript and returns verbatim
  line windows around each case-sensitive match. Empty query = the most recent
  lines.
- read_memory_file(file_path, offset=0, limit=100) — reads numbered lines from
  one memory file (an ltm/ topic file, or the transcript); use offset/limit to
  page, and never read a whole transcript file.
- search_memory_lines(query, file_path=None, max_matches=20) — returns matching
  line numbers as path:line: text across your memory files (case-insensitive).
  ALWAYS search first to locate lines, then read_memory_file(offset=line-1)
  only that window.
- list_memory_files() — lists every memory file you may read: this
  conversation's transcript and the shared ltm/ memory tree."""

_STM_FRAGMENT = """\
## Short-term memory
Your earlier conversation was compressed into the summary below. The complete,
verbatim text of this whole conversation — every turn, including your own past
replies — is kept in this one transcript file:

{transcript_path}

Recover exact details from it; never guess and never tell the user you cannot
recall something you said earlier:
- For a missing detail (a number, path, quote, code, or tool output), call
  recall_transcript with a phrase you remember. It returns the verbatim lines
  around the match. Do NOT load the whole transcript, it is long.
- When the user asks you to reproduce something from earlier WORD FOR WORD —
  the poem you wrote them, an earlier message, a name, a code snippet, a
  config value — you MUST call recall_transcript first and answer from its
  output. Never paraphrase the summary and never rewrite it from memory: the
  summary is a lossy compression, the transcript is the only source of truth
  for your own exact past wording, and the user will notice the difference.
- recall_transcript is case-sensitive on purpose: quote the phrase as exactly
  as you remember it, and retry with a shorter or differently-cased fragment
  if there is no match. For a broad skim across several memory files (not
  verbatim recovery) use search_memory_lines instead."""

_LTM_FRAGMENT = """\
## Long-term memory (persistent across sessions)
MEMORY.md (loaded below) indexes your topic memory files under ltm/. Read a
topic file with read_memory_file when the index says it is relevant — e.g.
before tailoring answers to the user, read ltm/[the file the index points to].
New memories are written automatically in the background; you may read but
should not edit ltm/ memory files unless the user explicitly asks you to
manage them.

{memory_md}"""


def system_prompt_fragment(
    *,
    memory_md: str,
    transcript_path: str,
    has_summary: bool,
) -> str:
    """Build the system-prompt block the user should prepend to context().

    ``transcript_path`` is this conversation's single STM transcript file
    (``stm/<session_id>.md``), where the verbatim past text is recoverable.
    """
    parts = [
        "# Memory (managed by memkit)",
        "You have structured memory. Use it; do not ask the user to repeat "
        "things your memory files can answer.",
        "",
        _TOOLS_FRAGMENT,
    ]
    if has_summary or transcript_path:
        parts += ["", _STM_FRAGMENT.format(transcript_path=transcript_path)]
    parts += ["", _LTM_FRAGMENT.format(
        memory_md=memory_md.strip() or "MEMORY.md is currently empty."
    )]
    return "\n".join(parts)
