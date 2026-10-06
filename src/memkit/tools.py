"""MEMORY tools — the agent's only window into its own memory folder.

These are NOT generic filesystem tools. They do not read project code, do not
touch anything outside the memory folder, and cannot be pointed at arbitrary
paths: every path is resolved and confined under the memory root. A host
project may already offer real file tools (read_file, edit_file, ...); these
tools are named ``*_memory_*`` for exactly that reason — they operate on the
agent's stored MEMORIES (the ``ltm/`` markdown tree) and on this
conversation's own transcript (``stm/<session_id>.md``).

Reads are additionally scoped to the CURRENT conversation: the agent may read
the shared LTM tree and its own transcript, but never another conversation's
transcript. A chatbot serving several users keeps their histories under the
same memory root, so without this an agent could pull one user's past chat
into another's context. Pinning ``session_id`` (see ``MemoryConfig``) is what
selects "its own" transcript.

Read-side tools (``read_memory_file``, ``search_memory_lines``,
``recall_transcript``, ``list_memory_files``) are for the user's agent and
cost tokens, so the prompt tells the agent to locate text first and then
recall only the window it needs. ``edit_memory_file`` / ``write_memory_file``
/ ``delete_memory_file`` are internal, restricted to LTM markdown files, and
only ever called by memkit's own background consolidation loop.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

# Longest single match line echoed back by search_memory_lines (keeps results scannable).
_MAX_MATCH_CHARS = 200
# The memkit-managed staging file for extracted-but-unconsolidated facts. It is
# transient internal state, not a memory the agent should read or cite, so the
# listing and search scans skip it. Mirror of ltm.INBOX_NAME (kept as a local
# literal because ltm imports tools, not the other way around).
_INBOX_NAME = "_inbox.md"
# Most files list_memory_files will name.
_MAX_LISTED = 500
# Hard cap on lines recall_transcript returns, however it is asked.
_MAX_RECALL_LINES = 400


class MemoryTools:
    def __init__(self, memory_root: Path, ltm_dir: Path,
                 session_id: str | None = None) -> None:
        self.root = Path(memory_root).resolve()
        self.ltm_dir = Path(ltm_dir).resolve()
        # The transcript this conversation owns. None means "unscoped": reads
        # fall back to the whole tree (kept for callers that just want the
        # memory tools without session isolation, e.g. tests of the LTM path).
        self.session_id = session_id
        self._own_transcript = (
            (self.root / "stm" / f"{session_id}.md").resolve()
            if session_id else None
        )

    # -- path guard ---------------------------------------------------------

    def _resolve(self, file_path: str) -> Path:
        """Resolve a supplied path and prove it stays inside the memory folder."""
        if not isinstance(file_path, str) or not file_path.strip():
            raise ValueError("file_path must be a non-empty string")
        candidate = Path(file_path.strip())
        if not candidate.is_absolute():
            candidate = self.root / candidate
        target = candidate.resolve()
        if not target.is_relative_to(self.root):
            raise ValueError(f"path '{file_path}' escapes the memory folder")
        return target

    def _resolve_readable(self, file_path: str) -> Path:
        """Like _resolve, but also refuses another conversation's transcript."""
        target = self._resolve(file_path)
        if self._is_foreign_transcript(target):
            raise ValueError(
                "that file belongs to a different conversation; an agent may "
                "only read its own session transcript and the ltm/ memory tree")
        return target

    def _is_foreign_transcript(self, target: Path) -> bool:
        """True when ``target`` is a transcript under stm/ that is not ours."""
        stm_dir = self.root / "stm"
        if not target.is_relative_to(stm_dir):
            return False          # ltm/ and anything else is shared
        if self._own_transcript is None:
            return False          # unscoped: everything is fair game
        return target != self._own_transcript

    def _resolve_ltm_write(self, file_path: str) -> Path:
        """Like _resolve, but confined to ltm/ and restricted to .md files."""
        target = self._resolve(file_path)
        if not target.is_relative_to(self.ltm_dir):
            raise ValueError("only memory files under the ltm/ directory can be written")
        if target.suffix.lower() != ".md":
            raise ValueError("only .md memory files can be written")
        return target

    def _rel(self, target: Path) -> str:
        return target.relative_to(self.root).as_posix()

    # -- read-side memory tools (exposed to the user's agent) ----------------

    def read_memory_file(self, file_path: str, offset: int = 0, limit: int = 100) -> str:
        """Reads a stored memory file (or this conversation's transcript) page by page.

        By default it reads up to 100 lines from the beginning. Use offset/limit
        for pagination on large memory files — never load a whole transcript at
        once. Results come back numbered by true file line number, so a
        search_memory_lines hit maps straight to read_memory_file(offset=n-1).
        This reads memories, not project files: it cannot see anything outside
        the memory folder.
        """
        try:
            target = self._resolve_readable(file_path)
        except ValueError as e:
            return f"Error: {e}"
        if not isinstance(offset, int) or not isinstance(limit, int):
            return "Error: offset and limit must be integers."
        if offset < 0:
            return "Error: offset must be >= 0."
        if limit < 0:
            return "Error: limit must be >= 0."
        if not target.exists():
            return f"Error: memory file '{file_path}' does not exist."
        if target.is_dir():
            return f"Error: '{file_path}' is a directory, not a file. Use list_memory_files()."
        lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
        if not lines:
            return "(memory file exists but is empty)"
        window = lines[offset: offset + limit]
        if not window:
            return f"(no lines at offset {offset}; file has {len(lines)} lines)"
        return "\n".join(f"{i + offset + 1}  {line}" for i, line in enumerate(window))

    def search_memory_lines(
        self,
        query: str,
        file_path: str | None = None,
        max_matches: int = 20,
    ) -> str:
        """Finds a snippet inside the stored memories and returns its line numbers.

        Returns matches as ``path:line_number: text`` so you can read exactly
        that region with read_memory_file(file_path, offset=line_number - 1)
        instead of loading a whole memory file. Case-insensitive; searches .md
        files under the memory folder (set file_path to limit it to one file).
        For the EXACT wording of this conversation use recall_transcript —
        this tool is for skimming many files, that one is for verbatim recovery.
        """
        if not isinstance(query, str) or not query.strip():
            return "Error: query must be a non-empty string."
        if not isinstance(max_matches, int) or max_matches < 1:
            max_matches = 20
        needle = query.strip().lower()

        if file_path:
            try:
                files = [self._resolve_readable(file_path)]
            except ValueError as e:
                return f"Error: {e}"
            if not files[0].is_file():
                return f"Error: memory file '{file_path}' does not exist."
        else:
            files = self._readable_markdown()

        out: list[str] = []
        errors: list[str] = []
        total = 0
        for f in files:
            rel = self._rel(f)
            try:
                lines = f.read_text(encoding="utf-8", errors="replace").splitlines()
            except FileNotFoundError:
                # A file the scan listed can be gone by read time — the LTM
                # worker deletes ltm/_inbox.md as soon as its facts are folded
                # into topic files. A vanished file is simply absent, not an
                # error to report.
                continue
            except OSError as e:
                errors.append(f"Error reading {rel}: {e}")
                continue
            for lineno, line in enumerate(lines, start=1):
                if needle in line.lower():
                    total += 1
                    if len(out) < max_matches:
                        text = line.strip()
                        if len(text) > _MAX_MATCH_CHARS:
                            text = text[:_MAX_MATCH_CHARS] + "..."
                        out.append(f"{rel}:{lineno}: {text}")
        if not out:
            if errors:
                return "\n".join(errors)
            return f"No matches for {query!r} in {len(files)} memory file(s) searched."
        header = f"{total} match(es) for {query!r}:" if total <= max_matches else (
            f"{total} match(es) for {query!r}, showing first {len(out)}:"
        )
        body = "\n".join(out)
        if errors:
            body += "\n" + "\n".join(errors)
        hint = (
            "\n\nTo see context around a hit, call "
            "read_memory_file(file_path, offset=line_number - 1, limit=20)."
            " To recover this conversation's exact words, call recall_transcript."
        )
        return header + "\n" + body + hint

    def recall_transcript(self, query: str = "", window: int = 20,
                          max_matches: int = 5) -> str:
        """Recovers the EXACT original text of THIS conversation from its transcript.

        Summarization compresses older turns; the words it dropped still live,
        byte-for-byte, in this conversation's one transcript file
        (``stm/<session_id>.md``). This tool reads ONLY that file — it cannot
        see other conversations or the ltm/ tree — and returns verbatim line
        windows around each match, numbered with true transcript line numbers
        (``[n]``). Case-SENSITIVE, because it exists for word-for-word
        recovery: quote the exact phrase you remember. An empty ``query``
        returns the most recent ``window`` lines of the transcript.
        """
        if self._own_transcript is None:
            return "Error: no session transcript is bound to this memory instance."
        if not self._own_transcript.is_file():
            return ("Error: this conversation has no transcript yet — nothing "
                    "has been summarized away from the live window.")
        if not isinstance(window, int) or window < 1:
            window = 20
        if not isinstance(max_matches, int) or max_matches < 1:
            max_matches = 5
        try:
            lines = self._own_transcript.read_text(
                encoding="utf-8", errors="replace").splitlines()
        except OSError as e:
            return f"Error: could not read the transcript: {e}"
        if not lines:
            return "(transcript exists but is empty)"

        def numbered(start: int, stop: int) -> list[str]:
            return [f"[{k + 1}] {lines[k]}" for k in range(start, stop)]

        if not isinstance(query, str):
            return "Error: query must be a string."
        if query == "":
            stop = len(lines)
            start = max(0, stop - min(window, _MAX_RECALL_LINES))
            n = stop - start
            head = (f"Verbatim tail of this conversation's transcript — "
                    f"the last {n} line(s):\n")
            return head + "\n".join(numbered(start, stop))

        matches = [i for i, line in enumerate(lines) if query in line]
        if not matches:
            return (f"No verbatim match for {query!r} in this conversation's "
                    "transcript. The search is case-sensitive and exact: try a "
                    "shorter phrase copied from what you remember, or search "
                    "with a different casing.")
        shown = matches[:max_matches]
        before = window // 2
        after = window - 1 - before
        spans: list[list[int]] = []
        for i in shown:
            s, e = max(0, i - before), min(len(lines), i + after + 1)
            if spans and s <= spans[-1][1]:
                spans[-1][1] = max(spans[-1][1], e)
            else:
                spans.append([s, e])
        # Bound total output even when windows are requested very wide.
        budget = _MAX_RECALL_LINES
        blocks: list[str] = []
        for s, e in spans:
            if budget <= 0:
                break
            e = min(e, s + budget)
            budget -= e - s
            blocks.append("\n".join(numbered(s, e)))
        if len(matches) > len(shown):
            head = (f"{len(matches)} verbatim match(es) for {query!r}; "
                    f"windows around the first {len(shown)}:")
        else:
            head = f"{len(matches)} verbatim match(es) for {query!r}:"
        return (head + "\n\n" + "\n\n...\n\n".join(blocks)
                + "\n\nThese lines are the exact original text, numbered by "
                  "true transcript line number.")

    def list_memory_files(self) -> str:
        """Lists every memory file this agent may read, with its size.

        That is this conversation's own transcript plus the shared ltm/ memory
        tree — nothing else on the machine. Use it to see which memories exist
        before searching or reading them.
        """
        files: list[Path] = []
        for path in sorted(self.root.rglob("*")):
            if (path.is_file() and "__pycache__" not in path.parts
                    and path.name != _INBOX_NAME
                    and not self._is_foreign_transcript(path.resolve())):
                files.append(path)
                if len(files) >= _MAX_LISTED:
                    break
        if not files:
            return "(no memory files yet)"
        rows = []
        for path in files:
            try:
                size = path.stat().st_size
            except OSError:
                size = -1
            rows.append(f"{self._rel(path)}  ({size} bytes)")
        return "\n".join(rows)

    # -- internal LTM memory-write tools (consolidation loop only) ------------

    def edit_memory_file(self, file_path: str, old_text: str, new_text: str,
                         replace_all: bool = False) -> str:
        """Edits one existing markdown memory file under ltm/ by exact text replacement.

        Prefer this over write_memory_file for small changes: it replaces
        ``old_text`` (copied verbatim from the read tools) with ``new_text``
        without you having to re-emit the whole file. ``old_text`` must match
        exactly one place unless ``replace_all`` is set. ``new_text=""`` deletes
        the matched text. The file must already exist — use write_memory_file
        to create one.
        """
        try:
            target = self._resolve_ltm_write(file_path)
        except ValueError as e:
            return f"Error: {e}"
        if not isinstance(old_text, str) or not old_text:
            return "Error: old_text must be a non-empty string copied exactly from the memory file."
        if not isinstance(new_text, str):
            return 'Error: new_text must be a string (pass "" to delete the matched text).'
        if not isinstance(replace_all, bool):
            replace_all = bool(replace_all)
        if not target.is_file():
            return (f"Error: memory file '{file_path}' does not exist. "
                    "Use write_memory_file to create it.")
        try:
            text = target.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            return f"Error: could not read '{file_path}': {e}"
        count = text.count(old_text)
        if count == 0:
            return (f"Error: old_text not found in {self._rel(target)}. "
                    "Read the file and copy the exact text (whitespace included).")
        if count > 1 and not replace_all:
            return (f"Error: old_text matches {count} places in "
                    f"{self._rel(target)}; widen it until it is unique, or set "
                    "replace_all=true.")
        new_text_full = text.replace(old_text, new_text)
        try:
            atomic_write(target, new_text_full)
        except OSError as e:
            return f"Error: could not edit '{file_path}': {e}"
        replaced = count if replace_all else 1
        return (f"Edited {self._rel(target)}: replaced {replaced} "
                f"occurrence(s) of {len(old_text)} chars with {len(new_text)} chars.")

    def write_memory_file(self, file_path: str, content: str) -> str:
        """Atomically creates or completely replaces a markdown memory file under ltm/.

        Use for new topic files or full rewrites; for a small change to an
        existing memory prefer edit_memory_file.
        """
        try:
            target = self._resolve_ltm_write(file_path)
        except ValueError as e:
            return f"Error: {e}"
        if not isinstance(content, str):
            return "Error: content must be a string."
        try:
            atomic_write(target, content)
        except OSError as e:
            return f"Error: could not write '{file_path}': {e}"
        return f"Wrote {len(content)} chars to memory file {self._rel(target)}"

    def delete_memory_file(self, file_path: str) -> str:
        """Deletes an obsolete markdown memory file under ltm/."""
        try:
            target = self._resolve_ltm_write(file_path)
        except ValueError as e:
            return f"Error: {e}"
        if not target.is_file():
            return f"Error: memory file '{file_path}' does not exist."
        if target.name == "MEMORY.md":
            return "Error: MEMORY.md is the memory index and cannot be deleted. Rewrite it instead."
        try:
            target.unlink()
        except OSError as e:
            return f"Error: could not delete '{file_path}': {e}"
        return f"Deleted memory file {self._rel(target)}"

    # -- helpers ---------------------------------------------------------------

    def _readable_markdown(self) -> list[Path]:
        """Every .md the current agent may search: ltm/ + its own transcript.

        Other conversations' transcripts are excluded — that is the isolation
        the per-session file exists to provide — and so is ltm/_inbox.md, which
        is memkit's transient staging file (its facts are folded into topic
        files, then it is deleted), not something the agent should cite.
        """
        files: list[Path] = []
        for path in sorted(self.root.rglob("*.md")):
            if (path.is_file() and path.name != _INBOX_NAME
                    and not self._is_foreign_transcript(path.resolve())):
                files.append(path)
                if len(files) >= _MAX_LISTED:
                    break
        return files

    def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """Dispatch one tool call. Never raises — errors come back as text."""
        fn = {
            "read_memory_file": self.read_memory_file,
            "search_memory_lines": self.search_memory_lines,
            "recall_transcript": self.recall_transcript,
            "list_memory_files": self.list_memory_files,
            "edit_memory_file": self.edit_memory_file,
            "write_memory_file": self.write_memory_file,
            "delete_memory_file": self.delete_memory_file,
        }.get(name)
        if fn is None:
            return f"Error: unknown tool '{name}'."
        if not isinstance(arguments, dict):
            return f"Error: arguments for '{name}' must be an object."
        allowed = {
            "read_memory_file": ("file_path", "offset", "limit"),
            "search_memory_lines": ("query", "file_path", "max_matches"),
            "recall_transcript": ("query", "window", "max_matches"),
            "list_memory_files": (),
            "edit_memory_file": ("file_path", "old_text", "new_text", "replace_all"),
            "write_memory_file": ("file_path", "content"),
            "delete_memory_file": ("file_path",),
        }[name]
        extra = set(arguments) - set(allowed)
        if extra:
            return f"Error: unexpected argument(s) for '{name}': {sorted(extra)}"
        try:
            return fn(**arguments)
        except Exception as e:  # a bad tool call must not kill the agent loop
            return f"Error: {name} failed: {type(e).__name__}: {e}"


TOOL_SPECS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "read_memory_file",
            "description": (
                "Reads a stored MEMORY file — a topic memory under ltm/ or this "
                "conversation's own transcript — NOT a project file. This tool "
                "and its siblings are memory tools and can only see inside the "
                "memory folder; they are distinct from any generic file tools "
                "the host project may provide. Reads up to 100 lines by default; "
                "use offset/limit to page through a large memory file and never "
                "load a whole transcript at once. Lines come back numbered by "
                "true file line number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path relative to the memory folder, e.g. 'ltm/user.md' or 'stm/<session_id>.md'."},
                    "offset": {"type": "integer", "description": "Zero-based line index to start at.", "default": 0},
                    "limit": {"type": "integer", "description": "Maximum number of lines to return.", "default": 100},
                },
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_memory_lines",
            "description": (
                "Searches this conversation's transcript and the shared memory "
                "files (ltm/) for a text snippet and returns matching line "
                "numbers, so you can read just those lines with "
                "read_memory_file instead of wasting tokens on whole memory "
                "files. Case-insensitive; for the exact, word-for-word text of "
                "this conversation prefer recall_transcript."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Text to look for (case-insensitive substring)."},
                    "file_path": {"type": "string", "description": "Optional single memory file to search."},
                    "max_matches": {"type": "integer", "description": "Maximum matches to return.", "default": 20},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "recall_transcript",
            "description": (
                "The ONE memory tool for recovering the exact original words of "
                "THIS conversation that summarization dropped. It reads only "
                "this conversation's own transcript file "
                "(stm/<session_id>.md) and returns verbatim line windows around "
                "each match, numbered with true transcript line numbers. The "
                "search is CASE-SENSITIVE and "
                "exact — quote the phrase as precisely as you remember it. "
                "Whenever the user asks what was said word-for-word (a poem, a "
                "quote, a name, a config value from earlier), you MUST answer "
                "from this tool's output, never from the summary. An empty "
                "query returns the last `window` lines of the transcript."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Exact phrase to recall (case-sensitive). Empty = the recent tail of the transcript.", "default": ""},
                    "window": {"type": "integer", "description": "Lines to return around each match (and for the empty-query tail).", "default": 20},
                    "max_matches": {"type": "integer", "description": "Maximum matches to show windows for.", "default": 5},
                },
                "required": [],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_memory_files",
            "description": (
                "Lists every MEMORY file you may read with its size: this "
                "conversation's transcript plus the shared ltm/ memory tree. "
                "It never lists project files and cannot reach outside the "
                "memory folder."
            ),
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_memory_file",
            "description": (
                "Edits one EXISTING markdown memory file under ltm/ by replacing "
                "an exact string with a new one — the right tool for small "
                "changes to a memory (fixing a fact, updating a bullet). "
                "old_text must be copied verbatim from read_memory_file and must "
                "match exactly one place, unless replace_all is true. Pass "
                "new_text=\"\" to delete the matched text. Use "
                "write_memory_file instead to create a file or rewrite it whole."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path under ltm/, e.g. 'ltm/user.md'. The file must exist."},
                    "old_text": {"type": "string", "description": "Exact text to replace, copied verbatim from the memory file."},
                    "new_text": {"type": "string", "description": "Replacement text. \"\" deletes the matched text."},
                    "replace_all": {"type": "boolean", "description": "Replace every occurrence instead of requiring a unique match.", "default": False},
                },
                "required": ["file_path", "old_text", "new_text"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_memory_file",
            "description": (
                "Creates or completely replaces a markdown MEMORY file under "
                "ltm/ — for new topic memories and full rewrites, not small "
                "edits (use edit_memory_file for those). Pass the full new "
                "content, not a diff."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {"type": "string", "description": "Path under ltm/, e.g. 'ltm/user.md'."},
                    "content": {"type": "string", "description": "Complete new contents of the memory file."},
                },
                "required": ["file_path", "content"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delete_memory_file",
            "description": (
                "Deletes an obsolete markdown MEMORY file under ltm/. "
                "MEMORY.md (the memory index) cannot be deleted — rewrite it "
                "with edit_memory_file or write_memory_file."
            ),
            "parameters": {
                "type": "object",
                "properties": {"file_path": {"type": "string", "description": "Path under ltm/."}},
                "required": ["file_path"],
                "additionalProperties": False,
            },
        },
    },
]


def read_specs(names: list[str] | None = None) -> list[dict[str, Any]]:
    """Tool specs, optionally filtered to a subset of names."""
    if names is None:
        return json.loads(json.dumps(TOOL_SPECS))
    keep = set(names)
    return [s for s in json.loads(json.dumps(TOOL_SPECS)) if s["function"]["name"] in keep]


def atomic_write(target: Path, content: str) -> None:
    """Write text durably: temp file in the same dir, fsync, then os.replace.

    Same-directory temp means the replace is atomic on both POSIX and Windows.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(content)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
