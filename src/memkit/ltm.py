"""Long-term memory: a file-based, Claude Code-style memory tree.

Layout under ``storage_dir/ltm/``:

* ``MEMORY.md``  — the index. One line per topic file:
  ``- [filename.md] — one-line hook``. Loaded into the agent's system prompt
  every turn; the agent reads the pointed-to topic files itself with
  read_memory_file.
* ``<topic>.md`` — free-form markdown, one dated bullet per memory:
  ``- (as of 2026-10-01) the fact`` so consolidation can prune stale ones.
* ``_inbox.md``  — extracted-but-not-yet-consolidated facts (memkit-managed;
  never listed in MEMORY.md).

Maintenance runs on the background worker thread (the sole writer):
extraction folds new facts into the inbox whenever STM archives; when the
total memory exceeds ``max_tokens_ltm`` the consolidation agent loop merges,
prunes and rewrites the tree autonomously. Session end triggers a final
extraction, so short sessions that never hit the cap still leave memories.
"""

from __future__ import annotations

import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from memkit.config import MemoryConfig
from memkit.llm import call_with_retries
from memkit.prompts import (LTM_CONSOLIDATE_SYSTEM, LTM_CONSOLIDATE_USER_HEADER,
                            LTM_EXTRACT_SYSTEM, LTM_EXTRACT_USER_HEADER)
from memkit.tokens import TokenCounter
from memkit.tools import MemoryTools, atomic_write

INDEX_NAME = "MEMORY.md"
INBOX_NAME = "_inbox.md"
_VALID_CATEGORIES = {"USER_FACTS", "PROJECT_FACTS", "FEEDBACK", "REFERENCES"}


class LongTermMemory:
    def __init__(self, config: MemoryConfig, counter: TokenCounter,
                 tools: MemoryTools) -> None:
        self.config = config
        self.counter = counter
        self.tools = tools
        self.ltm_dir = Path(config.ltm_dir)
        self._lock = threading.Lock()
        self._consolidating = False
        self._warned_no_tools = False
        self.ensure_initialized()

    # -- lifecycle -------------------------------------------------------------

    def ensure_initialized(self) -> None:
        self.ltm_dir.mkdir(parents=True, exist_ok=True)
        index = self.ltm_dir / INDEX_NAME
        if not index.exists():
            atomic_write(index, "# Memory Index\n\n(no topic files yet)\n")

    # -- reading -----------------------------------------------------------------

    def read_index(self) -> str:
        try:
            return (self.ltm_dir / INDEX_NAME).read_text(encoding="utf-8")
        except OSError:
            return "(MEMORY.md unavailable)"

    def total_tokens(self) -> int:
        """Tokens of the whole memory tree (index + topics + inbox)."""
        total = 0
        for path in self.ltm_dir.glob("*.md"):
            try:
                total += self.counter.count_text(
                    path.read_text(encoding="utf-8", errors="replace"))
            except OSError:
                continue
        return total

    def over_budget(self) -> bool:
        return self.total_tokens() > self.config.max_tokens_ltm

    # -- extraction -----------------------------------------------------------------

    def extract(self, text: str, source_label: str) -> list[str]:
        """Run the extractor LLM over `text`; append accepted facts to the inbox.

        Returns the fact lines accepted (empty list if none). Safe to call from
        the worker thread only.
        """
        if not text.strip():
            return []
        messages = [
            {"role": "system", "content": LTM_EXTRACT_SYSTEM},
            {"role": "user", "content": LTM_EXTRACT_USER_HEADER.format(
                index=self.read_index(), source_label=source_label) + "\n" + text},
        ]
        try:
            reply = call_with_retries(
                lambda: self.config.llm(messages), self.config.retries,
                self.config.logger)
        except Exception:
            self.config.logger.exception("memkit LTM: extraction LLM failed")
            return []
        facts = self._parse_facts(reply)
        if facts:
            with self._lock:
                inbox = self.ltm_dir / INBOX_NAME
                existing = inbox.read_text(encoding="utf-8") if inbox.exists() else ""
                atomic_write(inbox, existing + "\n".join(facts) + "\n")
        return facts

    def _parse_facts(self, reply: str) -> list[str]:
        out: list[str] = []
        for raw in (reply or "").splitlines():
            line = raw.strip().lstrip("-* ").strip()
            if not line or line.upper() == "NONE":
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 4:
                continue
            category, topic_file, intent, fact = parts[0], parts[1], parts[2], "|".join(parts[3:])
            if category.upper() not in _VALID_CATEGORIES:
                continue
            # sanitize the suggested filename: last path segment, always .md,
            # never empty — so a hostile/weird suggestion still lands in ltm/
            name = topic_file.replace("\\", "/").rstrip("/").split("/")[-1]
            if not name or name in (".", ".."):
                name = "misc"
            if not name.lower().endswith(".md"):
                name += ".md"
            topic_file = name
            if intent.lower() not in ("update", "new"):
                intent = "update"
            if not fact.strip():
                continue
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            out.append(f"{category.upper()} | {topic_file} | {intent} | (as of {today}) {fact.strip()}")
        return out

    # -- consolidation -----------------------------------------------------------------

    def inbox_pending(self) -> bool:
        """True if the inbox file exists and has fact lines."""
        inbox = self.ltm_dir / INBOX_NAME
        try:
            return inbox.is_file() and bool(
                inbox.read_text(encoding="utf-8").strip())
        except OSError:
            return False

    def take_inbox(self) -> list[str]:
        """Atomically read and clear the inbox; returns fact lines."""
        with self._lock:
            inbox = self.ltm_dir / INBOX_NAME
            if not inbox.exists():
                return []
            lines = [l.strip() for l in
                     inbox.read_text(encoding="utf-8").splitlines() if l.strip()]
            # Windows can hold a just-written file briefly (AV/indexer scan);
            # a locked inbox only delays cleanup, never costs data.
            for attempt in range(5):
                try:
                    inbox.unlink()
                    break
                except PermissionError:
                    time.sleep(0.05 * (attempt + 1))
            return lines

    def flush_inbox_to_files(self) -> int:
        """Append extracted facts to their topic files (creating new ones).

        Cheap and deterministic — no LLM call. The consolidation agent loop
        is reserved for merging/pruning once the budget is actually hit.
        Returns the number of facts written; on failure the unwritten facts
        go back to the inbox for the next pass.
        """
        facts = self.take_inbox()
        if not facts:
            return 0
        written = 0
        try:
            for i, line in enumerate(facts):
                parts = [p.strip() for p in line.split("|", 3)]
                if len(parts) < 4:
                    continue
                _category, topic_file, _intent, fact_text = parts
                fact_text = fact_text.strip()
                if not fact_text:
                    continue
                target = self.ltm_dir / topic_file
                try:
                    existing = (target.read_text(encoding="utf-8")
                                if target.exists() else "")
                except OSError:
                    continue
                if fact_text in existing:  # already remembered verbatim
                    written += 1
                    continue
                if not existing.strip():
                    title = (topic_file.removesuffix(".md")
                             .replace("_", " ").replace("-", " ").title())
                    existing = f"# {title}\n"
                atomic_write(target, existing.rstrip() + "\n" + f"- {fact_text}\n")
                written += 1
            self._repair_index()
        except Exception:
            self.restore_inbox(facts[written:])
            raise
        return written

    def restore_inbox(self, lines: list[str]) -> None:
        """Put fact lines back into the inbox.

        APPENDS (skipping verbatim duplicates) instead of overwriting: an
        extraction that completed while consolidation ran may have added new
        lines meanwhile — a blind overwrite would discard them."""
        if not lines:
            return
        with self._lock:
            inbox = self.ltm_dir / INBOX_NAME
            try:
                existing = (inbox.read_text(encoding="utf-8")
                            if inbox.exists() else "")
            except OSError:
                existing = ""
            have = {l.strip() for l in existing.splitlines() if l.strip()}
            to_add = [l for l in lines if l.strip() and l.strip() not in have]
            if not to_add:
                return
            body = existing if (not existing or existing.endswith("\n")) \
                else existing + "\n"
            atomic_write(inbox, body + "\n".join(to_add) + "\n")

    def consolidate(self) -> bool:
        """Merge inbox facts (if any) and prune the memory tree via the
        tool-calling agent loop.

        Returns True if the agent finished a pass. Never raises — a failed
        consolidation leaves the previous state intact and facts return to
        the inbox for the next pass.

        Without a tool-calling LLM (``llm_tools``), there is no agent to run:
        this logs once and returns False. Facts are still persisted verbatim
        into topic files by :meth:`flush_inbox_to_files`, so memory keeps
        working in inbox-only mode; only semantic merging/pruning is skipped.
        """
        from memkit.agent_loop import run_agent_loop

        if self.config.llm_tools is None:
            if not self._warned_no_tools:
                self._warned_no_tools = True
                self.config.logger.warning(
                    "memkit LTM: no tool-calling llm_tools, so consolidation "
                    "is skipped — extracted facts are still appended to topic "
                    "files, but the tree will keep growing past "
                    "max_tokens_ltm. Pass a tool-capable llm (or llm_tools) "
                    "to enable automatic merging and pruning.")
            return False

        with self._lock:
            if self._consolidating:
                return False
            self._consolidating = True
        try:
            facts = self.take_inbox()
            facts_text = "\n".join(facts) if facts else "(none — review and prune only)"
            max_ltm = self.config.max_tokens_ltm
            max_index = max(60, int(max_ltm * 0.10))
            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            system = LTM_CONSOLIDATE_SYSTEM.format(
                max_ltm_tokens=max_ltm, max_index_tokens=max_index, today=today)
            user = LTM_CONSOLIDATE_USER_HEADER.format(facts=facts_text)
            try:
                run_agent_loop(
                    llm_tools=self.config.llm_tools,
                    tools=self.tools,
                    system=system,
                    user=user,
                    max_steps=self.config.agent_loop_max_steps,
                    logger=self.config.logger,
                    retries=self.config.retries,
                )
            except Exception:
                self.config.logger.exception(
                    "memkit LTM: consolidation failed; facts returned to inbox")
                self.restore_inbox(facts)
                return False
            # Safety net: ensure the index exists and mentions every topic file.
            self._repair_index()
            return True
        finally:
            with self._lock:
                self._consolidating = False

    def _repair_index(self) -> None:
        index_path = self.ltm_dir / INDEX_NAME
        try:
            index = index_path.read_text(encoding="utf-8") if index_path.exists() else ""
        except OSError:
            index = ""
        topics = sorted(p for p in self.ltm_dir.glob("*.md")
                        if p.name not in (INDEX_NAME, INBOX_NAME))
        # Match on parsed [name] entries, not raw substring — a differently
        # formatted index line ("- [ name.md ] ..." or bold) must not make us
        # append a second entry for the same file.
        listed = set(_index_entries(index))
        missing = [p.name for p in topics if p.name not in listed]
        stale = [n for n in _index_entries(index)
                 if not (self.ltm_dir / n).exists() and n != "_inbox.md"]
        if not missing and not stale:
            if not index.strip():
                atomic_write(index_path, "# Memory Index\n\n(no topic files yet)\n")
            return
        # Decide staleness from the same parsed entries used above — a raw
        # "[name]" substring test would keep a restyled line for one file while
        # dropping an exact one for another on the very same line. A line whose
        # entries are ALL gone is dropped; a line that mixes live and gone
        # entries keeps its live ones and has the dead ones cut out of it (the
        # index must never advertise a file that is not there). Headers and
        # prose (no entries) are never dropped, and the "no topic files yet"
        # placeholder goes away as soon as the tree has a real topic file.
        lines: list[str] = []
        for line in index.splitlines():
            entries = _index_entries(line)
            if entries and all(n in stale for n in entries):
                continue
            if entries and any(n in stale for n in entries):
                kept = _drop_stale_entries(line, stale)
                if kept:
                    lines.append(kept)
                continue
            if line.strip() == "(no topic files yet)" and topics:
                continue
            lines.append(line)
        for name in missing:
            hook = _first_line(self.ltm_dir / name) or "no description yet"
            lines.append(f"- [{name}] — {hook[:120]}")
        body = "\n".join(lines).rstrip()
        if not body.strip():  # everything was stale: leave a valid, empty index
            body = "# Memory Index\n\n(no topic files yet)"
        atomic_write(index_path, body + "\n")


_ENTRY_RE = re.compile(r"\[\s*([\w.\-]+\.md)\s*\]")


def _index_entries(index: str) -> list[str]:
    # tolerate whitespace inside the brackets so a lightly restyled index line
    # is still recognised as listing that file (else repair appends a duplicate)
    return _ENTRY_RE.findall(index)


def _drop_stale_entries(line: str, stale: list[str]) -> str:
    """Cut the entries naming a deleted file out of an index line, keeping the
    live ones and the prose around them (``- [a.md] / [b.md] — pair`` becomes
    ``- [a.md] — pair`` when b.md is gone). Returns "" if nothing of the line
    is left, so a line that was nothing but stale entries disappears."""
    dead = set(stale)

    def _keep(m: re.Match) -> str:
        return "" if m.group(1) in dead else m.group(0)

    kept = _ENTRY_RE.sub(_keep, line)
    # The entries are joined by separators that must go with them, or a lone
    # survivor is left with dangling punctuation: strip any / , or — run that
    # now sits at the start or end of the line.
    kept = re.sub(r"^[\s/,;|\-—]+", "", kept)
    kept = re.sub(r"[\s/,;|\-—]+$", "", kept)
    return kept.strip() if _index_entries(kept) else ""


def _first_line(path: Path) -> str:
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                return line.lstrip("- ").strip()
    except OSError:
        pass
    return ""
