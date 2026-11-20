"""Shared text/table analysis primitives used by every agent."""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterator, List

from .ingestion import ParsedDocument, TableData
from .rag import KnowledgeBase
from .utils import clean, parse_dates

TASK_VERBS = ("develop", "build", "implement", "design", "study", "validate", "generate", "provide", "create",
              "produce", "document", "test", "deploy", "integrate", "prepare", "conduct", "define", "set up",
              "configure", "write", "review", "complete", "fix", "finalize", "finalise", "migrate", "refactor",
              "analyse", "analyze", "research", "plan", "demo", "present", "submit", "update", "improve",
              "optimize", "optimise", "verify", "evaluate", "train", "publish", "deliver", "ensure")
TASK_VERB_RE = re.compile(r"^\s*(?:" + "|".join(TASK_VERBS) + r")\b", re.I)

STRONG_ISSUE_RE = re.compile(
    r"\b(blocked|critical|severe|failed|failing|fails|crash\w*|overdue|missed|slipp\w+|"
    r"delayed|behind schedule|unable to|cannot|can't|not working|unresolved|escalat\w+)\b", re.I)
GENERIC_STATEMENT_RE = re.compile(r"\b(frequently|often|typically|generally|commonly|usually|struggle|allows?|transforms?|accepts?|enables?|identif(?:y|ies)|extracts?|produces?|generates?)\b", re.I)
RESOLVED_RE = re.compile(r"\b(fix(?:ed)?|resolved|corrected|passed|as expected|successfully|working now|handled|"
                         r"closed|completed|done|sorted|mitigated)\b", re.I)
UNRESOLVED_RE = re.compile(
    r"\b(error|fail\w*|mismatch|bug|exception|crash\w*|not working|issue|incorrect|wrong|pending|unresolved|"
    r"unable|cannot|can't|timeout|blocked|stuck|missing)\b", re.I)

HEADING_HINT = re.compile(r"^[A-Z][A-Za-z0-9 &/,\-()|~]{2,70}:?$")


@dataclass
class Unit:
    text: str
    doc: str
    locator: str
    section: str
    bullet: bool = False
    heading: bool = False

    @property
    def is_plan_content(self) -> bool:
        """True for descriptive plan text (task bullets) rather than status reporting."""
        return bool(TASK_VERB_RE.match(self.text)) or bool(
            re.search(r"module|outcome|objective|goal|deliverable|scope|statement", self.section, re.I))


def build_units(kb: KnowledgeBase) -> List[Unit]:
    """Flatten narrative documents into lines tagged with section context."""
    units: List[Unit] = []
    for doc in kb.documents.values():
        for seg in doc.segments:
            if seg.row_no is not None or seg.locator.endswith(", header"):
                continue
            section = ""
            for raw in seg.text.splitlines():
                line = clean(raw)
                if not line:
                    continue
                if line.startswith("## "):
                    section = line[3:].strip()
                    units.append(Unit(section, doc.name, seg.locator, section, heading=True))
                    continue
                bullet = line.startswith("• ")
                body = line[2:].strip() if bullet else line
                if not bullet and doc.ext in ("pdf", "txt", "md") and len(body) < 80 and HEADING_HINT.match(body) \
                        and (body.endswith(":") or body.isupper()):
                    section = body.rstrip(":")
                    units.append(Unit(section, doc.name, seg.locator, section, heading=True))
                    continue
                body = re.sub(r"^(project statement|statement)\s*:\s*", "", body, flags=re.I)
                parts = [body]
                if len(body) > 260 and not bullet:
                    from .utils import split_sentences
                    parts = split_sentences(body) or [body]
                for p in parts:
                    if len(p) >= 12:
                        units.append(Unit(p, doc.name, seg.locator, section, bullet=bullet))
    return units


# ------------------------------------------------------------------ tables ------
@dataclass
class Row:
    doc: str
    sheet: str
    index: int
    raw: Dict[str, str]
    roles: Dict[str, str]

    def get(self, role: str) -> str:
        col = self.roles.get(role)
        return clean(self.raw.get(col, "")) if col else ""

    def find(self, fragment: str) -> str:
        """Value of the first column whose name contains ``fragment``."""
        for c, v in self.raw.items():
            if fragment in c.lower() and clean(v):
                return clean(v)
        return ""

    @property
    def locator(self) -> str:
        return f"Sheet '{self.sheet}', record {self.index}"

    @property
    def title(self) -> str:
        t = self.get("title")
        if t:
            return t
        texts = [clean(v) for v in self.raw.values() if clean(v)]
        return max(texts, key=len) if texts else ""


def iter_tables(kb: KnowledgeBase, kinds: tuple = ()) -> Iterator[tuple]:
    for doc in kb.documents.values():
        for t in doc.tables:
            if not kinds or t.kind in kinds:
                yield doc, t


def iter_rows(doc: ParsedDocument, t: TableData) -> Iterator[Row]:
    for i, (_, r) in enumerate(t.df.iterrows(), start=1):
        yield Row(doc.name, t.sheet, i, {c: clean(v) for c, v in r.items()}, t.roles)


# ------------------------------------------------------------ status helpers ----
def norm_status(raw: str) -> str:
    s = re.sub(r"^\s*\d+\s*[-–.)]\s*", "", (raw or "").lower()).strip()
    if not s:
        return "unknown"
    if re.search(r"block|on hold|stuck|imped", s):
        return "blocked"
    if re.search(r"delay|overdue|late|at risk|slip", s):
        return "delayed"
    if re.search(r"complete|done|closed|resolved|passed|accepted|finished|released|deployed", s):
        return "done"
    if re.search(r"progress|doing|wip|ongoing|started|review|testing|develop", s):
        return "progress"
    if re.search(r"open|new|to ?do|backlog|not started|planned|pending|assigned|reopen", s):
        return "todo"
    return "unknown"


def row_date(row: Row, *roles: str) -> "datetime | None":
    for r in roles:
        ds = parse_dates(row.get(r))
        if ds:
            return ds[-1]
    return None
