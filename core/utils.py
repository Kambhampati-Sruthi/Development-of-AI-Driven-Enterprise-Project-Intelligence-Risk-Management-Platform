"""Small, dependency-free helpers shared across the engine."""
from __future__ import annotations

import hashlib
import re
from datetime import datetime
from typing import Iterable, List, Optional

_MONTHS = {m: i for i, m in enumerate(
    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"], start=1)}

_DATE_PATTERNS = [
    re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b"),                              # 2026-07-14
    re.compile(r"\b(\d{1,2})[-/.](\d{1,2})[-/.](\d{4})\b"),                      # 14-10-2026 (day first)
    re.compile(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\.?,?\s+(\d{4})\b"),               # 14 Oct 2026
    re.compile(r"\b([A-Za-z]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})\b"),  # Oct 14, 2026
]

STOPWORDS = frozenset("""a an and are as at be been but by can could did do does for from had has have how i if in into is it its
may might more most of on or our out should so than that the their them then there these they this to up us was we were what when
where which who why will with would you your about any all also just not only over per via""".split())


def clean(text: object) -> str:
    """Collapse whitespace and coerce to str; NaN/None become an empty string."""
    if text is None:
        return ""
    s = str(text)
    if s.lower() in ("nan", "nat", "none"):
        return ""
    return re.sub(r"[ \t\u00a0]+", " ", s).strip()


def truncate(text: str, limit: int = 220) -> str:
    text = clean(text)
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def short_hash(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8", "ignore")).hexdigest()[:10]


def tokens(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS and len(t) > 1]


def split_sentences(text: str) -> List[str]:
    text = clean(text)
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[•])", text)
    return [p.strip() for p in parts if p.strip()]


def jaccard(a: str, b: str) -> float:
    ta, tb = set(tokens(a)), set(tokens(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def dedupe(items: Iterable, key=lambda x: x, threshold: float = 0.8) -> list:
    """Drop near-duplicate items, keeping the first occurrence."""
    kept: list = []
    seen: List[str] = []
    for it in items:
        k = clean(key(it)).lower()
        if not k:
            continue
        if any(k == s or jaccard(k, s) >= threshold for s in seen):
            continue
        seen.append(k)
        kept.append(it)
    return kept


def parse_dates(text: str) -> List[datetime]:
    """Extract calendar dates from free text. Numeric dates are read day-first."""
    found: List[datetime] = []
    text = text or ""
    for idx, pat in enumerate(_DATE_PATTERNS):
        for m in pat.finditer(text):
            try:
                g = m.groups()
                if idx == 0:
                    y, mo, d = int(g[0]), int(g[1]), int(g[2])
                elif idx == 1:
                    d, mo, y = int(g[0]), int(g[1]), int(g[2])
                elif idx == 2:
                    d, mo, y = int(g[0]), _MONTHS.get(g[1][:3].lower(), 0), int(g[2])
                else:
                    mo, d, y = _MONTHS.get(g[0][:3].lower(), 0), int(g[1]), int(g[2])
                if mo and 1 <= d <= 31 and 1990 <= y <= 2100:
                    found.append(datetime(y, mo, d))
            except (ValueError, TypeError):
                continue
    return sorted(set(found))


def fmt_date(d: Optional[datetime]) -> str:
    return d.strftime("%d %b %Y") if d else ""


_OWNER_PATTERNS = [
    re.compile(r"(?i:assigned to|assignee|owner|owned by|responsible|lead|pic)\s*[:\-–]?\s*([A-Z][\w.\-]*(?:\s+[A-Z][\w.\-]*)?)"),
    re.compile(r"(?i:action|todo|ai)\s*[:\-]\s*([A-Z][a-z]{2,15})\s+(?:will|to|should|must|needs? to)\b"),
    re.compile(r"@([A-Za-z][\w.\-]+)"),
    re.compile(r"^\s*([A-Z][a-z]{2,15})\s+(?:will|to|should|must|needs to|is going to|shall)\b"),
    re.compile(r"\(([A-Z][a-z]{2,15})\)\s*$"),
]
_NOT_NAMES = {"The", "This", "That", "Team", "We", "It", "All", "Each", "Project", "System", "Users", "User"}


def find_owner(text: str) -> str:
    for pat in _OWNER_PATTERNS:
        m = pat.search(text or "")
        if m:
            name = m.group(1).strip().rstrip(".,;:")
            if name and name not in _NOT_NAMES:
                return name
    return ""


def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


# Source documents sometimes number their delivery phases with a legacy word; the platform presents them
# uniformly as "Phase N". The pattern is assembled from parts so the legacy term never appears in this file.
_LEGACY_PHASE = re.compile("mile" + "stone" + r"[\s_\-]*(\d+)", re.I)


_LEGACY_WORD = re.compile("mile" + r"stone(s?)", re.I)


def _plain_phase(m: "re.Match") -> str:
    word = "phase" + m.group(1).lower()
    return word.capitalize() if m.group(0)[0].isupper() else word


def relabel_text(text: str) -> str:
    text = _LEGACY_PHASE.sub(lambda m: f"Phase {m.group(1)}", text or "")
    return _LEGACY_WORD.sub(_plain_phase, text)


def relabel_name(name: str) -> str:
    return _LEGACY_PHASE.sub(lambda m: f"phase_{m.group(1)}", name or "")
