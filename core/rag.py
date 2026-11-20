"""RAG pipeline: chunking, vector indexing (TF-IDF) and cosine-similarity retrieval.

A sparse TF-IDF vector space with word uni+bi-grams is used deliberately: it is
deterministic, runs fully offline, needs no model download, and performs well on
short, terminology-heavy project artifacts. Fixed defaults live in ``config``.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_RELEVANCE, TOP_K
from .ingestion import ParsedDocument, Segment
from .utils import clean, jaccard, split_sentences, tokens, truncate

log = logging.getLogger(__name__)


_SUFFIXES = ("ations", "ation", "ingly", "ings", "ing", "edly", "ied", "ies", "ed", "es", "ly", "s")
_EXPAND = {   # light domain synonym expansion applied to queries only
    "fail": "error mismatch issue bug", "risk": "delay issue blocker dependency", "blocker": "blocked issue impediment pending",
    "defect": "bug issue", "bug": "defect issue", "delay": "slip behind overdue", "track": "status progress complete",
    "action": "todo task assigned", "decision": "decide approval pending", "issue": "error problem mismatch",
}


def stem(tok: str) -> str:
    for suf in _SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= 3:
            return tok[: -len(suf)]
    return tok


def analyze(text: str) -> List[str]:
    """Tokenise, drop stop-words, stem, then add bigrams."""
    uni = [stem(t) for t in tokens(text)]
    return uni + [f"{a} {b}" for a, b in zip(uni, uni[1:])]


def expand_query(query: str) -> str:
    extra = [_EXPAND[stem(t)] for t in tokens(query) if stem(t) in _EXPAND]
    return query + " " + " ".join(extra) if extra else query


class IndexingError(Exception):
    """Raised when the vector index cannot be built."""


@dataclass
class Chunk:
    chunk_id: str
    doc_name: str
    doc_kind: str
    locator: str
    text: str


@dataclass
class Hit:
    chunk: Chunk
    score: float


# ---------------------------------------------------------------- chunking -------
def _pack(text: str, size: int, overlap: int) -> List[str]:
    """Pack paragraphs/sentences into ~size-character windows with overlap."""
    units: List[str] = []
    for para in re.split(r"\n+", text):
        para = clean(para)
        if not para:
            continue
        if len(para) <= size:
            units.append(para)
        else:
            units.extend(split_sentences(para) or [para])
    out: List[str] = []
    cur = ""
    for u in units:
        while len(u) > size:                      # a single unit larger than a window
            if cur:
                out.append(cur)
                cur = ""
            out.append(u[:size])
            u = u[size - overlap:]
        if cur and len(cur) + len(u) + 1 > size:
            out.append(cur)
            tail = cur[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail
            cur = (tail + " " + u).strip()
        else:
            cur = (cur + "\n" + u).strip() if cur else u
    if cur:
        out.append(cur)
    return out


def chunk_document(doc: ParsedDocument, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[Chunk]:
    chunks: List[Chunk] = []
    n = 0

    def emit(text: str, loc: str) -> None:
        nonlocal n
        n += 1
        chunks.append(Chunk(f"{doc.doc_id}-{n:03d}", doc.name, doc.kind_label, loc, text))

    i, segs = 0, doc.segments
    while i < len(segs):
        seg: Segment = segs[i]
        if seg.locator.endswith(", header"):      # schema lines are metadata; rows repeat column names
            i += 1
            continue
        if seg.row_no is not None:               # group consecutive table rows
            group, first, last = [seg.text], seg.row_no, seg.row_no
            sheet = seg.locator.split(", record")[0]
            length = len(seg.text)
            j = i + 1
            while j < len(segs) and segs[j].row_no is not None and segs[j].locator.startswith(sheet) \
                    and length + len(segs[j].text) <= size:
                group.append(segs[j].text)
                length += len(segs[j].text)
                last = segs[j].row_no
                j += 1
            loc = f"{sheet}, record {first}" if first == last else f"{sheet}, records {first}–{last}"
            emit("\n".join(group), loc)
            i = j
            continue
        for piece in _pack(seg.text, size, overlap):
            emit(piece, seg.locator)
        i += 1
    return chunks


# ----------------------------------------------------------- knowledge base ------
class KnowledgeBase:
    """Holds parsed documents, their chunks, and the vector index."""

    def __init__(self) -> None:
        self.documents: Dict[str, ParsedDocument] = {}
        self.chunks: List[Chunk] = []
        self._vectorizer = None
        self._matrix = None
        self.index_error: Optional[str] = None

    # -- lifecycle ---------------------------------------------------------------
    @property
    def is_empty(self) -> bool:
        return not self.documents

    def add_document(self, doc: ParsedDocument) -> int:
        self.documents[doc.name] = doc
        return len(chunk_document(doc))

    def remove_document(self, name: str) -> None:
        self.documents.pop(name, None)

    def clear(self) -> None:
        self.documents.clear()
        self.chunks = []
        self._vectorizer = self._matrix = None
        self.index_error = None

    def build_index(self) -> None:
        """(Re)build chunks and the TF-IDF index from all documents."""
        self.chunks = []
        self._vectorizer = self._matrix = None
        self.index_error = None
        for doc in self.documents.values():
            try:
                self.chunks.extend(chunk_document(doc))
            except Exception as exc:
                log.exception("Chunking failed for %s", doc.name)
                doc.warnings.append(f"Chunking problem: {exc}")
        if not self.chunks:
            return
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(analyzer=analyze, sublinear_tf=True, min_df=1)
            self._matrix = self._vectorizer.fit_transform([c.text for c in self.chunks])
        except Exception as exc:                  # degrade to keyword search, never crash
            log.exception("Vector indexing failed")
            self._vectorizer = self._matrix = None
            self.index_error = f"Vector index unavailable, using keyword fallback ({exc})."

    # -- retrieval ---------------------------------------------------------------
    def search(self, query: str, k: int = TOP_K, min_score: float = MIN_RELEVANCE) -> List[Hit]:
        query = clean(query)
        if not query or not self.chunks:
            return []
        raw_query, query = query, expand_query(query)
        try:
            if self._vectorizer is not None and self._matrix is not None:
                qv = self._vectorizer.transform([query])
                scores = (self._matrix @ qv.T).toarray().ravel()      # L2-normalised ⇒ cosine
            else:
                scores = self._keyword_scores(query)
        except Exception as exc:
            log.warning("Search failed (%s); using keyword fallback", exc)
            scores = self._keyword_scores(raw_query)
        order = np.argsort(-scores)
        hits: List[Hit] = []
        for idx in order[: k * 3]:
            if scores[idx] < min_score:
                break
            if any(jaccard(self.chunks[idx].text, h.chunk.text) >= 0.9 for h in hits):
                continue                                   # near-duplicate evidence from a repeated document
            hits.append(Hit(self.chunks[idx], float(scores[idx])))
            if len(hits) >= k:
                break
        return hits

    def _keyword_scores(self, query: str) -> np.ndarray:
        q = set(tokens(query))
        if not q:
            return np.zeros(len(self.chunks))
        return np.array([len(q & set(tokens(c.text))) / len(q) for c in self.chunks], dtype=float)

    def rank_sentences(self, query: str, hits: List[Hit], limit: int) -> List[Tuple[str, Hit, float]]:
        """Pick the sentences inside the retrieved chunks that best answer the query."""
        cand: List[Tuple[str, Hit]] = []
        for h in hits:
            for line in h.chunk.text.split("\n"):
                line = clean(re.sub(r"^\[[^\]]+\]\s*", "", line.lstrip("•#- ").strip()))
                if len(line) < 25:
                    continue
                for s in (split_sentences(line) if len(line) > 260 else [line]):
                    cand.append((truncate(s, 320), h))
        if not cand:
            return []
        q_tokens = set(tokens(query))
        try:
            if self._vectorizer is not None:
                qv = self._vectorizer.transform([expand_query(query)])
                sv = self._vectorizer.transform([c[0] for c in cand])
                sims = (sv @ qv.T).toarray().ravel()
            else:
                raise ValueError("no vectorizer")
        except Exception:
            sims = np.array([len(q_tokens & set(tokens(c[0]))) / max(len(q_tokens), 1) for c in cand])
        scored = [(c[0], c[1], float(sims[i]) * 0.7 + c[1].score * 0.3) for i, c in enumerate(cand)]
        scored.sort(key=lambda x: -x[2])
        seen, out = [], []
        for s, h, sc in scored:
            key = s.lower()
            if sc < MIN_RELEVANCE or key in seen:
                continue
            seen.append(key)
            out.append((s, h, sc))
            if len(out) >= limit:
                break
        return out

    # -- stats -------------------------------------------------------------------
    def stats(self) -> Dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "characters": sum(d.char_count for d in self.documents.values()),
            "tables": sum(len(d.tables) for d in self.documents.values()),
            "vocabulary": len(self._vectorizer.vocabulary_) if self._vectorizer is not None else 0,
        }
