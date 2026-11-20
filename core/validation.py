"""Quality validation: retrieval accuracy, assistant grounding, and agent coverage across formats."""
from __future__ import annotations

import re
from typing import Dict, List

import numpy as np
import pandas as pd

from .assistant import NOT_FOUND, answer_question
from .config import TOP_K
from .health import HealthReport
from .models import PipelineResult
from .rag import KnowledgeBase
from .utils import jaccard, truncate

CANONICAL_QUESTIONS = [
    "Are we on track?", "What are our biggest risks?", "What are the open blockers?", "Who owns which work?",
    "What decisions are pending?", "Summarise the project scope", "What does the timeline look like?",
    "Which defects or test findings need attention?",
]
OFF_TOPIC_QUESTIONS = ["What is the capital of France?", "Give me a recipe for chocolate cake"]


def retrieval_validation(kb: KnowledgeBase, samples: int = 12, k: int = TOP_K) -> Dict:
    """Self-test: build a query from each sampled chunk's most distinctive terms and check it is retrieved."""
    vec, mat = kb._vectorizer, kb._matrix
    if vec is None or mat is None or not kb.chunks:
        return {"recall": None, "mrr": None, "n": 0, "rows": pd.DataFrame()}
    eligible = [i for i, c in enumerate(kb.chunks) if len(c.text) >= 80]
    if not eligible:
        return {"recall": None, "mrr": None, "n": 0, "rows": pd.DataFrame()}
    picks = [eligible[int(i)] for i in np.linspace(0, len(eligible) - 1, min(samples, len(eligible)))]
    picks = list(dict.fromkeys(picks))
    names = np.array(vec.get_feature_names_out())
    rows, hits_at_k, rr = [], 0, 0.0
    for idx in picks:
        row = mat.getrow(idx).toarray().ravel()
        order = [j for j in np.argsort(-row) if row[j] > 0 and " " not in names[j]][:5]
        if not order:
            continue
        query = " ".join(names[order])
        results = kb.search(query, k=k, min_score=0.0)
        target = kb.chunks[idx].text
        rank = next((r for r, h in enumerate(results, 1) if h.chunk.chunk_id == kb.chunks[idx].chunk_id
                     or jaccard(h.chunk.text, target) >= 0.9), None)
        hits_at_k += rank is not None
        rr += (1 / rank) if rank else 0.0
        rows.append({"Document": kb.chunks[idx].doc_name, "Location": kb.chunks[idx].locator, "Probe query": query,
                     "Rank": rank if rank else "miss"})
    n = len(rows)
    return {"recall": hits_at_k / n if n else None, "mrr": rr / n if n else None, "n": n, "rows": pd.DataFrame(rows)}


def assistant_validation(kb: KnowledgeBase, pipe: PipelineResult, health: HealthReport) -> Dict:
    rows: List[Dict] = []
    answered = cited = 0
    rel: List[float] = []
    cov: List[float] = []
    for q in CANONICAL_QUESTIONS:
        a = answer_question(kb, pipe, health, q)
        bullets = [l for l in a.text.splitlines() if l.strip().startswith("- ")]
        coverage = (sum(1 for l in bullets if re.search(r"\[\d+\]", l)) / len(bullets)) if bullets else (1.0 if re.search(r"\[\d+\]", a.text) else 0.0)
        ok = a.grounded and a.text != NOT_FOUND
        answered += ok
        cited += bool(a.sources)
        if ok:
            cov.append(coverage)
            rel.append(a.relevance)
        rows.append({"Question": q, "Intent": a.intent, "Answered": "yes" if ok else "no", "Sources": len(a.sources),
                     "Citation coverage": f"{coverage:.0%}", "Confidence": a.confidence, "Relevance": round(a.relevance, 2)})
    refused = 0
    for q in OFF_TOPIC_QUESTIONS:
        a = answer_question(kb, pipe, health, q)
        good = (not a.grounded) or a.text == NOT_FOUND
        refused += good
        rows.append({"Question": q + "  (off-topic control)", "Intent": a.intent, "Answered": "refused ✓" if good else "answered ✗",
                     "Sources": len(a.sources), "Citation coverage": "—", "Confidence": a.confidence, "Relevance": round(a.relevance, 2)})
    n = len(CANONICAL_QUESTIONS)
    return {"answer_rate": answered / n, "citation_rate": cited / n, "avg_coverage": float(np.mean(cov)) if cov else 0.0,
            "avg_relevance": float(np.mean(rel)) if rel else 0.0, "refusal_accuracy": refused / len(OFF_TOPIC_QUESTIONS),
            "rows": pd.DataFrame(rows)}


def agent_validation(kb: KnowledgeBase, pipe: PipelineResult) -> pd.DataFrame:
    """Which documents/formats actually contributed findings to each agent."""
    rows = []
    for name, doc in kb.documents.items():
        s = pipe.scope
        scope_n = (sum(1 for i in s.goals + s.deliverables + s.dated_events if i.doc == name)
                   + sum(1 for p in s.timeline if p.doc == name)
                   + sum(1 for r in s.responsibilities if any(src.startswith(name) for src in r.sources)))
        risk_n = sum(1 for r in pipe.risk.risks if r.doc == name)
        b = pipe.blockers
        blk_n = (sum(1 for x in b.blockers if x.doc == name) + sum(1 for x in b.decisions if x.doc == name)
                 + sum(1 for x in b.actions if x.doc == name))
        total = scope_n + risk_n + blk_n
        rows.append({"Document": name, "Format": f"{doc.ext.upper()} · {doc.kind_label}", "Scope findings": scope_n,
                     "Risk findings": risk_n, "Blocker / action findings": blk_n,
                     "Verdict": "Contributing" if total else "Indexed — no findings (often normal for reference text)"})
    return pd.DataFrame(rows)
