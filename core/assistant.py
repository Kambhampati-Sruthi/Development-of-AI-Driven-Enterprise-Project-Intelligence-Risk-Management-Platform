"""Conversational Project Intelligence Assistant (RAG Q&A, strictly grounded).

Every statement in an answer is built from retrieved document excerpts or from
agent findings that themselves carry document citations. If the documents do not
support an answer, the assistant says so instead of guessing.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import LLM_DEFAULT_MODEL, LLM_KEY_ENV, LLM_MODEL_ENV, MAX_ANSWER_SENTENCES, TOP_K
from .health import HealthReport
from .models import PipelineResult
from .rag import KnowledgeBase
from .utils import tokens, truncate

log = logging.getLogger(__name__)

NOT_FOUND = ("I couldn't find anything about that in the uploaded project documents, so I won't guess. "
             "Try rephrasing, or upload the document that covers it.")

INTENTS: List[Tuple[str, re.Pattern]] = [
    ("owners", re.compile(r"\bowners?\b|who owns|who (?:is|are) (?:responsible|assigned|working)|responsibilit\w+|ownership|team members?|"
                          r"who does what|owns which", re.I)),
    ("actions", re.compile(r"action item|to-?do|next steps?|who (?:is|are) (?:responsible|assigned)|assigned|who owns|owner|"
                           r"what should we do|pending tasks?", re.I)),
    ("decisions", re.compile(r"decision|decide|approval|sign-?off", re.I)),
    ("blockers", re.compile(r"blockers?|blocked|stuck|impediments?|issues?|problems?|defects?|bugs?", re.I)),
    ("risks", re.compile(r"risks?|threats?|danger|concerns?|biggest|worr", re.I)),
    ("on_track", re.compile(r"on track|status|health|progress|behind|ahead|how (?:are|is) we|going|on time|forecast|will we", re.I)),
    ("timeline", re.compile(r"timeline|schedule|deadlines?|\bwhen\b|phases?|sprints?|weeks?|due date|roadmap", re.I)),
    ("scope", re.compile(r"scope|goals?|objectives?|deliverables?|requirements?|features?|modules?|what is (?:this|the) project|"
                         r"project about|summary|overview|about the project", re.I)),
]

SYSTEM_PROMPT = (
    "You are a project intelligence assistant. Answer ONLY from the numbered context excerpts. "
    "Cite every claim with its bracketed number, e.g. [2]. If the context does not contain the answer, reply exactly: "
    f"\"{NOT_FOUND}\" Be concise and concrete; use short bullet points when listing."
)


@dataclass
class Source:
    n: int
    doc: str
    locator: str
    snippet: str
    score: float = 0.0


@dataclass
class Answer:
    text: str
    sources: List[Source] = field(default_factory=list)
    grounded: bool = True
    confidence: str = "Medium"
    intent: str = "general"
    mode: str = "extractive"
    relevance: float = 0.0
    followups: List[str] = field(default_factory=list)


class _Registry:
    def __init__(self, kb: KnowledgeBase) -> None:
        self.kb = kb
        self.items: Dict[Tuple[str, str], Source] = {}

    @staticmethod
    def _record_range(locator: str) -> Tuple[str, int, int]:
        m = re.search(r"^(.*?),?\s*records?\s+(\d+)(?:–(\d+))?$", locator)
        if not m:
            return locator, -1, -1
        lo = int(m.group(2))
        return m.group(1), lo, int(m.group(3) or lo)

    def _find_text(self, doc: str, locator: str) -> str:
        sheet, lo, hi = self._record_range(locator)
        for c in self.kb.chunks:
            if c.doc_name != doc:
                continue
            if c.locator == locator:
                return c.text
            c_sheet, c_lo, c_hi = self._record_range(c.locator)
            if lo >= 0 and c_sheet == sheet and c_lo <= lo and hi <= c_hi:
                return c.text
        return ""

    def cite(self, doc: str, locator: str, fallback: str = "", score: float = 0.0) -> int:
        key = (doc, locator)
        if key not in self.items:
            snippet = self._find_text(doc, locator) or fallback or locator
            self.items[key] = Source(len(self.items) + 1, doc, locator, truncate(snippet, 520), score)
        elif score > self.items[key].score:
            self.items[key].score = score
        return self.items[key].n

    def ordered(self) -> List[Source]:
        return sorted(self.items.values(), key=lambda s: s.n)


FOLLOWUPS = {
    "risks": ["What are the open blockers?", "Are we on track?", "Who owns which work?"],
    "blockers": ["What are our biggest risks?", "What decisions are pending?", "What are the action items?"],
    "decisions": ["What are the open blockers?", "What are the action items?", "Are we on track?"],
    "actions": ["Who owns which work?", "What are the open blockers?", "What does the timeline look like?"],
    "owners": ["What are the action items?", "What are our biggest risks?", "What does the timeline look like?"],
    "on_track": ["What are our biggest risks?", "What are the open blockers?", "What does the timeline look like?"],
    "timeline": ["Are we on track?", "What are the action items?", "What are our biggest risks?"],
    "scope": ["What does the timeline look like?", "Who owns which work?", "What are our biggest risks?"],
    "general": ["Are we on track?", "What are our biggest risks?", "Summarise the project scope"],
}
_FOLLOW_RE = re.compile(r"^(and|also|what about|how about|why|who|when|which|that|those|these|it|they|them|more|"
                        r"tell me more|explain|details?)\b", re.I)


def contextualize(question: str, history: Optional[List[Dict[str, str]]]) -> str:
    """Resolve short follow-ups ("and the owners?") by prefixing the previous user question."""
    prev = next((m["content"] for m in reversed(history or []) if m.get("role") == "user"), "")
    if prev and (len(tokens(question)) <= 3 or _FOLLOW_RE.match(question.strip())):
        return f"{prev} {question}"
    return question


def detect_intent(question: str) -> str:
    for name, rx in INTENTS:
        if rx.search(question):
            return name
    return "general"


def _bullets(lines: List[str]) -> str:
    return "\n".join(f"- {l}" for l in lines)


def _structured(intent: str, reg: _Registry, pipe: PipelineResult, health: HealthReport) -> Tuple[str, List[str]]:
    """Return (intro, evidence lines with citations) from analysed findings."""
    lines: List[str] = []
    intro = ""
    if intent == "risks":
        rs = pipe.risk.risks[:6]
        intro = (f"The uploaded documents surface **{len(pipe.risk.risks)} risk(s)**; the most significant are:"
                 if rs else "No risks were detected in the uploaded documents.")
        for r in rs:
            n = reg.cite(r.doc, r.locator, r.detail or r.title)
            tag = "inferred from structure" if r.evidence_type == "Inferred" else "stated in documents"
            lines.append(f"**{r.severity} · {r.category}** — {truncate(r.title, 170)} _({tag})_ [{n}]")
    elif intent == "blockers":
        open_b = [b for b in pipe.blockers.blockers if b.status == "Open"]
        intro = (f"There are **{len(open_b)} open blocker(s)/issue(s)**:" if open_b
                 else "No open blockers were found in the uploaded documents.")
        for b in open_b[:7]:
            n = reg.cite(b.doc, b.locator, b.text)
            who = f" — owner: {b.owner}" if b.owner else ""
            lines.append(f"**{b.severity}** — {truncate(b.text, 170)}{who} [{n}]")
        for b in [x for x in pipe.blockers.blockers if x.status == "Resolved"][:3]:
            n = reg.cite(b.doc, b.locator, b.text)
            lines.append(f"_Resolved:_ {truncate(b.text, 140)}{' → ' + truncate(b.resolution, 100) if b.resolution else ''} [{n}]")
    elif intent == "decisions":
        ds = pipe.blockers.decisions
        intro = f"**{len(ds)} pending decision(s)** were found:" if ds else "No pending decisions were found in the uploaded documents."
        for d in ds[:8]:
            lines.append(f"{truncate(d.text, 180)} [{reg.cite(d.doc, d.locator, d.text)}]")
    elif intent == "owners":
        rs = pipe.scope.responsibilities
        intro = (f"**{len(rs)} named owner(s)** appear in the documents:" if rs
                 else "No named owners were found in the uploaded documents — assigning owners is recommended.")
        for r in rs[:8]:
            src = r.sources[0].split(" · ")
            doc, loc = (src[0], src[1]) if len(src) > 1 else (src[0], "")
            lines.append(f"**{r.owner}** — {len(r.work)} item(s), e.g. {truncate('; '.join(r.work[:2]), 150)} [{reg.cite(doc, loc, '; '.join(r.work[:2]))}]")
    elif intent == "actions":
        acts = [a for a in pipe.blockers.actions if a.status != "Done"]
        intro = f"**{len(acts)} action item(s)** were identified (highest priority first):" if acts else "No action items were found."
        for a in acts[:9]:
            lines.append(f"**{a.action_id}** {truncate(a.text, 150)} — owner: {a.owner}, due: {a.due}, priority: {a.priority} "
                         f"[{reg.cite(a.doc, a.locator, a.text)}]")
    elif intent == "on_track":
        fc = pipe.risk.forecast
        if health.has_data:
            intro = (f"**{fc.level}.** Estimated on-time delivery likelihood is **{fc.on_time_pct:.0f}%** "
                     f"({fc.confidence.lower()} confidence); overall project health is **{health.overall:.0f}/100 ({health.band})**. "
                     "The signals behind this, from your documents:")
        for r in pipe.risk.risks[:4]:
            lines.append(f"{r.severity} {r.category.lower()} risk — {truncate(r.title, 150)} [{reg.cite(r.doc, r.locator, r.detail or r.title)}]")
        for b in [x for x in pipe.blockers.blockers if x.status == "Open"][:2]:
            lines.append(f"Open blocker — {truncate(b.text, 140)} [{reg.cite(b.doc, b.locator, b.text)}]")
        if fc.total:
            lines.insert(0, f"{fc.done} of {fc.total} tracked work items are complete.")
    elif intent == "timeline":
        tl = pipe.scope.timeline
        intro = "Here is the delivery timeline found in the documents:" if tl else "No timeline information was found."
        for p in tl[:8]:
            n = reg.cite(p.doc, p.locator, p.label)
            lines.append(f"**{truncate(p.label, 70)}**{' (' + p.window + ')' if p.window else ''} — {len(p.tasks)} task(s) [{n}]")
        for e in pipe.scope.dated_events[:3]:
            lines.append(f"{e.meta['date']:%d %b %Y}: {truncate(e.text, 120)} [{reg.cite(e.doc, e.locator, e.text)}]")
    elif intent == "scope":
        if pipe.scope.summary:
            g = pipe.scope.goals[0] if pipe.scope.goals else None
            intro = truncate(pipe.scope.summary, 360)
        for g in pipe.scope.goals[:4]:
            lines.append(f"**Goal** — {truncate(g.text, 170)} [{reg.cite(g.doc, g.locator, g.text)}]")
        for d in pipe.scope.deliverables[:5]:
            lines.append(f"**Deliverable** — {truncate(d.text, 150)} [{reg.cite(d.doc, d.locator, d.text)}]")
    return intro, lines


def _llm(question: str, sources: List[Source], history: Optional[List[Dict[str, str]]], api_key: str) -> Optional[str]:
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        ctx = "\n\n".join(f"[{s.n}] ({s.doc} · {s.locator})\n{s.snippet}" for s in sources)
        msgs = [{"role": m["role"], "content": m["content"]} for m in (history or [])[-4:] if m.get("content")]
        msgs.append({"role": "user", "content": f"Context excerpts:\n{ctx}\n\nQuestion: {question}"})
        resp = client.messages.create(model=os.environ.get(LLM_MODEL_ENV, LLM_DEFAULT_MODEL), max_tokens=700,
                                      system=SYSTEM_PROMPT, messages=msgs)
        text = "".join(getattr(b, "text", "") for b in resp.content).strip()
        return text or None
    except Exception as exc:
        log.warning("LLM synthesis unavailable (%s); using extractive answer.", exc)
        return None


def answer_question(kb: KnowledgeBase, pipe: PipelineResult, health: HealthReport, question: str,
                    history: Optional[List[Dict[str, str]]] = None) -> Answer:
    question = (question or "").strip()
    if not question:
        return Answer("Please type a question about your project.", grounded=False, confidence="Low")
    if kb.is_empty:
        return Answer("The knowledge base is empty. Upload project documents in **Knowledge Base** and I'll answer "
                      "from them, with citations.", grounded=False, confidence="Low")
    try:
        effective = contextualize(question, history)
        intent = detect_intent(question)
        if intent == "general":
            intent = detect_intent(effective)
        reg = _Registry(kb)
        hits = kb.search(effective, k=TOP_K)
        if intent != "general" and len(hits) < 2:
            seeds = {"risks": "risk delay dependency issue", "blockers": "blocked issue defect error pending",
                     "actions": "action assigned owner task", "decisions": "decision approval pending",
                     "timeline": "week sprint schedule date", "scope": "objective goal deliverable module",
                     "on_track": "status progress complete risk", "owners": "assigned owner responsible"}
            extra = kb.search(effective + " " + seeds.get(intent, ""), k=3)
            known = {h.chunk.chunk_id for h in hits}
            hits += [h for h in extra if h.chunk.chunk_id not in known]

        intro, evidence = _structured(intent, reg, pipe, health) if pipe.runs else ("", [])
        best = hits[0].score if hits else 0.0
        if not hits and not evidence:
            return Answer(NOT_FOUND, grounded=False, confidence="Low", intent=intent)

        for h in hits:                                            # register retrieved evidence first-class
            reg.cite(h.chunk.doc_name, h.chunk.locator, h.chunk.text, h.score)
        key_sents = kb.rank_sentences(effective, hits, MAX_ANSWER_SENTENCES)

        parts: List[str] = []
        if intro:
            parts.append(intro)
        if evidence:
            parts.append(_bullets(evidence))
        context_lines: List[str] = []
        if key_sents and (intent == "general" or not evidence or best >= 0.15):
            for s, h, _ in key_sents[: (MAX_ANSWER_SENTENCES if intent == "general" else 2)]:
                n = reg.cite(h.chunk.doc_name, h.chunk.locator, h.chunk.text, h.score)
                context_lines.append(f"{s} [{n}]")
            parts.append(("**From the documents:**\n" if evidence else "") + _bullets(context_lines))
        text = "\n\n".join(p for p in parts if p).strip()
        if not text:
            return Answer(NOT_FOUND, grounded=False, confidence="Low", intent=intent)

        sources = reg.ordered()
        mode = "extractive"
        key = os.environ.get(LLM_KEY_ENV, "")
        if key:
            enriched = _llm(question, sources, history, key)
            if enriched and (re.search(r"\[\d+\]", enriched) or enriched == NOT_FOUND):
                text, mode = enriched, "llm"
        used = {int(n) for n in re.findall(r"\[(\d+)\]", text)}
        sources = [s for s in sources if s.n in used] or sources[:TOP_K]
        remap = {s.n: i for i, s in enumerate(sources, start=1)}          # contiguous citation numbers
        text = re.sub(r"\[(\d+)\]", lambda m: f"[{remap[int(m.group(1))]}]" if int(m.group(1)) in remap else "", text)
        for s_ in sources:
            s_.n = remap[s_.n]
        confidence = "High" if best >= 0.3 or (evidence and best >= 0.15) else "Medium" if best >= 0.15 or evidence else "Low"
        return Answer(text, sources, True, confidence, intent, mode, relevance=min(1.0, best / 0.3) if hits else 0.6,
                      followups=FOLLOWUPS.get(intent, FOLLOWUPS["general"]))
    except Exception as exc:                                       # never surface a stack trace in the chat
        log.exception("Assistant failure")
        return Answer(f"Sorry — I hit an unexpected problem while answering ({exc.__class__.__name__}). "
                      "Please try rephrasing the question.", grounded=False, confidence="Low")
