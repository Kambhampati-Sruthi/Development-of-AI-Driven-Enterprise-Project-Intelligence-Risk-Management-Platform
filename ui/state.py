"""Session state, upload processing and pipeline orchestration for the UI."""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List

import streamlit as st

from core import docgen
from core.agents import run_pipeline
from core.health import HealthReport, compute_health
from core.ingestion import IngestionError, parse_file
from core.models import PipelineResult
from core.rag import KnowledgeBase

from .nav import PAGES, PAGE_ASSISTANT

log = logging.getLogger(__name__)


def init_state() -> None:
    ss = st.session_state
    ss.setdefault("kb", KnowledgeBase())
    ss.setdefault("pipe", PipelineResult())
    ss.setdefault("health", HealthReport())
    ss.setdefault("docs", {"stories": [], "stories_df": None, "risks_df": None, "actions_df": None})
    ss.setdefault("chat", [])
    ss.setdefault("upload_log", [])
    ss.setdefault("processed", set())
    ss.setdefault("uploader_key", 0)
    ss.setdefault("pending", None)
    ss.setdefault("notices", [])
    ss.setdefault("queued_question", None)
    ss.setdefault("nav", PAGES[0])
    ss.setdefault("feedback", {})
    ss.setdefault("validation", None)
    ss.setdefault("change_log", [])


def goto(page: str) -> None:
    """Button callback: switch page (callbacks run before the rerun, so widget state may be set here)."""
    st.session_state["nav"] = page


def ask(question: str) -> None:
    """Button callback: send a question to the assistant and open it."""
    q = (question or "").strip()
    if q:
        st.session_state["queued_question"] = q
        st.session_state["nav"] = PAGE_ASSISTANT


def ask_from_box() -> None:
    ask(st.session_state.get("dash_q", ""))
    st.session_state["dash_q"] = ""


def notify(kind: str, message: str) -> None:
    st.session_state.notices.append((kind, message))


def flush_notices() -> None:
    for kind, msg in st.session_state.notices:
        {"error": st.error, "warning": st.warning, "success": st.success}.get(kind, st.info)(msg)
    st.session_state.notices = []


def request(action: str, **payload: Any) -> None:
    """Queue an action; it executes at the top of the next run so every tab renders fresh data."""
    st.session_state.pending = {"action": action, **payload}
    st.rerun()


def refresh_analysis() -> None:
    """Rebuild the vector index, re-run every agent, and regenerate documents + health."""
    ss = st.session_state
    kb: KnowledgeBase = ss.kb
    bar = st.progress(0.0, text="Building vector index…")
    try:
        kb.build_index()
        if kb.index_error:
            notify("warning", kb.index_error)
    except Exception as exc:
        log.exception("Index build failed")
        notify("error", f"The vector index could not be built: {exc}")
    try:
        ss.pipe = run_pipeline(kb, progress=lambda f, m: bar.progress(min(f, 1.0), text=m))
        for run in ss.pipe.runs:
            if run.status == "error":
                notify("error", f"{run.name} failed: {run.message}. Other agents were not affected.")
    except Exception as exc:
        log.exception("Pipeline failed")
        ss.pipe = PipelineResult(generated_at=datetime.now())
        notify("error", f"The agent pipeline failed unexpectedly: {exc}")
    try:
        ss.health = compute_health(kb, ss.pipe)
    except Exception as exc:
        log.exception("Health scoring failed")
        ss.health = HealthReport()
        notify("error", f"Health scoring failed: {exc}")
    try:
        stories = docgen.user_stories(ss.pipe)
        ss.docs = {"stories": stories, "stories_df": docgen.stories_frame(stories),
                   "risks_df": docgen.risk_register(ss.pipe), "actions_df": docgen.action_register(ss.pipe)}
    except Exception as exc:
        log.exception("Document generation failed")
        notify("error", f"Documentation generation failed: {exc}")
    bar.empty()


def snapshot() -> Dict[str, Any]:
    """Headline metrics, used to show what each incremental upload changed."""
    ss = st.session_state
    pipe, h = ss.pipe, ss.health
    return {"docs": len(ss.kb.documents), "chunks": len(ss.kb.chunks), "risks": len(pipe.risk.risks),
            "blockers": sum(1 for b in pipe.blockers.blockers if b.status == "Open"), "actions": len(pipe.blockers.actions),
            "health": h.overall if h.has_data else None}


def process_uploads(files: List[Any]) -> None:
    ss = st.session_state
    added = 0
    before = snapshot()
    names: List[str] = []
    for f in files:
        sig = (f.name, f.size)
        if sig in ss.processed and f.name in ss.kb.documents:
            ss.upload_log.append({"file": f.name, "status": "skipped", "detail": "Already indexed (same name and size)."})
            continue
        try:
            doc = parse_file(f.name, f.getvalue())
            replaced = doc.name in ss.kb.documents
            ss.kb.add_document(doc)
            ss.processed.add(sig)
            added += 1
            names.append(("↻ " if replaced else "+ ") + doc.name)
            ss.upload_log.append({"file": doc.name, "status": "indexed",
                                  "detail": f"{doc.kind_label}; {len(doc.segments)} segment(s)"
                                            + (f"; {'; '.join(doc.warnings)}" if doc.warnings else "")})
        except IngestionError as exc:
            ss.upload_log.append({"file": f.name, "status": "failed", "detail": str(exc)})
            notify("error", f"**{f.name}** could not be ingested — {exc}")
        except Exception as exc:                       # defensive: one bad file never blocks the rest
            log.exception("Unexpected ingestion error")
            ss.upload_log.append({"file": f.name, "status": "failed", "detail": f"Unexpected error: {exc}"})
            notify("error", f"**{f.name}** failed unexpectedly — {exc}")
    ss.upload_log = ss.upload_log[-40:]
    if added or ss.kb.documents:
        refresh_analysis()
    if added:
        ss.change_log.append({"time": datetime.now(), "files": names, "before": before, "after": snapshot()})
        ss.change_log = ss.change_log[-15:]
        notify("success", f"Knowledge base updated with {added} document(s); the index and every agent were refreshed automatically.")


def _auto_ingest() -> None:
    """Incremental upload: files dropped into the uploader are ingested immediately, with no extra click."""
    ss = st.session_state
    files = ss.get(f"uploader_{ss.uploader_key}") or []
    if not files:
        return
    try:
        with st.spinner("Updating the knowledge base…"):
            process_uploads(list(files))
    except Exception as exc:
        log.exception("Incremental ingestion failed")
        notify("error", f"Could not update the knowledge base: {exc}")
    ss.uploader_key += 1                              # clears the uploader so the next batch starts fresh


def handle_pending() -> None:
    ss = st.session_state
    _auto_ingest()
    p, ss.pending = ss.pending, None
    if not p:
        return
    try:
        if p["action"] == "process":
            files = ss.get(f"uploader_{ss.uploader_key}") or []
            if not files:
                notify("warning", "Choose at least one file to process.")
            else:
                with st.spinner("Parsing documents, indexing and running agents…"):
                    process_uploads(list(files))
                ss.uploader_key += 1
        elif p["action"] == "rerun":
            if ss.kb.is_empty:
                notify("warning", "Upload documents first — the agents need content to analyse.")
            else:
                with st.spinner("Re-running agents…"):
                    refresh_analysis()
                notify("success", "Agents re-run on the current knowledge base.")
        elif p["action"] == "remove":
            for name in p.get("names", []):
                ss.kb.remove_document(name)
                ss.processed = {s for s in ss.processed if s[0] != name}
            if ss.kb.is_empty:
                ss.kb.clear()
                _reset_outputs()
            else:
                refresh_analysis()
            notify("success", f"Removed {len(p.get('names', []))} document(s).")
        elif p["action"] == "reset":
            ss.kb.clear()
            ss.processed = set()
            ss.upload_log = []
            ss.chat = []
            ss.change_log = []
            ss.validation = None
            _reset_outputs()
            ss.uploader_key += 1
            notify("success", "Knowledge base cleared.")
    except Exception as exc:
        log.exception("Action failed")
        notify("error", f"The action could not be completed: {exc}")


def _reset_outputs() -> None:
    ss = st.session_state
    ss.pipe = PipelineResult()
    ss.health = HealthReport()
    ss.docs = {"stories": [], "stories_df": None, "risks_df": None, "actions_df": None}
