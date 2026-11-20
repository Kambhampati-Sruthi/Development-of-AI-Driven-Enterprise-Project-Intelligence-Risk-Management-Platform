"""AI Project Intelligence & Risk Advisor — Streamlit entry point.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import logging
import os

import streamlit as st

from core.config import APP_NAME, APP_TAGLINE, LLM_KEY_ENV

st.set_page_config(page_title=APP_NAME, page_icon="🧭", layout="wide", initial_sidebar_state="expanded")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

from ui.components import hero, html_block  # noqa: E402  (must follow set_page_config)
from ui.nav import (PAGE_AGENTS, PAGE_ASSISTANT, PAGE_DASHBOARD, PAGE_DOCS, PAGE_HEALTH, PAGE_KB, PAGES)  # noqa: E402
from ui.state import flush_notices, handle_pending, init_state  # noqa: E402
from ui.styles import inject_css  # noqa: E402
from ui.views import (render_agent_center, render_assistant, render_dashboard, render_generated_docs,  # noqa: E402
                      render_health, render_knowledge_base)

VIEWS = {PAGE_DASHBOARD: render_dashboard, PAGE_ASSISTANT: render_assistant, PAGE_KB: render_knowledge_base,
         PAGE_AGENTS: render_agent_center, PAGE_DOCS: render_generated_docs, PAGE_HEALTH: render_health}


def _load_optional_llm_key() -> None:
    """Pick up an optional API key from Streamlit secrets (an environment variable also works)."""
    if os.environ.get(LLM_KEY_ENV):
        return
    try:
        key = st.secrets.get(LLM_KEY_ENV)
        if key:
            os.environ[LLM_KEY_ENV] = str(key)
    except Exception:
        pass                                         # no secrets file — the platform works fully without it


def _sidebar() -> None:
    ss = st.session_state
    with st.sidebar:
        html_block('<div class="side-brand"><div class="side-logo">🧭</div><div><div class="side-title">Project Intelligence<br>&amp; Risk Advisor</div>'
                   '<div class="side-sub">AI-powered delivery insight</div></div></div><div class="side-label">Navigate</div>')
        st.radio("Navigation", PAGES, key="nav", label_visibility="collapsed")
        stats = ss.kb.stats()
        h = ss.health
        score = f"{h.overall:.0f} · {h.band}" if h.has_data else "—"
        html_block('<div class="side-card"><div class="side-label" style="margin:0 0 6px 0">Knowledge base</div>'
                   f'<div class="side-stat"><span>Documents</span><b>{stats["documents"]}</b></div>'
                   f'<div class="side-stat"><span>Indexed chunks</span><b>{stats["chunks"]}</b></div>'
                   f'<div class="side-stat"><span>Project health</span><b>{score}</b></div></div>')
        st.caption("Upload more files any time — the knowledge base and every agent update automatically.")


def main() -> None:
    inject_css()
    init_state()
    _load_optional_llm_key()
    handle_pending()                                 # queued actions + incremental ingestion run before any page renders
    _sidebar()

    n_docs = len(st.session_state.kb.documents)
    hero(APP_NAME, APP_TAGLINE, f"{n_docs} document(s) indexed" if n_docs else "Ready for your documents")
    flush_notices()

    view = VIEWS.get(st.session_state.nav, render_dashboard)
    try:
        view()
    except Exception as exc:                         # a faulty view must never take down the whole app
        logging.getLogger(__name__).exception("View failed: %s", view.__name__)
        st.error(f"This page hit an unexpected problem ({exc.__class__.__name__}: {exc}). "
                 "Other pages are unaffected — try re-processing your documents.")


main()
