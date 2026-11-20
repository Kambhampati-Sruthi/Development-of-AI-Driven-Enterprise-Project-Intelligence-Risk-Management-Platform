"""Tab renderers. Every view renders a meaningful state with or without documents."""
from __future__ import annotations

from collections import Counter
from typing import List

import pandas as pd
import streamlit as st

from core import docgen
from core import report, validation
from core.assistant import answer_question
from core.agents import AGENT_CATALOG
from core.config import (CHUNK_OVERLAP, CHUNK_SIZE, HEALTH_WEIGHTS, MAX_FILE_MB, MIN_RELEVANCE, SUPPORTED_EXTENSIONS,
                         TOP_K)
from core.models import PipelineResult
from core.utils import truncate

from .components import (badge, bar_chart, card, dimension_bar, empty_state, gauge, heat_matrix, html_block, human_size,
                         kpi, row_items, section, show_df, stretch)
from .nav import PAGE_ASSISTANT, PAGE_DOCS
from .state import ask, ask_from_box, goto, request

SEV_ORDER = {"High": 0, "Medium": 1, "Low": 2}
ONBOARDING = ("Open Knowledge Base and upload your project files (briefs, sprint backlog, defect tracker, test plan, notes).",
              "Documents are parsed, chunked and indexed as soon as you add them, and every agent runs automatically.",
              "Explore insights here, review generated documents, and ask the assistant anything.")


def _ss():
    return st.session_state


def _open_blockers(pipe: PipelineResult):
    return [b for b in pipe.blockers.blockers if b.status == "Open"]


# =============================================================================
# Dashboard
# =============================================================================
def render_dashboard() -> None:
    ss = _ss()
    kb, pipe, health = ss.kb, ss.pipe, ss.health
    has = health.has_data
    fc = pipe.risk.forecast
    risks = pipe.risk.risks
    open_b = _open_blockers(pipe)
    high = sum(1 for r in risks if r.severity == "High")
    stats = kb.stats()

    with card():
        a, b = st.columns([5, 1])
        with a:
            st.text_input("Ask your project", key="dash_q", placeholder="💬  Ask the assistant — e.g. “Are we on track?” or “What are our biggest risks?”",
                          label_visibility="collapsed", on_change=ask_from_box)
        with b:
            stretch(st.button, "Ask Assistant", type="primary", key="dash_ask", on_click=ask_from_box)

    cols = st.columns(6)
    with cols[0]:
        kpi("Health score", f"{health.overall:.0f}" if has else "—", health.band if has else "Awaiting documents", "❤️",
            "green" if has and health.overall >= 80 else "amber" if has and health.overall >= 60 else "red" if has else "slate")
    with cols[1]:
        kpi("On-time likelihood", f"{fc.on_time_pct:.0f}%" if has else "—", fc.level if has else "Forecast pending", "🎯", "blue")
    with cols[2]:
        kpi("Risks detected", str(len(risks)) if has else "—", f"{high} high severity" if has else "Not analysed", "⚠️",
            "red" if high else "amber" if has else "slate")
    with cols[3]:
        kpi("Open blockers", str(len(open_b)) if has else "—", f"{len(pipe.blockers.decisions)} pending decisions" if has else "Not analysed",
            "🚧", "red" if open_b else "green" if has else "slate")
    with cols[4]:
        kpi("Action items", str(len(pipe.blockers.actions)) if has else "—", "Prioritised & owned" if has else "Not analysed", "✅", "violet")
    with cols[5]:
        kpi("Knowledge base", f"{stats['documents']} docs", f"{stats['chunks']} indexed chunks", "📚", "slate")

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        with card():
            section("Executive summary", "What the uploaded documents say, in one view")
            if has:
                st.markdown(f"**{pipe.scope.project_name or 'Project'}**")
                if pipe.scope.summary:
                    st.write(pipe.scope.summary)
                st.markdown(f"**Delivery outlook —** {fc.narrative}")
                if health.recommendations:
                    st.markdown("**Top recommendation —** " + health.recommendations[0])
            else:
                empty_state("🧭", "No project loaded yet", "Your executive summary, forecast and recommendations appear here once documents are indexed.", ONBOARDING)
    with right:
        with card():
            section("Overall health")
            gauge(health.overall if has else None, health.band if has else "No data", health.color if has else "#94a3b8", 220)
            for d in health.dimensions:
                dimension_bar(d.label, d.score)
            if not has:
                for key, lab in (("scope", "Scope clarity"), ("timeline", "Timeline risk"), ("blockers", "Blocker load")):
                    dimension_bar(lab, None, weight=HEALTH_WEIGHTS[key])

    c1, c2, c3 = st.columns(3)
    with c1:
        with card():
            section("Risk landscape", "By category, and probability × impact")
            if risks:
                cat = pd.DataFrame(Counter(r.category for r in risks).most_common(), columns=["Category", "Risks"])
                bar_chart(cat, "Category", "Risks", height=170)
                heat_matrix(Counter((r.probability, r.impact) for r in risks))
            else:
                st.caption("Risk charts appear after analysis." if not has else "No risks were detected.")
    with c2:
        with card():
            section("Top risks", "Highest severity first")
            row_items([(badge(r.severity), truncate(r.title, 140), f"{r.category} · {r.doc}") for r in risks[:5]],
                      "Risks will be listed here." if not has else "No risks detected.")
    with c3:
        with card():
            section("Blockers & pending decisions")
            rows = [(badge(b.severity), truncate(b.text, 130), f"Blocker · {b.doc}") for b in open_b[:4]]
            rows += [(badge("Decision", "brand"), truncate(d.text, 130), d.doc) for d in pipe.blockers.decisions[:3]]
            row_items(rows, "Blockers and decisions will be listed here." if not has else "No open blockers or pending decisions. 🎉")

    c4, c5 = st.columns(2)
    with c4:
        with card():
            section("Delivery timeline", "Phases, sprints and dated commitments")
            row_items([(badge(p.window or "Phase", "brand"), truncate(p.label, 90), f"{len(p.tasks)} task(s) · {p.doc}")
                       for p in pipe.scope.timeline[:7]], "Timeline phases will appear here.")
    with c5:
        with card():
            section("Who owns what", "Named responsibilities")
            rows = [(badge(r.owner, "neutral"), f"{len(r.work)} item(s): " + truncate("; ".join(r.work[:2]), 110), ", ".join(r.sources[:1]))
                    for r in pipe.scope.responsibilities[:6]]
            row_items(rows, "Owners will appear here once responsibilities are detected.")


# =============================================================================
# Knowledge Base
# =============================================================================
def render_knowledge_base() -> None:
    ss = _ss()
    kb = ss.kb
    with card():
        section("Add project documents", f"PDF, DOCX, CSV, Excel (XLS/XLSX), TXT · up to {MAX_FILE_MB} MB each · incremental: new files are indexed the moment you add them")
        st.file_uploader("Project documents", type=list(SUPPORTED_EXTENSIONS), accept_multiple_files=True,
                         key=f"uploader_{ss.uploader_key}", label_visibility="collapsed")
        b1, b2, _ = st.columns([1.3, 1.1, 3])
        with b1:
            if stretch(st.button, "🔄 Re-run analysis", key="btn_reanalyse", disabled=kb.is_empty):
                request("rerun")
        with b2:
            if stretch(st.button, "🗑️ Clear all", key="btn_reset", disabled=kb.is_empty):
                request("reset")
        st.caption(f"Retrieval is pre-tuned: {CHUNK_SIZE}-character chunks with {CHUNK_OVERLAP} overlap, TF-IDF vector index, "
                   f"cosine similarity, top-{TOP_K} evidence per question. Add new meeting notes or progress updates at any time.")

    if ss.change_log:
        with card():
            section("Latest knowledge-base update", "What your most recent upload changed")
            e = ss.change_log[-1]
            b4, a4 = e["before"], e["after"]

            def delta(key: str, label: str) -> str:
                x, y = b4.get(key), a4.get(key)
                if x is None or y is None:
                    return f"{label}: {y if y is not None else '—'}"
                d = y - x
                return f"{label}: {x:.0f} → {y:.0f}" if key == "health" else f"{label}: {x} → {y} ({'+' if d >= 0 else ''}{d})"
            st.markdown(f"**{e['time']:%H:%M:%S}** · " + ", ".join(e["files"]))
            st.caption(" · ".join([delta("docs", "Documents"), delta("chunks", "Chunks"), delta("risks", "Risks"),
                                   delta("blockers", "Open blockers"), delta("actions", "Actions"), delta("health", "Health")]))

    stats = kb.stats()
    k = st.columns(4)
    with k[0]:
        kpi("Documents", str(stats["documents"]), "indexed", "📄", "blue")
    with k[1]:
        kpi("Chunks", str(stats["chunks"]), "retrievable passages", "🧩", "violet")
    with k[2]:
        kpi("Tables", str(stats["tables"]), "spreadsheet sheets parsed", "📊", "green")
    with k[3]:
        kpi("Vocabulary", f"{stats['vocabulary']:,}", "indexed terms", "🔤", "slate")
    st.write("")

    with card():
        section("Indexed documents")
        if kb.is_empty:
            empty_state("📚", "The knowledge base is empty", "Drop files into the uploader above — they are indexed automatically.")
        else:
            per_doc = Counter(c.doc_name for c in kb.chunks)
            rows = [{"Document": d.name, "Type": d.kind_label, "Size": human_size(d.size_bytes), "Segments": len(d.segments),
                     "Chunks": per_doc.get(d.name, 0), "Notes": "; ".join(d.warnings) or "OK"} for d in kb.documents.values()]
            show_df(pd.DataFrame(rows))
            r1, r2 = st.columns([3, 1])
            with r1:
                sel = st.multiselect("Remove documents", list(kb.documents), key="remove_sel", label_visibility="collapsed",
                                     placeholder="Select documents to remove…")
            with r2:
                if stretch(st.button, "Remove selected", key="btn_remove", disabled=not sel):
                    request("remove", names=sel)

    c1, c2 = st.columns(2)
    with c1:
        with card():
            section("Ingestion log", "Most recent uploads")
            if ss.upload_log:
                row_items([(badge({"indexed": "Indexed", "failed": "Failed", "skipped": "Skipped"}[e["status"]],
                                  {"indexed": "ok", "failed": "error", "skipped": "neutral"}[e["status"]]), e["file"], e["detail"])
                           for e in reversed(ss.upload_log[-8:])])
            else:
                st.caption("Nothing uploaded yet.")
    with c2:
        with card():
            section("Retrieval explorer", "Test what the assistant would retrieve")
            q = st.text_input("Query", key="explore_q", placeholder="e.g. unresolved defects", label_visibility="collapsed")
            if q:
                hits = kb.search(q)
                if not hits:
                    st.caption("No passage scored above the relevance threshold — the assistant would answer “not found”.")
                for h in hits[:4]:
                    html_block(f'<div class="src"><b>{h.chunk.doc_name}</b> · {h.chunk.locator} · relevance {h.score:.2f}<br>'
                               f'{truncate(h.chunk.text, 260).replace("<", "&lt;")}</div>')
            else:
                st.caption(f"Relevance threshold: {MIN_RELEVANCE}. Type a query to inspect retrieval.")


# =============================================================================
# Agent Center
# =============================================================================
def _run_for(pipe: PipelineResult, name: str):
    return next((r for r in pipe.runs if r.name == name), None)


def render_agent_center() -> None:
    ss = _ss()
    pipe, docs = ss.pipe, ss.docs
    top = st.columns([4, 1.3])
    with top[0]:
        section("Multi-agent intelligence pipeline", "Four specialised agents analyse the knowledge base and cite their evidence")
    with top[1]:
        if stretch(st.button, "▶ Run all agents", type="primary", key="btn_rerun"):
            request("rerun")

    cols = st.columns(2)
    for i, (key, name, desc, icon) in enumerate(AGENT_CATALOG):
        with cols[i % 2]:
            run = _run_for(pipe, name)
            if key == "docs":
                ok = bool(docs.get("risks_df") is not None and ss.kb.documents)
                status, count, secs = ("Success" if ok else "Idle"), (len(docs["stories"]) + len(pipe.risk.risks) + len(pipe.blockers.actions)) if ok else 0, None
            else:
                status = "Idle" if run is None else run.status.capitalize()
                count, secs = (run.count if run else 0), (run.seconds if run else None)
            with card():
                html_block(f'<div style="display:flex;gap:12px;align-items:center"><div class="kpi-icon" style="background:#eef2ff;width:46px;height:46px;'
                           f'border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:21px">{icon}</div>'
                           f'<div style="flex:1"><div class="section-title" style="margin:0">{name}</div>'
                           f'<div class="section-sub" style="margin:0">{desc}</div></div>{badge(status, status.lower())}</div>')
                st.caption(f"{count} finding(s)" + (f" · {secs * 1000:.0f} ms" if secs is not None else "")
                           + (f" · {run.message}" if run and run.message else ""))

    SEC = ["🎯 Scope & Deliverables", "⚠️ Risks & Forecast", "🚧 Blockers & Actions", "📝 Documentation", "✔️ Validation"]
    sec = st.radio("Agent output", SEC, horizontal=True, key="agent_view", label_visibility="collapsed")
    scope = pipe.scope
    if sec == SEC[0]:
        if ss.kb.is_empty:
            empty_state("🎯", "Scope extraction is waiting for documents", "Goals, deliverables, timelines and owners are extracted here.", ONBOARDING)
        else:
            with card():
                section("Project summary")
                st.write(scope.summary or "No narrative summary was found in the documents.")
            a, b = st.columns(2)
            with a:
                with card():
                    section("Goals & objectives")
                    row_items([(badge("Goal", "brand"), truncate(g.text, 220), f"{g.doc} · {g.locator}") for g in scope.goals], "No explicit goals found.")
            with b:
                with card():
                    section("Deliverables")
                    show_df(pd.DataFrame([{"Deliverable": d.text, "Kind": d.meta.get("kind", ""), "Owner": d.meta.get("owner", ""),
                                           "Status": d.meta.get("status", ""), "Source": d.doc} for d in scope.deliverables]), height=300)
            with card():
                section("Timeline", "Phases and sprints with their tasks")
                if not scope.timeline:
                    st.caption("No phases or sprints were found.")
                for p in scope.timeline:
                    with st.expander(f"{truncate(p.label, 80)}" + (f"  ·  {p.window}" if p.window else "") + f"  ·  {len(p.tasks)} task(s)"):
                        for t in p.tasks:
                            st.markdown(f"- {t}")
                        st.caption(f"Source: {p.doc} · {p.locator}")
            with card():
                section("Responsibilities")
                show_df(pd.DataFrame([{"Owner": r.owner, "Items": len(r.work), "Work": "; ".join(r.work[:3]), "Sources": ", ".join(r.sources[:2])}
                                      for r in scope.responsibilities]))
    elif sec == SEC[1]:
        fc = pipe.risk.forecast
        if ss.kb.is_empty:
            empty_state("⚠️", "Risk detection is waiting for documents", "Schedule, dependency, quality and capacity risks plus a delivery forecast appear here.", ONBOARDING)
        else:
            k = st.columns(3)
            with k[0]:
                kpi("On-time likelihood", f"{fc.on_time_pct:.0f}%", fc.level, "🎯", "blue")
            with k[1]:
                kpi("Forecast confidence", fc.confidence, "based on evidence available", "🔎", "violet")
            with k[2]:
                kpi("Dependency gaps", str(len(pipe.risk.dependency_gaps)), "unmet or cascading", "🔗", "amber")
            st.write("")
            with card():
                section("Delivery forecast")
                st.write(fc.narrative)
                if fc.drivers:
                    st.markdown("**Drivers**\n" + "\n".join(f"- {d}" for d in fc.drivers))
                if fc.key_dates:
                    st.markdown("**Key dates**\n" + "\n".join(f"- {d}" for d in fc.key_dates))
            with card():
                section("Detected risks", "“Detected” = stated in a document · “Inferred” = derived from structural gaps")
                show_df(pd.DataFrame([{"ID": r.risk_id, "Severity": r.severity, "Category": r.category, "Risk": r.title, "Basis": r.evidence_type,
                                       "Source": f"{r.doc} · {r.locator}"} for r in pipe.risk.risks]))
            with card():
                section("Dependency gaps")
                row_items([(badge(r.severity), truncate(r.title, 160), f"{r.doc} · {r.locator}") for r in pipe.risk.dependency_gaps], "No dependency gaps detected.")
    elif sec == SEC[2]:
        if ss.kb.is_empty:
            empty_state("🚧", "Blocker identification is waiting for documents", "Blockers, pending decisions and assigned action items appear here.", ONBOARDING)
        else:
            bl = pipe.blockers
            with card():
                section("Blockers & issues")
                show_df(pd.DataFrame([{"Status": b.status, "Severity": b.severity, "Issue": b.text, "Owner": b.owner or "—",
                                       "Resolution": b.resolution, "Source": f"{b.doc} · {b.locator}"} for b in bl.blockers]))
            with card():
                section("Pending decisions")
                row_items([(badge("Decision", "brand"), truncate(d.text, 200), f"{d.doc} · {d.locator}") for d in bl.decisions], "No pending decisions detected.")
            with card():
                section("Action items")
                show_df(docs["actions_df"] if docs.get("actions_df") is not None else pd.DataFrame())
    elif sec == SEC[3]:
        if ss.kb.is_empty:
            empty_state("📝", "Documentation generation is waiting for documents", "User stories, a risk register and action items are authored automatically.", ONBOARDING)
        else:
            k = st.columns(3)
            with k[0]:
                kpi("User stories", str(len(docs["stories"])), "with acceptance criteria", "📖", "blue")
            with k[1]:
                kpi("Register entries", str(len(pipe.risk.risks)), "scored risks with mitigations", "🛡️", "red")
            with k[2]:
                kpi("Action items", str(len(pipe.blockers.actions)), "owner · due · priority", "✅", "green")
            stretch(st.button, "Open Generated Docs →", key="go_docs", on_click=goto, args=(PAGE_DOCS,))
    else:
        _render_agent_validation()


# =============================================================================
# Generated Docs
# =============================================================================
def render_generated_docs() -> None:
    ss = _ss()
    docs, pipe = ss.docs, ss.pipe
    stories: List[docgen.Story] = docs.get("stories") or []
    risks_df = docs.get("risks_df") if docs.get("risks_df") is not None else docgen.risk_register(PipelineResult())
    actions_df = docs.get("actions_df") if docs.get("actions_df") is not None else docgen.action_register(PipelineResult())
    stories_df = docs.get("stories_df") if docs.get("stories_df") is not None else docgen.stories_frame([])

    SEC = ["📖 User Stories", "🛡️ Risk Register", "✅ Action Items", "📄 Project Report", "⬇️ Export"]
    sec = st.radio("Document", SEC, horizontal=True, key="docs_view", label_visibility="collapsed")
    if sec == SEC[0]:
        if not stories:
            empty_state("📖", "No user stories yet", "Stories are written from your deliverables, goals, planned tasks and top risks.", ONBOARDING)
        else:
            q = st.text_input("Filter stories", key="story_filter", placeholder="Search by keyword…", label_visibility="collapsed")
            shown = [s for s in stories if not q or q.lower() in (s.title + s.story).lower()]
            st.caption(f"{len(shown)} of {len(stories)} stories")
            for s in shown:
                with st.expander(f"{s.story_id} · {s.title} — {s.priority} priority · {s.points} pts · {s.basis}"):
                    st.markdown(f"**{s.story}**")
                    st.markdown("**Acceptance criteria**\n" + "\n".join(f"{i}. {c}" for i, c in enumerate(s.acceptance, 1)))
                    st.caption(f"Source: {s.source}")
    elif sec == SEC[1]:
        if risks_df.empty:
            empty_state("🛡️", "The risk register is empty", "Each risk is scored (probability × impact), rated, mitigated and traced to its source.", ONBOARDING)
        else:
            counts = Counter(risks_df["Rating"])
            k = st.columns(4)
            for col, (lab, tone) in zip(k, (("Critical", "red"), ("High", "amber"), ("Medium", "blue"), ("Low", "green"))):
                with col:
                    kpi(lab, str(counts.get(lab, 0)), "risks", "⚠️", tone)
            st.write("")
            pick = st.multiselect("Rating", ["Critical", "High", "Medium", "Low"], default=["Critical", "High", "Medium", "Low"], key="risk_rating")
            show_df(risks_df[risks_df["Rating"].isin(pick)], height=420)
    elif sec == SEC[2]:
        if actions_df.empty:
            empty_state("✅", "No action items yet", "Actions are extracted from notes, trackers and plans, with owner, due date and priority.", ONBOARDING)
        else:
            c1, c2 = st.columns(2)
            with c1:
                owners = st.multiselect("Owner", sorted(actions_df["Owner"].unique()), key="act_owner")
            with c2:
                prios = st.multiselect("Priority", ["High", "Medium", "Low"], key="act_prio")
            view = actions_df
            if owners:
                view = view[view["Owner"].isin(owners)]
            if prios:
                view = view[view["Priority"].isin(prios)]
            show_df(view, height=420)
    elif sec == SEC[3]:
        with card():
            section("Project report", "A shareable executive report: summary, scope, health, risks, blockers, actions, stories")
            if ss.kb.is_empty:
                empty_state("📄", "The report is built from your documents", "Add documents to generate the full report.", ONBOARDING)
            else:
                try:
                    md = report.build_markdown(ss.kb, pipe, ss.health, stories)
                    docx_bytes = report.build_docx(ss.kb, pipe, ss.health, stories)
                except Exception as exc:
                    st.error(f"The report could not be generated: {exc}")
                    md, docx_bytes = "", b""
                c1, c2, _ = st.columns([1, 1, 2])
                with c1:
                    stretch(st.download_button, "⬇ Word report (.docx)", data=docx_bytes, file_name="project_report.docx", key="dl_report_docx",
                            mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", type="primary", disabled=not docx_bytes)
                with c2:
                    stretch(st.download_button, "⬇ Markdown", data=md.encode("utf-8"), file_name="project_report.md", mime="text/markdown",
                            key="dl_report_md", disabled=not md)
                with st.expander("Preview", expanded=True):
                    st.markdown(md)
    else:
        with card():
            section("Download generated documentation", "Excel pack · Markdown · CSV")
            try:
                xlsx = docgen.to_excel_bytes({"User Stories": stories_df, "Risk Register": risks_df, "Action Items": actions_df})
                md = docgen.to_markdown(stories, risks_df, actions_df).encode("utf-8")
            except Exception as exc:
                st.error(f"Export could not be prepared: {exc}")
                xlsx = md = b""
            c = st.columns(5)
            items = [("📦 Excel pack", xlsx, "project_documentation.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"),
                     ("📝 Markdown", md, "project_documentation.md", "text/markdown"),
                     ("📖 Stories CSV", docgen.to_csv_bytes(stories_df), "user_stories.csv", "text/csv"),
                     ("🛡️ Risks CSV", docgen.to_csv_bytes(risks_df), "risk_register.csv", "text/csv"),
                     ("✅ Actions CSV", docgen.to_csv_bytes(actions_df), "action_items.csv", "text/csv")]
            for col, (label, data, name, mime) in zip(c, items):
                with col:
                    stretch(st.download_button, label, data=data, file_name=name, mime=mime, key=f"dl_{name}", disabled=not data)
            if ss.kb.is_empty:
                st.caption("Exports contain column headers only until documents are processed.")


# =============================================================================
# Project Health
# =============================================================================
def render_health() -> None:
    ss = _ss()
    h, pipe = ss.health, ss.pipe
    a, b = st.columns([1, 2])
    with a:
        with card():
            section("Overall project health")
            gauge(h.overall if h.has_data else None, h.band if h.has_data else "No data", h.color if h.has_data else "#94a3b8", 260)
            if not h.has_data:
                st.caption("The score is computed from scope clarity, timeline risk and blocker load once documents are processed.")
    with b:
        with card():
            section("Dimension breakdown", "Weighted composite of three dimensions")
            if h.has_data:
                for d in h.dimensions:
                    dimension_bar(d.label, d.score, d.summary, d.weight)
            else:
                for key, lab, note in (("scope", "Scope clarity", "How clearly goals, deliverables, dates and owners are defined."),
                                       ("timeline", "Timeline risk", "Resilience of the schedule (100 = low risk)."),
                                       ("blockers", "Blocker load", "Open blockers and undecided questions (100 = unobstructed).")):
                    dimension_bar(lab, None, note, HEALTH_WEIGHTS[key])

    if h.has_data:
        cols = st.columns(3)
        for col, d in zip(cols, h.dimensions):
            with col:
                with card():
                    section(d.label)
                    st.markdown("\n".join(f"- {s}" for s in d.signals))
        c1, c2 = st.columns(2)
        with c1:
            with card():
                section("Recommendations", "Highest-leverage next steps")
                row_items([(badge(str(i), "brand"), r, "") for i, r in enumerate(h.recommendations, 1)])
        with c2:
            with card():
                section("Forecast drivers")
                fc = pipe.risk.forecast
                st.write(fc.narrative)
                st.markdown("\n".join(f"- {d}" for d in fc.drivers) or "_No drivers identified._")
    with st.expander("How the score is calculated"):
        st.markdown(
            f"- **Scope clarity ({int(HEALTH_WEIGHTS['scope'] * 100)}%)** — goals, deliverables, dated timeline, named owners and evidence breadth; reduced by scope/governance risks.\n"
            f"- **Timeline risk ({int(HEALTH_WEIGHTS['timeline'] * 100)}%)** — penalties for schedule and dependency risks blended with the on-time delivery forecast.\n"
            f"- **Blocker load ({int(HEALTH_WEIGHTS['blockers'] * 100)}%)** — severity-weighted open blockers and pending decisions.\n\n"
            "Bands: **Healthy** ≥ 80 · **Watch** 60–79 · **At Risk** 40–59 · **Critical** < 40.")




# =============================================================================
# Agent validation (across formats)
# =============================================================================
def _render_agent_validation() -> None:
    ss = _ss()
    if ss.kb.is_empty:
        empty_state("✔️", "Validation needs documents", "See which uploaded formats contributed findings to each agent, and how well retrieval performs.", ONBOARDING)
        return
    df = validation.agent_validation(ss.kb, ss.pipe)
    with card():
        section("Agent coverage by document", "Confirms every file format is analysed and shows who contributed what")
        show_df(df)
        contributing = int((df["Verdict"] == "Contributing").sum())
        st.caption(f"{contributing} of {len(df)} documents contributed findings across {df['Format'].str.split(' · ').str[0].nunique()} file format(s).")
    with card():
        section("Retrieval self-test", "Distinctive-term probes built from indexed chunks; each should retrieve its own chunk in the top results")
        res = validation.retrieval_validation(ss.kb)
        if res["n"]:
            k = st.columns(3)
            with k[0]:
                kpi("Recall@" + str(TOP_K), f"{res['recall']:.0%}", f"{res['n']} probes", "🎯", "green" if res["recall"] >= .8 else "amber")
            with k[1]:
                kpi("Mean reciprocal rank", f"{res['mrr']:.2f}", "1.00 = always ranked first", "📈", "blue")
            with k[2]:
                kpi("Index size", str(len(ss.kb.chunks)), "chunks", "🧩", "slate")
            with st.expander("Probe details"):
                show_df(res["rows"])
        else:
            st.caption("Not enough indexed text to run the self-test.")


# =============================================================================
# AI Assistant  (featured experience)
# =============================================================================
SUGGESTION_GROUPS = {
    "Status": ["Are we on track?", "Summarise the project scope"],
    "Risks & issues": ["What are our biggest risks?", "What are the open blockers?", "Which defects or test findings need attention?"],
    "Delivery": ["What does the timeline look like?", "What are the action items?"],
    "People & decisions": ["Who owns which work?", "What decisions are pending?"],
}


def _transcript(chat: list) -> str:
    out = ["# Project assistant transcript", ""]
    for m in chat:
        out.append(f"**{'You' if m['role'] == 'user' else 'Assistant'}:** {m['content']}\n")
        for n, doc, loc, snip, _ in m.get("sources", []):
            out.append(f"> [{n}] {doc} · {loc}")
        out.append("")
    return "\n".join(out)


def _render_msg(i: int, m: dict, last: bool) -> None:
    ss = _ss()
    with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🧭"):
        st.markdown(m["content"])
        if m["role"] != "assistant":
            return
        if m.get("meta"):
            st.caption(m["meta"])
        if m.get("sources"):
            with st.expander(f"📎 Sources ({len(m['sources'])})"):
                for n, doc, loc, snip, score in m["sources"]:
                    html_block(f'<div class="src"><span class="cite-chip">[{n}]</span><b>{doc}</b> · {loc}'
                               + (f' · relevance {score:.2f}' if score else "") + f'<br>{snip.replace("<", "&lt;")}</div>')
        fb = ss.feedback.get(i)
        c = st.columns([1, 1, 8])
        with c[0]:
            if stretch(st.button, "👍" if fb != 1 else "👍 ✓", key=f"up_{i}"):
                ss.feedback[i] = 1
                st.rerun()
        with c[1]:
            if stretch(st.button, "👎" if fb != -1 else "👎 ✓", key=f"down_{i}"):
                ss.feedback[i] = -1
                st.rerun()
        if last and m.get("followups"):
            st.caption("Suggested follow-ups")
            fc = st.columns(len(m["followups"]))
            for j, (col, q) in enumerate(zip(fc, m["followups"])):
                with col:
                    stretch(st.button, q, key=f"fu_{i}_{j}", on_click=ask, args=(q,))


def render_assistant() -> None:
    ss = _ss()
    prompt = st.chat_input("Ask about status, risks, blockers, owners, scope, timeline…", key="chat_box")   # docks to the bottom
    prompt = prompt or ss.queued_question
    ss.queued_question = None
    if prompt:
        history = [{"role": m["role"], "content": m["content"]} for m in ss.chat]
        with st.spinner("Searching the knowledge base…"):
            ans = answer_question(ss.kb, ss.pipe, ss.health, prompt, history=history)
        ss.chat.append({"role": "user", "content": prompt})
        ss.chat.append({"role": "assistant", "content": ans.text,
                        "sources": [(s.n, s.doc, s.locator, s.snippet, s.score) for s in ans.sources],
                        "followups": ans.followups, "relevance": ans.relevance, "grounded": ans.grounded,
                        "meta": (f"{ans.confidence} confidence · relevance {ans.relevance:.0%} · "
                                 f"{'LLM-synthesised' if ans.mode == 'llm' else 'extractive'} · intent: {ans.intent}") if ans.grounded
                        else "Not found in the uploaded documents"})

    stats = ss.kb.stats()
    head = st.columns([4, 1.1, 1.3])
    with head[0]:
        section("Conversational Project Intelligence Assistant",
                f"RAG over {stats['documents']} document(s) · {stats['chunks']} chunks · every answer is cited and grounded")
    with head[1]:
        stretch(st.button, "🧹 New chat", key="clear_chat", disabled=not ss.chat, on_click=lambda: ss.update(chat=[], feedback={}))
    with head[2]:
        stretch(st.download_button, "⬇ Transcript", data=_transcript(ss.chat).encode("utf-8"), file_name="assistant_transcript.md",
                mime="text/markdown", key="dl_chat", disabled=not ss.chat)

    main, side = st.columns([2.3, 1])
    with main:
        if not ss.chat:
            with card():
                html_block('<div style="text-align:center;padding:14px 6px"><div style="font-size:42px">🧭</div>'
                           '<div class="section-title" style="font-size:1.25rem">Ask anything about your project</div>'
                           '<div class="section-sub">I answer only from your uploaded documents, cite every source, and say so when I can’t find something.</div></div>')
                if ss.kb.is_empty:
                    st.info("The knowledge base is empty — upload documents in **Knowledge Base** to start. You can still try the interface.")
                st.markdown("**Good questions to start with**")
                cols = st.columns(2)
                for i, q in enumerate(["Are we on track?", "What are our biggest risks?", "What are the open blockers?", "Who owns which work?"]):
                    with cols[i % 2]:
                        stretch(st.button, q, key=f"start_{i}", on_click=ask, args=(q,))
        for i, m in enumerate(ss.chat):
            _render_msg(i, m, last=(i == len(ss.chat) - 1))
    with side:
        with card():
            section("Evidence for the latest answer")
            last = next((m for m in reversed(ss.chat) if m["role"] == "assistant"), None)
            if last and last.get("sources"):
                for n, doc, loc, snip, score in last["sources"]:
                    pct = int(min(1.0, score / 0.3) * 100) if score else 0
                    html_block(f'<div class="src"><span class="cite-chip">[{n}]</span><b>{doc}</b><br><span class="muted">{loc}</span>'
                               + (f'<div class="bar" style="margin-top:6px;height:6px"><span style="width:{pct}%;background:#4f46e5"></span></div>' if score else "")
                               + "</div>")
            else:
                st.caption("Sources appear here after you ask a question." if not last else "No supporting passages were found for the last question.")
        with card():
            section("Try asking")
            for g, qs in SUGGESTION_GROUPS.items():
                st.caption(g)
                for j, q in enumerate(qs):
                    stretch(st.button, q, key=f"sg_{g}_{j}", on_click=ask, args=(q,))
        with card():
            section("Answer quality")
            up = sum(1 for v in ss.feedback.values() if v == 1)
            down = sum(1 for v in ss.feedback.values() if v == -1)
            st.caption(f"Feedback this session: 👍 {up} · 👎 {down}")
            if stretch(st.button, "Validate assistant accuracy", key="run_validation", disabled=ss.kb.is_empty):
                with st.spinner("Running grounding and relevance checks…"):
                    ss.validation = validation.assistant_validation(ss.kb, ss.pipe, ss.health)
            v = ss.validation
            if v:
                st.markdown(f"- Answer rate: **{v['answer_rate']:.0%}**\n- Citation rate: **{v['citation_rate']:.0%}**\n"
                            f"- Citation coverage: **{v['avg_coverage']:.0%}**\n- Avg relevance: **{v['avg_relevance']:.0%}**\n"
                            f"- Off-topic refusal: **{v['refusal_accuracy']:.0%}**")
                with st.expander("Per-question results"):
                    show_df(v["rows"])
            elif ss.kb.is_empty:
                st.caption("Upload documents to run validation.")
