"""Project report: a shareable executive document assembled from the analysis (DOCX and Markdown)."""
from __future__ import annotations

import io
from datetime import datetime
from typing import List

from .docgen import MITIGATIONS, Story, rating
from .health import HealthReport
from .models import PipelineResult
from .rag import KnowledgeBase
from .utils import fmt_date, truncate


def build_markdown(kb: KnowledgeBase, pipe: PipelineResult, health: HealthReport, stories: List[Story]) -> str:
    fc, sc = pipe.risk.forecast, pipe.scope
    out = [f"# Project Intelligence Report — {sc.project_name or 'Project'}", "", f"_Generated {datetime.now():%d %b %Y %H:%M}_", "",
           "## 1. Executive summary", "", sc.summary or "No narrative summary was found in the uploaded documents.", ""]
    if health.has_data:
        out += [f"**Overall health:** {health.overall:.0f}/100 ({health.band})  ", f"**Delivery outlook:** {fc.narrative}", ""]
    out += ["## 2. Scope and deliverables", "", "**Goals**", ""] + [f"- {g.text} _({g.doc})_" for g in sc.goals] + ["", "**Deliverables**", ""]
    out += [f"- {d.text}" for d in sc.deliverables] + ["", "**Timeline**", ""]
    out += [f"- {p.label}{' — ' + p.window if p.window else ''} ({len(p.tasks)} tasks)" for p in sc.timeline] + [""]
    out += ["## 3. Project health", ""]
    out += [f"- **{d.label}:** {d.score:.0f}/100 (weight {d.weight:.0%}) — {'; '.join(d.signals)}" for d in health.dimensions]
    out += ["", "**Recommendations**", ""] + [f"{i}. {r}" for i, r in enumerate(health.recommendations, 1)] + [""]
    out += ["## 4. Risk register (top 12)", "", "| ID | Risk | Category | Rating | Mitigation |", "|---|---|---|---|---|"]
    for r in pipe.risk.risks[:12]:
        out.append(f"| {r.risk_id} | {r.title.replace('|', '/')} | {r.category} | {rating(r.score)} | {MITIGATIONS.get(r.category, '')} |")
    out += ["", "## 5. Blockers and pending decisions", ""]
    out += [f"- [{b.status}] {b.text}" for b in pipe.blockers.blockers] + [f"- [Decision] {d.text}" for d in pipe.blockers.decisions] + [""]
    out += ["## 6. Action items", "", "| ID | Action | Owner | Due | Priority |", "|---|---|---|---|---|"]
    out += [f"| {a.action_id} | {a.text.replace('|', '/')} | {a.owner} | {a.due} | {a.priority} |" for a in pipe.blockers.actions[:25]]
    out += ["", "## 7. User stories", ""] + [f"- **{s.story_id}** {s.story}" for s in stories[:15]]
    out += ["", "## 8. Evidence base", ""] + [f"- {d.name} — {d.kind_label}" for d in kb.documents.values()]
    return "\n".join(out) + "\n"


def build_docx(kb: KnowledgeBase, pipe: PipelineResult, health: HealthReport, stories: List[Story]) -> bytes:
    from docx import Document
    from docx.shared import Pt, RGBColor

    sc, fc = pipe.scope, pipe.risk.forecast
    doc = Document()
    doc.styles["Normal"].font.name = "Calibri"
    doc.styles["Normal"].font.size = Pt(10.5)
    title = doc.add_heading(f"Project Intelligence Report", 0)
    title.runs[0].font.color.rgb = RGBColor(0x1E, 0x1B, 0x4B)
    doc.add_paragraph(f"{sc.project_name or 'Project'} · generated {datetime.now():%d %b %Y %H:%M}")

    def table(headers: List[str], rows: List[List[str]]) -> None:
        t = doc.add_table(rows=1, cols=len(headers))
        try:
            t.style = "Light Grid Accent 1"
        except Exception:
            t.style = "Table Grid"
        for i, h in enumerate(headers):
            t.rows[0].cells[i].text = h
        for r in rows:
            cells = t.add_row().cells
            for i, v in enumerate(r):
                cells[i].text = str(v)

    doc.add_heading("1. Executive summary", 1)
    doc.add_paragraph(sc.summary or "No narrative summary was found in the uploaded documents.")
    if health.has_data:
        p = doc.add_paragraph()
        p.add_run(f"Overall health: {health.overall:.0f}/100 ({health.band}). ").bold = True
        p.add_run(fc.narrative)
    doc.add_heading("2. Scope and deliverables", 1)
    doc.add_heading("Goals", 2)
    for g in sc.goals:
        doc.add_paragraph(g.text, style="List Bullet")
    doc.add_heading("Deliverables", 2)
    for d in sc.deliverables:
        doc.add_paragraph(d.text, style="List Bullet")
    doc.add_heading("Timeline", 2)
    table(["Phase / sprint", "Window", "Tasks"], [[p.label, p.window or "—", len(p.tasks)] for p in sc.timeline] or [["—", "—", "—"]])
    doc.add_heading("3. Project health", 1)
    table(["Dimension", "Score", "Weight", "Signals"], [[d.label, f"{d.score:.0f}", f"{d.weight:.0%}", "; ".join(d.signals)] for d in health.dimensions]
          or [["—", "—", "—", "—"]])
    doc.add_heading("Recommendations", 2)
    for r in health.recommendations:
        doc.add_paragraph(r, style="List Number")
    doc.add_heading("4. Risk register", 1)
    table(["ID", "Risk", "Category", "Rating", "Mitigation"],
          [[r.risk_id, truncate(r.title, 140), r.category, rating(r.score), MITIGATIONS.get(r.category, "")] for r in pipe.risk.risks[:15]]
          or [["—", "No risks detected", "—", "—", "—"]])
    doc.add_heading("5. Blockers and pending decisions", 1)
    for b in pipe.blockers.blockers:
        doc.add_paragraph(f"[{b.status}] {b.text}", style="List Bullet")
    for d in pipe.blockers.decisions:
        doc.add_paragraph(f"[Decision] {d.text}", style="List Bullet")
    doc.add_heading("6. Action items", 1)
    table(["ID", "Action", "Owner", "Due", "Priority"],
          [[a.action_id, truncate(a.text, 130), a.owner, a.due, a.priority] for a in pipe.blockers.actions[:25]] or [["—", "None", "—", "—", "—"]])
    doc.add_heading("7. User stories", 1)
    for s in stories[:15]:
        doc.add_paragraph(f"{s.story_id}: {s.story}", style="List Bullet")
    doc.add_heading("8. Evidence base", 1)
    table(["Document", "Type", "Chunks"], [[d.name, d.kind_label, sum(1 for c in kb.chunks if c.doc_name == d.name)] for d in kb.documents.values()]
          or [["—", "—", "—"]])
    doc.add_paragraph("Findings marked Inferred are derived from structural gaps; all others are stated in the cited documents.").italic = True
    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
