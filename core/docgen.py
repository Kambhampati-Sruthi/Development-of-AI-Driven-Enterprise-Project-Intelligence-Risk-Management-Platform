"""Documentation Generation Agent: user stories, risk register, action items, exports."""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .models import Item, PipelineResult
from .utils import clean, truncate

MITIGATIONS: Dict[str, str] = {
    "Schedule": "Re-baseline the plan, add contingency buffer, and review progress against dates in a weekly checkpoint.",
    "Dependency": "Make the dependency explicit with an owner and due date; agree a fallback or stub so work can proceed.",
    "Resource": "Confirm capacity with the team, rebalance assignments, and name a backup for critical work.",
    "Technical": "Reproduce and isolate the issue, assign a developer, add a regression test, and timebox the fix.",
    "Quality": "Fix, retest and formally close; add the scenario to the regression suite and report coverage.",
    "Scope": "Clarify and baseline requirements with stakeholders; record decisions and control further changes.",
    "Governance": "Assign a single accountable owner and agree a communication and escalation routine.",
    "General": "Assess likelihood and impact with the team, assign an owner and define a mitigation date.",
}
_PP = {"develop": "developed", "build": "built", "implement": "implemented", "validate": "validated",
       "design": "designed", "study": "studied", "generate": "generated", "create": "created", "provide": "provided",
       "produce": "produced", "test": "tested", "deploy": "deployed", "integrate": "integrated",
       "document": "documented", "define": "defined", "complete": "completed", "fix": "fixed", "write": "written",
       "prepare": "prepared", "conduct": "conducted", "review": "reviewed", "configure": "configured",
       "verify": "verified", "evaluate": "evaluated", "ensure": "ensured", "improve": "improved"}


def rating(score: int) -> str:
    return "Critical" if score >= 20 else "High" if score >= 12 else "Medium" if score >= 6 else "Low"


# --------------------------------------------------------------------- risks ----
def risk_register(res: PipelineResult) -> pd.DataFrame:
    rows = [{
        "ID": r.risk_id, "Risk": r.title, "Category": r.category, "Probability (1-5)": r.probability,
        "Impact (1-5)": r.impact, "Score": r.score, "Rating": rating(r.score), "Basis": r.evidence_type,
        "Mitigation": MITIGATIONS.get(r.category, MITIGATIONS["General"]),
        "Owner": r.owner or "Unassigned", "Status": "Open", "Source": f"{r.doc} · {r.locator}",
    } for r in res.risk.risks]
    cols = ["ID", "Risk", "Category", "Probability (1-5)", "Impact (1-5)", "Score", "Rating", "Basis", "Mitigation",
            "Owner", "Status", "Source"]
    return pd.DataFrame(rows, columns=cols)


def action_register(res: PipelineResult) -> pd.DataFrame:
    rows = [{"ID": a.action_id, "Action": a.text, "Owner": a.owner, "Due": a.due, "Priority": a.priority,
             "Type": a.kind, "Status": a.status, "Source": f"{a.doc} · {a.locator}"} for a in res.blockers.actions]
    return pd.DataFrame(rows, columns=["ID", "Action", "Owner", "Due", "Priority", "Type", "Status", "Source"])


# -------------------------------------------------------------- user stories ----
@dataclass
class Story:
    story_id: str
    title: str
    story: str
    acceptance: List[str]
    priority: str
    points: int
    source: str
    basis: str


def _role(text: str) -> str:
    t = text.lower()
    if re.search(r"test|defect|validat|quality|bug", t):
        return "QA engineer"
    if re.search(r"risk|schedule|timeline|forecast|health|dashboard|report|insight|score", t):
        return "project manager"
    if re.search(r"assistant|question|query|chat|answer", t):
        return "team member"
    return "project stakeholder"


def _benefit(text: str) -> str:
    t = text.lower()
    rules = [(r"ingest|upload|knowledge base|parse|index|rag|retriev", "all project artifacts are available in one searchable knowledge base"),
             (r"risk|forecast|dependency", "threats to delivery are visible early enough to act on"),
             (r"dashboard|health|insight|score|summary", "I can judge delivery status at a glance"),
             (r"assistant|question|chat|answer", "I get trustworthy, cited answers without reading every document"),
             (r"test|validat|defect|bug|quality", "quality issues are caught before delivery"),
             (r"user stor|register|documentation|action item", "project documentation stays complete and consistent"),
             (r"scope|deliverable|extract|goal", "everyone shares the same understanding of what is being delivered")]
    for rx, b in rules:
        if re.search(rx, t):
            return b
    return "the project outcome is delivered on time and as agreed"


def _capability(text: str) -> str:
    text = clean(text)
    head = re.split(r"\s[—–-]\s|\.\s|;", text)[0]
    head = re.sub(r"^(?:\w+-?\d+\s*:\s*)", "", head)
    m = re.match(r"^(\w+)\s+(.*)$", head)
    if m and m.group(1).lower() in _PP and not (m.group(1).lower() == "document" and m.group(2)[:1].isupper()):
        return f"{truncate(m.group(2).strip(), 120)} to be {_PP[m.group(1).lower()]}"
    return f"the {truncate(head, 120)} capability in place"


def _criteria(text: str, cap: str) -> List[str]:
    t = text.lower()
    if re.search(r"ingest|upload|parse", t):
        return ["Given supported files (PDF, DOCX, CSV, Excel, TXT), when they are uploaded, then each is parsed and indexed without error.",
                "Given a corrupted or empty file, when it is uploaded, then a clear error is shown and other files are unaffected.",
                "Given indexed content, when searched, then results cite the originating document and location."]
    if re.search(r"risk|forecast", t):
        return ["Given project documents, when analysis runs, then risks are listed with category, severity and source.",
                "Given a risk, when reviewed, then probability, impact and a mitigation are recorded.",
                "Given the forecast, when sprint data changes, then the on-time likelihood updates accordingly."]
    if re.search(r"assistant|question|chat", t):
        return ["Given a question, when answered, then every statement is grounded in cited uploaded documents.",
                "Given a question with no support in the documents, when asked, then the assistant says so instead of guessing.",
                "Given an answer, when reviewed, then the source excerpts can be inspected."]
    if re.search(r"test|validat|defect", t):
        return ["Given the documented test cases, when executed, then actual results are recorded against expected results.",
                "Given a failed case, when logged, then a defect with owner and status is created.",
                "Given a fix, when retested, then the defect is closed with evidence."]
    return [f"Given the agreed scope, when work on {truncate(cap, 70)} is complete, then it is demonstrated to the owner.",
            "Given the deliverable, when reviewed, then it matches the documented requirement.",
            "Given completion, when recorded, then supporting evidence is traceable to the source document."]


def _points(text: str) -> int:
    n = len(text.split())
    steps = [1, 2, 3, 5, 8, 13]
    idx = 1 if n <= 12 else 2 if n <= 22 else 3 if n <= 34 else 4
    if re.search(r"pipeline|integration|agent|rag|dashboard|assistant|multi", text, re.I):
        idx = min(idx + 1, 5)
    return steps[idx]


def user_stories(res: PipelineResult) -> List[Story]:
    stories: List[Story] = []

    def add(title_src: str, src: str, basis: str, priority: str = "Medium", existing: str = "") -> None:
        title_src = clean(title_src)
        if existing:
            stories.append(Story("", truncate(re.sub(r"^As an? .*?, I want ", "", existing), 80) or title_src,
                                 existing, _criteria(existing, title_src), priority, _points(existing), src, "Existing"))
            return
        cap = _capability(title_src)
        story = f"As a {_role(title_src)}, I want {cap}, so that {_benefit(title_src)}."
        stories.append(Story("", truncate(title_src.split(" — ")[0], 80), story, _criteria(title_src, cap), priority,
                             _points(title_src), src, basis))

    for d in res.scope.deliverables:
        src = f"{d.doc} · {d.locator}"
        if d.meta.get("kind") == "User story":
            text = re.sub(r"^[\w-]+:\s*", "", d.text)
            if re.match(r"^as an? ", text, re.I):
                add(text, src, "Existing", "High", existing=text)
            else:
                add(text, src, "Derived")
        else:
            add(d.text, src, "Derived")
    for ph in res.scope.timeline:
        if ph.weeks:
            for task in ph.tasks:
                add(task, f"{ph.doc} · {ph.locator}", "Derived", "Medium")
    for g in res.scope.goals[:4]:
        add(g.text, f"{g.doc} · {g.locator}", "Derived", "High")
    for r in res.risk.risks[:3]:
        if r.severity == "High" or r.evidence_type == "Detected":
            add(f"Mitigate: {r.title}", f"{r.doc} · {r.locator}", "Risk-driven", "High")

    seen, out = set(), []
    for s in stories:
        key = re.sub(r"\W+", " ", s.story.lower())[:90]
        if key in seen:
            continue
        seen.add(key)
        out.append(s)
    for n, s in enumerate(out[:30], start=1):
        s.story_id = f"US-{n:02d}"
    return out[:30]


def stories_frame(stories: List[Story]) -> pd.DataFrame:
    return pd.DataFrame([{"ID": s.story_id, "Title": s.title, "User Story": s.story,
                          "Acceptance Criteria": "\n".join(f"{i}. {c}" for i, c in enumerate(s.acceptance, 1)),
                          "Priority": s.priority, "Points": s.points, "Basis": s.basis, "Source": s.source}
                         for s in stories],
                        columns=["ID", "Title", "User Story", "Acceptance Criteria", "Priority", "Points", "Basis", "Source"])


# ------------------------------------------------------------------- exports ----
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")


def to_excel_bytes(sheets: Dict[str, pd.DataFrame]) -> bytes:
    from openpyxl.styles import Alignment, Font, PatternFill
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        for name, df in sheets.items():
            df.to_excel(xl, sheet_name=name[:31], index=False)
            ws = xl.sheets[name[:31]]
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1E3A8A")
            for col in ws.columns:
                width = min(60, max(len(str(c.value or "")) for c in col) + 2)
                ws.column_dimensions[col[0].column_letter].width = max(10, width)
                for c in col[1:]:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
            ws.freeze_panes = "A2"
    return buf.getvalue()


def to_markdown(stories: List[Story], risks: pd.DataFrame, actions: pd.DataFrame) -> str:
    out = ["# Project Documentation Pack", "", "## User Stories", ""]
    for s in stories:
        out += [f"### {s.story_id} — {s.title}", "", s.story, "", f"*Priority: {s.priority} · Points: {s.points} · {s.basis} · Source: {s.source}*", "",
                "**Acceptance criteria**", ""] + [f"- {c}" for c in s.acceptance] + [""]
    out += ["## Risk Register", "", "| ID | Risk | Category | Score | Rating | Mitigation | Owner |", "|---|---|---|---|---|---|---|"]
    for _, r in risks.iterrows():
        out.append(f"| {r['ID']} | {str(r['Risk']).replace('|', '/')} | {r['Category']} | {r['Score']} | {r['Rating']} | {r['Mitigation']} | {r['Owner']} |")
    out += ["", "## Action Items", "", "| ID | Action | Owner | Due | Priority | Status |", "|---|---|---|---|---|---|"]
    for _, a in actions.iterrows():
        out.append(f"| {a['ID']} | {str(a['Action']).replace('|', '/')} | {a['Owner']} | {a['Due']} | {a['Priority']} | {a['Status']} |")
    return "\n".join(out) + "\n"
