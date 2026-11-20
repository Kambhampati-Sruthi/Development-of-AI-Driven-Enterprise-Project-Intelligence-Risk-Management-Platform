"""Multi-agent intelligence pipeline.

Agents are deterministic, evidence-based analysers: every finding carries the
document and location it was derived from, and findings inferred from structural
gaps (rather than stated in a document) are explicitly labelled ``Inferred``.
"""
from __future__ import annotations

import logging
import re
import time
from collections import OrderedDict
from datetime import datetime
from typing import Callable, Dict, List, Optional

from .models import (Action, AgentRun, Blocker, BlockerResult, Forecast, Item, Phase, PipelineResult,
                     Responsibility, Risk, RiskResult, ScopeResult)
from .nlp import (GENERIC_STATEMENT_RE, RESOLVED_RE, STRONG_ISSUE_RE, TASK_VERB_RE, UNRESOLVED_RE, Row, Unit,
                  build_units, iter_rows, iter_tables, norm_status, row_date)
from .rag import KnowledgeBase
from .utils import clamp, clean, dedupe, find_owner, fmt_date, parse_dates, split_sentences, truncate

log = logging.getLogger(__name__)

AGENT_CATALOG = [
    ("scope", "Scope & Deliverable Extraction Agent",
     "Extracts goals, deliverables, timelines and responsibilities from every uploaded artifact.", "🎯"),
    ("risk", "Risk Detection & Delivery Forecasting Agent",
     "Detects schedule, dependency, quality and capacity risks and forecasts on-time delivery.", "⚠️"),
    ("blockers", "Blocker & Action Item Identification Agent",
     "Surfaces blockers, pending decisions and assigned action items from notes, sprints and trackers.", "🚧"),
    ("docs", "Documentation Generation Agent",
     "Writes user stories, a scored risk register and structured action items.", "📝"),
]

GOAL_SECTION = re.compile(r"outcome|objective|goal|aim|purpose|vision|success", re.I)
DELIV_SECTION = re.compile(r"module|deliverable|feature|in scope|requirement|functional|output|scope", re.I)
GOAL_CUE = re.compile(r"\b(objective|goal|aims? to|purpose|in order to|intends? to|mission)\b", re.I)
DELIV_CUE = re.compile(r"\bdeliverables?\s*[:\-]|\bwill (?:produce|deliver)\b|\bshall (?:provide|deliver)\b", re.I)
SUMMARY_SECTION = re.compile(r"statement|overview|summary|about|introduction|background|abstract", re.I)
TIME_HEADING = re.compile(r"\b(?:week|sprint|phase|iteration|stage|release|quarter|q[1-4])s?\s*#?\s*\d", re.I)
WEEKS_RE = re.compile(r"weeks?\s*(\d+)\s*(?:[-–]|to)\s*(\d+)", re.I)
HOURS_RE = re.compile(r"~?\s*(\d+)\s*(?:hours?|hrs?)\b", re.I)

RISK_CATEGORIES = OrderedDict([
    ("Schedule", r"delay\w*|behind schedule|slip\w*|overrun\w*|missed deadline|tight (?:timeline|schedule)|overdue|"
                 r"postpone\w*|reschedul\w*|deadline|late\b"),
    ("Dependency", r"depends? on|dependenc\w+|dependent|waiting (?:for|on)|prerequisite|third[- ]party|external api|"
                   r"approval from|hand-?off|upstream|integration risk"),
    ("Resource", r"shortage|unavailab\w+|attrition|on leave|capacity|overallocat\w+|understaff\w*|skill gap|"
                 r"bandwidth|single point of failure"),
    ("Technical", r"bug|error|exception|crash\w*|performance|latency|scalab\w+|compatib\w+|vulnerab\w+|security|"
                  r"technical debt|not working|failing|mismatch|incompatib\w+"),
    ("Quality", r"defects?|test fail\w*|failed|regression|untested|coverage|rework|flaky"),
    ("Scope", r"scope creep|unclear requirement\w*|ambiguous|undefined|changing requirement\w*|\btbd\b|\btbc\b|"
              r"not finali[sz]ed|missing requirement\w*|incomplete"),
    ("Governance", r"miscommunication|no owner|unassigned|approval pending|unresponsive|conflict|"
                   r"lack of documentation|poor documentation"),
])
RISK_RES = {k: re.compile(r"\b(?:" + v + r")", re.I) for k, v in RISK_CATEGORIES.items()}
GENERAL_RISK_RE = re.compile(r"\b(?:risk of|at risk|high[- ]risk|risky|key risks?|major risks?|biggest risk|"
                             r"risks? (?:is|are|includes?|that|with))\b", re.I)
MEDIUM_RE = re.compile(r"\b(delay\w*|risk|issue|problem|pending|late|concern|tight|mismatch|error|bug|waiting)\b", re.I)


def _owner_name(o: str) -> str:
    o = clean(o)
    return o.title() if o and o.islower() else o


def _sev_pm(sev: str, category: str) -> tuple:
    p, i = {"High": (4, 4), "Medium": (3, 3), "Low": (2, 2)}[sev]
    if sev == "High" and category in ("Schedule", "Dependency"):
        i = 5                                              # delivery-critical: impact is maximal
    return p, i


def _severity(text: str) -> str:
    if STRONG_ISSUE_RE.search(text):
        return "High"
    return "Medium" if MEDIUM_RE.search(text) else "Low"


# =============================================================================
# 1. Scope & Deliverable Extraction Agent
# =============================================================================
def run_scope_agent(kb: KnowledgeBase, units: List[Unit]) -> ScopeResult:
    res = ScopeResult()
    body = [u for u in units if not u.heading]
    heads = [u for u in units if u.heading]

    if heads:
        res.project_name = heads[0].text
    elif kb.documents:
        res.project_name = re.sub(r"[_\-]+", " ", next(iter(kb.documents)).rsplit(".", 1)[0]).title()

    # --- summary
    summ = next((u for u in body if SUMMARY_SECTION.search(u.section) and len(u.text) > 100), None) \
        or next((u for u in body if len(u.text) > 140 and not u.bullet), None)
    if summ:
        res.summary = " ".join(split_sentences(summ.text)[:2])

    # --- goals
    goals: List[Item] = []
    for u in body:
        if len(u.text) < 30:
            continue
        if (GOAL_SECTION.search(u.section) and u.bullet) or (GOAL_CUE.search(u.text) and not GENERIC_STATEMENT_RE.search(u.text)):
            goals.append(Item(truncate(u.text, 300), u.doc, u.locator))
    res.goals = dedupe(goals, key=lambda i: i.text)[:8]

    # --- deliverables
    delivs: List[Item] = []
    for u in body:
        if u.bullet and DELIV_SECTION.search(u.section) and not GOAL_SECTION.search(u.section):
            delivs.append(Item(truncate(u.text, 220), u.doc, u.locator, {"kind": "Module / feature"}))
        elif DELIV_CUE.search(u.text) and len(u.text) > 30:
            delivs.append(Item(truncate(u.text, 220), u.doc, u.locator, {"kind": "Deliverable"}))
    for doc, t in iter_tables(kb, ("backlog",)):
        if "sprint" not in t.roles and "actual" not in t.roles and "dependency" not in t.roles:
            continue                                       # task grids are not deliverables
        for r in iter_rows(doc, t):
            if r.title:
                uid = r.find("id")
                label = f"{uid}: {r.title}" if uid and uid not in r.title else r.title
                delivs.append(Item(truncate(label, 220), doc.name, r.locator,
                                   {"kind": "User story", "status": r.get("status"), "owner": _owner_name(r.get("owner")),
                                    "sprint": r.get("sprint")}))
    res.deliverables = dedupe(delivs, key=lambda i: i.text)[:20]

    # --- timeline phases (narrative headings)
    phases: "OrderedDict[str, Phase]" = OrderedDict()
    cur: Optional[Phase] = None
    for u in units:
        if u.heading:
            cur = None
            if TIME_HEADING.search(u.text):
                wk, hr = WEEKS_RE.search(u.text), HOURS_RE.search(u.text)
                window = f"Weeks {wk.group(1)}–{wk.group(2)}" if wk else ""
                cur = Phase(u.text, window, [], u.doc, u.locator,
                            hours=int(hr.group(1)) if hr else None,
                            weeks=(int(wk.group(2)) - int(wk.group(1)) + 1) if wk else None)
                phases.setdefault(cur.label.lower(), cur)
                cur = phases[cur.label.lower()]
        elif cur is not None and u.bullet:
            if u.text not in cur.tasks:
                cur.tasks.append(truncate(u.text, 220))

    # --- timeline phases (tracker tables)
    for doc, t in iter_tables(kb, ("backlog", "generic")):
        groups: Dict[str, List[Row]] = {}
        for r in iter_rows(doc, t):
            key = r.get("sprint") or ("" if "sprint" in t.roles else t.sheet)
            if re.fullmatch(r"\d+", key or ""):
                key = f"Sprint {key}"
            if key and (r.title or r.get("start") or r.get("due")):
                groups.setdefault(key, []).append(r)
        if t.kind == "generic" and not ("start" in t.roles or "due" in t.roles):
            continue
        for key, rows in groups.items():
            dates = [d for r in rows for d in filter(None, [row_date(r, "start"), row_date(r, "due"), row_date(r, "date")])]
            window = f"{fmt_date(min(dates))} → {fmt_date(max(dates))}" if dates else ""
            tasks = [truncate(r.title, 160) for r in rows if r.title and t.kind == "backlog"]
            ph = phases.get(key.lower())
            if ph is None:
                phases[key.lower()] = Phase(key, window, tasks, doc.name, rows[0].locator)
            else:
                ph.window = ph.window or window
                ph.tasks.extend(x for x in tasks if x not in ph.tasks)
    res.timeline = list(phases.values())

    # --- dated events
    events: List[Item] = []
    for u in body:
        ds = parse_dates(u.text)
        if ds and not u.is_plan_content:
            events.append(Item(truncate(u.text, 200), u.doc, u.locator, {"date": ds[0]}))
    for doc, t in iter_tables(kb, ("backlog",)):
        for r in iter_rows(doc, t):
            d = row_date(r, "due", "date")
            if d and r.title:
                events.append(Item(truncate(r.title, 160), doc.name, r.locator,
                                   {"date": d, "done": norm_status(r.get("status")) == "done"}))
    res.dated_events = sorted(dedupe(events, key=lambda i: i.text), key=lambda i: i.meta["date"])[:20]

    # --- responsibilities
    owners: "OrderedDict[str, Responsibility]" = OrderedDict()

    def add_owner(name: str, work: str, src: str) -> None:
        name = _owner_name(name)
        if not name or len(name) > 40:
            return
        r = owners.setdefault(name.lower(), Responsibility(name, [], []))
        if work and work not in r.work:
            r.work.append(truncate(work, 140))
        if src not in r.sources:
            r.sources.append(src)

    for doc, t in iter_tables(kb):
        for r in iter_rows(doc, t):
            o = r.get("owner")
            if o and t.kind != "generic":
                prefix = "Fix defect: " if t.kind == "defects" else ""
                work = r.get("title") or r.find("activity")
                add_owner(o, (prefix + work) if work else "", f"{doc.name} · {r.locator}")
    for u in body:
        if not u.is_plan_content:
            o = find_owner(u.text)
            if o:
                add_owner(o, u.text, f"{u.doc} · {u.locator}")
    res.responsibilities = list(owners.values())
    return res


# =============================================================================
# 2. Risk Detection & Delivery Forecasting Agent
# =============================================================================
def run_risk_agent(kb: KnowledgeBase, units: List[Unit], scope: ScopeResult) -> RiskResult:
    now = datetime.now()
    risks: List[Risk] = []

    def add(title, category, severity, doc, locator, kind="Detected", owner="", detail=""):
        p, i = _sev_pm(severity, category)
        risks.append(Risk("", truncate(title, 150), category, severity, p, i, kind, doc, locator, owner,
                          truncate(detail, 260)))

    # --- narrative signals
    for u in units:
        if u.heading or len(u.text) < 25 or GENERIC_STATEMENT_RE.search(u.text):
            continue
        strong = bool(STRONG_ISSUE_RE.search(u.text))
        if u.is_plan_content and not strong:
            continue
        for cat, rx in RISK_RES.items():
            if rx.search(u.text):
                sev = _severity(u.text)
                add(u.text, cat, sev, u.doc, u.locator, owner=find_owner(u.text))
                break
        else:
            if GENERAL_RISK_RE.search(u.text) and not u.is_plan_content:
                add(u.text, "General", _severity(u.text), u.doc, u.locator)

    # --- tracker signals
    done = total = open_defects = failing = slipped = blocked = overdue = 0
    ids_status: Dict[str, str] = {}
    backlog_rows: List[tuple] = []
    for doc, t in iter_tables(kb):
        for r in iter_rows(doc, t):
            if t.kind == "defects":
                st = norm_status(r.get("status"))
                if st != "done":
                    open_defects += 1
                    sev = "High" if re.search(r"critical|high|blocker|severe", r.get("priority") + r.title, re.I) else "Medium"
                    add(f"Open defect: {r.title}", "Quality", sev, doc.name, r.locator, owner=_owner_name(r.get("owner")),
                        detail=f"Type: {r.get('type') or 'n/a'}; detected in {r.get('sprint') or 'n/a'}. {r.get('remarks')}")
                elif re.search(r"pending|retest|verify|not (?:yet )?verified", r.get("remarks"), re.I):
                    add(f"Defect closure not yet verified: {r.title}", "Quality", "Low", doc.name, r.locator,
                        owner=_owner_name(r.get("owner")), detail=r.get("remarks"))
            elif t.kind == "tests":
                actual = r.get("actual")
                if not actual:
                    add(f"Test case not executed: {r.title}", "Quality", "Medium", doc.name, r.locator)
                    continue
                bad, good = UNRESOLVED_RE.search(actual), RESOLVED_RE.search(actual)
                if bad and not good:
                    failing += 1
                    add(f"Unresolved test finding: {r.title}", "Technical", "High", doc.name, r.locator, detail=actual)
                elif bad and good:
                    add(f"Issue found during testing (fix claimed): {r.title}", "Technical", "Low", doc.name,
                        r.locator, detail=actual)
            elif t.kind == "backlog":
                st = norm_status(r.get("status"))
                uid = r.find("id") or r.title[:30]
                ids_status[uid.lower()] = st
                if r.title:
                    total += 1
                    done += st == "done"
                backlog_rows.append((doc, r, st))
                if st == "blocked":
                    blocked += 1
                    add(f"Blocked work item: {r.title}", "Schedule", "High", doc.name, r.locator,
                        owner=_owner_name(r.get("owner")))
                elif st == "delayed":
                    add(f"Delayed work item: {r.title}", "Schedule", "Medium", doc.name, r.locator,
                        owner=_owner_name(r.get("owner")))
                planned, actual_sprint = r.get("sprint"), r.get("actual")
                if planned and actual_sprint and planned.lower() != actual_sprint.lower():
                    slipped += 1
                    add(f"Scope slipped from {planned} to {actual_sprint}: {r.title}", "Schedule", "Medium",
                        doc.name, r.locator, owner=_owner_name(r.get("owner")))
                due = row_date(r, "due")
                if due and due < now and st != "done":
                    overdue += 1
                    add(f"Overdue ({fmt_date(due)}): {r.title}", "Schedule", "High", doc.name, r.locator,
                        owner=_owner_name(r.get("owner")))
            elif r.find("impediment"):
                imp, act = r.find("impediment"), r.find("action")
                if not RESOLVED_RE.search(act):
                    add(f"Impediment reported: {imp}", "Technical", "Medium", doc.name, r.locator, detail=act)
    for doc, r, st in backlog_rows:                         # unmet dependencies
        dep = r.get("dependency")
        if dep and st != "done":
            for d in re.findall(r"[A-Za-z]+-?\d+", dep):
                if ids_status.get(d.lower(), "todo") != "done":
                    add(f"Dependency gap: {r.title[:80]} depends on {d} which is not complete", "Dependency",
                        "Medium", doc.name, r.locator, owner=_owner_name(r.get("owner")))

    # --- structural (inferred) signals
    has_dates = bool(scope.dated_events) or any(p.window for p in scope.timeline)
    some_doc = next(iter(kb.documents), "")
    if kb.documents and not has_dates:
        add("No explicit delivery dates found in the uploaded documents", "Schedule", "Medium", some_doc, "All documents",
            "Inferred", detail="Without dated commitments, slippage cannot be measured.")
    if kb.documents and not scope.responsibilities:
        add("No named owners found for deliverables or tasks", "Governance", "Medium", some_doc, "All documents",
            "Inferred", detail="Unowned work is the most common cause of silent delivery slippage.")
    for ph in scope.timeline:
        if ph.hours and ph.tasks and len(ph.tasks) / ph.hours >= 0.35:
            add(f"{ph.label}: {len(ph.tasks)} tasks planned in a ~{ph.hours}-hour window", "Schedule", "Medium",
                ph.doc, ph.locator, "Inferred",
                detail="Task density is high relative to the stated effort budget; estimation buffer is thin.")
    text_phases = [p for p in scope.timeline if p.weeks]
    if len(text_phases) >= 2:
        first = text_phases[0]
        add(f"Sequential delivery plan: a slip in '{first.label}' cascades into {len(text_phases) - 1} later phase(s)",
            "Dependency", "Medium", first.doc, first.locator, "Inferred",
            detail="Later phases build on the outputs of earlier ones.")
    if kb.documents and not any(t.kind in ("tests", "defects") for _, t in iter_tables(kb)):
        add("No test plan or defect log provided — quality status is unknown", "Quality", "Low", some_doc,
            "All documents", "Inferred")

    # --- finalise register
    sev_rank = {"High": 0, "Medium": 1, "Low": 2}
    risks = dedupe(risks, key=lambda r: r.title, threshold=0.85)
    risks.sort(key=lambda r: (sev_rank[r.severity], -r.score, r.evidence_type != "Detected"))
    risks = risks[:30]
    for n, r in enumerate(risks, start=1):
        r.risk_id = f"R-{n:02d}"

    return RiskResult(risks=risks, forecast=_forecast(kb, scope, risks, done, total, open_defects, failing,
                                                       slipped, blocked, overdue, has_dates, now))


def _forecast(kb, scope, risks, done, total, open_defects, failing, slipped, blocked, overdue, has_dates, now) -> Forecast:
    if kb.is_empty:
        return Forecast()
    penalty = min(45, sum({"High": 6, "Medium": 3, "Low": 1}[r.severity] for r in risks))
    bonus = (done / total - 0.5) * 10 if total else 0
    conf_points = int(total > 0) + int(has_dates) + int(open_defects + failing > 0 or any(
        t.kind in ("tests", "defects") for _, t in iter_tables(kb)))
    confidence = ["Low", "Low", "Medium", "High"][conf_points]
    pct = clamp(85 - penalty + bonus, 5, 97)
    if confidence == "Low":
        pct = min(pct, 72)
    level = ("Likely on track" if pct >= 75 else "At risk of slippage" if pct >= 55
             else "Likely to slip" if pct >= 35 else "High likelihood of delay")
    drivers = [f"{r.severity}: {truncate(r.title, 90)}" for r in risks[:4]]
    if total:
        drivers.insert(0, f"Progress: {done}/{total} tracked work items complete ({round(100 * done / total)}%)")
    stats = [f"{n} {w}" for n, w in ((open_defects, "open defect(s)"), (failing, "unresolved test finding(s)"),
                                    (slipped, "item(s) slipped between sprints"), (blocked, "blocked item(s)"),
                                    (overdue, "overdue item(s)")) if n]
    narrative = (f"Estimated on-time delivery likelihood is {pct:.0f}% ({level.lower()}), "
                 f"with {confidence.lower()} confidence given the evidence available. ")
    narrative += ("Contributing signals: " + ", ".join(stats) + "." if stats else
                  "No tracker-level delay signals were found; the estimate rests on narrative risk signals.")
    key_dates = []
    for ev in scope.dated_events:
        d = ev.meta["date"]
        key_dates.append(f"{fmt_date(d)} — {truncate(ev.text, 90)} ({'completed' if ev.meta.get('done') else 'date passed' if d < now else 'upcoming'})")
    return Forecast(round(pct, 1), level, confidence, narrative, drivers, done, total, key_dates[:6])


# =============================================================================
# 3. Blocker & Action Item Identification Agent
# =============================================================================
BLOCKER_RE = re.compile(r"\b(blocked|blocker\s*:|impediment|stuck|unable to|cannot|can't|waiting (?:for|on)|"
                        r"held up|on hold|unresolved|outstanding issue|not working|failing|"
                        r"issues? (?:with|in|while|faced|encountered)|faced (?:some |the )?issues?|problems? (?:with|in|while))\b", re.I)
DECISION_RE = re.compile(r"\b(to be decided|undecided|decision|decide|tbd|tbc|to be confirmed|approval|approve|"
                         r"sign[- ]?off|need to choose|open question|clarif\w+|pending (?:confirmation|approval))\b", re.I)
ACTION_RE = re.compile(r"(?:\baction(?: item)?s?\s*[:\-]|\btodo\b|\bto-do\b|follow[- ]?up|\bneeds? to\b|\bshould\b|"
                       r"\bmust\b|\bassigned to\b|\bowner\s*:|\bnext steps?\b|\bwill\b)", re.I)


def _priority(text: str) -> str:
    if re.search(r"critical|urgent|blocker|asap|high|must|overdue", text, re.I):
        return "High"
    return "Low" if re.search(r"nice to have|optional|low", text, re.I) else "Medium"


def run_blocker_agent(kb: KnowledgeBase, units: List[Unit], scope: ScopeResult) -> BlockerResult:
    now = datetime.now()
    blockers: List[Blocker] = []
    decisions: List[Item] = []
    actions: List[Action] = []

    def act(text, owner, due, priority, kind, status, doc, loc):
        actions.append(Action("", truncate(text, 220), _owner_name(owner) or "Unassigned", due or "TBD", priority,
                              kind, status, doc, loc))

    # --- narrative
    phase_for = {}
    for ph in scope.timeline:
        for t in ph.tasks:
            phase_for.setdefault(t, ph)
    for u in units:
        if u.heading or len(u.text) < 20 or GENERIC_STATEMENT_RE.search(u.text):
            continue
        strong = bool(STRONG_ISSUE_RE.search(u.text))
        if u.is_plan_content and not strong and not DECISION_RE.search(u.text):
            ph = phase_for.get(truncate(u.text, 220))
            if ph and TASK_VERB_RE.match(u.text):
                act(u.text, find_owner(u.text), ph.window or ph.label, "Medium", "Planned task", "Planned", u.doc, u.locator)
            continue
        if BLOCKER_RE.search(u.text):
            resolved = bool(RESOLVED_RE.search(u.text)) and not strong
            blockers.append(Blocker(truncate(u.text, 240), "Resolved" if resolved else "Open", _severity(u.text),
                                    _owner_name(find_owner(u.text)), u.doc, u.locator))
        elif DECISION_RE.search(u.text):
            decisions.append(Item(truncate(u.text, 240), u.doc, u.locator, {"owner": find_owner(u.text)}))
        elif ACTION_RE.search(u.text):
            ds = parse_dates(u.text)
            act(u.text, find_owner(u.text), fmt_date(ds[0]) if ds else "", _priority(u.text), "Follow-up", "Open",
                u.doc, u.locator)

    # --- trackers
    for doc, t in iter_tables(kb):
        for r in iter_rows(doc, t):
            owner = r.get("owner")
            if t.kind == "defects":
                st = norm_status(r.get("status"))
                if st != "done":
                    blockers.append(Blocker(truncate(r.title, 240), "Open", _severity(r.title + " " + r.get("priority")),
                                            _owner_name(owner), doc.name, r.locator, r.get("action")))
                    act(f"Resolve and retest defect: {r.title}", owner, fmt_date(row_date(r, "action_date")), "High",
                        "Defect fix", "Open", doc.name, r.locator)
                elif re.search(r"pending|retest|verify|not (?:yet )?verified", r.get("remarks"), re.I):
                    act(f"Verify and formally close defect: {r.title} ({r.get('remarks')})", owner, "", "Medium",
                        "Defect closure", "Open", doc.name, r.locator)
            elif t.kind == "tests":
                actual = r.get("actual")
                if not actual:
                    act(f"Execute test case: {r.title}", owner, "", "Medium", "Testing", "Open", doc.name, r.locator)
                elif UNRESOLVED_RE.search(actual) and not RESOLVED_RE.search(actual):
                    blockers.append(Blocker(truncate(f"{r.title}: {actual}", 240), "Open", "High", _owner_name(owner),
                                            doc.name, r.locator))
                    act(f"Investigate failing test: {r.title}", owner, "", "High", "Testing", "Open", doc.name, r.locator)
            elif t.kind == "backlog":
                st = norm_status(r.get("status"))
                if st == "blocked":
                    blockers.append(Blocker(truncate(r.title, 240), "Open", "High", _owner_name(owner), doc.name,
                                            r.locator))
                if st != "done" and r.title and ("sprint" in t.roles or "owner" in t.roles):
                    due = row_date(r, "due")
                    act(f"Complete: {r.title}", owner, fmt_date(due) if due else r.get("sprint"),
                        "High" if st in ("blocked", "delayed") else "Medium", "Backlog item",
                        {"progress": "In Progress", "todo": "Open"}.get(st, "Open"), doc.name, r.locator)
            if r.find("impediment"):
                imp, taken = r.find("impediment"), r.find("action")
                solved = bool(RESOLVED_RE.search(taken))
                blockers.append(Blocker(truncate(imp, 240), "Resolved" if solved else "Open", "Medium",
                                        _owner_name(owner), doc.name, r.locator, taken))
            for col, val in r.raw.items():
                if "start doing" in col.lower() and val:
                    act(val, owner, "", "Medium", "Process improvement", "Open", doc.name, r.locator)
            rem = r.get("remarks")
            if rem and DECISION_RE.search(rem):
                decisions.append(Item(truncate(rem, 240), doc.name, r.locator, {"owner": _owner_name(owner)}))

    blockers = dedupe(blockers, key=lambda b: b.text)
    blockers.sort(key=lambda b: (b.status != "Open", {"High": 0, "Medium": 1, "Low": 2}[b.severity]))
    decisions = dedupe(decisions, key=lambda d: d.text)[:15]
    actions = dedupe(actions, key=lambda a: a.text)
    actions.sort(key=lambda a: ({"High": 0, "Medium": 1, "Low": 2}[a.priority], a.owner == "Unassigned"))
    actions = actions[:40]
    for n, a in enumerate(actions, start=1):
        a.action_id = f"A-{n:02d}"
    return BlockerResult(blockers[:25], decisions, actions)


# =============================================================================
# Orchestrator
# =============================================================================
def run_pipeline(kb: KnowledgeBase, progress: Optional[Callable[[float, str], None]] = None) -> PipelineResult:
    """Run all analysis agents. A failing agent never prevents the others from running."""
    out = PipelineResult(generated_at=datetime.now())
    if kb.is_empty:
        return out
    try:
        units = build_units(kb)
    except Exception as exc:
        log.exception("Unit extraction failed")
        units = []
        out.runs.append(AgentRun("Document analysis", "error", 0.0, 0, f"{exc.__class__.__name__}: {exc}"))

    def step(key: str, label: str, fn, store: Callable, counter: Callable, frac: float):
        if progress:
            progress(frac, f"{label}…")
        t0 = time.perf_counter()
        try:
            result = fn()
            store(result)
            out.runs.append(AgentRun(label, "success", time.perf_counter() - t0, counter(result)))
        except Exception as exc:
            log.exception("%s failed", label)
            out.runs.append(AgentRun(label, "error", time.perf_counter() - t0, 0, f"{exc.__class__.__name__}: {exc}"))

    name = {k: n for k, n, _, _ in AGENT_CATALOG}
    step("scope", name["scope"], lambda: run_scope_agent(kb, units), lambda r: setattr(out, "scope", r),
         lambda r: len(r.goals) + len(r.deliverables) + len(r.timeline) + len(r.responsibilities), 0.15)
    step("risk", name["risk"], lambda: run_risk_agent(kb, units, out.scope), lambda r: setattr(out, "risk", r),
         lambda r: len(r.risks), 0.45)
    step("blockers", name["blockers"], lambda: run_blocker_agent(kb, units, out.scope),
         lambda r: setattr(out, "blockers", r), lambda r: len(r.blockers) + len(r.decisions) + len(r.actions), 0.75)
    if progress:
        progress(1.0, "Analysis complete")
    return out
