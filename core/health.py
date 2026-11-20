"""Project health scoring with a dimension-wise breakdown."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .config import HEALTH_BANDS, HEALTH_WEIGHTS
from .models import PipelineResult
from .rag import KnowledgeBase
from .utils import clamp, truncate


@dataclass
class Dimension:
    key: str
    label: str
    score: float
    weight: float
    summary: str
    signals: List[str] = field(default_factory=list)


@dataclass
class HealthReport:
    has_data: bool = False
    overall: float = 0.0
    band: str = "No data"
    color: str = "#94a3b8"
    dimensions: List[Dimension] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


def band_for(score: float):
    for minimum, label, colour in HEALTH_BANDS:
        if score >= minimum:
            return label, colour
    return HEALTH_BANDS[-1][1], HEALTH_BANDS[-1][2]


def compute_health(kb: KnowledgeBase, res: PipelineResult) -> HealthReport:
    if kb.is_empty or not res.runs:
        return HealthReport()
    scope, risk, blk = res.scope, res.risk, res.blockers

    # ---- Scope clarity (how well-defined is the work?)
    sig: List[str] = []
    pts = 0.0
    g = min(len(scope.goals), 3) / 3 * 25
    d = min(len(scope.deliverables), 5) / 5 * 25
    has_dates = bool(scope.dated_events) or any(p.window for p in scope.timeline)
    t = min(len(scope.timeline), 2) / 2 * 12 + (8 if has_dates else 0)
    o = 20 if len(scope.responsibilities) >= 2 else 10 if scope.responsibilities else 0
    v = 10 if len(kb.documents) >= 2 else 5
    pts = g + d + t + o + v
    sig += [f"{len(scope.goals)} goal(s) identified", f"{len(scope.deliverables)} deliverable(s) identified",
            f"{len(scope.timeline)} timeline phase(s), dated commitments: {'yes' if has_dates else 'no'}",
            f"{len(scope.responsibilities)} named owner(s)"]
    scope_risks = [r for r in risk.risks if r.category in ("Scope", "Governance")]
    pts -= min(15, 4 * len(scope_risks))
    scope_score = clamp(pts)

    # ---- Timeline risk (higher score = lower risk)
    weights = {"High": 14, "Medium": 7, "Low": 2}
    timeline_risks = [r for r in risk.risks if r.category in ("Schedule", "Dependency")]
    penalty = min(80, sum(weights[r.severity] for r in timeline_risks))
    base = 100 - penalty
    fc = risk.forecast
    timeline_score = clamp(0.6 * base + 0.4 * fc.on_time_pct) if fc.total or fc.key_dates or timeline_risks else clamp(base)
    tsig = [f"{len(timeline_risks)} schedule/dependency risk(s)", f"forecast on-time likelihood {fc.on_time_pct:.0f}% "
            f"({fc.confidence.lower()} confidence)"]
    if fc.total:
        tsig.append(f"{fc.done}/{fc.total} tracked work items complete")

    # ---- Blocker count
    open_b = [b for b in blk.blockers if b.status == "Open"]
    bpen = sum({"High": 12, "Medium": 7, "Low": 3}[b.severity] for b in open_b) + min(15, 3 * len(blk.decisions))
    blocker_score = clamp(100 - bpen)
    bsig = [f"{len(open_b)} open blocker(s)", f"{len(blk.blockers) - len(open_b)} resolved",
            f"{len(blk.decisions)} pending decision(s)"]

    dims = [
        Dimension("scope", "Scope clarity", scope_score, HEALTH_WEIGHTS["scope"],
                  "How clearly goals, deliverables, dates and owners are defined.", sig),
        Dimension("timeline", "Timeline risk", timeline_score, HEALTH_WEIGHTS["timeline"],
                  "Resilience of the schedule (100 = low risk).", tsig),
        Dimension("blockers", "Blocker load", blocker_score, HEALTH_WEIGHTS["blockers"],
                  "Open blockers and undecided questions (100 = unobstructed).", bsig),
    ]
    overall = round(sum(x.score * x.weight for x in dims), 1)
    band, colour = band_for(overall)

    recs: List[str] = []
    if scope_score < 70:
        if not scope.responsibilities:
            recs.append("Assign a named owner to every deliverable and task so accountability is explicit.")
        if not has_dates:
            recs.append("Add dated commitments (start/finish or due dates) to each phase or sprint.")
        if len(scope.goals) < 2 or len(scope.deliverables) < 3:
            recs.append("Document clear goals and a concrete deliverable list in a scope or requirements document.")
    if timeline_score < 70 and timeline_risks:
        top = timeline_risks[0]
        recs.append(f"Address the top schedule risk first: {truncate(top.title, 110)} ({top.doc}).")
        if any("tasks planned" in r.title for r in timeline_risks):
            recs.append("Add buffer to dense phases and re-baseline dates where work has slipped between sprints.")
        else:
            recs.append("Re-baseline dates for blocked, overdue or slipped work and confirm the new commitments with owners.")
    if blocker_score < 70 and open_b:
        recs.append(f"Unblock first: {truncate(open_b[0].text, 110)} — assign an owner and a resolution date.")
    if blk.decisions:
        recs.append("Schedule a short decision meeting to close the pending decisions listed in the Agent Center.")
    if not recs:
        recs.append("Maintain the current cadence; refresh this analysis whenever trackers or notes change.")
    return HealthReport(True, overall, band, colour, dims, recs[:6])
