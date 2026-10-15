
from datetime import date, datetime


def _parse_date(value):
    """Convert a supported date value into a date object."""
    if not value:
        return None

    if isinstance(value, datetime):
        return value.date()

    if isinstance(value, date):
        return value

    value = str(value).strip()

    formats = (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
    )

    for fmt in formats:
        try:
            return datetime.strptime(value, fmt).date()
        except ValueError:
            continue

    return None


def _count_overdue(items, today):
    """Count incomplete items whose due dates have passed."""
    count = 0

    for item in items:
        status = str(item.get("status", "")).lower()

        if status in {
            "completed",
            "done",
            "closed",
            "resolved",
            "finished",
        }:
            continue

        due_date = _parse_date(item.get("due_date"))

        if due_date and due_date < today:
            count += 1

    return count


def calculate_project_health(
    analysis,
    today=None,
    extra_metrics=None,
):
    """
    Calculate an explainable project health score from agent results.

    Args:
        analysis: Output from ProjectAnalysisAgents.analyze().
        today: Optional date for deterministic testing.
        extra_metrics: Optional dictionary containing:
            schedule_completion (0-100)
            budget_variance_pct (non-negative percentage)
            quality_issue_count (non-negative integer)

    Returns:
        A dictionary containing the score, status, deductions,
        counts, and recommendations.

    Note:
        This is a heuristic indicator, not a validated prediction
        of project success.
    """
    analysis = analysis or {}
    today = today or date.today()
    extra_metrics = extra_metrics or {}

    scope = analysis.get("scope", {}) or {}
    risk_analysis = analysis.get("risk_analysis", {}) or {}
    blocker_analysis = analysis.get("blocker_analysis", {}) or {}
    action_analysis = analysis.get("action_analysis", {}) or {}

    risks = risk_analysis.get("risks", []) or []
    blockers = blocker_analysis.get("blockers", []) or []
    action_items = action_analysis.get("action_items", []) or []
    deliverables = scope.get("deliverables", []) or []

    # Risk deductions are severity-weighted. Each individual
    # risk contributes at most one deduction.
    risk_weights = {
        "critical": 18,
        "high": 12,
        "medium": 6,
        "low": 2,
    }

    risk_deduction = 0

    for risk in risks:
        severity = str(
            risk.get("severity", "medium")
        ).lower()

        risk_deduction += risk_weights.get(severity, 6)

    # Blockers receive a stronger penalty because they can
    # directly prevent delivery.
    blocker_deduction = 0

    for blocker in blockers:
        severity = str(
            blocker.get("severity", "medium")
        ).lower()

        if severity == "critical":
            blocker_deduction += 15
        elif severity == "high":
            blocker_deduction += 10
        else:
            blocker_deduction += 6

    overdue_actions = _count_overdue(action_items, today)
    overdue_deliverables = _count_overdue(deliverables, today)

    overdue_deduction = (
        overdue_actions * 5
        + overdue_deliverables * 8
    )

    # Optional project-level metrics are only used if supplied.
    schedule_deduction = 0
    schedule_completion = extra_metrics.get(
        "schedule_completion"
    )

    if schedule_completion is not None:
        schedule_completion = float(schedule_completion)

        if not 0 <= schedule_completion <= 100:
            raise ValueError(
                "schedule_completion must be between 0 and 100."
            )

        schedule_deduction = round(
            (100 - schedule_completion) * 0.25
        )

    budget_deduction = 0
    budget_variance = extra_metrics.get("budget_variance_pct")

    if budget_variance is not None:
        budget_variance = float(budget_variance)

        if budget_variance < 0:
            raise ValueError(
                "budget_variance_pct must be non-negative."
            )

        budget_deduction = min(
            15,
            round(budget_variance * 0.5),
        )

    quality_issue_count = int(
        extra_metrics.get("quality_issue_count", 0)
    )

    if quality_issue_count < 0:
        raise ValueError(
            "quality_issue_count must be non-negative."
        )

    quality_deduction = min(
        15,
        quality_issue_count * 3,
    )

    deductions = {
        "risks": min(35, risk_deduction),
        "blockers": min(30, blocker_deduction),
        "overdue_items": min(25, overdue_deduction),
        "schedule": min(25, schedule_deduction),
        "budget": budget_deduction,
        "quality": quality_deduction,
    }

    # A score of 100 means no detected penalty, not guaranteed
    # project success.
    total_deduction = sum(deductions.values())
    score = max(0, min(100, 100 - total_deduction))

    if score >= 85:
        status = "Healthy"
        explanation = "Few negative signals were detected."
    elif score >= 70:
        status = "Needs Attention"
        explanation = "Some project risks need active monitoring."
    elif score >= 50:
        status = "At Risk"
        explanation = "Several negative signals need mitigation."
    else:
        status = "Critical"
        explanation = "Multiple serious signals need urgent review."

    recommendations = []

    if risks:
        recommendations.append(
            "Review detected risks, confirm their severity, "
            "and assign mitigation owners."
        )

    if blockers:
        recommendations.append(
            "Resolve blockers and confirm the next action "
            "for each affected deliverable."
        )

    if overdue_actions or overdue_deliverables:
        recommendations.append(
            "Review overdue actions and deliverables, then "
            "agree on revised dates where necessary."
        )

    if schedule_completion is not None and schedule_completion < 80:
        recommendations.append(
            "Review the schedule, critical path, and resource allocation."
        )

    if budget_variance is not None and budget_variance > 10:
        recommendations.append(
            "Investigate the budget variance and update the forecast."
        )

    if quality_issue_count:
        recommendations.append(
            "Prioritize quality issues and verify acceptance criteria."
        )

    if not recommendations:
        recommendations.append(
            "Continue monitoring project updates and validate "
            "the available project data."
        )

    return {
        "health_score": score,
        "status": status,
        "explanation": explanation,
        "deductions": deductions,
        "counts": {
            "risks": len(risks),
            "blockers": len(blockers),
            "action_items": len(action_items),
            "deliverables": len(deliverables),
            "overdue_actions": overdue_actions,
            "overdue_deliverables": overdue_deliverables,
            "quality_issues": quality_issue_count,
        },
        "recommendations": recommendations,
        "calculation_note": (
            "Heuristic score based on supplied signals. "
            "Missing schedule, budget, or quality data is not "
            "treated as evidence of good performance."
        ),
    }
