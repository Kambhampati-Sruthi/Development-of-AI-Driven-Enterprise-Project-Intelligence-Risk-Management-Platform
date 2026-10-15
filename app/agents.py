
import re
from datetime import date, datetime


def _clean_text(value):
    """Normalize whitespace in extracted text."""
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _unique(items):
    """Return unique items while preserving their original order."""
    seen = set()
    result = []

    for item in items:
        key = _clean_text(item).lower()
        if key and key not in seen:
            seen.add(key)
            result.append(item)

    return result


def _parse_date(value):
    """Parse a YYYY-MM-DD date when available."""
    if not value:
        return None

    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None


def _extract_owner(text):
    """Extract a simple explicitly labelled owner or assignee."""
    match = re.search(
        r"\b(?:owner|assignee|assigned\s+to)\s*[:\-]\s*"
        r"([A-Za-z][A-Za-z .'-]{1,60})",
        text,
        re.IGNORECASE,
    )

    if match:
        return _clean_text(match.group(1)).rstrip(".,;")

    return None


def _extract_due_date(text):
    """Extract a date in YYYY-MM-DD format."""
    match = re.search(
        r"\b(?:due(?:\s+date)?|deadline|by)\s*[:\-]?\s*"
        r"(\d{4}-\d{2}-\d{2})\b",
        text,
        re.IGNORECASE,
    )

    if not match:
        match = re.search(r"\b(\d{4}-\d{2}-\d{2})\b", text)

    if match and _parse_date(match.group(1)):
        return match.group(1)

    return None


class ScopeDeliverablesAgent:
    """Extract project objectives, scope statements, and deliverables."""

    OBJECTIVE_PATTERN = re.compile(
        r"\b(?:objective|goal|purpose|aim)\s*[:\-]\s*(.+)",
        re.IGNORECASE,
    )

    DELIVERABLE_PATTERN = re.compile(
        r"\b(?:deliverable|output|milestone)\s*[:\-]\s*(.+)",
        re.IGNORECASE,
    )

    SCOPE_PATTERN = re.compile(
        r"\b(?:in scope|scope|out of scope)\s*[:\-]\s*(.+)",
        re.IGNORECASE,
    )

    def analyze(self, text):
        text = text or ""

        objectives = []
        deliverables = []
        scope_items = []

        for raw_line in text.splitlines():
            line = _clean_text(raw_line)

            if not line:
                continue

            objective_match = self.OBJECTIVE_PATTERN.search(line)
            deliverable_match = self.DELIVERABLE_PATTERN.search(line)
            scope_match = self.SCOPE_PATTERN.search(line)

            if objective_match:
                objectives.append({
                    "description": objective_match.group(1).strip(),
                    "source_text": line,
                })

            if deliverable_match:
                deliverables.append({
                    "description": deliverable_match.group(1).strip(),
                    "due_date": _extract_due_date(line),
                    "owner": _extract_owner(line),
                    "source_text": line,
                })

            if scope_match:
                scope_items.append({
                    "description": scope_match.group(1).strip(),
                    "source_text": line,
                })

        # Support simple bullet-list deliverables when the document
        # explicitly labels a deliverables section.
        in_deliverables_section = False

        for raw_line in text.splitlines():
            line = _clean_text(raw_line)

            if not line:
                continue

            if re.search(
                r"^\s*(?:project\s+)?deliverables\s*:?\s*$",
                line,
                re.IGNORECASE,
            ):
                in_deliverables_section = True
                continue

            if re.search(
                r"^\s*(?:objectives?|risks?|blockers?|action items?|"
                r"scope|assumptions|dependencies)\s*:?\s*$",
                line,
                re.IGNORECASE,
            ):
                in_deliverables_section = False
                continue

            if in_deliverables_section and re.match(
                r"^(?:[-*•]|\d+[.)])\s+", line
            ):
                description = re.sub(
                    r"^(?:[-*•]|\d+[.)])\s+",
                    "",
                    line,
                ).strip()

                if description:
                    deliverables.append({
                        "description": description,
                        "due_date": _extract_due_date(line),
                        "owner": _extract_owner(line),
                        "source_text": line,
                    })

        objectives = _unique(objectives)
        deliverables = _unique(deliverables)
        scope_items = _unique(scope_items)

        return {
            "objectives": objectives,
            "deliverables": deliverables,
            "scope_items": scope_items,
            "objective_count": len(objectives),
            "deliverable_count": len(deliverables),
            "scope_item_count": len(scope_items),
        }


class RiskDetectionAgent:
    """Detect risk signals using transparent keyword-based rules."""

    RISK_KEYWORDS = {
        "schedule": [
            "delay", "behind schedule", "late delivery",
            "missed deadline", "schedule risk", "slippage",
        ],
        "budget": [
            "budget overrun", "cost overrun", "overspend",
            "funding shortage", "budget risk",
        ],
        "resource": [
            "resource shortage", "staff shortage", "understaffed",
            "lack of resources", "resource constraint",
        ],
        "quality": [
            "quality issue", "defect", "failure", "testing issue",
            "high bug count", "performance issue",
        ],
        "dependency": [
            "dependency risk", "third-party delay",
            "vendor delay", "external dependency",
            "waiting on vendor",
        ],
        "scope": [
            "scope creep", "unclear requirements",
            "changing requirements", "requirements gap",
        ],
        "security": [
            "security vulnerability", "data breach",
            "security risk", "privacy risk",
        ],
    }

    SEVERITY_KEYWORDS = {
        "critical": [
            "critical", "severe", "project failure",
            "business outage", "production outage",
        ],
        "high": [
            "high risk", "major", "significant",
            "urgent", "release delay",
        ],
        "medium": [
            "medium risk", "moderate", "may affect",
            "potential issue",
        ],
        "low": [
            "low risk", "minor", "small impact",
        ],
    }

    def _severity(self, text):
        lowered = text.lower()

        for severity in ("critical", "high", "medium", "low"):
            if any(
                keyword in lowered
                for keyword in self.SEVERITY_KEYWORDS[severity]
            ):
                return severity

        if any(
            term in lowered
            for term in ("risk", "delay", "blocked", "failure", "threat")
        ):
            return "medium"

        return "low"

    def analyze(self, text):
        risks = []
        seen = set()

        for raw_line in (text or "").splitlines():
            line = _clean_text(raw_line)

            if not line:
                continue

            lowered = line.lower()
            matched_categories = []

            for category, keywords in self.RISK_KEYWORDS.items():
                if any(keyword in lowered for keyword in keywords):
                    matched_categories.append(category)

            explicitly_labelled = bool(
                re.search(r"\b(?:risk|threat)\s*[:\-]", line, re.I)
            )

            if not matched_categories and not explicitly_labelled:
                continue

            key = lowered
            if key in seen:
                continue
            seen.add(key)

            risks.append({
                "description": line,
                "category": (
                    matched_categories[0]
                    if matched_categories else "general"
                ),
                "matched_categories": matched_categories,
                "severity": self._severity(line),
                "signals": matched_categories or ["explicit risk statement"],
                "owner": _extract_owner(line),
                "due_date": _extract_due_date(line),
                "source_text": line,
                "status": "Detected - requires validation",
            })

        severity_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }
        risks.sort(
            key=lambda risk: severity_order.get(risk["severity"], 4)
        )

        return {
            "risks": risks,
            "risk_count": len(risks),
            "critical_count": sum(
                risk["severity"] == "critical" for risk in risks
            ),
            "high_count": sum(
                risk["severity"] == "high" for risk in risks
            ),
            "medium_count": sum(
                risk["severity"] == "medium" for risk in risks
            ),
            "low_count": sum(
                risk["severity"] == "low" for risk in risks
            ),
        }


class BlockerIdentificationAgent:
    """Identify potential blockers and impediments."""

    BLOCKER_KEYWORDS = [
        "blocked", "blocker", "waiting for approval",
        "pending approval", "awaiting approval",
        "dependency not resolved", "access denied",
        "environment unavailable", "cannot proceed",
        "unable to proceed", "work is stalled",
        "on hold", "unresolved dependency",
    ]

    def analyze(self, text):
        blockers = []
        seen = set()

        for raw_line in (text or "").splitlines():
            line = _clean_text(raw_line)

            if not line:
                continue

            lowered = line.lower()
            matched = [
                keyword
                for keyword in self.BLOCKER_KEYWORDS
                if keyword in lowered
            ]

            explicitly_labelled = bool(
                re.search(r"\bblocker\s*[:\-]", line, re.I)
            )

            if not matched and not explicitly_labelled:
                continue

            if lowered in seen:
                continue
            seen.add(lowered)

            blockers.append({
                "description": line,
                "severity": RiskDetectionAgent()._severity(line),
                "signals": matched or ["explicit blocker statement"],
                "owner": _extract_owner(line),
                "due_date": _extract_due_date(line),
                "status": "Unresolved signal; verify manually",
                "source_text": line,
            })

        return {
            "blockers": blockers,
            "blocker_count": len(blockers),
        }


class ActionItemAgent:
    """Identify action items, assignees, deadlines, and statuses."""

    ACTION_KEYWORDS = [
        "action item", "action:", "todo", "to-do",
        "task:", "next step", "follow up", "follow-up",
        "must complete", "need to complete", "should complete",
        "will complete", "responsible for",
    ]

    COMPLETED_KEYWORDS = [
        "completed", "done", "closed", "resolved",
    ]

    def analyze(self, text):
        action_items = []
        seen = set()

        for raw_line in (text or "").splitlines():
            line = _clean_text(raw_line)

            if not line:
                continue

            lowered = line.lower()

            is_action = any(
                keyword in lowered
                for keyword in self.ACTION_KEYWORDS
            )

            # Recognize simple task-list lines, for example:
            # - [ ] Complete integration testing
            # 1. Review requirements
            is_list_item = bool(
                re.match(
                    r"^(?:[-*•]\s*(?:\[[ xX]\]\s*)?|\d+[.)]\s+)",
                    line,
                )
            )

            if not is_action and not is_list_item:
                continue

            if lowered in seen:
                continue
            seen.add(lowered)

            if any(
                keyword in lowered
                for keyword in self.COMPLETED_KEYWORDS
            ):
                status = "Completed - verify source"
            elif re.search(r"\b(?:in progress|ongoing)\b", lowered):
                status = "In progress - verify source"
            else:
                status = "Needs verification"

            action_items.append({
                "description": line,
                "owner": _extract_owner(line),
                "due_date": _extract_due_date(line),
                "status": status,
                "source_text": line,
            })

        return {
            "action_items": action_items,
            "action_item_count": len(action_items),
        }


class ProjectAnalysisAgents:
    """Coordinate the agents for the project's analysis milestones."""

    def __init__(self):
        self.scope_agent = ScopeDeliverablesAgent()
        self.risk_agent = RiskDetectionAgent()
        self.blocker_agent = BlockerIdentificationAgent()
        self.action_agent = ActionItemAgent()

    def forecast_delivery(
        self,
        analysis,
        planned_end_date=None,
        today=None,
    ):
        """
        Produce a heuristic delivery-risk indicator.

        This is not a trained model and does not calculate a calibrated
        probability of delay.
        """
        analysis = analysis or {}
        today = today or date.today()

        if isinstance(today, str):
            today = _parse_date(today)

        if today is None:
            today = date.today()

        risk_data = analysis.get("risk_analysis", {}) or {}
        blocker_data = analysis.get("blocker_analysis", {}) or {}
        action_data = analysis.get("action_analysis", {}) or {}

        risks = risk_data.get("risks", []) or []
        blockers = blocker_data.get("blockers", []) or []
        actions = action_data.get("action_items", []) or []

        critical_risks = sum(
            str(risk.get("severity", "")).lower() == "critical"
            for risk in risks
        )
        high_risks = sum(
            str(risk.get("severity", "")).lower() == "high"
            for risk in risks
        )

        unresolved_blockers = [
            blocker for blocker in blockers
            if str(blocker.get("status", "")).lower()
            not in {"resolved", "closed", "completed"}
        ]

        overdue_actions = []

        for action in actions:
            due_date = _parse_date(action.get("due_date"))
            status = str(action.get("status", "")).lower()

            if (
                due_date is not None
                and due_date < today
                and status not in {
                    "completed", "done", "closed", "resolved"
                }
            ):
                overdue_actions.append(action)

        pressure = min(
            100,
            critical_risks * 25
            + high_risks * 15
            + len(unresolved_blockers) * 12
            + len(overdue_actions) * 8,
        )

        if pressure >= 60:
            forecast_status = "High delivery risk"
        elif pressure >= 25:
            forecast_status = "Delivery at risk"
        else:
            forecast_status = "No major delivery warning detected"

        planned_date = _parse_date(planned_end_date)
        days_until_due = (
            (planned_date - today).days
            if planned_date is not None else None
        )

        warnings = []

        if critical_risks:
            warnings.append(
                f"{critical_risks} critical risk(s) detected"
            )
        if high_risks:
            warnings.append(
                f"{high_risks} high risk(s) detected"
            )
        if unresolved_blockers:
            warnings.append(
                f"{len(unresolved_blockers)} unresolved blocker(s) detected"
            )
        if overdue_actions:
            warnings.append(
                f"{len(overdue_actions)} overdue action(s) detected"
            )
        if planned_date is None:
            warnings.append(
                "No valid planned end date supplied; "
                "calendar-based delivery assessment is limited"
            )
        elif days_until_due < 0:
            forecast_status = "Planned delivery date has passed"

        return {
            "forecast_status": forecast_status,
            "delivery_pressure_score": pressure,
            "planned_end_date": (
                planned_date.isoformat() if planned_date else None
            ),
            "days_until_planned_end": days_until_due,
            "critical_risk_count": critical_risks,
            "high_risk_count": high_risks,
            "unresolved_blocker_count": len(unresolved_blockers),
            "overdue_action_count": len(overdue_actions),
            "warnings": warnings,
            "calculation_note": (
                "Heuristic delivery-risk indicator, not a calibrated "
                "delay probability or guaranteed completion date."
            ),
        }

    def analyze(self, text, planned_end_date=None, today=None):
        """Run all analysis agents and produce a delivery forecast."""
        scope = self.scope_agent.analyze(text)
        risk_analysis = self.risk_agent.analyze(text)
        blocker_analysis = self.blocker_agent.analyze(text)
        action_analysis = self.action_agent.analyze(text)

        analysis = {
            "scope": scope,
            "risk_analysis": risk_analysis,
            "blocker_analysis": blocker_analysis,
            "action_analysis": action_analysis,
        }

        analysis["delivery_forecast"] = self.forecast_delivery(
            analysis,
            planned_end_date=planned_end_date,
            today=today,
        )

        return analysis
