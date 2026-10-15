
import re
from datetime import date, datetime


# Patterns for extracting common project-management information.
DATE_PATTERN = re.compile(
    r"\b(?:"
    r"\d{4}-\d{1,2}-\d{1,2}"
    r"|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}"
    r"|(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|"
    r"May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\s+\d{1,2}(?:,?\s+\d{4})?"
    r")\b",
    re.IGNORECASE,
)

OWNER_PATTERN = re.compile(
    r"\b(?:owner|assigned to|assignee|responsible)\s*[:=-]\s*"
    r"([A-Z][A-Za-z0-9_. -]{1,50})",
    re.IGNORECASE,
)

SEVERITY_KEYWORDS = {
    "critical": ("critical", "severe", "project-stopping"),
    "high": ("high risk", "major", "significant", "urgent"),
    "medium": ("medium risk", "moderate", "potential"),
    "low": ("low risk", "minor"),
}

RISK_KEYWORDS = (
    "risk",
    "delay",
    "delayed",
    "overdue",
    "behind schedule",
    "slippage",
    "dependency",
    "dependence",
    "failure",
    "failed",
    "uncertain",
    "shortage",
    "unavailable",
    "outage",
    "escalation",
    "scope creep",
    "budget overrun",
    "cost overrun",
    "quality issue",
    "security concern",
    "not on track",
    "may miss",
    "might miss",
    "at risk",
)

BLOCKER_KEYWORDS = (
    "blocked",
    "blocker",
    "blocking",
    "impediment",
    "cannot proceed",
    "can't proceed",
    "unable to proceed",
    "waiting for approval",
    "awaiting approval",
    "pending approval",
    "access denied",
    "environment unavailable",
    "dependency unresolved",
    "stuck",
)

ACTION_KEYWORDS = (
    "action:",
    "action item",
    "todo",
    "to-do",
    "task:",
    "follow up",
    "follow-up",
    "must complete",
    "need to",
    "needs to",
    "should complete",
    "will deliver",
    "responsible for",
    "assigned to",
)

DELIVERABLE_KEYWORDS = (
    "deliverable",
    "output",
    "milestone",
    "acceptance criteria",
    "release",
    "launch",
    "deployment",
    "prototype",
    "report",
    "design document",
    "test plan",
    "user guide",
    "documentation",
)

OBJECTIVE_KEYWORDS = (
    "objective",
    "goal",
    "purpose",
    "aim",
    "target",
    "business outcome",
    "project intends",
    "project will",
)


def _normalize_text(text):
    """Normalize input to a safe string."""
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    return text.replace("\x00", "").strip()


def _split_sentences(text):
    """Split text into sentence-like units while preserving useful context."""
    text = _normalize_text(text)

    if not text:
        return []

    # Split at newlines and common sentence-ending punctuation.
    parts = re.split(r"(?<=[.!?])\s+|\n+", text)

    return [
        re.sub(r"\s+", " ", part).strip(" \t-*•")
        for part in parts
        if part.strip(" \t-*•")
    ]


def _matches_any(text, keywords):
    lowered = text.lower()
    return [word for word in keywords if word in lowered]


def _extract_date(text):
    """Extract a date-like value; return None if absent or unparseable."""
    match = DATE_PATTERN.search(text)

    if not match:
        return None

    value = match.group(0).strip()

    formats = (
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%m-%d-%Y",
        "%d-%m-%Y",
        "%m/%d/%y",
        "%d/%m/%y",
        "%B %d, %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%b %d %Y",
        "%B %d",
        "%b %d",
    )

    for fmt in formats:
        try:
            parsed = datetime.strptime(value, fmt).date()

            # Dates without a year use the current year.
            if "%Y" not in fmt:
                parsed = parsed.replace(year=date.today().year)

            return parsed.isoformat()
        except ValueError:
            continue

    return value


def _extract_owner(text):
    match = OWNER_PATTERN.search(text)

    if not match:
        return None

    owner = match.group(1).strip(" .,:;-")
    return owner or None


def _severity(text, matched_keywords=None):
    """Estimate severity using explicit labels and risk wording."""
    lowered = text.lower()

    for level in ("critical", "high", "medium", "low"):
        if re.search(rf"\b{level}\b", lowered):
            return level

    if any(word in lowered for word in (
        "project-stopping",
        "production outage",
        "data loss",
        "security breach",
    )):
        return "critical"

    if any(word in lowered for word in (
        "urgent",
        "major",
        "severe",
        "miss the deadline",
        "miss deadline",
        "release blocked",
    )):
        return "high"

    if any(word in lowered for word in (
        "might miss",
        "may miss",
        "potential",
        "uncertain",
        "dependency",
    )):
        return "medium"

    return "medium"


def _unique_items(items, key_fields):
    """Deduplicate dictionaries while preserving their original order."""
    seen = set()
    output = []

    for item in items:
        key = tuple(item.get(field) for field in key_fields)

        if key in seen:
            continue

        seen.add(key)
        output.append(item)

    return output


class ScopeDeliverablesAgent:
    """Extract objectives, deliverables, milestones, and dates."""

    def analyze(self, text):
        sentences = _split_sentences(text)

        objectives = []
        deliverables = []

        for sentence in sentences:
            lowered = sentence.lower()

            if _matches_any(lowered, OBJECTIVE_KEYWORDS):
                objectives.append({
                    "description": sentence,
                    "source_text": sentence,
                })

            matched_deliverables = _matches_any(
                lowered,
                DELIVERABLE_KEYWORDS,
            )

            if matched_deliverables:
                deliverables.append({
                    "description": sentence,
                    "type": (
                        "milestone"
                        if "milestone" in matched_deliverables
                        else "deliverable"
                    ),
                    "due_date": _extract_date(sentence),
                    "keywords": matched_deliverables,
                })

        return {
            "objectives": _unique_items(
                objectives, ("description",)
            ),
            "deliverables": _unique_items(
                deliverables, ("description",)
            ),
            "objective_count": len(objectives),
            "deliverable_count": len(deliverables),
        }


class RiskDetectionAgent:
    """Identify possible risks using transparent keyword signals."""

    def analyze(self, text):
        risks = []

        for sentence in _split_sentences(text):
            matched = _matches_any(sentence, RISK_KEYWORDS)

            if not matched:
                continue

            risks.append({
                "description": sentence,
                "severity": _severity(sentence, matched),
                "signals": matched,
                "source_text": sentence,
                "due_date": _extract_date(sentence),
                "owner": _extract_owner(sentence),
            })

        risks = _unique_items(risks, ("description",))

        priority = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        risks.sort(
            key=lambda item: priority.get(
                item["severity"], 2
            )
        )

        return {
            "risks": risks,
            "risk_count": len(risks),
            "high_priority_count": sum(
                risk["severity"] in {"critical", "high"}
                for risk in risks
            ),
        }


class BlockerIdentificationAgent:
    """Identify sentences that may describe unresolved blockers."""

    def analyze(self, text):
        blockers = []

        for sentence in _split_sentences(text):
            matched = _matches_any(
                sentence,
                BLOCKER_KEYWORDS,
            )

            if not matched:
                continue

            blockers.append({
                "description": sentence,
                "severity": _severity(sentence, matched),
                "signals": matched,
                "owner": _extract_owner(sentence),
                "due_date": _extract_date(sentence),
                "status": "Unresolved signal; verify manually",
            })

        blockers = _unique_items(
            blockers, ("description",)
        )

        return {
            "blockers": blockers,
            "blocker_count": len(blockers),
        }


class ActionItemAgent:
    """Extract possible tasks, owners, dates, and status clues."""

    def analyze(self, text):
        action_items = []

        for sentence in _split_sentences(text):
            lowered = sentence.lower()
            matched = _matches_any(
                lowered,
                ACTION_KEYWORDS,
            )

            # Completed tasks should not automatically be listed
            # as open action items.
            completed = bool(re.search(
                r"\b(completed|finished|resolved|delivered|"
                r"closed|done)\b",
                lowered,
            ))

            if not matched and not re.match(
                r"^\s*[-*•]\s*\[[ xX]\]",
                sentence,
            ):
                continue

            action_items.append({
                "description": sentence,
                "owner": _extract_owner(sentence),
                "due_date": _extract_date(sentence),
                "status": (
                    "Completed"
                    if completed
                    else "Needs verification"
                ),
                "signals": matched,
            })

        action_items = _unique_items(
            action_items, ("description",)
        )

        return {
            "action_items": action_items,
            "action_item_count": len(action_items),
            "completed_count": sum(
                item["status"] == "Completed"
                for item in action_items
            ),
        }


class ProjectAnalysisAgents:
    """Unified interface for all four project-analysis agents."""

    def __init__(self):
        self.scope_agent = ScopeDeliverablesAgent()
        self.risk_agent = RiskDetectionAgent()
        self.blocker_agent = BlockerIdentificationAgent()
        self.action_agent = ActionItemAgent()

    def analyze(self, text):
        text = _normalize_text(text)

        if not text:
            return {
                "scope": {
                    "objectives": [],
                    "deliverables": [],
                    "objective_count": 0,
                    "deliverable_count": 0,
                },
                "risk_analysis": {
                    "risks": [],
                    "risk_count": 0,
                    "high_priority_count": 0,
                },
                "blocker_analysis": {
                    "blockers": [],
                    "blocker_count": 0,
                },
                "action_analysis": {
                    "action_items": [],
                    "action_item_count": 0,
                    "completed_count": 0,
                },
            }

        return {
            "scope": self.scope_agent.analyze(text),
            "risk_analysis": self.risk_agent.analyze(text),
            "blocker_analysis": self.blocker_agent.analyze(text),
            "action_analysis": self.action_agent.analyze(text),
        }
