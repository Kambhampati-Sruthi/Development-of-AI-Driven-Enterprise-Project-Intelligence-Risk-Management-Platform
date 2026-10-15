
import csv
import io
from datetime import datetime, timezone


class DocumentationGenerationAgent:
    """Generate project documentation from the analysis-agent results."""

    def __init__(self):
        self.generated_at = datetime.now(
            timezone.utc
        ).isoformat(timespec="seconds")

    @staticmethod
    def _text(value, default="Not specified"):
        if value is None or not str(value).strip():
            return default
        return str(value).strip()

    @staticmethod
    def _csv_content(rows, fieldnames):
        """Convert dictionaries into downloadable CSV text."""
        buffer = io.StringIO()
        writer = csv.DictWriter(
            buffer,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
        return buffer.getvalue()

    def generate_user_stories(self, analysis):
        """Generate draft user stories from project deliverables."""
        scope = (analysis or {}).get("scope", {}) or {}
        deliverables = scope.get("deliverables", []) or []
        objectives = scope.get("objectives", []) or []

        stories = []

        for index, deliverable in enumerate(deliverables, start=1):
            description = self._text(
                deliverable.get("description"),
                "Project deliverable",
            )

            stories.append({
                "id": f"US-{index:03d}",
                "title": description,
                "user_story": (
                    f"As a project stakeholder, I want "
                    f"{description.rstrip('.')} "
                    f"so that the project objective can be achieved."
                ),
                "acceptance_criteria": (
                    "The deliverable is completed, reviewed, "
                    "and accepted against agreed project requirements."
                ),
                "due_date": self._text(
                    deliverable.get("due_date"),
                    "",
                ),
                "status": "Draft - requires stakeholder validation",
            })

        if not stories:
            for index, objective in enumerate(objectives, start=1):
                description = self._text(
                    objective.get("description"),
                    "Project objective",
                )

                stories.append({
                    "id": f"US-{index:03d}",
                    "title": description,
                    "user_story": (
                        f"As a project stakeholder, I want "
                        f"{description.rstrip('.')} "
                        f"so that the intended business outcome is achieved."
                    ),
                    "acceptance_criteria": (
                        "Define measurable acceptance criteria "
                        "with the project owner."
                    ),
                    "due_date": "",
                    "status": "Draft - requires stakeholder validation",
                })

        return stories

    def generate_risk_register(self, analysis):
        """Generate a risk register from detected risk signals."""
        risk_analysis = (
            (analysis or {}).get("risk_analysis", {}) or {}
        )
        risks = risk_analysis.get("risks", []) or []

        register = []

        for index, risk in enumerate(risks, start=1):
            description = self._text(
                risk.get("description"),
                "Risk description requires review",
            )

            signals = risk.get("signals", []) or []
            severity = self._text(
                risk.get("severity"),
                "medium",
            ).lower()

            register.append({
                "risk_id": f"R-{index:03d}",
                "risk_description": description,
                "severity": severity,
                "signals": ", ".join(map(str, signals)),
                "owner": self._text(risk.get("owner")),
                "due_date": self._text(
                    risk.get("due_date"),
                    "",
                ),
                "mitigation_action": (
                    "Confirm impact and likelihood, assign an owner, "
                    "and agree on a mitigation plan."
                ),
                "status": "Open - requires project-team validation",
            })

        return register

    def generate_action_items(self, analysis):
        """Generate a structured action-item list."""
        action_analysis = (
            (analysis or {}).get("action_analysis", {}) or {}
        )
        blocker_analysis = (
            (analysis or {}).get("blocker_analysis", {}) or {}
        )

        source_items = action_analysis.get(
            "action_items", []
        ) or []

        source_blockers = blocker_analysis.get(
            "blockers", []
        ) or []

        items = []

        for index, action in enumerate(source_items, start=1):
            items.append({
                "action_id": f"A-{index:03d}",
                "description": self._text(
                    action.get("description")
                ),
                "owner": self._text(action.get("owner")),
                "due_date": self._text(
                    action.get("due_date"),
                    "",
                ),
                "status": self._text(
                    action.get("status"),
                    "Needs verification",
                ),
                "source_type": "Detected action item",
            })

        next_index = len(items) + 1

        for blocker in source_blockers:
            description = self._text(
                blocker.get("description")
            )

            items.append({
                "action_id": f"A-{next_index:03d}",
                "description": (
                    f"Investigate and resolve blocker: {description}"
                ),
                "owner": self._text(blocker.get("owner")),
                "due_date": self._text(
                    blocker.get("due_date"),
                    "",
                ),
                "status": "Open - requires assignment",
                "source_type": "Blocker follow-up",
            })

            next_index += 1

        return items

    def generate_all(self, analysis):
        """Generate all three required project documentation artifacts."""
        analysis = analysis or {}

        user_stories = self.generate_user_stories(analysis)
        risk_register = self.generate_risk_register(analysis)
        action_items = self.generate_action_items(analysis)

        return {
            "generated_at": self.generated_at,
            "user_stories": user_stories,
            "risk_register": risk_register,
            "action_items": action_items,
            "csv": {
                "user_stories": self._csv_content(
                    user_stories,
                    [
                        "id",
                        "title",
                        "user_story",
                        "acceptance_criteria",
                        "due_date",
                        "status",
                    ],
                ),
                "risk_register": self._csv_content(
                    risk_register,
                    [
                        "risk_id",
                        "risk_description",
                        "severity",
                        "signals",
                        "owner",
                        "due_date",
                        "mitigation_action",
                        "status",
                    ],
                ),
                "action_items": self._csv_content(
                    action_items,
                    [
                        "action_id",
                        "description",
                        "owner",
                        "due_date",
                        "status",
                        "source_type",
                    ],
                ),
            },
        }
