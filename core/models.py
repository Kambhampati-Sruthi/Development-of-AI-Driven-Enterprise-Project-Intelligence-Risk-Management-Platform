"""Typed result objects exchanged between agents, scoring, documents and the UI."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Item:
    text: str
    doc: str
    locator: str
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Phase:
    label: str
    window: str
    tasks: List[str]
    doc: str
    locator: str
    hours: Optional[int] = None
    weeks: Optional[int] = None


@dataclass
class Responsibility:
    owner: str
    work: List[str]
    sources: List[str]


@dataclass
class ScopeResult:
    project_name: str = ""
    summary: str = ""
    goals: List[Item] = field(default_factory=list)
    deliverables: List[Item] = field(default_factory=list)
    timeline: List[Phase] = field(default_factory=list)
    dated_events: List[Item] = field(default_factory=list)
    responsibilities: List[Responsibility] = field(default_factory=list)


@dataclass
class Risk:
    risk_id: str
    title: str
    category: str
    severity: str             # High | Medium | Low
    probability: int          # 1-5
    impact: int               # 1-5
    evidence_type: str        # Detected | Inferred
    doc: str
    locator: str
    owner: str = ""
    detail: str = ""

    @property
    def score(self) -> int:
        return self.probability * self.impact


@dataclass
class Forecast:
    on_time_pct: float = 0.0
    level: str = "Not enough data"
    confidence: str = "Low"
    narrative: str = ""
    drivers: List[str] = field(default_factory=list)
    done: int = 0
    total: int = 0
    key_dates: List[str] = field(default_factory=list)


@dataclass
class RiskResult:
    risks: List[Risk] = field(default_factory=list)
    forecast: Forecast = field(default_factory=Forecast)

    @property
    def dependency_gaps(self) -> List[Risk]:
        return [r for r in self.risks if r.category == "Dependency"]


@dataclass
class Blocker:
    text: str
    status: str               # Open | Resolved
    severity: str
    owner: str
    doc: str
    locator: str
    resolution: str = ""


@dataclass
class Action:
    action_id: str
    text: str
    owner: str
    due: str
    priority: str
    kind: str
    status: str
    doc: str
    locator: str


@dataclass
class BlockerResult:
    blockers: List[Blocker] = field(default_factory=list)
    decisions: List[Item] = field(default_factory=list)
    actions: List[Action] = field(default_factory=list)


@dataclass
class AgentRun:
    name: str
    status: str               # success | error | skipped
    seconds: float
    count: int
    message: str = ""


@dataclass
class PipelineResult:
    scope: ScopeResult = field(default_factory=ScopeResult)
    risk: RiskResult = field(default_factory=RiskResult)
    blockers: BlockerResult = field(default_factory=BlockerResult)
    runs: List[AgentRun] = field(default_factory=list)
    generated_at: Optional[datetime] = None
