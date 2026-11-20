"""End-to-end tests: ingestion → RAG → agents → health → documents → assistant → report."""
import os
import re

import pytest

from core import docgen, report, validation
from core.agents import run_pipeline
from core.assistant import NOT_FOUND, answer_question
from core.health import compute_health
from core.ingestion import IngestionError, parse_file
from core.rag import KnowledgeBase


# ---------------------------------------------------------------- ingestion ----
@pytest.mark.parametrize("name", ["brief.docx", "tracker.xlsx", "notes.txt", "tasks.csv"])
def test_formats_parse(sample_files, name):
    doc = parse_file(name, sample_files[name])
    assert doc.segments and doc.char_count > 50


def test_tables_are_classified(sample_files):
    doc = parse_file("tracker.xlsx", sample_files["tracker.xlsx"])
    kinds = {t.sheet: t.kind for t in doc.tables}
    assert kinds == {"Backlog": "backlog", "Defects": "defects", "UT": "tests"}


@pytest.mark.parametrize("name,data", [("a.pdf", b"%PDF-bad"), ("b.docx", b"nope"), ("c.txt", b""), ("d.exe", b"x"), ("e.xlsx", b"junk")])
def test_bad_files_raise_friendly_errors(name, data):
    with pytest.raises(IngestionError):
        parse_file(name, data)


# --------------------------------------------------------------------- RAG ----
def test_retrieval_finds_relevant_chunk(kb):
    hits = kb.search("payment gateway credentials vendor")
    assert hits and hits[0].chunk.doc_name == "notes.txt"


def test_retrieval_self_test_recall(kb):
    res = validation.retrieval_validation(kb)
    assert res["n"] > 0 and res["recall"] >= 0.8


def test_index_fallback_to_keywords(kb):
    kb._vectorizer = kb._matrix = None
    assert kb.search("payment gateway vendor")


# ------------------------------------------------------------------ agents ----
def test_all_agents_succeed(analysis):
    pipe, _ = analysis
    assert [r.status for r in pipe.runs] == ["success"] * 3


def test_scope_extraction(analysis):
    s = analysis[0].scope
    assert s.goals and s.deliverables and len(s.timeline) >= 2
    assert {r.owner for r in s.responsibilities} >= {"Ravi", "Asha"}


def test_risk_detection_covers_trackers_and_notes(analysis):
    titles = " | ".join(r.title for r in analysis[0].risk.risks)
    assert "Open defect" in titles and "slipped" in titles.lower() and "single point" in titles.lower()
    assert any(r.category == "Dependency" for r in analysis[0].risk.risks)


def test_blockers_decisions_actions(analysis):
    b = analysis[0].blockers
    assert any("gateway" in x.text.lower() for x in b.blockers)
    assert any("PostgreSQL" in d.text for d in b.decisions)
    assert any(a.owner == "Ravi" for a in b.actions)


def test_forecast_is_bounded(analysis):
    assert 5 <= analysis[0].risk.forecast.on_time_pct <= 97


def test_agents_isolated_from_failure(kb, monkeypatch):
    import core.agents as agents
    monkeypatch.setattr(agents, "run_risk_agent", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    pipe = agents.run_pipeline(kb)
    status = {r.name: r.status for r in pipe.runs}
    assert "error" in status.values() and list(status.values()).count("success") == 2


# ------------------------------------------------------------------ health ----
def test_health_dimensions(analysis):
    _, h = analysis
    assert h.has_data and 0 <= h.overall <= 100
    assert [d.key for d in h.dimensions] == ["scope", "timeline", "blockers"]
    assert abs(sum(d.weight for d in h.dimensions) - 1) < 1e-9 and h.recommendations


def test_health_empty():
    assert not compute_health(KnowledgeBase(), run_pipeline(KnowledgeBase())).has_data


# ----------------------------------------------------------------- docgen ----
def test_generated_documents(analysis):
    pipe, h = analysis
    stories = docgen.user_stories(pipe)
    assert stories and all(re.match(r"^(As an? .+, I want .+|As an? .+)", s.story) and len(s.acceptance) >= 3 for s in stories)
    rr, ar = docgen.risk_register(pipe), docgen.action_register(pipe)
    assert not rr.empty and set(rr["Rating"]) <= {"Critical", "High", "Medium", "Low"} and not ar.empty
    assert docgen.to_excel_bytes({"R": rr, "A": ar})[:2] == b"PK"


def test_report_exports(kb, analysis):
    pipe, h = analysis
    stories = docgen.user_stories(pipe)
    assert report.build_docx(kb, pipe, h, stories)[:2] == b"PK"
    assert "## 4. Risk register" in report.build_markdown(kb, pipe, h, stories)


# --------------------------------------------------------------- assistant ----
@pytest.mark.parametrize("q", ["Are we on track?", "What are our biggest risks?", "Who owns which work?", "What are the open blockers?"])
def test_answers_are_cited(kb, analysis, q):
    pipe, h = analysis
    a = answer_question(kb, pipe, h, q)
    assert a.grounded and a.sources and re.search(r"\[\d+\]", a.text)
    assert {int(n) for n in re.findall(r"\[(\d+)\]", a.text)} <= {s.n for s in a.sources}


def test_off_topic_is_refused(kb, analysis):
    pipe, h = analysis
    a = answer_question(kb, pipe, h, "What is the capital of France?")
    assert not a.grounded and a.text == NOT_FOUND


def test_empty_knowledge_base_message(analysis):
    pipe, h = analysis
    assert "empty" in answer_question(KnowledgeBase(), pipe, h, "Are we on track?").text.lower()


def test_follow_up_uses_context(kb, analysis):
    pipe, h = analysis
    first = answer_question(kb, pipe, h, "What are our biggest risks?")
    follow = answer_question(kb, pipe, h, "and the owners?", [{"role": "user", "content": "What are our biggest risks?"}, {"role": "assistant", "content": first.text}])
    assert follow.intent == "owners"


def test_assistant_validation_suite(kb, analysis):
    pipe, h = analysis
    v = validation.assistant_validation(kb, pipe, h)
    assert v["answer_rate"] >= 0.75 and v["refusal_accuracy"] == 1.0


# ------------------------------------------------------------- incremental ----
def test_incremental_upload_updates_everything(kb, sample_files):
    from core.ingestion import parse_file
    before = len(kb.chunks), len(run_pipeline(kb).blockers.blockers)
    kb.add_document(parse_file("update.txt", b"Blocked: the staging server is down and the QA team is stuck.\nAction: Meera to restore staging by 25 Nov 2026."))
    kb.build_index()
    pipe = run_pipeline(kb)
    assert len(kb.chunks) > before[0] and len(pipe.blockers.blockers) > before[1]
    assert answer_question(kb, pipe, compute_health(kb, pipe), "Is the staging server a problem?").grounded


# ---------------------------------------------------------------- hygiene ----
def test_no_legacy_phase_label_in_project_files():
    legacy = re.compile("mile" + "stone", re.I)
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    offenders = []
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in {".git", "__pycache__", ".pytest_cache", ".venv"}]
        for f in files:
            if f.endswith((".py", ".md", ".toml", ".txt")):
                path = os.path.join(base, f)
                if legacy.search(open(path, encoding="utf-8").read()):
                    offenders.append(os.path.relpath(path, root))
    assert not offenders, offenders
