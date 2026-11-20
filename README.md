# 🧭 AI Project Intelligence & Risk Advisor

🚀 **Live Application:** [https://projectintelligence.streamlit.app/](https://projectintelligence.streamlit.app/)

An enterprise-grade, multi-agent Streamlit platform that transforms scattered, unstructured project artifacts—project briefs, Agile templates, sprint backlogs, defect trackers, unit test plans, and meeting notes—into a centralized, living, and queryable intelligence layer.

---

## 📌 Table of Contents

1. [Problem Statement & Solution](#-problem-statement--solution)
2. [Key Features & Modules](#-key-features--modules)
3. [End-to-End System Architecture](#-end-to-end-system-architecture)
4. [How the Intelligence Pipeline Works](#-how-the-intelligence-pipeline-works)
5. [Project Structure](#-project-structure)
6. [Quick Start](#-quick-start)
7. [Optional LLM Answer Synthesis](#-optional-llm-answer-synthesis)
8. [Testing & Validation](#-testing--validation)
9. [Troubleshooting](#-troubleshooting)

---

## 🎯 Problem Statement & Solution

**The Problem:**  
In enterprise software delivery and project management, critical project knowledge is fragmented across PDFs, Word documents, CSV exports, and Excel workbooks. Manually cross-referencing sprint backlogs against defect trackers, spotting scope creep, identifying unowned tasks, and compiling executive status reports is slow, reactive, and error-prone.

**The Solution:**  
The **AI Project Intelligence & Risk Advisor** automates document ingestion, builds a deterministic Retrieval-Augmented Generation (RAG) knowledge base, and deploys four specialized analytical agents to continuously extract risks, blockers, scope gaps, and project health metrics—complete with strict source citations and exportable executive deliverables.

---

## ✨ Key Features & Modules

Navigation is powered by a persistent left sidebar (the page you are on is kept across uploads and actions). All pages are available from the first load and show helpful empty states until documents are indexed.

| Tab | Purpose |
| :--- | :--- |
| **Dashboard** | Ask-the-assistant bar, KPI cards, executive summary, health gauge, risk landscape (category chart + probability × impact heat map), top risks, blockers/decisions, timeline, ownership |
| **Knowledge Base** | Multi-format *incremental* upload (`PDF`, `DOCX`, `CSV`, `XLS/XLSX`, `TXT/MD`), update summaries, ingestion log, per-document stats, remove/clear, retrieval explorer |
| **Agent Center** | Four agents (**Scope**, **Risk & Forecast**, **Blocker & Action**, and **Orchestrator**) with status/timing and full outputs, plus a **Validation** view (per-format coverage, retrieval self-test) |
| **Generated Docs** | User stories, scored risk register, action items, **Word/Markdown project report**; export to Excel / CSV |
| **Project Health** | Overall score, dimension breakdown (scope clarity, timeline risk, blocker load), recommendations, forecast drivers |
| **AI Assistant ★** | Featured. Grounded RAG chat with numbered citations, evidence panel with relevance, follow-up handling, suggested follow-ups, feedback, transcript export, built-in accuracy validation; says “not found” instead of guessing |

---

## 🏗️ End-to-End System Architecture

```text
┌────────────────────────────────────────────────────────────────────────────┐
│                         Multi-Format Project Artifacts                     │
│        (PDF · DOCX · CSV · XLS/XLSX · TXT/MD — Briefs, Backlogs, Logs)     │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 1. INGESTION & TRACKER PARSING (core/ingestion.py, core/nlp.py)            │
│    • Extracts text segments & auto-detects table headers / column roles    │
│    • Natively interprets Sprint Backlogs, Defect Trackers & Test Plans    │
└─────────────────────────────────────┬──────────────────────────────────────┘
                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 2. DETERMINISTIC TF-IDF RAG INDEX (core/rag.py, core/config.py)            │
│    • 900-char chunks (150-char overlap) + unigram/bigram TF-IDF vectors    │
│    • Light stemming, domain query expansion & cosine similarity ranking    │
└───────────────────┬─────────────────────────────────┬──────────────────────┘
                    ▼                                 ▼
┌──────────────────────────────────────┐ ┌───────────────────────────────────┐
│ 3. MULTI-AGENT ENGINE (core/agents)  │ │ 4. GROUNDED AI ASSISTANT          │
│    • Scope Agent                     │ │    (core/assistant.py)            │
│    • Risk & Forecast Agent           │ │    • Intent-aware citation Q&A    │
│    • Blocker & Action Agent          │ │    • Optional Claude synthesis    │
│    • Central Orchestrator            │ │    • Strict "not found" guardrail │
└───────────────────┬──────────────────┘ └─────────────────┬─────────────────┘
                    ▼                                      ▼
┌────────────────────────────────────────────────────────────────────────────┐
│ 5. DELIVERABLES, HEALTH & STREAMLIT UI (docgen, health, report, ui/)       │
│    • Weighted Health Scoring · Heat Maps · DOCX/XLSX/CSV Exports           │
└────────────────────────────────────────────────────────────────────────────┘
⚙️ How the Intelligence Pipeline Works
RAG uses a deterministic TF-IDF vector space (word uni+bi-grams, light stemming, domain query expansion). It runs fully offline with no model download. Chunk size, overlap, and top-k are fixed constants in core/config.py.

Agents are evidence-based. Every finding carries its document and location. Findings derived from structural gaps (e.g., no owners, dense phases) are labelled Inferred; those stated in documents are Detected.

Tracker awareness. Sprint backlogs (planned vs. actual sprint, dependencies, status), defect trackers (open/closed, owner, remarks), unit test plans (expected vs. actual results), and stand-up/retrospective sheets are interpreted natively.

Resilience. Each file, the index build, each agent, each view, and each assistant answer is isolated by exception handling; one failure never takes down the rest. Vector-index failure degrades to keyword retrieval.

📁 Project Structure
app.py                  Streamlit entry point (sidebar navigation, error boundaries)
core/
  config.py             Fixed RAG defaults, scoring weights, optional LLM settings
  ingestion.py          Parsers (PDF, DOCX, CSV, Excel, TXT) → Segments + typed tables (header detection, column roles)
  rag.py                Chunking (900 chars / 150 overlap), TF-IDF vector index, cosine retrieval, sentence ranking
  nlp.py                Shared text/table primitives (sections, bullets, status normalisation)
  agents.py             Scope · Risk & Forecast · Blocker & Action agents + orchestrator
  docgen.py             User stories, risk register, action register, exports
  health.py             Weighted health score and recommendations
  assistant.py          Intent-aware, citation-first Q&A with follow-up handling (optional LLM synthesis)
  report.py             Executive project report (DOCX / Markdown)
  validation.py         Retrieval self-test, assistant grounding suite, per-format agent coverage
  models.py, utils.py   Typed results and helpers
ui/
  styles.py             Global theme/CSS      components.py   Cards, KPIs, gauge, heat map, tables
  state.py              Session, incremental ingestion, callbacks   views.py   One renderer per page   nav.py  Page names
tests/                  End-to-end + UI smoke tests (pytest)
docs/                   TECHNICAL_DOCUMENTATION.md · DEMO_SCRIPT.md · TEST_REPORT.md
.streamlit/config.toml  Theme and server settings
requirements.txt        Core production dependencies
requirements-dev.txt    Development & testing dependencies (pytest)

🚀 Quick Start
Option 1: Try the Live App
Open https://projectintelligence.streamlit.app/ in your browser.

Go to Knowledge Base in the left sidebar and drop in your files.

They are indexed automatically and every agent runs; add more files at any time and the knowledge base updates incrementally.

Option 2: Run Locally
git clone [https://github.com/Kambhampati-Sruthi/Development-of-AI-Driven-Enterprise-Project-Intelligence-Risk-Management-Platform.git](https://github.com/Kambhampati-Sruthi/Development-of-AI-Driven-Enterprise-Project-Intelligence-Risk-Management-Platform.git)
cd Development-of-AI-Driven-Enterprise-Project-Intelligence-Risk-Management-Platform

python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
Open the URL Streamlit prints (usually http://localhost:8501), go to Knowledge Base and drop in your files. They are indexed automatically and every agent runs; add more files at any time and the knowledge base updates incrementally.

pip install -r requirements-dev.txt && pytest -q  # end-to-end tests

🧪 Testing & Validation
pip install -r requirements-dev.txt && pytest -q

🛠️ Troubleshooting
Scanned PDF → “No text layer found”: run OCR first.

Nothing retrieved → use the Retrieval explorer in Knowledge Base to see what matches a query.

Re-analyse after editing files: upload the new version (same name replaces the old one) or press Re-run analysis.
```text