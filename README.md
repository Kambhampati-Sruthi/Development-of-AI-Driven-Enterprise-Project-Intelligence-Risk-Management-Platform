# 🧭 AI Project Intelligence & Risk Advisor

An enterprise-style Streamlit platform that turns scattered project artifacts — briefs, Agile templates and sprint
backlogs, defect trackers, unit test plans, meeting notes — into a living, queryable intelligence layer.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Open the URL Streamlit prints (usually http://localhost:8501), go to **Knowledge Base** and drop in your files. They are
indexed automatically and every agent runs; add more files at any time and the knowledge base updates incrementally.

```bash
pip install -r requirements-dev.txt && pytest -q      # end-to-end tests
```

> Legacy `.xls` files are read with `xlrd` (included in `requirements.txt`). If it is unavailable the app falls back to
> LibreOffice conversion (if installed), then to HTML/CSV sniffing, and finally reports a clear error.

## What you get

| Tab | Purpose |
|---|---|
| **Dashboard** | Ask-the-assistant bar, KPI cards, executive summary, health gauge, risk landscape (category chart + probability×impact heat map), top risks, blockers/decisions, timeline, ownership |
| **Knowledge Base** | Multi-format **incremental** upload (PDF, DOCX, CSV, XLS/XLSX, TXT/MD), update summaries, ingestion log, per-document stats, remove/clear, retrieval explorer |
| **Agent Center** | Four agents with status/timing and full outputs, plus a **Validation** view (per-format coverage, retrieval self-test) |
| **Generated Docs** | User stories, scored risk register, action items, **Word/Markdown project report**; export to Excel / CSV |
| **Project Health** | Overall score, dimension breakdown (scope clarity, timeline risk, blocker load), recommendations, forecast drivers |
| **AI Assistant ★** | Featured. Grounded RAG chat with numbered citations, evidence panel with relevance, follow-up handling, suggested follow-ups, feedback, transcript export, built-in accuracy validation; says “not found” instead of guessing |

Navigation is a persistent left sidebar (the page you are on is kept across uploads and actions). All pages are available from the first load and show helpful empty states until documents are indexed.

## Architecture

```
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
```

### Design notes

* **RAG** uses a deterministic TF-IDF vector space (word uni+bi-grams, light stemming, domain query expansion). It runs
  fully offline with no model download. Chunk size, overlap and top-k are fixed constants in `core/config.py`.
* **Agents are evidence-based.** Every finding carries its document and location. Findings derived from structural gaps
  (e.g. no owners, dense phases) are labelled **Inferred**; those stated in documents are **Detected**.
* **Tracker awareness.** Sprint backlogs (planned vs actual sprint, dependencies, status), defect trackers (open/closed,
  owner, remarks), unit test plans (expected vs actual results) and stand-up/retrospective sheets are interpreted natively.
* **Resilience.** Each file, the index build, each agent, each view and each assistant answer is isolated by exception
  handling; one failure never takes down the rest. Vector-index failure degrades to keyword retrieval.

### Optional LLM answer synthesis

The assistant is fully functional without any API key. To let an LLM phrase answers (still restricted to the retrieved,
numbered excerpts):

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key          # or add it to .streamlit/secrets.toml
```

## Troubleshooting

* **Scanned PDF** → “No text layer found”: run OCR first.
* **Nothing retrieved** → use the *Retrieval explorer* in Knowledge Base to see what matches a query.
* **Re-analyse** after editing files: upload the new version (same name replaces the old one) or press **Re-run analysis**.
