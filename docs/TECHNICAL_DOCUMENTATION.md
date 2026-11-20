# Technical Documentation — AI Project Intelligence & Risk Advisor

## 1. Purpose
Turn scattered project artifacts (proposals, SRS, meeting notes, sprint backlogs, defect trackers, test plans, progress
updates) into a unified, queryable knowledge base, and derive scope, risks, blockers, documentation and a project health
score from it. A conversational assistant answers questions grounded only in the uploaded documents.

## 2. System architecture

```
 Upload (PDF · DOCX · CSV · XLS/XLSX · TXT/MD)         incremental: files are ingested the moment they are added
        │
        ▼
 core/ingestion.py ──► ParsedDocument { segments[], tables[] }       (typed tables: backlog · defects · tests · generic)
        │
        ▼
 core/rag.py ─► chunker (900 chars, 150 overlap; table rows grouped) ─► TF-IDF vector index (uni+bi-grams, stemming)
        │                                                              cosine retrieval · sentence ranking
        ▼
 core/agents.py  (orchestrator; each agent isolated)
   1 Scope & Deliverable Extraction      → goals, deliverables, timeline phases, dated events, responsibilities
   2 Risk Detection & Delivery Forecast  → scored risks (Detected/Inferred), dependency gaps, on-time forecast
   3 Blocker & Action Identification     → blockers, pending decisions, action items (owner · due · priority)
        │
        ├─► core/docgen.py  (4 Documentation Generation Agent) user stories · risk register · action register · exports
        ├─► core/health.py  health score: scope clarity · timeline risk · blocker load + recommendations
        ├─► core/report.py  executive project report (DOCX / Markdown)
        └─► core/assistant.py  intent-aware, citation-first RAG Q&A (+ optional LLM synthesis)
                 core/validation.py  retrieval self-test · assistant grounding suite · per-format agent coverage
        ▼
 ui/ (Streamlit): sidebar navigation → Dashboard · AI Assistant ★ · Knowledge Base · Agent Center · Generated Docs · Project Health
```

## 3. Data model (`core/models.py`, `core/ingestion.py`)
| Object | Key fields |
|---|---|
| `ParsedDocument` | name, ext, segments (text + locator), tables, warnings |
| `TableData` | sheet, DataFrame, kind, `roles` (semantic column map: title, owner, status, sprint, due, expected, actual …) |
| `Chunk` | chunk_id, doc_name, locator, text |
| `Risk` | id, title, category, severity, probability (1-5), impact (1-5), evidence_type, doc, locator, owner |
| `Blocker` / `Action` | text, status, severity/priority, owner, due, source |
| `Forecast` | on_time_pct, level, confidence, narrative, drivers, key_dates |

## 4. RAG pipeline
* **Chunking** — paragraphs/sentences packed to ≤900 characters with 150 overlap; table rows grouped by sheet; column
  names are repeated in each row so a chunk is self-describing.
* **Indexing** — scikit-learn `TfidfVectorizer` with a custom analyzer (stop-word removal, light stemming, bigrams).
  Deterministic, offline, no model download. If indexing fails the system degrades to keyword overlap scoring.
* **Retrieval** — cosine similarity, top-6, relevance floor 0.07, near-duplicate suppression (repeated documents), light
  domain query expansion (e.g. *fail → error, mismatch, issue*).
* **Answering** — intent detection (status, risks, blockers, decisions, actions, owners, timeline, scope, general);
  structured intents are built from agent findings that carry citations, then supplemented with the best-matching
  sentences from retrieved chunks. Follow-up questions are resolved with the previous user question. Citations are
  renumbered contiguously; if nothing relevant is found the assistant says so instead of guessing.
* **Optional LLM** — with `ANTHROPIC_API_KEY` set, an LLM phrases the answer using only the numbered excerpts; output
  without citations is discarded in favour of the extractive answer.

## 5. Agent logic (summary)
* **Scope** — section-aware (outcome/objective/module headings), goal cue phrases, timeline headings containing
  week/sprint/phase patterns, tracker rows grouped by sprint, owners from owner columns and "Owner:/Action: X will" patterns.
* **Risk** — category lexicons (Schedule, Dependency, Resource, Technical, Quality, Scope, Governance) applied to status
  text only (plan descriptions are filtered); tracker rules: open defects, unresolved test findings (expected vs actual),
  blocked/delayed/overdue items, sprint slippage (planned ≠ actual), unmet dependencies; structural inference (no dates,
  no owners, dense phases, cascading phases, no quality evidence). Severity → probability × impact; rating bands
  Low <6, Medium 6-11, High 12-19, Critical ≥20.
* **Forecast** — `on_time = 85 − Σ penalties (High 6 / Medium 3 / Low 1, capped 45) + (done/total − 0.5)·10`, clamped 5-97 and
  capped at 72 when confidence is Low (confidence = evidence breadth: tracker data, dates, quality data).
* **Blockers & actions** — blocker/decision/action cue patterns, tracker rules (open defects, failing tests, blocked
  stories, stand-up impediments, "start doing" retrospective items), resolved-state detection, de-duplication.

## 6. Project health score (`core/health.py`)
`health = 0.35·scope + 0.35·timeline + 0.30·blockers`
* **Scope clarity** — goals (25) + deliverables (25) + timeline & dates (20) + named owners (20) + evidence breadth (10) − scope/governance risks (≤15).
* **Timeline risk** — `0.6·(100 − schedule/dependency penalties) + 0.4·on-time forecast`.
* **Blocker load** — `100 − Σ(open blockers: High 12 / Medium 7 / Low 3) − min(15, 3·pending decisions)`.
* Bands: Healthy ≥80 · Watch 60-79 · At Risk 40-59 · Critical <40.

## 7. Incremental upload
Dropping files into the uploader triggers ingestion immediately; same-name files replace their predecessor. The index is
rebuilt, all agents re-run, documents and health are regenerated, and a before/after summary of the change is shown.

## 8. Error handling
Per file (friendly `IngestionError`), index build (keyword fallback), each agent (isolated, other agents still run), each
page (error boundary), each assistant answer (never shows a stack trace), legacy `.xls` (xlrd → LibreOffice → HTML/CSV sniff).

## 9. Configuration (`core/config.py`)
Chunk size/overlap, top-k, relevance floor, health weights/bands, supported formats and LLM settings are fixed constants.

## 10. Extending
Add a risk category in `RISK_CATEGORIES`; add column synonyms in `_ROLE_SYNONYMS` (ingestion); add an assistant intent in
`INTENTS` + `_structured`; add a document type in `core/docgen.py`.
