# Test Report

Run locally with `pip install -r requirements-dev.txt && pytest -q`.

| Area | Tests | What is verified |
|---|---|---|
| Ingestion | formats, table classification, bad files | DOCX, XLSX, TXT, CSV parse; backlog/defect/test-plan sheets classified; corrupt, empty, unsupported and junk files give friendly errors |
| RAG | retrieval, self-test, fallback | relevant chunk ranked first; self-generated probes reach recall ≥ 80 %; keyword fallback if the vector index is unavailable |
| Agents | scope, risk, blockers, forecast, isolation | goals/deliverables/phases/owners; defect, slippage and dependency risks; blockers, decisions, actions with owners; forecast bounds; one failing agent never stops the others |
| Health | dimensions, empty state | three weighted dimensions summing to 1.0, score 0-100, recommendations |
| Documents | stories, register, exports, report | user-story format with ≥ 3 acceptance criteria; rated risk register; Excel and Word exports |
| Assistant | citations, refusal, empty KB, follow-ups, validation suite | every answer cites sources that exist; off-topic questions are refused; “empty knowledge base” guidance; follow-up resolved via context; answer-rate ≥ 75 % and 100 % off-topic refusal |
| Incremental upload | add file → everything updates | new document increases chunks and blockers and is answerable immediately |
| UI | all pages, empty and populated, uploader flow | every page renders; auto-ingestion via the uploader; chat round-trip with sources |
| Hygiene | legacy phase wording | no legacy phase label appears in any project file |

**Result in the authoring environment: 34 passed, 0 failed.** The same flows were also run end-to-end on the real
Agile template (.xls), defect tracker, unit test plan and brief documents (7 files): all three analysis agents succeeded,
retrieval self-test recall 100 %, project report generated.
