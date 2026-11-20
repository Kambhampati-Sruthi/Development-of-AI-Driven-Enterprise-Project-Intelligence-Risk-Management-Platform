"""Central, fixed configuration.

The RAG settings are intentionally constants rather than UI controls: they are
tuned defaults for project artifacts (briefs, notes, trackers, test plans).
"""
from __future__ import annotations

APP_NAME = "AI Project Intelligence & Risk Advisor"
APP_TAGLINE = "Turn scattered project artifacts into a living, queryable intelligence layer."

# ---- Ingestion ----------------------------------------------------------------
SUPPORTED_EXTENSIONS = ("pdf", "docx", "csv", "xlsx", "xls", "txt", "md")
MAX_FILE_MB = 25
MAX_TABLE_ROWS = 5000          # safety cap per sheet

# ---- RAG (fixed, optimised defaults) -------------------------------------------
CHUNK_SIZE = 900               # characters per chunk
CHUNK_OVERLAP = 150            # characters shared between neighbouring chunks
TOP_K = 6                      # chunks retrieved per question
MIN_RELEVANCE = 0.07           # below this cosine score a chunk is not evidence
MAX_ANSWER_SENTENCES = 5

# ---- Health scoring weights (sum to 1.0) ---------------------------------------
HEALTH_WEIGHTS = {"scope": 0.35, "timeline": 0.35, "blockers": 0.30}
HEALTH_BANDS = (            # (minimum score, label, colour)
    (80, "Healthy", "#16a34a"),
    (60, "Watch", "#d97706"),
    (40, "At Risk", "#ea580c"),
    (0, "Critical", "#dc2626"),
)

# ---- Optional LLM (answer synthesis only; the app works fully without it) -------
LLM_MODEL_ENV = "PROJECT_ADVISOR_LLM_MODEL"
LLM_DEFAULT_MODEL = "claude-sonnet-4-6"
LLM_KEY_ENV = "ANTHROPIC_API_KEY"
