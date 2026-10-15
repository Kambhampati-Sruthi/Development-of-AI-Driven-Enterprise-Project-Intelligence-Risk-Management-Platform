
from pathlib import Path
from datetime import datetime
import inspect
import json
import sys

import pandas as pd
import streamlit as st

from app.rag_pipeline import RAGPipeline


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI Project Intelligence & Risk Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PATHS AND DIRECTORY SETUP
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"


def ensure_directory(path: Path):
    """Create a directory, but never replace a conflicting file."""
    if path.exists() and not path.is_dir():
        raise NotADirectoryError(
            f"A file exists where a folder is required:\n{path}\n"
            "Rename or move the file before restarting the application."
        )

    path.mkdir(parents=True, exist_ok=True)


try:
    ensure_directory(DATA_DIR)
    ensure_directory(UPLOAD_DIR)
    ensure_directory(CHROMA_DIR)
except Exception as exc:
    st.error("Could not initialize the project folders.")
    st.code(f"{type(exc).__name__}: {exc}")
    st.info(
        "Check the paths above in File Explorer. Do not delete your "
        "existing ChromaDB data to resolve a folder conflict."
    )
    st.stop()


# =========================================================
# 3. STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            145deg, #08111f 0%, #0b1526 50%, #111c31 100%
        );
        color: #edf3ff;
    }

    [data-testid="stSidebar"] {
        background: #0b1220;
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    .hero {
        padding: 30px;
        border-radius: 22px;
        background: linear-gradient(
            120deg,
            rgba(37, 99, 235, 0.27),
            rgba(124, 58, 237, 0.22),
            rgba(8, 145, 178, 0.16)
        );
        border: 1px solid rgba(147, 197, 253, 0.25);
        margin-bottom: 24px;
    }

    .hero h1 {
        font-size: 2.15rem;
        color: #f8fafc;
        margin-bottom: 8px;
    }

    .hero p {
        color: #cbd5e1;
        margin-bottom: 0;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 2px;
        color: #7dd3fc;
        font-size: 0.75rem;
        font-weight: 700;
    }

    .section-card {
        padding: 20px;
        border-radius: 16px;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(148, 163, 184, 0.17);
        margin-bottom: 14px;
    }

    .section-card h3 {
        color: #f1f5f9;
        margin-top: 0;
    }

    .section-card p {
        color: #cbd5e1;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 15px;
        border-radius: 15px;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "search_history": [],
    "last_answer": None,
    "last_query": "",
    "risk_findings": None,
    "risk_totals": {},
}

for key, default_value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# =========================================================
# 5. PIPELINE INITIALIZATION
# =========================================================

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """
    Pass only constructor arguments supported by the installed
    RAGPipeline class.

    If the constructor has required arguments with different names,
    inspect app/rag_pipeline.py and add their names to the mapping.
    """
    signature = inspect.signature(RAGPipeline)
    parameters = signature.parameters

    argument_values = {
        "upload_dir": str(UPLOAD_DIR),
        "upload_path": str(UPLOAD_DIR),
        "uploads_dir": str(UPLOAD_DIR),
        "chroma_dir": str(CHROMA_DIR),
        "persist_directory": str(CHROMA_DIR),
        "persist_dir": str(CHROMA_DIR),
        "db_path": str(CHROMA_DIR),
        "persist_path": str(CHROMA_DIR),
        "collection_path": str(CHROMA_DIR),
    }

    accepts_extra_keywords = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in parameters.values()
    )

    if accepts_extra_keywords:
        return RAGPipeline(
            upload_dir=str(UPLOAD_DIR),
            chroma_dir=str(CHROMA_DIR),
        )

    kwargs = {
        name: argument_values[name]
        for name in parameters
        if name in argument_values
    }

    # Identify required constructor arguments we cannot supply.
    missing = [
        name
        for name, parameter in parameters.items()
        if name not in kwargs
        and parameter.default is inspect.Parameter.empty
        and parameter.kind
        not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        )
    ]

    if missing:
        raise TypeError(
            "RAGPipeline requires constructor argument(s) that this "
            "app cannot identify automatically: "
            + ", ".join(missing)
            + ". Check app/rag_pipeline.py."
        )

    return RAGPipeline(**kwargs)


def get_pipeline_safely():
    try:
        return get_pipeline()
    except Exception as exc:
        st.error("The AI search pipeline could not be initialized.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.info(
            "The exact constructor and its defaults are defined in "
            "app/rag_pipeline.py. If required arguments are reported, "
            "match them to that file."
        )
        return None


# =========================================================
# 6. GENERAL UTILITIES
# =========================================================

def list_uploaded_files():
    try:
        return sorted(
            [p for p in UPLOAD_DIR.iterdir() if p.is_file()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []


def human_size(size_bytes):
    size = float(size_bytes)

    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} TB"


def unique_upload_path(filename):
    """Avoid overwriting an existing upload with the same filename."""
    safe_name = Path(filename).name

    if not safe_name or safe_name in {".", ".."}:
        raise ValueError("Invalid uploaded filename.")

    destination = UPLOAD_DIR / safe_name

    if not destination.exists():
        return destination

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return destination.with_name(
        f"{destination.stem}_{timestamp}{destination.suffix}"
    )


def normalize_result(result):
    if result is None:
        return "No result was returned."

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ("answer", "response", "result", "text"):
            if result.get(key) is not None:
                return str(result[key])

        return json.dumps(result, indent=2, default=str)

    if isinstance(result, list):
        return "\n\n".join(str(item) for item in result)

    return str(result)


def call_with_supported_arguments(method, query, top_k=None):
    """
    Call a pipeline method using recognized argument names.
    """
    signature = inspect.signature(method)
    params = signature.parameters

    kwargs = {}

    query_names = (
        "query", "question", "text", "user_query", "search_query"
    )
    top_k_names = ("top_k", "k", "n_results", "num_results")

    query_parameter = next(
        (name for name in query_names if name in params),
        None,
    )

    top_k_parameter = next(
        (name for name in top_k_names if name in params),
        None,
    )

    if query_parameter:
        kwargs[query_parameter] = query
    elif any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in params.values()
    ):
        kwargs["query"] = query
    else:
        # Use the first positional argument if the method has one.
        positional = [
            p for p in params.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]
        if not positional:
            raise TypeError(
                f"Cannot identify the query argument for {method.__name__}."
            )
        kwargs[positional[0].name] = query

    if top_k is not None and top_k_parameter:
        kwargs[top_k_parameter] = top_k

    return method(**kwargs)


def run_semantic_search(pipeline, query, top_k):
    method_names = (
        "query",
        "search",
        "ask",
        "query_documents",
        "search_documents",
        "search_similar",
    )

    for name in method_names:
        method = getattr(pipeline, name, None)

        if callable(method):
            return call_with_supported_arguments(
                method, query, top_k
            )

    raise AttributeError(
        "No supported search method was found in RAGPipeline. "
        "Check the method names in app/rag_pipeline.py."
    )


def run_document_ingestion(pipeline, file_path):
    method_names = (
        "ingest_file",
        "add_document",
        "process_file",
        "ingest_document",
        "index_file",
        "add_file",
        "ingest",
        "add_documents",
    )

    for name in method_names:
        method = getattr(pipeline, name, None)

        if not callable(method):
            continue

        signature = inspect.signature(method)
        params = signature.parameters

        path_names = (
            "file_path", "path", "filepath", "document_path",
            "source_path", "file",
        )

        path_parameter = next(
            (key for key in path_names if key in params),
            None,
        )

        if path_parameter:
            return method(**{path_parameter: str(file_path)})

        positional = [
            p for p in params.values()
            if p.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if positional:
            return method(str(file_path))

    raise AttributeError(
        "No supported document-ingestion method was found in "
        "RAGPipeline. Check app/rag_pipeline.py."
    )


# =========================================================
# 7. SIDEBAR AND HEADER
# =========================================================

with st.sidebar:
    st.markdown("## 🧠 Project Intelligence")
    st.caption("AI-powered project knowledge and risk analysis")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Document Center",
            "Semantic Search",
            "Risk Signals",
            "System Diagnostics",
        ],
    )

    st.divider()
    st.metric("Stored documents", len(list_uploaded_files()))
    st.caption("Local embeddings • ChromaDB • Streamlit")
    st.divider()
    st.caption("Infosys Springboard Internship Project")

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Enterprise intelligence workspace</div>
        <h1>AI Project Intelligence &amp; Risk Advisor</h1>
        <p>
            Transform project documents into searchable knowledge,
            discover risk signals, and support better decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 8. OVERVIEW
# =========================================================

if page == "Overview":
    files = list_uploaded_files()
    total_size = sum(p.stat().st_size for p in files)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📄 Documents", len(files))
    c2.metric("💾 Stored data", human_size(total_size))
    c3.metric("🔎 Searches", len(st.session_state.search_history))
    c4.metric("⚠️ Risk analysis", "Keyword-based")

    left, right = st.columns([1.3, 1])

    with left:
        st.markdown(
            """
            <div class="section-card">
                <h3>🚀 Platform capabilities</h3>
                <p><b>Document intelligence</b><br>
                Upload project files for local indexing.</p>
                <p><b>Semantic search</b><br>
                Retrieve relevant project knowledge by meaning.</p>
                <p><b>Risk signals</b><br>
                Identify risk-related language for human review.</p>
                <p><b>Local-first processing</b><br>
                Use local embeddings and a persistent vector database.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <h3>🧭 Getting started</h3>
                <p>1. Open Document Center.</p>
                <p>2. Select one or more files to upload.</p>
                <p>3. Index the documents.</p>
                <p>4. Ask questions in Semantic Search.</p>
                <p>5. Review Risk Signals.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("### Recently uploaded files")

    if files:
        st.dataframe(
            pd.DataFrame([
                {
                    "Filename": p.name,
                    "Size": human_size(p.stat().st_size),
                    "Modified": datetime.fromtimestamp(
                        p.stat().st_mtime
                    ).strftime("%d %b %Y, %H:%M"),
                }
                for p in files[:10]
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No uploaded files yet. Visit Document Center to begin.")


# =========================================================
# 9. DOCUMENT CENTER — MULTIPLE FILE UPLOADS
# =========================================================

elif page == "Document Center":
    st.markdown("## 📂 Document Center")
    st.write(
        "Select multiple project documents in one operation, then save "
        "and index each document."
    )

    st.info(
        "Accepted formats depend on your ingestion code. Common formats "
        "include PDF, DOCX, TXT, MD, CSV, JSON, XLSX, PPTX, HTML, and code."
    )

    uploaded_files = st.file_uploader(
        "Choose one or more documents",
        type=None,
        accept_multiple_files=True,
        help="Hold Ctrl or Shift while selecting files to choose multiple.",
        key="project_document_uploader",
    )

    if uploaded_files:
        st.success(f"{len(uploaded_files)} file(s) selected.")

        st.dataframe(
            pd.DataFrame([
                {
                    "Filename": file.name,
                    "Type": file.type or "Unknown",
                    "Size": human_size(file.size),
                }
                for file in uploaded_files
            ]),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "📥 Save and index selected documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        pipeline = get_pipeline_safely()

        if pipeline is not None:
            successes = []
            failures = []

            progress = st.progress(0)
            status = st.empty()

            for index, uploaded_file in enumerate(uploaded_files):
                saved_path = None

                try:
                    saved_path = unique_upload_path(uploaded_file.name)
                    status.write(
                        f"Processing {index + 1}/{len(uploaded_files)}: "
                        f"{uploaded_file.name}"
                    )

                    # Save the original upload first.
                    with saved_path.open("wb") as output:
                        output.write(uploaded_file.getbuffer())

                    # Index the saved file.
                    run_document_ingestion(pipeline, saved_path)
                    successes.append(saved_path.name)

                except Exception as exc:
                    failures.append({
                        "Filename": uploaded_file.name,
                        "Error": f"{type(exc).__name__}: {exc}",
                    })

                finally:
                    progress.progress((index + 1) / len(uploaded_files))

            status.empty()

            if successes:
                st.success(
                    f"Successfully indexed {len(successes)} file(s)."
                )
                for name in successes:
                    st.write(f"✓ {name}")

            if failures:
                st.error(
                    f"{len(failures)} file(s) could not be indexed."
                )
                st.dataframe(
                    pd.DataFrame(failures),
                    use_container_width=True,
                    hide_index=True,
                )

            # Do not rerun immediately: keep the result messages visible.

    st.divider()
    st.markdown("### Files saved in the uploads folder")

    stored_files = list_uploaded_files()

    if stored_files:
        st.dataframe(
            pd.DataFrame([
                {
                    "Filename": p.name,
                    "Extension": p.suffix.lower() or "None",
                    "Size": human_size(p.stat().st_size),
                    "Modified": datetime.fromtimestamp(
                        p.stat().st_mtime
                    ).strftime("%d %b %Y, %H:%M"),
                }
                for p in stored_files
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("The uploads folder is empty.")


# =========================================================
# 10. SEMANTIC SEARCH
# =========================================================

elif page == "Semantic Search":
    st.markdown("## 🔎 Semantic Search")
    st.write("Ask a question about the indexed project documentation.")

    with st.form("semantic_search_form"):
        query = st.text_area(
            "Your question",
            placeholder=(
                "Example: Which dependencies might delay project delivery?"
            ),
            height=110,
        )

        top_k = st.slider(
            "Number of passages to retrieve",
            min_value=1,
            max_value=10,
            value=5,
        )

        submitted = st.form_submit_button(
            "Search project knowledge",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            pipeline = get_pipeline_safely()

            if pipeline is not None:
                try:
                    with st.spinner("Searching indexed documents..."):
                        result = run_semantic_search(
                            pipeline, query.strip(), top_k
                        )

                    st.session_state.last_answer = result
                    st.session_state.last_query = query.strip()
                    st.session_state.search_history.append({
                        "Question": query.strip(),
                        "Time": datetime.now().strftime(
                            "%d %b %Y, %H:%M:%S"
                        ),
                    })

                except Exception as exc:
                    st.error("Semantic search failed.")
                    st.code(f"{type(exc).__name__}: {exc}")

    if st.session_state.last_answer is not None:
        st.divider()
        st.markdown("### Result")
        st.markdown(f"**Question:** {st.session_state.last_query}")
        st.write(normalize_result(st.session_state.last_answer))

        if isinstance(st.session_state.last_answer, dict):
            with st.expander("View result details"):
                st.json(st.session_state.last_answer)

        elif isinstance(st.session_state.last_answer, list):
            with st.expander("View retrieved passages"):
                for index, item in enumerate(
                    st.session_state.last_answer, start=1
                ):
                    st.markdown(f"**Passage {index}**")
                    st.write(item)
                    st.divider()

    st.divider()
    st.markdown("### Recent searches")

    if st.session_state.search_history:
        st.dataframe(
            pd.DataFrame(
                list(reversed(st.session_state.search_history[-10:]))
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Search history will appear here.")


# =========================================================
# 11. RISK SIGNALS
# =========================================================

elif page == "Risk Signals":
    st.markdown("## ⚠️ Risk Signals")
    st.write(
        "Scan text extracted from uploaded documents for phrases that "
        "may indicate project delivery, budget, dependency, resource, "
        "or quality concerns."
    )

    st.warning(
        "These are keyword matches, not validated risk predictions. "
        "Review the context before taking action."
    )

    risk_categories = {
        "Schedule and delivery": [
            "delay", "delayed", "behind schedule", "deadline",
            "overdue", "milestone", "slippage",
        ],
        "Budget and cost": [
            "budget overrun", "cost overrun", "overspend",
            "funding", "budget risk", "cost increase",
        ],
        "Dependencies and blockers": [
            "dependency", "dependencies", "blocked", "blocker",
            "third-party", "vendor delay", "integration issue",
        ],
        "Resources and staffing": [
            "resource shortage", "staffing", "understaffed",
            "capacity", "unavailable", "key person",
        ],
        "Quality and technical issues": [
            "critical bug", "defect", "failure",
            "security vulnerability", "data loss", "performance issue",
        ],
    }

    stored_files = list_uploaded_files()

    if not stored_files:
        st.info("Upload documents in Document Center before scanning.")
    else:
        selected_categories = st.multiselect(
            "Select categories",
            options=list(risk_categories.keys()),
            default=list(risk_categories.keys()),
        )

        if st.button(
            "🔍 Scan uploaded documents",
            type="primary",
            use_container_width=True,
        ):
            try:
                from app.ingestion import extract_text_from_file

                findings = []
                totals = {category: 0 for category in selected_categories}

                for path in stored_files:
                    try:
                        text = extract_text_from_file(str(path)) or ""

                        for line_number, line in enumerate(
                            text.splitlines(), start=1
                        ):
                            normalized = line.casefold()

                            for category in selected_categories:
                                matches = [
                                    term for term in risk_categories[category]
                                    if term in normalized
                                ]

                                if matches:
                                    totals[category] += 1
                                    findings.append({
                                        "Category": category,
                                        "Document": path.name,
                                        "Line": line_number,
                                        "Matched terms": ", ".join(matches),
                                        "Excerpt": line.strip()[:500],
                                    })

                    except Exception as exc:
                        st.warning(
                            f"{path.name}: {type(exc).__name__}: {exc}"
                        )

                st.session_state.risk_findings = findings
                st.session_state.risk_totals = totals

            except Exception as exc:
                st.error("Risk scan could not start.")
                st.code(f"{type(exc).__name__}: {exc}")

        findings = st.session_state.risk_findings

        if findings is not None:
            st.divider()
            st.markdown("### Scan summary")

            if st.session_state.risk_totals:
                metric_columns = st.columns(
                    min(4, len(st.session_state.risk_totals))
                )

                for index, (category, count) in enumerate(
                    st.session_state.risk_totals.items()
                ):
                    with metric_columns[index % len(metric_columns)]:
                        st.metric(category, count)

            st.metric("Matching excerpts", len(findings))

            if findings:
                findings_df = pd.DataFrame(findings)
                st.dataframe(
                    findings_df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "⬇️ Download findings as CSV",
                    data=findings_df.to_csv(index=False).encode("utf-8"),
                    file_name="project_risk_signals.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            else:
                st.success(
                    "No configured risk keywords were found. "
                    "This does not guarantee the project is risk-free."
                )


# =========================================================
# 12. SYSTEM DIAGNOSTICS
# =========================================================

elif page == "System Diagnostics":
    st.markdown("## 🛠️ System Diagnostics")

    st.markdown("### Folder checks")

    folder_rows = []

    for path in (ROOT, DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
        folder_rows.append({
            "Path": str(path),
            "Exists": path.exists(),
            "Is folder": path.is_dir(),
            "Is file": path.is_file(),
        })

    st.dataframe(
        pd.DataFrame(folder_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Python environment")

    st.code(
        f"Python executable: {sys.executable}\n"
        f"Python version: {sys.version.split()[0]}\n"
        f"Project root: {ROOT}\n"
        f"Working directory: {Path.cwd()}"
    )

    st.markdown("### RAG pipeline constructor")

    try:
        st.code(str(inspect.signature(RAGPipeline)))
    except Exception as exc:
        st.error(f"Could not inspect constructor: {exc}")

    st.markdown("### Pipeline test")

    if st.button("Test pipeline initialization", type="primary"):
        pipeline = get_pipeline_safely()

        if pipeline is not None:
            st.success("RAGPipeline initialized successfully.")
            st.caption(
                "Ingestion and search still need to be tested separately."
            )

    st.markdown("### Upload directory")

    stored_files = list_uploaded_files()

    if stored_files:
        st.dataframe(
            pd.DataFrame([
                {
                    "Filename": path.name,
                    "Size": human_size(path.stat().st_size),
                }
                for path in stored_files
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No uploaded files found.")


# =========================================================
# 13. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Project Intelligence & Risk Advisor | "
    "Infosys Springboard Internship Project | "
    "Local-first document intelligence"
)
