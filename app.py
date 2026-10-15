from pathlib import Path
from datetime import datetime
import json
import traceback

import streamlit as st
import pandas as pd

from app.rag_pipeline import RAGPipeline


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Project Intelligence | AI Risk Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# 2. PROJECT PATHS
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"


# =========================================================
# 3. SAFE DIRECTORY INITIALIZATION
# =========================================================

def ensure_directory(folder: Path):
    """
    Create a directory if it does not exist.
    Report a clear error if a file occupies the required path.
    """
    if folder.exists() and not folder.is_dir():
        raise NotADirectoryError(
            f"The path exists as a FILE, but a folder is required:\n{folder}\n\n"
            "Rename or move the conflicting file, then restart the app."
        )

    folder.mkdir(parents=True, exist_ok=True)

    if not folder.is_dir():
        raise OSError(f"Could not create or access directory: {folder}")


try:
    ensure_directory(DATA_DIR)
    ensure_directory(UPLOAD_DIR)
    ensure_directory(CHROMA_DIR)

except Exception as exc:
    st.error("Could not initialize the project folders.")

    st.markdown("### Diagnostic information")
    st.code(f"{type(exc).__name__}: {exc}")

    st.markdown("### Expected project folders")
    st.code(
        f"{DATA_DIR}\n"
        f"{UPLOAD_DIR}\n"
        f"{CHROMA_DIR}"
    )

    st.warning(
        "Open these paths in Windows File Explorer. "
        "If a file exists where a folder is required, rename or move "
        "that file instead of deleting it. Do not delete your ChromaDB "
        "folder if it contains indexed data."
    )

    st.stop()


# =========================================================
# 4. CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(
            145deg,
            #08111f 0%,
            #0b1526 48%,
            #111c31 100%
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
        font-size: 2.25rem;
        margin-bottom: 8px;
        color: #f8fafc;
    }

    .hero p {
        font-size: 1.02rem;
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

    .small-muted {
        color: #94a3b8;
        font-size: 0.88rem;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.78);
        border: 1px solid rgba(148, 163, 184, 0.18);
        padding: 17px;
        border-radius: 15px;
    }

    div[data-testid="stMetricLabel"] {
        color: #cbd5e1;
    }

    div[data-testid="stMetricValue"] {
        color: #f8fafc;
    }

    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
    }

    [data-testid="stFileUploader"] {
        background: rgba(15, 23, 42, 0.45);
        border-radius: 12px;
        padding: 12px;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.2);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 5. SESSION STATE
# =========================================================

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "last_query" not in st.session_state:
    st.session_state.last_query = ""


# =========================================================
# 6. PIPELINE INITIALIZATION
# =========================================================

@st.cache_resource(show_spinner=False)
def get_pipeline():
    """
    Reuse one RAG pipeline during the Streamlit session.
    """
    return RAGPipeline(
        upload_dir=str(UPLOAD_DIR),
        chroma_dir=str(CHROMA_DIR),
    )


def get_pipeline_safely():
    try:
        return get_pipeline()
    except Exception as exc:
        st.error("The AI search pipeline could not be initialized.")
        st.code(f"{type(exc).__name__}: {exc}")
        st.info(
            "Check app/rag_pipeline.py and requirements.txt. "
            "The first model initialization may also require downloading "
            "the embedding model."
        )
        return None


def list_uploaded_files():
    """
    List files saved in the upload directory.
    """
    try:
        return sorted(
            [
                path for path in UPLOAD_DIR.iterdir()
                if path.is_file()
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
    except OSError:
        return []


def human_size(size_bytes):
    """
    Convert bytes into a readable size.
    """
    size = float(size_bytes)

    for unit in ["B", "KB", "MB", "GB"]:
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024

    return f"{size:.1f} GB"


def extract_result_text(result):
    """
    Normalize common RAG result formats into displayable text.
    """
    if result is None:
        return "No answer was returned."

    if isinstance(result, str):
        return result

    if isinstance(result, dict):
        for key in ["answer", "response", "result", "text"]:
            if result.get(key):
                return str(result[key])

        return json.dumps(result, indent=2, default=str)

    return str(result)


def run_semantic_search(pipeline, query, top_k):
    """
    Call a compatible search method exposed by RAGPipeline.

    This adapter supports common method names. If your pipeline uses a
    different method signature, the error will be displayed on screen.
    """
    candidate_methods = [
        "query",
        "search",
        "ask",
        "query_documents",
        "search_documents",
    ]

    for method_name in candidate_methods:
        method = getattr(pipeline, method_name, None)

        if callable(method):
            try:
                return method(query, top_k=top_k)
            except TypeError:
                try:
                    return method(query)
                except TypeError:
                    continue

    raise AttributeError(
        "No compatible search method was found in RAGPipeline. "
        "Expected one of: query, search, ask, query_documents, "
        "search_documents."
    )


def run_document_ingestion(pipeline, saved_path):
    """
    Find a compatible ingestion method on RAGPipeline.
    """
    candidate_methods = [
        "ingest_file",
        "add_document",
        "process_file",
        "ingest_document",
        "index_file",
        "add_file",
    ]

    for method_name in candidate_methods:
        method = getattr(pipeline, method_name, None)

        if callable(method):
            try:
                return method(str(saved_path))
            except TypeError:
                continue

    raise AttributeError(
        "No compatible document-ingestion method was found in "
        "RAGPipeline. Check app/rag_pipeline.py for its ingestion method."
    )


# =========================================================
# 7. SIDEBAR NAVIGATION
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
        index=0,
    )

    st.divider()

    st.markdown("### Project status")

    current_files = list_uploaded_files()

    st.metric("Stored documents", len(current_files))

    st.caption("Local embeddings • ChromaDB • Streamlit")

    st.divider()

    st.caption("Infosys Springboard Internship Project")
    st.caption("AI Project Intelligence & Risk Advisor")


# =========================================================
# 8. SHARED HEADER
# =========================================================

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">Enterprise intelligence workspace</div>
        <h1>AI Project Intelligence &amp; Risk Advisor</h1>
        <p>
            Transform project documents into searchable knowledge,
            discover important signals, and support better project decisions.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 9. OVERVIEW PAGE
# =========================================================

if page == "Overview":

    files = list_uploaded_files()

    total_size = sum(
        path.stat().st_size
        for path in files
        if path.exists()
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📄 Documents", len(files))
    col2.metric("💾 Stored data", human_size(total_size))
    col3.metric("🔎 Searches", len(st.session_state.search_history))
    col4.metric("⚠️ Risk keywords", "Heuristic")

    st.markdown("## Welcome to your intelligence workspace")

    left, right = st.columns([1.35, 1])

    with left:
        st.markdown(
            """
            <div class="section-card">
                <h3>🚀 What this platform does</h3>
                <p>
                    <b>Document intelligence</b><br>
                    Upload project files and prepare them for local indexing.
                </p>
                <p>
                    <b>Semantic search</b><br>
                    Search for meaning and context rather than relying only
                    on exact keyword matches.
                </p>
                <p>
                    <b>Risk signal discovery</b><br>
                    Flag document passages containing risk-related language
                    for human review.
                </p>
                <p>
                    <b>Local-first architecture</b><br>
                    Use local embedding models and a persistent vector
                    database without requiring a commercial LLM API key.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with right:
        st.markdown(
            """
            <div class="section-card">
                <h3>🧭 Quick start</h3>
                <p>1. Open <b>Document Center</b>.</p>
                <p>2. Upload project documents.</p>
                <p>3. Index the documents into the vector store.</p>
                <p>4. Open <b>Semantic Search</b>.</p>
                <p>5. Explore <b>Risk Signals</b> and validate the results.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("## Recently stored documents")

    if files:
        rows = []

        for path in files[:8]:
            stat = path.stat()

            rows.append(
                {
                    "Document": path.name,
                    "Size": human_size(stat.st_size),
                    "Last modified": datetime.fromtimestamp(
                        stat.st_mtime
                    ).strftime("%d %b %Y, %I:%M %p"),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info(
            "No documents are stored yet. Open Document Center to upload "
            "your first project document."
        )

    st.caption(
        "Note: keyword-based risk indicators are preliminary signals, "
        "not validated predictions of project failure."
    )


# =========================================================
# 10. DOCUMENT CENTER PAGE
# =========================================================

elif page == "Document Center":

    st.markdown("## 📂 Document Center")
    st.write(
        "Upload project documentation and add it to the local search index."
    )

    st.info(
        "Supported formats depend on your ingestion implementation. "
        "Common formats include PDF, DOCX, TXT, Markdown, CSV, JSON, "
        "XLSX, PPTX, HTML, and source-code files."
    )

    uploaded_files = st.file_uploader(
        "Choose project documents",
        type=None,
        accept_multiple_files=True,
        help="Select one or more files from your computer.",
    )

    pipeline = get_pipeline_safely()

    if st.button(
        "📥 Save and index documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files or pipeline is None,
    ):

        successful = 0
        failures = []

        progress = st.progress(0)
        status = st.empty()

        for index, uploaded_file in enumerate(uploaded_files):

            try:
                safe_name = Path(uploaded_file.name).name

                if not safe_name:
                    raise ValueError("The uploaded file has no valid name.")

                saved_path = UPLOAD_DIR / safe_name

                # Avoid overwriting an existing file with the same name.
                if saved_path.exists():
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    saved_path = UPLOAD_DIR / (
                        f"{saved_path.stem}_{timestamp}{saved_path.suffix}"
                    )

                status.write(f"Processing: {safe_name}")

                with open(saved_path, "wb") as file_handle:
                    file_handle.write(uploaded_file.getbuffer())

                run_document_ingestion(pipeline, saved_path)

                successful += 1

            except Exception as exc:
                failures.append(
                    {
                        "file": uploaded_file.name,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

            progress.progress((index + 1) / len(uploaded_files))

        status.empty()

        if successful:
            st.success(
                f"Successfully processed {successful} document(s)."
            )
            st.cache_data.clear()

        if failures:
            st.error(
                f"{len(failures)} document(s) could not be indexed."
            )

            for failure in failures:
                with st.expander(f"Error: {failure['file']}"):
                    st.code(failure["error"])

        st.rerun()

    st.divider()

    st.markdown("### Stored files")

    files = list_uploaded_files()

    if files:
        rows = []

        for path in files:
            stat = path.stat()

            rows.append(
                {
                    "Filename": path.name,
                    "File type": path.suffix.lower() or "No extension",
                    "Size": human_size(stat.st_size),
                    "Modified": datetime.fromtimestamp(
                        stat.st_mtime
                    ).strftime("%d %b %Y, %H:%M"),
                }
            )

        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

    else:
        st.info("Your uploads folder is currently empty.")


# =========================================================
# 11. SEMANTIC SEARCH PAGE
# =========================================================

elif page == "Semantic Search":

    st.markdown("## 🔎 Semantic Search")
    st.write(
        "Ask a question about your indexed project documentation."
    )

    with st.form("semantic_search_form"):

        query = st.text_area(
            "Your question",
            placeholder=(
                "Example: Which project dependencies could delay delivery?"
            ),
            height=110,
        )

        top_k = st.slider(
            "Number of results to retrieve",
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
            st.warning("Enter a question before searching.")

        else:
            pipeline = get_pipeline_safely()

            if pipeline is not None:
                try:
                    with st.spinner(
                        "Searching the local project knowledge base..."
                    ):
                        result = run_semantic_search(
                            pipeline,
                            query.strip(),
                            top_k,
                        )

                    st.session_state.last_answer = result
                    st.session_state.last_query = query.strip()

                    st.session_state.search_history.append(
                        {
                            "query": query.strip(),
                            "timestamp": datetime.now().strftime(
                                "%d %b %Y, %H:%M:%S"
                            ),
                        }
                    )

                except Exception as exc:
                    st.error("The semantic search failed.")
                    st.code(f"{type(exc).__name__}: {exc}")

    if st.session_state.last_answer is not None:

        st.divider()
        st.markdown("### Search result")

        st.markdown(
            f"**Question:** {st.session_state.last_query}"
        )

        st.markdown(
            extract_result_text(st.session_state.last_answer)
        )

        if isinstance(st.session_state.last_answer, dict):
            with st.expander("View raw result details"):
                st.json(
                    st.session_state.last_answer,
                    expanded=True,
                )

        elif isinstance(st.session_state.last_answer, list):
            with st.expander("View retrieved passages"):
                for index, item in enumerate(
                    st.session_state.last_answer,
                    start=1,
                ):
                    st.markdown(f"**Result {index}**")
                    st.write(item)
                    st.divider()

    st.divider()

    st.markdown("### Recent search history")

    if st.session_state.search_history:
        st.dataframe(
            pd.DataFrame(
                list(reversed(st.session_state.search_history[-10:]))
            ),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.caption("Your search history will appear here.")


# =========================================================
# 12. RISK SIGNALS PAGE
# =========================================================

elif page == "Risk Signals":

    st.markdown("## ⚠️ Risk Signals")
    st.write(
        "Scan stored text-based documents for language that may indicate "
        "delivery, budget, dependency, or resource risks."
    )

    st.warning(
        "This is a preliminary keyword-based scan. It does not understand "
        "all context, calculate a validated risk probability, or replace "
        "a project manager's assessment."
    )

    risk_categories = {
        "Schedule and delivery": [
            "delay",
            "delayed",
            "behind schedule",
            "deadline",
            "overdue",
            "milestone",
            "slippage",
        ],
        "Budget and cost": [
            "budget overrun",
            "cost overrun",
            "overspend",
            "funding",
            "budget risk",
            "cost increase",
        ],
        "Dependencies and blockers": [
            "dependency",
            "dependencies",
            "blocked",
            "blocker",
            "third-party",
            "vendor delay",
            "integration issue",
        ],
        "Resources and staffing": [
            "resource shortage",
            "staffing",
            "understaffed",
            "capacity",
            "unavailable",
            "key person",
        ],
        "Quality and technical issues": [
            "critical bug",
            "defect",
            "failure",
            "security vulnerability",
            "data loss",
            "performance issue",
        ],
    }

    files = list_uploaded_files()

    if not files:
        st.info(
            "Upload project documents first to scan them for risk signals."
        )

    else:
        st.markdown("### Scan settings")

        selected_categories = st.multiselect(
            "Risk categories",
            options=list(risk_categories.keys()),
            default=list(risk_categories.keys()),
        )

        max_preview = st.slider(
            "Maximum matching excerpts per category",
            min_value=1,
            max_value=20,
            value=5,
        )

        if st.button(
            "🔍 Scan documents for risk signals",
            type="primary",
            use_container_width=True,
        ):

            try:
                from app.ingestion import extract_text_from_file

                all_findings = []
                category_totals = {
                    category: 0
                    for category in selected_categories
                }

                for path in files:

                    try:
                        extracted_text = extract_text_from_file(str(path))

                        if not extracted_text:
                            continue

                        lines = extracted_text.splitlines()

                        # Fall back to the entire text if it contains no
                        # meaningful line breaks.
                        if not lines:
                            lines = [extracted_text]

                        for line_number, line in enumerate(
                            lines,
                            start=1,
                        ):
                            normalized = line.lower()

                            for category in selected_categories:
                                matching_terms = [
                                    term
                                    for term in risk_categories[category]
                                    if term in normalized
                                ]

                                if matching_terms:
                                    category_totals[category] += 1

                                    all_findings.append(
                                        {
                                            "Category": category,
                                            "Document": path.name,
                                            "Line": line_number,
                                            "Matched terms": ", ".join(
                                                matching_terms
                                            ),
                                            "Excerpt": line.strip()[:500],
                                        }
                                    )

                    except Exception as exc:
                        st.warning(
                            f"Could not scan {path.name}: "
                            f"{type(exc).__name__}: {exc}"
                        )

                st.session_state["risk_findings"] = all_findings
                st.session_state["risk_totals"] = category_totals

            except Exception as exc:
                st.error("Could not start the risk scan.")
                st.code(f"{type(exc).__name__}: {exc}")

        findings = st.session_state.get("risk_findings", None)

        if findings is not None:

            totals = st.session_state.get("risk_totals", {})

            st.divider()
            st.markdown("### Scan summary")

            metric_columns = st.columns(
                max(1, min(4, len(totals) or 1))
            )

            if totals:
                for index, (category, count) in enumerate(totals.items()):
                    with metric_columns[index % len(metric_columns)]:
                        st.metric(category, count)

            st.metric("Total matching excerpts", len(findings))

            if findings:
                findings_df = pd.DataFrame(findings)

                st.markdown("### Detected signals")

                st.dataframe(
                    findings_df,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = findings_df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "⬇️ Export findings as CSV",
                    data=csv_data,
                    file_name="project_risk_signals.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

                for category in selected_categories:
                    category_findings = [
                        finding
                        for finding in findings
                        if finding["Category"] == category
                    ]

                    if category_findings:
                        with st.expander(
                            f"{category} — {len(category_findings)} finding(s)"
                        ):
                            for finding in category_findings[
                                :max_preview
                            ]:
                                st.markdown(
                                    f"**{finding['Document']}** "
                                    f"(line {finding['Line']})"
                                )
                                st.caption(
                                    "Matched: "
                                    + finding["Matched terms"]
                                )
                                st.write(finding["Excerpt"])
                                st.divider()

            else:
                st.success(
                    "No configured risk keywords were found in the "
                    "scanned text. This does not guarantee the absence "
                    "of project risks."
                )


# =========================================================
# 13. SYSTEM DIAGNOSTICS PAGE
# =========================================================

elif page == "System Diagnostics":

    st.markdown("## 🛠️ System Diagnostics")
    st.write(
        "Inspect the local folders and check whether the application "
        "can initialize its AI pipeline."
    )

    folder_info = []

    for folder in [ROOT, DATA_DIR, UPLOAD_DIR, CHROMA_DIR]:
        folder_info.append(
            {
                "Path": str(folder),
                "Exists": folder.exists(),
                "Is directory": folder.is_dir(),
                "Is file": folder.is_file(),
            }
        )

    st.markdown("### Folder checks")

    st.dataframe(
        pd.DataFrame(folder_info),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Runtime information")

    st.code(
        f"Python executable: {__import__('sys').executable}\n"
        f"Python version: {__import__('sys').version.split()[0]}\n"
        f"Project root: {ROOT}\n"
        f"Working directory: {Path.cwd()}"
    )

    st.markdown("### AI pipeline")

    if st.button("Test pipeline initialization", type="primary"):
        pipeline = get_pipeline_safely()

        if pipeline is not None:
            st.success(
                "The RAGPipeline object initialized successfully."
            )
            st.caption(
                "This confirms initialization only. It does not guarantee "
                "that document ingestion or search will succeed."
            )

    st.markdown("### Stored documents")

    files = list_uploaded_files()

    if files:
        st.write(f"Found {len(files)} file(s) in the uploads folder.")

        for path in files:
            st.write(
                f"• {path.name} — {human_size(path.stat().st_size)}"
            )
    else:
        st.info("No uploaded files were found.")


# =========================================================
# 14. FOOTER
# =========================================================

st.divider()

st.caption(
    "AI Project Intelligence & Risk Advisor | "
    "Infosys Springboard Internship Project | "
    "Local-first document intelligence"
)
