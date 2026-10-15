
from pathlib import Path
from datetime import date, datetime
import inspect
import json

import pandas as pd
import streamlit as st

from app.rag_pipeline import RAGPipeline
from app.ingestion import extract_text_from_file
from app.agents import ProjectAnalysisAgents
from app.health import calculate_project_health
from app.documentation_agent import DocumentationGenerationAgent
from app.assistant import ProjectIntelligenceAssistant


# =========================================================
# 1. PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Project Intelligence | Risk Advisor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# =========================================================
# 2. DIRECTORIES
# =========================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma_db"

for folder in (DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
    if folder.exists() and not folder.is_dir():
        st.error(f"A file exists where a directory is required: {folder}")
        st.stop()
    folder.mkdir(parents=True, exist_ok=True)


# =========================================================
# 3. WEBSITE-STYLE CSS
# =========================================================

st.markdown(
    """
    <style>
    :root {
        --page-bg: #080d19;
        --panel-bg: #10182a;
        --panel-soft: #131e33;
        --border: rgba(148, 163, 184, 0.16);
        --muted: #9aaac2;
        --text: #edf3ff;
        --blue: #67a5ff;
        --violet: #a78bfa;
        --cyan: #67e8f9;
    }

    .stApp {
        background:
            radial-gradient(
                ellipse at 10% 0%,
                rgba(59, 130, 246, 0.12),
                transparent 34%
            ),
            radial-gradient(
                ellipse at 95% 10%,
                rgba(139, 92, 246, 0.10),
                transparent 28%
            ),
            var(--page-bg);
        color: var(--text);
    }

    [data-testid="stSidebar"],
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
    }

    .block-container {
        max-width: 1500px;
        padding-top: 1.1rem;
        padding-bottom: 3rem;
    }

    header[data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stDecoration"] {
        display: none;
    }

    h1, h2, h3 {
        color: #f3f6ff !important;
        letter-spacing: -0.025em;
    }

    p, label, .stMarkdown {
        color: #d3ddef;
    }

    .top-brand {
        display: flex;
        align-items: center;
        gap: 13px;
        padding: 10px 0 18px 0;
    }

    .brand-icon {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 46px;
        height: 46px;
        border-radius: 15px;
        background: linear-gradient(135deg, #2563eb, #7c3aed);
        box-shadow: 0 8px 30px rgba(59, 130, 246, 0.23);
        font-size: 24px;
    }

    .brand-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 750;
        line-height: 1.35;
    }

    .brand-subtitle {
        color: #91a3bd;
        font-size: 0.78rem;
        margin-top: 2px;
    }

    .top-status {
        text-align: right;
        color: #a8b8d0;
        font-size: 0.8rem;
        padding-top: 12px;
    }

    .hero {
        position: relative;
        overflow: hidden;
        padding: 32px;
        margin: 10px 0 24px 0;
        border: 1px solid rgba(147, 197, 253, 0.19);
        border-radius: 24px;
        background:
            linear-gradient(
                112deg,
                rgba(30, 64, 175, 0.28),
                rgba(76, 29, 149, 0.20) 58%,
                rgba(8, 145, 178, 0.10)
            );
    }

    .hero-eyebrow {
        text-transform: uppercase;
        letter-spacing: 2px;
        font-size: 0.72rem;
        font-weight: 800;
        color: #93c5fd;
        margin-bottom: 12px;
    }

    .hero-title {
        color: #f8fafc;
        font-size: clamp(1.75rem, 3vw, 2.65rem);
        font-weight: 800;
        line-height: 1.16;
        letter-spacing: -1.5px;
        margin-bottom: 12px;
    }

    .hero-description {
        max-width: 780px;
        color: #c4d1e4;
        font-size: 1rem;
        line-height: 1.75;
    }

    .hero-footnote {
        color: #9cb6da;
        font-size: 0.78rem;
        margin-top: 18px;
    }

    .section-kicker {
        color: #8ab9ff;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-size: 0.7rem;
        font-weight: 800;
        margin-bottom: 5px;
    }

    .section-description {
        color: var(--muted);
        margin-top: -4px;
        margin-bottom: 18px;
        font-size: 0.9rem;
    }

    .panel {
        background: linear-gradient(
            145deg,
            rgba(19, 30, 51, 0.95),
            rgba(13, 22, 39, 0.96)
        );
        border: 1px solid var(--border);
        border-radius: 19px;
        padding: 21px;
        margin-bottom: 15px;
    }

    .panel-title {
        color: #f1f5ff;
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 9px;
    }

    .panel-text {
        color: #9fb0c9;
        font-size: 0.86rem;
        line-height: 1.7;
    }

    .kpi-label {
        color: #9fb0c9;
        font-size: 0.78rem;
        font-weight: 600;
    }

    .kpi-note {
        color: #8297b5;
        font-size: 0.75rem;
        margin-top: 4px;
    }

    div[data-testid="stMetric"] {
        background: linear-gradient(
            145deg,
            rgba(19, 30, 51, 0.97),
            rgba(13, 22, 39, 0.97)
        );
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 18px 18px 15px 18px;
        min-height: 112px;
    }

    div[data-testid="stMetricLabel"] {
        color: #a7b6cd !important;
    }

    div[data-testid="stMetricValue"] {
        color: #f4f7ff !important;
        font-weight: 800;
    }

    div[data-testid="stMetricDelta"] {
        font-size: 0.76rem;
    }

    .stButton > button,
    .stDownloadButton > button {
        border-radius: 11px;
        min-height: 42px;
        font-weight: 650;
        border: 1px solid rgba(125, 164, 225, 0.22);
        transition: all 0.18s ease;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(100deg, #2563eb, #6d4aff);
        color: white;
        border: none;
    }

    .stButton > button:hover,
    .stDownloadButton > button:hover {
        border-color: #7c9ee8;
        transform: translateY(-1px);
    }

    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea,
    div[data-testid="stDateInput"] input {
        background: #0c1424;
        color: #f1f5ff;
        border-color: rgba(148, 163, 184, 0.25);
        border-radius: 10px;
    }

    [data-testid="stFileUploader"] {
        background: rgba(12, 20, 36, 0.6);
        border: 1px dashed rgba(125, 164, 225, 0.4);
        border-radius: 15px;
        padding: 10px;
    }

    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stProgressBar"] > div > div {
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    }

    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    hr {
        border-color: rgba(148, 163, 184, 0.15);
        margin: 22px 0;
    }

    .footer {
        color: #70839f;
        font-size: 0.78rem;
        text-align: center;
        padding: 16px 0 0 0;
    }

    @media (max-width: 800px) {
        .hero {
            padding: 23px;
        }

        .top-status {
            text-align: left;
            padding: 0 0 10px 0;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 4. SESSION STATE
# =========================================================

DEFAULT_STATE = {
    "page": "Overview",
    "search_history": [],
    "last_answer": None,
    "last_query": "",
    "project_analysis": None,
    "project_health": None,
    "project_documentation": None,
    "analysis_signature": None,
    "analysis_errors": [],
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


# =========================================================
# 5. FILE AND DISPLAY UTILITIES
# =========================================================

def list_uploaded_files():
    try:
        return sorted(
            [path for path in UPLOAD_DIR.iterdir() if path.is_file()],
            key=lambda path: path.stat().st_mtime,
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
    name = Path(filename).name

    if not name or name in {".", ".."}:
        raise ValueError("Invalid uploaded filename.")

    destination = UPLOAD_DIR / name

    if not destination.exists():
        return destination

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return destination.with_name(
        f"{destination.stem}_{stamp}{destination.suffix}"
    )


def invalidate_analysis():
    st.session_state.project_analysis = None
    st.session_state.project_health = None
    st.session_state.project_documentation = None
    st.session_state.analysis_signature = None
    st.session_state.analysis_errors = []


def current_file_signature():
    """Detect when the uploaded file list or modification times change."""
    return tuple(
        (
            path.name,
            path.stat().st_size,
            path.stat().st_mtime_ns,
        )
        for path in list_uploaded_files()
    )


# =========================================================
# 6. PIPELINE INITIALIZATION
# =========================================================

@st.cache_resource(show_spinner=False)
def get_pipeline():
    return RAGPipeline(
        upload_dir=str(UPLOAD_DIR),
        chroma_dir=str(CHROMA_DIR),
    )


def get_pipeline_safely():
    try:
        return get_pipeline()
    except Exception as exc:
        st.error("The local RAG pipeline could not initialize.")
        st.code(f"{type(exc).__name__}: {exc}")
        return None


# =========================================================
# 7. INGESTION COMPATIBILITY
# =========================================================

def index_saved_file(pipeline, file_path):
    """
    Call the available file-indexing method using its actual signature.
    """
    path = Path(file_path)

    method_names = (
        "index_file",
        "ingest_file",
        "add_document",
        "process_file",
        "ingest_document",
        "add_file",
        "ingest",
    )

    path_names = (
        "file_path",
        "path",
        "filepath",
        "document_path",
        "source_path",
        "file",
    )

    for method_name in method_names:
        method = getattr(pipeline, method_name, None)

        if not callable(method):
            continue

        signature = inspect.signature(method)
        parameters = signature.parameters

        path_parameter = next(
            (name for name in path_names if name in parameters),
            None,
        )

        if path_parameter:
            return method(**{path_parameter: str(path)})

        positional = [
            parameter
            for parameter in parameters.values()
            if parameter.kind in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        ]

        if positional:
            return method(str(path))

    batch_method = getattr(pipeline, "index_files", None)

    if callable(batch_method):
        return batch_method([str(path)])

    raise AttributeError(
        "No compatible file-indexing method was found in "
        "app/rag_pipeline.py. Check its index_file/index_files methods."
    )


# =========================================================
# 8. PROJECT ANALYSIS
# =========================================================

def analyze_uploaded_project(planned_end_date):
    """
    Extract text from saved files, run the project agents, calculate
    project health, and generate the three required documentation types.
    """
    files = list_uploaded_files()

    if not files:
        return None, None, None, [
            "Upload project documents before running the analysis."
        ]

    text_parts = []
    errors = []

    for path in files:
        try:
            extracted = extract_text_from_file(str(path))

            if extracted and extracted.strip():
                text_parts.append(
                    f"\n\nDOCUMENT: {path.name}\n{extracted}"
                )
            else:
                errors.append(
                    f"{path.name}: no extractable text was found."
                )

        except Exception as exc:
            errors.append(
                f"{path.name}: {type(exc).__name__}: {exc}"
            )

    combined_text = "\n".join(text_parts).strip()

    if not combined_text:
        return None, None, None, errors or [
            "No readable text was extracted from the uploaded files."
        ]

    agents = ProjectAnalysisAgents()

    analysis = agents.analyze(
        combined_text,
        planned_end_date=planned_end_date.isoformat(),
    )

    health = calculate_project_health(analysis)

    documentation = DocumentationGenerationAgent().generate_all(
        analysis
    )

    return analysis, health, documentation, errors


def get_project_analysis():
    """Return the current analysis, if one has already been generated."""
    return (
        st.session_state.project_analysis,
        st.session_state.project_health,
        st.session_state.project_documentation,
    )


# =========================================================
# 9. TOP NAVIGATION
# =========================================================

top_left, top_right = st.columns([3.2, 1])

with top_left:
    st.markdown(
        """
        <div class="top-brand">
            <div class="brand-icon">🧠</div>
            <div>
                <div class="brand-title">Project Intelligence</div>
                <div class="brand-subtitle">
                    AI Project Intelligence &amp; Risk Advisor
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with top_right:
    st.markdown(
        """
        <div class="top-status">
            <span style="color:#6ee7b7;">●</span>
            Local workspace<br>
            <span style="font-size:0.72rem;">Project analysis environment</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

NAV_ITEMS = [
    "Overview",
    "Documents",
    "Intelligence",
    "Risk Analysis",
    "Diagnostics",
]

selected_page = st.radio(
    "Main navigation",
    NAV_ITEMS,
    horizontal=True,
    label_visibility="collapsed",
    key="top_navigation",
)

st.session_state.page = selected_page

st.markdown("---")


# =========================================================
# 10. OVERVIEW DASHBOARD
# =========================================================

if selected_page == "Overview":
    files = list_uploaded_files()
    total_size = sum(path.stat().st_size for path in files)

    analysis, health, documentation = get_project_analysis()

    st.markdown(
        """
        <div class="hero">
            <div class="hero-eyebrow">Enterprise project workspace</div>
            <div class="hero-title">
                Project Intelligence Dashboard
            </div>
            <div class="hero-description">
                Turn project documentation into searchable knowledge.
                Identify risks, monitor blockers, assess delivery pressure,
                and generate structured project artifacts from one workspace.
            </div>
            <div class="hero-footnote">
                LOCAL EMBEDDINGS &nbsp; • &nbsp; CHROMADB
                &nbsp; • &nbsp; DOCUMENT-BASED ANALYSIS
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="section-kicker">Project at a glance</div>',
                unsafe_allow_html=True)
    st.markdown("### Workspace overview")

    if health:
        health_score = health.get("health_score", "—")
        health_status = health.get("status", "Not calculated")
    else:
        health_score = "—"
        health_status = "Run project analysis"

    risk_count = (
        analysis.get("risk_analysis", {}).get("risk_count", 0)
        if analysis else 0
    )
    blocker_count = (
        analysis.get("blocker_analysis", {}).get("blocker_count", 0)
        if analysis else 0
    )
    action_count = (
        analysis.get("action_analysis", {}).get("action_item_count", 0)
        if analysis else 0
    )

    metric_cols = st.columns(5)

    metric_cols[0].metric("Documents", len(files))
    metric_cols[1].metric("Stored data", human_size(total_size))
    metric_cols[2].metric("Health score", health_score)
    metric_cols[3].metric("Risk signals", risk_count)
    metric_cols[4].metric("Blockers", blocker_count)

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown(
            """
            <div class="panel">
                <div class="panel-title">Workspace activity</div>
                <div class="panel-text">
                    <b>Document intelligence</b><br>
                    Upload project plans, requirements, status reports,
                    and meeting notes.<br><br>
                    <b>Knowledge retrieval</b><br>
                    Ask questions about your indexed documents and inspect
                    retrieved source passages.<br><br>
                    <b>Project analysis</b><br>
                    Extract scope, identify risks and blockers, and
                    produce a transparent delivery-risk indicator.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Recent documents")

        if files:
            recent_df = pd.DataFrame([
                {
                    "Document": path.name,
                    "Size": human_size(path.stat().st_size),
                    "Last modified": datetime.fromtimestamp(
                        path.stat().st_mtime
                    ).strftime("%d %b %Y, %H:%M"),
                }
                for path in files[:8]
            ])

            st.dataframe(
                recent_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info(
                "Your workspace is empty. Open Documents to upload files."
            )

    with right:
        st.markdown("### Project status")

        if health:
            st.metric("Overall project health", health_score, health_status)

            if health.get("explanation"):
                st.write(health["explanation"])

            if health.get("recommendations"):
                st.markdown("**Recommended actions**")
                for recommendation in health["recommendations"]:
                    st.write(f"• {recommendation}")
        else:
            st.markdown(
                """
                <div class="panel">
                    <div class="panel-title">Analysis not yet generated</div>
                    <div class="panel-text">
                        Upload project documents and run project analysis
                        to calculate the health score, risk summary,
                        blocker count, and delivery forecast.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        if analysis:
            forecast = analysis.get("delivery_forecast", {})

            st.markdown("**Delivery forecast**")
            st.write(
                forecast.get(
                    "forecast_status",
                    "No forecast is available.",
                )
            )

            st.caption(
                "This is a heuristic warning indicator, not a calibrated "
                "probability of delay."
            )

        st.markdown("### Quick actions")

        if st.button(
            "Open document workspace →",
            use_container_width=True,
            type="primary",
        ):
            st.session_state.top_navigation = "Documents"
            st.rerun()

        if st.button(
            "Open risk analysis →",
            use_container_width=True,
        ):
            st.session_state.top_navigation = "Risk Analysis"
            st.rerun()

    st.markdown("---")
    st.markdown("### Project analysis")

    st.write(
        "Run analysis after uploading or updating project documents. "
        "The analysis uses extracted document text and rule-based agents."
    )

    planned_end_date = st.date_input(
        "Planned project completion date",
        value=date.today(),
        help="Used to calculate the remaining calendar days in the delivery forecast.",
        key="overview_planned_end_date",
    )

    if st.button(
        "Analyze project documents",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            "Extracting documents and analyzing project signals..."
        ):
            try:
                analysis, health, documentation, errors = (
                    analyze_uploaded_project(planned_end_date)
                )

                st.session_state.project_analysis = analysis
                st.session_state.project_health = health
                st.session_state.project_documentation = documentation
                st.session_state.analysis_errors = errors
                st.session_state.analysis_signature = current_file_signature()

                if analysis:
                    st.success("Project analysis completed.")
                else:
                    st.warning("Project analysis did not produce a result.")

            except Exception as exc:
                st.error("Project analysis failed.")
                st.code(f"{type(exc).__name__}: {exc}")

    for error in st.session_state.analysis_errors:
        st.warning(error)

    if st.session_state.project_analysis:
        analysis = st.session_state.project_analysis

        with st.expander("View extracted scope and deliverables"):
            st.json(analysis.get("scope", {}))

        with st.expander("View risk and blocker analysis"):
            st.json({
                "risk_analysis": analysis.get("risk_analysis", {}),
                "blocker_analysis": analysis.get("blocker_analysis", {}),
            })

        with st.expander("View delivery forecast and action items"):
            st.json({
                "delivery_forecast": analysis.get("delivery_forecast", {}),
                "action_analysis": analysis.get("action_analysis", {}),
            })


# =========================================================
# 11. DOCUMENT WORKSPACE
# =========================================================

elif selected_page == "Documents":
    st.markdown('<div class="section-kicker">Knowledge base</div>',
                unsafe_allow_html=True)
    st.markdown("## Document workspace")

    st.markdown(
        '<div class="section-description">'
        'Upload multiple project files, save them locally, and index them '
        'for semantic retrieval.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded_files = st.file_uploader(
        "Select project documents",
        accept_multiple_files=True,
        type=None,
        help=(
            "Common supported formats include PDF, DOCX, TXT, CSV, "
            "XLSX, PPTX, JSON, HTML, and code files. Actual support "
            "depends on app/ingestion.py."
        ),
        key="website_document_uploader",
    )

    if uploaded_files:
        st.info(f"{len(uploaded_files)} document(s) selected.")

        st.dataframe(
            pd.DataFrame([
                {
                    "Filename": file.name,
                    "Format": Path(file.name).suffix.lower(),
                    "Size": human_size(file.size),
                }
                for file in uploaded_files
            ]),
            use_container_width=True,
            hide_index=True,
        )

    if st.button(
        "Save and index documents",
        type="primary",
        use_container_width=True,
        disabled=not uploaded_files,
    ):
        pipeline = get_pipeline_safely()

        if pipeline is not None:
            successful = []
            failed = []

            progress = st.progress(0)
            status_box = st.empty()

            for index, uploaded_file in enumerate(uploaded_files):
                saved_path = None

                try:
                    saved_path = unique_upload_path(uploaded_file.name)

                    status_box.write(
                        f"Processing {index + 1} of {len(uploaded_files)}: "
                        f"{uploaded_file.name}"
                    )

                    with saved_path.open("wb") as output:
                        output.write(uploaded_file.getbuffer())

                    index_saved_file(pipeline, saved_path)
                    successful.append(saved_path.name)

                except Exception as exc:
                    failed.append({
                        "Document": uploaded_file.name,
                        "Error": f"{type(exc).__name__}: {exc}",
                    })

                progress.progress((index + 1) / len(uploaded_files))

            status_box.empty()

            if successful:
                st.success(
                    f"Successfully indexed {len(successful)} document(s)."
                )
                invalidate_analysis()

                with st.expander("Successfully indexed"):
                    for filename in successful:
                        st.write(f"✓ {filename}")

            if failed:
                st.error(f"{len(failed)} document(s) failed.")
                st.dataframe(
                    pd.DataFrame(failed),
                    use_container_width=True,
                    hide_index=True,
                )

    st.markdown("---")
    st.markdown("### Stored documents")

    files = list_uploaded_files()

    if files:
        files_df = pd.DataFrame([
            {
                "Filename": path.name,
                "Format": path.suffix.lower(),
                "Size": human_size(path.stat().st_size),
                "Modified": datetime.fromtimestamp(
                    path.stat().st_mtime
                ).strftime("%d %b %Y, %H:%M"),
            }
            for path in files
        ])

        st.dataframe(
            files_df,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "Download file inventory (CSV)",
            data=files_df.to_csv(index=False).encode("utf-8"),
            file_name="project_document_inventory.csv",
            mime="text/csv",
        )
    else:
        st.info("No files have been uploaded yet.")


# =========================================================
# 12. PROJECT INTELLIGENCE ASSISTANT
# =========================================================

elif selected_page == "Intelligence":
    st.markdown('<div class="section-kicker">RAG knowledge assistant</div>',
                unsafe_allow_html=True)
    st.markdown("## Ask your project documents")

    st.markdown(
        '<div class="section-description">'
        'Search the indexed project knowledge base and inspect the source '
        'passages returned by the local retrieval pipeline.'
        '</div>',
        unsafe_allow_html=True,
    )

    with st.form("project_question_form"):
        question = st.text_area(
            "Project question",
            placeholder=(
                "Example: What risks could delay the project launch?"
            ),
            height=115,
        )

        submitted = st.form_submit_button(
            "Search project knowledge",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        if not question.strip():
            st.warning("Enter a project question first.")
        else:
            pipeline = get_pipeline_safely()

            if pipeline is not None:
                try:
                    with st.spinner(
                        "Retrieving relevant project passages..."
                    ):
                        assistant = ProjectIntelligenceAssistant(
                            rag_pipeline=pipeline,
                            top_k=5,
                        )
                        result = assistant.ask(question.strip())

                    st.session_state.last_answer = result
                    st.session_state.last_query = question.strip()

                    st.session_state.search_history.append({
                        "Question": question.strip(),
                        "Time": datetime.now().strftime(
                            "%d %b %Y, %H:%M:%S"
                        ),
                    })

                except Exception as exc:
                    st.error("Project document search failed.")
                    st.code(f"{type(exc).__name__}: {exc}")

    result = st.session_state.last_answer

    if result is not None:
        st.markdown("---")
        st.markdown("### Retrieved answer")

        if isinstance(result, dict):
            st.write(result.get("answer", json.dumps(result, indent=2)))

            sources = result.get("sources", [])

            if sources:
                st.markdown("### Sources")

                st.dataframe(
                    pd.DataFrame(sources),
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.write(str(result))

        st.caption(
            "The current API-free assistant presents retrieved excerpts. "
            "Treat them as supporting evidence and verify the source text."
        )

    if st.session_state.search_history:
        st.markdown("---")
        st.markdown("### Recent questions")

        st.dataframe(
            pd.DataFrame(
                list(reversed(st.session_state.search_history[-10:]))
            ),
            use_container_width=True,
            hide_index=True,
        )


# =========================================================
# 13. RISK ANALYSIS AND PROJECT HEALTH
# =========================================================

elif selected_page == "Risk Analysis":
    st.markdown('<div class="section-kicker">Delivery monitoring</div>',
                unsafe_allow_html=True)
    st.markdown("## Project risks and health")

    st.markdown(
        '<div class="section-description">'
        'Review extracted risks, blockers, action items, project health, '
        'and delivery warning signals.'
        '</div>',
        unsafe_allow_html=True,
    )

    analysis, health, documentation = get_project_analysis()

    if analysis is None:
        st.info(
            "No project analysis is available yet. Open Overview and click "
            "'Analyze project documents' first."
        )
    else:
        forecast = analysis.get("delivery_forecast", {})
        risk_data = analysis.get("risk_analysis", {})
        blocker_data = analysis.get("blocker_analysis", {})
        action_data = analysis.get("action_analysis", {})
        scope_data = analysis.get("scope", {})

        health_score = (
            health.get("health_score", "—")
            if health else "—"
        )
        health_status = (
            health.get("status", "Not calculated")
            if health else "Not calculated"
        )

        cols = st.columns(4)
        cols[0].metric("Project health", health_score, health_status)
        cols[1].metric(
            "Risk signals",
            risk_data.get("risk_count", len(risk_data.get("risks", []))),
        )
        cols[2].metric(
            "Blockers",
            blocker_data.get(
                "blocker_count",
                len(blocker_data.get("blockers", [])),
            ),
        )
        cols[3].metric(
            "Action items",
            action_data.get(
                "action_item_count",
                len(action_data.get("action_items", [])),
            ),
        )

        st.markdown("---")
        st.markdown("### Delivery forecast")

        st.markdown(
            f"**Status:** {forecast.get('forecast_status', 'Unavailable')}"
        )

        forecast_cols = st.columns(3)
        forecast_cols[0].metric(
            "Delivery pressure",
            forecast.get("delivery_pressure_score", "—"),
            help="Heuristic indicator, not a probability of delay.",
        )
        forecast_cols[1].metric(
            "Days until planned end",
            forecast.get("days_until_planned_end", "—"),
        )
        forecast_cols[2].metric(
            "Overdue actions",
            forecast.get("overdue_action_count", 0),
        )

        for warning in forecast.get("warnings", []):
            st.warning(warning)

        if health:
            st.markdown("---")
            st.markdown("### Health score explanation")

            if health.get("explanation"):
                st.write(health["explanation"])

            if health.get("deductions"):
                st.dataframe(
                    pd.DataFrame([
                        {"Factor": key, "Impact": value}
                        for key, value in health["deductions"].items()
                    ]),
                    use_container_width=True,
                    hide_index=True,
                )

            if health.get("recommendations"):
                st.markdown("**Recommendations**")
                for recommendation in health["recommendations"]:
                    st.write(f"• {recommendation}")

        st.markdown("---")
        st.markdown("### Risk register")

        risks = risk_data.get("risks", []) or []

        if risks:
            risk_df = pd.DataFrame(risks)
            st.dataframe(
                risk_df,
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download risk register (CSV)",
                data=risk_df.to_csv(index=False).encode("utf-8"),
                file_name="project_risk_register.csv",
                mime="text/csv",
            )
        else:
            st.success("No risk signals were detected by the current rules.")

        st.markdown("---")
        st.markdown("### Blockers")

        blockers = blocker_data.get("blockers", []) or []

        if blockers:
            st.dataframe(
                pd.DataFrame(blockers),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No blockers were detected by the current rules.")

        st.markdown("---")
        st.markdown("### Action items")

        actions = action_data.get("action_items", []) or []

        if actions:
            st.dataframe(
                pd.DataFrame(actions),
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No action items were detected by the current rules.")

        st.markdown("---")
        st.markdown("### Scope and deliverables")

        scope_cols = st.columns(3)
        scope_cols[0].metric(
            "Objectives",
            scope_data.get("objective_count", 0),
        )
        scope_cols[1].metric(
            "Deliverables",
            scope_data.get("deliverable_count", 0),
        )
        scope_cols[2].metric(
            "Scope statements",
            scope_data.get("scope_item_count", 0),
        )

        with st.expander("View scope extraction details"):
            st.json(scope_data)

        st.markdown("---")
        st.markdown("### Generated project documentation")

        if documentation:
            story_df = pd.DataFrame(
                documentation.get("user_stories", [])
            )
            risk_register_df = pd.DataFrame(
                documentation.get("risk_register", [])
            )
            action_df = pd.DataFrame(
                documentation.get("action_items", [])
            )

            story_tab, register_tab, action_tab = st.tabs([
                "User Stories",
                "Risk Register",
                "Action Items",
            ])

            with story_tab:
                if not story_df.empty:
                    st.dataframe(
                        story_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download user stories (CSV)",
                        data=story_df.to_csv(index=False).encode("utf-8"),
                        file_name="project_user_stories.csv",
                        mime="text/csv",
                        key="download_user_stories",
                    )
                else:
                    st.info(
                        "No user stories could be drafted from the "
                        "detected objectives or deliverables."
                    )

            with register_tab:
                if not risk_register_df.empty:
                    st.dataframe(
                        risk_register_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download documentation risk register (CSV)",
                        data=risk_register_df.to_csv(
                            index=False
                        ).encode("utf-8"),
                        file_name="project_documentation_risk_register.csv",
                        mime="text/csv",
                        key="download_documentation_risks",
                    )
                else:
                    st.info("No risk entries are available.")

            with action_tab:
                if not action_df.empty:
                    st.dataframe(
                        action_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        "Download action items (CSV)",
                        data=action_df.to_csv(index=False).encode("utf-8"),
                        file_name="project_action_items.csv",
                        mime="text/csv",
                        key="download_action_items",
                    )
                else:
                    st.info("No action-item entries are available.")

        st.caption(
            "Risk detection, delivery forecasting, and health scoring use "
            "heuristics. Validate the results against actual project records."
        )


# =========================================================
# 14. SYSTEM DIAGNOSTICS
# =========================================================

elif selected_page == "Diagnostics":
    st.markdown('<div class="section-kicker">Environment status</div>',
                unsafe_allow_html=True)
    st.markdown("## System diagnostics")

    st.markdown(
        '<div class="section-description">'
        'Check the local workspace, dependencies, stored files, and '
        'RAG pipeline initialization.'
        '</div>',
        unsafe_allow_html=True,
    )

    folder_rows = []

    for folder in (ROOT, DATA_DIR, UPLOAD_DIR, CHROMA_DIR):
        folder_rows.append({
            "Path": str(folder),
            "Exists": folder.exists(),
            "Directory": folder.is_dir(),
        })

    st.markdown("### Workspace folders")
    st.dataframe(
        pd.DataFrame(folder_rows),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### Runtime")
    st.code(
        f"Python: {__import__('sys').version.split()[0]}\n"
        f"Project root: {ROOT}\n"
        f"Upload directory: {UPLOAD_DIR}\n"
        f"Chroma directory: {CHROMA_DIR}"
    )

    st.markdown("### Pipeline constructor")
    try:
        st.code(str(inspect.signature(RAGPipeline)))
    except Exception as exc:
        st.error(f"Could not inspect RAGPipeline: {exc}")

    if st.button(
        "Test local RAG pipeline",
        type="primary",
        use_container_width=True,
    ):
        pipeline = get_pipeline_safely()

        if pipeline is not None:
            st.success("RAGPipeline initialized successfully.")

            stats_method = getattr(
                pipeline,
                "get_dashboard_stats",
                None,
            )

            if callable(stats_method):
                try:
                    st.markdown("### Knowledge base statistics")
                    st.json(stats_method())
                except Exception as exc:
                    st.warning(
                        "The pipeline initialized, but statistics "
                        f"could not be loaded: {exc}"
                    )

    st.markdown("### Uploaded file inventory")
    files = list_uploaded_files()

    if files:
        st.dataframe(
            pd.DataFrame([
                {
                    "Filename": path.name,
                    "Size": human_size(path.stat().st_size),
                    "Extension": path.suffix.lower(),
                }
                for path in files
            ]),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No uploaded files are currently stored.")


# =========================================================
# 15. FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <div class="footer">
        AI Project Intelligence &amp; Risk Advisor
        &nbsp; • &nbsp; Infosys Springboard Internship Project
        <br>
        Local-first document intelligence and project risk analysis
    </div>
    """,
    unsafe_allow_html=True,
)
