"""UI smoke test: every page renders with and without documents, and incremental upload works.

Uses Streamlit's AppTest when Streamlit is installed; otherwise falls back to a lightweight stub so the
test still runs in minimal environments.
"""
import glob
import os
import sys
from unittest.mock import MagicMock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _install_stub():
    class SS(dict):
        def __getattr__(self, k):
            try:
                return self[k]
            except KeyError:
                raise AttributeError(k)

        def __setattr__(self, k, v):
            self[k] = v

    st = MagicMock(name="st")
    st.session_state = SS()
    st.columns.side_effect = lambda spec, **k: [MagicMock() for _ in range(spec if isinstance(spec, int) else len(spec))]
    st.button.return_value = False
    st.text_input.return_value = ""
    st.chat_input.return_value = None
    st.file_uploader.return_value = []
    st.multiselect.side_effect = lambda label, options=None, default=None, **k: (default or [])
    st.radio.side_effect = lambda label, options, **k: st.session_state.get(k.get("key"), options[0])
    st.secrets = MagicMock()
    st.secrets.get.return_value = None
    sys.modules["streamlit"] = st
    return st


def test_all_pages_render_empty_and_populated(sample_files):
    try:
        import streamlit  # noqa: F401
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.join(ROOT, "app.py"), default_timeout=60).run()
        assert not at.exception
        return
    except ImportError:
        pass
    st = _install_stub()
    sys.path.insert(0, ROOT)
    import ui.state as S
    import ui.views as V
    from ui.nav import PAGES

    class F:
        def __init__(self, n, d):
            self.name, self._d, self.size = n, d, len(d)

        def getvalue(self):
            return self._d

    S.init_state()
    ss = st.session_state
    views = [V.render_dashboard, V.render_assistant, V.render_knowledge_base, V.render_agent_center, V.render_generated_docs, V.render_health]
    for fn in views:                                   # empty state
        fn()
    # incremental ingestion through the uploader
    ss[f"uploader_{ss.uploader_key}"] = [F(n, d) for n, d in sample_files.items()]
    S._auto_ingest()
    assert len(ss.kb.documents) == 4 and ss.health.has_data and ss.change_log
    ss[f"uploader_{ss.uploader_key}"] = [F("extra.txt", b"Blocked: the build server is down and the team is stuck waiting.")]
    S._auto_ingest()
    assert len(ss.kb.documents) == 5 and len(ss.change_log) == 2
    for fn in views:                                   # populated state, every agent view section
        for section in ("🎯 Scope & Deliverables", "⚠️ Risks & Forecast", "🚧 Blockers & Actions", "📝 Documentation", "✔️ Validation"):
            ss["agent_view"] = section
            if fn is V.render_agent_center:
                fn()
        for section in ("📖 User Stories", "🛡️ Risk Register", "✅ Action Items", "📄 Project Report", "⬇️ Export"):
            ss["docs_view"] = section
            if fn is V.render_generated_docs:
                fn()
        fn()
    st.chat_input.return_value = "What are our biggest risks?"
    V.render_assistant()
    assert len(ss.chat) == 2 and ss.chat[1]["sources"]
    assert len(PAGES) == 6
