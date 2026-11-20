"""Global stylesheet: modern card-based executive look."""
from __future__ import annotations

import streamlit as st

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{
  --bg:#f6f7fb; --card:#ffffff; --ink:#0f172a; --muted:#64748b; --line:#e5e7eb;
  --brand:#4f46e5; --brand2:#7c3aed; --ok:#16a34a; --warn:#d97706; --bad:#dc2626; --info:#0284c7;
  --radius:16px; --shadow:0 1px 2px rgba(15,23,42,.04),0 8px 24px rgba(15,23,42,.06);
}
html, body, [class*="css"], .stMarkdown, .stTextInput, .stButton, button, input, textarea{font-family:'Inter',system-ui,sans-serif !important;}
.stApp{background:var(--bg);}
.block-container{padding-top:1.4rem !important; padding-bottom:3rem; max-width:1320px;}
header[data-testid="stHeader"]{background:transparent;}
#MainMenu, footer{visibility:hidden;}

/* ---------- hero ---------- */
.hero{background:linear-gradient(120deg,#1e1b4b 0%,#4338ca 55%,#7c3aed 100%);color:#fff;border-radius:20px;
  padding:26px 30px;margin-bottom:18px;box-shadow:0 12px 32px rgba(67,56,202,.28);display:flex;align-items:center;gap:20px;}
.hero-logo{font-size:38px;background:rgba(255,255,255,.14);border-radius:16px;width:68px;height:68px;display:flex;align-items:center;justify-content:center;}
.hero h1{margin:0;font-size:1.65rem;font-weight:800;letter-spacing:-.02em;color:#fff;}
.hero p{margin:4px 0 0 0;color:rgba(255,255,255,.82);font-size:.95rem;}
.hero-pill{margin-left:auto;background:rgba(255,255,255,.16);border:1px solid rgba(255,255,255,.25);padding:8px 14px;border-radius:999px;font-size:.82rem;font-weight:600;white-space:nowrap;}

/* ---------- tabs ---------- */
.stTabs [data-baseweb="tab-list"]{gap:6px;background:var(--card);padding:6px;border-radius:14px;border:1px solid var(--line);box-shadow:var(--shadow);flex-wrap:wrap;}
.stTabs [data-baseweb="tab"]{height:44px;border-radius:10px;padding:0 18px;font-weight:600;color:#475569;background:transparent;}
.stTabs [data-baseweb="tab"]:hover{background:#eef2ff;color:var(--brand);}
.stTabs [aria-selected="true"]{background:linear-gradient(120deg,var(--brand),var(--brand2)) !important;color:#fff !important;}
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"]{display:none !important;}
.stTabs [data-baseweb="tab-panel"]{padding-top:1.1rem;}
.stTabs .stTabs [data-baseweb="tab-list"]{box-shadow:none;background:#f1f5f9;}

/* ---------- cards ---------- */
.card, [class*="st-key-card"], div[data-testid="stVerticalBlockBorderWrapper"]{border-radius:var(--radius) !important;}
.card{background:var(--card);border:1px solid var(--line);box-shadow:var(--shadow);padding:20px 22px;margin-bottom:14px;}
[class*="st-key-card"]{background:var(--card);border:1px solid var(--line) !important;box-shadow:var(--shadow);padding:6px 10px;}
.card h3, .section-title{margin:0 0 4px 0;font-size:1.02rem;font-weight:700;color:var(--ink);letter-spacing:-.01em;}
.section-sub{color:var(--muted);font-size:.86rem;margin-bottom:12px;}
.muted{color:var(--muted);}

/* ---------- KPI ---------- */
.kpi{background:var(--card);border:1px solid var(--line);border-radius:var(--radius);box-shadow:var(--shadow);padding:16px 18px;display:flex;gap:14px;align-items:center;height:100%;min-height:94px;}
.kpi-icon{width:46px;height:46px;border-radius:12px;display:flex;align-items:center;justify-content:center;font-size:21px;flex:none;}
.kpi-blue .kpi-icon{background:#eef2ff}.kpi-green .kpi-icon{background:#dcfce7}.kpi-amber .kpi-icon{background:#fef3c7}
.kpi-red .kpi-icon{background:#fee2e2}.kpi-violet .kpi-icon{background:#f3e8ff}.kpi-slate .kpi-icon{background:#f1f5f9}
.kpi-label{font-size:.74rem;font-weight:600;color:var(--muted);text-transform:uppercase;letter-spacing:.06em;}
.kpi-value{font-size:1.65rem;font-weight:800;color:var(--ink);line-height:1.15;letter-spacing:-.02em;}
.kpi-sub{font-size:.78rem;color:var(--muted);margin-top:2px;}

/* ---------- badges ---------- */
.badge{display:inline-block;padding:2px 10px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.02em;white-space:nowrap;}
.b-high,.b-critical,.b-error,.b-open{background:#fee2e2;color:#b91c1c}
.b-medium,.b-warn{background:#fef3c7;color:#b45309}
.b-low,.b-info{background:#dbeafe;color:#1d4ed8}
.b-ok,.b-success,.b-resolved{background:#dcfce7;color:#15803d}
.b-neutral,.b-idle{background:#f1f5f9;color:#475569}
.b-brand{background:#eef2ff;color:#4338ca}

/* ---------- list rows ---------- */
.row-item{display:flex;gap:12px;align-items:flex-start;padding:11px 0;border-bottom:1px solid #f1f5f9;}
.row-item:last-child{border-bottom:none;}
.row-main{flex:1;min-width:0;font-size:.9rem;color:var(--ink);line-height:1.45;}
.row-meta{font-size:.76rem;color:var(--muted);margin-top:3px;}
.empty{border:1.5px dashed #cbd5e1;border-radius:var(--radius);padding:26px;text-align:center;background:#fbfcfe;color:var(--muted);}
.empty h4{margin:6px 0 4px 0;color:var(--ink);font-size:1.05rem;}
.empty .big{font-size:34px}
.steps{display:flex;gap:12px;flex-wrap:wrap;margin-top:14px;justify-content:center;}
.step{background:#fff;border:1px solid var(--line);border-radius:12px;padding:10px 14px;font-size:.84rem;text-align:left;max-width:260px;}
.step b{color:var(--brand)}

/* ---------- gauge / bars ---------- */
.gauge-wrap{text-align:center}
.gauge-label{font-weight:800;font-size:1.05rem;margin-top:-6px}
.dim{margin:12px 0}
.dim-top{display:flex;justify-content:space-between;font-size:.86rem;font-weight:600;margin-bottom:5px}
.bar{height:10px;border-radius:999px;background:#eef2f7;overflow:hidden}
.bar > span{display:block;height:100%;border-radius:999px}
.dim-note{font-size:.76rem;color:var(--muted);margin-top:4px}

/* ---------- heat map ---------- */
.heat{border-collapse:separate;border-spacing:4px;margin:auto;font-size:.78rem}
.heat td{width:46px;height:38px;text-align:center;border-radius:8px;font-weight:700;color:#0f172a}
.heat th{font-weight:600;color:var(--muted);font-size:.7rem}

/* ---------- chat ---------- */
.stChatMessage{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:12px 16px;}
.cite-chip{display:inline-block;background:#eef2ff;color:#4338ca;border-radius:6px;padding:1px 7px;font-size:.74rem;font-weight:700;margin-right:6px}
.src{border-left:3px solid var(--brand);background:#f8fafc;padding:8px 12px;border-radius:0 10px 10px 0;margin:8px 0;font-size:.82rem;color:#334155;white-space:pre-wrap}

/* ---------- controls ---------- */
.stButton > button, .stDownloadButton > button{border-radius:10px;font-weight:600;border:1px solid var(--line);padding:.5rem 1rem;}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"]{background:linear-gradient(120deg,var(--brand),var(--brand2));border:none;color:#fff;}
.stButton > button:hover{border-color:var(--brand);color:var(--brand);}
.stButton > button[kind="primary"]:hover{color:#fff;filter:brightness(1.06);}
[data-testid="stFileUploaderDropzone"]{border-radius:14px;border:1.5px dashed #a5b4fc;background:#f8f9ff;}
.stDataFrame{border-radius:12px;overflow:hidden;border:1px solid var(--line);}
[data-testid="stExpander"]{border-radius:12px !important;border:1px solid var(--line) !important;background:var(--card);}
@media (max-width: 780px){ .hero{flex-direction:column;align-items:flex-start} .hero-pill{margin-left:0} }

/* ---------- sidebar navigation ---------- */
section[data-testid="stSidebar"]{background:linear-gradient(185deg,#1e1b4b 0%,#312e81 55%,#4338ca 100%);border-right:none;}
section[data-testid="stSidebar"] > div{padding-top:1rem;}
section[data-testid="stSidebar"] *{color:#e0e7ff;}
.side-brand{display:flex;gap:12px;align-items:center;margin:4px 0 18px 2px;}
.side-logo{font-size:26px;background:rgba(255,255,255,.14);border-radius:12px;width:46px;height:46px;display:flex;align-items:center;justify-content:center;}
.side-title{font-weight:800;font-size:1.0rem;line-height:1.15;color:#fff !important;letter-spacing:-.01em;}
.side-sub{font-size:.72rem;color:#a5b4fc !important;}
.side-label{font-size:.68rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;color:#a5b4fc !important;margin:6px 0 8px 4px;}
section[data-testid="stSidebar"] div[role="radiogroup"]{gap:6px;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label{background:rgba(255,255,255,.07);border:1px solid rgba(255,255,255,.08);border-radius:12px;padding:11px 14px;width:100%;margin:0;cursor:pointer;transition:all .15s;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:hover{background:rgba(255,255,255,.16);}
section[data-testid="stSidebar"] div[role="radiogroup"] > label > div:first-child{display:none;}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked){background:#ffffff;border-color:#fff;box-shadow:0 6px 18px rgba(0,0,0,.25);}
section[data-testid="stSidebar"] div[role="radiogroup"] > label:has(input:checked) *{color:#312e81 !important;font-weight:700;}
section[data-testid="stSidebar"] div[role="radiogroup"] p{font-size:.93rem;margin:0;}
.side-card{background:rgba(255,255,255,.09);border:1px solid rgba(255,255,255,.14);border-radius:14px;padding:12px 14px;margin-top:18px;font-size:.8rem;}
.side-card b{color:#fff !important;}
.side-stat{display:flex;justify-content:space-between;margin:3px 0;}

/* ---------- in-page section pills (horizontal radios) ---------- */
[data-testid="stMain"] div[role="radiogroup"]{gap:8px;flex-wrap:wrap;}
[data-testid="stMain"] div[role="radiogroup"] > label{background:#fff;border:1px solid var(--line);border-radius:999px;padding:7px 16px;margin:0;box-shadow:0 1px 2px rgba(15,23,42,.04);cursor:pointer;}
[data-testid="stMain"] div[role="radiogroup"] > label > div:first-child{display:none;}
[data-testid="stMain"] div[role="radiogroup"] > label:hover{border-color:var(--brand);}
[data-testid="stMain"] div[role="radiogroup"] > label:has(input:checked){background:linear-gradient(120deg,var(--brand),var(--brand2));border-color:transparent;}
[data-testid="stMain"] div[role="radiogroup"] > label:has(input:checked) *{color:#fff !important;}
[data-testid="stMain"] div[role="radiogroup"] p{font-size:.86rem;font-weight:600;margin:0;}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
