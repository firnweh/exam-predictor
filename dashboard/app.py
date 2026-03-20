import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import get_questions_df, get_topics_hierarchy
from analysis.trend_analyzer import topic_frequency_by_year, find_hot_cold_topics, detect_cycles
from analysis.predictor_v2 import predict_topics_v2, backtest, HOLDOUT_YEARS
from analysis.predictor_v3 import predict_chapters_v3, predict_microtopics_v3, backtest_v3, backtest_single_year
from analysis.predictor import predict_topics

# Prajna model (optional — falls back to v3 if not trained)
SLM_AVAILABLE = False
try:
    from analysis.slm_model import predict_with_slm, backtest_slm
    import os as _os
    # Check if any Prajna model exists
    if _os.path.exists("models") and any(f.endswith(".pt") for f in _os.listdir("models")):
        SLM_AVAILABLE = True
except ImportError:
    pass

# Chatbot
try:
    from analysis.chatbot import PrajnaChatbot
    HAS_CHATBOT = True
except ImportError:
    HAS_CHATBOT = False
from analysis.deep_analysis import (
    get_topic_deep_dive, get_topic_tree, get_syllabus_coverage,
    get_subject_weightage_timeline, get_difficulty_evolution,
)
from data.syllabus import NEET_SYLLABUS, JEE_SYLLABUS
from data.historical_events import TIMELINE, NEET_2024_REMOVED, NEET_2024_ADDED, JEE_2024_REMOVED
from utils.pdf_generator import generate_paper_pdf

DB_PATH = "data/exam.db"

# --- Color Palette (Modern Muted) ---
C_HIGH = "#10b981"    # emerald
C_MED = "#f59e0b"     # amber
C_LOW = "#ef4444"     # red
C_SPEC = "#94a3b8"    # slate
C_BLUE = "#6366f1"    # indigo
C_PURPLE = "#a855f7"  # purple
C_TEAL = "#14b8a6"    # teal accent
C_ROSE = "#f43f5e"    # rose accent
C_SKY = "#38bdf8"     # sky accent
CONF_COLORS = {"HIGH": C_HIGH, "MEDIUM": C_MED, "LOW": C_LOW, "SPECULATIVE": C_SPEC}
SUBJ_COLORS = ["#6366f1", "#f43f5e", "#10b981", "#f59e0b", "#a855f7"]
SYL_COLORS = {"RETAINED": C_HIGH, "MODIFIED": C_MED, "NEW": C_BLUE, "REMOVED": C_LOW}

PLOT_LAYOUT = dict(
    plot_bgcolor  = "#0f0f1a",
    paper_bgcolor = "#131320",
    font = dict(family="Inter, system-ui, sans-serif", size=12, color="#8888aa"),
    margin  = dict(l=10, r=10, t=36, b=10),
    hoverlabel = dict(bgcolor="#1e1e30", font_color="#e2e8f0", font_size=12),
)
_LEGEND = dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#8888aa", size=11))
_GRID = dict(
    gridcolor     = "rgba(255,255,255,0.05)",
    zerolinecolor = "rgba(255,255,255,0.08)",
    linecolor     = "rgba(255,255,255,0.06)",
    tickfont      = dict(color="#8888aa", size=11),
)

def _style_fig(fig):
    """Apply subtle grid styling to any plotly figure."""
    fig.update_xaxes(**_GRID)
    fig.update_yaxes(**_GRID)
    return fig

_orig_plotly_chart = st.plotly_chart
def styled_plotly_chart(fig, **kwargs):
    _style_fig(fig)
    return _orig_plotly_chart(fig, **kwargs)
st.plotly_chart = styled_plotly_chart


# --- Page Config ---
st.set_page_config(page_title="PRAJNA — Deep Dive by Physics Wallah", page_icon="🧠", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  /* ── Global ── */
  .stApp { background:#0f0f1a !important; font-family:'Inter',system-ui,sans-serif; }
  .block-container { padding-top:0 !important; max-width:1480px; padding-left:1.5rem; padding-right:1.5rem; }
  [data-testid="stHeader"] { background:transparent !important; }
  [data-testid="stSidebar"] { display:none !important; }
  section[data-testid="stSidebarContent"] { display:none !important; }

  /* ── Compact topbar ── */
  .prajna-topbar {
    display:flex; align-items:center; justify-content:space-between;
    background:#0f0f1a; border-bottom:1px solid rgba(255,255,255,0.06);
    padding:0 1.5rem; height:56px;
    position:sticky; top:0; z-index:200;
    margin:0 -1.5rem 0 -1.5rem;
  }
  .prajna-topbar-left { display:flex; align-items:center; gap:12px; }
  .prajna-logo-circle {
    width:36px; height:36px; border-radius:50%; background:white;
    display:flex; align-items:center; justify-content:center;
    font-size:13px; font-weight:900; color:#0f0f1a;
    font-family:'Space Grotesk',sans-serif; flex-shrink:0;
  }
  .prajna-brand { font-size:17px; font-weight:700; color:#e2e8f0; letter-spacing:-0.3px; }
  .prajna-topbar-links { display:flex; align-items:center; gap:8px; }
  .prajna-nav-link {
    font-size:12px; font-weight:600; padding:5px 12px; border-radius:20px;
    text-decoration:none; border:1px solid; transition:opacity .2s;
  }
  .prajna-nav-link:hover { opacity:.8; }
  .prajna-nav-student  { color:#a78bfa; background:rgba(167,139,250,.12); border-color:rgba(167,139,250,.4); }
  .prajna-nav-api      { color:#34d399; background:rgba(52,211,153,.12);  border-color:rgba(52,211,153,.4);  }

  /* ── Filter bar ── */
  .filter-bar-wrap {
    background:#131320; border-bottom:1px solid rgba(255,255,255,0.06);
    padding:10px 1.5rem; margin:0 -1.5rem 1rem -1.5rem;
  }
  .filter-bar-label {
    font-size:11px; font-weight:600; color:rgba(255,255,255,0.4);
    text-transform:uppercase; letter-spacing:.5px; margin-bottom:3px;
  }

  /* ── Stat chips (DB summary bar) ── */
  .stat-bar {
    display:flex; flex-wrap:wrap; gap:8px;
    background:#131320; padding:10px 1.5rem;
    margin:0 -1.5rem .5rem -1.5rem;
    border-bottom:1px solid rgba(255,255,255,0.05);
  }
  .stat-chip {
    background:rgba(255,255,255,0.05); border:1px solid rgba(255,255,255,0.08);
    border-radius:10px; padding:7px 14px;
    display:flex; flex-direction:column; align-items:flex-start;
  }
  .stat-chip-val { font-size:18px; font-weight:800; color:white; font-family:'Space Grotesk',sans-serif; line-height:1; }
  .stat-chip-lbl { font-size:9px; font-weight:600; color:rgba(255,255,255,0.4); text-transform:uppercase; letter-spacing:.5px; margin-top:2px; }

  /* ── Section headings ── */
  .section-divider {
    font-size:15px; font-weight:700; color:#e2e8f0; letter-spacing:-0.2px;
    border-left:3px solid #6366f1; padding-left:12px;
    margin:28px 0 14px 0; display:flex; align-items:center; gap:8px;
  }
  .section-badge {
    font-size:10px; font-weight:700; padding:2px 8px; border-radius:10px;
    background:rgba(99,102,241,.15); color:#a5b4fc;
    border:1px solid rgba(99,102,241,.3);
  }

  /* ── KPI / metric cards ── */
  div[data-testid="stMetric"] {
    background:#131320 !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:14px !important; padding:18px 20px !important;
    box-shadow:0 4px 20px rgba(0,0,0,0.25);
    transition:transform .2s,box-shadow .2s;
  }
  div[data-testid="stMetric"]:hover {
    transform:translateY(-2px);
    border-color:rgba(99,102,241,.35) !important;
    box-shadow:0 8px 28px rgba(99,102,241,.12);
  }
  div[data-testid="stMetric"] label {
    font-size:10px !important; font-weight:600 !important;
    color:rgba(255,255,255,0.4) !important;
    text-transform:uppercase !important; letter-spacing:.5px !important;
    white-space:normal !important;
  }
  div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size:26px !important; font-weight:800 !important; color:#ffffff !important;
  }
  div[data-testid="stMetricDelta"] span { font-size:11px !important; }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    gap:2px; background:#131320;
    border-radius:12px; padding:4px;
    border:1px solid rgba(255,255,255,0.07);
  }
  .stTabs [data-baseweb="tab"] {
    border-radius:8px; padding:9px 16px;
    font-weight:600; font-size:13px; color:rgba(255,255,255,0.45);
    transition:all .2s;
  }
  .stTabs [data-baseweb="tab"][aria-selected="true"] {
    background:rgba(99,102,241,.18) !important;
    color:#a5b4fc !important;
    box-shadow:0 2px 8px rgba(99,102,241,.15);
    border-bottom:2px solid #6366f1 !important;
  }
  .stTabs [data-baseweb="tab-highlight"] { background:transparent !important; }

  /* ── Selectboxes / inputs ── */
  [data-testid="stSelectbox"] > div > div,
  [data-testid="stNumberInput"] > div > div > input,
  [data-testid="stTextInput"] > div > div > input {
    border-radius:8px !important;
    border-color:rgba(255,255,255,0.1) !important;
    background:#131320 !important; color:#e2e8f0 !important;
    font-size:13px !important;
  }
  [data-testid="stSelectbox"] label,
  [data-testid="stNumberInput"] label,
  [data-testid="stTextInput"] label {
    font-size:11px !important; color:rgba(255,255,255,0.4) !important;
    text-transform:uppercase; letter-spacing:.5px;
  }

  /* ── Buttons ── */
  .stButton > button {
    border-radius:8px; font-weight:600; font-size:13px;
    border:1px solid rgba(255,255,255,0.12) !important;
    background:#131320 !important; color:#e2e8f0 !important;
    transition:all .2s;
  }
  .stButton > button[kind="primary"] {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border:none !important; color:white !important;
    box-shadow:0 4px 16px rgba(99,102,241,.35);
  }
  .stButton > button[kind="primary"]:hover {
    box-shadow:0 6px 24px rgba(99,102,241,.5); transform:translateY(-1px);
  }

  /* ── Expanders ── */
  .streamlit-expanderHeader { font-weight:600; font-size:13px; color:#cbd5e1; }
  details {
    background:#131320 !important;
    border:1px solid rgba(255,255,255,0.07) !important;
    border-radius:12px !important;
  }

  /* ── DataFrames ── */
  [data-testid="stDataFrame"] {
    border-radius:12px; overflow:hidden;
    border:1px solid rgba(255,255,255,0.07);
  }
  [data-testid="stDataFrame"] * { color:#e2e8f0 !important; }
  [data-testid="stDataFrame"] th {
    background:rgba(99,102,241,.12) !important;
    color:#a5b4fc !important; font-weight:700 !important;
  }
  [data-testid="stDataFrame"] td { background:rgba(255,255,255,0.02) !important; }
  [data-testid="stDataFrame"] tr:hover td { background:rgba(99,102,241,.06) !important; }

  /* ── Plotly containers ── */
  .stPlotlyChart {
    background:#131320; border-radius:14px; padding:4px;
    border:1px solid rgba(255,255,255,0.06);
  }

  /* ── Download buttons ── */
  .stDownloadButton > button {
    border-radius:8px; font-weight:500;
    border:1px solid rgba(255,255,255,0.12) !important;
    background:#131320 !important; color:#cbd5e1 !important;
  }
  .stDownloadButton > button:hover {
    border-color:rgba(99,102,241,.5) !important; color:#a5b4fc !important;
  }

  /* ── Badges ── */
  .badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:11px; font-weight:700; letter-spacing:.3px; }
  .badge-high     { background:rgba(16,185,129,.15); color:#6ee7b7; border:1px solid rgba(16,185,129,.3); }
  .badge-medium   { background:rgba(245,158,11,.15);  color:#fcd34d; border:1px solid rgba(245,158,11,.3);  }
  .badge-low      { background:rgba(239,68,68,.15);   color:#fca5a5; border:1px solid rgba(239,68,68,.3);   }
  .badge-spec     { background:rgba(148,163,184,.1);  color:#94a3b8; border:1px solid rgba(148,163,184,.2); }
  .badge-retained { background:rgba(16,185,129,.15);  color:#6ee7b7; border:1px solid rgba(16,185,129,.3);  }
  .badge-modified { background:rgba(245,158,11,.15);  color:#fcd34d; border:1px solid rgba(245,158,11,.3);  }
  .badge-new      { background:rgba(99,102,241,.15);  color:#a5b4fc; border:1px solid rgba(99,102,241,.3);  }
  .badge-removed  { background:rgba(239,68,68,.15);   color:#fca5a5; border:1px solid rgba(239,68,68,.3);   }

  /* ── Slider ── */
  .stSlider > div > div > div > div { background:linear-gradient(90deg,#6366f1,#a855f7) !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width:5px; height:5px; }
  ::-webkit-scrollbar-track { background:transparent; }
  ::-webkit-scrollbar-thumb { background:rgba(99,102,241,.3); border-radius:3px; }
  ::-webkit-scrollbar-thumb:hover { background:rgba(99,102,241,.5); }

  /* ── Misc ── */
  .stApp, p, li, span { color:#cbd5e1; }
  h1,h2,h3,h4 { color:#f1f5f9 !important; }
  .stCaption,[data-testid="stCaptionContainer"] { color:rgba(255,255,255,.35) !important; font-size:12px; }
  [data-testid="stAlert"] { border-radius:12px; background:rgba(255,255,255,.04); border:1px solid rgba(255,255,255,.1); }
  .stSpinner > div { border-top-color:#6366f1 !important; }

  /* ── Footer ── */
  .prajna-footer {
    background:linear-gradient(135deg,#0f0f1a,#131320);
    border-top:1px solid rgba(99,102,241,.2);
    padding:28px 1.5rem; margin:48px -1.5rem 0 -1.5rem;
  }
  .team-card {
    background:#131320; border:1px solid rgba(255,255,255,.07);
    border-radius:12px; padding:14px 18px; text-align:center;
  }
  .team-name  { font-size:14px; font-weight:700; color:#e2e8f0; }
  .team-org   { font-size:10px; color:rgba(255,255,255,.4); margin:2px 0; text-transform:uppercase; letter-spacing:.5px; }
  .team-code  { font-size:11px; color:#a5b4fc; font-weight:600; }
  .team-phone { font-size:10px; color:rgba(255,255,255,.3); }

  /* ── Collapsible prediction cards ── */
  details summary::-webkit-details-marker { display:none; }
  details summary::marker { display:none; }
  details[open] .pred-chevron { transform:rotate(90deg); }
  details[open] { border-color:rgba(99,102,241,.3) !important; }

  /* Hide Streamlit chrome ── */
  #MainMenu { visibility:hidden; }
  footer { visibility:hidden; }
  [data-testid="stToolbar"]     { display:none !important; }
  [data-testid="stDeployButton"]{ display:none !important; }

  /* ── Loading overlay ── */
  [data-testid="stStatusWidget"] { display:none !important; }
  .stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4) !important;
    border-radius: 4px !important;
  }
  .stProgress > div > div {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 4px !important;
    height: 6px !important;
  }
  /* ── Suppress the grey "running" overlay on rerun ── */
  .stApp[data-teststate="running"] .main           { opacity: 1 !important; }
  .stApp[data-teststate="running"] .block-container{ opacity: 1 !important; }
  .stApp[data-teststate="running"] section         { opacity: 1 !important; }
  .stApp[data-teststate="running"] [data-testid]   { opacity: 1 !important; }
  /* Streamlit ≥ 1.29 uses a stale overlay iframe — hide it */
  iframe[title="streamlit_loading_overlay"]         { display: none !important; }
  /* Animated indigo loading bar at the very top */
  .stApp[data-teststate="running"]::before {
    content: '';
    position: fixed; top: 0; left: 0; right: 0; height: 3px; z-index: 9999;
    background: linear-gradient(90deg, #6366f1, #a855f7, #06b6d4, #6366f1);
    background-size: 200% 100%;
    animation: prajna-loading 1.2s linear infinite;
  }
  @keyframes prajna-loading {
    0%   { background-position: 100% 0; }
    100% { background-position: -100% 0; }
  }
</style>
""", unsafe_allow_html=True)

# ── Compact Topbar ───────────────────────────────────────────────────────────
st.markdown("""
<div class="prajna-topbar">
  <div class="prajna-topbar-left">
    <div class="prajna-logo-circle">PW</div>
    <span class="prajna-brand">PRAJNA Intelligence</span>
  </div>
  <div class="prajna-topbar-links">
    <a class="prajna-nav-link prajna-nav-student"
       href="http://localhost:8765/student-dashboard.html" target="_blank">Student IQ ↗</a>
    <a class="prajna-nav-link prajna-nav-api"
       href="http://localhost:8765/intelligence-dashboard.html" target="_blank">Intelligence ↗</a>
  </div>
</div>
""", unsafe_allow_html=True)

if not os.path.exists(DB_PATH):
    st.error("Database not found. Run the loader first: `python run.py`")
    st.stop()

df = get_questions_df(DB_PATH)
holdout_str = ", ".join(str(y) for y in sorted(HOLDOUT_YEARS))


# ================================================================
# FILTER BAR
# ================================================================
# ── DB stat bar ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="stat-bar">
  <div class="stat-chip"><div class="stat-chip-val">23,119</div><div class="stat-chip-lbl">Questions</div></div>
  <div class="stat-chip"><div class="stat-chip-val">292</div><div class="stat-chip-lbl">Papers</div></div>
  <div class="stat-chip"><div class="stat-chip-val">48 yrs</div><div class="stat-chip-lbl">1978–2026</div></div>
  <div class="stat-chip"><div class="stat-chip-val">36</div><div class="stat-chip-lbl">NEET Papers</div></div>
  <div class="stat-chip"><div class="stat-chip-val">178</div><div class="stat-chip-lbl">JEE Main</div></div>
  <div class="stat-chip"><div class="stat-chip-val">78</div><div class="stat-chip-lbl">JEE Adv.</div></div>
  <div class="stat-chip"><div class="stat-chip-val">755</div><div class="stat-chip-lbl">Micro-Topics</div></div>
  <div class="stat-chip"><div class="stat-chip-val">143</div><div class="stat-chip-lbl">Chapters</div></div>
  <div class="stat-chip"><div class="stat-chip-val">82.1%</div><div class="stat-chip-lbl">Backtest</div></div>
</div>
""", unsafe_allow_html=True)

# ── Filter Bar ───────────────────────────────────────────────────────────────
st.markdown('<div class="filter-bar-wrap">', unsafe_allow_html=True)
f1, f2, f3, f4, f5 = st.columns([1.3, 1.1, 0.8, 0.8, 0.7])
with f1:
    exams = ["All"] + sorted(df["exam"].unique().tolist())
    selected_exam = st.selectbox("Exam", exams, key="gx_exam", label_visibility="visible")
with f2:
    subjects = ["All"] + sorted(df["subject"].unique().tolist())
    selected_subject = st.selectbox("Subject", subjects, key="gx_subj", label_visibility="visible")
with f3:
    target_year = st.number_input("Year", value=2026, min_value=2010, max_value=2035, key="gx_yr")
with f4:
    top_n = st.selectbox("Top K", [20, 40, 60, 80, 100, 150, 200], index=1, key="gx_topn")
with f5:
    pred_level = st.selectbox("Level", ["Chapter", "Micro-Topic"], key="gx_level")
st.markdown('</div>', unsafe_allow_html=True)

# ── Refresh toast when any filter changes ────────────────────────────────────
_filter_sig = (selected_exam, selected_subject, target_year, top_n, pred_level)
if "last_filter_sig" not in st.session_state:
    st.session_state["last_filter_sig"] = _filter_sig
elif st.session_state["last_filter_sig"] != _filter_sig:
    st.toast("⚡ Refreshing predictions…", icon="🔄")
    st.session_state["last_filter_sig"] = _filter_sig

exam_filter = selected_exam if selected_exam != "All" else None

filtered = df.copy()
if selected_exam != "All":
    filtered = filtered[filtered["exam"] == selected_exam]
if selected_subject != "All":
    filtered = filtered[filtered["subject"] == selected_subject]

# Run predictions once (cached)
@st.cache_data(ttl=300)
def get_predictions_v2(db, year, exam):
    return predict_topics_v2(db, target_year=year, exam=exam)

@st.cache_data(ttl=300)
def get_predictions_v3(db, year, exam, k):
    # Each K gets its own independent reranking — not a slice of a larger set
    return predict_chapters_v3(db, target_year=year, exam=exam, top_k=k)

@st.cache_data(ttl=300)
def get_predictions_micro_v3(db, year, exam, k):
    # Each K gets its own independent reranking
    return predict_microtopics_v3(db, target_year=year, exam=exam, top_k=k)

@st.cache_data(ttl=600)
def run_backtest_single(db, test_year, exam, k, level):
    return backtest_single_year(db, test_year=test_year, exam=exam, k=k, level=level)

@st.cache_data(ttl=600)
def get_slm_predictions(db, year, exam, k, level):
    if not SLM_AVAILABLE:
        return []
    try:
        if exam is None:
            # "All" selected — merge predictions from each per-exam SLM
            all_preds = []
            for ex in ["NEET", "JEE Main", "JEE Advanced"]:
                try:
                    preds = predict_with_slm(db, target_year=year, exam=ex, top_k=k, level=level)
                    all_preds.extend(preds)
                except FileNotFoundError:
                    pass
            # Deduplicate by chapter, keep highest score
            seen = {}
            for p in sorted(all_preds, key=lambda x: x["final_score"], reverse=True):
                key = p.get("normalized_chapter", p["chapter"]).lower()
                if key not in seen:
                    seen[key] = p
            return sorted(seen.values(), key=lambda x: x["final_score"], reverse=True)[:k]
        return predict_with_slm(db, target_year=year, exam=exam, top_k=k, level=level)
    except FileNotFoundError:
        return []

# --- Load predictions for selected K (separate reranking per K) ---
# ── Loading indicator ─────────────────────────────────────────────────────────
_load_bar = st.empty()
_load_status = st.empty()
_prog = _load_bar.progress(0, text="")
_load_status.caption("⚡ PRAJNA loading predictions...")

_prog.progress(15, text="Loading micro-topic predictions...")
preds_micro = get_predictions_micro_v3(DB_PATH, target_year, exam_filter, top_n)
active_micro = [p for p in preds_micro if p["syllabus_status"] != "REMOVED"]
if selected_subject != "All":
    active_micro = [p for p in active_micro if p["subject"] == selected_subject]

_prog.progress(40, text="Loading chapter predictions...")
preds_v3 = get_predictions_v3(DB_PATH, target_year, exam_filter, top_n)
active_v3 = [p for p in preds_v3 if p["syllabus_status"] != "REMOVED"]
if selected_subject != "All":
    active_v3 = [p for p in active_v3 if p["subject"] == selected_subject]

# Prajna model predictions (if available)
if SLM_AVAILABLE:
    _prog.progress(60, text="Loading Prajna model predictions...")
    slm_preds_raw = get_slm_predictions(DB_PATH, target_year, exam_filter, top_n, "chapter")
    active_slm = [p for p in slm_preds_raw if p["syllabus_status"] != "REMOVED"]
    if selected_subject != "All":
        active_slm = [p for p in active_slm if p["subject"] == selected_subject]
else:
    active_slm = []

_prog.progress(80, text="Computing revision schedule...")
# Active list depends on selected level
pred_list = active_micro if pred_level == "Micro-Topic" else active_v3
pred_list = pred_list[:top_n]

# v2: micro-topic level (for deep analysis / lesson plan)
_prog.progress(95, text="Finalising...")
predictions_v2 = get_predictions_v2(DB_PATH, target_year, exam_filter)
active_preds_v2 = [p for p in predictions_v2 if p["syllabus_status"] != "REMOVED"]

_prog.progress(100, text="Done!")
import time as _time; _time.sleep(0.25)
_load_bar.empty()
_load_status.empty()


# --- Tabs ---
tab_main, tab_backtest, tab_deep, tab_lesson, tab_revision, tab_timeline, tab_explorer, tab_paper, tab_chat, tab_api = st.tabs([
    "📊 Predictions", "🎯 Backtest", "🔬 Topic Deep Dive", "📚 Lesson Plan", "📅 Revision Plan",
    "📈 Historical Timeline", "❓ Question Explorer", "📄 Paper Generator", "🤖 Ask PRAJNA", "🔌 API Docs",
])


# ================================================================
# TAB 1: PREDICTIONS DASHBOARD
# ================================================================
with tab_main:

    if not pred_list:
        st.warning("No predictions available for the selected filters.")
        st.stop()

    # ── SECTION 1: KPI STRIP ──
    st.markdown('<div class="section-divider">Executive Summary <span class="section-badge">LIVE</span></div>', unsafe_allow_html=True)

    all_active_for_kpi = pred_list  # already filtered and K-limited
    high_prob = sum(1 for p in all_active_for_kpi if p["appearance_probability"] >= 0.7)
    total_exp_qs = sum(p["expected_questions"] for p in pred_list)
    src_preds = preds_micro if pred_level == "Micro-Topic" else preds_v3
    new_topics = sum(1 for p in src_preds if p["syllabus_status"] == "NEW")
    removed_topics = sum(1 for p in src_preds if p["syllabus_status"] == "REMOVED")
    avg_conf = np.mean([p["confidence_score"] for p in pred_list]) if pred_list else 0
    rising = sum(1 for p in pred_list if p["trend_direction"] == "RISING")
    declining = sum(1 for p in pred_list if p["trend_direction"] == "DECLINING")
    shift = "More rising topics" if rising > declining * 1.5 else "Classic topics leading" if declining > rising * 1.5 else "Balanced mix"

    from collections import Counter
    subj_dist = Counter(p["subject"] for p in pred_list)
    subj_str = " | ".join(f"{s}: {c}" for s, c in subj_dist.most_common())

    k1, k2, k3, k4, k5 = st.columns(5)
    level_label = "Micro-Topics" if pred_level == "Micro-Topic" else "Chapters"
    k1.metric(f"High-Prob {level_label}", f"{high_prob} (>70%)")
    k2.metric(f"Expected Qs (Top {top_n})", f"~{total_exp_qs:.0f}")
    k3.metric("Syllabus Changes", f"+{new_topics} new", delta=f"-{removed_topics} removed", delta_color="inverse")
    k4.metric("Model Confidence", f"{avg_conf:.0%}")
    k5.metric("Pattern Shift", shift)

    st.caption(f"Top-{top_n} {pred_level} predictions | Reranked independently for K={top_n} | Subject split: {subj_str} | {len(filtered):,} questions in DB")

    # ── SECTION 2: RANKED PREDICTION TABLE ──
    is_micro = pred_level == "Micro-Topic"
    st.markdown(f'<div class="section-divider">Ranked {pred_level} Predictions — Top {top_n} <span class="section-badge">REAL ENGINE</span></div>', unsafe_allow_html=True)
    st.caption(f"Subject-balanced reranking for K={top_n}. {'Micro-topic + parent chapter.' if is_micro else 'Chapter-level aggregation.'}")

    # ── Detailed Prediction Cards ─────────────────────────────────────────────
    SUBJ_HEX    = {"Biology": "#22c55e", "Chemistry": "#06b6d4", "Physics": "#f59e0b", "Mathematics": "#a855f7"}
    CONF_HEX    = {"HIGH": "#10b981", "MEDIUM": "#f59e0b", "LOW": "#ef4444", "SPECULATIVE": "#94a3b8"}
    TREND_ARROW = {"RISING": "↑", "STABLE": "→", "DECLINING": "↓", "NEW": "★", "REMOVED": "✗"}

    cards_html = []
    for i, p in enumerate(pred_list, 1):
        sc       = SUBJ_HEX.get(p["subject"], "#6366f1")
        cc       = CONF_HEX.get(p["confidence"], "#6366f1")
        prob     = p["appearance_probability"]
        prob_pct = f"{prob:.0%}"
        trend    = TREND_ARROW.get(p["trend_direction"], "→")
        conf     = p["confidence"]
        name     = p.get("micro_topic", p["chapter"]) if is_micro else p["chapter"]
        exp_q    = p["expected_questions"]
        q_min    = p["expected_qs_min"]
        q_max    = p["expected_qs_max"]
        fmts     = ", ".join(p["likely_formats"][:2])
        diff     = round(p["likely_difficulty"], 1)
        last     = p["last_appeared"]
        subj     = p["subject"]
        syllabus = p.get("syllabus_status", "RETAINED")
        training = p.get("training_years", "1978–2026")
        # Normalise legacy "...-2023" range to include 2026 data
        if training and training.endswith("-2023"):
            training = training.replace("-2023", "–2026")

        # Reason pills (first 2 shown inline in summary)
        reasons = p.get("reasons", [])
        reason_pills = "".join(
            f'<span style="background:rgba(99,102,241,.12);color:#a5b4fc;border:1px solid rgba(99,102,241,.25);'
            f'border-radius:6px;padding:2px 9px;font-size:11px;margin-right:6px;white-space:nowrap">{r}</span>'
            for r in reasons[:2]
        )
        all_reasons_html = "".join(
            f'<li style="font-size:12px;color:#94a3b8;margin:3px 0;line-height:1.4">{r}</li>'
            for r in reasons
        )

        # Full-width signal bars — all non-zero signals sorted by value
        sig = p.get("signal_breakdown", {})
        sig_flat  = {k: (v if isinstance(v, (int, float)) else 0) for k, v in sig.items()}
        sig_items = sorted(sig_flat.items(), key=lambda x: x[1], reverse=True)
        sig_bars  = ""
        for sname, sval in sig_items:
            if sval <= 0:
                continue
            pct = min(int(sval * 100), 100)
            # Green for high signal, purple for lower (matches screenshot gradient)
            bar_color = f"linear-gradient(90deg,#22c55e,#10b981)" if sval >= 0.5 else "linear-gradient(90deg,#6366f1,#a855f7)"
            sig_bars += f"""
            <div style="display:flex;align-items:center;gap:10px;margin:5px 0">
              <span style="font-size:11px;color:#8888aa;min-width:110px;flex-shrink:0">{sname.replace('_',' ')}</span>
              <div style="flex:1;background:rgba(255,255,255,0.06);border-radius:3px;height:7px">
                <div style="width:{pct}%;background:{bar_color};height:7px;border-radius:3px"></div>
              </div>
              <span style="font-size:11px;color:#8888aa;min-width:32px;text-align:right;flex-shrink:0">{sval:.2f}</span>
            </div>"""

        # Probability bar width for summary row
        prob_bar_pct = min(int(prob * 100), 100)
        training_str = f"Training: {training}" if training else ""

        card = f"""
        <details style="background:#131320;border:1px solid rgba(255,255,255,0.07);border-radius:12px;
                         margin:5px 0;overflow:hidden">
          <!-- ── Collapsed summary row ── -->
          <summary style="display:flex;align-items:center;gap:0;padding:12px 16px;
                           cursor:pointer;list-style:none;user-select:none;
                           transition:background .15s"
                   onmouseover="this.style.background='rgba(255,255,255,0.025)'"
                   onmouseout="this.style.background='transparent'">
            <!-- Rank -->
            <span style="font-size:11px;color:#8888aa;min-width:28px;flex-shrink:0">#{i}</span>
            <!-- Name + subject + pills -->
            <div style="flex:1;min-width:0;padding-right:16px">
              <div style="font-size:14px;font-weight:700;color:#f1f5f9;margin-bottom:2px">{name}</div>
              <div style="font-size:11px;color:#8888aa;margin-bottom:6px">{subj}</div>
              <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:5px">{reason_pills}</div>
              <div style="font-size:11px;color:#6b7280">{fmts} · diff: {diff} · last: {last}</div>
            </div>
            <!-- Right stats -->
            <div style="display:flex;align-items:center;gap:10px;flex-shrink:0">
              <span style="font-size:13px;color:#8888aa">{trend}</span>
              <span style="background:{cc}22;color:{cc};border:1px solid {cc}44;
                            border-radius:6px;padding:3px 10px;font-size:11px;font-weight:800">{conf}</span>
              <span style="font-size:12px;color:#e2e8f0;font-weight:600;white-space:nowrap">~{exp_q:.1f}Q ({q_min}–{q_max})</span>
              <div style="width:80px;background:rgba(255,255,255,0.06);border-radius:3px;height:6px;flex-shrink:0">
                <div style="width:{prob_bar_pct}%;background:linear-gradient(90deg,#22c55e,#10b981);height:6px;border-radius:3px"></div>
              </div>
              <span style="font-size:14px;font-weight:800;color:{cc};min-width:36px;text-align:right">{prob_pct}</span>
            </div>
          </summary>

          <!-- ── Expanded body ── -->
          <div style="background:rgba(99,102,241,0.04);border-top:1px solid rgba(99,102,241,0.15);
                       padding:14px 18px 14px 44px">
            <!-- PRAJNA Model Signals header -->
            <div style="font-size:12px;font-weight:700;color:#6366f1;margin-bottom:10px">PRAJNA Model Signals:</div>
            <!-- All reasons as bullets -->
            <ul style="margin:0 0 14px 0;padding-left:18px">
              {all_reasons_html if all_reasons_html else '<li style="font-size:12px;color:#8888aa">No reasons available</li>'}
            </ul>
            <!-- Signal breakdown bars -->
            {f'<div style="font-size:11px;color:#8888aa;margin-bottom:8px">Signal Breakdown:</div>{sig_bars}' if sig_bars else ''}
            <!-- Footer -->
            <div style="font-size:11px;color:#8888aa;margin-top:12px">
              Syllabus: <b style="color:#94a3b8">{syllabus}</b>{(" · " + training_str) if training_str else ""}
            </div>
          </div>
        </details>"""
        cards_html.append(card)

    st.markdown("\n".join(cards_html), unsafe_allow_html=True)

    # Downloads
    dc1, dc2 = st.columns(2)
    dl_data = [{
        "Subject": p["subject"], "Chapter": p["chapter"], "Micro_Topic": p["micro_topic"],
        "Appearance_Prob": p["appearance_probability"],
        "Expected_Qs": p["expected_questions"],
        "Qs_Min": p["expected_qs_min"], "Qs_Max": p["expected_qs_max"],
        "Format": ", ".join(p["likely_formats"][:2]),
        "Difficulty": p["likely_difficulty"], "Trend": p["trend_direction"],
        "Syllabus": p["syllabus_status"], "Confidence": p["confidence"],
        "Confidence_Score": p["confidence_score"],
        "Appearances": p["total_appearances"], "Last_Asked": p["last_appeared"],
    } for p in preds_micro]
    dl_df = pd.DataFrame(dl_data)
    with dc1:
        st.download_button(f"Download Top {top_n} (CSV)", dl_df.head(top_n).to_csv(index=False),
                           f"top{top_n}_{target_year}.csv", "text/csv")
    with dc2:
        st.download_button(f"Download ALL {len(dl_df)} (CSV)", dl_df.to_csv(index=False),
                           f"all_{target_year}.csv", "text/csv")

    # ── SECTION 3: TOP PROBABILITY + EXPECTED QUESTIONS ──
    st.markdown(f'<div class="section-divider">{pred_level} Probability & Expected Weightage <span class="section-badge">REAL ENGINE</span></div>', unsafe_allow_html=True)

    name_key = "micro_topic" if is_micro else "chapter"
    bar_data = pd.DataFrame([{
        "Name": p[name_key],
        "Chapter": p["chapter"],
        "Probability": p["appearance_probability"],
        "Confidence": p["confidence"],
        "Expected Qs": p["expected_questions"],
        "Format": ", ".join(p["likely_formats"][:2]),
    } for p in pred_list[:15]])

    bar_col1, bar_col2 = st.columns(2)

    with bar_col1:
        fig = px.bar(
            bar_data, x="Probability", y="Name", orientation="h",
            color="Confidence", color_discrete_map=CONF_COLORS,
            text=bar_data["Probability"].apply(lambda x: f"{x:.0%}"),
            custom_data=["Chapter", "Expected Qs", "Format"],
        )
        fig.update_traces(
            textposition="outside", textfont_size=11,
            hovertemplate="<b>%{y}</b><br>Chapter: %{customdata[0]}<br>P(Appear): %{x:.0%}<br>Exp Qs: %{customdata[1]}<br>Format: %{customdata[2]}<extra></extra>",
        )
        fig.update_layout(
            **PLOT_LAYOUT, title=f"P({pred_level} Appears)",
            height=max(380, len(bar_data) * 32),
            yaxis=dict(autorange="reversed", title=""),
            xaxis=dict(title="Appearance Probability", tickformat=".0%", range=[0, 1.12]),
            legend=dict(**_LEGEND, orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with bar_col2:
        fig = px.bar(
            bar_data, x="Expected Qs", y="Name", orientation="h",
            color="Confidence", color_discrete_map=CONF_COLORS,
            text=bar_data["Expected Qs"].apply(lambda x: f"{x:.1f}"),
            custom_data=["Chapter"],
        )
        fig.update_traces(
            textposition="outside", textfont_size=11,
            hovertemplate="<b>%{y}</b><br>Chapter: %{customdata[0]}<br>Exp Qs: %{x:.1f}<extra></extra>",
        )
        fig.update_layout(
            **PLOT_LAYOUT, title="Expected Questions (if appears)",
            height=max(380, len(bar_data) * 32),
            yaxis=dict(autorange="reversed", title=""),
            xaxis=dict(title="Expected Questions"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── SECTION 4: WEIGHTAGE TREND + QUESTION TYPE MIX ──
    left_col, right_col = st.columns(2)

    with left_col:
        st.markdown('<div class="section-divider">Weightage Over Years</div>', unsafe_allow_html=True)
        wt = get_subject_weightage_timeline(DB_PATH, exam=exam_filter)
        if not wt.empty:
            subj_cols = [c for c in wt.columns if c != "year"]
            fig = px.area(wt, x="year", y=subj_cols,
                          labels={"value": "% of Paper", "variable": "Subject"},
                          color_discrete_sequence=SUBJ_COLORS)
            fig.update_layout(**PLOT_LAYOUT, height=360, yaxis_title="% of Paper",
                              legend=dict(**_LEGEND, orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        # Interactive topic/micro trend
        if is_micro:
            trend_choices = [f"{p['micro_topic']} ({p['chapter']})" for p in pred_list[:10]]
            trend_map = {f"{p['micro_topic']} ({p['chapter']})": p["micro_topic"] for p in pred_list[:10]}
            sel_trend = st.selectbox("Micro-topic weightage trend", trend_choices, key="trend_sel")
            sel_name = trend_map.get(sel_trend, "")
            topic_yr = filtered[filtered["micro_topic"] == sel_name].groupby("year").size().reset_index(name="count")
        else:
            trend_choices = [p["chapter"] for p in pred_list[:10]]
            sel_trend = st.selectbox("Chapter weightage trend", trend_choices, key="trend_sel")
            sel_name = sel_trend
            topic_yr = filtered[filtered["topic"] == sel_name].groupby("year").size().reset_index(name="count")
            if not topic_yr.empty:
                fig = px.bar(topic_yr, x="year", y="count", color_discrete_sequence=[C_BLUE])
                fig.update_layout(**PLOT_LAYOUT, height=220, xaxis_title="", yaxis_title="Questions")
                st.plotly_chart(fig, use_container_width=True)

    with right_col:
        st.markdown('<div class="section-divider">Question Type Evolution</div>', unsafe_allow_html=True)
        type_yr = filtered.groupby(["year", "question_type"]).size().reset_index(name="count")
        if not type_yr.empty:
            fig = px.bar(type_yr, x="year", y="count", color="question_type", barmode="stack",
                         color_discrete_sequence=SUBJ_COLORS,
                         labels={"count": "Questions", "question_type": "Type"})
            fig.update_layout(**PLOT_LAYOUT, height=360,
                              legend=dict(**_LEGEND, orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        # Predicted format donut
        st.markdown(f"**Predicted {target_year} format mix** (from top {top_n})")
        tcounts = {}
        for p in pred_list:
            for t in p["likely_formats"]:
                tcounts[t] = tcounts.get(t, 0) + 1
        if tcounts:
            tdf = pd.DataFrame([{"Type": k, "Count": v} for k, v in sorted(tcounts.items(), key=lambda x: -x[1])])
            fig = px.pie(tdf, values="Count", names="Type", hole=0.45,
                         color_discrete_sequence=SUBJ_COLORS)
            fig.update_layout(**PLOT_LAYOUT, height=260, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

    # ── SECTION 5: SYLLABUS CHANGE INTELLIGENCE ──
    st.markdown('<div class="section-divider">Syllabus Change Intelligence <span class="section-badge">2026</span></div>', unsafe_allow_html=True)

    # Use chapter-level v3 for syllabus section (cleaner, no duplicate chapters)
    all_active_ch = active_v3
    retained_list = [p for p in all_active_ch if p["syllabus_status"] == "RETAINED"]
    modified_list = [p for p in all_active_ch if p["syllabus_status"] == "MODIFIED"]
    new_list = [p for p in all_active_ch if p["syllabus_status"] == "NEW"]
    removed_list = [p for p in preds_v3 if p["syllabus_status"] == "REMOVED"]

    sc1, sc2, sc3 = st.columns(3)

    with sc1:
        st.markdown(f"#### Retained ({len(retained_list)})")
        st.caption("Strong historical data — PYQ highly relevant")
        for p in sorted(retained_list, key=lambda x: -x["appearance_probability"])[:8]:
            st.markdown(f'<span class="badge badge-retained">RETAINED</span> **{p["chapter"]}** — {p["appearance_probability"]:.0%}',
                        unsafe_allow_html=True)

    with sc2:
        st.markdown(f"#### Modified ({len(modified_list)})")
        st.caption("Topic scope changed — PYQ partially relevant")
        if modified_list:
            for p in sorted(modified_list, key=lambda x: -x["appearance_probability"])[:8]:
                st.markdown(f'<span class="badge badge-modified">MODIFIED</span> **{p["chapter"]}** — {p["appearance_probability"]:.0%}',
                            unsafe_allow_html=True)
        else:
            st.info("No modified topics detected for this exam.")

    with sc3:
        st.markdown(f"#### New ({len(new_list)}) / Removed ({len(removed_list)})")
        st.caption("New = estimated via proxies; Removed = zero probability")
        for p in sorted(new_list, key=lambda x: -x["appearance_probability"])[:5]:
            st.markdown(f'<span class="badge badge-new">NEW</span> **{p["chapter"]}** — {p["appearance_probability"]:.0%}',
                        unsafe_allow_html=True)
        if removed_list:
            st.markdown("---")
            for p in removed_list[:5]:
                st.markdown(f'<span class="badge badge-removed">REMOVED</span> ~~{p["chapter"]}~~',
                            unsafe_allow_html=True)

    # Syllabus status scatter map
    syl_map = []
    for p in all_active_ch[:60]:
        syl_map.append({"Topic": p["chapter"], "Subject": p["subject"],
                        "Status": p["syllabus_status"], "Probability": p["appearance_probability"]})
    if syl_map:
        sdf = pd.DataFrame(syl_map)
        fig = px.strip(sdf, x="Probability", y="Subject", color="Status",
                       color_discrete_map=SYL_COLORS, hover_name="Topic",
                       stripmode="overlay")
        fig.update_traces(marker=dict(size=8, opacity=0.7))
        fig.update_layout(**PLOT_LAYOUT, height=260, xaxis=dict(tickformat=".0%"),
                          legend=dict(**_LEGEND, orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    # ── SECTION 6: WHY THIS PREDICTION? ──
    st.markdown('<div class="section-divider">Why This Prediction? — Score Decomposition <span class="section-badge">REAL ENGINE</span></div>', unsafe_allow_html=True)

    if is_micro:
        explain_opts = [f"{p['micro_topic']} · {p['chapter']} ({p['appearance_probability']:.0%}, ~{p['expected_questions']:.0f} Qs)" for p in pred_list]
    else:
        explain_opts = [f"{p['chapter']} ({p['appearance_probability']:.0%}, ~{p['expected_questions']:.0f} Qs)" for p in pred_list]
    sel_explain = st.selectbox(f"Select a {pred_level.lower()} to see score drivers", explain_opts, key="explain_sel")

    if sel_explain:
        idx = explain_opts.index(sel_explain)
        p = pred_list[idx]

        em1, em2, em3, em4, em5 = st.columns(5)
        em1.metric("P(Appear)", f"{p['appearance_probability']:.0%}")
        em2.metric("Expected Qs", f"{p['expected_questions']:.1f} ({p['expected_qs_min']}-{p['expected_qs_max']})")
        em3.metric("Confidence", p["confidence"])
        em4.metric("Trend", p["trend_direction"])
        em5.metric("Total Appearances", p["total_appearances"])

        if p["signal_breakdown"]:
            sigs = [{"Signal": k.replace("_", " ").title(), "Value": v["value"]}
                    for k, v in p["signal_breakdown"].items()]
            sdf = pd.DataFrame(sigs).sort_values("Value", ascending=True)

            contrib_col, detail_col = st.columns([3, 2])
            with contrib_col:
                colors = ["#10b981" if c > 0.5 else "#6366f1" if c > 0.2 else "#94a3b8" for c in sdf["Value"]]
                fig = go.Figure(go.Bar(
                    y=sdf["Signal"], x=sdf["Value"], orientation="h",
                    marker_color=colors,
                    text=sdf["Value"].apply(lambda x: f"{x:.2f}"),
                    textposition="outside",
                ))
                title_name = p["micro_topic"] if is_micro else p["chapter"]
                fig.update_layout(**PLOT_LAYOUT, height=320, xaxis_title="Signal Value",
                                  yaxis=dict(autorange="reversed"),
                                  title=f"Signal Breakdown: {title_name}")
                st.plotly_chart(fig, use_container_width=True)

            with detail_col:
                st.markdown("**Reasoning:**")
                for r in p["reasons"]:
                    st.markdown(f"- {r}")
                if is_micro:
                    st.markdown(f"**Micro-Topic:** {p['micro_topic']}")
                st.markdown(f"**Chapter:** {p['chapter']}")
                st.markdown(f"**Format:** {', '.join(p['likely_formats'])}")
                st.markdown(f"**Difficulty:** {p['likely_difficulty']}")
                st.markdown(f"**Syllabus:** {p['syllabus_status']}")
                st.markdown(f"**Training data:** {p['training_years']}")

    # ── SECTION 7: CONFIDENCE & RISK SCATTER ──
    st.markdown('<div class="section-divider">Confidence vs Probability — Risk Map <span class="section-badge">LIVE</span></div>', unsafe_allow_html=True)

    risk_col, tier_col = st.columns([2, 1])

    with risk_col:
        sc_source = active_micro if is_micro else active_v3
        sc_data = [{
            "Name": p["micro_topic"] if is_micro else p["chapter"],
            "Chapter": p["chapter"],
            "Probability": p["appearance_probability"],
            "Confidence Score": p["confidence_score"],
            "Expected Qs": max(p["expected_questions"], 0.5),
            "Confidence": p["confidence"], "Subject": p["subject"]}
                   for p in sc_source[:80]]
        if sc_data:
            scdf = pd.DataFrame(sc_data)
            fig = px.scatter(scdf, x="Probability", y="Confidence Score",
                             size="Expected Qs", hover_name="Name",
                             hover_data={"Chapter": True},
                             color="Confidence", color_discrete_map=CONF_COLORS, symbol="Subject")
            fig.update_layout(**PLOT_LAYOUT, height=440,
                              xaxis=dict(title="Appearance Probability", tickformat=".0%"),
                              yaxis=dict(title="Confidence Score"))
            fig.add_hline(y=0.6, line_dash="dash", line_color=C_SPEC, opacity=0.4)
            fig.add_vline(x=0.5, line_dash="dash", line_color=C_SPEC, opacity=0.4)
            fig.add_annotation(x=0.78, y=0.88, text="HIGH PROB + HIGH CONF",
                               showarrow=False, font=dict(size=10, color=C_HIGH))
            fig.add_annotation(x=0.22, y=0.88, text="LOW PROB + HIGH CONF",
                               showarrow=False, font=dict(size=10, color=C_BLUE))
            fig.add_annotation(x=0.78, y=0.32, text="HIGH PROB + LOW CONF",
                               showarrow=False, font=dict(size=10, color=C_MED))
            fig.add_annotation(x=0.22, y=0.32, text="SPECULATIVE",
                               showarrow=False, font=dict(size=10, color=C_SPEC))
            st.plotly_chart(fig, use_container_width=True)

    with tier_col:
        st.markdown("#### Confidence Tiers")
        for tier, label, desc in [
            ("HIGH", "High", "Strongly supported by data"),
            ("MEDIUM", "Medium", "Some support, some volatility"),
            ("LOW", "Low", "Weak trend data"),
            ("SPECULATIVE", "Speculative", "Mostly syllabus-driven"),
        ]:
            tier_src = active_micro if is_micro else active_v3
            tier_items = [p for p in tier_src[:80] if p["confidence"] == tier]
            if tier_items:
                st.markdown(f"**{label}** ({len(tier_items)})")
                for tp in tier_items[:4]:
                    if is_micro:
                        st.caption(f"  {tp['micro_topic']} ({tp['chapter']}) — {tp['appearance_probability']:.0%}")
                    else:
                        st.caption(f"  {tp['chapter']} — {tp['appearance_probability']:.0%} (~{tp['expected_questions']:.0f} Qs)")

    # ── SECTION 8: PAPER BLUEPRINT SIMULATOR ──
    st.markdown('<div class="section-divider">Paper Blueprint Simulator <span class="section-badge">REAL ENGINE</span></div>', unsafe_allow_html=True)
    st.caption(f"Estimated paper structure for {selected_exam if selected_exam != 'All' else 'NEET/JEE'} {target_year}")

    bp_exam = selected_exam if selected_exam != "All" else "NEET"
    if "NEET" in bp_exam:
        sections = [
            {"Section": "A (Compulsory)", "Type": "MCQ Single Correct", "Questions": 35, "Marks/Q": 4},
            {"Section": "B (Choice)", "Type": "MCQ (pick 10/15)", "Questions": 15, "Marks/Q": 4},
        ]
        bp_subjects = ["Physics", "Chemistry", "Biology"]
    elif "Advanced" in str(bp_exam):
        sections = [
            {"Section": "Paper 1 – MCQ", "Type": "Single Correct", "Questions": 20, "Marks/Q": 3},
            {"Section": "Paper 1 – Numerical", "Type": "Numerical", "Questions": 10, "Marks/Q": 4},
            {"Section": "Paper 2 – MCQ", "Type": "Multiple Correct", "Questions": 20, "Marks/Q": 4},
            {"Section": "Paper 2 – Numerical", "Type": "Numerical", "Questions": 10, "Marks/Q": 4},
        ]
        bp_subjects = ["Physics", "Chemistry", "Mathematics"]
    else:
        sections = [
            {"Section": "A", "Type": "MCQ Single Correct", "Questions": 20, "Marks/Q": 4},
            {"Section": "B", "Type": "Numerical Value", "Questions": 10, "Marks/Q": 4},
        ]
        bp_subjects = ["Physics", "Chemistry", "Mathematics"]

    bp_left, bp_right = st.columns([1, 2])

    with bp_left:
        sec_df = pd.DataFrame(sections)
        sec_df["Total"] = sec_df["Questions"] * sec_df["Marks/Q"]
        st.dataframe(sec_df, hide_index=True, use_container_width=True)
        st.metric("Total Marks (per subject)", sec_df["Total"].sum())
        st.metric("Total Questions (per subject)", sec_df["Questions"].sum())

    with bp_right:
        bp_rows = []
        for subj in bp_subjects:
            sp = [p for p in active_v3 if p["subject"] == subj][:8]
            for r, p in enumerate(sp, 1):
                bp_rows.append({"Subject": subj, "Chapter": p["chapter"],
                                "Prob.": p["appearance_probability"],
                                "Exp. Qs": p["expected_questions"],
                                "Format": ", ".join(p["likely_formats"][:1]),
                                "Difficulty": p["likely_difficulty"]})
        if bp_rows:
            bpdf = pd.DataFrame(bp_rows)
            fig = px.bar(bpdf, x="Prob.", y="Chapter", color="Subject", orientation="h",
                         color_discrete_sequence=SUBJ_COLORS, facet_row="Subject")
            fig.update_layout(**PLOT_LAYOUT, height=max(480, len(bp_rows) * 20),
                              showlegend=False, xaxis=dict(tickformat=".0%"))
            fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
            st.plotly_chart(fig, use_container_width=True)

    # ── SECTION 9: BACKTESTING ──
    st.markdown('<div class="section-divider">Model Performance — Backtesting <span class="section-badge">23K DB</span></div>', unsafe_allow_html=True)
    st.caption("Train on years before test year, predict, compare against actual paper.")

    if st.button("Run Backtest v3 (2019–2023)", key="bt_run"):
        with st.spinner("Running v3 backtest (chapter-level, coverage metrics)..."):
            bt = backtest_v3(DB_PATH, exam=exam_filter, k=50)
            if bt:
                btdf = pd.DataFrame(bt)
                fig = px.bar(btdf, x="test_year",
                             y=["precision_at_k", "coverage_at_k", "heavy_topic_recall", "avg_subject_coverage"],
                             barmode="group",
                             color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#a855f7"],
                             labels={"value": "Score", "variable": "Metric"})
                fig.update_layout(**PLOT_LAYOUT, height=340,
                                  legend=dict(**_LEGEND, orientation="h", yanchor="bottom", y=1.02))
                st.plotly_chart(fig, use_container_width=True)

                bm1, bm2, bm3, bm4 = st.columns(4)
                bm1.metric(f"Avg Precision@{bt[0]['k']}", f"{btdf['precision_at_k'].mean():.1%}")
                bm2.metric(f"Avg Coverage@{bt[0]['k']}", f"{btdf['coverage_at_k'].mean():.1%}")
                bm3.metric("Avg Heavy-Topic Recall", f"{btdf['heavy_topic_recall'].mean():.1%}")
                bm4.metric("Avg Subject Coverage", f"{btdf['avg_subject_coverage'].mean():.1%}")

                st.caption(f"Combined Score: {btdf['combined_score'].mean():.1%} (0.35*Prec + 0.40*Cov + 0.15*Heavy + 0.10*Subj)")
                st.dataframe(btdf[["test_year", "precision_at_k", "coverage_at_k", "heavy_topic_recall",
                                   "avg_subject_coverage", "rank_correlation", "combined_score",
                                   "unique_chapters", "questions_covered", "actual_questions"]],
                             hide_index=True, use_container_width=True)
            else:
                st.warning("No backtest data for this filter.")

    # Quick PDF from predictions
    st.markdown("---")
    qp1, qp2 = st.columns([2, 1])
    with qp1:
        pred_qs = st.slider("Questions for practice paper", 10, 90, 30, key="main_qs")
    with qp2:
        if st.button("Generate Predicted-Topics PDF", key="main_pdf", type="primary"):
            if is_micro:
                top_items = [p["micro_topic"] for p in active_micro[:30]]
                pool = filtered[filtered["micro_topic"].isin(top_items)]
            else:
                top_items = [p["chapter"] for p in active_v3[:20]]
                pool = filtered[filtered["topic"].isin(top_items)]
            if not pool.empty:
                practice = pool.sample(n=min(pred_qs, len(pool)), random_state=42)
                ename = selected_exam if selected_exam != "All" else "NEET"
                pdf_bytes = generate_paper_pdf(practice, title=f"Predicted — {ename} {target_year}",
                                               exam_name=ename, include_answers=True)
                st.download_button("Download PDF", pdf_bytes, f"predicted_{target_year}.pdf", "application/pdf")


# ================================================================
# TAB 2: INTERACTIVE BACKTEST
# ================================================================
with tab_backtest:
    st.markdown('<div class="section-divider">Interactive Backtest — Select a Year <span class="section-badge">23K DB</span></div>', unsafe_allow_html=True)
    st.caption("Train on all data before the selected year. Predict that year. Match against actual paper questions.")

    bt_c1, bt_c2, bt_c3, bt_c4 = st.columns([1.5, 1, 1, 1])
    with bt_c1:
        # Usable test years: those in DB (need at least 5 years of training data before them)
        all_years_in_db = sorted(df["exam" if exam_filter is None else "exam"].unique() if False else
                                  df["year"].unique().tolist())
        usable_years = [y for y in all_years_in_db if y >= 1990 and y <= 2025]
        bt_year = st.select_slider("Test Year", options=usable_years, value=2022)
    with bt_c2:
        bt_level = st.selectbox("Prediction Level", ["chapter", "micro"], index=0,
                                 format_func=lambda x: "Chapter" if x == "chapter" else "Micro-Topic",
                                 key="bt_level")
    with bt_c3:
        bt_k = st.selectbox("K (predictions)", [20, 40, 60, 80, 100], index=2, key="bt_k")
    with bt_c4:
        st.markdown("<br>", unsafe_allow_html=True)
        run_bt = st.button("▶ Run Backtest", type="primary", key="run_bt")

    if run_bt:
        with st.spinner(f"Training on data before {bt_year}, predicting {bt_year}..."):
            bt_exam = exam_filter
            summary, actual_df = run_backtest_single(DB_PATH, bt_year, bt_exam, bt_k, bt_level)

        if summary is None:
            st.warning(f"No data found for year {bt_year} with exam filter '{bt_exam}'.")
        else:
            st.success(f"Backtest complete: predicted Top-{bt_k} {'chapters' if bt_level == 'chapter' else 'micro-topics'} for {bt_year}")

            # ── KPI Row ──
            bk1, bk2, bk3, bk4, bk5, bk6 = st.columns(6)
            bk1.metric("Combined Score", f"{summary['combined_score']:.1%}",
                       help="0.35×Precision + 0.40×Coverage + 0.15×HeavyRecall + 0.10×SubjCov")
            bk2.metric(f"Precision@{bt_k}", f"{summary['precision_at_k']:.1%}",
                       help="Predicted topics that actually appeared")
            bk3.metric(f"Coverage@{bt_k}", f"{summary['coverage_at_k']:.1%}",
                       help="Fraction of actual exam questions covered by predictions")
            bk4.metric("Heavy-Topic Recall", f"{summary['heavy_topic_recall']:.1%}",
                       help="Recall of topics with 3+ questions")
            bk5.metric("Subj. Coverage", f"{summary['avg_subject_coverage']:.1%}")
            bk6.metric("Questions Covered", f"{summary['questions_covered']}/{summary['actual_questions']}")

            # ── Subject breakdown ──
            st.markdown('<div class="section-divider">Subject-wise Coverage</div>', unsafe_allow_html=True)
            subj_data = [{"Subject": s, "Coverage": v, "Type": "Covered"}
                         for s, v in summary["subject_coverage"].items()]
            if subj_data:
                sdf = pd.DataFrame(subj_data)
                fig = px.bar(sdf, x="Subject", y="Coverage", color="Subject",
                             color_discrete_sequence=SUBJ_COLORS,
                             text=sdf["Coverage"].apply(lambda x: f"{x:.0%}"))
                fig.update_traces(textposition="outside")
                fig.update_layout(**PLOT_LAYOUT, height=280,
                                  yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                                  showlegend=False, title=f"Subject Coverage — {bt_year} Paper")
                st.plotly_chart(fig, use_container_width=True)

            # ── Hit / Miss Table ──
            st.markdown('<div class="section-divider">Topic-by-Topic Breakdown</div>', unsafe_allow_html=True)
            bd_col1, bd_col2 = st.columns(2)

            breakdown_df = pd.DataFrame(summary["topic_breakdown"])
            if not breakdown_df.empty:
                hit_df = breakdown_df[breakdown_df["status"] == "HIT"].sort_values("actual_qs", ascending=False)
                miss_df = breakdown_df[breakdown_df["status"] == "MISSED"].sort_values("actual_qs", ascending=False)

                with bd_col1:
                    st.markdown(f"#### ✅ Hits ({len(hit_df)}) — {summary['questions_covered']} Qs covered")
                    hit_display = hit_df[["topic", "subject", "actual_qs", "predicted_rank", "is_heavy"]].copy()
                    hit_display.columns = ["Topic", "Subject", "Actual Qs", "Pred. Rank", "Heavy?"]
                    st.dataframe(
                        hit_display.style.background_gradient(subset=["Actual Qs"], cmap="Greens"),
                        hide_index=True, use_container_width=True, height=400
                    )

                with bd_col2:
                    st.markdown(f"#### ❌ Missed ({len(miss_df)}) — topics we didn't predict")
                    miss_display = miss_df[["topic", "subject", "actual_qs", "is_heavy"]].copy()
                    miss_display.columns = ["Topic", "Subject", "Actual Qs", "Heavy?"]
                    st.dataframe(
                        miss_display.style.background_gradient(subset=["Actual Qs"], cmap="Reds"),
                        hide_index=True, use_container_width=True, height=400
                    )

            # ── False Positives ──
            with st.expander(f"False Positives — {summary['false_positives']} predicted but not asked"):
                if summary["fp_breakdown"]:
                    fp_df = pd.DataFrame(summary["fp_breakdown"])
                    fp_df.columns = ["Topic", "Subject", "Pred. Rank", "P(Appear)", "Confidence"]
                    st.dataframe(fp_df, hide_index=True, use_container_width=True)

            # ── Rank correlation chart ──
            st.markdown('<div class="section-divider">Prediction Quality Summary <span class="section-badge">23K DB</span></div>', unsafe_allow_html=True)
            sum_col1, sum_col2 = st.columns(2)

            with sum_col1:
                metrics_radar = {
                    "Precision": summary["precision_at_k"],
                    "Coverage": summary["coverage_at_k"],
                    "Heavy Recall": summary["heavy_topic_recall"],
                    "Subj. Balance": summary["avg_subject_coverage"],
                }
                fig = go.Figure(go.Bar(
                    x=list(metrics_radar.keys()),
                    y=list(metrics_radar.values()),
                    marker_color=["#6366f1", "#10b981", "#f59e0b", "#a855f7"],
                    text=[f"{v:.0%}" for v in metrics_radar.values()],
                    textposition="outside",
                ))
                fig.update_layout(**PLOT_LAYOUT, height=300,
                                  yaxis=dict(tickformat=".0%", range=[0, 1.15]),
                                  title=f"Model Performance — {bt_year} (K={bt_k})")
                st.plotly_chart(fig, use_container_width=True)

            with sum_col2:
                st.markdown("#### Summary")
                st.markdown(f"- **Year tested:** {bt_year}")
                st.markdown(f"- **Exam:** {bt_exam or 'All'}")
                st.markdown(f"- **Level:** {'Chapter' if bt_level == 'chapter' else 'Micro-Topic'}")
                st.markdown(f"- **K:** {bt_k} predictions")
                st.markdown(f"- **Actual paper:** {summary['actual_topics']} topics, {summary['actual_questions']} questions")
                st.markdown(f"- **Hits:** {summary['hits']} / {summary['actual_topics']} topics")
                st.markdown(f"- **Questions covered:** {summary['questions_covered']} / {summary['actual_questions']}")
                st.markdown(f"- **Heavy topics (3+ Qs) hit:** {summary['heavy_topics_hit']} / {summary['heavy_topics_total']}")
                st.markdown(f"- **Rank correlation:** {summary['rank_correlation']:.3f}")
                st.markdown(f"---")
                st.markdown(f"**Combined Score: {summary['combined_score']:.1%}**")

            # ── Multi-K comparison for same year ──
            st.markdown('<div class="section-divider">Coverage vs K — Same Year</div>', unsafe_allow_html=True)
            st.caption("How does coverage improve as you increase K?")
            if st.button("Compute Coverage@K curve", key="cov_curve"):
                with st.spinner("Computing for K = 20, 40, 60, 80, 100..."):
                    curve_rows = []
                    for ck in [20, 40, 60, 80, 100]:
                        cs, _ = run_backtest_single(DB_PATH, bt_year, bt_exam, ck, bt_level)
                        if cs:
                            curve_rows.append({"K": ck, "Coverage": cs["coverage_at_k"],
                                               "Precision": cs["precision_at_k"],
                                               "Combined": cs["combined_score"]})
                    if curve_rows:
                        cdf = pd.DataFrame(curve_rows)
                        fig = px.line(cdf, x="K", y=["Coverage", "Precision", "Combined"],
                                      markers=True, color_discrete_sequence=["#10b981", "#6366f1", "#f59e0b"])
                        fig.update_layout(**PLOT_LAYOUT, height=320,
                                          yaxis=dict(tickformat=".0%"),
                                          title=f"Coverage@K vs K — {bt_year}")
                        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Select a year and click **▶ Run Backtest** to see prediction performance against the actual paper.")

        # Preview: show multi-year trend of combined scores
        if st.checkbox("Preview: multi-year backtest (2018–2023)", key="bt_preview"):
            with st.spinner("Running across 2018–2023..."):
                bt_exam_prev = exam_filter
                prev_rows = []
                for yr in [2018, 2019, 2020, 2021, 2022, 2023]:
                    s, _ = run_backtest_single(DB_PATH, yr, bt_exam_prev, 50, "chapter")
                    if s:
                        prev_rows.append({
                            "Year": yr, "Precision": s["precision_at_k"],
                            "Coverage": s["coverage_at_k"],
                            "Heavy Recall": s["heavy_topic_recall"],
                            "Combined": s["combined_score"],
                        })
                if prev_rows:
                    prev_df = pd.DataFrame(prev_rows)
                    fig = px.line(prev_df, x="Year",
                                  y=["Precision", "Coverage", "Heavy Recall", "Combined"],
                                  markers=True,
                                  color_discrete_sequence=["#6366f1", "#10b981", "#f59e0b", "#f43f5e"])
                    fig.update_layout(**PLOT_LAYOUT, height=360,
                                      yaxis=dict(tickformat=".0%"),
                                      title="Multi-Year Model Performance (Chapter@50)")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(prev_df, hide_index=True, use_container_width=True)


# ================================================================
# TAB 3: DEEP TOPIC ANALYSIS
# ================================================================
with tab_deep:
    st.markdown('<div class="section-divider">Deep Topic Analysis <span class="section-badge">LIVE</span></div>', unsafe_allow_html=True)
    st.caption("Select a topic to see its complete history, questions, and patterns.")

    all_topics = sorted(filtered["topic"].unique().tolist())

    # Reset stale topic selection when filters change (subject/exam changed the topic list)
    if st.session_state.get("dt_topic", "") not in ([""] + all_topics):
        st.session_state["dt_topic"] = ""

    ca, cb = st.columns(2)
    with ca:
        sel_topic = st.selectbox("Chapter / Topic", [""] + all_topics, key="dt_topic")
    with cb:
        if sel_topic:
            mopts = sorted(filtered[filtered["topic"] == sel_topic]["micro_topic"].unique().tolist())
        else:
            mopts = sorted(filtered["micro_topic"].unique().tolist())
        sel_micro = st.selectbox("Micro-topic (optional)", ["All"] + mopts, key="dt_micro")

    subject_filter_deep = selected_subject if selected_subject != "All" else None
    if sel_topic:
        dive = get_topic_deep_dive(DB_PATH, sel_topic, exam=exam_filter, subject=subject_filter_deep)
        if dive:
            # ── Active filter badge ────────────────────────────────────────────
            filter_parts = []
            if exam_filter:      filter_parts.append(f"📋 {exam_filter}")
            if subject_filter_deep: filter_parts.append(f"📚 {subject_filter_deep}")
            if sel_micro and sel_micro != "All": filter_parts.append(f"🔬 {sel_micro}")
            if filter_parts:
                st.caption("Active filters: " + "  ·  ".join(filter_parts))

            # ── Derive the working question set ───────────────────────────────
            # When a micro-topic is selected, ALL charts and metrics use that
            # filtered subset — not just the questions table.
            qdf_all = dive["questions"]
            if sel_micro and sel_micro != "All":
                qdf_view = qdf_all[qdf_all["micro_topic"] == sel_micro]
            else:
                qdf_view = qdf_all

            # ── Metrics ───────────────────────────────────────────────────────
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Questions Shown", len(qdf_view),
                      delta=f"{len(qdf_all)} total" if sel_micro and sel_micro != "All" else None)
            m2.metric("First Appeared", int(qdf_view["year"].min()) if not qdf_view.empty else dive["first_year"])
            m3.metric("Last Appeared",  int(qdf_view["year"].max()) if not qdf_view.empty else dive["last_year"])
            m4.metric("Span", f"{int(qdf_view['year'].max()) - int(qdf_view['year'].min())} yrs"
                      if not qdf_view.empty and qdf_view['year'].max() != qdf_view['year'].min()
                      else f"{dive['last_year'] - dive['first_year']} yrs")

            # ── Timeline: recompute from qdf_view so micro filter applies ─────
            ydf = qdf_view.groupby("year").size().reset_index(name="count") if not qdf_view.empty else dive["year_counts"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ydf["year"], y=ydf["count"], marker_color="#818cf8", name="Questions"))
            if len(ydf) > 2:
                ma = pd.Series(ydf["count"].values).rolling(3, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=ydf["year"], y=ma, mode="lines", name="3-yr Avg",
                                         line=dict(color="#f43f5e", width=2, dash="dot")))
            chart_title = f"'{sel_micro}' in {sel_topic}" if (sel_micro and sel_micro != "All") else f"'{sel_topic}'"
            fig.update_layout(**PLOT_LAYOUT, height=320, title=f"{chart_title} — Questions per Year")
            st.plotly_chart(fig, use_container_width=True)

            dl, dr = st.columns(2)
            with dl:
                # Difficulty trend — recompute from filtered questions
                if not qdf_view.empty:
                    ddf = qdf_view.groupby("year")["difficulty"].mean().reset_index()
                    ddf.columns = ["year", "difficulty"]
                else:
                    ddf = dive["difficulty_trend"]
                if not ddf.empty and len(ddf) > 1:
                    fig = px.line(ddf, x="year", y="difficulty", markers=True,
                                  color_discrete_sequence=["#f43f5e"])
                    fig.update_layout(**PLOT_LAYOUT, height=260, title="Difficulty Trend", yaxis_range=[1, 5])
                    st.plotly_chart(fig, use_container_width=True)
            with dr:
                # Type distribution — recompute from filtered questions
                type_dist = qdf_view["question_type"].value_counts().to_dict() if not qdf_view.empty else dive["type_distribution"]
                tdf = pd.DataFrame(list(type_dist.items()), columns=["Type", "Count"])
                if not tdf.empty:
                    fig = px.pie(tdf, values="Count", names="Type", color_discrete_sequence=SUBJ_COLORS)
                    fig.update_layout(**PLOT_LAYOUT, height=260, title="Question Types")
                    st.plotly_chart(fig, use_container_width=True)

            # Subtopic breakdown (only shown when no micro-topic filter active)
            if not (sel_micro and sel_micro != "All"):
                sdf = dive["subtopic_counts"]
                if not sdf.empty:
                    fig = px.bar(sdf, x="count", y="micro_topic", orientation="h",
                                 color="avg_difficulty", color_continuous_scale="RdYlGn_r",
                                 title="Subtopic Frequency (click a bar to filter above)")
                    fig.update_layout(**PLOT_LAYOUT, height=max(280, len(sdf) * 26),
                                      yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

            # Cross-exam (always full-topic scope, useful for comparison context)
            if len(dive["exam_counts"]) > 1 and not exam_filter:
                edf = pd.DataFrame(list(dive["exam_counts"].items()), columns=["Exam", "Questions"])
                fig = px.pie(edf, values="Questions", names="Exam", title="Cross-Exam Presence")
                fig.update_layout(**PLOT_LAYOUT, height=260)
                st.plotly_chart(fig, use_container_width=True)

            # Questions list — always uses qdf_view (already micro-filtered above)
            st.markdown(f"#### Questions ({len(qdf_view)} shown)")
            for _, row in qdf_view.iterrows():
                with st.expander(f"[{row['exam']} {row['year']}] {row['micro_topic']} — Diff: {row['difficulty']}"):
                    st.markdown(f"**Q:** {row['question_text'][:500]}")
                    st.markdown(f"**A:** {row['answer']}")
                    st.markdown(f"**Type:** {row['question_type']}  |  **Shift:** {row['shift']}")
        else:
            st.info("No data found for this topic with the current filters.")
    else:
        st.info("Select a chapter above to explore its full history, difficulty trend, and question breakdown.")

        # Hot & Cold topics
        st.markdown('<div class="section-divider">Hot & Cold Topics <span class="section-badge">LIVE</span></div>', unsafe_allow_html=True)
        hc1, hc2 = st.columns(2)
        hot, cold = find_hot_cold_topics(DB_PATH, recent_years=3)
        with hc1:
            st.markdown("**Hot** (frequent recently)")
            hdata = [{"Topic": m, "Count (3 yrs)": c} for _, m, c in hot[:15]]
            if hdata:
                st.dataframe(pd.DataFrame(hdata), hide_index=True, use_container_width=True)
        with hc2:
            st.markdown("**Cold** (dormant — could return)")
            cdata = [{"Topic": m, "Gap (years)": g} for _, m, g in cold[:15]]
            if cdata:
                st.dataframe(pd.DataFrame(cdata), hide_index=True, use_container_width=True)

        # Cyclical
        cycles = detect_cycles(DB_PATH)
        if cycles:
            st.markdown('<div class="section-divider">Cyclical Topics</div>', unsafe_allow_html=True)
            cdf = pd.DataFrame(cycles)
            fig = px.scatter(cdf, x="avg_gap", y="consistency", size="estimated_cycle_years",
                             hover_name="micro_topic", color="consistency",
                             color_continuous_scale="Viridis",
                             labels={"avg_gap": "Avg Gap (years)", "consistency": "Cycle Consistency"})
            fig.update_layout(**PLOT_LAYOUT, height=380)
            st.plotly_chart(fig, use_container_width=True)


# ================================================================
# TAB 3: LESSON PLAN
# ================================================================
with tab_lesson:
    st.markdown('<div class="section-divider">Syllabus-Based Lesson Plan <span class="section-badge">AI</span></div>', unsafe_allow_html=True)
    st.caption(f"Official syllabus mapped to historical data. Priority = prediction score. Trained excluding {holdout_str}.")

    exam_ch = st.selectbox("Exam Syllabus", ["NEET", "JEE Main", "JEE Advanced"], key="lp_exam")
    syllabus = NEET_SYLLABUS if exam_ch == "NEET" else JEE_SYLLABUS
    cov_df = get_syllabus_coverage(DB_PATH, exam_ch)

    preds_v1 = predict_topics(DB_PATH, target_year=2026, exam=exam_ch if exam_ch == "NEET" else None)
    pred_scores = {p["micro_topic"]: p["score"] for p in preds_v1}

    syl_subjects = list(syllabus.keys())
    sel_sub = st.selectbox("Subject", syl_subjects, key="lp_sub")

    if sel_sub:
        chapters = syllabus[sel_sub]
        for ch_name, subtopics in chapters.items():
            ch_cov = cov_df[cov_df["chapter"] == ch_name]
            total_qs = ch_cov["questions_found"].sum()

            ch_priority = 0
            for st_name in subtopics:
                for k, v in pred_scores.items():
                    if any(w in k.lower() for w in st_name.lower().split() if len(w) > 3):
                        ch_priority = max(ch_priority, v)

            plabel = "HIGH" if ch_priority > 0.15 else "MEDIUM" if ch_priority > 0.08 else "LOW"
            pclass = "badge-high" if plabel == "HIGH" else "badge-medium" if plabel == "MEDIUM" else "badge-spec"

            with st.expander(f"**{ch_name}** — {int(total_qs)} Qs — {plabel} priority"):
                sub_data = []
                for stn in subtopics:
                    sr = ch_cov[ch_cov["subtopic"] == stn]
                    qf = int(sr["questions_found"].sum()) if not sr.empty else 0
                    last = int(sr["last_appeared"].max()) if not sr.empty and sr["last_appeared"].max() > 0 else 0
                    ylist = []
                    if not sr.empty:
                        for yl in sr["years_appeared"]:
                            if isinstance(yl, list):
                                ylist.extend(yl)
                    ystr = ", ".join(str(y) for y in sorted(set(ylist))[-5:]) if ylist else "-"
                    bs = max((v for k, v in pred_scores.items()
                              if any(w in k.lower() for w in stn.lower().split() if len(w) > 3)), default=0)
                    focus = "Must Study" if bs > 0.15 else "Important" if bs > 0.08 else "Review" if qf > 0 else "Low Yield"
                    sub_data.append({"Subtopic": stn, "Questions": qf, "Last Seen": last,
                                     "Recent Years": ystr, "Priority": focus, "Score": round(bs, 3)})

                stdf = pd.DataFrame(sub_data).sort_values("Score", ascending=False)
                st.dataframe(stdf, hide_index=True, use_container_width=True)

                if stdf["Questions"].sum() > 0:
                    fig = px.bar(stdf[stdf["Questions"] > 0], x="Questions", y="Subtopic", orientation="h",
                                 color="Score", color_continuous_scale="YlOrRd")
                    fig.update_layout(**PLOT_LAYOUT, height=max(180, len(stdf) * 22),
                                      yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)

    # Overall priority
    st.markdown('<div class="section-divider">Top 20 Focus Subtopics <span class="section-badge">AI</span></div>', unsafe_allow_html=True)
    if not cov_df.empty:
        summary = cov_df.copy()
        summary["pred_score"] = summary["subtopic"].apply(
            lambda s: max((v for k, v in pred_scores.items()
                          if any(w in k.lower() for w in s.lower().split() if len(w) > 3)), default=0))
        summary["priority"] = summary["pred_score"] * 0.6 + (summary["frequency"] / max(summary["frequency"].max(), 1)) * 0.4
        top20 = summary.sort_values("priority", ascending=False).head(20)
        fig = px.bar(top20, x="priority", y="subtopic", orientation="h", color="subject",
                     hover_data=["chapter", "questions_found", "last_appeared"],
                     color_discrete_sequence=SUBJ_COLORS)
        fig.update_layout(**PLOT_LAYOUT, height=550, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        # ── Download buttons ──
        dl_lp1, dl_lp2 = st.columns(2)
        with dl_lp1:
            dl_top20 = top20[["subtopic", "chapter", "subject", "questions_found",
                               "last_appeared", "pred_score", "priority"]].copy()
            dl_top20.columns = ["Subtopic", "Chapter", "Subject", "Questions Found",
                                 "Last Appeared", "Prediction Score", "Priority Score"]
            dl_top20["Priority Label"] = dl_top20["Priority Score"].apply(
                lambda x: "Must Study" if x > 0.15 else "Important" if x > 0.08 else "Review")
            st.download_button(
                "⬇ Download Top 20 Important Topics (CSV)",
                dl_top20.to_csv(index=False),
                f"important_topics_{exam_ch}_{target_year}.csv", "text/csv",
            )
        with dl_lp2:
            # Full lesson plan: all subtopics with priority
            full_plan = summary.sort_values(["subject", "chapter", "priority"], ascending=[True, True, False])
            full_plan = full_plan[["subtopic", "chapter", "subject", "questions_found",
                                   "last_appeared", "pred_score", "priority"]].copy()
            full_plan.columns = ["Subtopic", "Chapter", "Subject", "Questions Found",
                                  "Last Appeared", "Prediction Score", "Priority Score"]
            full_plan["Priority Label"] = full_plan["Priority Score"].apply(
                lambda x: "Must Study" if x > 0.15 else "Important" if x > 0.08 else "Review" if x > 0 else "Low Yield")
            st.download_button(
                "⬇ Download Full Lesson Plan (CSV)",
                full_plan.to_csv(index=False),
                f"lesson_plan_{exam_ch}_{target_year}.csv", "text/csv",
            )


# ================================================================
# TAB 4: REVISION PLAN
# ================================================================
with tab_revision:
    st.markdown('<div class="section-divider">📅 Prajna Revision Plan <span class="section-badge">AI</span></div>', unsafe_allow_html=True)
    st.caption(f"Personalised {target_year} revision schedule based on PRAJNA predictions — prioritised by appearance probability & confidence.")

    SUBJ_HEX_LOCAL = {"Biology": "#22c55e", "Chemistry": "#06b6d4", "Physics": "#f59e0b", "Mathematics": "#a855f7"}
    CONF_HEX_LOCAL = {"HIGH": "#10b981", "MEDIUM": "#f59e0b", "LOW": "#ef4444", "SPECULATIVE": "#94a3b8"}

    # ── Controls ──────────────────────────────────────────────────────────────
    rc1, rc2, rc3 = st.columns([1, 1, 1])
    with rc1:
        days_available = st.number_input("Days until exam", value=60, min_value=7, max_value=365, key="rev_days")
    with rc2:
        hours_per_day = st.number_input("Study hours/day", value=6, min_value=1, max_value=16, key="rev_hours")
    with rc3:
        focus_subj = st.selectbox("Focus subject", ["All"] + list(SUBJ_HEX_LOCAL.keys()), key="rev_subj")

    total_hours = days_available * hours_per_day

    # ── Filter predictions ─────────────────────────────────────────────────────
    rev_list = [p for p in pred_list if p["syllabus_status"] != "REMOVED"]
    if focus_subj != "All":
        rev_list = [p for p in rev_list if p["subject"] == focus_subj]

    if not rev_list:
        st.warning("No predictions available for selected filters.")
    else:
        # ── Allocate hours proportional to appearance probability ──────────────
        total_prob = sum(p["appearance_probability"] for p in rev_list) or 1
        phases = [
            ("🔴 Phase 1 — Must Revise", lambda p: p["appearance_probability"] >= 0.85, "#ef4444"),
            ("🟡 Phase 2 — High Priority", lambda p: 0.65 <= p["appearance_probability"] < 0.85, "#f59e0b"),
            ("🟢 Phase 3 — Good to Cover", lambda p: p["appearance_probability"] < 0.65, "#22c55e"),
        ]

        # ── Summary stat bar ──────────────────────────────────────────────────
        p1 = [p for p in rev_list if p["appearance_probability"] >= 0.85]
        p2 = [p for p in rev_list if 0.65 <= p["appearance_probability"] < 0.85]
        p3 = [p for p in rev_list if p["appearance_probability"] < 0.65]

        sm1, sm2, sm3, sm4 = st.columns(4)
        sm1.metric("Total Study Hours", f"{total_hours}h")
        sm2.metric("Must Revise", f"{len(p1)} topics", f"{len(p1)/len(rev_list):.0%} of plan")
        sm3.metric("High Priority", f"{len(p2)} topics")
        sm4.metric("Good to Cover", f"{len(p3)} topics")

        st.markdown("---")

        # ── Render each phase ────────────────────────────────────────────────
        for phase_label, phase_filter, phase_color in phases:
            phase_topics = [p for p in rev_list if phase_filter(p)]
            if not phase_topics:
                continue

            phase_prob_sum = sum(p["appearance_probability"] for p in phase_topics)
            phase_hours = round((phase_prob_sum / total_prob) * total_hours)

            st.markdown(f"""
            <div style="background:#131320;border:1px solid rgba(255,255,255,0.07);border-radius:12px;
                         padding:14px 18px;margin:12px 0;border-left:4px solid {phase_color}">
              <div style="display:flex;align-items:center;justify-content:space-between">
                <span style="font-size:15px;font-weight:700;color:#f1f5f9">{phase_label}</span>
                <span style="font-size:12px;color:{phase_color};font-weight:600">~{phase_hours}h allocated · {len(phase_topics)} topics</span>
              </div>
            </div>""", unsafe_allow_html=True)

            for p in phase_topics:
                sc = SUBJ_HEX_LOCAL.get(p["subject"], "#6366f1")
                cc = CONF_HEX_LOCAL.get(p["confidence"], "#6366f1")
                topic_prob = p["appearance_probability"]
                topic_hours = max(1, round((topic_prob / total_prob) * total_hours))
                chapter = p.get("micro_topic", p["chapter"]) if pred_level == "Micro-Topic" else p["chapter"]
                formats = ", ".join(p["likely_formats"][:2])
                reasons = p.get("reasons", [])
                reason_str = reasons[0] if reasons else ""

                days_needed = max(1, round(topic_hours / hours_per_day))

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                             border-radius:10px;padding:12px 16px;margin:4px 0 4px 16px;
                             border-left:3px solid {sc}">
                  <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                    <span style="font-size:13px;font-weight:700;color:#e2e8f0;flex:1">{chapter}</span>
                    <span style="background:{sc}22;color:{sc};border:1px solid {sc}44;
                                  border-radius:20px;padding:1px 8px;font-size:10px;font-weight:600">{p['subject']}</span>
                    <span style="background:{cc}22;color:{cc};border:1px solid {cc}44;
                                  border-radius:20px;padding:1px 8px;font-size:10px;font-weight:700">{p['confidence']}</span>
                    <span style="color:#6366f1;font-size:12px;font-weight:700">{topic_prob:.0%}</span>
                  </div>
                  <div style="display:flex;gap:20px;margin-bottom:4px">
                    <span style="font-size:11px;color:#8888aa">⏱ <b style="color:#e2e8f0">{topic_hours}h</b> ({days_needed}d)</span>
                    <span style="font-size:11px;color:#8888aa">~<b style="color:#e2e8f0">{p['expected_questions']:.0f} Qs</b> expected</span>
                    <span style="font-size:11px;color:#8888aa">Format: <b style="color:#e2e8f0">{formats}</b></span>
                    <span style="font-size:11px;color:#8888aa">Last: <b style="color:#e2e8f0">{p['last_appeared']}</b></span>
                  </div>
                  {f'<div style="font-size:11px;color:#6366f1;font-style:italic">💡 {reason_str}</div>' if reason_str else ''}
                </div>""", unsafe_allow_html=True)

        # ── Weekly schedule summary ───────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-divider">📆 Weekly Schedule Overview <span class="section-badge">AI</span></div>', unsafe_allow_html=True)

        weeks = max(1, days_available // 7)
        phase_names = ["Must Revise (Phase 1)", "High Priority (Phase 2)", "Good to Cover (Phase 3)"]
        phase_counts = [len(p1), len(p2), len(p3)]
        phase_hours_list = []
        for topics_list in [p1, p2, p3]:
            ph = sum(p["appearance_probability"] for p in topics_list)
            phase_hours_list.append(round((ph / total_prob) * total_hours))

        week_rows = []
        week_num = 1
        for pname, phours, pcount in zip(phase_names, phase_hours_list, phase_counts):
            if pcount == 0:
                continue
            phase_days = max(1, round(phours / hours_per_day))
            phase_weeks = max(1, round(phase_days / 7))
            week_rows.append({
                "Phase": pname,
                "Weeks": f"Wk {week_num}–{week_num + phase_weeks - 1}",
                "Topics": pcount,
                "Hours": f"{phours}h",
                "Daily Target": f"{hours_per_day}h/day",
            })
            week_num += phase_weeks

        if week_rows:
            wdf = pd.DataFrame(week_rows)
            st.dataframe(wdf, hide_index=True, use_container_width=True)

        st.caption(f"Total: {days_available} days · {total_hours}h · {len(rev_list)} topics across {weeks} weeks")


# ================================================================
# TAB 5: HISTORICAL TIMELINE
# ================================================================
with tab_timeline:
    st.markdown('<div class="section-divider">Historical Timeline — Syllabus, Policy & News <span class="section-badge">2026</span></div>', unsafe_allow_html=True)
    st.caption("How exam patterns, syllabus changes, and real-world events correlate with question trends.")

    tl_df = pd.DataFrame(TIMELINE, columns=["year", "category", "event", "impact"])
    cat_filter = st.multiselect("Filter categories", ["syllabus", "policy", "news"],
                                default=["syllabus", "policy", "news"], key="tl_cat")
    tl_filt = tl_df[tl_df["category"].isin(cat_filter)]

    cat_colors = {"syllabus": "#f43f5e", "policy": "#6366f1", "news": "#10b981"}
    fig = go.Figure()
    for cat in cat_filter:
        cdf = tl_filt[tl_filt["category"] == cat]
        fig.add_trace(go.Scatter(
            x=cdf["year"], y=[cat] * len(cdf), mode="markers+text",
            marker=dict(size=14, color=cat_colors.get(cat, C_SPEC)),
            text=cdf["year"].astype(str), textposition="top center",
            name=cat.title(), hovertext=cdf["event"], hoverinfo="text"))
    fig.update_layout(**PLOT_LAYOUT, height=280, title="Event Timeline")
    st.plotly_chart(fig, use_container_width=True)

    for _, row in tl_filt.sort_values("year", ascending=False).iterrows():
        icon = {"syllabus": "S", "policy": "P", "news": "N"}.get(row["category"], "?")
        with st.expander(f"[{icon}] **{row['year']}** — {row['event'][:80]}"):
            st.markdown(f"**Category:** {row['category'].title()}")
            st.markdown(f"**Event:** {row['event']}")
            st.markdown(f"**Impact:** {row['impact']}")

    # 2024 changes
    st.markdown('<div class="section-divider">2024 Syllabus Overhaul <span class="section-badge">2026</span></div>', unsafe_allow_html=True)
    tc1, tc2 = st.columns(2)
    with tc1:
        st.markdown("**Removed from NEET 2024:**")
        for item in NEET_2024_REMOVED:
            st.write(f"- ~~{item}~~")
        st.markdown("**Removed from JEE Main 2024:**")
        for item in JEE_2024_REMOVED:
            st.write(f"- ~~{item}~~")
    with tc2:
        st.markdown("**Added to NEET 2024:**")
        for item in NEET_2024_ADDED:
            st.write(f"- **{item}** (NEW)")

    # Correlation patterns
    st.markdown('<div class="section-divider">News → Question Correlation Patterns <span class="section-badge">LIVE</span></div>', unsafe_allow_html=True)
    st.markdown("""
| Pattern | Evidence | Strength |
|---------|----------|----------|
| Syllabus changes → questions | 2024: removed topics disappeared | Very Strong |
| NCERT changes lag 1 year | New textbooks 2023 → new syllabus 2024 | Strong |
| Space missions → Physics | Chandrayaan years: more gravitation Qs | Moderate |
| Health events → Biology | COVID: immunity/virology emphasis | Moderate |
| Nobel Prizes → subtle emphasis | Related areas get more Qs next year | Weak |
| Environment → Ecology Qs | Climate events → ecology spikes | Moderate |
""")


# ================================================================
# TAB 5: QUESTION EXPLORER
# ================================================================
with tab_explorer:
    st.markdown('<div class="section-divider">Question Explorer <span class="section-badge">23K DB</span></div>', unsafe_allow_html=True)

    ec1, ec2, ec3, ec4 = st.columns(4)
    with ec1:
        q_search = st.text_input("Search questions", "", key="qe_s")
    with ec2:
        q_topic = st.selectbox("Topic", ["All"] + sorted(filtered["topic"].unique().tolist()), key="qe_t")
    with ec3:
        q_diff = st.selectbox("Difficulty", ["All", 1, 2, 3, 4, 5], key="qe_d")
    with ec4:
        q_type = st.selectbox("Type", ["All"] + sorted(filtered["question_type"].unique().tolist()), key="qe_ty")

    yr = st.slider("Year range", int(filtered["year"].min()), int(filtered["year"].max()),
                    (int(filtered["year"].min()), int(filtered["year"].max())), key="qe_yr")

    edf = filtered[(filtered["year"] >= yr[0]) & (filtered["year"] <= yr[1])].copy()
    if q_search:
        edf = edf[edf["question_text"].str.contains(q_search, case=False, na=False)
                   | edf["micro_topic"].str.contains(q_search, case=False, na=False)
                   | edf["topic"].str.contains(q_search, case=False, na=False)]
    if q_topic != "All":
        edf = edf[edf["topic"] == q_topic]
    if q_diff != "All":
        edf = edf[edf["difficulty"] == q_diff]
    if q_type != "All":
        edf = edf[edf["question_type"] == q_type]

    st.write(f"Found {len(edf)} questions")

    for _, row in edf.head(50).iterrows():
        with st.expander(f"[{row['exam']} {row['year']}] {row['subject']} > {row['topic']} > {row['micro_topic']}"):
            st.markdown(f"**Q:** {row['question_text'][:500]}")
            st.markdown(f"**A:** {row['answer']}")
            dm = {1: "Easy", 2: "Easy", 3: "Moderate", 4: "Hard", 5: "Very Hard"}
            st.markdown(f"**Difficulty:** {dm.get(row['difficulty'], '?')} ({row['difficulty']}) | "
                        f"**Type:** {row['question_type']} | **Marks:** {row['marks']}")

    if len(edf) > 50:
        st.info(f"Showing first 50 of {len(edf)}. Use filters to narrow down.")

    if not edf.empty:
        h1, h2 = st.columns(2)
        with h1:
            fig = px.histogram(edf, x="difficulty", nbins=5, color="subject",
                               color_discrete_sequence=SUBJ_COLORS, title="Difficulty Distribution")
            fig.update_layout(**PLOT_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True)
        with h2:
            fig = px.histogram(edf, x="year", color="subject",
                               color_discrete_sequence=SUBJ_COLORS, title="Questions Per Year")
            fig.update_layout(**PLOT_LAYOUT, height=320)
            st.plotly_chart(fig, use_container_width=True)


# ================================================================
# TAB 6: PAPER GENERATOR
# ================================================================
with tab_paper:
    st.markdown('<div class="section-divider">Practice Paper Generator <span class="section-badge">AI</span></div>', unsafe_allow_html=True)

    pc1, pc2, pc3 = st.columns(3)
    with pc1:
        paper_mode = st.selectbox("Paper Type", [
            "High Probability Topics", "Dormant Topics (Surprise)",
            "Chapter-Wise Practice", "Full Balanced Mock", "Weak Areas Focus",
        ], key="pg_mode")
    with pc2:
        num_q = st.slider("Questions", 10, 90, 30, key="pg_n")
    with pc3:
        inc_ans = st.checkbox("Include answers", value=True, key="pg_ans")

    ch_topic = None
    if paper_mode == "Chapter-Wise Practice":
        ch_topic = st.selectbox("Chapter", sorted(filtered["topic"].unique().tolist()), key="pg_ch")

    paper_title = st.text_input("Title", f"Practice Paper — {selected_exam} {paper_mode}", key="pg_title")

    if st.button("Generate PDF", type="primary", key="pg_gen"):
        with st.spinner("Generating..."):
            if paper_mode == "High Probability Topics":
                top_ch = [p["chapter"] for p in active_v3[:20]]
                pool = filtered[filtered["topic"].isin(top_ch)]
            elif paper_mode == "Dormant Topics (Surprise)":
                _, cold = find_hot_cold_topics(DB_PATH, recent_years=5)
                pool = filtered[filtered["micro_topic"].isin([c[1] for c in cold[:20]])]
            elif paper_mode == "Chapter-Wise Practice" and ch_topic:
                pool = filtered[filtered["topic"] == ch_topic]
            elif paper_mode == "Weak Areas Focus":
                pool = filtered[filtered["difficulty"] >= 4]
            else:
                pool = filtered

            if pool.empty:
                st.warning("No questions found.")
            else:
                practice = pool.sample(n=min(num_q, len(pool)), random_state=42)
                st.subheader("Preview")
                for i, (_, row) in enumerate(practice.iterrows(), 1):
                    with st.expander(f"Q{i}. [{row['micro_topic']}] — {row['exam']} {row['year']}"):
                        st.write(row["question_text"][:500])
                        if inc_ans:
                            st.write(f"**Answer:** {row['answer']}")

                ename = selected_exam if selected_exam != "All" else "NEET"
                pdf = generate_paper_pdf(practice, title=paper_title, exam_name=ename, include_answers=inc_ans)
                st.download_button("Download PDF", pdf, "practice_paper.pdf", "application/pdf", type="primary")
                st.success(f"Paper with {len(practice)} questions!")

# ================================================================
# TAB 8: ASK PRAJNA (Chatbot)
# ================================================================
with tab_chat:
    st.markdown('<div class="section-divider">🤖 Ask PRAJNA — Exam Intelligence Chatbot <span class="section-badge">PRAJNA</span></div>', unsafe_allow_html=True)
    st.caption("Ask questions about 23,119 exam questions across 48 years. Powered by semantic search + intent detection.")

    # Model badge
    if SLM_AVAILABLE:
        st.markdown("""
        <div style="display:inline-flex;gap:8px;margin-bottom:16px;">
            <span style="background:rgba(16,185,129,0.15);color:#10b981;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:700;">
                ✓ Prajna Model Loaded (22M params)
            </span>
            <span style="background:rgba(99,102,241,0.15);color:#a5b4fc;padding:4px 12px;border-radius:12px;font-size:12px;font-weight:700;">
                Chatbot Active
            </span>
        </div>
        """, unsafe_allow_html=True)

    # Example queries
    with st.expander("💡 Example questions you can ask", expanded=False):
        st.markdown("""
        - **Counts**: "How many Physics questions in NEET?"
        - **Trends**: "What topics are trending in JEE Main?"
        - **Difficulty**: "What are the hardest topics in NEET?"
        - **Gaps**: "Which topics are overdue for NEET 2026?"
        - **Comparison**: "Compare Physics vs Chemistry in NEET"
        - **Topic details**: "Tell me about Thermodynamics in JEE"
        """)

    # Chat interface
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "details" in msg and msg["details"]:
                details = msg["details"]
                if details.get("by_year"):
                    yr_df = pd.DataFrame(list(details["by_year"].items()), columns=["Year", "Questions"])
                    fig = px.bar(yr_df, x="Year", y="Questions", color_discrete_sequence=["#6366f1"])
                    fig.update_layout(**PLOT_LAYOUT, height=250, title="Questions by Year")
                    st.plotly_chart(fig, use_container_width=True)
                if details.get("rising"):
                    rdf = pd.DataFrame(details["rising"][:8])
                    if not rdf.empty and "topic" in rdf.columns:
                        fig = px.bar(rdf, x="trend_ratio", y="topic", orientation="h",
                                     color="subject", color_discrete_sequence=SUBJ_COLORS)
                        fig.update_layout(**PLOT_LAYOUT, height=280, title="Rising Topics",
                                          yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)
                if details.get("gap_topics"):
                    gdf = pd.DataFrame(details["gap_topics"][:10])
                    if not gdf.empty:
                        fig = px.bar(gdf, x="overdue_ratio", y="topic", orientation="h",
                                     color="subject", color_discrete_sequence=SUBJ_COLORS,
                                     hover_data=["last_appeared", "gap_years", "avg_gap"])
                        fig.update_layout(**PLOT_LAYOUT, height=300, title="Overdue Topics",
                                          yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

    # Chat input
    user_query = st.chat_input("Ask about exam patterns, topics, trends...")

    if user_query:
        st.session_state.chat_messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        if HAS_CHATBOT:
            with st.chat_message("assistant"):
                with st.spinner("Searching 23,119 questions..."):
                    if "chatbot_instance" not in st.session_state:
                        st.session_state.chatbot_instance = PrajnaChatbot(DB_PATH)
                    bot = st.session_state.chatbot_instance
                    response = bot.ask(user_query)

                st.markdown(response["answer"])
                details = response.get("details", {})

                # Render charts for specific response types
                if details.get("by_year"):
                    yr_df = pd.DataFrame(list(details["by_year"].items()), columns=["Year", "Questions"])
                    fig = px.bar(yr_df, x="Year", y="Questions", color_discrete_sequence=["#6366f1"])
                    fig.update_layout(**PLOT_LAYOUT, height=250, title="Questions by Year")
                    st.plotly_chart(fig, use_container_width=True)
                if details.get("rising"):
                    rdf = pd.DataFrame(details["rising"][:8])
                    if not rdf.empty and "topic" in rdf.columns:
                        fig = px.bar(rdf, x="trend_ratio", y="topic", orientation="h",
                                     color="subject", color_discrete_sequence=SUBJ_COLORS)
                        fig.update_layout(**PLOT_LAYOUT, height=280, title="Rising Topics",
                                          yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)
                if details.get("gap_topics"):
                    gdf = pd.DataFrame(details["gap_topics"][:10])
                    if not gdf.empty:
                        fig = px.bar(gdf, x="overdue_ratio", y="topic", orientation="h",
                                     color="subject", color_discrete_sequence=SUBJ_COLORS,
                                     hover_data=["last_appeared", "gap_years", "avg_gap"])
                        fig.update_layout(**PLOT_LAYOUT, height=300, title="Overdue Topics",
                                          yaxis=dict(autorange="reversed"))
                        st.plotly_chart(fig, use_container_width=True)

                st.session_state.chat_messages.append({
                    "role": "assistant", "content": response["answer"],
                    "details": details,
                })
        else:
            with st.chat_message("assistant"):
                st.warning("Chatbot module not available. Install: `pip install sentence-transformers`")
                st.session_state.chat_messages.append({
                    "role": "assistant", "content": "Chatbot module not available.",
                    "details": {},
                })


with tab_api:
    st.markdown('<div class="section-divider">🔌 PRAJNA Intelligence API <span class="section-badge">REST</span></div>', unsafe_allow_html=True)
    st.caption("FastAPI service running on http://localhost:8001 — all endpoints return JSON. Interactive docs at /docs.")

    # ── Quick links ────────────────────────────────────────────────────────────
    ql1, ql2, ql3 = st.columns(3)
    with ql1:
        st.markdown("""
        <a href="http://localhost:8001/docs" target="_blank"
           style="display:block;background:linear-gradient(135deg,#6366f1,#8b5cf6);
                  color:white;text-align:center;padding:12px;border-radius:10px;
                  text-decoration:none;font-weight:700;font-size:13px">
          📖 Swagger UI ↗
        </a>""", unsafe_allow_html=True)
    with ql2:
        st.markdown("""
        <a href="http://localhost:8001/redoc" target="_blank"
           style="display:block;background:#131320;border:1px solid rgba(255,255,255,0.12);
                  color:#e2e8f0;text-align:center;padding:12px;border-radius:10px;
                  text-decoration:none;font-weight:700;font-size:13px">
          📄 ReDoc ↗
        </a>""", unsafe_allow_html=True)
    with ql3:
        st.markdown("""
        <a href="http://localhost:8001/openapi.json" target="_blank"
           style="display:block;background:#131320;border:1px solid rgba(255,255,255,0.12);
                  color:#e2e8f0;text-align:center;padding:12px;border-radius:10px;
                  text-decoration:none;font-weight:700;font-size:13px">
          🗂 OpenAPI JSON ↗
        </a>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Endpoint groups ────────────────────────────────────────────────────────
    API_GROUPS = [
        {
            "prefix": "/api/v1/data",
            "label": "Data Bridge",
            "color": "#6366f1",
            "desc": "Direct access to real PRAJNA engine — bypasses mock adapter",
            "endpoints": [
                ("GET", "/predict", "Run predict_chapters_v3 or predict_microtopics_v3", "?exam_type=NEET&year=2026&top_n=40&level=chapter"),
                ("GET", "/hot-cold-topics", "Hot & cold topic analysis from full history", "?exam_type=NEET&top_n=10"),
                ("GET", "/topic-deep-dive", "Full topic history, difficulty breakdown", "?topic=Thermodynamics&exam_type=NEET"),
                ("GET", "/backtest", "Train-on-past predict-one-year backtest", "?exam_type=NEET&test_year=2023&k=40"),
                ("GET", "/lesson-plan", "Syllabus-mapped lesson plan with priority", "?exam_type=NEET&year=2026"),
                ("GET", "/subject-timeline", "Subject weightage % across all years", "?exam_type=NEET"),
                ("GET", "/topics-list", "All distinct chapters/topics in the DB", "?exam_type=NEET"),
            ],
        },
        {
            "prefix": "/api/v1/predictions",
            "label": "Predictions",
            "color": "#10b981",
            "desc": "Prediction adapter endpoints (chapter and micro-topic level)",
            "endpoints": [
                ("GET", "/batch-summary", "High-level batch summary for exam/year", "?exam_type=NEET&year=2026"),
                ("GET", "/chapter-detail", "Full chapter-level prediction detail", "?exam_type=NEET&year=2026&chapter=Thermodynamics"),
                ("GET", "/subject-summary", "Subject-level prediction summary", "?exam_type=NEET&year=2026"),
                ("GET", "/microtopic-list", "Micro-topic predictions with priority scores", "?exam_type=NEET&year=2026"),
                ("GET", "/all-microtopics-ranked", "Ranked micro-topics across all subjects", "?exam_type=NEET&year=2026&top_k=50"),
            ],
        },
        {
            "prefix": "/api/v1/copilot",
            "label": "PRAJNA Copilot",
            "color": "#f59e0b",
            "desc": "Natural language question answering over exam intelligence",
            "endpoints": [
                ("POST", "/ask", "Ask a natural language question about upcoming topics", '{"question":"Which chapters are most likely in NEET 2026?","exam_type":"NEET","year":2026}'),
            ],
        },
        {
            "prefix": "/api/v1/insights",
            "label": "Insights",
            "color": "#a855f7",
            "desc": "AI-generated intelligence summaries per topic/chapter/subject",
            "endpoints": [
                ("POST", "/micro-topic", "Explain a micro-topic's exam importance", '{"topic":"Photoelectric Effect","subject":"Physics","exam_type":"NEET"}'),
                ("POST", "/chapter", "Chapter intelligence summary", '{"chapter":"Thermodynamics","exam_type":"NEET","year":2026}'),
                ("POST", "/subject-strategy", "Full subject revision strategy", '{"subject":"Chemistry","exam_type":"NEET","year":2026}'),
                ("GET", "/top-topics", "Top N predicted topics across all subjects", "?exam_type=NEET&year=2026&top_n=10"),
            ],
        },
        {
            "prefix": "/api/v1/reports",
            "label": "Reports",
            "color": "#06b6d4",
            "desc": "Revision plans and trend analysis reports",
            "endpoints": [
                ("POST", "/revision-plan", "Generate a complete revision plan", '{"exam_type":"NEET","year":2026,"top_k":40}'),
                ("GET", "/trend-analysis", "Topic trend analysis vs previous year", "?exam_type=NEET&year=2026"),
            ],
        },
    ]

    METHOD_COLORS = {"GET": "#10b981", "POST": "#f59e0b", "DELETE": "#ef4444"}

    for group in API_GROUPS:
        gc = group["color"]
        with st.expander(f"**{group['label']}** — `{group['prefix']}`", expanded=False):
            st.caption(group["desc"])
            for method, path, summary, example in group["endpoints"]:
                mc = METHOD_COLORS.get(method, "#6366f1")
                full_url = f"http://localhost:8001{group['prefix']}{path}"
                if method == "GET":
                    example_url = full_url + example
                    curl_cmd = f"curl '{example_url}'"
                else:
                    curl_cmd = f"curl -X POST '{full_url}' -H 'Content-Type: application/json' -d '{example}'"

                st.markdown(f"""
                <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                             border-radius:10px;padding:12px 16px;margin:6px 0;border-left:3px solid {mc}">
                  <div style="display:flex;align-items:center;gap:10px;margin-bottom:6px">
                    <span style="background:{mc}22;color:{mc};border:1px solid {mc}44;
                                  border-radius:6px;padding:2px 8px;font-size:11px;font-weight:800;
                                  font-family:monospace;flex-shrink:0">{method}</span>
                    <span style="font-family:monospace;font-size:13px;color:#a5b4fc;flex:1">{group['prefix']}<b style="color:#f1f5f9">{path}</b></span>
                  </div>
                  <div style="font-size:12px;color:#94a3b8;margin-bottom:8px">{summary}</div>
                  <div style="background:#0f0f1a;border-radius:6px;padding:8px 12px;
                               font-family:monospace;font-size:11px;color:#6ee7b7;
                               overflow-x:auto;white-space:nowrap">{curl_cmd}</div>
                </div>""", unsafe_allow_html=True)

    # ── Base URL note ──────────────────────────────────────────────────────────
    st.markdown("---")
    st.info("💡 **Start the API:** `cd intelligence && uvicorn services.api.main:app --port 8001 --reload`  \nAll endpoints return JSON. CORS is enabled for localhost origins.", icon="🔌")


# ================================================================
# PRAJNA FOOTER — Team Credits
# ================================================================
st.markdown("""
<div class="prajna-footer">
  <div style="text-align:center; margin-bottom:24px;">
    <div style="font-size:11px; font-weight:700; letter-spacing:2px; color:rgba(255,255,255,0.3); text-transform:uppercase; margin-bottom:8px;">Built for AI Hackathon 2026</div>
    <div style="font-size:20px; font-weight:800; color:white; font-family:'Space Grotesk',sans-serif;">PRAJNA — Deep Dive by Physics Wallah</div>
    <div style="font-size:13px; color:rgba(255,255,255,0.4); margin-top:4px;">Predictive Research & Analysis for JEE/NEET Intelligence</div>
  </div>
  <div style="display:flex; gap:16px; justify-content:center; flex-wrap:wrap; margin-bottom:24px;">
    <div class="team-card" style="min-width:220px;">
      <div class="team-name">Aman Arora</div>
      <div class="team-org">Curious Labs</div>
      <div class="team-code">PW16130</div>
      <div class="team-phone">📞 7505484120</div>
    </div>
    <div class="team-card" style="min-width:220px;">
      <div class="team-name">Himanshu Sharma</div>
      <div class="team-org">Curious Labs</div>
      <div class="team-code">PW1925</div>
      <div class="team-phone">📞 9694948108</div>
    </div>
  </div>
  <div style="text-align:center; margin-bottom:16px;">
    <a href="https://firnweh.github.io/exam-predictor/" target="_blank" rel="noopener"
       style="display:inline-flex; align-items:center; gap:8px; background:rgba(99,102,241,0.12); border:1px solid rgba(99,102,241,0.3); border-radius:20px; padding:6px 18px; color:#a5b4fc; text-decoration:none; font-size:12px; font-weight:600; letter-spacing:0.03em; transition:background 0.2s;">
      🌐 View PRAJNA Webpage — Full Technical Documentation
    </a>
  </div>
  <div style="text-align:center; font-size:11px; color:rgba(255,255,255,0.2);">
    PRAJNA v1.0 · 23,119 questions · 292 papers · 1978–2026 · Model: 3-Stage Prajna with subject-balanced reranking
  </div>
</div>
""", unsafe_allow_html=True)
