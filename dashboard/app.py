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
from analysis.predictor_v3 import predict_chapters_v3, predict_microtopics_v3, backtest_v3
from analysis.predictor import predict_topics
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
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", size=12, color="#334155"),
    margin=dict(l=10, r=10, t=40, b=10),
)
_GRID = dict(gridcolor="rgba(148,163,184,0.15)", zerolinecolor="rgba(148,163,184,0.2)")

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
st.set_page_config(page_title="Exam Predictor", page_icon="📊", layout="wide", initial_sidebar_state="collapsed")

# --- Custom CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Global ── */
    .stApp {
        background: linear-gradient(135deg, #f0f4ff 0%, #faf5ff 50%, #ecfdf5 100%);
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }
    [data-testid="stHeader"] { background: transparent; }
    .block-container { padding-top: 1rem; max-width: 1440px; }

    /* ── Hero title area ── */
    .hero-title {
        font-size: 28px; font-weight: 800; letter-spacing: -0.5px;
        background: linear-gradient(135deg, #6366f1, #a855f7, #ec4899);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        background-clip: text; margin-bottom: 2px;
    }
    .hero-subtitle {
        font-size: 14px; color: #64748b; font-weight: 400; margin-bottom: 20px;
    }

    /* ── Section dividers ── */
    .section-divider {
        font-size: 16px; font-weight: 700; color: #1e293b;
        letter-spacing: -0.3px;
        border-left: 3px solid; padding-left: 12px;
        border-image: linear-gradient(to bottom, #6366f1, #a855f7) 1;
        margin: 32px 0 16px 0;
    }

    /* ── Glass metric cards ── */
    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(12px); -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.8);
        border-radius: 16px; padding: 18px 20px;
        box-shadow: 0 4px 16px rgba(99, 102, 241, 0.06), 0 1px 3px rgba(0,0,0,0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(99, 102, 241, 0.12), 0 2px 6px rgba(0,0,0,0.06);
    }
    div[data-testid="stMetric"] label {
        font-size: 11px; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.3px;
        white-space: normal !important; line-height: 1.3;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 26px; font-weight: 800; color: #1e293b;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        font-size: 11px;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(8px);
        border-radius: 12px; padding: 4px;
        border: 1px solid rgba(226, 232, 240, 0.6);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 10px; padding: 10px 20px;
        font-weight: 600; font-size: 13px;
        color: #64748b; transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: white !important;
        color: #6366f1 !important;
        box-shadow: 0 2px 8px rgba(99, 102, 241, 0.12);
    }

    /* ── Selectboxes / inputs ── */
    [data-testid="stSelectbox"] > div > div,
    [data-testid="stNumberInput"] > div > div > input,
    [data-testid="stTextInput"] > div > div > input {
        border-radius: 10px !important;
        border-color: #e2e8f0 !important;
        background: rgba(255,255,255,0.8) !important;
    }

    /* ── Buttons ── */
    .stButton > button {
        border-radius: 10px; font-weight: 600; font-size: 13px;
        border: 1px solid #e2e8f0; padding: 8px 20px;
        transition: all 0.2s ease;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
        border: none !important; color: white !important;
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.3);
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
        transform: translateY(-1px);
    }

    /* ── Expander ── */
    .streamlit-expanderHeader {
        font-weight: 600; font-size: 14px; color: #334155;
        border-radius: 10px;
    }
    details {
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(226, 232, 240, 0.6) !important;
        border-radius: 12px !important;
    }

    /* ── Dataframe ── */
    [data-testid="stDataFrame"] {
        border-radius: 12px; overflow: hidden;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.04);
        border: 1px solid rgba(226, 232, 240, 0.6);
    }

    /* ── Download buttons ── */
    .stDownloadButton > button {
        border-radius: 10px; font-weight: 500;
        border: 1px solid #e2e8f0 !important;
        background: rgba(255,255,255,0.8) !important;
        transition: all 0.2s;
    }
    .stDownloadButton > button:hover {
        background: white !important;
        border-color: #6366f1 !important;
        color: #6366f1 !important;
    }

    /* ── Badge styles ── */
    .badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 11px; font-weight: 700; letter-spacing: 0.3px;
    }
    .badge-high { background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; }
    .badge-medium { background: linear-gradient(135deg, #fef3c7, #fde68a); color: #92400e; }
    .badge-low { background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b; }
    .badge-spec { background: linear-gradient(135deg, #f1f5f9, #e2e8f0); color: #475569; }
    .badge-retained { background: linear-gradient(135deg, #d1fae5, #a7f3d0); color: #065f46; }
    .badge-modified { background: linear-gradient(135deg, #fef3c7, #fde68a); color: #92400e; }
    .badge-new { background: linear-gradient(135deg, #e0e7ff, #c7d2fe); color: #3730a3; }
    .badge-removed { background: linear-gradient(135deg, #fee2e2, #fecaca); color: #991b1b; }

    /* ── Plotly chart containers ── */
    .stPlotlyChart {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(8px);
        border-radius: 12px; padding: 4px;
        border: 1px solid rgba(226, 232, 240, 0.4);
    }

    /* ── Slider ── */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #6366f1, #a855f7) !important;
    }

    /* ── Captions ── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #94a3b8 !important; font-size: 12px;
    }

    /* ── Info / Warning boxes ── */
    [data-testid="stAlert"] {
        border-radius: 12px; border: none;
        background: rgba(255, 255, 255, 0.6);
        backdrop-filter: blur(8px);
    }

    /* ── Spinner ── */
    .stSpinner > div { border-top-color: #6366f1 !important; }

    /* ── Scrollbar ── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: transparent; }
    ::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

    /* Hide hamburger & footer */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

if not os.path.exists(DB_PATH):
    st.error("Database not found. Run the loader first: `python run.py`")
    st.stop()

df = get_questions_df(DB_PATH)
holdout_str = ", ".join(str(y) for y in sorted(HOLDOUT_YEARS))


# ================================================================
# FILTER BAR
# ================================================================
st.markdown('<div class="hero-title">Exam Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">What\'s likely to come, why the model thinks so, and how preparation should shift.</div>', unsafe_allow_html=True)

f1, f2, f3, f4, f5 = st.columns([1.2, 1, 1, 0.8, 0.8])
with f1:
    exams = ["All"] + sorted(df["exam"].unique().tolist())
    selected_exam = st.selectbox("Exam", exams, key="gx_exam")
with f2:
    subjects = ["All"] + sorted(df["subject"].unique().tolist())
    selected_subject = st.selectbox("Subject", subjects, key="gx_subj")
with f3:
    target_year = st.number_input("Predict for", value=2026, min_value=2024, max_value=2030, key="gx_yr")
with f4:
    top_n = st.selectbox("Show top", [10, 15, 20, 30, 50, 100], index=2, key="gx_topn")
with f5:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(f"Trained excluding {holdout_str}")

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
    return predict_chapters_v3(db, target_year=year, exam=exam, top_k=k)

@st.cache_data(ttl=300)
def get_predictions_micro_v3(db, year, exam, k):
    return predict_microtopics_v3(db, target_year=year, exam=exam, top_k=k)

# v3 micro-topic level (primary for Predictions tab)
preds_micro = get_predictions_micro_v3(DB_PATH, target_year, exam_filter, max(top_n * 3, 150))
active_micro = [p for p in preds_micro if p["syllabus_status"] != "REMOVED"]
if selected_subject != "All":
    active_micro = [p for p in active_micro if p["subject"] == selected_subject]
pred_list = active_micro[:top_n]

# v3 chapter-level (for blueprint simulator + backtest)
preds_v3 = get_predictions_v3(DB_PATH, target_year, exam_filter, max(top_n, 50))
active_v3 = [p for p in preds_v3 if p["syllabus_status"] != "REMOVED"]
if selected_subject != "All":
    active_v3 = [p for p in active_v3 if p["subject"] == selected_subject]

# v2: micro-topic level (for deep analysis / lesson plan)
predictions_v2 = get_predictions_v2(DB_PATH, target_year, exam_filter)
active_preds_v2 = [p for p in predictions_v2 if p["syllabus_status"] != "REMOVED"]


# --- Tabs ---
tab_main, tab_deep, tab_lesson, tab_timeline, tab_explorer, tab_paper = st.tabs([
    "Predictions", "Topic Deep Dive", "Lesson Plan",
    "Historical Timeline", "Question Explorer", "Paper Generator",
])


# ================================================================
# TAB 1: PREDICTIONS DASHBOARD
# ================================================================
with tab_main:

    if not pred_list:
        st.warning("No predictions available for the selected filters.")
        st.stop()

    # ── SECTION 1: KPI STRIP ──
    st.markdown('<div class="section-divider">Executive Summary</div>', unsafe_allow_html=True)

    all_active_micro = [p for p in preds_micro if p["syllabus_status"] != "REMOVED"]
    if selected_subject != "All":
        all_active_micro = [p for p in all_active_micro if p["subject"] == selected_subject]

    high_prob = sum(1 for p in all_active_micro if p["appearance_probability"] >= 0.7)
    total_exp_qs = sum(p["expected_questions"] for p in pred_list)
    new_topics = sum(1 for p in preds_micro if p["syllabus_status"] == "NEW")
    removed_topics = sum(1 for p in preds_micro if p["syllabus_status"] == "REMOVED")
    avg_conf = np.mean([p["confidence_score"] for p in pred_list]) if pred_list else 0
    rising = sum(1 for p in all_active_micro[:50] if p["trend_direction"] == "RISING")
    declining = sum(1 for p in all_active_micro[:50] if p["trend_direction"] == "DECLINING")
    shift = "More rising topics" if rising > declining * 1.5 else "Classic topics leading" if declining > rising * 1.5 else "Balanced mix"

    from collections import Counter
    subj_dist = Counter(p["subject"] for p in pred_list)
    subj_str = " | ".join(f"{s}: {c}" for s, c in subj_dist.most_common())

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("High-Prob Micro-Topics", f"{high_prob} (>70%)")
    k2.metric(f"Expected Qs (Top {top_n})", f"~{total_exp_qs:.0f}")
    k3.metric("Syllabus Changes", f"+{new_topics} new", delta=f"-{removed_topics} removed", delta_color="inverse")
    k4.metric("Model Confidence", f"{avg_conf:.0%}")
    k5.metric("Pattern Shift", shift)

    st.caption(f"{len(all_active_micro)} unique micro-topics | {len(pred_list)} in shortlist | Subject split: {subj_str} | {len(filtered):,} questions in DB")

    # ── SECTION 2: MAIN PREDICTION TABLE (Micro-Topic-Level v3) ──
    st.markdown('<div class="section-divider">Ranked Micro-Topic Predictions</div>', unsafe_allow_html=True)
    st.caption("Each row = one unique micro-topic. Subject-balanced ranking. Parent chapter shown for context.")

    TREND_ICONS = {"RISING": "↑ Rising", "STABLE": "→ Stable", "DECLINING": "↓ Declining", "NEW": "★ New", "REMOVED": "✗ Removed"}
    TREND_COLORS = {"↑ Rising": "#10b981", "→ Stable": "#6366f1", "↓ Declining": "#f43f5e", "★ New": "#a855f7", "✗ Removed": "#94a3b8"}

    table_rows = []
    for i, p in enumerate(pred_list, 1):
        table_rows.append({
            "#": i,
            "Subject": p["subject"],
            "Micro-Topic": p["micro_topic"],
            "Chapter": p["chapter"],
            "P(Appear)": p["appearance_probability"],
            "Exp. Qs": p["expected_questions"],
            "Range": f"{p['expected_qs_min']}–{p['expected_qs_max']}",
            "Trend": TREND_ICONS.get(p["trend_direction"], "?"),
            "Last Year": int(p["last_appeared"]),
            "Format": ", ".join(p["likely_formats"][:2]),
            "Diff.": round(p["likely_difficulty"], 1),
            "Syllabus": p["syllabus_status"],
            "Confidence": p["confidence"],
        })

    tbl = pd.DataFrame(table_rows)
    st.dataframe(
        tbl.style
        .background_gradient(subset=["P(Appear)"], cmap="RdYlGn", vmin=0, vmax=1)
        .background_gradient(subset=["Exp. Qs"], cmap="PuBu", vmin=0, vmax=8)
        .format({"P(Appear)": "{:.0%}", "Exp. Qs": "{:.1f}", "Diff.": "{:.1f}"}),
        hide_index=True,
        use_container_width=True,
        height=min(700, len(table_rows) * 37 + 38),
    )

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

    # ── SECTION 3: TOP MICRO-TOPIC PROBABILITY + EXPECTED QUESTIONS ──
    st.markdown('<div class="section-divider">Micro-Topic Probability & Expected Weightage</div>', unsafe_allow_html=True)

    bar_data = pd.DataFrame([{
        "Micro-Topic": p["micro_topic"],
        "Chapter": p["chapter"],
        "Probability": p["appearance_probability"],
        "Confidence": p["confidence"],
        "Expected Qs": p["expected_questions"],
        "Format": ", ".join(p["likely_formats"][:2]),
    } for p in pred_list[:15]])

    bar_col1, bar_col2 = st.columns(2)

    with bar_col1:
        fig = px.bar(
            bar_data, x="Probability", y="Micro-Topic", orientation="h",
            color="Confidence", color_discrete_map=CONF_COLORS,
            text=bar_data["Probability"].apply(lambda x: f"{x:.0%}"),
            custom_data=["Chapter", "Expected Qs", "Format"],
        )
        fig.update_traces(
            textposition="outside", textfont_size=11,
            hovertemplate="<b>%{y}</b><br>Chapter: %{customdata[0]}<br>P(Appear): %{x:.0%}<br>Exp Qs: %{customdata[1]}<br>Format: %{customdata[2]}<extra></extra>",
        )
        fig.update_layout(
            **PLOT_LAYOUT, title="P(Micro-Topic Appears)",
            height=max(380, len(bar_data) * 32),
            yaxis=dict(autorange="reversed", title=""),
            xaxis=dict(title="Appearance Probability", tickformat=".0%", range=[0, 1.12]),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

    with bar_col2:
        fig = px.bar(
            bar_data, x="Expected Qs", y="Micro-Topic", orientation="h",
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
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
            st.plotly_chart(fig, use_container_width=True)

        # Interactive micro-topic trend
        trend_choices = [f"{p['micro_topic']} ({p['chapter']})" for p in pred_list[:10]]
        trend_map = {f"{p['micro_topic']} ({p['chapter']})": p["micro_topic"] for p in pred_list[:10]}
        if trend_choices:
            sel_trend = st.selectbox("Micro-topic weightage trend", trend_choices, key="trend_sel")
            sel_micro_name = trend_map.get(sel_trend, "")
            topic_yr = filtered[filtered["micro_topic"] == sel_micro_name].groupby("year").size().reset_index(name="count")
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
                              legend=dict(orientation="h", yanchor="bottom", y=1.02))
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
    st.markdown('<div class="section-divider">Syllabus Change Intelligence</div>', unsafe_allow_html=True)

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
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

    # ── SECTION 6: WHY THIS PREDICTION? ──
    st.markdown('<div class="section-divider">Why This Prediction? — Score Decomposition</div>', unsafe_allow_html=True)

    explain_opts = [f"{p['micro_topic']} · {p['chapter']} ({p['appearance_probability']:.0%}, ~{p['expected_questions']:.0f} Qs)" for p in pred_list]
    sel_explain = st.selectbox("Select a micro-topic to see score drivers", explain_opts, key="explain_sel")

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
                fig.update_layout(**PLOT_LAYOUT, height=320, xaxis_title="Signal Value",
                                  yaxis=dict(autorange="reversed"),
                                  title=f"Signal Breakdown: {p['micro_topic']}")
                st.plotly_chart(fig, use_container_width=True)

            with detail_col:
                st.markdown("**Reasoning:**")
                for r in p["reasons"]:
                    st.markdown(f"- {r}")
                st.markdown(f"**Micro-Topic:** {p['micro_topic']}")
                st.markdown(f"**Chapter:** {p['chapter']}")
                st.markdown(f"**Format:** {', '.join(p['likely_formats'])}")
                st.markdown(f"**Difficulty:** {p['likely_difficulty']}")
                st.markdown(f"**Syllabus:** {p['syllabus_status']}")
                st.markdown(f"**Training data:** {p['training_years']}")

    # ── SECTION 7: CONFIDENCE & RISK SCATTER ──
    st.markdown('<div class="section-divider">Confidence vs Probability — Risk Map</div>', unsafe_allow_html=True)

    risk_col, tier_col = st.columns([2, 1])

    with risk_col:
        sc_data = [{"Micro-Topic": p["micro_topic"], "Chapter": p["chapter"],
                    "Probability": p["appearance_probability"],
                    "Confidence Score": p["confidence_score"],
                    "Expected Qs": max(p["expected_questions"], 0.5),
                    "Confidence": p["confidence"], "Subject": p["subject"]}
                   for p in active_micro[:80]]
        if sc_data:
            scdf = pd.DataFrame(sc_data)
            fig = px.scatter(scdf, x="Probability", y="Confidence Score",
                             size="Expected Qs", hover_name="Micro-Topic",
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
            tier_items = [p for p in active_micro[:80] if p["confidence"] == tier]
            if tier_items:
                st.markdown(f"**{label}** ({len(tier_items)})")
                for tp in tier_items[:4]:
                    st.caption(f"  {tp['micro_topic']} ({tp['chapter']}) — {tp['appearance_probability']:.0%} (~{tp['expected_questions']:.0f} Qs)")

    # ── SECTION 8: PAPER BLUEPRINT SIMULATOR ──
    st.markdown('<div class="section-divider">Paper Blueprint Simulator</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-divider">Model Performance — Backtesting</div>', unsafe_allow_html=True)
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
                                  legend=dict(orientation="h", yanchor="bottom", y=1.02))
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
            top_micros = [p["micro_topic"] for p in active_micro[:30]]
            pool = filtered[filtered["micro_topic"].isin(top_micros)]
            if not pool.empty:
                practice = pool.sample(n=min(pred_qs, len(pool)), random_state=42)
                ename = selected_exam if selected_exam != "All" else "NEET"
                pdf_bytes = generate_paper_pdf(practice, title=f"Predicted — {ename} {target_year}",
                                               exam_name=ename, include_answers=True)
                st.download_button("Download PDF", pdf_bytes, f"predicted_{target_year}.pdf", "application/pdf")


# ================================================================
# TAB 2: DEEP TOPIC ANALYSIS
# ================================================================
with tab_deep:
    st.markdown('<div class="section-divider">Deep Topic Analysis</div>', unsafe_allow_html=True)
    st.caption("Select a topic to see its complete history, questions, and patterns.")

    all_topics = sorted(filtered["topic"].unique().tolist())
    ca, cb = st.columns(2)
    with ca:
        sel_topic = st.selectbox("Chapter / Topic", [""] + all_topics, key="dt_topic")
    with cb:
        if sel_topic:
            mopts = sorted(filtered[filtered["topic"] == sel_topic]["micro_topic"].unique().tolist())
        else:
            mopts = sorted(filtered["micro_topic"].unique().tolist())
        sel_micro = st.selectbox("Micro-topic (optional)", ["All"] + mopts, key="dt_micro")

    if sel_topic:
        dive = get_topic_deep_dive(DB_PATH, sel_topic, exam=exam_filter)
        if dive:
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Questions", dive["total_questions"])
            m2.metric("First Appeared", dive["first_year"])
            m3.metric("Last Appeared", dive["last_year"])
            m4.metric("Span", f"{dive['last_year'] - dive['first_year']} years")

            # Timeline bar + moving avg
            ydf = dive["year_counts"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=ydf["year"], y=ydf["count"], marker_color="#818cf8", name="Questions"))
            if len(ydf) > 2:
                ma = pd.Series(ydf["count"].values).rolling(3, min_periods=1).mean()
                fig.add_trace(go.Scatter(x=ydf["year"], y=ma, mode="lines", name="3-yr Avg",
                                         line=dict(color="#f43f5e", width=2, dash="dot")))
            fig.update_layout(**PLOT_LAYOUT, height=320, title=f"'{sel_topic}' — Questions per Year")
            st.plotly_chart(fig, use_container_width=True)

            dl, dr = st.columns(2)
            with dl:
                ddf = dive["difficulty_trend"]
                if not ddf.empty and len(ddf) > 1:
                    fig = px.line(ddf, x="year", y="difficulty", markers=True,
                                  color_discrete_sequence=["#f43f5e"])
                    fig.update_layout(**PLOT_LAYOUT, height=260, title="Difficulty Trend", yaxis_range=[1, 5])
                    st.plotly_chart(fig, use_container_width=True)
            with dr:
                tdf = pd.DataFrame(list(dive["type_distribution"].items()), columns=["Type", "Count"])
                if not tdf.empty:
                    fig = px.pie(tdf, values="Count", names="Type", color_discrete_sequence=SUBJ_COLORS)
                    fig.update_layout(**PLOT_LAYOUT, height=260, title="Question Types")
                    st.plotly_chart(fig, use_container_width=True)

            # Subtopic breakdown
            sdf = dive["subtopic_counts"]
            if not sdf.empty:
                fig = px.bar(sdf, x="count", y="micro_topic", orientation="h",
                             color="avg_difficulty", color_continuous_scale="RdYlGn_r",
                             title="Subtopic Frequency")
                fig.update_layout(**PLOT_LAYOUT, height=max(280, len(sdf) * 26),
                                  yaxis=dict(autorange="reversed"))
                st.plotly_chart(fig, use_container_width=True)

            # Cross-exam
            if len(dive["exam_counts"]) > 1:
                edf = pd.DataFrame(list(dive["exam_counts"].items()), columns=["Exam", "Questions"])
                fig = px.pie(edf, values="Questions", names="Exam", title="Cross-Exam Presence")
                fig.update_layout(**PLOT_LAYOUT, height=260)
                st.plotly_chart(fig, use_container_width=True)

            # Questions
            st.markdown("#### All Questions")
            qdf = dive["questions"]
            if sel_micro and sel_micro != "All":
                qdf = qdf[qdf["micro_topic"] == sel_micro]
            st.write(f"Showing {len(qdf)} questions")
            for _, row in qdf.iterrows():
                with st.expander(f"[{row['exam']} {row['year']}] {row['micro_topic']} — Diff: {row['difficulty']}"):
                    st.markdown(f"**Q:** {row['question_text'][:500]}")
                    st.markdown(f"**A:** {row['answer']}")
                    st.markdown(f"**Type:** {row['question_type']}  |  **Shift:** {row['shift']}")
        else:
            st.info("No data found for this topic.")
    else:
        st.info("Select a topic above to explore.")
        # Default treemap
        tree = get_topic_tree(DB_PATH, exam=exam_filter)
        if selected_subject != "All":
            tree = tree[tree["subject"] == selected_subject]
        if not tree.empty:
            fig = px.treemap(tree, path=["subject", "topic", "micro_topic"],
                             values="count", color="avg_difficulty",
                             color_continuous_scale="RdYlGn_r", color_continuous_midpoint=3,
                             title="Topic Hierarchy (size = questions, color = difficulty)")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)

        # Hot & Cold topics
        st.markdown('<div class="section-divider">Hot & Cold Topics</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-divider">Syllabus-Based Lesson Plan</div>', unsafe_allow_html=True)
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
                    last = int(sr["last_appeared"].max()) if not sr.empty and sr["last_appeared"].max() > 0 else "-"
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
    st.markdown('<div class="section-divider">Top 20 Focus Subtopics</div>', unsafe_allow_html=True)
    if not cov_df.empty:
        summary = cov_df.copy()
        summary["pred_score"] = summary["subtopic"].apply(
            lambda s: max((v for k, v in pred_scores.items()
                          if any(w in k.lower() for w in s.lower().split() if len(w) > 3)), default=0))
        summary["priority"] = summary["pred_score"] * 0.6 + (summary["frequency"] / max(summary["frequency"].max(), 1)) * 0.4
        summary = summary.sort_values("priority", ascending=False).head(20)
        fig = px.bar(summary, x="priority", y="subtopic", orientation="h", color="subject",
                     hover_data=["chapter", "questions_found", "last_appeared"],
                     color_discrete_sequence=SUBJ_COLORS)
        fig.update_layout(**PLOT_LAYOUT, height=550, yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)


# ================================================================
# TAB 4: HISTORICAL TIMELINE
# ================================================================
with tab_timeline:
    st.markdown('<div class="section-divider">Historical Timeline — Syllabus, Policy & News</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-divider">2024 Syllabus Overhaul</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-divider">News → Question Correlation Patterns</div>', unsafe_allow_html=True)
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
    st.markdown('<div class="section-divider">Question Explorer</div>', unsafe_allow_html=True)

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
    st.markdown('<div class="section-divider">Practice Paper Generator</div>', unsafe_allow_html=True)

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
