"""
Data Bridge Router
==================
Thin REST wrappers around the existing PRAJNA analysis modules
(trend_analyzer, deep_analysis, predictor_v3).  Bridges the
intelligence HTML dashboard to the real 23K-question SQLite database
without rewriting any analysis logic.

All heavy computation stays in the proven analysis/*.py modules.
This router is purely I/O: call the function, serialize to JSON.
"""
from __future__ import annotations

import os
import sys
import uuid
import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query

log = logging.getLogger(__name__)
router = APIRouter()

# ── Locate repo root + inject into sys.path ────────────────────────────────
# intelligence/ lives one level below repo root.
_HERE = os.path.dirname(os.path.abspath(__file__))
# routers/ -> api/ -> services/ -> intelligence/ -> repo-root
_REPO = os.path.abspath(os.path.join(_HERE, "..", "..", "..", ".."))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

DB_PATH = os.path.join(_REPO, "data", "exam.db")


def _import_analysis():
    """Lazy import analysis modules so the API still starts even if the DB
    is missing — endpoints will 503 gracefully."""
    try:
        from utils.db import get_questions_df                                      # noqa: F401
        from analysis.trend_analyzer import (                                      # noqa: F401
            find_hot_cold_topics, detect_cycles, topic_frequency_by_year,
        )
        from analysis.deep_analysis import (                                       # noqa: F401
            get_topic_deep_dive, get_subject_weightage_timeline,
        )
        from analysis.predictor_v3 import (                                        # noqa: F401
            predict_chapters_v3, predict_microtopics_v3, backtest_single_year,
        )
        return True
    except Exception as exc:
        log.warning("analysis modules unavailable: %s", exc)
        return False


_ANALYSIS_AVAILABLE = _import_analysis()


def _db_ok():
    if not _ANALYSIS_AVAILABLE:
        raise HTTPException(503, "Analysis modules not available — check repo path")
    if not os.path.exists(DB_PATH):
        raise HTTPException(503, f"exam.db not found at {DB_PATH}")


# ── 1. Hot / Cold Topics ───────────────────────────────────────────────────

@router.get("/hot-cold-topics", summary="Hot & Cold topic analysis from full history")
async def hot_cold_topics(
    exam_type:   str = Query("NEET", description="NEET | JEE Main | JEE Advanced"),
    recent_years: int = Query(3, ge=1, le=10),
    top_n:       int = Query(15, ge=5, le=50),
):
    """
    Hot topics = appeared frequently in last N years.
    Cold topics = historically frequent but recently dormant (due for return).
    Cyclical topics = appear on regular intervals.
    """
    _db_ok()
    from analysis.trend_analyzer import find_hot_cold_topics, detect_cycles

    exam_map = {"neet": "NEET", "jee_main": "JEE Main", "jee_advanced": "JEE Advanced"}
    exam = exam_map.get(exam_type.lower(), exam_type)

    try:
        hot, cold = find_hot_cold_topics(DB_PATH, recent_years=recent_years)
        cycles = detect_cycles(DB_PATH)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    def _parse_row(row):
        # Format: ((subject, topic), micro_topic, count_or_gap)
        if isinstance(row, (list, tuple)) and len(row) == 3:
            subj_topic = row[0]
            subject = subj_topic[0] if isinstance(subj_topic, (tuple, list)) else str(subj_topic)
            micro    = str(row[1])
            val      = int(row[2]) if hasattr(row[2], '__int__') else row[2]
            return subject, micro, val
        return "", str(row), 0

    hot_parsed  = [_parse_row(r) for r in (hot  or [])[:top_n]]
    cold_parsed = [_parse_row(r) for r in (cold or [])[:top_n]]

    return {
        "success": True,
        "request_id": str(uuid.uuid4()),
        "exam": exam,
        "recent_years": recent_years,
        "hot_topics": [
            {"subject": s, "micro_topic": mt, "count_recent": c}
            for s, mt, c in hot_parsed
        ],
        "cold_topics": [
            {"subject": s, "micro_topic": mt, "gap_years": g}
            for s, mt, g in cold_parsed
        ],
        "cyclical_topics": [
            {
                "micro_topic":          str(cy.get("micro_topic", "")),
                "topic":                str(cy.get("topic", "")),
                "avg_gap":              round(float(cy.get("avg_gap", 0)), 1),
                "consistency":          round(float(cy.get("consistency", 0)), 2),
                "estimated_cycle_years": round(float(cy.get("estimated_cycle_years", 0)), 1),
                "appearances":          [int(y) for y in cy.get("appearances", [])[-5:]],
            }
            for cy in (cycles or [])[:top_n]
        ],
    }


# ── 2. Topic Deep Dive ────────────────────────────────────────────────────

@router.get("/topic-deep-dive", summary="Full topic history, difficulty, question breakdown")
async def topic_deep_dive(
    topic:     str = Query(..., description="Chapter / topic name"),
    exam_type: str = Query("NEET", description="NEET | JEE Main | JEE Advanced"),
):
    """
    Returns complete historical data for one topic:
    - questions-per-year timeline
    - 3-year moving average
    - difficulty trend
    - question type distribution
    - subtopic frequency breakdown
    - cross-exam presence
    - all individual questions
    """
    _db_ok()
    from analysis.deep_analysis import get_topic_deep_dive

    exam_map = {"neet": "NEET", "jee_main": "JEE Main", "jee_advanced": "JEE Advanced"}
    exam = exam_map.get(exam_type.lower(), exam_type)

    try:
        dive = get_topic_deep_dive(DB_PATH, topic, exam=exam)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    if not dive:
        raise HTTPException(404, f"No data found for topic '{topic}'")

    # Serialize DataFrames → lists of dicts
    def df_to_list(df):
        if df is None or (hasattr(df, "empty") and df.empty):
            return []
        return df.to_dict(orient="records") if hasattr(df, "to_dict") else []

    return {
        "success": True,
        "request_id": str(uuid.uuid4()),
        "topic": topic,
        "exam": exam,
        "total_questions": dive.get("total_questions", 0),
        "first_year": dive.get("first_year"),
        "last_year": dive.get("last_year"),
        "span_years": (dive.get("last_year", 0) or 0) - (dive.get("first_year", 0) or 0),
        "year_counts": df_to_list(dive.get("year_counts")),
        "difficulty_trend": df_to_list(dive.get("difficulty_trend")),
        "type_distribution": dive.get("type_distribution", {}),
        "subtopic_counts": df_to_list(dive.get("subtopic_counts")),
        "exam_counts": dive.get("exam_counts", {}),
        "questions": df_to_list(dive.get("questions")),
    }


# ── 3. Backtest ───────────────────────────────────────────────────────────

@router.get("/backtest", summary="Train-on-past, predict-one-year backtest")
async def run_backtest(
    test_year:   int = Query(..., ge=2010, le=2025, description="Year to predict"),
    exam_type:   str = Query("NEET", description="NEET | JEE Main | JEE Advanced"),
    top_n:       int = Query(60, ge=10, le=150),
    level:       str = Query("chapter", description="chapter | micro"),
):
    """
    Trains PRAJNA on all data before test_year, predicts test_year,
    then compares against the actual paper.  Returns hit-rate, coverage,
    matched topics, missed topics.
    """
    _db_ok()
    from analysis.predictor_v3 import backtest_single_year

    exam_map = {"neet": "NEET", "jee_main": "JEE Main", "jee_advanced": "JEE Advanced"}
    exam = exam_map.get(exam_type.lower(), exam_type)

    try:
        result = backtest_single_year(
            DB_PATH,
            test_year=test_year,
            exam=exam,
            k=top_n,
            level=level,
        )
    except Exception as exc:
        log.exception("Backtest failed")
        raise HTTPException(500, f"Backtest error: {exc}")

    if result is None:
        raise HTTPException(404, f"No data for exam='{exam}' year={test_year}")

    # Normalize — backtest_single_year may return dict or tuple
    if isinstance(result, dict):
        summary = result
    elif isinstance(result, (list, tuple)) and len(result) >= 1:
        summary = result[0] if isinstance(result[0], dict) else {"raw": str(result)}
    else:
        summary = {"raw": str(result)}

    return {
        "success": True,
        "request_id": str(uuid.uuid4()),
        "test_year": test_year,
        "exam": exam,
        "top_n": top_n,
        "level": level,
        "summary": summary,
    }


# ── 4. Lesson Plan ────────────────────────────────────────────────────────

@router.get("/lesson-plan", summary="Syllabus-mapped lesson plan with prediction priority")
async def lesson_plan(
    exam_type: str = Query("NEET", description="NEET | JEE Main | JEE Advanced"),
    year:      int = Query(2025, ge=2024, le=2035),
    top_n:     int = Query(60, ge=10, le=150),
):
    """
    Maps the official NEET/JEE syllabus to historical appearance data
    and returns a prioritised lesson plan — sorted by predicted importance.
    """
    _db_ok()
    from analysis.predictor_v3 import predict_chapters_v3, predict_microtopics_v3
    from data.syllabus import NEET_SYLLABUS, JEE_SYLLABUS

    exam_map = {"neet": "NEET", "jee_main": "JEE Main", "jee_advanced": "JEE Advanced"}
    exam = exam_map.get(exam_type.lower(), exam_type)

    syllabus = NEET_SYLLABUS if "neet" in exam_type.lower() else JEE_SYLLABUS

    try:
        chapters = predict_chapters_v3(DB_PATH, target_year=year, exam=exam, top_k=top_n)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    # Build lesson plan: each chapter → predicted priority + syllabus topics
    plan = []
    for ch in (chapters or []):
        ch_name = ch.get("chapter") or ch.get("topic") or ch.get("micro_topic", "")
        subject  = ch.get("subject", "")
        # Find matching syllabus entry
        syllabus_topics: List[str] = []
        for subj_key, chapters_dict in syllabus.items():
            if isinstance(chapters_dict, dict):
                for chap, topics in chapters_dict.items():
                    if chap.lower() == ch_name.lower() or ch_name.lower() in chap.lower():
                        if isinstance(topics, list):
                            syllabus_topics = topics
                        break

        plan.append({
            "chapter":             ch_name,
            "subject":             subject,
            "appearance_probability": round(ch.get("appearance_probability", 0), 3),
            "expected_questions":  ch.get("expected_questions", 0),
            "confidence_score":    round(ch.get("confidence_score", 0), 3),
            "trend_direction":     ch.get("trend_direction", ""),
            "syllabus_topics":     syllabus_topics[:10],  # cap at 10 for size
            "priority_band":       (
                "A" if ch.get("appearance_probability", 0) >= 0.7
                else "B" if ch.get("appearance_probability", 0) >= 0.45
                else "C"
            ),
        })

    return {
        "success": True,
        "request_id": str(uuid.uuid4()),
        "exam": exam,
        "year": year,
        "top_n": top_n,
        "total_chapters": len(plan),
        "lesson_plan": plan,
    }


# ── 5. Subject Weightage Timeline ─────────────────────────────────────────

@router.get("/subject-timeline", summary="Subject weightage % across all years")
async def subject_timeline(
    exam_type: str = Query("NEET", description="NEET | JEE Main | JEE Advanced"),
):
    """Year-by-year breakdown of how many questions each subject contributed."""
    _db_ok()
    from analysis.deep_analysis import get_subject_weightage_timeline

    exam_map = {"neet": "NEET", "jee_main": "JEE Main", "jee_advanced": "JEE Advanced"}
    exam = exam_map.get(exam_type.lower(), exam_type)

    try:
        df = get_subject_weightage_timeline(DB_PATH, exam=exam)
    except Exception as exc:
        raise HTTPException(500, str(exc))

    if df is None or (hasattr(df, "empty") and df.empty):
        return {"success": True, "request_id": str(uuid.uuid4()), "rows": []}

    return {
        "success": True,
        "request_id": str(uuid.uuid4()),
        "exam": exam,
        "rows": df.to_dict(orient="records") if hasattr(df, "to_dict") else [],
    }


# ── 6. Real Predictions (PRAJNA heuristic + SLM) ─────────────────────────

@router.get("/predict", summary="Run PRAJNA predict_chapters_v3 or predict_microtopics_v3")
async def real_predict(
    exam_type: str  = Query("NEET",    description="NEET | JEE Main | JEE Advanced"),
    year:      int  = Query(2026,      ge=2010, le=2035),
    top_n:     int  = Query(60,        ge=5,    le=200),
    level:     str  = Query("chapter", description="chapter | micro"),
    subject:   str  = Query("",        description="Optional subject filter"),
):
    """
    Runs the real PRAJNA 3-stage prediction engine on the live 23K-question DB.
    - level='chapter'  → predict_chapters_v3   (heuristic + SLM reranking)
    - level='micro'    → predict_microtopics_v3 (subject-balanced reranking)
    Returns appearance_probability, expected_questions, confidence, trend_direction,
    signal_breakdown, reasons, syllabus_status — everything Streamlit shows.
    """
    _db_ok()
    from analysis.predictor_v3 import predict_chapters_v3, predict_microtopics_v3

    exam_map = {"neet": "NEET", "jee_main": "JEE Main", "jee_advanced": "JEE Advanced"}
    exam = exam_map.get(exam_type.lower(), exam_type)

    try:
        if level == "micro":
            try:
                from analysis.predictor_v4 import predict_microtopics_v4
                preds = predict_microtopics_v4(DB_PATH, target_year=year, exam=exam, top_k=top_n)
            except Exception as exc:
                log.warning("predictor_v4 failed, falling back to v3: %s", exc)
                preds = predict_microtopics_v3(DB_PATH, target_year=year, exam=exam, top_k=top_n)
        else:
            preds = predict_chapters_v3(DB_PATH, target_year=year, exam=exam, top_k=top_n)
    except Exception as exc:
        log.exception("Real predict failed")
        raise HTTPException(500, str(exc))

    # Subject filter (client can also do this, but server-side is faster)
    if subject and subject.strip().lower() not in ("", "all", "all subjects"):
        preds = [p for p in preds if (p.get("subject") or "").lower() == subject.strip().lower()]

    # Normalize numpy scalar types → plain Python (JSON-safe)
    def _clean(v):
        if hasattr(v, "item"):          return v.item()   # numpy scalar
        if isinstance(v, dict):         return {kk: _clean(vv) for kk, vv in v.items()}
        if isinstance(v, (list, tuple)):return [_clean(i) for i in v]
        return v

    cleaned = [_clean(p) for p in preds]

    # Summary stats (same KPIs shown in Streamlit)
    high_prob  = [p for p in cleaned if p.get("appearance_probability", 0) >= 0.70]
    rising     = [p for p in cleaned if p.get("trend_direction") == "RISING"]
    critical   = [p for p in cleaned if p.get("confidence") == "HIGH"
                  and p.get("appearance_probability", 0) >= 0.50]
    total_exp  = sum(p.get("expected_questions", 0) for p in cleaned[:top_n])
    avg_conf   = (sum(p.get("confidence_score", 0) for p in cleaned)
                  / max(len(cleaned), 1))

    return {
        "success":    True,
        "request_id": str(uuid.uuid4()),
        "exam":       exam,
        "year":       year,
        "level":      level,
        "total":      len(cleaned),
        "summary": {
            "high_prob_count":   len(high_prob),
            "expected_total_qs": round(float(total_exp), 1),
            "rising_count":      len(rising),
            "avg_confidence":    round(float(avg_conf), 3),
            "critical_count":    len(critical),
        },
        "predictions": cleaned,
    }


# ── 7. All Topics List ────────────────────────────────────────────────────

@router.get("/topics-list", summary="All distinct topics/chapters in the database")
async def topics_list(
    exam_type: str = Query("NEET", description="NEET | JEE Main | JEE Advanced"),
):
    """Returns every distinct chapter+subject in the DB for the topic deep-dive selector."""
    _db_ok()
    import sqlite3

    exam_map = {"neet": "NEET", "jee_main": "JEE Main", "jee_advanced": "JEE Advanced"}
    exam = exam_map.get(exam_type.lower(), exam_type)

    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT DISTINCT subject, topic FROM questions WHERE exam=? ORDER BY subject, topic",
            (exam,),
        )
        rows = [{"subject": r[0], "topic": r[1]} for r in cur.fetchall()]
        conn.close()
    except Exception as exc:
        raise HTTPException(500, str(exc))

    return {
        "success": True,
        "request_id": str(uuid.uuid4()),
        "exam": exam,
        "topics": rows,
        "count": len(rows),
    }
