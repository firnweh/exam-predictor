"""
Exam Predictor v4 — Hierarchical Micro-Topic Engine (Approaches A+B+C combined)

Architecture:
  Stage 1: Chapter gate — reuses predict_chapters_v3() (~94% precision)
  Stage 2: Conditional micro scoring — scores micros WITHIN each predicted
           chapter's bucket using 10 signals (7 original + 3 new)
  Stage 3: Weight-optimised ranking — loads weights from weights_cache.json

New signals vs v3:
  parent_inheritance — chapter final_score passed down to all its micro-topics
  recency_burst      — fraction of last 3 years the micro appeared
  dispersion         — rewards micro-topics with dense (2-3 Qs) appearances
"""

import json
import os
import numpy as np
from collections import Counter
from utils.db import get_questions_df
from analysis.predictor_v3 import (
    predict_chapters_v3, _normalize_chapter, _syllabus_status,
    _expected_questions, _predict_format, HOLDOUT_YEARS,
)

# ================================================================
# WEIGHT MANAGEMENT
# ================================================================

WEIGHTS_CACHE_PATH = os.path.join(os.path.dirname(__file__), "weights_cache.json")

DEFAULT_SIGNAL_WEIGHTS = {
    "recency_freq":       0.20,
    "appearance_rate":    0.15,
    "recent_3yr":         0.10,
    "recent_5yr":         0.03,
    "gap_return":         0.10,
    "trend_slope":        0.05,
    "cycle_match":        0.05,
    "parent_inheritance": 0.15,
    "recency_burst":      0.12,
    "dispersion":         0.05,
}
_total = sum(DEFAULT_SIGNAL_WEIGHTS.values())
DEFAULT_SIGNAL_WEIGHTS = {k: v / _total for k, v in DEFAULT_SIGNAL_WEIGHTS.items()}

DEFAULT_FINAL_WEIGHTS = {
    "app_prob": 0.40,
    "exp_qs":   0.25,
    "yield":    0.10,
    "cross":    0.10,
    "parent":   0.15,
}
_total_fw = sum(DEFAULT_FINAL_WEIGHTS.values())
DEFAULT_FINAL_WEIGHTS = {k: v / _total_fw for k, v in DEFAULT_FINAL_WEIGHTS.items()}


def _normalise_weights(w):
    """Clamp negatives to 0.01, then normalise to sum=1."""
    clamped = {k: max(0.01, v) for k, v in w.items()}
    total = sum(clamped.values())
    return {k: v / total for k, v in clamped.items()}


def _load_weights():
    """
    Load optimised weights from cache file if it exists.
    Falls back to DEFAULT_* if cache is missing or corrupt.
    Returns (signal_weights, final_weights).
    """
    if os.path.exists(WEIGHTS_CACHE_PATH):
        try:
            with open(WEIGHTS_CACHE_PATH) as f:
                cache = json.load(f)
            sw = cache.get("signal_weights", DEFAULT_SIGNAL_WEIGHTS)
            fw = cache.get("final_weights", DEFAULT_FINAL_WEIGHTS)
            return _normalise_weights(sw), _normalise_weights(fw)
        except Exception:
            pass
    return dict(DEFAULT_SIGNAL_WEIGHTS), dict(DEFAULT_FINAL_WEIGHTS)


# ================================================================
# STAGE 2: APPEARANCE PROBABILITY (10 signals)
# ================================================================

def _appearance_probability_v4(years_appeared, total_years_range, target_year,
                                max_year, parent_score, total_questions,
                                signal_weights):
    """
    Estimate P(micro-topic appears in target year).

    Extends v3's 7 signals with 3 new ones:
      parent_inheritance — parent chapter confidence propagated down
      recency_burst      — appeared in 2+ of last 3 years -> strong signal
      dispersion         — rewards 2-3 Qs/appearance (not thin 1-Q scatter)

    Returns:
        (probability: float, signals: dict)
    """
    if not years_appeared:
        return 0.0, {}

    signals = {}

    # Signal 1: Recency-weighted frequency (exponential decay)
    decay = 1.5
    rwf = sum(1.0 / ((target_year - y) ** decay + 1) for y in years_appeared)
    max_possible = sum(
        1.0 / ((target_year - y) ** decay + 1)
        for y in range(total_years_range[0], total_years_range[1] + 1)
    )
    signals["recency_freq"] = min(rwf / (max_possible + 0.01), 1.0)

    # Signal 2: Raw appearance rate
    year_span = total_years_range[1] - total_years_range[0] + 1
    signals["appearance_rate"] = min(len(set(years_appeared)) / year_span, 1.0)

    # Signal 3 & 4: Recent presence
    recent_3 = sum(1 for y in years_appeared if y >= max_year - 2)
    recent_5 = sum(1 for y in years_appeared if y >= max_year - 4)
    signals["recent_3yr"] = min(recent_3 / 3, 1.0)
    signals["recent_5yr"] = min(recent_5 / 5, 1.0)

    # Signal 5: Gap return probability
    gap = target_year - max(years_appeared)
    gaps_sorted = sorted(set(years_appeared))
    inter_gaps = [gaps_sorted[i + 1] - gaps_sorted[i] for i in range(len(gaps_sorted) - 1)]
    mean_gap = np.mean(inter_gaps) if inter_gaps else year_span
    gap_ratio = gap / (mean_gap + 0.1)
    signals["gap_return"] = min(gap_ratio, 2.0) / 2.0

    # Signal 6: Trend slope (linear regression on last 10 years)
    recent_window = list(range(max(total_years_range[0], max_year - 9), max_year + 1))
    year_counts = Counter(years_appeared)
    y_vals = [year_counts.get(yr, 0) for yr in recent_window]
    if len(y_vals) >= 3 and sum(y_vals) > 0:
        x = np.arange(len(y_vals))
        slope = np.polyfit(x, y_vals, 1)[0]
        signals["trend_slope"] = min(max((slope + 0.5) / 1.0, 0), 1.0)
    else:
        signals["trend_slope"] = 0.5

    # Signal 7: Cycle match
    cycle_score = 0.0
    if inter_gaps and len(set(years_appeared)) >= 4:
        avg_gap = np.mean(inter_gaps)
        if np.var(inter_gaps) <= 1.5 and avg_gap > 0:
            years_since = target_year - max(years_appeared)
            remainder = years_since % round(avg_gap)
            if remainder == 0:
                cycle_score = 1.0
            elif remainder <= 1 or (round(avg_gap) - remainder) <= 1:
                cycle_score = 0.5
    signals["cycle_match"] = cycle_score

    # Signal 8: Parent chapter confidence flows down
    signals["parent_inheritance"] = float(parent_score)

    # Signal 9: Recency burst — same numerator as recent_3yr but independent weight
    signals["recency_burst"] = min(recent_3 / 3.0, 1.0)

    # Signal 10: Dispersion — rewards heavy/dense appearances (2-3 Qs/yr)
    avg_qs_when_present = total_questions / max(len(set(years_appeared)), 1)
    signals["dispersion"] = min(avg_qs_when_present / 3.0, 1.0)

    # Weighted combination
    score = sum(signals[k] * signal_weights.get(k, 0.0) for k in signals)
    return min(score * 1.8, 0.99), signals


def predict_microtopics_v4(db_path="data/exam.db", target_year=2026, exam=None,
                            top_k=200, chapter_k=70,
                            signal_weights=None, final_weights=None):
    """
    Hierarchical micro-topic prediction (v4).

    Stage 1: predict_chapters_v3() -> top chapter_k chapters with scores.
    Stage 2: For each predicted chapter, score all its micro-topics using
             _appearance_probability_v4() (10 signals including parent_inheritance).
             Micro-topics whose parent chapter is NOT in top chapter_k are excluded.
    """
    if signal_weights is None or final_weights is None:
        cached_sw, cached_fw = _load_weights()
        signal_weights = signal_weights if signal_weights is not None else cached_sw
        final_weights  = final_weights  if final_weights  is not None else cached_fw

    # Stage 1: Chapter gate
    chapter_preds = predict_chapters_v3(
        db_path, target_year=target_year, exam=exam, top_k=chapter_k
    )

    chapter_score_map = {}
    for cp in chapter_preds:
        if cp.get("syllabus_status") != "REMOVED":
            chapter_score_map[cp["chapter"].lower()] = cp["final_score"]
            norm = cp.get("normalized_chapter", "")
            if norm:
                chapter_score_map[norm.lower()] = cp["final_score"]

    # Stage 2: Conditional micro scoring
    full_df = get_questions_df(db_path)
    df = full_df[~full_df["year"].isin(HOLDOUT_YEARS)]
    if exam:
        df = df[df["exam"] == exam]

    if df.empty:
        return []

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    cross_df = get_questions_df(db_path)
    cross_df = cross_df[~cross_df["year"].isin(HOLDOUT_YEARS)]

    predictions = []

    for (subject, chapter, micro_topic), group in df.groupby(
        ["subject", "topic", "micro_topic"]
    ):
        normalized = _normalize_chapter(chapter)
        parent_score = chapter_score_map.get(
            chapter.lower(),
            chapter_score_map.get(normalized.lower(), 0.0)
        )
        if parent_score == 0.0:
            continue

        syl_status, syl_gate = _syllabus_status(chapter, exam)
        if syl_gate == 0.0:
            continue

        years_appeared   = sorted(group["year"].unique())
        qs_per_year      = group.groupby("year").size().to_dict()
        question_types   = group["question_type"].value_counts().to_dict()
        difficulty_list  = group["difficulty"].dropna().tolist()
        recent_diffs     = [d for y, d in zip(group["year"], group["difficulty"])
                            if y >= max_year - 4]
        total_questions  = len(group)

        app_prob, app_signals = _appearance_probability_v4(
            years_appeared, (min_year, max_year), target_year, max_year,
            parent_score, total_questions, signal_weights
        )
        app_prob *= syl_gate

        exp_qs, exp_min, exp_max, wt_conf = _expected_questions(
            qs_per_year, years_appeared, target_year, max_year
        )

        likely_types, likely_diff, format_dom = _predict_format(
            question_types, difficulty_list, recent_diffs
        )

        cross_score = min(
            cross_df[cross_df["micro_topic"] == micro_topic]["exam"].nunique() / 3, 1.0
        )
        app_signals["cross_exam"] = cross_score

        slope = app_signals.get("trend_slope", 0.5)
        if slope > 0.6:
            trend_dir = "RISING"
        elif slope < 0.4:
            trend_dir = "DECLINING"
        else:
            trend_dir = "STABLE"
        if syl_status == "NEW":
            trend_dir = "NEW"

        recent_qs    = [v for y, v in qs_per_year.items() if y >= max_year - 2]
        yield_bonus  = min(np.mean(recent_qs) / 3.0, 1.0) if recent_qs else 0.0

        fw = final_weights
        fw_sum = sum(fw.values()) or 1.0
        final_score = (
            fw.get("app_prob", 0.40) * app_prob +
            fw.get("exp_qs",   0.25) * min(exp_qs / 4.0, 1.0) +
            fw.get("yield",    0.10) * yield_bonus +
            fw.get("cross",    0.10) * cross_score +
            fw.get("parent",   0.15) * parent_score
        ) / fw_sum

        signal_breakdown = {k: {"value": round(v, 3)} for k, v in app_signals.items()}
        signal_breakdown["expected_qs"]    = {"value": round(exp_qs, 1)}
        signal_breakdown["parent_score"]   = {"value": round(parent_score, 3)}

        predictions.append({
            "micro_topic":           micro_topic,
            "chapter":               chapter,
            "normalized_chapter":    normalized,
            "subject":               subject,
            "appearance_probability": round(app_prob, 3),
            "expected_questions":    round(exp_qs, 1),
            "expected_qs_min":       exp_min,
            "expected_qs_max":       exp_max,
            "likely_formats":        likely_types,
            "likely_difficulty":     likely_diff,
            "format_dominance":      format_dom,
            "confidence":            "HIGH" if app_prob > 0.6 else "MEDIUM" if app_prob > 0.3 else "LOW",
            "confidence_score":      round(app_prob * 0.7 + parent_score * 0.3, 3),
            "final_score":           round(final_score, 4),
            "signal_breakdown":      signal_breakdown,
            "trend_direction":       trend_dir,
            "syllabus_status":       syl_status,
            "total_appearances":     len(years_appeared),
            "total_questions":       total_questions,
            "last_appeared":         int(max(years_appeared)),
            "parent_final_score":    round(parent_score, 3),
            "training_years":        f"{min_year}-{max_year}",
        })

    predictions.sort(key=lambda x: x["final_score"], reverse=True)
    return predictions[:top_k]
