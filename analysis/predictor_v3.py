"""
Exam Predictor v3 — Chapter-level 3-stage model with constraint-based reranking.

Architecture:
  Stage 1: Appearance model — P(chapter appears in paper)
  Stage 2: Weightage model — Expected questions if chapter appears
  Stage 3: Format model — Likely question types

Post-processing:
  - Subject-balanced reranking with quotas
  - Unique chapter enforcement (no duplicate slots)
  - Diversification penalty for correlated chapters

Evaluation:
  - Coverage@K (fraction of actual paper covered)
  - Heavy-topic recall (chapters with 3+ questions)
  - Subject coverage balance
  - Rank correlation (predicted importance vs actual)
"""

import numpy as np
import pandas as pd
from collections import Counter
from utils.db import get_questions_df
from analysis.trend_analyzer import detect_cycles
from data.historical_events import NEET_2024_REMOVED, NEET_2024_ADDED, JEE_2024_REMOVED

HOLDOUT_YEARS = {2024, 2025, 2026}

# ================================================================
# CHAPTER NAME NORMALIZATION (Taxonomy Layer)
# ================================================================

# Map variant names → canonical chapter name
CHAPTER_ALIASES = {
    "heat and thermodynamics": "thermodynamics",
    "heat & thermodynamics": "thermodynamics",
    "thermal properties of matter": "thermodynamics",
    "atoms and nuclei": "atoms and nuclei",
    "atoms": "atoms and nuclei",
    "nuclei": "atoms and nuclei",
    "equilibrum": "equilibrium",
    "chemical equilibrium": "equilibrium",
    "ionic equilibrium": "equilibrium",
    "ray optics and optical instruments": "ray optics",
    "ray optics": "ray optics",
    "geometrical optics": "ray optics",
    "wave optics": "wave optics",
    "semiconductor electronics": "semiconductor electronics",
    "semiconductor and p n junction diode": "semiconductor electronics",
    "p block elements": "p block elements",
    "p-block elements": "p block elements",
    "d and f block elements": "d and f block elements",
    "d-block and f-block elements": "d and f block elements",
    "d block elements": "d and f block elements",
    "coordination compounds": "coordination compounds",
    "coordination chemistry": "coordination compounds",
    "units and measurements": "units and measurements",
    "units and measurement": "units and measurements",
    "physical world and measurement": "units and measurements",
    "electrostatic potential and capacitance": "electrostatics",
    "electric charges and fields": "electrostatics",
    "electrostatics": "electrostatics",
    "electromagnetic induction": "electromagnetic induction",
    "electromagnetic induction and alternating current": "electromagnetic induction",
    "alternating current": "alternating current",
    "ac circuits and power in ac circuits": "alternating current",
    "moving charges and magnetism": "magnetism",
    "magnetism and matter": "magnetism",
    "chemical kinetics": "chemical kinetics",
    "reaction kinetics": "chemical kinetics",
}


def _normalize_chapter(name):
    """Normalize chapter name using alias map."""
    key = name.strip().lower()
    return CHAPTER_ALIASES.get(key, name)


def _syllabus_status(topic, exam):
    """Determine syllabus status for a chapter."""
    topic_lower = topic.lower()
    removed_list = NEET_2024_REMOVED if exam == "NEET" else JEE_2024_REMOVED if exam else []
    added_list = NEET_2024_ADDED if exam == "NEET" else []

    for r in removed_list:
        if r.lower() in topic_lower or topic_lower in r.lower():
            return "REMOVED", 0.0
    for a in added_list:
        if a.lower() in topic_lower or topic_lower in a.lower():
            return "NEW", 0.5
    return "RETAINED", 1.0


# ================================================================
# STAGE 1: APPEARANCE MODEL
# ================================================================

def _appearance_probability(years_appeared, total_years_range, target_year, max_year, decay=1.5):
    """
    Estimate P(chapter appears in target year).
    Uses recency-weighted frequency + gap-return + trend.
    """
    if not years_appeared:
        return 0.0, {}

    signals = {}

    # Signal 1: Recency-weighted frequency (exponential decay)
    rwf = sum(1.0 / ((target_year - y) ** decay + 1) for y in years_appeared)
    max_possible = sum(1.0 / ((target_year - y) ** decay + 1) for y in range(total_years_range[0], total_years_range[1] + 1))
    rwf_norm = min(rwf / (max_possible + 0.01), 1.0)
    signals["recency_freq"] = rwf_norm

    # Signal 2: Raw appearance rate
    year_span = total_years_range[1] - total_years_range[0] + 1
    appearance_rate = len(set(years_appeared)) / year_span
    signals["appearance_rate"] = min(appearance_rate, 1.0)

    # Signal 3: Recent presence (last 3 and 5 years)
    recent_3 = sum(1 for y in years_appeared if y >= max_year - 2)
    recent_5 = sum(1 for y in years_appeared if y >= max_year - 4)
    signals["recent_3yr"] = min(recent_3 / 3, 1.0)
    signals["recent_5yr"] = min(recent_5 / 5, 1.0)

    # Signal 4: Gap return probability
    gap = target_year - max(years_appeared)
    gaps = sorted(set(years_appeared))
    inter_gaps = [gaps[i+1] - gaps[i] for i in range(len(gaps)-1)]
    mean_gap = np.mean(inter_gaps) if inter_gaps else year_span
    gap_ratio = gap / (mean_gap + 0.1)
    gap_return = min(gap_ratio, 2.0) / 2.0  # peaks at 2x mean gap
    signals["gap_return"] = gap_return

    # Signal 5: Trend slope (linear regression on last 10 years)
    recent_window = list(range(max(total_years_range[0], max_year - 9), max_year + 1))
    year_counts = Counter(years_appeared)
    y_vals = [year_counts.get(yr, 0) for yr in recent_window]
    if len(y_vals) >= 3 and sum(y_vals) > 0:
        x = np.arange(len(y_vals))
        slope = np.polyfit(x, y_vals, 1)[0]
        slope_norm = min(max((slope + 0.5) / 1.0, 0), 1.0)
    else:
        slope_norm = 0.5
    signals["trend_slope"] = slope_norm

    # Signal 6: Cycle match
    cycle_score = 0.0
    if inter_gaps and len(set(years_appeared)) >= 4:
        avg_gap = np.mean(inter_gaps)
        variance = np.var(inter_gaps)
        if variance <= 1.5 and avg_gap > 0:
            years_since = target_year - max(years_appeared)
            remainder = years_since % round(avg_gap)
            if remainder == 0:
                cycle_score = 1.0
            elif remainder <= 1 or (round(avg_gap) - remainder) <= 1:
                cycle_score = 0.5
    signals["cycle_match"] = cycle_score

    # Weighted combination
    weights = {
        "recency_freq": 0.25,
        "appearance_rate": 0.20,
        "recent_3yr": 0.15,
        "recent_5yr": 0.05,
        "gap_return": 0.15,
        "trend_slope": 0.12,
        "cycle_match": 0.08,
    }

    score = sum(signals[k] * weights[k] for k in weights)
    return min(score * 1.8, 0.99), signals


# ================================================================
# STAGE 2: WEIGHTAGE MODEL
# ================================================================

def _expected_questions(qs_per_year, years_appeared, target_year, max_year):
    """
    Predict expected number of questions if chapter appears.
    Returns (expected_qs, min_qs, max_qs, confidence).
    """
    if not qs_per_year:
        return 1, 1, 2, 0.0

    values = list(qs_per_year.values())
    mean_qs = np.mean(values)
    median_qs = np.median(values)

    # Recency-weighted average (recent years count more)
    weighted_sum = 0
    weight_total = 0
    for yr, count in qs_per_year.items():
        w = 1.0 / ((target_year - yr) ** 0.5 + 1)
        weighted_sum += count * w
        weight_total += w
    recency_avg = weighted_sum / weight_total if weight_total > 0 else mean_qs

    # Recent trend
    recent_vals = [v for y, v in qs_per_year.items() if y >= max_year - 4]
    recent_avg = np.mean(recent_vals) if recent_vals else mean_qs

    # Blend: 50% recency-weighted, 30% recent avg, 20% overall median
    expected = 0.50 * recency_avg + 0.30 * recent_avg + 0.20 * median_qs

    # Confidence based on consistency
    cv = np.std(values) / mean_qs if mean_qs > 0 else 1.0
    weightage_confidence = max(0, 1 - cv)

    # Range
    q25 = max(1, int(np.percentile(values, 25)))
    q75 = int(np.percentile(values, 75))

    return round(expected, 1), q25, max(q75, q25 + 1), round(weightage_confidence, 2)


# ================================================================
# STAGE 3: FORMAT MODEL
# ================================================================

def _predict_format(question_types, difficulty_list, recent_difficulties):
    """Predict likely question formats and difficulty."""
    # Format prediction
    if question_types:
        type_ranked = sorted(question_types.items(), key=lambda x: -x[1])
        likely_types = [t[0] for t in type_ranked[:3]]
        dominant_pct = type_ranked[0][1] / sum(question_types.values())
    else:
        likely_types = ["MCQ_single"]
        dominant_pct = 1.0

    # Difficulty prediction (recency-weighted)
    if recent_difficulties:
        likely_diff = round(np.mean(recent_difficulties), 1)
    elif difficulty_list:
        likely_diff = round(np.mean(difficulty_list), 1)
    else:
        likely_diff = 3.0

    return likely_types, likely_diff, round(dominant_pct, 2)


# ================================================================
# CONFIDENCE SCORING
# ================================================================

def _confidence_score(appearance_prob, weightage_confidence, appearances, syl_status,
                      recent_3yr, trend_slope):
    """
    Composite confidence score separate from probability.
    A topic can have high probability but low confidence (volatile)
    or high confidence but low probability (stable but rare).
    """
    factors = {
        "data_depth": min(appearances / 15, 1.0),  # more data = more confident
        "weightage_stability": weightage_confidence,
        "recent_evidence": min(recent_3yr / 2, 1.0),
        "trend_clarity": abs(trend_slope - 0.5) * 2,  # clear trend = more confident
        "syllabus_certainty": 1.0 if syl_status == "RETAINED" else 0.5 if syl_status == "MODIFIED" else 0.3,
    }

    weights = {"data_depth": 0.30, "weightage_stability": 0.25, "recent_evidence": 0.20,
               "trend_clarity": 0.10, "syllabus_certainty": 0.15}

    score = sum(factors[k] * weights[k] for k in weights)

    if score >= 0.65:
        label = "HIGH"
    elif score >= 0.45:
        label = "MEDIUM"
    elif score >= 0.25:
        label = "LOW"
    else:
        label = "SPECULATIVE"

    return round(score, 3), label


# ================================================================
# SUBJECT BALANCING + CONSTRAINT LAYER
# ================================================================

# Approximate subject quotas (fraction of total paper)
SUBJECT_QUOTAS = {
    "NEET": {"Biology": 0.50, "Physics": 0.25, "Chemistry": 0.25},
    "JEE Main": {"Physics": 0.33, "Chemistry": 0.33, "Mathematics": 0.34},
    "JEE Advanced": {"Physics": 0.33, "Chemistry": 0.33, "Mathematics": 0.34},
}

# Hard subject guard — prevents cross-exam leakage (Maths in NEET, Bio in JEE)
EXAM_VALID_SUBJECTS = {
    "NEET":         {"Biology", "Physics", "Chemistry"},
    "JEE Main":     {"Physics", "Chemistry", "Mathematics"},
    "JEE Advanced": {"Physics", "Chemistry", "Mathematics"},
}


def _subject_balanced_rerank(predictions, exam, top_k=50):
    """
    Rerank predictions with subject quotas.
    Ensures each subject gets proportional representation.
    Also enforces hard subject guard to prevent cross-exam leakage
    (e.g. Mathematics in NEET, Biology in JEE).
    """
    # Hard subject guard: drop predictions that don't belong to this exam
    valid_subjects = EXAM_VALID_SUBJECTS.get(exam, set())
    if valid_subjects:
        predictions = [p for p in predictions if p["subject"] in valid_subjects]

    quotas = SUBJECT_QUOTAS.get(exam, {})
    if not quotas:
        # Default: equal split
        subjects = list(set(p["subject"] for p in predictions))
        quotas = {s: 1.0 / len(subjects) for s in subjects}

    # Allocate slots per subject
    slots = {s: max(3, int(top_k * q)) for s, q in quotas.items()}

    # Ensure total = top_k
    total = sum(slots.values())
    if total < top_k:
        # Give extra to subject with most predictions
        biggest = max(quotas, key=quotas.get)
        slots[biggest] += top_k - total

    # Fill per-subject
    per_subject = {}
    for p in predictions:
        s = p["subject"]
        if s not in per_subject:
            per_subject[s] = []
        per_subject[s].append(p)

    result = []
    for s, slot_count in slots.items():
        subj_preds = per_subject.get(s, [])
        # Already sorted by final_score
        result.extend(subj_preds[:slot_count])

    # Re-sort by final_score
    result.sort(key=lambda x: x["final_score"], reverse=True)
    return result[:top_k]


# ================================================================
# MAIN PREDICTION FUNCTION
# ================================================================

def predict_chapters_v3(db_path="data/exam.db", target_year=2026, exam=None, top_k=50):
    """
    Chapter-level 3-stage prediction.

    Returns list of dicts, one per unique chapter:
      - chapter, subject
      - appearance_probability
      - expected_questions (count, min, max)
      - likely_formats, likely_difficulty
      - confidence, confidence_score
      - final_score (combined)
      - signal_breakdown
      - top_micro_topic
      - trend_direction
      - syllabus_status
    """
    full_df = get_questions_df(db_path)

    # Training data: exclude holdout
    df = full_df[~full_df["year"].isin(HOLDOUT_YEARS)]
    if exam:
        df = df[df["exam"] == exam]
        full_df = full_df[full_df["exam"] == exam]

    if df.empty:
        return []

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    # Cross-exam data
    cross_df = get_questions_df(db_path)
    cross_df = cross_df[~cross_df["year"].isin(HOLDOUT_YEARS)]

    # Chapter-level aggregation
    chapter_groups = df.groupby(["subject", "topic"])

    predictions = []

    for (subject, chapter), group in chapter_groups:
        normalized_chapter = _normalize_chapter(chapter)

        # Syllabus gate
        syl_status, syl_gate = _syllabus_status(chapter, exam)
        if syl_gate == 0.0:
            predictions.append({
                "chapter": chapter, "normalized_chapter": normalized_chapter,
                "subject": subject,
                "appearance_probability": 0.0, "expected_questions": 0,
                "expected_qs_min": 0, "expected_qs_max": 0,
                "likely_formats": [], "likely_difficulty": 0,
                "format_dominance": 0, "confidence": "HIGH",
                "confidence_score": 0.95, "final_score": 0.0,
                "signal_breakdown": {}, "reasons": ["Removed from syllabus"],
                "top_micro_topic": "", "trend_direction": "REMOVED",
                "syllabus_status": "REMOVED", "total_appearances": 0,
                "total_questions": 0, "last_appeared": 0,
                "training_years": f"{min_year}-{max_year}",
            })
            continue

        years_appeared = sorted(group["year"].unique())
        total_appearances = len(years_appeared)

        # Questions per year (for weightage model)
        qs_per_year = group.groupby("year").size().to_dict()
        question_types = group["question_type"].value_counts().to_dict()
        difficulty_list = group["difficulty"].dropna().tolist()
        recent_diffs = [d for y, d in zip(group["year"], group["difficulty"]) if y >= max_year - 4]

        # Top micro-topic
        micro_counts = group["micro_topic"].value_counts()
        top_micro = micro_counts.index[0] if len(micro_counts) > 0 else chapter

        # --- STAGE 1: Appearance ---
        app_prob, app_signals = _appearance_probability(
            years_appeared, (min_year, max_year), target_year, max_year
        )
        app_prob *= syl_gate  # Apply syllabus gate

        # --- STAGE 2: Weightage ---
        exp_qs, exp_min, exp_max, wt_conf = _expected_questions(
            qs_per_year, years_appeared, target_year, max_year
        )

        # --- STAGE 3: Format ---
        likely_types, likely_diff, format_dom = _predict_format(
            question_types, difficulty_list, recent_diffs
        )

        # --- Cross-exam signal ---
        micro_cross = cross_df[cross_df["topic"] == chapter]
        exams_present = micro_cross["exam"].unique()
        cross_score = min(len(exams_present) / 3, 1.0)
        app_signals["cross_exam"] = cross_score

        # --- Trend direction ---
        slope = app_signals.get("trend_slope", 0.5)
        if slope > 0.6:
            trend_dir = "RISING"
        elif slope < 0.4:
            trend_dir = "DECLINING"
        else:
            trend_dir = "STABLE"
        if syl_status == "NEW":
            trend_dir = "NEW"

        # --- Confidence ---
        conf_score, conf_label = _confidence_score(
            app_prob, wt_conf, total_appearances, syl_status,
            app_signals.get("recent_3yr", 0), slope
        )

        # --- Recent yield bonus (promotes heavy chapters in last 3 years) ---
        recent_qs = [v for y, v in qs_per_year.items() if y >= max_year - 2]
        recent_yield = np.mean(recent_qs) if recent_qs else 0
        yield_bonus = min(recent_yield / 6.0, 1.0)  # 6+ avg qs recently = max

        # --- Final combined score ---
        # 0.40 * P(appear) + 0.30 * normalized_expected_qs + 0.15 * yield_bonus + 0.10 * cross + 0.05 * syllabus
        normalized_exp_qs = min(exp_qs / 8.0, 1.0)  # 8+ questions = max
        final_score = (
            0.40 * app_prob +
            0.30 * normalized_exp_qs +
            0.15 * yield_bonus +
            0.10 * cross_score +
            0.05 * syl_gate
        )

        # Build reasons
        reasons = []
        if app_prob > 0.6:
            reasons.append(f"High appearance probability ({app_prob:.0%}) — appeared {total_appearances} times")
        if exp_qs >= 3:
            reasons.append(f"Heavy chapter: expected ~{exp_qs:.0f} questions (range {exp_min}-{exp_max})")
        if trend_dir == "RISING":
            reasons.append("Rising trend — increasing frequency recently")
        elif trend_dir == "DECLINING":
            reasons.append("Declining trend — less frequent recently")
        if app_signals.get("gap_return", 0) > 0.5:
            gap = target_year - max(years_appeared)
            reasons.append(f"Gap return signal: not seen in {gap} years")
        if app_signals.get("cycle_match", 0) > 0.5:
            reasons.append("Cycle match — periodic reappearance pattern")
        if cross_score > 0.5:
            reasons.append(f"Cross-exam presence: {', '.join(exams_present)}")
        if syl_status == "NEW":
            reasons.append("Newly added to syllabus — estimated through proxies")
        if not reasons:
            reasons.append("Low signal — limited data")

        # Full signal breakdown
        signal_breakdown = {}
        for k, v in app_signals.items():
            signal_breakdown[k] = {"value": round(v, 3)}
        signal_breakdown["expected_qs"] = {"value": round(exp_qs, 1)}
        signal_breakdown["weightage_confidence"] = {"value": round(wt_conf, 2)}

        predictions.append({
            "chapter": chapter,
            "normalized_chapter": normalized_chapter,
            "subject": subject,
            "appearance_probability": round(app_prob, 3),
            "expected_questions": round(exp_qs, 1),
            "expected_qs_min": exp_min,
            "expected_qs_max": exp_max,
            "likely_formats": likely_types,
            "likely_difficulty": likely_diff,
            "format_dominance": format_dom,
            "confidence": conf_label,
            "confidence_score": conf_score,
            "final_score": round(final_score, 4),
            "signal_breakdown": signal_breakdown,
            "reasons": reasons,
            "top_micro_topic": top_micro,
            "trend_direction": trend_dir,
            "syllabus_status": syl_status,
            "total_appearances": total_appearances,
            "total_questions": len(group),
            "last_appeared": int(max(years_appeared)),
            "training_years": f"{min_year}-{max_year}",
        })

    # Sort by final_score
    predictions.sort(key=lambda x: x["final_score"], reverse=True)

    # Deduplicate by normalized chapter name (keep highest scoring variant)
    seen = set()
    deduped = []
    for p in predictions:
        norm = p["normalized_chapter"].lower()
        if norm not in seen:
            seen.add(norm)
            deduped.append(p)

    # Apply subject-balanced reranking
    active = [p for p in deduped if p["syllabus_status"] != "REMOVED"]
    removed = [p for p in deduped if p["syllabus_status"] == "REMOVED"]

    if exam:
        reranked = _subject_balanced_rerank(active, exam, top_k=top_k)
    else:
        reranked = active[:top_k]

    return reranked + removed


# ================================================================
# MICRO-TOPIC LEVEL PREDICTION (granular)
# ================================================================

def predict_microtopics_v3(db_path="data/exam.db", target_year=2026, exam=None, top_k=100):
    """
    Micro-topic-level 3-stage prediction.

    Like predict_chapters_v3 but grouped at (subject, topic, micro_topic) level.
    Returns list of dicts with: micro_topic, chapter (parent), subject,
    appearance_probability, expected_questions, expected_qs_min/max,
    likely_formats, likely_difficulty, confidence, confidence_score,
    final_score, signal_breakdown, reasons, trend_direction, syllabus_status.
    """
    full_df = get_questions_df(db_path)

    df = full_df[~full_df["year"].isin(HOLDOUT_YEARS)]
    if exam:
        df = df[df["exam"] == exam]
        full_df = full_df[full_df["exam"] == exam]

    if df.empty:
        return []

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    cross_df = get_questions_df(db_path)
    cross_df = cross_df[~cross_df["year"].isin(HOLDOUT_YEARS)]

    # Group at micro-topic level
    micro_groups = df.groupby(["subject", "topic", "micro_topic"])

    predictions = []

    for (subject, chapter, micro_topic), group in micro_groups:
        normalized_chapter = _normalize_chapter(chapter)

        # Syllabus gate is at chapter level
        syl_status, syl_gate = _syllabus_status(chapter, exam)
        if syl_gate == 0.0:
            predictions.append({
                "micro_topic": micro_topic, "chapter": chapter,
                "normalized_chapter": normalized_chapter, "subject": subject,
                "appearance_probability": 0.0, "expected_questions": 0,
                "expected_qs_min": 0, "expected_qs_max": 0,
                "likely_formats": [], "likely_difficulty": 0,
                "format_dominance": 0, "confidence": "HIGH",
                "confidence_score": 0.95, "final_score": 0.0,
                "signal_breakdown": {}, "reasons": ["Removed from syllabus"],
                "trend_direction": "REMOVED", "syllabus_status": "REMOVED",
                "total_appearances": 0, "total_questions": 0,
                "last_appeared": 0, "training_years": f"{min_year}-{max_year}",
            })
            continue

        years_appeared = sorted(group["year"].unique())
        total_appearances = len(years_appeared)
        qs_per_year = group.groupby("year").size().to_dict()
        question_types = group["question_type"].value_counts().to_dict()
        difficulty_list = group["difficulty"].dropna().tolist()
        recent_diffs = [d for y, d in zip(group["year"], group["difficulty"]) if y >= max_year - 4]

        # --- STAGE 1: Appearance ---
        app_prob, app_signals = _appearance_probability(
            years_appeared, (min_year, max_year), target_year, max_year
        )
        app_prob *= syl_gate

        # --- STAGE 2: Weightage ---
        exp_qs, exp_min, exp_max, wt_conf = _expected_questions(
            qs_per_year, years_appeared, target_year, max_year
        )

        # --- STAGE 3: Format ---
        likely_types, likely_diff, format_dom = _predict_format(
            question_types, difficulty_list, recent_diffs
        )

        # --- Cross-exam signal (at chapter level for micro-topics) ---
        ch_cross = cross_df[cross_df["topic"] == chapter]
        exams_present = ch_cross["exam"].unique()
        cross_score = min(len(exams_present) / 3, 1.0)
        app_signals["cross_exam"] = cross_score

        # --- Trend direction ---
        slope = app_signals.get("trend_slope", 0.5)
        if slope > 0.6:
            trend_dir = "RISING"
        elif slope < 0.4:
            trend_dir = "DECLINING"
        else:
            trend_dir = "STABLE"
        if syl_status == "NEW":
            trend_dir = "NEW"

        # --- Confidence ---
        conf_score, conf_label = _confidence_score(
            app_prob, wt_conf, total_appearances, syl_status,
            app_signals.get("recent_3yr", 0), slope
        )

        # --- Recent yield bonus ---
        recent_qs_mt = [v for y, v in qs_per_year.items() if y >= max_year - 2]
        recent_yield_mt = np.mean(recent_qs_mt) if recent_qs_mt else 0
        yield_bonus_mt = min(recent_yield_mt / 3.0, 1.0)  # 3+ avg qs recently = max

        # --- Final score ---
        normalized_exp_qs = min(exp_qs / 4.0, 1.0)  # micro-topics: 4+ qs = max
        final_score = (
            0.40 * app_prob +
            0.30 * normalized_exp_qs +
            0.15 * yield_bonus_mt +
            0.10 * cross_score +
            0.05 * syl_gate
        )

        # Build reasons
        reasons = []
        if app_prob > 0.6:
            reasons.append(f"High appearance probability ({app_prob:.0%}) — appeared {total_appearances} times")
        if exp_qs >= 2:
            reasons.append(f"Recurring micro-topic: expected ~{exp_qs:.0f} questions (range {exp_min}-{exp_max})")
        if trend_dir == "RISING":
            reasons.append("Rising trend — increasing frequency recently")
        elif trend_dir == "DECLINING":
            reasons.append("Declining trend — less frequent recently")
        if app_signals.get("gap_return", 0) > 0.5:
            gap = target_year - max(years_appeared)
            reasons.append(f"Gap return signal: not seen in {gap} years")
        if app_signals.get("cycle_match", 0) > 0.5:
            reasons.append("Cycle match — periodic reappearance pattern")
        if cross_score > 0.5:
            reasons.append(f"Cross-exam presence in: {', '.join(exams_present)}")
        if syl_status == "NEW":
            reasons.append("Parent chapter newly added — estimated via proxies")
        if not reasons:
            reasons.append("Low signal — limited data for this micro-topic")

        signal_breakdown = {}
        for k, v in app_signals.items():
            signal_breakdown[k] = {"value": round(v, 3)}
        signal_breakdown["expected_qs"] = {"value": round(exp_qs, 1)}
        signal_breakdown["weightage_confidence"] = {"value": round(wt_conf, 2)}

        predictions.append({
            "micro_topic": micro_topic,
            "chapter": chapter,
            "normalized_chapter": normalized_chapter,
            "subject": subject,
            "appearance_probability": round(app_prob, 3),
            "expected_questions": round(exp_qs, 1),
            "expected_qs_min": exp_min,
            "expected_qs_max": exp_max,
            "likely_formats": likely_types,
            "likely_difficulty": likely_diff,
            "format_dominance": format_dom,
            "confidence": conf_label,
            "confidence_score": conf_score,
            "final_score": round(final_score, 4),
            "signal_breakdown": signal_breakdown,
            "reasons": reasons,
            "trend_direction": trend_dir,
            "syllabus_status": syl_status,
            "total_appearances": total_appearances,
            "total_questions": len(group),
            "last_appeared": int(max(years_appeared)),
            "training_years": f"{min_year}-{max_year}",
        })

    # Sort by final_score
    predictions.sort(key=lambda x: x["final_score"], reverse=True)

    # Apply subject-balanced reranking
    active = [p for p in predictions if p["syllabus_status"] != "REMOVED"]
    removed = [p for p in predictions if p["syllabus_status"] == "REMOVED"]

    if exam:
        reranked = _subject_balanced_rerank(active, exam, top_k=top_k)
    else:
        reranked = active[:top_k]

    return reranked + removed


# ================================================================
# BACKTESTING V3
# ================================================================

def backtest_v3(db_path="data/exam.db", test_years=None, exam=None, k=50):
    """
    Backtest with new metrics:
      - precision@K (chapters)
      - coverage@K (fraction of actual paper questions covered)
      - heavy_topic_recall (chapters with 3+ questions)
      - subject_coverage (per-subject coverage balance)
      - unique_chapters (how many unique chapters in top-K)
      - rank_correlation (predicted importance vs actual)
    """
    if test_years is None:
        test_years = [2019, 2020, 2021, 2022, 2023]

    full_df = get_questions_df(db_path)
    if exam:
        full_df = full_df[full_df["exam"] == exam]

    results = []

    for test_year in test_years:
        actual = full_df[full_df["year"] == test_year]
        if actual.empty:
            continue

        actual_chapters = set(actual["topic"].unique())
        actual_qs_per_chapter = actual.groupby("topic").size().to_dict()
        actual_total_qs = len(actual)

        # Heavy chapters (3+ questions)
        heavy_chapters = {ch for ch, count in actual_qs_per_chapter.items() if count >= 3}

        # Actual subject distribution
        actual_subj_qs = actual.groupby("subject").size().to_dict()
        actual_total = sum(actual_subj_qs.values())

        # Temporarily set holdout
        import analysis.predictor_v3 as self_mod
        orig = self_mod.HOLDOUT_YEARS
        self_mod.HOLDOUT_YEARS = {y for y in range(test_year, 2030)}

        preds = predict_chapters_v3(db_path, target_year=test_year, exam=exam, top_k=k)

        self_mod.HOLDOUT_YEARS = orig

        if not preds:
            continue

        pred_chapters = [p["chapter"] for p in preds if p["syllabus_status"] != "REMOVED"][:k]
        pred_set = set(pred_chapters)

        # --- Metrics ---

        # Precision@K
        hits = pred_set & actual_chapters
        precision = len(hits) / k if k > 0 else 0

        # Coverage@K (questions covered)
        covered_qs = sum(actual_qs_per_chapter.get(ch, 0) for ch in pred_set)
        coverage = covered_qs / actual_total_qs if actual_total_qs > 0 else 0

        # Heavy-topic recall
        heavy_hits = pred_set & heavy_chapters
        heavy_recall = len(heavy_hits) / len(heavy_chapters) if heavy_chapters else 0

        # Subject coverage
        pred_subj_qs = {}
        for ch in pred_set:
            ch_qs = actual_qs_per_chapter.get(ch, 0)
            ch_subj = actual[actual["topic"] == ch]["subject"].mode()
            if len(ch_subj) > 0:
                s = ch_subj.iloc[0]
                pred_subj_qs[s] = pred_subj_qs.get(s, 0) + ch_qs

        subj_coverage = {}
        for s, qs in actual_subj_qs.items():
            covered = pred_subj_qs.get(s, 0)
            subj_coverage[s] = round(covered / qs, 3) if qs > 0 else 0

        # Average subject coverage
        avg_subj_cov = np.mean(list(subj_coverage.values())) if subj_coverage else 0

        # Unique chapters
        unique_count = len(pred_set)

        # Rank correlation: predicted rank vs actual question count
        from scipy.stats import spearmanr
        pred_ranks = {ch: i for i, ch in enumerate(pred_chapters)}
        common = pred_set & actual_chapters
        if len(common) >= 5:
            pred_r = [pred_ranks[ch] for ch in common]
            actual_r = [actual_qs_per_chapter[ch] for ch in common]
            rank_corr, _ = spearmanr(pred_r, [-x for x in actual_r])  # negate: lower rank = better
        else:
            rank_corr = 0

        # Combined score
        combined = (0.35 * precision + 0.40 * coverage +
                    0.15 * heavy_recall + 0.10 * avg_subj_cov)

        results.append({
            "test_year": test_year,
            "actual_chapters": len(actual_chapters),
            "actual_questions": actual_total_qs,
            "precision_at_k": round(precision, 3),
            "coverage_at_k": round(coverage, 3),
            "heavy_topic_recall": round(heavy_recall, 3),
            "subject_coverage": subj_coverage,
            "avg_subject_coverage": round(avg_subj_cov, 3),
            "unique_chapters": unique_count,
            "rank_correlation": round(rank_corr, 3),
            "combined_score": round(combined, 3),
            "k": k,
            "questions_covered": covered_qs,
            "heavy_topics_hit": len(heavy_hits),
            "heavy_topics_total": len(heavy_chapters),
        })

    return results


# ================================================================
# INTERACTIVE SINGLE-YEAR BACKTEST
# ================================================================

def backtest_single_year(db_path="data/exam.db", test_year=2020, exam=None,
                         k=50, level="chapter"):
    """
    Train on data strictly before test_year, predict for test_year,
    compare predictions against actual paper.

    level: "chapter" or "micro"

    Returns (summary_dict, actual_df) where summary_dict has:
      - precision_at_k, coverage_at_k, heavy_topic_recall
      - subject_coverage per subject
      - combined_score (0.35P + 0.40C + 0.15H + 0.10S)
      - hit_topics, missed_topics, false_positives
      - per-topic breakdown with predicted_rank, actual_qs
    """
    import analysis.predictor_v3 as self_mod
    from scipy.stats import spearmanr

    full_df = get_questions_df(db_path)
    if exam:
        full_df = full_df[full_df["exam"] == exam]

    actual = full_df[full_df["year"] == test_year]
    if actual.empty:
        return None, None

    # Override holdout: exclude test_year and everything after
    orig_holdout = self_mod.HOLDOUT_YEARS
    self_mod.HOLDOUT_YEARS = set(range(test_year, 2030))

    try:
        if level == "micro":
            preds = predict_microtopics_v3(db_path, target_year=test_year, exam=exam, top_k=k)
        else:
            preds = predict_chapters_v3(db_path, target_year=test_year, exam=exam, top_k=k)
    finally:
        self_mod.HOLDOUT_YEARS = orig_holdout

    if not preds:
        return None, actual

    active_preds = [p for p in preds if p["syllabus_status"] != "REMOVED"][:k]

    # Build actual paper stats
    if level == "micro":
        actual_key = "micro_topic"
        pred_key = "micro_topic"
    else:
        actual_key = "topic"
        pred_key = "chapter"

    actual_set = set(actual[actual_key].unique())
    actual_qs_map = actual.groupby(actual_key).size().to_dict()
    actual_subj_qs = actual.groupby("subject").size().to_dict()
    actual_total = len(actual)

    pred_list_names = [p[pred_key] for p in active_preds]
    pred_set = set(pred_list_names)
    pred_ranks = {name: i + 1 for i, name in enumerate(pred_list_names)}

    # Core metrics
    hits = pred_set & actual_set
    missed = actual_set - pred_set
    false_pos = pred_set - actual_set

    precision = len(hits) / k if k > 0 else 0

    covered_qs = sum(actual_qs_map.get(t, 0) for t in pred_set)
    coverage = covered_qs / actual_total if actual_total > 0 else 0

    heavy_actual = {t for t, c in actual_qs_map.items() if c >= 3}
    heavy_hits = pred_set & heavy_actual
    heavy_recall = len(heavy_hits) / len(heavy_actual) if heavy_actual else 0

    # Subject coverage
    subj_covered_qs = {}
    for t in pred_set:
        qs = actual_qs_map.get(t, 0)
        subj = actual[actual[actual_key] == t]["subject"].mode()
        if len(subj) > 0:
            s = subj.iloc[0]
            subj_covered_qs[s] = subj_covered_qs.get(s, 0) + qs

    subj_coverage = {}
    for s, qs in actual_subj_qs.items():
        subj_coverage[s] = round(subj_covered_qs.get(s, 0) / qs, 3) if qs > 0 else 0

    avg_subj_cov = np.mean(list(subj_coverage.values())) if subj_coverage else 0

    combined = 0.35 * precision + 0.40 * coverage + 0.15 * heavy_recall + 0.10 * avg_subj_cov

    # Rank correlation
    common = hits
    rank_corr = 0.0
    if len(common) >= 5:
        pred_r = [pred_ranks[t] for t in common]
        actual_r = [actual_qs_map[t] for t in common]
        rank_corr, _ = spearmanr(pred_r, [-x for x in actual_r])

    # Per-topic breakdown (for hit/miss table)
    topic_breakdown = []
    for t in sorted(actual_set, key=lambda x: -actual_qs_map.get(x, 0)):
        status = "HIT" if t in pred_set else "MISSED"
        rank = pred_ranks.get(t, None)
        # Get subject for this topic
        t_subj = actual[actual[actual_key] == t]["subject"].mode()
        t_subj = t_subj.iloc[0] if len(t_subj) > 0 else "Unknown"
        topic_breakdown.append({
            "topic": t,
            "subject": t_subj,
            "actual_qs": actual_qs_map.get(t, 0),
            "status": status,
            "predicted_rank": rank,
            "is_heavy": t in heavy_actual,
        })

    # False positives (predicted but not asked)
    fp_breakdown = []
    for p in active_preds:
        name = p[pred_key]
        if name not in actual_set:
            fp_breakdown.append({
                "topic": name,
                "subject": p["subject"],
                "predicted_rank": pred_ranks.get(name, 0),
                "appearance_prob": p["appearance_probability"],
                "confidence": p["confidence"],
            })

    summary = {
        "test_year": test_year,
        "exam": exam or "All",
        "level": level,
        "k": k,
        "precision_at_k": round(precision, 3),
        "coverage_at_k": round(coverage, 3),
        "heavy_topic_recall": round(heavy_recall, 3),
        "avg_subject_coverage": round(avg_subj_cov, 3),
        "rank_correlation": round(rank_corr, 3),
        "combined_score": round(combined, 3),
        "hits": len(hits),
        "misses": len(missed),
        "false_positives": len(false_pos),
        "actual_topics": len(actual_set),
        "actual_questions": actual_total,
        "questions_covered": covered_qs,
        "heavy_topics_hit": len(heavy_hits),
        "heavy_topics_total": len(heavy_actual),
        "subject_coverage": subj_coverage,
        "topic_breakdown": topic_breakdown,
        "fp_breakdown": fp_breakdown,
    }

    return summary, actual
