# Predictor v4 Micro-Topic Accuracy Engine — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Improve NEET micro-topic prediction from combined score ~0.53 → ~0.68+ by implementing a hierarchical two-stage engine with 3 new signals and a hill-climbing weight optimiser.

**Architecture:** Stage 1 runs `predict_chapters_v3()` as a chapter gate (~94% precision). Stage 2 scores micro-topics *conditionally within* each predicted chapter's bucket using 10 signals (7 original + 3 new: parent_inheritance, recency_burst, dispersion). Stage 3 loads weight-optimised coefficients from `weights_cache.json` at predict time.

**Tech Stack:** Python 3.9, numpy, pandas, scipy, pytest, existing `utils/db.py` + `analysis/predictor_v3.py`

**Run tests with:** `export PATH="$PATH:/Users/aman/Library/Python/3.9/bin" && pytest tests/ -q`

**Baseline to preserve:** 16 passing tests in `tests/`. Chapter-level combined score ≥ 0.820.

---

## Task 1: Weight management + 3 new signal functions

**Files:**
- Create: `analysis/predictor_v4.py`
- Create: `tests/test_predictor_v4.py`

### Step 1: Write failing tests for weight loading and new signals

```python
# tests/test_predictor_v4.py
import os, json, pytest
import numpy as np

# We import after creation, so use importlib for safety
def _import():
    from analysis.predictor_v4 import (
        _load_weights, _normalise_weights,
        DEFAULT_SIGNAL_WEIGHTS, DEFAULT_FINAL_WEIGHTS,
        _appearance_probability_v4,
        WEIGHTS_CACHE_PATH,
    )
    return _load_weights, _normalise_weights, DEFAULT_SIGNAL_WEIGHTS, DEFAULT_FINAL_WEIGHTS, _appearance_probability_v4, WEIGHTS_CACHE_PATH


def test_default_weights_sum_to_one():
    _, _norm, sw, fw, _, _ = _import()
    assert abs(sum(sw.values()) - 1.0) < 0.01
    assert abs(sum(fw.values()) - 1.0) < 0.01


def test_normalise_weights_clamps_negatives():
    _, _norm, _, _, _, _ = _import()
    w = {"a": -0.5, "b": 0.5}
    normed = _norm(w)
    assert all(v >= 0.01 for v in normed.values())
    assert abs(sum(normed.values()) - 1.0) < 1e-6


def test_load_weights_falls_back_to_defaults_when_no_cache(tmp_path, monkeypatch):
    import analysis.predictor_v4 as v4
    monkeypatch.setattr(v4, "WEIGHTS_CACHE_PATH", str(tmp_path / "no_such_file.json"))
    sw, fw = v4._load_weights()
    assert "parent_inheritance" in sw
    assert "app_prob" in fw


def test_load_weights_reads_cache(tmp_path, monkeypatch):
    import analysis.predictor_v4 as v4
    cache = {
        "signal_weights": {"parent_inheritance": 0.5, "recency_freq": 0.5},
        "final_weights":  {"app_prob": 1.0},
    }
    p = tmp_path / "weights_cache.json"
    p.write_text(json.dumps(cache))
    monkeypatch.setattr(v4, "WEIGHTS_CACHE_PATH", str(p))
    sw, fw = v4._load_weights()
    assert sw["parent_inheritance"] == 0.5


def test_new_signals_in_appearance_probability():
    _, _, _, _, app_prob_v4, _ = _import()
    years = list(range(2010, 2022))
    prob, signals = app_prob_v4(
        years_appeared=years,
        total_years_range=(2000, 2022),
        target_year=2023,
        max_year=2022,
        parent_score=0.8,
        total_questions=30,
        signal_weights={
            "recency_freq": 0.10, "appearance_rate": 0.10,
            "recent_3yr": 0.10, "recent_5yr": 0.05,
            "gap_return": 0.05, "trend_slope": 0.05, "cycle_match": 0.05,
            "parent_inheritance": 0.20, "recency_burst": 0.15, "dispersion": 0.15,
        }
    )
    assert 0.0 <= prob <= 1.0
    assert "parent_inheritance" in signals
    assert "recency_burst" in signals
    assert "dispersion" in signals
    # parent_inheritance should equal parent_score
    assert signals["parent_inheritance"] == pytest.approx(0.8)


def test_dispersion_rewards_dense_appearances():
    _, _, _, _, app_prob_v4, _ = _import()
    flat_weights = {k: 0.10 for k in [
        "recency_freq","appearance_rate","recent_3yr","recent_5yr",
        "gap_return","trend_slope","cycle_match",
        "parent_inheritance","recency_burst","dispersion"
    ]}
    # 20 total questions / 10 years = 2 Qs/yr
    _, signals_dense = app_prob_v4(
        list(range(2012,2022)), (2000,2022), 2023, 2022, 0.5, 20, flat_weights
    )
    # 10 total questions / 10 years = 1 Qs/yr
    _, signals_thin = app_prob_v4(
        list(range(2012,2022)), (2000,2022), 2023, 2022, 0.5, 10, flat_weights
    )
    assert signals_dense["dispersion"] > signals_thin["dispersion"]
```

### Step 2: Run tests — expect ImportError (file doesn't exist yet)

```bash
cd /Users/aman/exam-predictor
export PATH="$PATH:/Users/aman/Library/Python/3.9/bin"
pytest tests/test_predictor_v4.py -v 2>&1 | head -20
```
Expected: `ModuleNotFoundError: No module named 'analysis.predictor_v4'`

### Step 3: Create `analysis/predictor_v4.py` with weight management + new signal function

```python
# analysis/predictor_v4.py
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
# Normalise on definition so they always sum to 1
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
      recency_burst      — appeared in 2+ of last 3 years → strong signal
      dispersion         — rewards 2-3 Qs/appearance (not thin 1-Q scatter)

    Args:
        years_appeared:    list of years this micro-topic appeared
        total_years_range: (min_year, max_year) of training data
        target_year:       year to predict for
        max_year:          last year in training data
        parent_score:      parent chapter's final_score (0-1)
        total_questions:   total questions for this micro-topic in training data
        signal_weights:    dict of 10 signal weights (must sum to 1)

    Returns:
        (probability: float, signals: dict)
    """
    if not years_appeared:
        return 0.0, {}

    signals = {}

    # ── Original 7 signals (same formulas as predictor_v3) ──────────────────

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

    # ── 3 new signals ────────────────────────────────────────────────────────

    # Signal 8: Parent chapter confidence flows down
    signals["parent_inheritance"] = float(parent_score)

    # Signal 9: Recency burst — same numerator as recent_3yr but independent weight
    signals["recency_burst"] = min(recent_3 / 3.0, 1.0)

    # Signal 10: Dispersion — rewards heavy/dense appearances (2-3 Qs/yr)
    avg_qs_when_present = total_questions / max(len(set(years_appeared)), 1)
    signals["dispersion"] = min(avg_qs_when_present / 3.0, 1.0)

    # ── Weighted combination ─────────────────────────────────────────────────
    score = sum(signals[k] * signal_weights.get(k, 0.0) for k in signals)
    return min(score * 1.8, 0.99), signals
```

### Step 4: Run tests — expect them to pass

```bash
cd /Users/aman/exam-predictor
export PATH="$PATH:/Users/aman/Library/Python/3.9/bin"
pytest tests/test_predictor_v4.py -v
```
Expected: **6 passed**

### Step 5: Commit

```bash
git add analysis/predictor_v4.py tests/test_predictor_v4.py
git commit -m "feat: predictor v4 — weight management and 10-signal appearance function"
```

---

## Task 2: `predict_microtopics_v4()` — hierarchical Stage 1+2 pipeline

**Files:**
- Modify: `analysis/predictor_v4.py` (append function)
- Modify: `tests/test_predictor_v4.py` (append tests)

### Step 1: Append integration tests

```python
# Append to tests/test_predictor_v4.py

def _make_db(tmp_path):
    """Helper: create a minimal test DB with known data."""
    from utils.db import init_db, insert_questions
    db = str(tmp_path / "test_v4.db")
    init_db(db)
    qs = []
    # Chapter A: heavy (appears every year 2010-2022, multiple micro-topics)
    for yr in range(2010, 2023):
        qs.append({
            "id": f"A_{yr}_1", "exam": "NEET", "year": yr, "shift": "P1",
            "subject": "Biology", "topic": "Genetics",
            "micro_topic": "Mendelian Inheritance",
            "question_text": "q", "question_type": "MCQ_single",
            "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4,
        })
        qs.append({
            "id": f"A_{yr}_2", "exam": "NEET", "year": yr, "shift": "P1",
            "subject": "Biology", "topic": "Genetics",
            "micro_topic": "Chromosomal Theory",
            "question_text": "q", "question_type": "MCQ_single",
            "difficulty": 3, "concepts_tested": [], "answer": "B", "marks": 4,
        })
    # Chapter B: sparse (appears only 3 years)
    for yr in [2015, 2018, 2021]:
        qs.append({
            "id": f"B_{yr}", "exam": "NEET", "year": yr, "shift": "P1",
            "subject": "Physics", "topic": "Optics",
            "micro_topic": "Interference",
            "question_text": "q", "question_type": "MCQ_single",
            "difficulty": 3, "concepts_tested": [], "answer": "C", "marks": 4,
        })
    insert_questions(db, qs)
    return db


def test_predict_microtopics_v4_returns_list(tmp_path):
    from analysis.predictor_v4 import predict_microtopics_v4
    db = _make_db(tmp_path)
    results = predict_microtopics_v4(db, target_year=2024, exam="NEET", top_k=50)
    assert isinstance(results, list)
    assert len(results) > 0


def test_predict_microtopics_v4_output_schema(tmp_path):
    from analysis.predictor_v4 import predict_microtopics_v4
    db = _make_db(tmp_path)
    results = predict_microtopics_v4(db, target_year=2024, exam="NEET", top_k=50)
    required = {
        "micro_topic", "chapter", "subject",
        "appearance_probability", "expected_questions",
        "final_score", "trend_direction", "parent_final_score",
    }
    for key in required:
        assert key in results[0], f"Missing key: {key}"


def test_heavy_chapter_micros_rank_above_sparse(tmp_path):
    from analysis.predictor_v4 import predict_microtopics_v4
    db = _make_db(tmp_path)
    results = predict_microtopics_v4(db, target_year=2024, exam="NEET", top_k=50)
    scores = {r["micro_topic"]: r["final_score"] for r in results}
    # Mendelian Inheritance (13 years) should score higher than Interference (3 years)
    assert scores.get("Mendelian Inheritance", 0) > scores.get("Interference", 0), \
        "Heavy chapter micro-topic should outrank sparse chapter micro-topic"


def test_parent_score_propagated(tmp_path):
    from analysis.predictor_v4 import predict_microtopics_v4
    db = _make_db(tmp_path)
    results = predict_microtopics_v4(db, target_year=2024, exam="NEET", top_k=50)
    for r in results:
        assert 0 <= r["parent_final_score"] <= 1.0
        assert r["parent_final_score"] > 0, "Parent score should be >0 (chapter was in gate)"


def test_scores_sorted_descending(tmp_path):
    from analysis.predictor_v4 import predict_microtopics_v4
    db = _make_db(tmp_path)
    results = predict_microtopics_v4(db, target_year=2024, exam="NEET", top_k=50)
    scores = [r["final_score"] for r in results]
    assert scores == sorted(scores, reverse=True)
```

### Step 2: Run — expect failures (function not yet implemented)

```bash
cd /Users/aman/exam-predictor
export PATH="$PATH:/Users/aman/Library/Python/3.9/bin"
pytest tests/test_predictor_v4.py::test_predict_microtopics_v4_returns_list -v
```
Expected: `ImportError` or `AttributeError: module has no attribute predict_microtopics_v4`

### Step 3: Append `predict_microtopics_v4()` to `analysis/predictor_v4.py`

```python
# Append to analysis/predictor_v4.py  (after _appearance_probability_v4)

def predict_microtopics_v4(db_path="data/exam.db", target_year=2026, exam=None,
                            top_k=200, chapter_k=70,
                            signal_weights=None, final_weights=None):
    """
    Hierarchical micro-topic prediction (v4).

    Stage 1: predict_chapters_v3() → top chapter_k chapters with scores.
    Stage 2: For each predicted chapter, score all its micro-topics using
             _appearance_probability_v4() (10 signals including parent_inheritance).
             Micro-topics whose parent chapter is NOT in top chapter_k are excluded.

    Args:
        db_path:         path to exam.db
        target_year:     year to predict for (default 2026)
        exam:            "NEET" | "JEE Main" | None for all
        top_k:           max micro-topics to return (default 200)
        chapter_k:       chapter gate size — how many chapters to allow (default 70)
        signal_weights:  override signal weights (default: load from cache)
        final_weights:   override final-score weights (default: load from cache)

    Returns:
        list of dicts sorted by final_score descending
    """
    # Load weights (from cache or defaults)
    if signal_weights is None or final_weights is None:
        cached_sw, cached_fw = _load_weights()
        signal_weights = signal_weights if signal_weights is not None else cached_sw
        final_weights  = final_weights  if final_weights  is not None else cached_fw

    # ── Stage 1: Chapter gate ────────────────────────────────────────────────
    chapter_preds = predict_chapters_v3(
        db_path, target_year=target_year, exam=exam, top_k=chapter_k
    )

    # Build lookup: canonical lower-case chapter name → final_score
    chapter_score_map = {}
    for cp in chapter_preds:
        if cp.get("syllabus_status") != "REMOVED":
            chapter_score_map[cp["chapter"].lower()] = cp["final_score"]
            norm = cp.get("normalized_chapter", "")
            if norm:
                chapter_score_map[norm.lower()] = cp["final_score"]

    # ── Stage 2: Conditional micro scoring ──────────────────────────────────
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
        # Parent gate — skip micro-topics whose chapter didn't make the cut
        normalized = _normalize_chapter(chapter)
        parent_score = chapter_score_map.get(
            chapter.lower(),
            chapter_score_map.get(normalized.lower(), 0.0)
        )
        if parent_score == 0.0:
            continue

        # Syllabus gate
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

        # Appearance probability (10 signals)
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

        # Cross-exam signal
        cross_score = min(
            cross_df[cross_df["micro_topic"] == micro_topic]["exam"].nunique() / 3, 1.0
        )
        app_signals["cross_exam"] = cross_score

        # Trend direction
        slope = app_signals.get("trend_slope", 0.5)
        if slope > 0.6:
            trend_dir = "RISING"
        elif slope < 0.4:
            trend_dir = "DECLINING"
        else:
            trend_dir = "STABLE"
        if syl_status == "NEW":
            trend_dir = "NEW"

        # Yield bonus
        recent_qs    = [v for y, v in qs_per_year.items() if y >= max_year - 2]
        yield_bonus  = min(np.mean(recent_qs) / 3.0, 1.0) if recent_qs else 0.0

        # Final score (parameterised, normalised)
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
```

### Step 4: Run all new tests

```bash
cd /Users/aman/exam-predictor
export PATH="$PATH:/Users/aman/Library/Python/3.9/bin"
pytest tests/test_predictor_v4.py -v
```
Expected: **11 passed**

### Step 5: Run full test suite to confirm no regression

```bash
pytest tests/ -q
```
Expected: **22 passed** (16 original + 6 Task 1 + 5 new Task 2 = but some are integration so exact count may vary — all must pass)

### Step 6: Smoke-test on real DB

```bash
cd /Users/aman/exam-predictor
python3 -c "
from analysis.predictor_v4 import predict_microtopics_v4
results = predict_microtopics_v4('data/exam.db', target_year=2026, exam='NEET', top_k=10)
for r in results[:5]:
    print(r['micro_topic'], '|', r['chapter'], '| score:', r['final_score'], '| parent:', r['parent_final_score'])
"
```
Expected: 5 rows printed, all with `parent_final_score > 0`

### Step 7: Commit

```bash
git add analysis/predictor_v4.py tests/test_predictor_v4.py
git commit -m "feat: predictor v4 — hierarchical predict_microtopics_v4 with parent gate + 3 new signals"
```

---

## Task 3: Backtest v4 and measure baseline improvement

**Files:**
- Create: `analysis/backtest_v4.py` (standalone script, no tests needed — output is the metric)

### Step 1: Create the backtest comparison script

```python
# analysis/backtest_v4.py
"""
Compare predictor v3 vs v4 micro-topic combined scores across 2019-2023.
Run: python3 -m analysis.backtest_v4 --exam NEET --k 200
"""
import argparse
import numpy as np
import analysis.predictor_v4 as v4_mod
from utils.db import get_questions_df

DB = "data/exam.db"


def _score_preds(preds, actual_df, k):
    pred_micros = set(p["micro_topic"] for p in preds[:k])
    actual_micros = set(actual_df["micro_topic"].unique())
    qs_map = actual_df.groupby("micro_topic").size().to_dict()
    total = len(actual_df)
    heavy = {m for m, q in qs_map.items() if q >= 3}

    hits     = pred_micros & actual_micros
    precision = len(hits) / k
    coverage  = sum(qs_map.get(m, 0) for m in hits) / total
    heavy_r   = len(hits & heavy) / len(heavy) if heavy else 0

    subj_qs = actual_df.groupby("subject").size().to_dict()
    pred_subj = {}
    for m in hits:
        qs = qs_map.get(m, 0)
        s_series = actual_df[actual_df["micro_topic"] == m]["subject"].mode()
        if len(s_series):
            pred_subj[s_series.iloc[0]] = pred_subj.get(s_series.iloc[0], 0) + qs
    avg_sc = np.mean([pred_subj.get(s, 0) / q for s, q in subj_qs.items()]) if subj_qs else 0

    return {
        "precision": round(precision, 3),
        "coverage":  round(coverage, 3),
        "heavy_r":   round(heavy_r, 3),
        "avg_sc":    round(avg_sc, 3),
        "combined":  round(0.35 * precision + 0.40 * coverage + 0.15 * heavy_r + 0.10 * avg_sc, 3),
    }


def run_comparison(exam="NEET", test_years=None, k=200):
    if test_years is None:
        test_years = [2019, 2020, 2021, 2022, 2023]

    full_df = get_questions_df(DB)
    if exam:
        full_df = full_df[full_df["exam"] == exam]

    orig_holdout = v4_mod.HOLDOUT_YEARS
    print(f"\n{'Year':<6} {'v3_comb':>9} {'v4_comb':>9} {'delta':>8}  coverage_v3  coverage_v4")
    print("-" * 60)

    v3_all, v4_all = [], []

    for yr in test_years:
        actual = full_df[full_df["year"] == yr]
        if actual.empty:
            continue

        v4_mod.HOLDOUT_YEARS = set(range(yr, 2030))
        try:
            # v3 micro-level
            from analysis.predictor_v3 import predict_microtopics_v3
            p_v3 = predict_microtopics_v3(DB, target_year=yr, exam=exam, top_k=k)
            # v4 micro-level
            from analysis.predictor_v4 import predict_microtopics_v4
            p_v4 = predict_microtopics_v4(DB, target_year=yr, exam=exam, top_k=k)
        finally:
            v4_mod.HOLDOUT_YEARS = orig_holdout

        s3 = _score_preds(p_v3, actual, k)
        s4 = _score_preds(p_v4, actual, k)
        delta = s4["combined"] - s3["combined"]
        print(f"{yr:<6} {s3['combined']:>9.3f} {s4['combined']:>9.3f} {delta:>+8.3f}  "
              f"{s3['coverage']:>11.3f}  {s4['coverage']:>11.3f}")
        v3_all.append(s3["combined"])
        v4_all.append(s4["combined"])

    print("-" * 60)
    print(f"{'AVG':<6} {np.mean(v3_all):>9.3f} {np.mean(v4_all):>9.3f} "
          f"{np.mean(v4_all)-np.mean(v3_all):>+8.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", default="NEET")
    parser.add_argument("--k", type=int, default=200)
    args = parser.parse_args()
    run_comparison(exam=args.exam, k=args.k)
```

### Step 2: Run baseline comparison (v3 vs v4 before weight optimisation)

```bash
cd /Users/aman/exam-predictor
python3 -m analysis.backtest_v4 --exam NEET --k 200
```
Expected output format:
```
Year   v3_comb  v4_comb    delta  coverage_v3  coverage_v4
------------------------------------------------------------
2019     0.452    0.XXX   +X.XXX        0.412        0.XXX
...
AVG      0.530    0.XXX   +X.XXX
```
Record the AVG v4_comb — this is the pre-optimisation baseline.

### Step 3: Commit

```bash
git add analysis/backtest_v4.py
git commit -m "feat: backtest_v4 comparison script — v3 vs v4 micro combined scores"
```

---

## Task 4: Weight optimiser — hill-climbing over signal + final weights

**Files:**
- Create: `analysis/weight_optimiser.py`
- Create: `tests/test_weight_optimiser.py`

### Step 1: Write failing tests

```python
# tests/test_weight_optimiser.py
import os, json, pytest


def test_optimiser_improves_over_baseline(tmp_path):
    """
    On a tiny synthetic DB, optimiser must produce a score >= starting score.
    (We only verify it doesn't regress — not that it hits a specific value.)
    """
    from utils.db import init_db, insert_questions
    from analysis.weight_optimiser import optimise_weights, _backtest_score
    from analysis.predictor_v4 import DEFAULT_SIGNAL_WEIGHTS, DEFAULT_FINAL_WEIGHTS

    db = str(tmp_path / "opt_test.db")
    init_db(db)
    qs = []
    for yr in range(2010, 2022):
        for mt in ["Mitosis", "Meiosis", "Dna Replication"]:
            qs.append({
                "id": f"{mt[:3]}_{yr}", "exam": "NEET", "year": yr, "shift": "P1",
                "subject": "Biology", "topic": "Cell Division", "micro_topic": mt,
                "question_text": "q", "question_type": "MCQ_single",
                "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4,
            })
    insert_questions(db, qs)

    base_score = _backtest_score(db, "NEET", [2020, 2021],
                                  DEFAULT_SIGNAL_WEIGHTS, DEFAULT_FINAL_WEIGHTS, k=20)
    sw, fw, train_score, val_score = optimise_weights(
        db_path=db, exam="NEET", n_rounds=3, delta=0.05, k=20, verbose=False
    )
    assert train_score >= base_score - 0.01  # must not regress


def test_optimiser_saves_cache(tmp_path, monkeypatch):
    from utils.db import init_db, insert_questions
    from analysis.weight_optimiser import optimise_weights
    import analysis.predictor_v4 as v4

    cache_path = str(tmp_path / "weights_cache.json")
    monkeypatch.setattr(v4, "WEIGHTS_CACHE_PATH", cache_path)

    db = str(tmp_path / "opt_test2.db")
    init_db(db)
    qs = [{"id": f"q{i}", "exam": "NEET", "year": 2015+i, "shift": "P1",
            "subject": "Physics", "topic": "Optics", "micro_topic": "Lenses",
            "question_text": "q", "question_type": "MCQ_single",
            "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4}
           for i in range(6)]
    insert_questions(db, qs)

    sw, fw, ts, vs = optimise_weights(db_path=db, exam="NEET", n_rounds=1,
                                       delta=0.05, k=10, verbose=False,
                                       cache_path=cache_path)
    assert os.path.exists(cache_path)
    with open(cache_path) as f:
        saved = json.load(f)
    assert "signal_weights" in saved
    assert "final_weights" in saved
    assert "train_score" in saved
```

### Step 2: Run tests — expect ImportError

```bash
cd /Users/aman/exam-predictor
export PATH="$PATH:/Users/aman/Library/Python/3.9/bin"
pytest tests/test_weight_optimiser.py -v
```
Expected: `ModuleNotFoundError: No module named 'analysis.weight_optimiser'`

### Step 3: Create `analysis/weight_optimiser.py`

```python
# analysis/weight_optimiser.py
"""
Hill-climbing weight optimiser for PRAJNA Predictor v4.

Algorithm: Coordinate-wise ascent — perturb one weight at a time by ±delta,
keep if backtest combined score improves, repeat for n_rounds passes.

Training years: 2015-2021 (7 years)
Validation years: 2022-2023 (held out — never used during optimisation)

Usage:
    python3 -m analysis.weight_optimiser --exam NEET --rounds 40 --k 200
"""

import argparse, copy, json, os
import numpy as np
import analysis.predictor_v4 as v4_mod
from analysis.predictor_v4 import (
    predict_microtopics_v4,
    DEFAULT_SIGNAL_WEIGHTS,
    DEFAULT_FINAL_WEIGHTS,
    WEIGHTS_CACHE_PATH,
    _normalise_weights,
)
from utils.db import get_questions_df

TRAIN_YEARS = list(range(2015, 2022))   # 2015–2021
VAL_YEARS   = [2022, 2023]              # never touched during hill-climbing


def _backtest_score(db_path, exam, test_years, signal_weights, final_weights, k=200):
    """
    Run micro-topic backtest for test_years.
    Returns average combined score: 0.35P + 0.40C + 0.15H + 0.10S
    """
    full_df = get_questions_df(db_path)
    if exam:
        full_df = full_df[full_df["exam"] == exam]

    orig = v4_mod.HOLDOUT_YEARS
    scores = []

    for yr in test_years:
        actual = full_df[full_df["year"] == yr]
        if actual.empty:
            continue

        v4_mod.HOLDOUT_YEARS = set(range(yr, 2030))
        try:
            preds = predict_microtopics_v4(
                db_path, target_year=yr, exam=exam, top_k=k,
                signal_weights=signal_weights, final_weights=final_weights,
            )
        finally:
            v4_mod.HOLDOUT_YEARS = orig

        if not preds:
            continue

        pred_set = set(p["micro_topic"] for p in preds)
        actual_set = set(actual["micro_topic"].unique())
        qs_map = actual.groupby("micro_topic").size().to_dict()
        total = len(actual)
        heavy = {m for m, q in qs_map.items() if q >= 3}

        hits     = pred_set & actual_set
        precision = len(hits) / k
        coverage  = sum(qs_map.get(m, 0) for m in hits) / (total or 1)
        heavy_r   = len(hits & heavy) / (len(heavy) or 1)

        subj_qs = actual.groupby("subject").size().to_dict()
        pred_subj = {}
        for m in hits:
            qs = qs_map.get(m, 0)
            s_series = actual[actual["micro_topic"] == m]["subject"].mode()
            if len(s_series):
                pred_subj[s_series.iloc[0]] = pred_subj.get(s_series.iloc[0], 0) + qs
        avg_sc = (np.mean([pred_subj.get(s, 0) / q for s, q in subj_qs.items()])
                  if subj_qs else 0)

        combined = 0.35*precision + 0.40*coverage + 0.15*heavy_r + 0.10*avg_sc
        scores.append(combined)

    return float(np.mean(scores)) if scores else 0.0


def optimise_weights(db_path="data/exam.db", exam="NEET",
                     n_rounds=40, delta=0.05, k=200, verbose=True,
                     cache_path=None):
    """
    Coordinate-wise hill-climbing over signal_weights + final_weights.

    Each round loops over every weight key and tries ±delta.
    Keeps the perturbation if the TRAIN_YEARS backtest improves.
    Stops early if a full round produces no improvement.

    Args:
        db_path:    path to exam.db
        exam:       "NEET" | "JEE Main"
        n_rounds:   maximum hill-climbing rounds (default 40)
        delta:      step size for each perturbation (default 0.05)
        k:          number of micro-topic predictions per backtest year (default 200)
        verbose:    print progress (default True)
        cache_path: override path to save weights_cache.json

    Returns:
        (signal_weights, final_weights, train_score, val_score)
    """
    sw = copy.deepcopy(DEFAULT_SIGNAL_WEIGHTS)
    fw = copy.deepcopy(DEFAULT_FINAL_WEIGHTS)
    best_score = _backtest_score(db_path, exam, TRAIN_YEARS, sw, fw, k)

    if verbose:
        print(f"Starting train score: {best_score:.4f}")

    for round_num in range(n_rounds):
        improved = False

        # Optimise signal weights
        for key in list(sw.keys()):
            for sign in (+1, -1):
                candidate = copy.deepcopy(sw)
                candidate[key] += sign * delta
                candidate = _normalise_weights(candidate)
                score = _backtest_score(db_path, exam, TRAIN_YEARS, candidate, fw, k)
                if score > best_score + 1e-6:
                    best_score, sw, improved = score, candidate, True
                    if verbose:
                        print(f"  R{round_num+1} signal.{key} {'+' if sign>0 else '-'}{delta} "
                              f"→ {score:.4f}")

        # Optimise final-score weights
        for key in list(fw.keys()):
            for sign in (+1, -1):
                candidate = copy.deepcopy(fw)
                candidate[key] += sign * delta
                candidate = _normalise_weights(candidate)
                score = _backtest_score(db_path, exam, TRAIN_YEARS, sw, candidate, k)
                if score > best_score + 1e-6:
                    best_score, fw, improved = score, candidate, True
                    if verbose:
                        print(f"  R{round_num+1} final.{key} {'+' if sign>0 else '-'}{delta} "
                              f"→ {score:.4f}")

        if not improved:
            if verbose:
                print(f"  R{round_num+1}: converged — stopping early")
            break

    # Validate on held-out years
    val_score = _backtest_score(db_path, exam, VAL_YEARS, sw, fw, k)
    if verbose:
        print(f"\nFinal train score: {best_score:.4f}  |  Val (2022-2023): {val_score:.4f}")

    # Save cache
    out_path = cache_path or WEIGHTS_CACHE_PATH
    cache = {
        "exam": exam,
        "train_score": round(best_score, 4),
        "val_score":   round(val_score,  4),
        "signal_weights": sw,
        "final_weights":  fw,
    }
    with open(out_path, "w") as f:
        json.dump(cache, f, indent=2)
    if verbose:
        print(f"Weights saved → {out_path}")

    return sw, fw, best_score, val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise v4 prediction weights")
    parser.add_argument("--exam",   default="NEET",  help="NEET | JEE Main")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--k",      type=int, default=200)
    args = parser.parse_args()
    print(f"Optimising {args.exam} | rounds={args.rounds} | k={args.k}")
    optimise_weights(exam=args.exam, n_rounds=args.rounds, k=args.k)
```

### Step 4: Run tests

```bash
cd /Users/aman/exam-predictor
export PATH="$PATH:/Users/aman/Library/Python/3.9/bin"
pytest tests/test_weight_optimiser.py -v
```
Expected: **2 passed** (these take ~20-30s each on the tiny synthetic DB)

### Step 5: Run the real optimiser on the live DB (takes ~3-5 min)

```bash
cd /Users/aman/exam-predictor
python3 -m analysis.weight_optimiser --exam NEET --rounds 40 --k 200
```
Expected output ends with:
```
Final train score: 0.XXXX  |  Val (2022-2023): 0.XXXX
Weights saved → analysis/weights_cache.json
```
If val_score < train_score - 0.05, the model is overfitting — reduce `--rounds` to 20 and re-run.

### Step 6: Re-run backtest comparison to see improvement

```bash
python3 -m analysis.backtest_v4 --exam NEET --k 200
```
Expected: AVG v4_comb > 0.650 (up from v3 ~0.530)

### Step 7: Commit

```bash
git add analysis/weight_optimiser.py tests/test_weight_optimiser.py analysis/weights_cache.json
git commit -m "feat: hill-climbing weight optimiser for predictor v4 — saves weights_cache.json"
```

---

## Task 5: Wire v4 into the Intelligence API

**Files:**
- Modify: `intelligence/services/api/routers/data_bridge.py` lines ~338-405 (the `/predict` endpoint)

### Step 1: Open the file and locate the `/predict` endpoint

```bash
grep -n "def real_predict\|predict_microtopics_v3\|predict_chapters_v3" \
    intelligence/services/api/routers/data_bridge.py
```

### Step 2: Update the `/predict` endpoint to route `level=micro` to v4

Find the block inside `real_predict()` that calls `predict_microtopics_v3` and replace:

```python
# BEFORE (in data_bridge.py, inside real_predict()):
        if level == "micro":
            preds = predict_microtopics_v3(DB_PATH, target_year=year, exam=exam, top_k=top_n)
        else:
            preds = predict_chapters_v3(DB_PATH, target_year=year, exam=exam, top_k=top_n)
```

```python
# AFTER:
        if level == "micro":
            try:
                from analysis.predictor_v4 import predict_microtopics_v4
                preds = predict_microtopics_v4(DB_PATH, target_year=year, exam=exam, top_k=top_n)
            except Exception as exc:
                log.warning("predictor_v4 failed, falling back to v3: %s", exc)
                preds = predict_microtopics_v3(DB_PATH, target_year=year, exam=exam, top_k=top_n)
        else:
            preds = predict_chapters_v3(DB_PATH, target_year=year, exam=exam, top_k=top_n)
```

### Step 3: Restart the API and smoke-test

```bash
pm2 restart prajna-intelligence
sleep 3

curl -s "http://localhost:8001/api/v1/data/predict?exam_type=neet&year=2026&top_n=10&level=micro" \
  | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('success:', d.get('success'))
print('total:', d.get('total'))
p = d.get('predictions', [])
if p:
    print('first micro:', p[0].get('micro_topic'), '| parent_score:', p[0].get('signal_breakdown',{}).get('parent_score',{}).get('value','N/A'))
"
```
Expected: `success: True` and a `parent_score` value in the first prediction.

### Step 4: Verify chapter-level still works (no regression)

```bash
curl -s "http://localhost:8001/api/v1/data/predict?exam_type=neet&year=2026&top_n=5&level=chapter" \
  | python3 -c "import sys,json; d=json.load(sys.stdin); print('chapters:', len(d.get('predictions',[])), 'success:', d.get('success'))"
```
Expected: `chapters: 5  success: True`

### Step 5: Commit

```bash
git add intelligence/services/api/routers/data_bridge.py
git commit -m "feat: route level=micro API calls to predictor_v4 with v3 fallback"
```

---

## Task 6: Update Streamlit dashboard to show v4 scores

**Files:**
- Modify: `dashboard/app.py`

### Step 1: Find where micro-topic predictions are displayed in Streamlit

```bash
grep -n "predict_microtopics\|micro_topic\|level.*micro\|Micro" dashboard/app.py | head -20
```

### Step 2: Update the predictions call to use v4 + show improvement badge

Find the section in `app.py` that calls the prediction function and add a version indicator. Locate the relevant predict call (likely in the "Predictions" tab section) and wrap it:

```python
# In dashboard/app.py — find the micro predictions section and replace the call:

# BEFORE:
# preds = predict_microtopics_v3(DB_PATH, target_year=target_year, exam=exam_map[selected_exam], top_k=top_n)

# AFTER:
try:
    from analysis.predictor_v4 import predict_microtopics_v4
    preds = predict_microtopics_v4(
        DB_PATH, target_year=target_year,
        exam=exam_map[selected_exam], top_k=top_n
    )
    st.markdown(
        '<span style="background:#22c55e20;color:#22c55e;border:1px solid #22c55e40;'
        'border-radius:8px;padding:2px 10px;font-size:.75rem;font-weight:700">'
        '⚡ PRAJNA v4 Engine</span>',
        unsafe_allow_html=True
    )
except Exception:
    from analysis.predictor_v3 import predict_microtopics_v3
    preds = predict_microtopics_v3(
        DB_PATH, target_year=target_year,
        exam=exam_map[selected_exam], top_k=top_n
    )
    st.caption("v3 engine (v4 unavailable)")
```

Also add `parent_final_score` as a new column in the Streamlit predictions table where micro-topic results are displayed, so users can see the parent chapter confidence.

### Step 3: Restart Streamlit and verify

```bash
pm2 restart prajna-intelligence 2>/dev/null; true
# Streamlit auto-reloads on file change — just save the file
# Then open http://localhost:8501 → Predictions tab → check "⚡ PRAJNA v4 Engine" badge
```

### Step 4: Commit

```bash
git add dashboard/app.py
git commit -m "feat: Streamlit uses predictor_v4 for micro predictions with v3 fallback badge"
```

---

## Task 7: Final validation — compare v3 vs v4 across all years

### Step 1: Run the full comparison

```bash
cd /Users/aman/exam-predictor
python3 -m analysis.backtest_v4 --exam NEET --k 200
```

### Step 2: Check success criteria

```
Target:
  Micro combined score AVG 2019-2023 ≥ 0.680  (was 0.530)
  Micro coverage@200 AVG              ≥ 0.720  (was 0.649)
  Chapter combined score AVG          ≥ 0.820  (unchanged — uses v3)
```

If combined score < 0.650, re-run the optimiser with more rounds:
```bash
python3 -m analysis.weight_optimiser --exam NEET --rounds 80 --k 200
```
Then re-run backtest_v4.

### Step 3: Commit final tag

```bash
git add .
git commit -m "feat: PRAJNA predictor v4 complete — micro combined score improved from 0.530 → {actual score}"
git tag v4.0-micro-engine
```

---

## Appendix: Troubleshooting

| Symptom | Fix |
|---------|-----|
| `predict_chapters_v3` returns empty list | Check `HOLDOUT_YEARS` — must be overridden during backtest |
| Val score << train score (>0.05 gap) | Overfitting — reduce `--rounds` to 15-20 |
| API returns v3 results despite v4 installed | Check pm2 restart completed: `pm2 logs prajna-intelligence --lines 5` |
| `weights_cache.json` not found | Run `python3 -m analysis.weight_optimiser` first |
| Chapter gate misses known chapters | Increase `chapter_k` from 70 to 80 in `predict_microtopics_v4` call |
