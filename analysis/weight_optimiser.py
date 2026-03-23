"""
Hill-climbing weight optimiser for PRAJNA Predictor v4.

Algorithm: Coordinate-wise ascent — perturb one weight at a time by ±delta,
keep if backtest combined score improves, repeat for n_rounds passes.

Training years: 2015-2021
Validation years: 2022-2023 (held out)

Usage:
    python3 -m analysis.weight_optimiser --exam NEET --rounds 40 --k 200
"""

import argparse, copy, json, os
import numpy as np
import analysis.predictor_v3 as v3_mod
import analysis.predictor_v4 as v4_mod
from analysis.predictor_v4 import (
    predict_microtopics_v4,
    DEFAULT_SIGNAL_WEIGHTS,
    DEFAULT_FINAL_WEIGHTS,
    WEIGHTS_CACHE_PATH,
    _normalise_weights,
)
from utils.db import get_questions_df

TRAIN_YEARS = list(range(2015, 2022))   # 2015-2021
VAL_YEARS   = [2022, 2023]              # never touched during hill-climbing


def _backtest_score(db_path, exam, test_years, signal_weights, final_weights, k=200):
    """
    Run micro-topic backtest for test_years.
    Returns average combined score: 0.35P + 0.40C + 0.15H + 0.10S
    """
    full_df = get_questions_df(db_path)
    if exam:
        full_df = full_df[full_df["exam"] == exam]

    orig_v3 = v3_mod.HOLDOUT_YEARS
    orig_v4 = v4_mod.HOLDOUT_YEARS
    scores = []

    for yr in test_years:
        actual = full_df[full_df["year"] == yr]
        if actual.empty:
            continue

        holdout = set(range(yr, 2030))
        v3_mod.HOLDOUT_YEARS = holdout
        v4_mod.HOLDOUT_YEARS = holdout
        try:
            preds = predict_microtopics_v4(
                db_path, target_year=yr, exam=exam, top_k=k,
                signal_weights=signal_weights, final_weights=final_weights,
            )
        finally:
            v3_mod.HOLDOUT_YEARS = orig_v3
            v4_mod.HOLDOUT_YEARS = orig_v4

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
    Returns (signal_weights, final_weights, train_score, val_score)
    """
    sw = copy.deepcopy(DEFAULT_SIGNAL_WEIGHTS)
    fw = copy.deepcopy(DEFAULT_FINAL_WEIGHTS)
    best_score = _backtest_score(db_path, exam, TRAIN_YEARS, sw, fw, k)

    if verbose:
        print(f"Starting train score: {best_score:.4f}")

    for round_num in range(n_rounds):
        improved = False

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
                              f"-> {score:.4f}")

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
                              f"-> {score:.4f}")

        if not improved:
            if verbose:
                print(f"  R{round_num+1}: converged — stopping early")
            break

    val_score = _backtest_score(db_path, exam, VAL_YEARS, sw, fw, k)
    if verbose:
        print(f"\nFinal train score: {best_score:.4f}  |  Val (2022-2023): {val_score:.4f}")

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
        print(f"Weights saved -> {out_path}")

    return sw, fw, best_score, val_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optimise v4 prediction weights")
    parser.add_argument("--exam",   default="NEET",  help="NEET | JEE Main")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--k",      type=int, default=200)
    args = parser.parse_args()
    print(f"Optimising {args.exam} | rounds={args.rounds} | k={args.k}")
    optimise_weights(exam=args.exam, n_rounds=args.rounds, k=args.k)
