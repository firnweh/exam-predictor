import os, json, pytest
import numpy as np

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
