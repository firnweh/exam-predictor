import os, json, pytest


def test_optimiser_improves_over_baseline(tmp_path):
    """On a tiny synthetic DB, optimiser must produce a score >= starting score."""
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
    assert train_score >= base_score - 0.01


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
