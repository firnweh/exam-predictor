import os
from utils.db import init_db, get_questions_df
from utils.loader import load_all_extracted
from analysis.trend_analyzer import topic_frequency_by_year
from analysis.difficulty_classifier import classify_difficulty
from analysis.pattern_finder import subject_weightage_over_time
from analysis.predictor import predict_topics

TEST_DB = "data/test_e2e.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_full_pipeline():
    init_db(TEST_DB)
    total = load_all_extracted(TEST_DB, "data/extracted")
    assert total >= 6

    df = get_questions_df(TEST_DB)
    assert len(df) >= 6
    assert "JEE Advanced" in df["exam"].values
    assert "NEET" in df["exam"].values

    freq = topic_frequency_by_year(TEST_DB)
    assert not freq.empty

    classified = classify_difficulty(TEST_DB)
    assert "difficulty_label" in classified.columns

    weights = subject_weightage_over_time(TEST_DB)
    assert not weights.empty

    predictions = predict_topics(TEST_DB, target_year=2026)
    assert len(predictions) > 0
    assert all("score" in p for p in predictions)
    assert all("reasons" in p for p in predictions)
