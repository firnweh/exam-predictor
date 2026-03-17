import os
from utils.db import init_db, insert_questions
from analysis.predictor import predict_topics

TEST_DB = "data/test_predictor.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _q(id, year, topic, micro_topic, exam="JEE Advanced"):
    return {
        "id": id, "exam": exam, "year": year, "shift": "P1",
        "subject": "Physics", "topic": topic, "micro_topic": micro_topic,
        "question_text": "...", "question_type": "MCQ_single",
        "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4,
    }


def test_predict_topics_returns_ranked_list():
    questions = []
    for i, year in enumerate(range(2015, 2023)):
        questions.append(_q(f"K{i}", year, "Mechanics", "Kinematics"))
    for i, year in enumerate([2018, 2020, 2022]):
        questions.append(_q(f"D{i}", year, "Waves", "Doppler Effect"))
    questions.append(_q("T0", 2016, "Heat", "Carnot Cycle"))
    insert_questions(TEST_DB, questions)

    predictions = predict_topics(TEST_DB, target_year=2024)
    assert len(predictions) > 0
    assert "micro_topic" in predictions[0]
    assert "score" in predictions[0]
    top_5_micros = [p["micro_topic"] for p in predictions[:5]]
    assert "Kinematics" in top_5_micros


def test_predict_topics_includes_reasoning():
    questions = [_q(f"Q{i}", 2020 + i, "Mechanics", "Kinematics") for i in range(3)]
    insert_questions(TEST_DB, questions)
    predictions = predict_topics(TEST_DB, target_year=2024)
    assert "reasons" in predictions[0]
