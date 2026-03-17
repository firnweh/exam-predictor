import os
from utils.db import init_db, insert_questions
from analysis.trend_analyzer import (
    topic_frequency_by_year,
    find_hot_cold_topics,
    detect_cycles,
)

TEST_DB = "data/test_trends.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _make_question(id, year, topic, micro_topic, exam="JEE Advanced", subject="Physics"):
    return {
        "id": id, "exam": exam, "year": year, "shift": "P1",
        "subject": subject, "topic": topic, "micro_topic": micro_topic,
        "question_text": "...", "question_type": "MCQ_single",
        "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4,
    }


def test_topic_frequency_by_year():
    questions = [
        _make_question("Q1", 2020, "Mechanics", "Kinematics"),
        _make_question("Q2", 2020, "Mechanics", "Kinematics"),
        _make_question("Q3", 2021, "Mechanics", "Kinematics"),
        _make_question("Q4", 2020, "Optics", "Refraction"),
    ]
    insert_questions(TEST_DB, questions)
    freq = topic_frequency_by_year(TEST_DB)
    assert freq.loc[("Mechanics", "Kinematics"), 2020] == 2
    assert freq.loc[("Mechanics", "Kinematics"), 2021] == 1
    assert freq.loc[("Optics", "Refraction"), 2020] == 1


def test_hot_cold_topics():
    questions = []
    for i, year in enumerate(range(2018, 2023)):
        questions.append(_make_question(f"K{i}", year, "Mechanics", "Kinematics"))
    questions.append(_make_question("T0", 2018, "Heat", "Thermodynamics"))
    insert_questions(TEST_DB, questions)

    hot, cold = find_hot_cold_topics(TEST_DB, recent_years=3, current_year=2022)
    hot_micros = [t[1] for t in hot]
    cold_micros = [t[1] for t in cold]
    assert "Kinematics" in hot_micros
    assert "Thermodynamics" in cold_micros


def test_detect_cycles():
    questions = []
    for i, year in enumerate([2010, 2013, 2016, 2019, 2022]):
        questions.append(_make_question(f"C{i}", year, "Waves", "Doppler Effect"))
    for i, year in enumerate(range(2010, 2023)):
        questions.append(_make_question(f"F{i}", year, "Mechanics", "Friction"))
    insert_questions(TEST_DB, questions)

    cycles = detect_cycles(TEST_DB, min_occurrences=4)
    doppler_entry = [c for c in cycles if c["micro_topic"] == "Doppler Effect"]
    assert len(doppler_entry) == 1
    assert doppler_entry[0]["estimated_cycle_years"] == 3
