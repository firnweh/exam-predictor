import os
from utils.db import init_db, insert_questions
from analysis.difficulty_classifier import classify_difficulty, get_difficulty_distribution

TEST_DB = "data/test_difficulty.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_classify_difficulty():
    questions = []
    for i in range(20):
        questions.append({
            "id": f"Q{i}", "exam": "JEE Advanced", "year": 2020,
            "shift": "P1", "subject": "Physics", "topic": "Mechanics",
            "micro_topic": "Kinematics", "question_text": "...",
            "question_type": "MCQ_single" if i < 10 else "integer",
            "difficulty": (i % 5) + 1, "concepts_tested": [],
            "answer": "A", "marks": 3 if i < 10 else 4,
        })
    insert_questions(TEST_DB, questions)
    result = classify_difficulty(TEST_DB)
    assert "difficulty_label" in result.columns
    assert set(result["difficulty_label"].unique()).issubset({"Easy", "Moderate", "Hard", "Very Hard"})


def test_difficulty_distribution():
    questions = [
        {"id": f"Q{i}", "exam": "JEE Advanced", "year": 2020,
         "shift": "P1", "subject": "Physics", "topic": "Mechanics",
         "micro_topic": "Kinematics", "question_text": "...",
         "question_type": "MCQ_single", "difficulty": (i % 5) + 1,
         "concepts_tested": [], "answer": "A", "marks": 4}
        for i in range(20)
    ]
    insert_questions(TEST_DB, questions)
    dist = get_difficulty_distribution(TEST_DB)
    assert "Physics" in dist.index.get_level_values("subject")
