# tests/test_db.py
import os
import sqlite3
import json
from utils.db import init_db, insert_questions, get_all_questions, get_topics_hierarchy

TEST_DB = "data/test_exam.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_init_db_creates_tables():
    init_db(TEST_DB)
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert "questions" in tables
    assert "topics" in tables


def test_insert_and_retrieve_questions():
    init_db(TEST_DB)
    questions = [
        {
            "id": "JEE_ADV_2019_P1_Q1",
            "exam": "JEE Advanced",
            "year": 2019,
            "shift": "Paper 1",
            "subject": "Physics",
            "topic": "Mechanics",
            "micro_topic": "Rotational Dynamics",
            "question_text": "A uniform rod of length L...",
            "question_type": "MCQ_single",
            "difficulty": 3,
            "concepts_tested": ["moment of inertia", "angular momentum"],
            "answer": "B",
            "marks": 3,
        }
    ]
    insert_questions(TEST_DB, questions)
    result = get_all_questions(TEST_DB)
    assert len(result) == 1
    assert result[0]["id"] == "JEE_ADV_2019_P1_Q1"
    assert result[0]["subject"] == "Physics"
    assert result[0]["concepts_tested"] == ["moment of inertia", "angular momentum"]


def test_topics_hierarchy():
    init_db(TEST_DB)
    questions = [
        {
            "id": "Q1", "exam": "JEE Advanced", "year": 2019, "shift": "P1",
            "subject": "Physics", "topic": "Mechanics", "micro_topic": "Rotational Dynamics",
            "question_text": "...", "question_type": "MCQ_single", "difficulty": 3,
            "concepts_tested": [], "answer": "B", "marks": 3,
        },
        {
            "id": "Q2", "exam": "NEET", "year": 2020, "shift": "N/A",
            "subject": "Physics", "topic": "Mechanics", "micro_topic": "Newton's Laws",
            "question_text": "...", "question_type": "MCQ_single", "difficulty": 2,
            "concepts_tested": [], "answer": "A", "marks": 4,
        },
    ]
    insert_questions(TEST_DB, questions)
    hierarchy = get_topics_hierarchy(TEST_DB)
    assert "Physics" in hierarchy
    assert "Mechanics" in hierarchy["Physics"]
    assert "Rotational Dynamics" in hierarchy["Physics"]["Mechanics"]
    assert "Newton's Laws" in hierarchy["Physics"]["Mechanics"]
