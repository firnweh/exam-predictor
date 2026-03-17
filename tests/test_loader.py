import os
import json
from utils.loader import load_json_file, load_all_extracted
from utils.db import init_db, get_all_questions

TEST_DB = "data/test_loader.db"
TEST_JSON_DIR = "data/test_extracted"


def setup_function():
    for f in [TEST_DB]:
        if os.path.exists(f):
            os.remove(f)
    os.makedirs(TEST_JSON_DIR, exist_ok=True)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    import shutil
    if os.path.exists(TEST_JSON_DIR):
        shutil.rmtree(TEST_JSON_DIR)


def test_load_json_file():
    init_db(TEST_DB)
    questions = [
        {
            "id": "NEET_2020_Q1",
            "exam": "NEET",
            "year": 2020,
            "shift": "N/A",
            "subject": "Biology",
            "topic": "Genetics",
            "micro_topic": "Mendelian Genetics",
            "question_text": "In a monohybrid cross...",
            "question_type": "MCQ_single",
            "difficulty": 2,
            "concepts_tested": ["dominance", "segregation"],
            "answer": "C",
            "marks": 4,
        }
    ]
    filepath = os.path.join(TEST_JSON_DIR, "neet_2020.json")
    with open(filepath, "w") as f:
        json.dump(questions, f)

    count = load_json_file(TEST_DB, filepath)
    assert count == 1
    result = get_all_questions(TEST_DB)
    assert len(result) == 1
    assert result[0]["exam"] == "NEET"


def test_load_all_extracted():
    init_db(TEST_DB)
    for i, name in enumerate(["jee_2019.json", "neet_2020.json"]):
        q = [{
            "id": f"Q{i}",
            "exam": "JEE Advanced" if i == 0 else "NEET",
            "year": 2019 + i,
            "shift": "P1",
            "subject": "Physics",
            "topic": "Mechanics",
            "micro_topic": "Kinematics",
            "question_text": "...",
            "question_type": "MCQ_single",
            "difficulty": 2,
            "concepts_tested": [],
            "answer": "A",
            "marks": 4,
        }]
        with open(os.path.join(TEST_JSON_DIR, name), "w") as f:
            json.dump(q, f)

    total = load_all_extracted(TEST_DB, TEST_JSON_DIR)
    assert total == 2
    result = get_all_questions(TEST_DB)
    assert len(result) == 2
