import json
import os
from utils.db import insert_questions


def load_json_file(db_path, json_path):
    with open(json_path, "r") as f:
        questions = json.load(f)
    insert_questions(db_path, questions)
    return len(questions)


def load_all_extracted(db_path, extracted_dir="data/extracted"):
    total = 0
    for filename in sorted(os.listdir(extracted_dir)):
        if filename.endswith(".json"):
            filepath = os.path.join(extracted_dir, filename)
            count = load_json_file(db_path, filepath)
            print(f"Loaded {count} questions from {filename}")
            total += count
    print(f"Total: {total} questions loaded")
    return total
