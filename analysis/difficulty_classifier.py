import pandas as pd
from utils.db import get_questions_df


QUESTION_TYPE_MAP = {
    "MCQ_single": 1, "MCQ_multi": 2, "integer": 3,
    "numerical": 3, "matrix_match": 4, "assertion_reason": 2, "subjective": 5,
}

DIFFICULTY_LABELS = {1: "Easy", 2: "Easy", 3: "Moderate", 4: "Hard", 5: "Very Hard"}


def classify_difficulty(db_path="data/exam.db"):
    df = get_questions_df(db_path)
    df["type_score"] = df["question_type"].map(QUESTION_TYPE_MAP).fillna(2)
    df["difficulty_label"] = df["difficulty"].map(DIFFICULTY_LABELS)
    return df


def get_difficulty_distribution(db_path="data/exam.db", exam=None):
    df = classify_difficulty(db_path)
    if exam:
        df = df[df["exam"] == exam]
    dist = df.groupby(["subject", "topic", "difficulty_label"]).size().unstack(fill_value=0)
    return dist


def difficulty_over_time(db_path="data/exam.db", exam=None):
    df = classify_difficulty(db_path)
    if exam:
        df = df[df["exam"] == exam]
    return df.groupby(["year", "subject"])["difficulty"].mean().unstack(fill_value=0)
