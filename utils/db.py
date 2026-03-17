# utils/db.py
import sqlite3
import json


def init_db(db_path="data/exam.db"):
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS questions (
            id TEXT PRIMARY KEY,
            exam TEXT NOT NULL,
            year INTEGER NOT NULL,
            shift TEXT,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            micro_topic TEXT NOT NULL,
            question_text TEXT,
            question_type TEXT,
            difficulty INTEGER,
            concepts_tested TEXT,
            answer TEXT,
            marks INTEGER
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS topics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject TEXT NOT NULL,
            topic TEXT NOT NULL,
            micro_topic TEXT NOT NULL,
            UNIQUE(subject, topic, micro_topic)
        )
    """)
    conn.commit()
    conn.close()


def insert_questions(db_path, questions):
    conn = sqlite3.connect(db_path)
    for q in questions:
        concepts = json.dumps(q.get("concepts_tested", []))
        conn.execute(
            """INSERT OR REPLACE INTO questions
            (id, exam, year, shift, subject, topic, micro_topic,
             question_text, question_type, difficulty, concepts_tested, answer, marks)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                q["id"], q["exam"], q["year"], q["shift"],
                q["subject"], q["topic"], q["micro_topic"],
                q["question_text"], q["question_type"], q["difficulty"],
                concepts, q["answer"], q["marks"],
            ),
        )
        conn.execute(
            "INSERT OR IGNORE INTO topics (subject, topic, micro_topic) VALUES (?, ?, ?)",
            (q["subject"], q["topic"], q["micro_topic"]),
        )
    conn.commit()
    conn.close()


def get_all_questions(db_path="data/exam.db"):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    rows = conn.execute("SELECT * FROM questions").fetchall()
    conn.close()
    result = []
    for row in rows:
        d = dict(row)
        d["concepts_tested"] = json.loads(d["concepts_tested"])
        result.append(d)
    return result


def get_questions_df(db_path="data/exam.db"):
    import pandas as pd
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM questions", conn)
    conn.close()
    df["concepts_tested"] = df["concepts_tested"].apply(json.loads)
    return df


def get_topics_hierarchy(db_path="data/exam.db"):
    conn = sqlite3.connect(db_path)
    rows = conn.execute(
        "SELECT DISTINCT subject, topic, micro_topic FROM topics ORDER BY subject, topic, micro_topic"
    ).fetchall()
    conn.close()
    hierarchy = {}
    for subject, topic, micro_topic in rows:
        hierarchy.setdefault(subject, {}).setdefault(topic, []).append(micro_topic)
    return hierarchy
