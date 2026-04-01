"""Question Bank API — powered by qbg.db (1.14M questions)"""
from fastapi import APIRouter, Query
from pathlib import Path
import sqlite3

router = APIRouter(prefix="/api/v1/qbank", tags=["Question Bank"])
QBG_DB = str(Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "qbg.db")

def _get_db():
    conn = sqlite3.connect(QBG_DB)
    conn.row_factory = sqlite3.Row
    return conn

def _rows_to_list(rows):
    return [{k: row[k] for k in row.keys()} for row in rows]

@router.get("/search")
async def search_questions(
    query: str = Query(..., min_length=2),
    subject: str = Query(None),
    difficulty: str = Query(None),
    top_n: int = Query(10, ge=1, le=50),
):
    conn = _get_db()
    try:
        sql = "SELECT q.* FROM questions_fts fts JOIN questions q ON q.rowid = fts.rowid WHERE fts.questions_fts MATCH ?"
        params = [query]
        if subject:
            sql += " AND q.subject = ?"
            params.append(subject)
        if difficulty:
            sql += " AND q.difficulty = ?"
            params.append(difficulty)
        sql += f" LIMIT {top_n}"
        rows = conn.execute(sql, params).fetchall()
        return {"success": True, "count": len(rows), "questions": _rows_to_list(rows)}
    except Exception as e:
        return {"success": False, "error": str(e), "questions": []}
    finally:
        conn.close()

@router.get("/random")
async def random_questions(
    subject: str = Query(None),
    difficulty: str = Query(None),
    type: str = Query(None),
    count: int = Query(10, ge=1, le=100),
):
    conn = _get_db()
    try:
        sql = "SELECT * FROM questions WHERE 1=1"
        params = []
        if subject:
            sql += " AND subject = ?"; params.append(subject)
        if difficulty:
            sql += " AND difficulty = ?"; params.append(difficulty)
        if type:
            sql += " AND type = ?"; params.append(type)
        sql += f" ORDER BY RANDOM() LIMIT {count}"
        rows = conn.execute(sql, params).fetchall()
        return {"success": True, "count": len(rows), "questions": _rows_to_list(rows)}
    finally:
        conn.close()

@router.get("/stats")
async def question_stats():
    conn = _get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM questions").fetchone()[0]
        subjects = conn.execute("SELECT subject, COUNT(*) as count FROM questions GROUP BY subject ORDER BY count DESC").fetchall()
        difficulties = conn.execute("SELECT difficulty, COUNT(*) as count FROM questions GROUP BY difficulty ORDER BY count DESC").fetchall()
        types = conn.execute("SELECT type, COUNT(*) as count FROM questions GROUP BY type ORDER BY count DESC LIMIT 10").fetchall()
        return {
            "success": True, "total": total,
            "by_subject": _rows_to_list(subjects),
            "by_difficulty": _rows_to_list(difficulties),
            "by_type": _rows_to_list(types),
        }
    finally:
        conn.close()

@router.post("/mock-test")
async def generate_mock_test(body: dict):
    subject = body.get("subject")
    difficulty = body.get("difficulty")
    count = min(body.get("count", 30), 100)
    q_type = body.get("type", "single correct choice")
    conn = _get_db()
    try:
        sql = "SELECT * FROM questions WHERE type = ?"
        params = [q_type]
        if subject:
            sql += " AND subject = ?"; params.append(subject)
        if difficulty and difficulty != "mixed":
            sql += " AND difficulty = ?"; params.append(difficulty)
        sql += f" ORDER BY RANDOM() LIMIT {count}"
        rows = conn.execute(sql, params).fetchall()
        questions = _rows_to_list(rows)
        # Strip answers for test mode
        for q in questions:
            q.pop("correct_answer", None)
            q.pop("answer_clean", None)
            q.pop("text_solution", None)
        return {"success": True, "count": len(questions), "questions": questions}
    finally:
        conn.close()
