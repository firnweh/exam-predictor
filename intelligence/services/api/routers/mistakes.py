"""Mistake Analysis API — wraps MistakeAnalyzer + MistakePredictor"""
from fastapi import APIRouter, Query
from functools import lru_cache
from pathlib import Path
import pandas as pd
import sys

# Ensure analysis module is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent.parent))

from analysis.mistake_analyzer import MistakeAnalyzer
from analysis.mistake_predictor import MistakePredictor
from analysis.predictor_v3 import predict_chapters_v3

router = APIRouter(prefix="/api/v1/mistakes", tags=["Mistake Analysis"])

DATA_DIR = Path(__file__).resolve().parent.parent.parent.parent.parent / "data"
STUDENT_DIR = DATA_DIR / "student_data"
EXAM_DB = str(DATA_DIR / "exam.db")

@lru_cache(maxsize=4)
def _load_results(exam_type: str) -> pd.DataFrame:
    fname = f"{exam_type}_results_v2.csv"
    path = STUDENT_DIR / fname
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(str(path))

@lru_cache(maxsize=4)
def _load_students() -> pd.DataFrame:
    path = STUDENT_DIR / "students_v2.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(str(path))

def _build_prajna_importance(exam_type: str) -> dict:
    try:
        exam = "neet" if exam_type == "neet" else "jee_main"
        preds = predict_chapters_v3(EXAM_DB, 2026, exam, top_k=50)
        return {p.get("chapter", ""): p.get("appearance_probability", 0) for p in preds}
    except Exception:
        return {}

@router.get("/danger-zones")
async def danger_zones(
    exam_type: str = Query("neet"),
    error_threshold: float = Query(0.4),
    top_n: int = Query(15),
):
    df = _load_results(exam_type)
    if df.empty:
        return {"success": False, "error": "No results data", "zones": []}
    analyzer = MistakeAnalyzer(df)
    prajna = _build_prajna_importance(exam_type)
    dz = analyzer.danger_zones(prajna, error_threshold=error_threshold, top_n=top_n)
    return {"success": True, "count": len(dz), "zones": dz.to_dict(orient="records")}

@router.get("/cofailure")
async def cofailure(
    exam_type: str = Query("neet"),
    top_n: int = Query(15),
):
    df = _load_results(exam_type)
    if df.empty:
        return {"success": False, "error": "No results data", "pairs": []}
    analyzer = MistakeAnalyzer(df)
    pairs = analyzer.cofailure_pairs(top_n=top_n)
    return {"success": True, "count": len(pairs), "pairs": pairs}

@router.get("/time-accuracy")
async def time_accuracy(exam_type: str = Query("neet")):
    df = _load_results(exam_type)
    if df.empty:
        return {"success": False, "error": "No results data", "data": []}
    analyzer = MistakeAnalyzer(df)
    tva = analyzer.time_vs_accuracy()
    return {"success": True, "count": len(tva), "data": tva.to_dict(orient="records")}

@router.get("/predict")
async def predict_mistakes(
    student_id: str = Query(...),
    exam_type: str = Query("neet"),
):
    df = _load_results(exam_type)
    students_df = _load_students()
    if df.empty:
        return {"success": False, "error": "No results data", "predictions": []}

    prajna = _build_prajna_importance(exam_type)

    # Build abilities df from students
    abilities_cols = [c for c in students_df.columns if c.startswith("ability_")]
    abilities_df = students_df[["student_id"] + abilities_cols] if abilities_cols and not students_df.empty else pd.DataFrame({"student_id": []})

    # Topic difficulty from exam.db
    topic_difficulty = {}  # simplified — would need DB query for full implementation

    mp = MistakePredictor()
    X, y = mp.build_features(df, abilities_df, topic_difficulty, prajna, train_exams=range(1, 9))
    if len(X) > 0:
        mp.train(X, y)
        result = mp.predict_for_student(df, abilities_df, topic_difficulty, prajna, student_id=student_id)
        return {"success": True, "predictions": result}
    return {"success": False, "error": "Insufficient data", "predictions": []}

@router.get("/feature-importance")
async def feature_importance(exam_type: str = Query("neet")):
    df = _load_results(exam_type)
    students_df = _load_students()
    if df.empty:
        return {"success": False, "error": "No data", "importances": {}}

    prajna = _build_prajna_importance(exam_type)
    abilities_cols = [c for c in students_df.columns if c.startswith("ability_")]
    abilities_df = students_df[["student_id"] + abilities_cols] if abilities_cols and not students_df.empty else pd.DataFrame({"student_id": []})

    mp = MistakePredictor()
    X, y = mp.build_features(df, abilities_df, {}, prajna, train_exams=range(1, 9))
    if len(X) > 0:
        mp.train(X, y)
        return {"success": True, "importances": mp.feature_importances()}
    return {"success": False, "error": "Insufficient data", "importances": {}}
