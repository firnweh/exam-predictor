"""
PRAJNA Student Analyzer
========================
Analyzes student mock-exam performance across 3-level topic hierarchy and
cross-references with the PRAJNA SLM's exam-importance predictions to
surface personalized feedback: strengths, weaknesses, and SLM-powered
priority focus areas.

Usage:
    from analysis.student_analyzer import StudentAnalyzer
    sa = StudentAnalyzer(exam_type='neet')
    sa.load_data()
    profile = sa.get_full_profile('STU001')
    json_summary = sa.build_all_summaries()  # for dashboard JSON
"""

import os
import sys
import json
import math
import csv
import random
from pathlib import Path
from collections import defaultdict

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "student_data"
MODELS_DIR = BASE_DIR / "models"

# ── SLM chapter importance (precomputed priorities from backtest / SLM output) ─
# These reflect the PRAJNA SLM's prediction for 2026 exams.
# Scale 0–1: higher = more likely to appear in the real exam.
NEET_CHAPTER_IMPORTANCE = {
    # Physics
    "Physical World And Measurement": 0.45,
    "Kinematics": 0.78,
    "Laws Of Motion": 0.82,
    "Work Energy And Power": 0.75,
    "Rotational Motion": 0.72,
    "Gravitation": 0.65,
    "Properties Of Solids And Liquids": 0.55,
    "Thermodynamics": 0.70,
    "Kinetic Theory Of Gases": 0.60,
    "Thermodynamics Laws": 0.68,
    "Oscillations And Waves": 0.74,
    "Electrostatics": 0.88,
    "Current Electricity": 0.82,
    "Magnetic Effects Of Current": 0.79,
    "Electromagnetic Induction": 0.74,
    "Alternating Current": 0.68,
    "Electromagnetic Waves": 0.52,
    "Optics": 0.85,
    "Dual Nature Of Matter": 0.70,
    "Atoms And Nuclei": 0.76,
    "Electronic Devices": 0.62,
    # Chemistry
    "Some Basic Concepts Of Chemistry": 0.72,
    "Structure Of Atom": 0.78,
    "Classification Of Elements": 0.68,
    "Chemical Bonding": 0.85,
    "States Of Matter": 0.62,
    "Chemical Thermodynamics": 0.74,
    "Equilibrium": 0.82,
    "Redox Reactions": 0.65,
    "Hydrogen": 0.48,
    "S Block Elements": 0.55,
    "P Block Elements": 0.72,
    "Organic Chemistry Basic Principles": 0.88,
    "Hydrocarbons": 0.79,
    "Environmental Chemistry": 0.45,
    "Solid State": 0.70,
    "Solutions": 0.72,
    "Electrochemistry": 0.78,
    "Chemical Kinetics": 0.76,
    "Surface Chemistry": 0.58,
    "D And F Block Elements": 0.68,
    "Coordination Compounds": 0.75,
    "Haloalkanes And Haloarenes": 0.72,
    "Alcohols Phenols Ethers": 0.70,
    "Aldehydes Ketones Carboxylic Acids": 0.78,
    "Amines": 0.65,
    "Biomolecules": 0.70,
    "Polymers": 0.48,
    "Chemistry In Everyday Life": 0.42,
    # Biology
    "Diversity In Living World": 0.72,
    "Structural Organisation In Plants And Animals": 0.68,
    "Cell Structure And Function": 0.82,
    "Plant Physiology": 0.78,
    "Human Physiology": 0.90,
    "Reproduction": 0.85,
    "Genetics And Evolution": 0.92,
    "Biology And Human Welfare": 0.65,
    "Biotechnology": 0.78,
    "Ecology And Environment": 0.72,
}

JEE_SUBCHAPTER_IMPORTANCE = {
    # Physics - Mechanics
    "Kinematics": 0.82,
    "Newton's Laws": 0.85,
    "Work Power Energy": 0.78,
    "Centre Of Mass": 0.72,
    "Rotational Mechanics": 0.80,
    "Gravitation": 0.68,
    "Simple Harmonic Motion": 0.75,
    "Fluid Mechanics": 0.65,
    # Physics - Waves & Thermo
    "Wave Motion": 0.70,
    "Sound Waves": 0.68,
    "Heat And Temperature": 0.65,
    "KTG And Thermodynamics": 0.78,
    # Physics - EM
    "Electrostatics": 0.88,
    "Capacitance": 0.80,
    "Current Electricity": 0.85,
    "Magnetic Effect": 0.82,
    "Electromagnetic Induction": 0.78,
    "Alternating Current": 0.72,
    # Physics - Optics
    "Geometrical Optics": 0.82,
    "Wave Optics": 0.75,
    # Physics - Modern
    "Photoelectric Effect": 0.72,
    "Atomic Models": 0.68,
    "Nuclear Physics": 0.70,
    "Semiconductors": 0.62,
    # Chemistry - Physical
    "Mole Concept": 0.85,
    "Atomic Structure": 0.78,
    "Chemical Bonding": 0.82,
    "Gaseous State": 0.70,
    "Chemical Thermodynamics": 0.75,
    "Chemical Equilibrium": 0.80,
    "Ionic Equilibrium": 0.82,
    "Chemical Kinetics": 0.78,
    "Electrochemistry": 0.75,
    "Solutions": 0.70,
    "Surface Chemistry": 0.60,
    "Solid State": 0.68,
    # Chemistry - Inorganic
    "Periodic Table": 0.72,
    "s-Block Elements": 0.62,
    "p-Block Elements": 0.75,
    "d-Block Elements": 0.70,
    "Coordination Chemistry": 0.78,
    "Metallurgy": 0.58,
    "Qualitative Analysis": 0.55,
    "Hydrogen": 0.52,
    # Chemistry - Organic
    "General Organic Chemistry": 0.88,
    "Hydrocarbons": 0.82,
    "Halides": 0.78,
    "Oxygen Compounds": 0.80,
    "Nitrogen Compounds": 0.72,
    "Biomolecules": 0.65,
    "Polymers": 0.55,
    "Practical Organic": 0.70,
    # Mathematics - Algebra
    "Quadratic Equations": 0.80,
    "Complex Numbers": 0.75,
    "Sequences And Series": 0.72,
    "Permutation Combination": 0.78,
    "Binomial Theorem": 0.72,
    "Matrices Determinants": 0.82,
    "Mathematical Induction": 0.55,
    # Math - Trig
    "Trigonometric Functions": 0.78,
    "Trigonometric Equations": 0.72,
    "Inverse Trigonometry": 0.70,
    "Properties Of Triangles": 0.65,
    # Math - Coord Geo
    "Straight Lines": 0.82,
    "Circles": 0.85,
    "Parabola": 0.80,
    "Ellipse": 0.75,
    "Hyperbola": 0.72,
    # Math - Calculus
    "Limits": 0.85,
    "Continuity And Differentiability": 0.78,
    "Differentiation": 0.82,
    "Application Of Derivatives": 0.88,
    "Indefinite Integrals": 0.82,
    "Definite Integrals": 0.85,
    "Area Under Curves": 0.80,
    "Differential Equations": 0.78,
    # Math - Vectors & 3D
    "Vectors": 0.78,
    "3D Geometry": 0.80,
    # Math - Probability
    "Probability": 0.82,
    "Distributions": 0.72,
    "Statistics": 0.65,
}

# Performance classification thresholds
THRESHOLDS = {
    "Mastered":    (80, 101),
    "Strong":      (65, 80),
    "Developing":  (45, 65),
    "Weak":        (25, 45),
    "Critical":    (0,  25),
}

LEVEL_COLOR = {
    "Mastered":   "#22c55e",
    "Strong":     "#84cc16",
    "Developing": "#f59e0b",
    "Weak":       "#f97316",
    "Critical":   "#ef4444",
}

def classify_level(accuracy_pct):
    for label, (lo, hi) in THRESHOLDS.items():
        if lo <= accuracy_pct < hi:
            return label
    return "Critical"

def trend_slope(values):
    """Linear trend slope over a list of values. Positive = improving."""
    n = len(values)
    if n < 2:
        return 0.0
    x_mean = (n - 1) / 2
    y_mean = sum(values) / n
    num = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    den = sum((i - x_mean) ** 2 for i in range(n))
    return num / den if den > 0 else 0.0

def consistency_score(values):
    """0–100 where 100 = perfectly consistent. Inverse of coefficient of variation."""
    if not values or all(v == 0 for v in values):
        return 0.0
    mean = sum(values) / len(values)
    if mean == 0:
        return 0.0
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    std = variance ** 0.5
    cv = std / (mean + 1e-6)
    return round(max(0, min(100, 100 * (1 - cv))), 1)

# ── Core Analyzer ─────────────────────────────────────────────────────────────

class StudentAnalyzer:
    def __init__(self, exam_type="neet"):
        assert exam_type in ("neet", "jee"), "exam_type must be 'neet' or 'jee'"
        self.exam_type = exam_type
        self.results = []       # list of dicts from CSV
        self.summary = []       # list of dicts from summary CSV
        self.students = []      # list of student profiles
        self._student_map = {}  # student_id → profile dict
        self._results_by_student = defaultdict(list)
        self._summary_by_student = defaultdict(list)
        self._importance = (
            NEET_CHAPTER_IMPORTANCE if exam_type == "neet" else JEE_SUBCHAPTER_IMPORTANCE
        )
        self._chapter_key = "chapter" if exam_type == "neet" else "sub_chapter"

    # ── Data loading ───────────────────────────────────────────────────────────
    def load_data(self):
        prefix = "neet" if self.exam_type == "neet" else "jee"
        res_path = DATA_DIR / f"{prefix}_results_v2.csv"
        sum_path = DATA_DIR / f"{prefix}_summary_v2.csv"
        stu_path = DATA_DIR / "students_v2.csv"

        self.students = list(self._read_csv(stu_path))
        self._student_map = {s["student_id"]: s for s in self.students}

        self.results = list(self._read_csv(res_path))
        for row in self.results:
            self._results_by_student[row["student_id"]].append(row)

        self.summary = list(self._read_csv(sum_path))
        for row in self.summary:
            self._summary_by_student[row["student_id"]].append(row)

        print(f"Loaded {len(self.results):,} result rows for {len(self.students)} students ({self.exam_type.upper()})")

    def _read_csv(self, path):
        with open(path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Cast numeric fields
                for k in ("exam_no","total_qs","attempted","correct","wrong","not_attempted"):
                    if k in row:
                        row[k] = int(row[k])
                for k in ("score","max_score","accuracy_pct","time_min","total_score","percentage","rank","percentile"):
                    if k in row:
                        try:
                            row[k] = float(row[k])
                        except (ValueError, TypeError):
                            pass
                yield row

    # ── Chapter performance per student ───────────────────────────────────────
    def get_chapter_performance(self, student_id):
        """
        Returns dict: chapter_name → {
          avg_accuracy, trend, consistency, level, importance,
          priority_score, exam_accuracies: [e1..e10]
        }
        """
        rows = self._results_by_student.get(student_id, [])
        ck = self._chapter_key
        # Group by chapter, then by exam_no
        chapter_exam_acc = defaultdict(dict)
        for row in rows:
            chap = row[ck]
            en = int(row["exam_no"])
            acc = float(row["accuracy_pct"])
            chapter_exam_acc[chap][en] = acc

        result = {}
        for chap, exam_dict in chapter_exam_acc.items():
            exam_accs = [exam_dict.get(e, 0.0) for e in range(1, 11)]
            avg_acc = sum(exam_accs) / len(exam_accs)
            slope = trend_slope(exam_accs)
            cons = consistency_score(exam_accs)
            level = classify_level(avg_acc)
            importance = self._importance.get(chap, 0.60)
            # Priority = how much attention this topic needs
            # High priority = weak topic that is also important for the exam
            weakness_score = max(0, 1 - avg_acc / 100)
            priority_score = round(weakness_score * importance, 4)

            result[chap] = {
                "avg_accuracy": round(avg_acc, 1),
                "trend": round(slope, 3),          # per-exam improvement
                "consistency": cons,
                "level": level,
                "importance": importance,
                "priority_score": priority_score,
                "exam_accuracies": [round(a, 1) for a in exam_accs],
            }
        return result

    # ── Subject-level summary ─────────────────────────────────────────────────
    def get_subject_performance(self, student_id):
        """Returns dict: subject → {avg_accuracy, trend, level, total_score_pct}"""
        rows = self._results_by_student.get(student_id, [])
        subj_acc = defaultdict(list)
        subj_score = defaultdict(lambda: [0.0, 0.0])  # [earned, max]
        for row in rows:
            subj = row["subject"]
            subj_acc[subj].append(float(row["accuracy_pct"]))
            subj_score[subj][0] += float(row["score"])
            subj_score[subj][1] += float(row["max_score"])

        result = {}
        for subj, accs in subj_acc.items():
            avg = sum(accs) / len(accs)
            # Group by exam and compute per-exam averages for trend
            exam_subj = defaultdict(list)
            for row in rows:
                if row["subject"] == subj:
                    exam_subj[int(row["exam_no"])].append(float(row["accuracy_pct"]))
            exam_avgs = [sum(exam_subj.get(e, [0])) / max(len(exam_subj.get(e, [1])), 1)
                         for e in range(1, 11)]
            slope = trend_slope(exam_avgs)
            earned, mx = subj_score[subj]
            result[subj] = {
                "avg_accuracy": round(avg, 1),
                "trend": round(slope, 3),
                "level": classify_level(avg),
                "score_pct": round(100 * earned / mx, 1) if mx > 0 else 0.0,
                "exam_averages": [round(a, 1) for a in exam_avgs],
            }
        return result

    # ── Exam-level trajectory ─────────────────────────────────────────────────
    def get_exam_trajectory(self, student_id):
        """Returns list of 10 dicts: exam_no, total_score, percentage, rank, percentile"""
        rows = self._summary_by_student.get(student_id, [])
        rows_sorted = sorted(rows, key=lambda r: int(r["exam_no"]))
        return [
            {
                "exam_no": int(r["exam_no"]),
                "exam_label": r["exam_label"],
                "total_score": float(r["total_score"]),
                "percentage": float(r["percentage"]),
                "rank": int(r["rank"]),
                "percentile": float(r["percentile"]),
            }
            for r in rows_sorted
        ]

    # ── Weakness / Strength identification ────────────────────────────────────
    def identify_weak_zones(self, student_id, top_n=10):
        """Returns top N chapters sorted by priority_score (weakness × importance)."""
        chap_perf = self.get_chapter_performance(student_id)
        sorted_chaps = sorted(
            chap_perf.items(),
            key=lambda x: x[1]["priority_score"],
            reverse=True,
        )
        return [
            {"chapter": ch, **data}
            for ch, data in sorted_chaps[:top_n]
            if data["level"] in ("Weak", "Critical", "Developing")
        ]

    def identify_strengths(self, student_id, top_n=8):
        """Returns top N chapters by accuracy where student is Strong/Mastered."""
        chap_perf = self.get_chapter_performance(student_id)
        sorted_chaps = sorted(
            chap_perf.items(),
            key=lambda x: x[1]["avg_accuracy"],
            reverse=True,
        )
        return [
            {"chapter": ch, **data}
            for ch, data in sorted_chaps[:top_n]
            if data["level"] in ("Mastered", "Strong")
        ]

    def get_slm_priority_focus(self, student_id, top_n=8):
        """
        SLM-powered focus list: chapters where the student is weak AND
        the SLM predicts high appearance probability in the real exam.
        These are the highest-ROI study targets.
        """
        chap_perf = self.get_chapter_performance(student_id)
        candidates = []
        for chap, data in chap_perf.items():
            if data["level"] in ("Weak", "Critical", "Developing"):
                importance = data["importance"]
                weakness = max(0, 1 - data["avg_accuracy"] / 100)
                slm_priority = round(weakness * importance * 100, 1)
                candidates.append({
                    "chapter": chap,
                    "accuracy": data["avg_accuracy"],
                    "level": data["level"],
                    "slm_importance": round(importance * 100, 0),
                    "slm_priority_score": slm_priority,
                    "trend": data["trend"],
                    "consistency": data["consistency"],
                })
        candidates.sort(key=lambda x: x["slm_priority_score"], reverse=True)
        return candidates[:top_n]

    def get_micro_topic_breakdown(self, student_id, chapter_name, top_n=5):
        """Returns per-micro-topic accuracy for a given chapter."""
        ck = self._chapter_key
        rows = [r for r in self._results_by_student.get(student_id, [])
                if r[ck] == chapter_name]
        mt_acc = defaultdict(list)
        for row in rows:
            mt_acc[row["micro_topic"]].append(float(row["accuracy_pct"]))
        result = []
        for mt, accs in mt_acc.items():
            avg = sum(accs) / len(accs)
            result.append({"micro_topic": mt, "avg_accuracy": round(avg, 1), "level": classify_level(avg)})
        result.sort(key=lambda x: x["avg_accuracy"])
        return result[:top_n]  # weakest first

    # ── Consistency and improvement metrics ───────────────────────────────────
    def get_overall_metrics(self, student_id):
        """Key headline metrics: overall accuracy, improvement rate, consistency."""
        traj = self.get_exam_trajectory(student_id)
        if not traj:
            return {}
        scores = [t["percentage"] for t in traj]
        best_rank = min(t["rank"] for t in traj)
        latest_rank = traj[-1]["rank"]
        best_pct = max(scores)
        latest_pct = scores[-1]
        avg_pct = sum(scores) / len(scores)
        improvement = scores[-1] - scores[0]  # from exam 1 to exam 10
        cons = consistency_score(scores)
        slope = trend_slope(scores)
        return {
            "avg_percentage": round(avg_pct, 1),
            "best_percentage": round(best_pct, 1),
            "latest_percentage": round(latest_pct, 1),
            "improvement": round(improvement, 1),
            "best_rank": int(best_rank),
            "latest_rank": int(latest_rank),
            "consistency_score": cons,
            "trend_per_exam": round(slope, 3),
            "trajectory": scores,
        }

    # ── Full profile for dashboard ────────────────────────────────────────────
    def get_full_profile(self, student_id):
        profile = self._student_map.get(student_id, {})
        return {
            "student": profile,
            "metrics": self.get_overall_metrics(student_id),
            "subjects": self.get_subject_performance(student_id),
            "trajectory": self.get_exam_trajectory(student_id),
            "weak_zones": self.identify_weak_zones(student_id),
            "strengths": self.identify_strengths(student_id),
            "slm_focus": self.get_slm_priority_focus(student_id),
            "chapter_performance": self.get_chapter_performance(student_id),
        }

    # ── Build compact JSON for all students (used by HTML dashboard) ──────────
    def build_all_summaries(self):
        """
        Builds a compact JSON-serializable dict for all students.
        Suitable for embedding in the HTML dashboard.
        """
        out = {"exam_type": self.exam_type, "students": []}
        for stu in self.students:
            sid = stu["student_id"]
            metrics = self.get_overall_metrics(sid)
            subjects = self.get_subject_performance(sid)
            chap_perf = self.get_chapter_performance(sid)
            slm_focus = self.get_slm_priority_focus(sid, top_n=5)
            strengths = self.identify_strengths(sid, top_n=5)
            trajectory = self.get_exam_trajectory(sid)

            # Compact chapter data: {chapter: [avg_acc, level, importance]}
            chap_compact = {
                ch: [d["avg_accuracy"], d["level"][0], round(d["importance"] * 100)]
                for ch, d in chap_perf.items()
            }

            out["students"].append({
                "id": sid,
                "name": stu["name"],
                "city": stu.get("city", ""),
                "coaching": stu.get("coaching", ""),
                "target": stu.get("target", ""),
                "abilities": {
                    "phy": float(stu.get("ability_physics", 0.5)),
                    "chem": float(stu.get("ability_chemistry", 0.5)),
                    "bio": float(stu.get("ability_biology", 0.5)),
                    "maths": float(stu.get("ability_mathematics", 0.5)),
                },
                "metrics": metrics,
                "subjects": {
                    k: {"acc": v["avg_accuracy"], "level": v["level"], "trend": v["trend"],
                        "exams": v["exam_averages"]}
                    for k, v in subjects.items()
                },
                "trajectory": [t["percentage"] for t in trajectory],
                "ranks": [t["rank"] for t in trajectory],
                "chapters": chap_compact,
                "slm_focus": slm_focus,
                "strengths": [{"chapter": s["chapter"], "acc": s["avg_accuracy"]} for s in strengths],
            })
        return out


# ── CLI: build dashboard JSON ─────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PRAJNA Student Analyzer")
    parser.add_argument("--exam", choices=["neet", "jee"], default="neet")
    parser.add_argument("--output", default=None, help="Output JSON path")
    args = parser.parse_args()

    sa = StudentAnalyzer(exam_type=args.exam)
    sa.load_data()

    out_path = args.output or str(BASE_DIR / "docs" / f"student_summary_{args.exam}.json")
    print(f"Building summaries for {len(sa.students)} students …")
    summaries = sa.build_all_summaries()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summaries, f, separators=(",", ":"))
    size_kb = os.path.getsize(out_path) / 1024
    print(f"✓ Written {size_kb:.0f} KB → {out_path}")
