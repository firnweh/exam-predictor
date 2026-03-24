"""
PRAJNA Mistake Predictor — Student View Logistic Regression
============================================================
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATURE_NAMES = [
    "rolling_accuracy", "ability_score", "topic_difficulty",
    "exam_importance", "avg_time_spent", "streak", "exam_number",
]

class MistakePredictor:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()

    def build_features(self, results_df, abilities_df, topic_difficulty,
                       prajna_importance, train_exams=range(1, 9)):
        df = results_df[results_df["exam_no"].isin(train_exams)].copy()
        if df.empty:
            return np.empty((0, 7)), np.empty(0)

        ability_map = {}
        for _, row in abilities_df.iterrows():
            sid = row["student_id"]
            for col in abilities_df.columns:
                if col.startswith("ability_"):
                    subj = col.replace("ability_", "").capitalize()
                    ability_map[(sid, subj)] = row[col]

        rows = []
        labels = []

        for (sid, topic), grp in df.groupby(["student_id", "micro_topic"]):
            grp = grp.sort_values("exam_no")
            subj = grp["subject"].iloc[0]
            ability = ability_map.get((sid, subj), 0.5)
            diff = topic_difficulty.get(topic, 3.0) / 5.0
            importance = prajna_importance.get(topic, 0.5)

            accs = grp["accuracy_pct"].values
            times = grp["time_min"].values
            exams = grp["exam_no"].values

            streak = 0
            for a in accs:
                if a >= 50:
                    streak = streak + 1 if streak > 0 else 1
                else:
                    streak = streak - 1 if streak < 0 else -1

            for i in range(1, len(accs)):
                rolling_acc = np.mean(accs[:i]) / 100.0
                avg_time = np.mean(times[:i])
                max_time = df["time_min"].max() or 1
                norm_time = avg_time / max_time

                feat = [
                    rolling_acc, ability, diff, importance, norm_time,
                    np.clip(streak / 5.0, -1, 1), exams[i] / 10.0,
                ]
                rows.append(feat)
                labels.append(1 if accs[i] < 50 else 0)

        if not rows:
            return np.empty((0, 7)), np.empty(0)
        return np.array(rows, dtype=np.float64), np.array(labels, dtype=np.int32)

    def train(self, X, y):
        if len(X) == 0:
            return
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        if self.model is None or len(X) == 0:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importances(self):
        if self.model is None:
            return {}
        coefs = np.abs(self.model.coef_[0])
        total = coefs.sum() or 1
        return {name: round(float(c / total) * 100, 1)
                for name, c in zip(FEATURE_NAMES, coefs)}

    def predict_for_student(self, results_df, abilities_df, topic_difficulty,
                            prajna_importance, student_id):
        sdf = results_df[results_df["student_id"] == student_id]
        if sdf.empty or self.model is None:
            return []

        ability_map = {}
        for _, row in abilities_df.iterrows():
            if row["student_id"] == student_id:
                for col in abilities_df.columns:
                    if col.startswith("ability_"):
                        subj = col.replace("ability_", "").capitalize()
                        ability_map[subj] = row[col]

        results = []
        for topic, grp in sdf.groupby("micro_topic"):
            grp = grp.sort_values("exam_no")
            subj = grp["subject"].iloc[0]
            ability = ability_map.get(subj, 0.5)
            diff = topic_difficulty.get(topic, 3.0) / 5.0
            importance = prajna_importance.get(topic, 0.5)

            accs = grp["accuracy_pct"].values
            times = grp["time_min"].values
            rolling_acc = np.mean(accs) / 100.0
            avg_time = np.mean(times)
            max_time = results_df["time_min"].max() or 1

            streak = 0
            for a in accs:
                if a >= 50:
                    streak = streak + 1 if streak > 0 else 1
                else:
                    streak = streak - 1 if streak < 0 else -1

            feat = np.array([[
                rolling_acc, ability, diff, importance,
                avg_time / max_time,
                np.clip(streak / 5.0, -1, 1),
                grp["exam_no"].max() / 10.0,
            ]])
            p = self.predict_proba(feat)[0]

            results.append({
                "micro_topic": topic, "subject": subj,
                "past_accuracy": round(float(np.mean(accs)), 1),
                "p_mistake": round(float(p), 3),
                "importance": importance,
                "trend": "improving" if len(accs) > 1 and accs[-1] > accs[0] else "declining",
            })

        results.sort(key=lambda x: x["p_mistake"], reverse=True)
        return results

    def save(self, path):
        if self.model is None:
            return
        data = {
            "coef": self.model.coef_.tolist(),
            "intercept": self.model.intercept_.tolist(),
            "classes": self.model.classes_.tolist(),
            "scaler_mean": self.scaler.mean_.tolist(),
            "scaler_scale": self.scaler.scale_.tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f)

    def load(self, path):
        with open(path, "r") as f:
            data = json.load(f)
        self.model = LogisticRegression()
        self.model.coef_ = np.array(data["coef"])
        self.model.intercept_ = np.array(data["intercept"])
        self.model.classes_ = np.array(data["classes"])
        self.scaler.mean_ = np.array(data["scaler_mean"])
        self.scaler.scale_ = np.array(data["scaler_scale"])
