# Mistake Analysis Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a "Mistake Analysis" Streamlit tab with Center View (aggregate patterns) and Student View (logistic regression predictions).

**Architecture:** Two standalone Python modules (`mistake_analyzer.py` for aggregations, `mistake_predictor.py` for logistic regression) wired into a new Streamlit tab via sidebar nav.

**Tech Stack:** pandas, sklearn (LogisticRegression), plotly, streamlit, joblib (for model serialization)

---

### Task 1: Center View — Mistake Analyzer Module

**Files:**
- Create: `analysis/mistake_analyzer.py`
- Test: `tests/test_mistake_analyzer.py`

**Step 1: Write the failing tests**

```python
# tests/test_mistake_analyzer.py
import pytest
import pandas as pd
from analysis.mistake_analyzer import MistakeAnalyzer


@pytest.fixture
def sample_results():
    """Minimal results DataFrame matching neet_results_v2.csv schema."""
    return pd.DataFrame({
        "student_id": ["S1","S1","S1","S2","S2","S2"],
        "coaching": ["PW Kota"]*3 + ["PW Delhi"]*3,
        "subject": ["Physics","Physics","Chemistry","Physics","Physics","Chemistry"],
        "chapter": ["Kinematics","Optics","Bonding","Kinematics","Optics","Bonding"],
        "micro_topic": ["1D Motion","Refraction","Ionic","1D Motion","Refraction","Ionic"],
        "total_qs": [5,4,3,5,4,3],
        "correct": [1,3,2,2,1,1],
        "wrong": [4,1,1,3,3,2],
        "accuracy_pct": [20.0,75.0,66.7,40.0,25.0,33.3],
        "time_min": [8.0,5.0,4.0,6.0,7.0,5.0],
        "exam_no": [1]*6,
    })


@pytest.fixture
def analyzer(sample_results):
    return MistakeAnalyzer(sample_results)


def test_error_rates(analyzer):
    er = analyzer.error_rates()
    assert "micro_topic" in er.columns
    assert "error_rate" in er.columns
    row = er[er["micro_topic"] == "1D Motion"].iloc[0]
    assert abs(row["error_rate"] - 0.70) < 0.01


def test_danger_zones(analyzer):
    prajna = {"1D Motion": 0.90, "Refraction": 0.80, "Ionic": 0.50}
    dz = analyzer.danger_zones(prajna, error_threshold=0.5)
    assert len(dz) >= 1
    assert "danger_score" in dz.columns
    top = dz.iloc[0]
    assert top["micro_topic"] == "1D Motion"
    assert abs(top["danger_score"] - 0.63) < 0.01


def test_cofailure_matrix(analyzer):
    cf = analyzer.cofailure_pairs(fail_threshold=50)
    assert isinstance(cf, list)
    for pair in cf:
        assert "topic_a" in pair
        assert "topic_b" in pair
        assert "cofailure_pct" in pair


def test_time_vs_accuracy(analyzer):
    tva = analyzer.time_vs_accuracy()
    assert "micro_topic" in tva.columns
    assert "avg_time" in tva.columns
    assert "avg_accuracy" in tva.columns
    assert "subject" in tva.columns
```

**Step 2: Run tests to verify they fail**

Run: `export PATH="$PATH:/Users/aman/Library/Python/3.9/bin" && cd /Users/aman/exam-predictor && python -m pytest tests/test_mistake_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'analysis.mistake_analyzer'`

**Step 3: Implement mistake_analyzer.py**

```python
# analysis/mistake_analyzer.py
"""
PRAJNA Mistake Analyzer — Center View Aggregations
===================================================
Computes aggregate mistake patterns from student results:
- error_rates: per-topic wrong/total ratio
- danger_zones: high error + high PRAJNA probability
- cofailure_pairs: P(fail B | fail A) — correlated weaknesses
- time_vs_accuracy: scatter data for teaching insight
"""
import pandas as pd
import numpy as np
from collections import defaultdict


class MistakeAnalyzer:
    def __init__(self, results_df: pd.DataFrame):
        self.df = results_df

    def error_rates(self, group_by=None):
        """Per-topic error rate = avg(wrong / total_qs) across all students."""
        cols = ["micro_topic", "subject"]
        if group_by:
            cols.append(group_by)
        g = self.df.groupby(cols, as_index=False).agg(
            total_wrong=("wrong", "sum"),
            total_qs=("total_qs", "sum"),
            student_count=("student_id", "nunique"),
        )
        g["error_rate"] = (g["total_wrong"] / g["total_qs"]).clip(0, 1)
        return g.sort_values("error_rate", ascending=False).reset_index(drop=True)

    def danger_zones(self, prajna_probs: dict, error_threshold=0.4, top_n=15):
        """Topics where students fail AND PRAJNA predicts appearance."""
        er = self.error_rates()
        er["prajna_prob"] = er["micro_topic"].map(prajna_probs).fillna(0)
        er["danger_score"] = er["error_rate"] * er["prajna_prob"]
        dz = er[er["error_rate"] >= error_threshold].copy()
        dz = dz.sort_values("danger_score", ascending=False).head(top_n)
        return dz.reset_index(drop=True)

    def cofailure_pairs(self, fail_threshold=50, min_students=2, top_n=15):
        """P(fail B | fail A) — which topic failures correlate."""
        agg = self.df.groupby(["student_id", "micro_topic"], as_index=False).agg(
            acc=("accuracy_pct", "mean")
        )
        agg["failed"] = (agg["acc"] < fail_threshold).astype(int)
        pivot = agg.pivot_table(index="student_id", columns="micro_topic",
                                values="failed", fill_value=0)

        topics = list(pivot.columns)
        pairs = []
        for i, a in enumerate(topics):
            fails_a = pivot[a].sum()
            if fails_a < min_students:
                continue
            for b in topics[i + 1:]:
                both = ((pivot[a] == 1) & (pivot[b] == 1)).sum()
                if both < min_students:
                    continue
                p_b_given_a = both / fails_a
                fails_b = pivot[b].sum()
                p_a_given_b = both / fails_b if fails_b else 0
                pairs.append({
                    "topic_a": a, "topic_b": b,
                    "cofailure_pct": round(max(p_b_given_a, p_a_given_b) * 100, 1),
                    "both_fail_count": int(both),
                })
        pairs.sort(key=lambda x: x["cofailure_pct"], reverse=True)
        return pairs[:top_n]

    def time_vs_accuracy(self):
        """Avg time spent vs avg accuracy per topic."""
        g = self.df.groupby(["micro_topic", "subject"], as_index=False).agg(
            avg_time=("time_min", "mean"),
            avg_accuracy=("accuracy_pct", "mean"),
            student_count=("student_id", "nunique"),
        )
        return g.sort_values("avg_accuracy").reset_index(drop=True)
```

**Step 4: Run tests to verify they pass**

Run: `export PATH="$PATH:/Users/aman/Library/Python/3.9/bin" && cd /Users/aman/exam-predictor && python -m pytest tests/test_mistake_analyzer.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add analysis/mistake_analyzer.py tests/test_mistake_analyzer.py
git commit -m "feat: mistake analyzer — center view aggregations (error rates, danger zones, co-failure, time-vs-accuracy)"
```

---

### Task 2: Student View — Mistake Predictor Module

**Files:**
- Create: `analysis/mistake_predictor.py`
- Test: `tests/test_mistake_predictor.py`

**Step 1: Write the failing tests**

```python
# tests/test_mistake_predictor.py
import pytest
import pandas as pd
import numpy as np
from analysis.mistake_predictor import MistakePredictor


@pytest.fixture
def sample_data():
    """10-exam trajectory for 2 students on 3 topics."""
    rows = []
    rng = np.random.RandomState(42)
    for sid in ["S1", "S2"]:
        for exam in range(1, 11):
            for topic, subj, base_acc in [
                ("1D Motion", "Physics", 30),
                ("Refraction", "Physics", 60),
                ("Ionic Bond", "Chemistry", 70),
            ]:
                acc = min(100, base_acc + exam * 3 + rng.randint(-5, 5))
                total = 5
                correct = int(round(acc / 100 * total))
                wrong = total - correct
                rows.append({
                    "student_id": sid, "exam_no": exam,
                    "subject": subj, "micro_topic": topic,
                    "total_qs": total, "correct": correct,
                    "wrong": wrong, "accuracy_pct": acc,
                    "time_min": rng.uniform(3, 10),
                })
    return pd.DataFrame(rows)


@pytest.fixture
def abilities():
    return pd.DataFrame({
        "student_id": ["S1", "S2"],
        "ability_physics": [0.4, 0.7],
        "ability_chemistry": [0.6, 0.5],
    })


@pytest.fixture
def topic_difficulty():
    return {"1D Motion": 3.2, "Refraction": 2.8, "Ionic Bond": 2.5}


@pytest.fixture
def prajna_importance():
    return {"1D Motion": 0.85, "Refraction": 0.72, "Ionic Bond": 0.60}


def test_feature_engineering(sample_data, abilities, topic_difficulty, prajna_importance):
    mp = MistakePredictor()
    X, y = mp.build_features(sample_data, abilities, topic_difficulty, prajna_importance,
                             train_exams=range(1, 9))
    assert X.shape[0] > 0
    assert X.shape[1] == 7
    assert len(y) == X.shape[0]
    assert set(np.unique(y)).issubset({0, 1})


def test_train_and_predict(sample_data, abilities, topic_difficulty, prajna_importance):
    mp = MistakePredictor()
    X_train, y_train = mp.build_features(sample_data, abilities, topic_difficulty,
                                          prajna_importance, train_exams=range(1, 9))
    mp.train(X_train, y_train)
    assert mp.model is not None

    X_test, y_test = mp.build_features(sample_data, abilities, topic_difficulty,
                                        prajna_importance, train_exams=range(9, 11))
    preds = mp.predict_proba(X_test)
    assert len(preds) == X_test.shape[0]
    assert all(0 <= p <= 1 for p in preds)


def test_feature_importances(sample_data, abilities, topic_difficulty, prajna_importance):
    mp = MistakePredictor()
    X, y = mp.build_features(sample_data, abilities, topic_difficulty,
                              prajna_importance, train_exams=range(1, 9))
    mp.train(X, y)
    fi = mp.feature_importances()
    assert isinstance(fi, dict)
    assert len(fi) == 7
    assert "rolling_accuracy" in fi


def test_student_predictions(sample_data, abilities, topic_difficulty, prajna_importance):
    mp = MistakePredictor()
    X, y = mp.build_features(sample_data, abilities, topic_difficulty,
                              prajna_importance, train_exams=range(1, 9))
    mp.train(X, y)
    result = mp.predict_for_student(sample_data, abilities, topic_difficulty,
                                     prajna_importance, student_id="S1")
    assert isinstance(result, list)
    assert len(result) > 0
    assert "micro_topic" in result[0]
    assert "p_mistake" in result[0]
```

**Step 2: Run tests to verify they fail**

Run: `export PATH="$PATH:/Users/aman/Library/Python/3.9/bin" && cd /Users/aman/exam-predictor && python -m pytest tests/test_mistake_predictor.py -v`
Expected: FAIL — `ModuleNotFoundError`

**Step 3: Implement mistake_predictor.py**

```python
# analysis/mistake_predictor.py
"""
PRAJNA Mistake Predictor — Student View Logistic Regression
============================================================
Trains a logistic regression model to predict P(mistake) per student per topic.
Features: rolling_accuracy, ability_score, topic_difficulty, exam_importance,
          avg_time_spent, streak, exam_number.
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
        """Build feature matrix from results + enrichment data."""
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
                    rolling_acc,
                    ability,
                    diff,
                    importance,
                    norm_time,
                    np.clip(streak / 5.0, -1, 1),
                    exams[i] / 10.0,
                ]
                rows.append(feat)
                labels.append(1 if accs[i] < 50 else 0)

        if not rows:
            return np.empty((0, 7)), np.empty(0)

        X = np.array(rows, dtype=np.float64)
        y = np.array(labels, dtype=np.int32)
        return X, y

    def train(self, X, y):
        """Train logistic regression on feature matrix."""
        if len(X) == 0:
            return
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        self.model = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        self.model.fit(X_scaled, y)

    def predict_proba(self, X):
        """Return P(mistake) for each row."""
        if self.model is None or len(X) == 0:
            return np.array([])
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importances(self):
        """Return dict of feature name -> absolute coefficient as percentage."""
        if self.model is None:
            return {}
        coefs = np.abs(self.model.coef_[0])
        total = coefs.sum() or 1
        return {name: round(float(c / total) * 100, 1)
                for name, c in zip(FEATURE_NAMES, coefs)}

    def predict_for_student(self, results_df, abilities_df, topic_difficulty,
                            prajna_importance, student_id):
        """Predict P(mistake) per topic for a specific student."""
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
        """Save trained model using JSON-safe format."""
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
        """Load trained model from JSON."""
        with open(path, "r") as f:
            data = json.load(f)
        self.model = LogisticRegression()
        self.model.coef_ = np.array(data["coef"])
        self.model.intercept_ = np.array(data["intercept"])
        self.model.classes_ = np.array(data["classes"])
        self.scaler.mean_ = np.array(data["scaler_mean"])
        self.scaler.scale_ = np.array(data["scaler_scale"])
```

**Step 4: Run tests to verify they pass**

Run: `export PATH="$PATH:/Users/aman/Library/Python/3.9/bin" && cd /Users/aman/exam-predictor && python -m pytest tests/test_mistake_predictor.py -v`
Expected: 4 passed

**Step 5: Commit**

```bash
git add analysis/mistake_predictor.py tests/test_mistake_predictor.py
git commit -m "feat: mistake predictor — logistic regression P(mistake) per student per topic"
```

---

### Task 3: Wire into Streamlit Dashboard

**Files:**
- Modify: `dashboard/app.py` — add nav item + tab handler

**Step 1: Add nav item to sidebar radio list (around line 643)**

Add `"🧪 Mistake Analysis"` between `"🤖 Ask PRAJNA"` and `"🔌 API Docs"`.

**Step 2: Add the tab handler before the API Docs handler**

Add the full Mistake Analysis tab code (center view with danger zones, co-failure, time-vs-accuracy + student view with predicted miss probability, personal danger zones, improvement trajectory, feature importances). Uses `MistakeAnalyzer` for center view and `MistakePredictor` for student view.

Key patterns to follow:
- Use `@st.cache_data(ttl=600)` for data loading
- Use `@st.cache_resource` for model training
- Use `plotly_dark` template with `paper_bgcolor="#0f0f1a"` and `plot_bgcolor="#131320"` for charts
- Use `st.dataframe(..., use_container_width=True, hide_index=True)` for tables

**Step 3: Verify in browser**

Run: `cd /Users/aman/exam-predictor && streamlit run dashboard/app.py`
Navigate to sidebar → "🧪 Mistake Analysis"
- Toggle Center View: danger zones table, co-failure pairs, scatter plot
- Toggle Student View: select student, predicted miss table, danger zones, trajectory, feature importance

**Step 4: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: mistake analysis Streamlit tab — center + student views with logistic regression"
```

---

### Task 4: Final Validation & Push

**Step 1: Run all tests**

Run: `export PATH="$PATH:/Users/aman/Library/Python/3.9/bin" && cd /Users/aman/exam-predictor && python -m pytest tests/ -v --tb=short`
Expected: All tests pass

**Step 2: Push to GitHub**

```bash
git push origin main
```

**Step 3: Redeploy**

```bash
vercel --prod --yes
```
