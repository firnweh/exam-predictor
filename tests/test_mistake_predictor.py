import pytest
import pandas as pd
import numpy as np
from analysis.mistake_predictor import MistakePredictor

@pytest.fixture
def sample_data():
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
