# Exam Predictor Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a local Python system that analyzes 40 years of JEE/NEET question papers to find trends, classify difficulty, detect patterns, and predict likely topics.

**Architecture:** One-time LLM extraction produces structured JSON, loaded into SQLite. Python analysis engine computes trends, difficulty, patterns, and predictions. Streamlit dashboard displays results. Hosted online via Streamlit Community Cloud or ngrok tunnel from local machine.

**Tech Stack:** Python 3.11+, SQLite, Pandas, Scikit-learn, Plotly, Streamlit

---

### Task 1: Project Scaffolding & Dependencies

**Files:**
- Create: `requirements.txt`
- Create: `data/raw/.gitkeep`
- Create: `data/extracted/.gitkeep`

**Step 1: Create requirements.txt**

```txt
pandas>=2.0.0
scikit-learn>=1.3.0
plotly>=5.15.0
streamlit>=1.28.0
fpdf2>=2.7.0
```

**Step 2: Create directory structure**

Run:
```bash
cd /Users/aman/exam-predictor
mkdir -p data/raw data/extracted extraction analysis dashboard utils tests
touch data/raw/.gitkeep data/extracted/.gitkeep
```

**Step 3: Create virtual environment and install**

Run:
```bash
cd /Users/aman/exam-predictor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
Expected: All packages install successfully.

**Step 4: Commit**

```bash
git add requirements.txt data/
git commit -m "feat: add project scaffolding and dependencies"
```

---

### Task 2: Extraction Prompt Template

**Files:**
- Create: `extraction/prompt_template.md`

**Step 1: Write the extraction prompt**

This is the prompt the user will paste into Claude/ChatGPT along with question paper text. It must produce consistent, parseable JSON.

```markdown
# Exam Question Extraction Prompt

You are an expert on Indian competitive exams (JEE Main, JEE Advanced, NEET).

I will paste questions from an exam paper. For EACH question, output a JSON object with these exact fields:

- "id": string — format "{EXAM}{YEAR}_{SHIFT}_{QN}" e.g. "JEE_ADV_2019_P1_Q12"
- "exam": string — one of "JEE Main", "JEE Advanced", "NEET"
- "year": integer — e.g. 2019
- "shift": string — e.g. "Paper 1", "Shift 1", "Morning", or "N/A"
- "subject": string — one of "Physics", "Chemistry", "Mathematics", "Biology"
- "topic": string — broad topic e.g. "Electromagnetism", "Organic Chemistry"
- "micro_topic": string — specific subtopic e.g. "Faraday's Law of Induction"
- "question_text": string — full question text
- "question_type": string — one of "MCQ_single", "MCQ_multi", "integer", "numerical", "matrix_match", "assertion_reason", "subjective"
- "difficulty": integer 1-5 — (1=easy, 5=extremely hard) based on concept depth, calculation complexity, and multi-step reasoning
- "concepts_tested": list of strings — the specific concepts needed to solve
- "answer": string — correct answer or "N/A" if unknown
- "marks": integer — marks for correct answer, or 4 as default

Output as a JSON array. Example:

```json
[
  {
    "id": "JEE_ADV_2019_P1_Q1",
    "exam": "JEE Advanced",
    "year": 2019,
    "shift": "Paper 1",
    "subject": "Physics",
    "topic": "Mechanics",
    "micro_topic": "Rotational Dynamics",
    "question_text": "A uniform rod of length L...",
    "question_type": "MCQ_single",
    "difficulty": 3,
    "concepts_tested": ["moment of inertia", "angular momentum conservation"],
    "answer": "B",
    "marks": 3
  }
]
```

IMPORTANT RULES:
1. Use consistent topic names. Refer to standard textbook chapter names.
2. micro_topic should be specific enough to identify the exact concept (not just "Mechanics" but "Projectile Motion on Inclined Plane").
3. Difficulty rating guide:
   - 1: Direct formula application
   - 2: Single concept, some calculation
   - 3: Multi-concept or tricky application
   - 4: Complex multi-step or unusual approach needed
   - 5: Olympiad-level or requires deep insight
4. Output ONLY the JSON array, nothing else.

---

EXAM: [PASTE EXAM NAME HERE]
YEAR: [PASTE YEAR HERE]
SHIFT: [PASTE SHIFT HERE]

QUESTIONS:
[PASTE QUESTIONS HERE]
```

**Step 2: Commit**

```bash
git add extraction/prompt_template.md
git commit -m "feat: add LLM extraction prompt template"
```

---

### Task 3: Database Schema & Helpers

**Files:**
- Create: `utils/db.py`
- Create: `tests/test_db.py`

**Step 1: Write the failing test**

```python
# tests/test_db.py
import os
import sqlite3
import json
from utils.db import init_db, insert_questions, get_all_questions, get_topics_hierarchy

TEST_DB = "data/test_exam.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_init_db_creates_tables():
    init_db(TEST_DB)
    conn = sqlite3.connect(TEST_DB)
    cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = {row[0] for row in cursor.fetchall()}
    conn.close()
    assert "questions" in tables
    assert "topics" in tables


def test_insert_and_retrieve_questions():
    init_db(TEST_DB)
    questions = [
        {
            "id": "JEE_ADV_2019_P1_Q1",
            "exam": "JEE Advanced",
            "year": 2019,
            "shift": "Paper 1",
            "subject": "Physics",
            "topic": "Mechanics",
            "micro_topic": "Rotational Dynamics",
            "question_text": "A uniform rod of length L...",
            "question_type": "MCQ_single",
            "difficulty": 3,
            "concepts_tested": ["moment of inertia", "angular momentum"],
            "answer": "B",
            "marks": 3,
        }
    ]
    insert_questions(TEST_DB, questions)
    result = get_all_questions(TEST_DB)
    assert len(result) == 1
    assert result[0]["id"] == "JEE_ADV_2019_P1_Q1"
    assert result[0]["subject"] == "Physics"
    assert result[0]["concepts_tested"] == ["moment of inertia", "angular momentum"]


def test_topics_hierarchy():
    init_db(TEST_DB)
    questions = [
        {
            "id": "Q1",
            "exam": "JEE Advanced",
            "year": 2019,
            "shift": "P1",
            "subject": "Physics",
            "topic": "Mechanics",
            "micro_topic": "Rotational Dynamics",
            "question_text": "...",
            "question_type": "MCQ_single",
            "difficulty": 3,
            "concepts_tested": [],
            "answer": "B",
            "marks": 3,
        },
        {
            "id": "Q2",
            "exam": "NEET",
            "year": 2020,
            "shift": "N/A",
            "subject": "Physics",
            "topic": "Mechanics",
            "micro_topic": "Newton's Laws",
            "question_text": "...",
            "question_type": "MCQ_single",
            "difficulty": 2,
            "concepts_tested": [],
            "answer": "A",
            "marks": 4,
        },
    ]
    insert_questions(TEST_DB, questions)
    hierarchy = get_topics_hierarchy(TEST_DB)
    assert "Physics" in hierarchy
    assert "Mechanics" in hierarchy["Physics"]
    assert "Rotational Dynamics" in hierarchy["Physics"]["Mechanics"]
    assert "Newton's Laws" in hierarchy["Physics"]["Mechanics"]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/aman/exam-predictor && source venv/bin/activate && python -m pytest tests/test_db.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.db'`

**Step 3: Write the implementation**

```python
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
```

**Step 4: Create `utils/__init__.py` and `tests/__init__.py`**

```bash
touch utils/__init__.py tests/__init__.py
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_db.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
git add utils/ tests/
git commit -m "feat: add SQLite database schema and helpers"
```

---

### Task 4: JSON Loader (extracted JSON → SQLite)

**Files:**
- Create: `utils/loader.py`
- Create: `tests/test_loader.py`

**Step 1: Write the failing test**

```python
# tests/test_loader.py
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
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'utils.loader'`

**Step 3: Write the implementation**

```python
# utils/loader.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_loader.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add utils/loader.py tests/test_loader.py
git commit -m "feat: add JSON loader for extracted question data"
```

---

### Task 5: Trend Analyzer

**Files:**
- Create: `analysis/trend_analyzer.py`
- Create: `tests/test_trend_analyzer.py`

**Step 1: Write the failing test**

```python
# tests/test_trend_analyzer.py
import os
from utils.db import init_db, insert_questions
from analysis.trend_analyzer import (
    topic_frequency_by_year,
    find_hot_cold_topics,
    detect_cycles,
)

TEST_DB = "data/test_trends.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _make_question(id, year, topic, micro_topic, exam="JEE Advanced", subject="Physics"):
    return {
        "id": id, "exam": exam, "year": year, "shift": "P1",
        "subject": subject, "topic": topic, "micro_topic": micro_topic,
        "question_text": "...", "question_type": "MCQ_single",
        "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4,
    }


def test_topic_frequency_by_year():
    questions = [
        _make_question("Q1", 2020, "Mechanics", "Kinematics"),
        _make_question("Q2", 2020, "Mechanics", "Kinematics"),
        _make_question("Q3", 2021, "Mechanics", "Kinematics"),
        _make_question("Q4", 2020, "Optics", "Refraction"),
    ]
    insert_questions(TEST_DB, questions)
    freq = topic_frequency_by_year(TEST_DB)
    assert freq.loc[("Mechanics", "Kinematics"), 2020] == 2
    assert freq.loc[("Mechanics", "Kinematics"), 2021] == 1
    assert freq.loc[("Optics", "Refraction"), 2020] == 1


def test_hot_cold_topics():
    questions = []
    # "Kinematics" appears every year 2018-2022 — hot
    for i, year in enumerate(range(2018, 2023)):
        questions.append(_make_question(f"K{i}", year, "Mechanics", "Kinematics"))
    # "Thermodynamics" only appeared in 2018 — cold
    questions.append(_make_question("T0", 2018, "Heat", "Thermodynamics"))
    insert_questions(TEST_DB, questions)

    hot, cold = find_hot_cold_topics(TEST_DB, recent_years=3, current_year=2022)
    hot_micros = [t[1] for t in hot]
    cold_micros = [t[1] for t in cold]
    assert "Kinematics" in hot_micros
    assert "Thermodynamics" in cold_micros


def test_detect_cycles():
    questions = []
    # Topic appears every 3 years: 2010, 2013, 2016, 2019, 2022
    for i, year in enumerate([2010, 2013, 2016, 2019, 2022]):
        questions.append(_make_question(f"C{i}", year, "Waves", "Doppler Effect"))
    # Fill gaps with another topic so the data spans full range
    for i, year in enumerate(range(2010, 2023)):
        questions.append(_make_question(f"F{i}", year, "Mechanics", "Friction"))
    insert_questions(TEST_DB, questions)

    cycles = detect_cycles(TEST_DB, min_occurrences=4)
    doppler_entry = [c for c in cycles if c["micro_topic"] == "Doppler Effect"]
    assert len(doppler_entry) == 1
    assert doppler_entry[0]["estimated_cycle_years"] == 3
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_trend_analyzer.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# analysis/trend_analyzer.py
import pandas as pd
from utils.db import get_questions_df


def topic_frequency_by_year(db_path="data/exam.db", exam=None):
    df = get_questions_df(db_path)
    if exam:
        df = df[df["exam"] == exam]
    freq = df.groupby(["topic", "micro_topic", "year"]).size().unstack(fill_value=0)
    return freq


def find_hot_cold_topics(db_path="data/exam.db", recent_years=3, current_year=None):
    df = get_questions_df(db_path)
    if current_year is None:
        current_year = df["year"].max()

    recent_cutoff = current_year - recent_years + 1

    topic_last_year = df.groupby(["topic", "micro_topic"])["year"].max()
    topic_recent_count = (
        df[df["year"] >= recent_cutoff]
        .groupby(["topic", "micro_topic"])
        .size()
    )

    hot = topic_recent_count.sort_values(ascending=False)
    hot_topics = [(idx, idx[1], count) for idx, count in hot.items()]

    cold_topics = []
    for idx, last_year in topic_last_year.items():
        if last_year < recent_cutoff:
            gap = current_year - last_year
            cold_topics.append((idx, idx[1], gap))
    cold_topics.sort(key=lambda x: x[2], reverse=True)

    return hot_topics, cold_topics


def detect_cycles(db_path="data/exam.db", min_occurrences=4):
    df = get_questions_df(db_path)
    results = []

    for (topic, micro_topic), group in df.groupby(["topic", "micro_topic"]):
        years = sorted(group["year"].unique())
        if len(years) < min_occurrences:
            continue

        gaps = [years[i + 1] - years[i] for i in range(len(years) - 1)]
        if not gaps:
            continue

        avg_gap = sum(gaps) / len(gaps)
        variance = sum((g - avg_gap) ** 2 for g in gaps) / len(gaps)

        if variance <= 1.5:
            results.append({
                "topic": topic,
                "micro_topic": micro_topic,
                "estimated_cycle_years": round(avg_gap),
                "appearances": years,
                "avg_gap": round(avg_gap, 1),
                "consistency": round(1 - (variance / (avg_gap ** 2 + 0.01)), 2),
            })

    results.sort(key=lambda x: x["consistency"], reverse=True)
    return results
```

**Step 4: Create `analysis/__init__.py`**

```bash
touch analysis/__init__.py
```

**Step 5: Run test to verify it passes**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_trend_analyzer.py -v`
Expected: 3 passed

**Step 6: Commit**

```bash
git add analysis/ tests/test_trend_analyzer.py
git commit -m "feat: add trend analyzer with frequency, hot/cold, and cycle detection"
```

---

### Task 6: Difficulty Classifier

**Files:**
- Create: `analysis/difficulty_classifier.py`
- Create: `tests/test_difficulty.py`

**Step 1: Write the failing test**

```python
# tests/test_difficulty.py
import os
from utils.db import init_db, insert_questions
from analysis.difficulty_classifier import classify_difficulty, get_difficulty_distribution

TEST_DB = "data/test_difficulty.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_classify_difficulty():
    questions = []
    for i in range(20):
        questions.append({
            "id": f"Q{i}", "exam": "JEE Advanced", "year": 2020,
            "shift": "P1", "subject": "Physics", "topic": "Mechanics",
            "micro_topic": "Kinematics", "question_text": "...",
            "question_type": "MCQ_single" if i < 10 else "integer",
            "difficulty": (i % 5) + 1, "concepts_tested": [],
            "answer": "A", "marks": 3 if i < 10 else 4,
        })
    insert_questions(TEST_DB, questions)
    result = classify_difficulty(TEST_DB)
    assert "difficulty_label" in result.columns
    assert set(result["difficulty_label"].unique()).issubset({"Easy", "Moderate", "Hard", "Very Hard"})


def test_difficulty_distribution():
    questions = [
        {"id": f"Q{i}", "exam": "JEE Advanced", "year": 2020,
         "shift": "P1", "subject": "Physics", "topic": "Mechanics",
         "micro_topic": "Kinematics", "question_text": "...",
         "question_type": "MCQ_single", "difficulty": (i % 5) + 1,
         "concepts_tested": [], "answer": "A", "marks": 4}
        for i in range(20)
    ]
    insert_questions(TEST_DB, questions)
    dist = get_difficulty_distribution(TEST_DB)
    assert "Physics" in dist.index.get_level_values("subject")
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_difficulty.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# analysis/difficulty_classifier.py
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
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_difficulty.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add analysis/difficulty_classifier.py tests/test_difficulty.py
git commit -m "feat: add difficulty classifier with labels and distribution"
```

---

### Task 7: Pattern Finder

**Files:**
- Create: `analysis/pattern_finder.py`
- Create: `tests/test_pattern_finder.py`

**Step 1: Write the failing test**

```python
# tests/test_pattern_finder.py
import os
from utils.db import init_db, insert_questions
from analysis.pattern_finder import (
    topic_cooccurrence,
    subject_weightage_over_time,
    cross_exam_correlation,
)

TEST_DB = "data/test_patterns.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _q(id, year, subject, topic, micro_topic, exam="JEE Advanced"):
    return {
        "id": id, "exam": exam, "year": year, "shift": "P1",
        "subject": subject, "topic": topic, "micro_topic": micro_topic,
        "question_text": "...", "question_type": "MCQ_single",
        "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4,
    }


def test_topic_cooccurrence():
    # Same paper, same year — these topics co-occur
    questions = [
        _q("Q1", 2020, "Physics", "Mechanics", "Kinematics"),
        _q("Q2", 2020, "Physics", "Mechanics", "Projectile Motion"),
        _q("Q3", 2020, "Physics", "Optics", "Refraction"),
    ]
    insert_questions(TEST_DB, questions)
    matrix = topic_cooccurrence(TEST_DB)
    assert matrix.loc["Kinematics", "Projectile Motion"] > 0
    assert matrix.loc["Kinematics", "Refraction"] > 0


def test_subject_weightage():
    questions = [
        _q(f"P{i}", 2020, "Physics", "M", "K") for i in range(3)
    ] + [
        _q(f"C{i}", 2020, "Chemistry", "O", "R") for i in range(7)
    ]
    insert_questions(TEST_DB, questions)
    weights = subject_weightage_over_time(TEST_DB)
    assert weights.loc[2020, "Physics"] < weights.loc[2020, "Chemistry"]


def test_cross_exam_correlation():
    # Topic appears in JEE 2020, then NEET 2021
    questions = [
        _q("J1", 2020, "Physics", "Waves", "Doppler", exam="JEE Advanced"),
        _q("N1", 2021, "Physics", "Waves", "Doppler", exam="NEET"),
    ]
    insert_questions(TEST_DB, questions)
    corr = cross_exam_correlation(TEST_DB)
    doppler = [c for c in corr if c["micro_topic"] == "Doppler"]
    assert len(doppler) == 1
    assert doppler[0]["lag_years"] == 1
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_pattern_finder.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# analysis/pattern_finder.py
import pandas as pd
from itertools import combinations
from utils.db import get_questions_df


def topic_cooccurrence(db_path="data/exam.db"):
    df = get_questions_df(db_path)
    all_micros = df["micro_topic"].unique()
    cooccur = pd.DataFrame(0, index=all_micros, columns=all_micros)

    for (exam, year, shift), group in df.groupby(["exam", "year", "shift"]):
        micros = group["micro_topic"].unique()
        for a, b in combinations(micros, 2):
            cooccur.loc[a, b] += 1
            cooccur.loc[b, a] += 1

    return cooccur


def subject_weightage_over_time(db_path="data/exam.db", exam=None):
    df = get_questions_df(db_path)
    if exam:
        df = df[df["exam"] == exam]
    counts = df.groupby(["year", "subject"]).size().unstack(fill_value=0)
    totals = counts.sum(axis=1)
    weights = counts.div(totals, axis=0)
    return weights


def cross_exam_correlation(db_path="data/exam.db"):
    df = get_questions_df(db_path)
    results = []

    for micro_topic in df["micro_topic"].unique():
        subset = df[df["micro_topic"] == micro_topic]
        exams = subset["exam"].unique()
        if len(exams) < 2:
            continue

        for exam_a in exams:
            for exam_b in exams:
                if exam_a == exam_b:
                    continue
                years_a = sorted(subset[subset["exam"] == exam_a]["year"].unique())
                years_b = sorted(subset[subset["exam"] == exam_b]["year"].unique())

                for ya in years_a:
                    for yb in years_b:
                        lag = yb - ya
                        if 0 < lag <= 3:
                            results.append({
                                "micro_topic": micro_topic,
                                "from_exam": exam_a,
                                "to_exam": exam_b,
                                "from_year": ya,
                                "to_year": yb,
                                "lag_years": lag,
                            })

    results.sort(key=lambda x: x["lag_years"])
    return results
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_pattern_finder.py -v`
Expected: 3 passed

**Step 5: Commit**

```bash
git add analysis/pattern_finder.py tests/test_pattern_finder.py
git commit -m "feat: add pattern finder with co-occurrence, weightage, cross-exam"
```

---

### Task 8: Predictor

**Files:**
- Create: `analysis/predictor.py`
- Create: `tests/test_predictor.py`

**Step 1: Write the failing test**

```python
# tests/test_predictor.py
import os
from utils.db import init_db, insert_questions
from analysis.predictor import predict_topics

TEST_DB = "data/test_predictor.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)
    init_db(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def _q(id, year, topic, micro_topic, exam="JEE Advanced"):
    return {
        "id": id, "exam": exam, "year": year, "shift": "P1",
        "subject": "Physics", "topic": topic, "micro_topic": micro_topic,
        "question_text": "...", "question_type": "MCQ_single",
        "difficulty": 3, "concepts_tested": [], "answer": "A", "marks": 4,
    }


def test_predict_topics_returns_ranked_list():
    questions = []
    # Frequent topic — should rank high
    for i, year in enumerate(range(2015, 2023)):
        questions.append(_q(f"K{i}", year, "Mechanics", "Kinematics"))
    # Cyclical topic every 2 years — due in 2024
    for i, year in enumerate([2018, 2020, 2022]):
        questions.append(_q(f"D{i}", year, "Waves", "Doppler Effect"))
    # Dormant topic — last seen 2016
    questions.append(_q("T0", 2016, "Heat", "Carnot Cycle"))
    insert_questions(TEST_DB, questions)

    predictions = predict_topics(TEST_DB, target_year=2024)
    assert len(predictions) > 0
    assert "micro_topic" in predictions[0]
    assert "score" in predictions[0]
    # Kinematics should be near the top (appears every year)
    top_5_micros = [p["micro_topic"] for p in predictions[:5]]
    assert "Kinematics" in top_5_micros


def test_predict_topics_includes_reasoning():
    questions = [_q(f"Q{i}", 2020 + i, "Mechanics", "Kinematics") for i in range(3)]
    insert_questions(TEST_DB, questions)
    predictions = predict_topics(TEST_DB, target_year=2024)
    assert "reasons" in predictions[0]
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_predictor.py -v`
Expected: FAIL

**Step 3: Write the implementation**

```python
# analysis/predictor.py
from utils.db import get_questions_df
from analysis.trend_analyzer import detect_cycles


WEIGHTS = {
    "frequency_trend": 0.30,
    "cycle_match": 0.25,
    "gap_bonus": 0.20,
    "cross_exam": 0.15,
    "recency": 0.10,
}


def predict_topics(db_path="data/exam.db", target_year=2026, exam=None):
    df = get_questions_df(db_path)
    if exam:
        df = df[df["exam"] == exam]

    all_micros = df.groupby(["topic", "micro_topic"]).agg(
        total_count=("year", "size"),
        years_list=("year", lambda x: sorted(x.unique())),
        last_year=("year", "max"),
        first_year=("year", "min"),
    ).reset_index()

    cycles = detect_cycles(db_path)
    cycle_map = {c["micro_topic"]: c for c in cycles}

    max_year = df["year"].max()
    min_year = df["year"].min()
    year_span = max_year - min_year + 1

    # Load full dataset for cross-exam signals
    full_df = get_questions_df(db_path)

    predictions = []

    for _, row in all_micros.iterrows():
        reasons = []
        score = 0.0

        # 1. Frequency trend
        freq = row["total_count"] / year_span
        freq_score = min(freq * 10, 1.0)
        score += WEIGHTS["frequency_trend"] * freq_score
        if freq_score > 0.5:
            reasons.append(f"High frequency: appeared {row['total_count']} times in {year_span} years")

        # 2. Cycle match
        cycle_score = 0.0
        if row["micro_topic"] in cycle_map:
            cycle = cycle_map[row["micro_topic"]]
            cycle_len = cycle["estimated_cycle_years"]
            years_since = target_year - row["last_year"]
            if years_since % cycle_len == 0:
                cycle_score = 1.0
                reasons.append(f"Cycle match: appears every ~{cycle_len} years, due in {target_year}")
            elif abs(years_since % cycle_len) <= 1:
                cycle_score = 0.5
                reasons.append(f"Near cycle: ~{cycle_len}-year cycle, close to due")
        score += WEIGHTS["cycle_match"] * cycle_score

        # 3. Gap bonus
        gap = target_year - row["last_year"]
        gap_score = min(gap / 10, 1.0) if gap >= 3 else 0.0
        score += WEIGHTS["gap_bonus"] * gap_score
        if gap >= 3:
            reasons.append(f"Gap bonus: not seen in {gap} years")

        # 4. Cross-exam signal
        micro_df = full_df[full_df["micro_topic"] == row["micro_topic"]]
        exams_with_topic = micro_df["exam"].unique()
        cross_score = min(len(exams_with_topic) / 3, 1.0)
        score += WEIGHTS["cross_exam"] * cross_score
        if len(exams_with_topic) > 1:
            reasons.append(f"Cross-exam: appears in {', '.join(exams_with_topic)}")

        # 5. Recency
        recency = max(0, 1 - (gap / year_span))
        score += WEIGHTS["recency"] * recency

        if not reasons:
            reasons.append("Low signal — limited historical data")

        predictions.append({
            "topic": row["topic"],
            "micro_topic": row["micro_topic"],
            "score": round(score, 4),
            "total_appearances": row["total_count"],
            "last_appeared": row["last_year"],
            "reasons": reasons,
        })

    predictions.sort(key=lambda x: x["score"], reverse=True)
    return predictions
```

**Step 4: Run test to verify it passes**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/test_predictor.py -v`
Expected: 2 passed

**Step 5: Commit**

```bash
git add analysis/predictor.py tests/test_predictor.py
git commit -m "feat: add weighted topic predictor with reasoning"
```

---

### Task 9: Streamlit Dashboard

**Files:**
- Create: `dashboard/app.py`

**Step 1: Write the dashboard**

```python
# dashboard/app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.db import get_questions_df, get_topics_hierarchy
from analysis.trend_analyzer import topic_frequency_by_year, find_hot_cold_topics, detect_cycles
from analysis.difficulty_classifier import classify_difficulty, difficulty_over_time
from analysis.pattern_finder import topic_cooccurrence, subject_weightage_over_time
from analysis.predictor import predict_topics

DB_PATH = "data/exam.db"

st.set_page_config(page_title="Exam Predictor", page_icon="📊", layout="wide")
st.title("Exam Predictor — JEE & NEET Analysis")

if not os.path.exists(DB_PATH):
    st.error("Database not found. Run the loader first: python run.py")
    st.stop()

df = get_questions_df(DB_PATH)

# Sidebar filters
st.sidebar.header("Filters")
exams = ["All"] + sorted(df["exam"].unique().tolist())
selected_exam = st.sidebar.selectbox("Exam", exams)
subjects = ["All"] + sorted(df["subject"].unique().tolist())
selected_subject = st.sidebar.selectbox("Subject", subjects)

filtered = df.copy()
if selected_exam != "All":
    filtered = filtered[filtered["exam"] == selected_exam]
if selected_subject != "All":
    filtered = filtered[filtered["subject"] == selected_subject]

st.sidebar.metric("Total Questions", len(filtered))
st.sidebar.metric("Year Range", f"{filtered['year'].min()}–{filtered['year'].max()}")

# --- Tabs ---
tab1, tab2, tab3, tab4 = st.tabs([
    "Topic Heatmap", "Predictions", "Question Explorer", "Practice Sets"
])

with tab1:
    st.header("Topic Frequency Heatmap")
    exam_filter = selected_exam if selected_exam != "All" else None
    freq = topic_frequency_by_year(DB_PATH, exam=exam_filter)
    if selected_subject != "All":
        subject_topics = df[df["subject"] == selected_subject][["topic", "micro_topic"]].drop_duplicates()
        valid_idx = [idx for idx in freq.index if idx in list(zip(subject_topics["topic"], subject_topics["micro_topic"]))]
        freq = freq.loc[valid_idx] if valid_idx else freq

    if not freq.empty:
        fig = px.imshow(
            freq.values,
            labels=dict(x="Year", y="Topic", color="Questions"),
            x=[str(c) for c in freq.columns],
            y=[f"{t} > {m}" for t, m in freq.index],
            color_continuous_scale="YlOrRd",
            aspect="auto",
        )
        fig.update_layout(height=max(400, len(freq) * 25))
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Hot & Cold Topics")
    col1, col2 = st.columns(2)
    hot, cold = find_hot_cold_topics(DB_PATH, recent_years=3)
    with col1:
        st.markdown("**Hot Topics** (frequent recently)")
        for topic_idx, micro, count in hot[:15]:
            st.write(f"- **{micro}** ({count} times in last 3 years)")
    with col2:
        st.markdown("**Cold Topics** (dormant)")
        for topic_idx, micro, gap in cold[:15]:
            st.write(f"- **{micro}** (last seen {gap} years ago)")

    st.subheader("Cyclical Topics")
    cycles = detect_cycles(DB_PATH)
    if cycles:
        cycle_df = pd.DataFrame(cycles)
        st.dataframe(cycle_df[["topic", "micro_topic", "estimated_cycle_years", "avg_gap", "consistency"]])

with tab2:
    st.header("Topic Predictions")
    target_year = st.number_input("Predict for year", value=2026, min_value=2024, max_value=2030)
    predictions = predict_topics(
        DB_PATH,
        target_year=target_year,
        exam=exam_filter,
    )
    top_n = st.slider("Show top N predictions", 10, 100, 50)

    pred_df = pd.DataFrame(predictions[:top_n])
    if not pred_df.empty:
        fig = px.bar(
            pred_df, x="score", y="micro_topic", orientation="h",
            color="score", color_continuous_scale="Viridis",
            title=f"Top {top_n} Predicted Topics for {target_year}",
        )
        fig.update_layout(height=max(400, top_n * 25), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig, use_container_width=True)

        for _, row in pred_df.iterrows():
            with st.expander(f"{row['micro_topic']} (score: {row['score']})"):
                st.write(f"**Topic:** {row['topic']}")
                st.write(f"**Total appearances:** {row['total_appearances']}")
                st.write(f"**Last appeared:** {row['last_appeared']}")
                st.write("**Reasons:**")
                for reason in row["reasons"]:
                    st.write(f"- {reason}")

with tab3:
    st.header("Question Explorer")
    search = st.text_input("Search questions", "")
    col1, col2 = st.columns(2)
    with col1:
        topic_filter = st.selectbox("Topic", ["All"] + sorted(filtered["topic"].unique().tolist()))
    with col2:
        diff_filter = st.selectbox("Difficulty", ["All", 1, 2, 3, 4, 5])

    explorer_df = filtered.copy()
    if search:
        explorer_df = explorer_df[
            explorer_df["question_text"].str.contains(search, case=False, na=False)
            | explorer_df["micro_topic"].str.contains(search, case=False, na=False)
        ]
    if topic_filter != "All":
        explorer_df = explorer_df[explorer_df["topic"] == topic_filter]
    if diff_filter != "All":
        explorer_df = explorer_df[explorer_df["difficulty"] == diff_filter]

    st.write(f"Showing {len(explorer_df)} questions")
    st.dataframe(
        explorer_df[["id", "exam", "year", "subject", "topic", "micro_topic", "difficulty", "question_type"]],
        use_container_width=True,
    )

    if not explorer_df.empty:
        st.subheader("Difficulty Distribution")
        fig = px.histogram(explorer_df, x="difficulty", nbins=5, color="subject")
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Questions Per Year")
        fig = px.histogram(explorer_df, x="year", color="subject")
        st.plotly_chart(fig, use_container_width=True)

with tab4:
    st.header("Smart Practice Sets")
    mode = st.selectbox("Mode", [
        "High Probability Topics",
        "Dormant Topics (Surprise Factor)",
        "Full Balanced Mock",
    ])
    num_questions = st.slider("Number of questions", 10, 90, 30)

    if st.button("Generate Practice Set"):
        if mode == "High Probability Topics":
            preds = predict_topics(DB_PATH, target_year=2026)
            top_micros = [p["micro_topic"] for p in preds[:20]]
            pool = filtered[filtered["micro_topic"].isin(top_micros)]
            practice = pool.sample(n=min(num_questions, len(pool)), random_state=42)
        elif mode == "Dormant Topics (Surprise Factor)":
            _, cold = find_hot_cold_topics(DB_PATH, recent_years=5)
            cold_micros = [c[1] for c in cold[:20]]
            pool = filtered[filtered["micro_topic"].isin(cold_micros)]
            practice = pool.sample(n=min(num_questions, len(pool)), random_state=42)
        else:
            practice = filtered.sample(n=min(num_questions, len(filtered)), random_state=42)

        st.dataframe(
            practice[["id", "exam", "year", "subject", "topic", "micro_topic", "difficulty", "question_text"]],
            use_container_width=True,
        )

        export_text = ""
        for i, (_, row) in enumerate(practice.iterrows(), 1):
            export_text += f"Q{i}. [{row['exam']} {row['year']}] [{row['micro_topic']}] (Difficulty: {row['difficulty']})\n"
            export_text += f"{row['question_text']}\n"
            export_text += f"Answer: {row['answer']}\n\n"

        st.download_button("Download Practice Set", export_text, "practice_set.txt", "text/plain")
```

**Step 2: Test manually**

Run: `cd /Users/aman/exam-predictor && source venv/bin/activate && streamlit run dashboard/app.py`
Expected: Opens browser, shows "Database not found" error (since no data loaded yet — that's correct).

**Step 3: Commit**

```bash
git add dashboard/app.py
git commit -m "feat: add Streamlit dashboard with heatmap, predictions, explorer, practice sets"
```

---

### Task 10: Sample Data & Run Script

**Files:**
- Create: `data/extracted/sample_jee_2020.json`
- Create: `tests/test_e2e.py`
- Create: `run.py`

**Step 1: Create sample data file**

Create `data/extracted/sample_jee_2020.json` with 6 sample questions across different topics/exams. (See design doc for the JSON schema.)

**Step 2: Write the run script**

```python
# run.py
"""Load data into SQLite and launch the dashboard."""
import subprocess
import sys


def main():
    from utils.db import init_db
    from utils.loader import load_all_extracted

    db_path = "data/exam.db"

    print("Initializing database...")
    init_db(db_path)

    print("Loading extracted questions...")
    total = load_all_extracted(db_path)

    if total == 0:
        print("\nNo questions found in data/extracted/")
        print("Add JSON files extracted via the prompt template, then run again.")
        return

    print(f"\nLoaded {total} questions into {db_path}")
    print("\nLaunching dashboard...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "dashboard/app.py"])


if __name__ == "__main__":
    main()
```

**Step 3: Write the e2e test**

```python
# tests/test_e2e.py
import os
from utils.db import init_db, get_questions_df
from utils.loader import load_all_extracted
from analysis.trend_analyzer import topic_frequency_by_year
from analysis.difficulty_classifier import classify_difficulty
from analysis.pattern_finder import subject_weightage_over_time
from analysis.predictor import predict_topics

TEST_DB = "data/test_e2e.db"


def setup_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def teardown_function():
    if os.path.exists(TEST_DB):
        os.remove(TEST_DB)


def test_full_pipeline():
    init_db(TEST_DB)
    total = load_all_extracted(TEST_DB, "data/extracted")
    assert total >= 6

    df = get_questions_df(TEST_DB)
    assert len(df) >= 6
    assert "JEE Advanced" in df["exam"].values
    assert "NEET" in df["exam"].values

    freq = topic_frequency_by_year(TEST_DB)
    assert not freq.empty

    classified = classify_difficulty(TEST_DB)
    assert "difficulty_label" in classified.columns

    weights = subject_weightage_over_time(TEST_DB)
    assert not weights.empty

    predictions = predict_topics(TEST_DB, target_year=2026)
    assert len(predictions) > 0
    assert all("score" in p for p in predictions)
    assert all("reasons" in p for p in predictions)
```

**Step 4: Run all tests**

Run: `cd /Users/aman/exam-predictor && python -m pytest tests/ -v`
Expected: All tests pass

**Step 5: Commit**

```bash
git add data/extracted/sample_jee_2020.json run.py tests/test_e2e.py
git commit -m "feat: add sample data, run script, and end-to-end test"
```

---

### Task 11: Local Hosting via ngrok

**Files:**
- Modify: `requirements.txt` (add `pyngrok`)
- Create: `serve.py`

**Step 1: Add pyngrok to requirements**

Add `pyngrok>=7.0.0` to `requirements.txt` and run `pip install pyngrok`.

**Step 2: Write serve.py for public hosting from your laptop**

```python
# serve.py
"""Host the dashboard publicly from your laptop using ngrok."""
import subprocess
import sys
import threading
import time


def start_streamlit():
    subprocess.run([
        sys.executable, "-m", "streamlit", "run", "dashboard/app.py",
        "--server.port", "8501",
        "--server.headless", "true",
    ])


def start_ngrok():
    from pyngrok import ngrok
    time.sleep(3)  # wait for streamlit to start
    public_url = ngrok.connect(8501)
    print(f"\n{'=' * 50}")
    print(f"Your dashboard is live at: {public_url}")
    print(f"Share this URL with anyone!")
    print(f"{'=' * 50}\n")
    print("Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        ngrok.disconnect(public_url)


def main():
    from utils.db import init_db
    from utils.loader import load_all_extracted

    db_path = "data/exam.db"
    init_db(db_path)
    total = load_all_extracted(db_path)

    if total == 0:
        print("No questions found in data/extracted/. Add JSON files first.")
        return

    print(f"Loaded {total} questions. Starting server...")

    st_thread = threading.Thread(target=start_streamlit, daemon=True)
    st_thread.start()
    start_ngrok()


if __name__ == "__main__":
    main()
```

**Step 3: Commit**

```bash
git add serve.py requirements.txt
git commit -m "feat: add ngrok-based public hosting from local machine"
```

---

### Task 12: Final Integration Test

**Step 1: Run full pipeline locally**

```bash
cd /Users/aman/exam-predictor
source venv/bin/activate
python run.py
```

Expected: Loads sample data, launches dashboard at localhost:8501.

**Step 2: Verify all 4 tabs work in browser**

- Topic Heatmap shows color grid
- Predictions shows ranked bar chart
- Question Explorer shows searchable table
- Practice Sets generates downloadable set

**Step 3: Test public hosting**

```bash
python serve.py
```

Expected: Prints a public ngrok URL. Opening it in any browser shows the dashboard.

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat: exam predictor v1 complete"
```
