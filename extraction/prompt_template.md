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
