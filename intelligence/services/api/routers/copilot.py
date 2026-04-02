"""
Copilot Router — Enhanced Natural Language Q&A
Combines PRAJNA prediction engine with qbg.db question bank (1.14M questions)
"""

from __future__ import annotations
import re
import os
import uuid
import base64
import sqlite3
import httpx
import logging
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = "llama3.2:3b"
OLLAMA_VISION_MODEL = "qwen2.5vl:3b"
OLLAMA_TIMEOUT = 90  # seconds
from fastapi import APIRouter, Depends, UploadFile, File, Form
from packages.schemas.contracts import CopilotRequest, CopilotResponse
from services.api.deps import (
    aggregator_dep, insight_generator_dep, prediction_adapter_dep,
)
from services.topic_intelligence.aggregator import TopicIntelligenceAggregator
from services.insight_engine.generator import InsightGenerator
from services.prediction_adapter.client import PredictionAdapter

router = APIRouter()

QBG_DB = str(Path(__file__).resolve().parent.parent.parent.parent.parent / "data" / "qbg.db")

# Keywords that indicate a concept/explanation question vs strategy question
CONCEPT_KEYWORDS = [
    "explain", "what is", "what are", "define", "describe", "how does",
    "how do", "why does", "why do", "what happens", "what occur",
    "difference between", "derive", "prove", "formula", "equation",
    "law", "principle", "theorem", "mechanism", "process", "function",
    "structure", "diagram", "example", "solve", "calculate", "find the",
    "evaluate", "meaning of", "concept of", "types of", "steps of",
    "phases of", "stages of", "properties of", "characteristics of",
    "reaction", "cycle", "cell division", "meiosis", "mitosis",
    "photosynthesis", "respiration", "digestion", "excretion",
]

STRATEGY_KEYWORDS = [
    "prioritize", "priority", "focus", "strategy", "study plan",
    "revision", "predict", "important", "likely", "upcoming",
    "probability", "trending", "rising", "which topic", "which chapter",
    "how many hours", "weak", "strong", "roi", "exam brief",
]

# Topic names for topic-aware search (must match labels in qbg.db)
KNOWN_TOPICS = {
    "kinematics": "Kinematics",
    "laws of motion": "Laws of Motion",
    "newton": "Laws of Motion",
    "work energy": "Work Energy Power",
    "work energy power": "Work Energy Power",
    "gravitation": "Gravitation",
    "gravity": "Gravitation",
    "thermodynamics": "Thermodynamics",
    "oscillations": "Oscillations",
    "shm": "Oscillations",
    "simple harmonic": "Oscillations",
    "waves": "Waves",
    "sound": "Waves",
    "electrostatics": "Electrostatics",
    "coulomb": "Electrostatics",
    "capacitor": "Electrostatics",
    "current electricity": "Current Electricity",
    "ohm": "Current Electricity",
    "kirchhoff": "Current Electricity",
    "magnetism": "Magnetism",
    "magnetic": "Magnetism",
    "electromagnetic induction": "Electromagnetic Induction",
    "faraday": "Electromagnetic Induction",
    "alternating current": "Alternating Current",
    "optics": "Optics",
    "lens": "Optics",
    "mirror": "Optics",
    "refraction": "Optics",
    "modern physics": "Modern Physics",
    "photoelectric": "Modern Physics",
    "bohr": "Modern Physics",
    "nuclear": "Modern Physics",
    "semiconductor": "Semiconductor",
    "diode": "Semiconductor",
    "transistor": "Semiconductor",
    "rotational motion": "Rotational Motion",
    "moment of inertia": "Rotational Motion",
    "torque": "Rotational Motion",
    "fluid mechanics": "Fluid Mechanics",
    "bernoulli": "Fluid Mechanics",
    "viscosity": "Fluid Mechanics",
    "elasticity": "Elasticity",
    "stress strain": "Elasticity",
    "kinetic theory": "Kinetic Theory",
    "ideal gas": "Kinetic Theory",
    "organic chemistry": "Organic Chemistry",
    "alkane": "Organic Chemistry",
    "alkene": "Organic Chemistry",
    "benzene": "Organic Chemistry",
    "inorganic chemistry": "Inorganic Chemistry",
    "periodic table": "Inorganic Chemistry",
    "coordination compound": "Inorganic Chemistry",
    "d-block": "Inorganic Chemistry",
    "p-block": "Inorganic Chemistry",
    "physical chemistry": "Physical Chemistry",
    "electrochemistry": "Physical Chemistry",
    "chemical kinetics": "Physical Chemistry",
    "equilibrium": "Physical Chemistry",
    "solutions": "Solutions",
    "molarity": "Solutions",
    "colligative": "Solutions",
    "chemical bonding": "Chemical Bonding",
    "hybridization": "Chemical Bonding",
    "vsepr": "Chemical Bonding",
    "atomic structure": "Atomic Structure",
    "quantum number": "Atomic Structure",
    "electron configuration": "Atomic Structure",
    "redox": "Redox Reactions",
    "oxidation reduction": "Redox Reactions",
    "genetics": "Genetics",
    "dna": "Genetics",
    "rna": "Genetics",
    "heredity": "Genetics",
    "mendel": "Genetics",
    "cell biology": "Cell Biology",
    "mitosis": "Cell Biology",
    "meiosis": "Cell Biology",
    "cell cycle": "Cell Biology",
    "ecology": "Ecology",
    "ecosystem": "Ecology",
    "biodiversity": "Ecology",
    "food chain": "Ecology",
    "human physiology": "Human Physiology",
    "digestion": "Human Physiology",
    "respiration": "Human Physiology",
    "circulation": "Human Physiology",
    "excretion": "Human Physiology",
    "nervous system": "Human Physiology",
    "endocrine": "Human Physiology",
    "plant physiology": "Plant Physiology",
    "photosynthesis": "Plant Physiology",
    "transpiration": "Plant Physiology",
    "reproduction": "Reproduction",
    "fertilization": "Reproduction",
    "gametogenesis": "Reproduction",
    "evolution": "Evolution",
    "natural selection": "Evolution",
    "darwin": "Evolution",
    "molecular biology": "Molecular Biology",
    "transcription": "Molecular Biology",
    "translation": "Molecular Biology",
    "replication": "Molecular Biology",
    "biotechnology": "Biotechnology",
    "genetic engineering": "Biotechnology",
    "algebra": "Algebra",
    "quadratic": "Algebra",
    "polynomial": "Algebra",
    "permutation": "Algebra",
    "combination": "Algebra",
    "calculus": "Calculus",
    "differentiation": "Calculus",
    "integration": "Calculus",
    "derivative": "Calculus",
    "limit": "Calculus",
    "coordinate geometry": "Coordinate Geometry",
    "parabola": "Coordinate Geometry",
    "ellipse": "Coordinate Geometry",
    "hyperbola": "Coordinate Geometry",
    "conic": "Coordinate Geometry",
    "trigonometry": "Trigonometry",
    "vectors": "Vectors",
    "vector": "Vectors",
    "probability": "Probability",
    "bayes": "Probability",
    "matrices": "Matrices",
    "matrix": "Matrices",
    "determinant": "Matrices",
    "statistics": "Statistics",
    "3d geometry": "3D Geometry",
    "direction cosine": "3D Geometry",
}

# Regex to detect useless solutions like "(1)", "(a)", "(2)", "(d)", "option b", etc.
_BAD_SOLUTION_RE = re.compile(
    r'^\s*(\([1-4a-d]\)|\(?option\s*[1-4a-d]\)?|[1-4a-d]\.?\s*)$',
    re.IGNORECASE,
)

# Phrases that indicate GPT "thinking" rather than actual answer content
THINKING_PREFIXES = [
    'we need to', 'we should', 'let me', 'i need to', 'the user',
    'likely', 'probably', 'so the answer', 'the sentence',
    'they want', 'the question', 'this is a', 'user asks',
    'user says', 'user writes', 'we have', 'we can', 'let\'s',
    'i\'ll', 'i will', 'maybe', 'perhaps', 'it seems', 'it looks',
    'asks:', 'asks \"', 'says:', 'writes:', 'straightforward',
    'provide', 'respond', 'answer:', 'succinctly', 'briefly',
    'here we', 'here is', 'so we', 'so,', 'thus,', 'hence,',
    'note:', 'ok,', 'ok.', 'alright', 'well,', 'now,',
]


def _clean_gpt_analysis(raw: str) -> str:
    """Clean GPT analysis text: remove channel markers, thinking prefix, format for display."""
    if not raw:
        return ""

    # Remove channel markers
    text = re.sub(r'<\|channel\|>\w+<\|message\|>', '\n', raw)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove leading "analysis" / "assistant" / "final" words
    text = re.sub(r'^(analysis|assistant|final|user|system)\s*', '', text.strip(), flags=re.IGNORECASE)

    # Split into sentences
    sentences = re.split(r'(?<=[.!?\n])\s+', text.strip())
    if not sentences:
        return text.strip()

    # Skip thinking/reasoning sentences, keep answer sentences
    clean_sentences = []
    found_answer = False
    for s in sentences:
        s = s.strip()
        if not s:
            continue
        s_lower = s.lower()

        # Skip meta-thinking sentences
        if not found_answer and any(s_lower.startswith(p) for p in THINKING_PREFIXES):
            continue

        # Found first non-thinking sentence — this is the start of the real answer
        found_answer = True
        # Also skip single-word "sentences" that are leftovers
        if len(s) < 10:
            continue
        clean_sentences.append(s)

    result = ' '.join(clean_sentences) if clean_sentences else text.strip()

    # Clean up repeated whitespace and newlines
    result = re.sub(r'\n{3,}', '\n\n', result)
    result = re.sub(r'  +', ' ', result)

    return result.strip()


def _find_topic(words: list[str]) -> str | None:
    """Check if any search word(s) match a known topic name.

    Tries multi-word combinations first (e.g. "organic chemistry"),
    then single words (e.g. "kinematics").
    """
    query_lower = " ".join(words).lower()

    # Try full query match first
    if query_lower in KNOWN_TOPICS:
        return KNOWN_TOPICS[query_lower]

    # Try 2-word and 3-word combos
    for n in (3, 2):
        for i in range(len(words) - n + 1):
            combo = " ".join(words[i:i+n]).lower()
            if combo in KNOWN_TOPICS:
                return KNOWN_TOPICS[combo]

    # Try single words
    for w in words:
        wl = w.lower()
        if wl in KNOWN_TOPICS:
            return KNOWN_TOPICS[wl]

    return None


def _has_good_solution(row: dict) -> bool:
    """Check if a question has a real, useful solution (not just option labels)."""
    sol = row.get("text_solution") or ""
    if not sol or len(sol.strip()) < 10:
        return False
    # Filter out solutions that are just option labels like "(2)", "(d)", etc.
    clean = re.sub(r'<[^>]+>', '', sol).strip()
    if _BAD_SOLUTION_RE.match(clean):
        return False
    return True


def _is_concept_question(question: str) -> bool:
    q = question.lower().strip()
    concept_score = sum(1 for kw in CONCEPT_KEYWORDS if kw in q)
    strategy_score = sum(1 for kw in STRATEGY_KEYWORDS if kw in q)
    # Default to CONCEPT when both scores are 0 — most questions ask to learn
    # Only route to strategy when strategy keywords are explicitly present
    return concept_score >= strategy_score


def _search_qbg(query: str, subject: str | None = None, top_n: int = 5) -> list[dict]:
    """Search qbg.db for relevant questions with solutions.

    Strategy:
    0. Topic-aware search — if query matches a known topic, search by topic column first
    1. LIKE search (finds questions containing the topic keywords)
    2. Single-word LIKE fallback
    3. FTS5 fallback for broader semantic matching

    Results are ranked by solution quality: prefer questions with real explanations.
    """
    try:
        if not Path(QBG_DB).exists():
            return []
        conn = sqlite3.connect(QBG_DB)
        conn.row_factory = sqlite3.Row

        # Extract key topic words (remove stopwords)
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                      'what', 'how', 'why', 'when', 'where', 'which', 'who',
                      'do', 'does', 'did', 'can', 'could', 'should', 'would',
                      'explain', 'describe', 'define', 'give', 'me', 'with',
                      'and', 'or', 'of', 'in', 'on', 'for', 'to', 'from', 'by',
                      'it', 'its', 'this', 'that', 'these', 'those', 'some', 'any',
                      'examples', 'example', 'about', 'between', 'using', 'find'}
        clean_query = re.sub(r'[^\w\s]', ' ', query).strip().lower()
        words = [w for w in clean_query.split() if w not in stopwords and len(w) > 2][:6]
        if not words:
            words = clean_query.split()[:4]

        rows = []

        # Strategy 0: Topic-aware search — use topic column if query matches a topic
        topic_match = _find_topic(words)
        if topic_match:
            topic_sql = """
                SELECT qbgid, subject, difficulty, question_clean,
                       answer_clean, text_solution, gpt_analysis
                FROM questions
                WHERE topic = ?
                AND text_solution IS NOT NULL
                AND LENGTH(text_solution) > 50
            """
            topic_params = [topic_match]
            if subject:
                topic_sql += " AND subject = ?"
                topic_params.append(subject)
            # Rank by solution quality: longer solutions first, then GPT analysis presence
            topic_sql += """
                ORDER BY
                    (CASE WHEN gpt_analysis IS NOT NULL AND LENGTH(gpt_analysis) > 50 THEN 0 ELSE 1 END),
                    LENGTH(text_solution) DESC
                LIMIT ?
            """
            topic_params.append(top_n * 3)  # fetch more, then filter
            topic_rows = conn.execute(topic_sql, topic_params).fetchall()

            # Filter out bad solutions and take top_n
            for row in topic_rows:
                rd = {k: row[k] for k in row.keys()}
                if _has_good_solution(rd):
                    rows.append(row)
                if len(rows) >= top_n:
                    break

        # Pick the most specific/rare words for LIKE search
        generic_words = {'happens', 'during', 'process', 'division', 'reaction',
                         'method', 'effect', 'change', 'type', 'form', 'nature',
                         'following', 'given', 'correct', 'statement', 'true', 'false'}
        specific_words = sorted(words, key=lambda w: (w.lower() not in generic_words, len(w)), reverse=True)

        # Strategy 1: LIKE search with top 2-3 most specific words
        if len(rows) < top_n:
            use_words = specific_words[:min(3, len(specific_words))]
            like_conditions = " AND ".join(f"question_clean LIKE ?" for w in use_words)
            like_params = [f"%{w}%" for w in use_words]

            like_sql = f"""
                SELECT qbgid, subject, difficulty, question_clean,
                       answer_clean, text_solution, gpt_analysis
                FROM questions
                WHERE {like_conditions}
                AND text_solution IS NOT NULL
                AND LENGTH(text_solution) > 20
            """
            if subject:
                like_sql += " AND subject = ?"
                like_params.append(subject)
            like_sql += """
                ORDER BY
                    (CASE WHEN gpt_analysis IS NOT NULL AND gpt_analysis != '' THEN 0 ELSE 1 END),
                    LENGTH(text_solution) DESC
            """
            like_sql += f" LIMIT {top_n * 3}"

            like_rows = conn.execute(like_sql, like_params).fetchall()
            # Deduplicate by qbgid and filter bad solutions
            seen_ids = {row["qbgid"] for row in rows} if rows else set()
            for row in like_rows:
                if row["qbgid"] in seen_ids:
                    continue
                rd = {k: row[k] for k in row.keys()}
                if _has_good_solution(rd):
                    rows.append(row)
                    seen_ids.add(row["qbgid"])
                if len(rows) >= top_n:
                    break

        # Strategy 2: If LIKE returns too few, try with just the most specific word
        if len(rows) < 3 and specific_words:
            like_params2 = [f"%{specific_words[0]}%"]
            like_sql2 = """
                SELECT qbgid, subject, difficulty, question_clean,
                       answer_clean, text_solution, gpt_analysis
                FROM questions
                WHERE question_clean LIKE ?
                AND text_solution IS NOT NULL
                AND LENGTH(text_solution) > 20
            """
            if subject:
                like_sql2 += " AND subject = ?"
                like_params2.append(subject)
            like_sql2 += """
                ORDER BY
                    (CASE WHEN gpt_analysis IS NOT NULL AND gpt_analysis != '' THEN 0 ELSE 1 END),
                    LENGTH(text_solution) DESC
            """
            like_sql2 += f" LIMIT {top_n * 3}"
            like_rows2 = conn.execute(like_sql2, like_params2).fetchall()
            seen_ids = {row["qbgid"] for row in rows} if rows else set()
            for row in like_rows2:
                if row["qbgid"] in seen_ids:
                    continue
                rd = {k: row[k] for k in row.keys()}
                if _has_good_solution(rd):
                    rows.append(row)
                    seen_ids.add(row["qbgid"])
                if len(rows) >= top_n:
                    break

        # Strategy 3: FTS5 fallback
        if len(rows) < 2:
            fts_query = ' OR '.join(words)
            fts_sql = """
                SELECT q.qbgid, q.subject, q.difficulty, q.question_clean,
                       q.answer_clean, q.text_solution, q.gpt_analysis
                FROM questions_fts fts
                JOIN questions q ON q.rowid = fts.rowid
                WHERE fts.questions_fts MATCH ?
            """
            fts_params = [fts_query]
            if subject:
                fts_sql += " AND q.subject = ?"
                fts_params.append(subject)
            fts_sql += f" ORDER BY rank LIMIT {top_n * 3}"
            fts_rows = conn.execute(fts_sql, fts_params).fetchall()
            seen_ids = {row["qbgid"] for row in rows} if rows else set()
            for row in fts_rows:
                if row["qbgid"] in seen_ids:
                    continue
                rows.append(row)
                seen_ids.add(row["qbgid"])
                if len(rows) >= top_n:
                    break

        conn.close()
        return [{k: row[k] for k in row.keys()} for row in rows]
    except Exception:
        return []


def _is_broad_query(question: str) -> bool:
    """Detect if the query is too broad/vague for a specific question match."""
    q = question.lower().strip()
    # Broad queries: short, generic topic names, no specific numbers/formulas
    broad_patterns = [
        r'^(explain|describe|what is|what are|tell me about)\s+\w+\s+\w+$',  # "explain X Y"
        r'reaction mechanism',
        r'types of',
        r'laws of',
        r'properties of',
        r'difference between .+ and',
        r'classification of',
    ]
    # Short queries are only broad if they don't contain specific terms
    specific_terms = {'formula', 'law', 'equation', 'theorem', 'principle',
                      'difference', 'define', 'solve', 'calculate', 'derive'}
    if len(q.split()) <= 4 and not any(c.isdigit() for c in q) and not any(t in q for t in specific_terms):
        return True
    for p in broad_patterns:
        if re.search(p, q):
            return True
    return False


async def _build_topic_overview(question: str, qbg_results: list[dict]) -> str:
    """For broad queries, show a topic overview with categorized practice questions."""
    parts = []
    parts.append(f"📚 **{question}**\n")

    # Group practice questions by difficulty
    easy = [q for q in qbg_results if q.get("difficulty") == "easy"]
    medium = [q for q in qbg_results if q.get("difficulty") == "medium"]
    hard = [q for q in qbg_results if q.get("difficulty") == "hard"]

    parts.append("This is a broad topic. Here are practice questions at different difficulty levels:\n")

    for label, qs in [("🟢 Easy", easy), ("🟡 Medium", medium), ("🔴 Hard", hard)]:
        if not qs:
            continue
        parts.append(f"\n**{label}:**")
        for q in qs[:2]:
            q_text = re.sub(r'<[^>]+>', '', q.get("question_clean", ""))
            if len(q_text) > 180:
                q_text = q_text[:180] + "..."
            answer = q.get("answer_clean", "—")
            if len(answer) > 100:
                answer = answer[:100] + "..."
            parts.append(f"• {q_text}")
            parts.append(f"  → **Answer:** {answer}\n")

    # If any have GPT analysis, show the best explanation
    best_gpt = None
    for q in qbg_results:
        gpt = q.get("gpt_analysis", "")
        if gpt and len(gpt) > 100:
            best_gpt = gpt
            break

    if best_gpt:
        clean = _clean_gpt_analysis(best_gpt)
        if clean and len(clean) > 50:
            if len(clean) > 600:
                clean = clean[:600] + "..."
            parts.append(f"\n**Concept note:**\n{clean}")

    parts.append("\n💡 **Tip:** Ask a more specific question for detailed explanations, e.g.:")
    parts.append("• \"What is SN1 reaction mechanism?\"")
    parts.append("• \"Explain electrophilic addition with example\"")
    parts.append("• \"Difference between SN1 and SN2\"")

    return "\n".join(parts)


def _clean_html(text: str) -> str:
    """Remove HTML/MathML tags, keep LaTeX."""
    text = re.sub(r'<math[^>]*>.*?</math>', '', text, flags=re.DOTALL)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'&nbsp;', ' ', text)
    text = re.sub(r'&[a-z]+;', '', text)
    return re.sub(r'\s+', ' ', text).strip()


async def _llm_format(question: str, context_items: list[dict]) -> str | None:
    """Use local Llama 3.2 (via Ollama) to structure a clean answer from qbg.db results.

    Returns None if Ollama is offline — caller should fall back to rule-based formatting.
    """
    # Build context from top 3 results
    context_parts = []
    for q in context_items[:3]:
        q_text = _clean_html(q.get("question_clean", ""))
        answer = _clean_html(q.get("answer_clean", ""))
        solution = _clean_html(q.get("text_solution", ""))
        if q_text:
            context_parts.append(f"Q: {q_text[:300]}")
        if answer:
            context_parts.append(f"A: {answer[:200]}")
        if solution and len(solution) > 20:
            context_parts.append(f"Solution: {solution[:400]}")
        context_parts.append("")

    context = "\n".join(context_parts)

    prompt = f"""You are PRAJNA, an AI tutor for NEET/JEE exam preparation.

A student asked: "{question}"

Here are relevant questions and answers from the question bank:

{context}

Based on the above, give a clear, structured answer to the student's question.
Rules:
- Be direct and educational
- Use LaTeX for formulas: \\(V = IR\\) for inline, \\[E = mc^2\\] for display
- Keep it concise (3-5 sentences for simple questions, up to 8 for complex ones)
- Include the key formula or concept first, then explain
- End with one practical exam tip if relevant
- Do NOT say "based on the question bank" or reference the source data"""

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            resp = await client.post(OLLAMA_URL, json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "num_predict": 1200,
                }
            })
            if resp.status_code == 200:
                data = resp.json()
                answer = data.get("response", "").strip()
                if answer and len(answer) > 20:
                    return answer
    except Exception as e:
        logger.warning(f"Ollama unavailable: {e}")

    return None


async def _build_concept_answer(question: str, qbg_results: list[dict]) -> str:
    """Build a clean answer using question bank data.

    Tries Llama 3.2 (via Ollama) first for structured answers.
    Falls back to rule-based formatting if Ollama is offline.
    Uses text_solution (human-written) and answer_clean directly.
    No GPT analysis — it's raw thinking text that reads poorly.
    """
    if not qbg_results:
        return f"I couldn't find questions matching \"{question}\" in the question bank. Try more specific terms."

    if _is_broad_query(question):
        return await _build_topic_overview(question, qbg_results)

    # Try Llama 3.2 for structured answer
    llm_answer = await _llm_format(question, qbg_results)
    if llm_answer:
        # Add practice questions below the LLM answer
        others = qbg_results[1:4]
        practice = ""
        if others:
            practice = "\n\n📝 **Practice:**\n"
            for i, q in enumerate(others, 1):
                q_text = _clean_html(q.get("question_clean", ""))
                if len(q_text) > 150:
                    q_text = q_text[:150] + "..."
                ans = _clean_html(q.get("answer_clean", "—"))
                if len(ans) > 80:
                    ans = ans[:80] + "..."
                diff = (q.get("difficulty") or "unknown").upper()
                practice += f"\n{i}. **[{diff}]** {q_text}\n   → {ans}\n"
        return f"🧠 **{question}**\n\n{llm_answer}{practice}"

    # Fallback: rule-based formatting (Ollama offline)
    with_solution = [q for q in qbg_results if q.get("text_solution") and len(str(q["text_solution"])) > 30]
    best = with_solution[0] if with_solution else qbg_results[0]

    parts = []
    parts.append(f"📚 **{question}**\n")

    # Show the answer first — clearly
    answer = best.get("answer_clean", "")
    if answer:
        clean_a = _clean_html(answer)
        if len(clean_a) > 300:
            clean_a = clean_a[:300] + "..."
        parts.append(f"**Answer:** {clean_a}\n")

    # Show human-written solution
    solution = str(best.get("text_solution") or "")
    if solution and len(solution) > 30:
        clean_sol = _clean_html(solution)
        if len(clean_sol) > 600:
            clean_sol = clean_sol[:600] + "..."
        parts.append(f"**Explanation:**\n{clean_sol}\n")

    # Source question context
    best_q = _clean_html(best.get("question_clean", ""))
    if best_q:
        if len(best_q) > 200:
            best_q = best_q[:200] + "..."
        parts.append(f"*Based on:* {best_q}\n")

    # Practice questions
    others = [q for q in qbg_results if q.get("qbgid") != best.get("qbgid")]
    if others:
        parts.append(f"\n📝 **Practice ({len(others)} more):**\n")
        for i, q in enumerate(others[:3], 1):
            q_text = _clean_html(q.get("question_clean", ""))
            if len(q_text) > 150:
                q_text = q_text[:150] + "..."
            ans = _clean_html(q.get("answer_clean", "—"))
            if len(ans) > 80:
                ans = ans[:80] + "..."
            diff = (q.get("difficulty") or "unknown").upper()
            parts.append(f"{i}. **[{diff}]** {q_text}")
            parts.append(f"   → {ans}\n")

    return "\n".join(parts)


@router.post("/ask", response_model=CopilotResponse,
             summary="Ask a natural language question about upcoming exam topics")
async def ask_copilot(
    request:    CopilotRequest,
    adapter:    PredictionAdapter            = Depends(prediction_adapter_dep),
    aggregator: TopicIntelligenceAggregator  = Depends(aggregator_dep),
    generator:  InsightGenerator             = Depends(insight_generator_dep),
):
    """
    Enhanced Q&A combining PRAJNA predictions with 1.14M question bank.

    - Concept questions → searches qbg.db, returns explanations + practice
    - Strategy questions → uses prediction engine for exam-focused advice
    """
    req_id = str(uuid.uuid4())
    is_concept = _is_concept_question(request.question)

    if is_concept:
        # Search qbg.db for related content
        qbg_results = _search_qbg(
            request.question,
            subject=request.subject_filter,
            top_n=5,
        )
        answer = await _build_concept_answer(request.question, qbg_results)

        # Build follow-ups based on the content
        subject_hint = qbg_results[0]["subject"] if qbg_results else "Physics"
        follow_ups = [
            f"Give me 5 more practice problems on this topic",
            f"What is the exam probability for {subject_hint} topics?",
            f"Explain the solution step by step",
        ]

        return CopilotResponse(
            success=True,
            request_id=req_id,
            question=request.question,
            answer=answer,
            confidence=0.75 if qbg_results else 0.2,
            insights=[],
            sources=[{"source": "qbg-pcmr", "type": "question_bank"}] if qbg_results else [],
            follow_up_questions=follow_ups,
        )

    # Strategy question → use prediction engine (original behavior)
    micro_preds = await adapter.get_predictions(
        exam_type=request.exam_type,
        target_year=request.target_year,
        subject=request.subject_filter,
    )
    batch = aggregator.build_batch(micro_preds)
    insight = await generator.answer_copilot_question(request, batch)

    top_names = [t.micro_topic_name for t in micro_preds[:3]] if micro_preds else []
    exam_label = str(request.exam_type).split(".")[-1].upper()
    follow_ups = [
        f"Which {top_names[0]} sub-topics are most likely for {exam_label} {request.target_year}?"
        if top_names else f"What are the top chapters for {exam_label} {request.target_year}?",
        f"How many hours should I allocate to {top_names[1] if len(top_names) > 1 else 'each subject'}?",
        "Which weak chapters overlap with high-probability predictions?",
    ]

    return CopilotResponse(
        success=True, request_id=req_id,
        question=request.question,
        answer=insight.narrative,
        confidence=insight.confidence,
        insights=[insight],
        sources=[{"source": e.source_name, "type": str(e.evidence_type)} for e in insight.evidence],
        follow_up_questions=follow_ups,
    )


# ── OCR Endpoint — Image/PDF → Extract text → Solve ──────────────────────────

async def _ocr_with_qwen(image_base64: str, prompt: str = "Extract all text, questions, and mathematical formulas from this image. Preserve LaTeX notation for any math expressions.") -> str:
    """Use Qwen2.5-VL via Ollama to extract text from an image."""
    try:
        async with httpx.AsyncClient(timeout=180) as client:  # 3 min for vision model
            resp = await client.post(OLLAMA_CHAT_URL, json={
                "model": OLLAMA_VISION_MODEL,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                        "images": [image_base64],
                    }
                ],
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 800,
                },
            })
            if resp.status_code == 200:
                data = resp.json()
                msg = data.get("message", {})
                return msg.get("content", "").strip()
    except Exception as e:
        logger.error(f"Qwen OCR failed: {e}")
    return ""


def _pdf_to_images(pdf_path: str) -> list[str]:
    """Convert PDF pages to base64 images.

    Uses pdf2image if available, falls back to PyMuPDF.
    Returns list of base64-encoded PNG images.
    """
    images_b64 = []
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(pdf_path)
        for page_num in range(min(len(doc), 5)):  # max 5 pages
            page = doc[page_num]
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for clarity
            img_bytes = pix.tobytes("png")
            images_b64.append(base64.b64encode(img_bytes).decode())
        doc.close()
    except ImportError:
        logger.warning("PyMuPDF not installed. Install with: pip install PyMuPDF")
        # Fallback: try pdf2image
        try:
            from pdf2image import convert_from_path
            import io
            pages = convert_from_path(pdf_path, first_page=1, last_page=5, dpi=200)
            for page in pages:
                buf = io.BytesIO()
                page.save(buf, format="PNG")
                images_b64.append(base64.b64encode(buf.getvalue()).decode())
        except ImportError:
            logger.error("Neither PyMuPDF nor pdf2image installed for PDF processing")
    return images_b64


@router.post("/ocr-solve",
             summary="Upload image/PDF → OCR extract text → solve with PRAJNA SLM")
async def ocr_and_solve(
    file: UploadFile = File(...),
    exam_type: str = Form("neet"),
    subject_filter: str = Form(None),
):
    """
    Upload a photograph or PDF of a question:
    1. Qwen2.5-VL extracts the text (OCR)
    2. Extracted text is searched in qbg.db (1.14M questions)
    3. PRAJNA SLM structures a clean answer

    Supports: .jpg, .jpeg, .png, .pdf
    """
    req_id = str(uuid.uuid4())

    # Validate file type
    filename = file.filename or "upload"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in ("jpg", "jpeg", "png", "pdf", "webp"):
        return {
            "success": False,
            "request_id": req_id,
            "error": f"Unsupported file type: .{ext}. Use .jpg, .png, or .pdf",
        }

    # Read file content
    content = await file.read()
    if len(content) > 20 * 1024 * 1024:  # 20MB limit
        return {
            "success": False,
            "request_id": req_id,
            "error": "File too large. Maximum 20MB.",
        }

    # Convert to base64 images
    images_b64 = []
    if ext == "pdf":
        # Save to temp file for PDF processing
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name
        try:
            images_b64 = _pdf_to_images(tmp_path)
        finally:
            os.unlink(tmp_path)
    else:
        # Direct image
        images_b64 = [base64.b64encode(content).decode()]

    if not images_b64:
        return {
            "success": False,
            "request_id": req_id,
            "error": "Could not process the file. Try a different image or PDF.",
        }

    # OCR each page/image with Qwen
    extracted_texts = []
    for i, img_b64 in enumerate(images_b64):
        text = await _ocr_with_qwen(img_b64)
        if text:
            extracted_texts.append(text)

    if not extracted_texts:
        return {
            "success": False,
            "request_id": req_id,
            "extracted_text": "",
            "error": "OCR could not extract text from the image. Ensure the image is clear and well-lit.",
        }

    full_text = "\n\n".join(extracted_texts)

    # Search qbg.db for matching questions
    qbg_results = _search_qbg(full_text, subject=subject_filter, top_n=5)

    # Build answer using existing pipeline
    answer = await _build_concept_answer(full_text, qbg_results)

    # Build follow-ups
    follow_ups = [
        "Explain this solution step by step",
        "Give me 5 similar practice problems",
        "What topic is this from and how important is it for the exam?",
    ]

    return {
        "success": True,
        "request_id": req_id,
        "extracted_text": full_text,
        "answer": answer,
        "confidence": 0.75 if qbg_results else 0.3,
        "pages_processed": len(images_b64),
        "sources": [{"source": "qbg-pcmr + Qwen2.5-VL OCR", "type": "vision_ocr"}] if qbg_results else [],
        "follow_up_questions": follow_ups,
    }
