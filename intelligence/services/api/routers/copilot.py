"""
Copilot Router — Enhanced Natural Language Q&A
Combines PRAJNA prediction engine with qbg.db question bank (1.14M questions)
"""

from __future__ import annotations
import re
import uuid
import sqlite3
from pathlib import Path
from fastapi import APIRouter, Depends
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


def _is_concept_question(question: str) -> bool:
    q = question.lower().strip()
    concept_score = sum(1 for kw in CONCEPT_KEYWORDS if kw in q)
    strategy_score = sum(1 for kw in STRATEGY_KEYWORDS if kw in q)
    return concept_score > strategy_score


def _search_qbg(query: str, subject: str | None = None, top_n: int = 5) -> list[dict]:
    """Search qbg.db for relevant questions with solutions.

    Strategy: LIKE search first (finds questions containing the topic keywords),
    then FTS5 fallback for broader semantic matching.
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

        # Pick the most specific/rare words for LIKE search
        # Generic words that appear in many subjects — deprioritize
        generic_words = {'happens', 'during', 'process', 'division', 'reaction',
                         'method', 'effect', 'change', 'type', 'form', 'nature',
                         'following', 'given', 'correct', 'statement', 'true', 'false'}
        # Score: domain-specific words first, then by length
        specific_words = sorted(words, key=lambda w: (w.lower() not in generic_words, len(w)), reverse=True)

        # Strategy 1: LIKE search with top 2-3 most specific words
        use_words = specific_words[:min(3, len(specific_words))]
        like_conditions = " AND ".join(f"question_clean LIKE ?" for w in use_words)
        like_params = [f"%{w}%" for w in use_words]

        like_sql = f"""
            SELECT qbgid, subject, difficulty, question_clean,
                   answer_clean, text_solution, gpt_analysis
            FROM questions
            WHERE {like_conditions}
        """
        if subject:
            like_sql += " AND subject = ?"
            like_params.append(subject)
        # Prefer questions with GPT analysis
        like_sql += " ORDER BY (CASE WHEN gpt_analysis IS NOT NULL AND gpt_analysis != '' THEN 0 ELSE 1 END), RANDOM()"
        like_sql += f" LIMIT {top_n}"

        rows = conn.execute(like_sql, like_params).fetchall()

        # Strategy 2: If LIKE returns too few, try with just the most specific word
        if len(rows) < 3 and specific_words:
            like_conditions2 = "question_clean LIKE ?"
            like_params2 = [f"%{specific_words[0]}%"]
            like_sql2 = f"""
                SELECT qbgid, subject, difficulty, question_clean,
                       answer_clean, text_solution, gpt_analysis
                FROM questions
                WHERE {like_conditions2}
            """
            if subject:
                like_sql2 += " AND subject = ?"
                like_params2.append(subject)
            like_sql2 += " ORDER BY (CASE WHEN gpt_analysis IS NOT NULL AND gpt_analysis != '' THEN 0 ELSE 1 END), RANDOM()"
            like_sql2 += f" LIMIT {top_n}"
            rows = conn.execute(like_sql2, like_params2).fetchall()

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
            fts_sql += f" ORDER BY rank LIMIT {top_n}"
            rows = conn.execute(fts_sql, fts_params).fetchall()

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


def _build_topic_overview(question: str, qbg_results: list[dict]) -> str:
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


def _build_concept_answer(question: str, qbg_results: list[dict]) -> str:
    """Build a clean answer using question bank data (no LLM calls).

    Uses text_solution (human-written) and answer_clean directly.
    No GPT analysis — it's raw thinking text that reads poorly.
    """
    if not qbg_results:
        return f"I couldn't find questions matching \"{question}\" in the question bank. Try more specific terms."

    if _is_broad_query(question):
        return _build_topic_overview(question, qbg_results)

    # Prefer questions with text_solution (human-written explanation)
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
        answer = _build_concept_answer(request.question, qbg_results)

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
