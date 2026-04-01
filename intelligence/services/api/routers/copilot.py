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
    "how do", "why does", "why do", "difference between", "derive",
    "prove", "formula", "equation", "law", "principle", "theorem",
    "mechanism", "process", "function", "structure", "diagram",
    "example", "solve", "calculate", "find the", "evaluate",
]

STRATEGY_KEYWORDS = [
    "prioritize", "priority", "focus", "strategy", "study plan",
    "revision", "predict", "important", "likely", "upcoming",
    "probability", "trending", "rising", "which topic", "which chapter",
    "how many hours", "weak", "strong", "roi", "exam brief",
]


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

        # Strategy 1: LIKE search with key words — most accurate for topic matching
        like_conditions = " AND ".join(f"question_clean LIKE ?" for w in words[:3])
        like_params = [f"%{w}%" for w in words[:3]]

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

        # Strategy 2: If LIKE returns too few, try with fewer words
        if len(rows) < 3 and len(words) > 1:
            like_conditions2 = " AND ".join(f"question_clean LIKE ?" for w in words[:2])
            like_params2 = [f"%{w}%" for w in words[:2]]
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


def _build_concept_answer(question: str, qbg_results: list[dict]) -> str:
    """Build an answer for concept questions using qbg.db results."""
    if not qbg_results:
        return f"I don't have specific content on \"{question}\" in my question bank. Try rephrasing or asking about a specific topic."

    # Use GPT analysis if available, otherwise use text_solution
    best = None
    for q in qbg_results:
        if q.get("gpt_analysis") and len(q["gpt_analysis"]) > 50:
            best = q
            break
    if not best:
        for q in qbg_results:
            if q.get("text_solution") and len(q["text_solution"]) > 50:
                best = q
                break
    if not best:
        best = qbg_results[0]

    parts = []
    parts.append(f"📚 **Related to: \"{question}\"**\n")

    # Build explanation from best match
    explanation = best.get("gpt_analysis") or best.get("text_solution") or ""
    if explanation:
        # Clean: remove HTML tags, channel tokens, GPT formatting noise
        clean = re.sub(r'<[^>]+>', '', explanation)
        clean = re.sub(r'<\|channel\|>[^<]*<\|message\|>', '', clean)
        # Split on "assistantfinal" or "assistant" marker — take only the final answer part
        for marker in ['assistantfinal', 'assistant final', 'assistant\n']:
            if marker in clean.lower():
                idx = clean.lower().index(marker)
                clean = clean[idx + len(marker):]
                break
        # Remove leading analysis/thinking text (before the actual answer)
        clean = re.sub(r'^(analysis|assistant|final|user|system)\s*', '', clean, flags=re.IGNORECASE)
        clean = clean.strip()
        # Remove "We need to..." thinking prefix if still present
        if clean.lower().startswith(('we need to', 'we should', 'let me', 'i need to', 'the user')):
            # Find the first sentence that looks like an actual answer
            sentences = re.split(r'(?<=[.!?])\s+', clean)
            # Skip thinking sentences, keep answer sentences
            answer_sentences = []
            started = False
            for s in sentences:
                s_lower = s.lower().strip()
                if not started and any(s_lower.startswith(p) for p in ('we need', 'we should', 'let me', 'i need', 'the user', 'likely', 'probably', 'so ')):
                    continue
                started = True
                answer_sentences.append(s)
            if answer_sentences:
                clean = ' '.join(answer_sentences)
        clean = clean.strip()
        if len(clean) > 1200:
            clean = clean[:1200] + "..."
        parts.append(f"**Explanation:**\n{clean}\n")

    # Show the source question for context
    best_q = best.get("question_clean", "")
    best_a = best.get("answer_clean", "")
    if best_q:
        parts.append(f"\n**Source question:** {best_q[:200]}")
    if best_a:
        parts.append(f"**Answer:** {best_a[:200]}\n")

    # Show related practice questions
    parts.append(f"\n📝 **Practice Questions ({len(qbg_results)} found):**\n")
    for i, q in enumerate(qbg_results[:3], 1):
        q_text = q.get("question_clean", "")
        if len(q_text) > 200:
            q_text = q_text[:200] + "..."
        answer = q.get("answer_clean", "—")
        difficulty = q.get("difficulty", "unknown")
        parts.append(f"{i}. [{difficulty.upper()}] {q_text}")
        parts.append(f"   → Answer: {answer}\n")

    parts.append("\n💡 Try asking me to **explain the solution** for any of these, or ask for **more practice problems**.")

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
