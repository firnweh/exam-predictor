"""
Scrape NEET/AIPMT questions from ExamSIDE's SvelteKit API.
Extracts questions with options, answers, topics, and explanations.

Usage:
    python extraction/scrape_examside.py

Output: JSON files in data/extracted/ (one per paper)
"""

import requests
import json
import os
import re
import time
import sys

BASE_URL = "https://questions.examside.com/past-years/year-wise/medical/neet"
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}
OUTPUT_DIR = "data/extracted"

# All NEET/AIPMT papers available on ExamSIDE
PAPERS = [
    "neet-2025", "neet-2024-re-examination", "neet-2024",
    "neet-2023-manipur", "neet-2023",
    "neet-2022-phase-2", "neet-2022",
    "neet-2021", "neet-2020-phase-1", "neet-2019", "neet-2018",
    "neet-2017", "neet-2016-phase-2", "neet-2016-phase-1",
    "aipmt-2015", "aipmt-2015-cancelled-paper", "aipmt-2014",
    "neet-2013-karnataka", "neet-2013",
    "aipmt-2012-mains", "aipmt-2012-prelims",
    "aipmt-2011-mains", "aipmt-2011-prelims",
    "aipmt-2010-mains", "aipmt-2010-prelims",
    "aipmt-2009", "aipmt-2008", "aipmt-2007", "aipmt-2006",
    "aipmt-2005", "aipmt-2004", "aipmt-2003", "aipmt-2002",
    "aipmt-2001", "aipmt-2000",
]


def clean_html(html):
    """Strip HTML tags from content."""
    if not html or not isinstance(html, str):
        return ""
    text = re.sub(r"<[^>]+>", "", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&#39;", "'").replace("&quot;", '"')
    return text.strip()


def resolve(data, idx):
    """Resolve a SvelteKit deduplication index to its actual value."""
    if isinstance(idx, int) and 0 <= idx < len(data):
        return data[idx]
    return idx


def get_question_ids(paper_key):
    """Fetch all question IDs for a paper from the listing page."""
    url = f"{BASE_URL}/{paper_key}/__data.json"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    nodes = r.json()["nodes"]

    d = nodes[1]["data"]
    questions = []

    # Find subject groups (Biology, Chemistry, Physics)
    # d[0] has a 'questions' key pointing to the subjects list
    subjects_idx = d[0].get("questions")
    if subjects_idx is None:
        return []

    subjects_list = resolve(d, subjects_idx)
    if not isinstance(subjects_list, list):
        return []

    for subj_idx in subjects_list:
        subj = resolve(d, subj_idx)
        if isinstance(subj, dict) and "questions" in subj:
            q_indices = resolve(d, subj["questions"])
            if isinstance(q_indices, list):
                for qi in q_indices:
                    q = resolve(d, qi)
                    if isinstance(q, dict) and "question_id" in q:
                        qid = resolve(d, q["question_id"])
                        questions.append(qid)

    return questions


def extract_question(paper_key, question_id, data):
    """Extract a single question from the individual question page data."""
    d = data

    # Find the question metadata dict (has 24 keys with question_id, chapter, etc.)
    meta = None
    q_content_en = None

    for i, item in enumerate(d):
        if isinstance(item, dict) and "question_id" in item and "chapter" in item:
            resolved_id = resolve(d, item["question_id"])
            if resolved_id == question_id:
                meta = item
                break

    if not meta:
        # Try to use the first metadata dict (usually d[5])
        for i, item in enumerate(d):
            if isinstance(item, dict) and len(item) > 20 and "question_id" in item:
                meta = item
                break

    if not meta:
        return None

    # Resolve metadata
    subject = resolve(d, meta.get("subject", ""))
    chapter = resolve(d, meta.get("chapter", ""))
    topic = resolve(d, meta.get("topic", ""))
    year = resolve(d, meta.get("year", 0))
    q_type = resolve(d, meta.get("type", "mcq"))
    marks = resolve(d, meta.get("marks", 4))
    difficulty = resolve(d, meta.get("difficulty"))

    # Find the English question content dict
    question_ref = resolve(d, meta.get("question"))
    if isinstance(question_ref, dict) and "en" in question_ref:
        q_content_en = resolve(d, question_ref["en"])
    else:
        # Search for content dict near the metadata
        for i, item in enumerate(d):
            if isinstance(item, dict) and "content" in item and "options" in item and "correct_options" in item:
                q_content_en = item
                break

    if not q_content_en:
        return None

    # Extract question text
    content_html = resolve(d, q_content_en.get("content", ""))
    question_text = clean_html(content_html)

    # Extract options
    options_list = resolve(d, q_content_en.get("options", []))
    options = {}
    if isinstance(options_list, list):
        for opt_idx in options_list:
            opt = resolve(d, opt_idx)
            if isinstance(opt, dict):
                identifier = resolve(d, opt.get("identifier", ""))
                opt_content = resolve(d, opt.get("content", ""))
                options[identifier] = clean_html(opt_content)

    # Extract correct answer
    correct_ref = resolve(d, q_content_en.get("correct_options", []))
    correct_options = []
    if isinstance(correct_ref, list):
        correct_options = [resolve(d, c) for c in correct_ref]
    elif isinstance(correct_ref, str):
        correct_options = [correct_ref]

    # Extract explanation
    explanation_html = resolve(d, q_content_en.get("explanation", ""))
    explanation = clean_html(explanation_html)

    # Determine exam name from paper_key
    if "aipmt" in paper_key:
        exam = "NEET"  # Store as NEET for consistency
    else:
        exam = "NEET"

    # Determine question type
    if q_type == "mcq":
        q_type_mapped = "MCQ_single"
    elif q_type == "mcq_multiple":
        q_type_mapped = "MCQ_multi"
    elif q_type == "integer" or q_type == "numerical":
        q_type_mapped = "numerical"
    else:
        q_type_mapped = "MCQ_single"

    # Format chapter/topic for readability
    def format_slug(slug):
        if not slug or not isinstance(slug, str):
            return ""
        return slug.replace("-", " ").title()

    return {
        "id": f"NEET_{year}_{question_id}",
        "exam": exam,
        "year": year if isinstance(year, int) else 0,
        "shift": paper_key,
        "subject": format_slug(subject) if subject else "Unknown",
        "topic": format_slug(chapter) if chapter else "General",
        "micro_topic": format_slug(topic) if topic else format_slug(chapter),
        "question_text": question_text,
        "question_type": q_type_mapped,
        "difficulty": difficulty if isinstance(difficulty, int) else 3,
        "concepts_tested": [],
        "answer": ",".join(correct_options) if correct_options else "N/A",
        "marks": marks if isinstance(marks, int) else 4,
    }


def scrape_paper(paper_key):
    """Scrape all questions for a single paper."""
    print(f"\n{'='*60}")
    print(f"Scraping: {paper_key}")
    print(f"{'='*60}")

    # Step 1: Get question IDs from listing page
    try:
        question_ids = get_question_ids(paper_key)
    except Exception as e:
        print(f"  ERROR fetching listing: {e}")
        return []

    print(f"  Found {len(question_ids)} questions")

    if not question_ids:
        return []

    # Step 2: Fetch individual question pages (they have 5 questions each)
    questions = []
    fetched_ids = set()

    for i, qid in enumerate(question_ids):
        if qid in fetched_ids:
            continue

        try:
            url = f"{BASE_URL}/{paper_key}/{qid}/__data.json"
            r = requests.get(url, headers=HEADERS)
            r.raise_for_status()
            nodes = r.json()["nodes"]

            # Node 2 has the question detail data
            if len(nodes) > 2:
                d = nodes[2]["data"]

                # Extract ALL questions from this page (it may have multiple)
                for j, item in enumerate(d):
                    if isinstance(item, dict) and "question_id" in item and "chapter" in item and len(item) > 20:
                        this_qid = resolve(d, item["question_id"])
                        if this_qid in fetched_ids:
                            continue

                        # Find the content for this question
                        question_ref = resolve(d, item.get("question"))
                        if isinstance(question_ref, dict) and "en" in question_ref:
                            content_dict = resolve(d, question_ref["en"])
                            if isinstance(content_dict, dict) and "content" in content_dict:
                                q = extract_from_meta_and_content(d, item, content_dict)
                                if q and q["question_text"]:
                                    questions.append(q)
                                    fetched_ids.add(this_qid)

            # Rate limiting
            time.sleep(0.5)

            if (i + 1) % 10 == 0:
                print(f"  Progress: {len(fetched_ids)}/{len(question_ids)} questions extracted")

        except Exception as e:
            print(f"  ERROR on question {qid}: {e}")
            time.sleep(1)

    print(f"  Total extracted: {len(questions)} questions")
    return questions


def extract_from_meta_and_content(d, meta, content_dict):
    """Extract question data from metadata and content dicts."""
    subject = resolve(d, meta.get("subject", ""))
    chapter = resolve(d, meta.get("chapter", ""))
    topic = resolve(d, meta.get("topic", ""))
    year = resolve(d, meta.get("year", 0))
    q_type = resolve(d, meta.get("type", "mcq"))
    marks = resolve(d, meta.get("marks", 4))
    difficulty = resolve(d, meta.get("difficulty"))
    question_id = resolve(d, meta.get("question_id", ""))
    paper_key = resolve(d, meta.get("yearKey", ""))

    # Content
    content_html = resolve(d, content_dict.get("content", ""))
    question_text = clean_html(content_html)

    # Options
    options_list = resolve(d, content_dict.get("options", []))
    options = {}
    if isinstance(options_list, list):
        for opt_idx in options_list:
            opt = resolve(d, opt_idx)
            if isinstance(opt, dict):
                identifier = resolve(d, opt.get("identifier", ""))
                opt_content = resolve(d, opt.get("content", ""))
                options[identifier] = clean_html(opt_content)

    # Correct answer
    correct_ref = resolve(d, content_dict.get("correct_options", []))
    correct_options = []
    if isinstance(correct_ref, list):
        correct_options = [resolve(d, c) for c in correct_ref]

    def format_slug(slug):
        if not slug or not isinstance(slug, str):
            return ""
        return slug.replace("-", " ").title()

    q_type_mapped = "MCQ_single"
    if q_type == "mcq_multiple":
        q_type_mapped = "MCQ_multi"
    elif q_type in ("integer", "numerical"):
        q_type_mapped = "numerical"

    return {
        "id": f"NEET_{year}_{question_id}",
        "exam": "NEET",
        "year": year if isinstance(year, int) else 0,
        "shift": str(paper_key),
        "subject": format_slug(subject) if subject else "Unknown",
        "topic": format_slug(chapter) if chapter else "General",
        "micro_topic": format_slug(topic) if topic else format_slug(chapter),
        "question_text": question_text,
        "question_type": q_type_mapped,
        "difficulty": difficulty if isinstance(difficulty, int) else 3,
        "concepts_tested": [],
        "answer": ",".join(correct_options) if correct_options else "N/A",
        "marks": marks if isinstance(marks, int) else 4,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Allow selecting specific papers via command line
    papers_to_scrape = PAPERS
    if len(sys.argv) > 1:
        papers_to_scrape = sys.argv[1:]

    total_questions = 0

    for paper_key in papers_to_scrape:
        output_file = os.path.join(OUTPUT_DIR, f"{paper_key}.json")

        # Skip if already scraped
        if os.path.exists(output_file):
            with open(output_file) as f:
                existing = json.load(f)
            print(f"\nSkipping {paper_key} — already have {len(existing)} questions")
            total_questions += len(existing)
            continue

        questions = scrape_paper(paper_key)

        if questions:
            with open(output_file, "w") as f:
                json.dump(questions, f, indent=2, ensure_ascii=False)
            print(f"  Saved to {output_file}")
            total_questions += len(questions)
        else:
            print(f"  No questions extracted for {paper_key}")

        # Be polite between papers
        time.sleep(1)

    print(f"\n{'='*60}")
    print(f"DONE! Total questions extracted: {total_questions}")
    print(f"Files saved in: {OUTPUT_DIR}/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
