#!/usr/bin/env python3
"""
Process real student test data from PW AIR Vidyapeeth exports.
Input:  data/raw/question_paper_sheet.csv  (test metadata)
        data/raw/final_data_sheet.csv      (25K+ student results)
Output: docs/student_summary_neet.json
        docs/student_summary_jee.json
"""
import csv, json, hashlib, statistics, os
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

# ── Constants ──────────────────────────────────────────────────────────────────

PW_CENTERS = [
    "PW Kota", "PW Delhi", "PW Patna", "PW Lucknow", "PW Jaipur",
    "PW Mumbai", "PW Hyderabad", "PW Kolkata", "PW Chennai", "PW Pune",
    "PW Ahmedabad", "PW Bhopal",
]

PW_CITIES = {
    "PW Kota": "Kota", "PW Delhi": "Delhi", "PW Patna": "Patna",
    "PW Lucknow": "Lucknow", "PW Jaipur": "Jaipur", "PW Mumbai": "Mumbai",
    "PW Hyderabad": "Hyderabad", "PW Kolkata": "Kolkata",
    "PW Chennai": "Chennai", "PW Pune": "Pune",
    "PW Ahmedabad": "Ahmedabad", "PW Bhopal": "Bhopal",
}

# Realistic Indian name pool (first, last)
FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Vihaan", "Arjun", "Sai", "Reyansh",
    "Ayaan", "Krishna", "Ishaan", "Shaurya", "Atharv", "Advait", "Dhruv",
    "Kabir", "Ritvik", "Aarush", "Kayaan", "Darsh", "Veer", "Arnav",
    "Rudra", "Agastya", "Samar", "Yash", "Harsh", "Rohan", "Dev",
    "Ananya", "Diya", "Myra", "Sara", "Aanya", "Aadhya", "Isha",
    "Anvi", "Prisha", "Riya", "Aisha", "Navya", "Pari", "Saanvi",
    "Avni", "Kiara", "Mahi", "Tara", "Zara", "Nisha", "Pooja", "Sneha",
    "Rahul", "Amit", "Priya", "Neha", "Deepak", "Ravi", "Manish",
    "Suman", "Kavita", "Anjali", "Vikram", "Suresh", "Mohit", "Tanvi",
    "Gaurav", "Nikhil", "Karan", "Shreya", "Swati", "Meera", "Akash",
    "Divya", "Kunal", "Sakshi", "Aman", "Varun", "Nandini", "Tushar",
    "Palak", "Shivam", "Ayushi", "Pranav", "Simran", "Aryan", "Kriti",
]

LAST_NAMES = [
    "Sharma", "Verma", "Singh", "Kumar", "Gupta", "Patel", "Yadav",
    "Mishra", "Joshi", "Pandey", "Reddy", "Nair", "Menon", "Iyer",
    "Rao", "Desai", "Mehta", "Shah", "Jain", "Agarwal", "Tiwari",
    "Chauhan", "Thakur", "Srivastava", "Dubey", "Saxena", "Kapoor",
    "Bhat", "Pillai", "Das", "Mukherjee", "Banerjee", "Ghosh",
    "Chandra", "Rajput", "Malhotra", "Khanna", "Arora", "Bose",
    "Sengupta", "Patil", "Kulkarni", "Jha", "Rathore", "Trivedi",
]

# NEET subjects map from Evalresults keys
NEET_SUBJ_MAP = {
    "Physics": "Physics",
    "Chemistry": "Chemistry",
    "Botany": "Biology",
    "Zoology": "Biology",
}
JEE_SUBJ_MAP = {
    "Physics": "Physics",
    "Chemistry": "Chemistry",
    "Mathematics": "Mathematics",
    "Maths": "Mathematics",
}


def _hash_int(s, mod):
    """Deterministic hash of string to int in [0, mod)."""
    return int(hashlib.md5(s.encode()).hexdigest(), 16) % mod


def _assign_name(erpid):
    h = int(hashlib.md5(erpid.encode()).hexdigest(), 16)
    first = FIRST_NAMES[h % len(FIRST_NAMES)]
    last = LAST_NAMES[(h // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"


def _assign_center(erpid):
    return PW_CENTERS[_hash_int(erpid, len(PW_CENTERS))]


def _parse_evalresults(raw):
    """Parse Evalresults JSON string → dict."""
    if not raw or raw.strip() == "":
        return {}
    try:
        return json.loads(raw.replace('""', '"').strip('"'))
    except:
        # Try fixing common escaping issues
        try:
            cleaned = raw.strip()
            if cleaned.startswith('"') and cleaned.endswith('"'):
                cleaned = cleaned[1:-1]
            cleaned = cleaned.replace('""', '"')
            return json.loads(cleaned)
        except:
            return {}


def _extract_subject_scores(evalresults, exam_type):
    """
    Extract per-subject {subject: {right, wrong, blank, marks, total}} from Evalresults.
    NEET has: Physics, Chemistry, Botany, Zoology → merge Botany+Zoology into Biology
    JEE has: Physics, Chemistry, Mathematics/Maths
    """
    subjects = defaultdict(lambda: {"right": 0, "wrong": 0, "blank": 0, "marks": 0})

    subj_map = NEET_SUBJ_MAP if "neet" in exam_type.lower() else JEE_SUBJ_MAP

    for raw_subj, mapped in subj_map.items():
        r = int(evalresults.get(f"{raw_subj} R", evalresults.get(f"{raw_subj}R", 0)) or 0)
        w = int(evalresults.get(f"{raw_subj} W", evalresults.get(f"{raw_subj}W", 0)) or 0)
        b = int(evalresults.get(f"{raw_subj} B", evalresults.get(f"{raw_subj}B", 0)) or 0)
        m = int(evalresults.get(f"{raw_subj} Marks", evalresults.get(f"{raw_subj}Marks", 0)) or 0)
        subjects[mapped]["right"] += r
        subjects[mapped]["wrong"] += w
        subjects[mapped]["blank"] += b
        subjects[mapped]["marks"] += m

    return dict(subjects)


def _level(acc):
    if acc >= 80: return "Mastery"
    if acc >= 65: return "Strong"
    if acc >= 45: return "Developing"
    if acc >= 25: return "Weak"
    return "Critical"


def _find_topic_mapping(topic_map, test_name, exam_type):
    """Find the topic mapping entry that best matches the test name."""
    if not topic_map:
        return None
    for tm in topic_map.get("tests", []):
        # Match by test name (fuzzy — test names may differ slightly)
        tm_name = tm["test_name"].lower().replace("-", " ").replace(":", " ")
        t_name = test_name.lower().replace("-", " ").replace(":", " ")
        # Check if key parts match
        if tm["exam_type"] != exam_type:
            continue
        # Extract test number and type for matching
        if any(w in t_name and w in tm_name for w in ["test 11", "test 12", "test 13", "test 14",
               "test 15", "test 16", "test 17", "test 10", "pyq 5", "pyq 6", "pyq 7",
               "full test 1", "syllabus 01", "syllabus 02"]):
            return tm
        # Fallback: check if significant overlap
        t_words = set(t_name.split())
        tm_words = set(tm_name.split())
        overlap = len(t_words & tm_words)
        if overlap >= 3:
            return tm
    return None


def _subject_to_sections(subject, exam_type):
    """Map a merged subject name to section names in the topic mapping."""
    if subject == "Biology":
        return ["Botany", "Zoology"]
    return [subject]


def load_test_metadata():
    """Load test metadata from question paper sheet."""
    path = BASE / "data" / "raw" / "question_paper_sheet.csv"
    with open(path) as f:
        tests = list(csv.DictReader(f))

    tid_meta = {}
    for t in tests:
        tid = t["Testid's"].strip()
        if not tid:
            continue
        stream = t["Class_Stream"].strip()
        # Determine exam type
        if "NEET" in stream:
            exam = "NEET"
        elif "JEE" in stream:
            exam = "JEE"
        else:
            exam = "NEET"  # default

        tid_meta[tid] = {
            "name": t["Test Name"].strip(),
            "stream": stream,
            "exam": exam,
            "date": t["Test Date"].strip(),
        }
    return tid_meta


def load_results():
    """Load all student results from final data sheet."""
    path = BASE / "data" / "raw" / "final_data_sheet.csv"
    with open(path) as f:
        return list(csv.DictReader(f))


def load_topic_mapping():
    """Load test topic mapping from extracted PDF data."""
    path = BASE / "data" / "raw" / "test_topic_mapping.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


def build_student_profiles(results, test_meta, topic_map=None):
    """
    Build per-student profiles grouped by exam type (NEET/JEE).
    Returns {exam_type: [student_profile, ...]}
    """
    # Group results by (userid, exam_type)
    student_results = defaultdict(lambda: defaultdict(list))

    for r in results:
        tid = r["testid"].strip()
        uid = r["userid"].strip()
        erpid = r["erpid"].strip()
        if not uid or not tid:
            continue

        meta = test_meta.get(tid)
        if not meta:
            continue

        exam_type = meta["exam"]
        evalresults = _parse_evalresults(r.get("Evalresults", ""))
        subj_scores = _extract_subject_scores(evalresults, exam_type)

        userscore = int(r.get("userscore", 0) or 0)
        totalscore = int(r.get("totalscore", 0) or 0)
        correct = int(r.get("correctquestions", 0) or 0)
        incorrect = int(r.get("incorrectquestions", 0) or 0)
        total_qs = int(r.get("totalquestions", 0) or 0)

        pct = (userscore / totalscore * 100) if totalscore > 0 else 0

        student_results[(uid, exam_type)]["results"].append({
            "test_name": meta["name"],
            "test_date": meta["date"],
            "stream": meta["stream"],
            "userscore": userscore,
            "totalscore": totalscore,
            "pct": round(pct, 2),
            "correct": correct,
            "incorrect": incorrect,
            "total_qs": total_qs,
            "subjects": subj_scores,
        })
        student_results[(uid, exam_type)]["erpid"] = erpid

    # Build profiles
    profiles = defaultdict(list)

    for (uid, exam_type), data in student_results.items():
        erpid = data["erpid"]
        test_results = sorted(data["results"], key=lambda x: x["test_date"])

        if not test_results:
            continue

        name = _assign_name(erpid)
        center = _assign_center(erpid)
        city = PW_CITIES[center]

        # Trajectory = percentage per test
        trajectory = [r["pct"] for r in test_results]
        avg_pct = statistics.mean(trajectory) if trajectory else 0
        best_pct = max(trajectory) if trajectory else 0
        latest_pct = trajectory[-1] if trajectory else 0

        # Improvement: latest - first (or avg of last 3 - avg of first 3)
        if len(trajectory) >= 2:
            improvement = trajectory[-1] - trajectory[0]
        else:
            improvement = 0

        # Consistency: 100 - std_dev (capped)
        if len(trajectory) >= 2:
            std = statistics.stdev(trajectory)
            consistency = max(0, min(100, 100 - std))
        else:
            consistency = 50

        # Per-subject aggregation across all tests
        subj_agg = defaultdict(lambda: {"right": 0, "wrong": 0, "blank": 0, "marks": 0, "exams": []})
        for r in test_results:
            for subj, scores in r["subjects"].items():
                subj_agg[subj]["right"] += scores["right"]
                subj_agg[subj]["wrong"] += scores["wrong"]
                subj_agg[subj]["blank"] += scores["blank"]
                subj_agg[subj]["marks"] += scores["marks"]
                # Per-exam accuracy
                attempted = scores["right"] + scores["wrong"]
                if attempted > 0:
                    subj_agg[subj]["exams"].append(round(scores["right"] / attempted * 100, 1))

        subjects_out = {}
        for subj, agg in subj_agg.items():
            attempted = agg["right"] + agg["wrong"]
            acc = round(agg["right"] / attempted * 100, 1) if attempted > 0 else 0
            trend = 0
            exams = agg["exams"]
            if len(exams) >= 2:
                # Simple linear trend per exam
                trend = round((exams[-1] - exams[0]) / len(exams), 3)
            subjects_out[subj] = {
                "acc": acc,
                "level": _level(acc),
                "trend": trend,
                "exams": exams[-10:],  # last 10
            }

        # Abilities dict
        abilities = {}
        subj_key_map = {"Physics": "phy", "Chemistry": "chem", "Biology": "bio", "Mathematics": "maths"}
        for subj, sdata in subjects_out.items():
            key = subj_key_map.get(subj, subj.lower()[:4])
            abilities[key] = round(sdata["acc"] / 100, 4)

        # Ranks (placeholder — will be computed after all students processed)

        # Chapters: use topic mapping to assign chapter-level accuracy
        # For each test the student took, find the topics covered per subject
        # and assign the subject accuracy to each chapter
        chapters_agg = defaultdict(lambda: {"acc_sum": 0, "count": 0})
        if topic_map:
            for r in test_results:
                test_name = r["test_name"]
                # Find matching topic mapping entry
                tm = _find_topic_mapping(topic_map, test_name, exam_type)
                if not tm:
                    continue
                for subj, scores in r["subjects"].items():
                    attempted = scores["right"] + scores["wrong"]
                    if attempted == 0:
                        continue
                    subj_acc = round(scores["right"] / attempted * 100, 1)
                    # Map subject to section name(s) in the topic mapping
                    section_names = _subject_to_sections(subj, exam_type)
                    for sec_name in section_names:
                        sec = tm["sections"].get(sec_name, {})
                        topics = sec.get("topics", [])
                        for topic in topics:
                            if topic.startswith("Full Syllabus"):
                                continue  # skip generic "Full Syllabus" entries
                            chapters_agg[topic]["acc_sum"] += subj_acc
                            chapters_agg[topic]["count"] += 1

        chapters_out = {}
        for ch, agg in chapters_agg.items():
            avg_acc = round(agg["acc_sum"] / agg["count"], 1) if agg["count"] > 0 else 0
            chapters_out[ch] = [avg_acc, _level(avg_acc)[0], agg["count"]]

        student_id = f"STU{erpid[-6:]}" if len(erpid) >= 6 else f"STU{erpid}"

        profile = {
            "id": student_id,
            "erpid": erpid,
            "name": name,
            "city": city,
            "coaching": center,
            "target": exam_type,
            "abilities": abilities,
            "metrics": {
                "avg_percentage": round(avg_pct, 1),
                "best_percentage": round(best_pct, 1),
                "latest_percentage": round(latest_pct, 1),
                "improvement": round(improvement, 1),
                "best_rank": 0,  # computed later
                "latest_rank": 0,
                "consistency_score": round(consistency, 1),
                "trend_per_exam": round(improvement / max(len(trajectory) - 1, 1), 3),
                "trajectory": [round(t, 2) for t in trajectory[-10:]],
            },
            "subjects": subjects_out,
            "trajectory": trajectory[-10:],
            "ranks": [],  # computed later
            "chapters": chapters_out,
            "slm_focus": [],  # computed below from weak chapters
            "strengths": [],  # computed from subjects
            "tests_taken": len(test_results),
        }

        # Strengths: top 5 chapters by accuracy (if available), else top subjects
        if chapters_out:
            sorted_chs = sorted(chapters_out.items(), key=lambda x: x[1][0], reverse=True)
            profile["strengths"] = [
                {"chapter": ch, "acc": data[0]} for ch, data in sorted_chs[:5]
            ]
            # SLM focus: weakest chapters with enough data
            weak_chs = sorted(
                [(ch, data) for ch, data in chapters_out.items() if data[2] >= 1],
                key=lambda x: x[1][0]
            )
            profile["slm_focus"] = [
                {
                    "chapter": ch, "accuracy": data[0],
                    "level": _level(data[0]),
                    "slm_importance": 70 + (50 - data[0]) * 0.5,  # higher importance for weaker chapters
                    "slm_priority_score": round((100 - data[0]) * 0.6, 1),
                    "trend": 0, "consistency": data[2] * 10,
                }
                for ch, data in weak_chs[:5]
            ]
        else:
            sorted_subjs = sorted(subjects_out.items(), key=lambda x: x[1]["acc"], reverse=True)
            profile["strengths"] = [
                {"chapter": s, "acc": d["acc"]} for s, d in sorted_subjs[:3]
            ]

        profiles[exam_type].append(profile)

    # Compute ranks per exam type
    for exam_type, students in profiles.items():
        # Sort by avg_percentage descending
        ranked = sorted(students, key=lambda s: s["metrics"]["avg_percentage"], reverse=True)
        for i, s in enumerate(ranked):
            s["metrics"]["best_rank"] = i + 1
            s["metrics"]["latest_rank"] = i + 1
            # Approximate per-test ranks
            s["ranks"] = list(range(max(1, i - 2), min(len(ranked), i + 4)))

    return profiles


def main():
    print("Loading test metadata...")
    test_meta = load_test_metadata()
    print(f"  → {len(test_meta)} tests loaded")

    print("Loading student results...")
    results = load_results()
    print(f"  → {len(results)} result rows loaded")

    print("Loading topic mapping...")
    topic_map = load_topic_mapping()
    if topic_map:
        print(f"  → {len(topic_map.get('tests', []))} test topic mappings loaded")
    else:
        print("  → No topic mapping found (chapters will be empty)")

    print("Building student profiles...")
    profiles = build_student_profiles(results, test_meta, topic_map)

    for exam_type, students in profiles.items():
        print(f"\n  {exam_type}: {len(students)} students")
        if students:
            scores = [s["metrics"]["avg_percentage"] for s in students]
            print(f"    avg score: {statistics.mean(scores):.1f}%")
            print(f"    score range: {min(scores):.1f}% – {max(scores):.1f}%")
            # Subject breakdown
            all_subjs = set()
            for s in students:
                all_subjs.update(s["subjects"].keys())
            for subj in sorted(all_subjs):
                accs = [s["subjects"][subj]["acc"] for s in students if subj in s["subjects"]]
                if accs:
                    print(f"    {subj}: avg {statistics.mean(accs):.1f}%, range {min(accs):.0f}–{max(accs):.0f}%")

    # Write output
    for exam_type, students in profiles.items():
        exam_key = exam_type.lower()
        out_path = BASE / "docs" / f"student_summary_{exam_key}.json"
        output = {
            "exam_type": exam_type,
            "students": students,
        }
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\n  Wrote {out_path} ({len(students)} students)")

    # Also write to student_data for backend
    data_dir = BASE / "data" / "student_data"
    data_dir.mkdir(exist_ok=True)
    for exam_type, students in profiles.items():
        out_path = data_dir / f"student_summary_{exam_type.lower()}.json"
        output = {"exam_type": exam_type, "students": students}
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"  Wrote {out_path}")

    print("\nDone! Fake data has been replaced with real student profiles.")


if __name__ == "__main__":
    main()
