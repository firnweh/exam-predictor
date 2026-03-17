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
