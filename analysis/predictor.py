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
