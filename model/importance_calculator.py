import json


def compute_importance(frequency_by_topic_year: dict, current_year: int, decay: float = 0.9) -> dict:
    raw_scores = {}
    for topic_id, year_counts in frequency_by_topic_year.items():
        score = sum(
            count * (decay ** (current_year - year))
            for year, count in year_counts.items()
        )
        raw_scores[topic_id] = score

    total = sum(raw_scores.values())
    if total == 0:
        raise ValueError("Total frequency is zero - check your input data.")

    return {topic_id: round(score / total * 100, 2) for topic_id, score in raw_scores.items()}
