"""
Topic importance engine — docs/spec.md Section 4.

Input: frequency_by_topic_year, a dict like:
    {
      "BIO-07-01": {2019: 2, 2021: 1, 2023: 3},
      "BIO-07-02": {2020: 1, 2022: 2},
      ...
    }
(how many times each topic appeared in each year's paper)

Output: normalized importance scores summing to 100 across all topics.
Run this once whenever new year data is added — NOT per student, NOT per paper.

This is Model Support's file to own and run. Output should be written to
model/topic_importance.json for the backend to read.
"""
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
        raise ValueError("Total frequency is zero — check your input data.")

    return {topic_id: round(score / total * 100, 2) for topic_id, score in raw_scores.items()}


# No __main__ block here on purpose — use build_topic_importance.py to
# generate the real output file from actual tagged question data. This file
# is a pure library so nothing accidentally overwrites topic_importance.json
# with placeholder numbers.
