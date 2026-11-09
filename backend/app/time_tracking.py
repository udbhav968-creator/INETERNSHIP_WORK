import json
from typing import Optional
from .db_models import StudentTopicState
from .paths import DATA_DIR

PACE_THRESHOLD = 1.2

with open(DATA_DIR / "config" / "time_benchmarks.json", encoding="utf-8") as f:
    _DEFAULT_BENCHMARKS = json.load(f)


def expected_time(difficulty: str, question_expected_seconds: Optional[int] = None) -> int:
    if question_expected_seconds is not None and question_expected_seconds > 0:
        return int(question_expected_seconds)
    return int(_DEFAULT_BENCHMARKS.get(difficulty, 90))


def update_pace(
    state: StudentTopicState,
    is_correct: bool,
    time_taken_seconds: float | int,
    difficulty: str,
    expected_time_seconds: Optional[int] = None,
) -> StudentTopicState:
    if not is_correct:
        return state

    denom = float(max(1, expected_time(difficulty, expected_time_seconds)))
    time_val = max(1.0, float(time_taken_seconds if time_taken_seconds is not None else 1.0))
    ratio = time_val / denom
    n = state.attempts_counted or 0

    if state.avg_time_ratio is None:
        state.avg_time_ratio = round(ratio, 4)
    else:
        state.avg_time_ratio = round((state.avg_time_ratio * n + ratio) / (n + 1), 4)

    state.attempts_counted = n + 1
    state.pace_flag = "slow" if state.avg_time_ratio > PACE_THRESHOLD else "fast"
    return state
