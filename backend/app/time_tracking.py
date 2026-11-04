"""
Pace tracking — docs/spec.md Section 8 + Part I Section 7 Fix #3.

Runs ALONGSIDE the state machine, never inside it. Level changes only from
process_answer() in state_machine.py; pace_flag changes only from here.
Only correct answers count toward the average — an incorrect fast guess
is noise, not "fast mastery".

Part I Section 7 Fix #3:
Supports per-question expected_time_seconds from Question.expected_time_seconds
with automatic fallback to time_benchmarks.json.
"""
import json
from typing import Optional
from .db_models import StudentTopicState
from .paths import DATA_DIR

PACE_THRESHOLD = 1.2  # avg_time_ratio above this => "slow"

with open(DATA_DIR / "config" / "time_benchmarks.json", encoding="utf-8") as f:
    _DEFAULT_BENCHMARKS = json.load(f)


def expected_time(difficulty: str, question_expected_seconds: Optional[int] = None) -> int:
    if question_expected_seconds is not None and question_expected_seconds > 0:
        return int(question_expected_seconds)
    return int(_DEFAULT_BENCHMARKS.get(difficulty, 90))


def update_pace(
    state: StudentTopicState,
    is_correct: bool,
    time_taken_seconds: int,
    difficulty: str,
    expected_time_seconds: Optional[int] = None,
) -> StudentTopicState:
    if not is_correct:
        return state  # exclude incorrect attempts from the time-ratio average

    denom = max(1, expected_time(difficulty, expected_time_seconds))
    ratio = time_taken_seconds / denom
    n = state.attempts_counted or 0

    if state.avg_time_ratio is None:
        state.avg_time_ratio = ratio
    else:
        state.avg_time_ratio = (state.avg_time_ratio * n + ratio) / (n + 1)

    state.attempts_counted = n + 1
    state.pace_flag = "slow" if state.avg_time_ratio > PACE_THRESHOLD else "fast"
    return state
