"""
Pace tracking — docs/spec.md Section 8.

Runs ALONGSIDE the state machine, never inside it. Level changes only from
process_answer() in state_machine.py; pace_flag changes only from here.
Only correct answers count toward the average — an incorrect fast guess
is noise, not "fast mastery".
"""
import json
from .db_models import StudentTopicState

PACE_THRESHOLD = 1.2  # avg_time_ratio above this => "slow"

from .paths import DATA_DIR

with open(DATA_DIR / "config" / "time_benchmarks.json") as f:
    _DEFAULT_BENCHMARKS = json.load(f)


def expected_time(difficulty: str) -> int:
    return _DEFAULT_BENCHMARKS[difficulty]


def update_pace(state: StudentTopicState, is_correct: bool, time_taken_seconds: int, difficulty: str) -> StudentTopicState:
    if not is_correct:
        return state  # exclude incorrect attempts from the time-ratio average

    ratio = time_taken_seconds / expected_time(difficulty)
    n = state.attempts_counted or 0

    if state.avg_time_ratio is None:
        state.avg_time_ratio = ratio
    else:
        state.avg_time_ratio = (state.avg_time_ratio * n + ratio) / (n + 1)

    state.attempts_counted = n + 1
    state.pace_flag = "slow" if state.avg_time_ratio > PACE_THRESHOLD else "fast"
    return state
