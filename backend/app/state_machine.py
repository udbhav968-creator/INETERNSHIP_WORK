"""
The per-topic mastery state machine — docs/spec.md Section 7.

This is the ONE place promotion/demotion logic lives. Nothing else should
reimplement these rules — import and call process_answer() everywhere.

Rules (rolling-window based, not single-answer-triggered):
  basic    -> medium   : 2 correct out of last 3
  medium   -> advanced : 3 correct IN A ROW (a true streak, not just a window)
  medium   -> basic    : 2 wrong out of last 3
  advanced -> medium   : last 2 answers both wrong
"""
from .db_models import StudentTopicState

WINDOW_SIZE = 3


def process_answer(state: StudentTopicState, is_correct: bool) -> StudentTopicState:
    """Mutates and returns the given state object. Caller is responsible for
    committing it to the DB session."""

    results = list(state.last_3_results or [])
    results.append(is_correct)
    results = results[-WINDOW_SIZE:]
    state.last_3_results = results

    if state.level == "basic":
        state.streak = (state.streak + 1) if is_correct else 0
        if len(results) >= WINDOW_SIZE and sum(results) >= 2:
            state.level = "medium"
            state.streak = 0
            state.last_3_results = []

    elif state.level == "medium":
        state.streak = (state.streak + 1) if is_correct else 0
        wrong_count = results.count(False)
        if state.streak >= 3:
            state.level = "advanced"
            state.streak = 0
            state.last_3_results = []
        elif len(results) >= WINDOW_SIZE and wrong_count >= 2:
            state.level = "basic"
            state.streak = 0
            state.last_3_results = []

    elif state.level == "advanced":
        if len(results) >= 2 and results[-2:] == [False, False]:
            state.level = "medium"
            state.streak = 0
            state.last_3_results = []
        # otherwise: stays advanced. Paper generator periodically injects a
        # probe question from this topic to confirm retention over time.

    return state
