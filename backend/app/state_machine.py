from .db_models import StudentTopicState
from .paths import PROJECT_ROOT
from model.state_machine.bkt_model import compute_bkt_update, resolve_level_with_hysteresis

WINDOW_SIZE = 3


def process_answer(
    state: StudentTopicState,
    is_correct: bool,
    difficulty: str = "medium",
    time_ratio: float = 1.0,
    option_switches: int = 0,
    hesitation_ms: int = 0,
    elapsed_hours: float = 0.0,
) -> StudentTopicState:
    results = list(state.last_3_results or [])
    results.append(bool(is_correct))
    results = results[-WINDOW_SIZE:]
    state.last_3_results = results

    current_mastery = getattr(state, "mastery_prob", None)
    if current_mastery is None:
        current_mastery = 0.35 if state.level == "basic" else (0.62 if state.level == "medium" else 0.85)
    attempts_so_far = (getattr(state, "attempts_counted", 0) or 0) + len(results)

    sim_time_taken = max(1, int(round(float(time_ratio) * 90)))
    p_next, ci_95 = compute_bkt_update(
        prior_mastery=float(current_mastery),
        is_correct=bool(is_correct),
        difficulty=difficulty or state.level or "medium",
        time_taken_seconds=sim_time_taken,
        expected_time_seconds=90,
        option_switches=int(option_switches),
        hesitation_ms=int(hesitation_ms),
        attempts_count=int(attempts_so_far),
        elapsed_hours=float(elapsed_hours),
        streak=int(state.streak or 0),
    )
    state.mastery_prob = p_next
    state.confidence_interval = ci_95

    if state.level == "basic":
        state.streak = (state.streak + 1) if is_correct else 0
        if sum(1 for r in results if r) >= 2:
            new_lvl = resolve_level_with_hysteresis(state.level, p_next, results)
            if new_lvl != state.level:
                state.level = new_lvl
                state.streak = 0
                state.last_3_results = []

    elif state.level == "medium":
        state.streak = (state.streak + 1) if is_correct else 0
        wrong_count = results.count(False)
        if state.streak >= 3:
            new_lvl = resolve_level_with_hysteresis(state.level, p_next, results)
            if new_lvl != state.level:
                state.level = new_lvl
                state.streak = 0
                state.last_3_results = []
        elif wrong_count >= 2:
            new_lvl = resolve_level_with_hysteresis(state.level, p_next, results)
            if new_lvl != state.level:
                state.level = new_lvl
                state.streak = 0
                state.last_3_results = []

    elif state.level == "advanced":
        state.streak = (state.streak + 1) if is_correct else 0
        if len(results) >= 2 and results[-2:] == [False, False]:
            new_lvl = resolve_level_with_hysteresis(state.level, p_next, results)
            if new_lvl != state.level:
                state.level = new_lvl
                state.streak = 0
                state.last_3_results = []

    return state
