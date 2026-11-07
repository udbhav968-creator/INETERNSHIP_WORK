import json
import random
import time
from pathlib import Path
import numpy as np

from .dina_cognitive_diagnosis import compute_dina_and_rt_profile
from .feature_store_and_drift import FEATURE_STORE
from .irt_3pl_model import (
    IRTItemParams,
    fisher_information,
    get_item_params,
    irt_3pl_probability,
    theta_to_percentile,
    update_student_theta_map,
)
from .knowledge_graph_dag import diagnose_prerequisite_bottlenecks
from .linucb_bandit import (
    ZPD_TARGET_PROB,
    LinUCBStudentState,
    build_context_vector,
)
from .rag_socratic_engine import generate_socratic_remediation
from .rf_compact_runtime import COMPACT_RF_RUNTIME
from .state_machine.bkt_model import (
    compute_bkt_update,
    compute_ebbinghaus_forgetting_prior,
    compute_retention_urgency,
    resolve_level_with_hysteresis,
)
from .study_plan_and_coverage_engine import compute_syllabus_coverage_and_improvement_plan

MODEL_DIR = Path(__file__).resolve().parent
HARNESS_REPORT_PATH = MODEL_DIR / "harness_benchmark_report.json"


def _percentiles_ms(samples_ms: list[float]) -> dict[str, float]:
    arr = sorted(samples_ms)
    n = len(arr)
    if n == 0:
        return {"p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    return {
        "p50_ms": round(arr[int(0.50 * (n - 1))], 4),
        "p95_ms": round(arr[int(0.95 * (n - 1))], 4),
        "p99_ms": round(arr[int(0.99 * (n - 1))], 4),
    }


def simulate_student_personas() -> dict:
    master_level = "basic"
    master_prob = 0.35
    master_ci = 0.35
    master_theta = 0.0
    master_streak = 0
    for step in range(1, 15):
        item = IRTItemParams(
            a=1.35,
            b=-0.8 if master_level == "basic" else (0.1 if master_level == "medium" else 1.1),
            c=0.22,
        )
        master_prob, master_ci = compute_bkt_update(
            prior_mastery=master_prob,
            is_correct=True,
            difficulty=master_level,
            time_taken_seconds=52,
            expected_time_seconds=90,
            option_switches=0,
            hesitation_ms=2500,
            attempts_count=step,
            streak=master_streak,
        )
        master_streak += 1
        master_level = resolve_level_with_hysteresis(master_level, master_prob, master_ci)
        master_theta = update_student_theta_map(master_theta, True, item)

    struggler_level = "advanced"
    struggler_prob = 0.75
    struggler_ci = 0.25
    struggler_theta = 0.5
    for step in range(1, 12):
        item = IRTItemParams(a=1.40, b=0.2, c=0.22)
        struggler_prob, struggler_ci = compute_bkt_update(
            prior_mastery=struggler_prob,
            is_correct=False,
            difficulty=struggler_level,
            time_taken_seconds=155,
            expected_time_seconds=90,
            option_switches=2,
            hesitation_ms=25000,
            attempts_count=step,
            streak=0,
        )
        struggler_level = resolve_level_with_hysteresis(struggler_level, struggler_prob, struggler_ci)
        struggler_theta = update_student_theta_map(struggler_theta, False, item)

    bottlenecks = diagnose_prerequisite_bottlenecks(
        {"Rotational Motion": struggler_prob, "Laws of Motion": 0.28, "Kinematics": 0.30}
    )

    p_deliberate, _ = compute_bkt_update(
        prior_mastery=0.40,
        is_correct=True,
        difficulty="medium",
        time_taken_seconds=68,
        expected_time_seconds=90,
        option_switches=0,
        hesitation_ms=4000,
        attempts_count=3,
    )
    p_impulsive, _ = compute_bkt_update(
        prior_mastery=0.40,
        is_correct=True,
        difficulty="medium",
        time_taken_seconds=5,
        expected_time_seconds=90,
        option_switches=4,
        hesitation_ms=55000,
        attempts_count=3,
    )

    peak_m = 0.88
    decayed_7d = compute_ebbinghaus_forgetting_prior(peak_m, elapsed_hours=168.0, streak=1)
    urgency_7d = compute_retention_urgency(peak_m, elapsed_hours=168.0, streak=1)

    return {
        "strong_master": {
            "final_level": master_level,
            "final_mastery_prob": master_prob,
            "final_theta": master_theta,
            "estimated_percentile": theta_to_percentile(master_theta),
            "invariants_satisfied": bool(master_level == "advanced" and master_prob >= 0.85 and master_theta > 0.8),
        },
        "foundational_struggler": {
            "final_level": struggler_level,
            "final_mastery_prob": struggler_prob,
            "final_theta": struggler_theta,
            "detected_bottlenecks": len(bottlenecks),
            "invariants_satisfied": bool(
                struggler_level == "basic" and struggler_prob < 0.35 and struggler_theta < 0.0 and len(bottlenecks) >= 1
            ),
        },
        "impulsive_guesser_telemetry_audit": {
            "deliberate_posterior": p_deliberate,
            "impulsive_guess_posterior": p_impulsive,
            "damping_delta": round(p_deliberate - p_impulsive, 4),
            "invariants_satisfied": bool(p_deliberate > p_impulsive + 0.05),
        },
        "spaced_repetition_learner": {
            "peak_mastery": peak_m,
            "decayed_mastery_after_168h": decayed_7d,
            "retention_urgency": urgency_7d,
            "invariants_satisfied": bool(0.35 <= decayed_7d < peak_m and urgency_7d > 0.20),
        },
    }


def benchmark_linucb_vs_random_ablation(n_rounds: int = 120) -> dict:
    rng = random.Random(42)
    student_theta = 0.15
    bandit = LinUCBStudentState(alpha=0.35)

    candidate_arms = [
        {"qid": "ARM_EASY", "diff": "basic", "exp_t": 50, "params": IRTItemParams(a=0.95, b=-1.60, c=0.25), "imp": 1.5, "mastery": 0.82},
        {"qid": "ARM_ZPD_OPT", "diff": "medium", "exp_t": 85, "params": IRTItemParams(a=1.65, b=-0.22, c=0.20), "imp": 4.2, "mastery": 0.48},
        {"qid": "ARM_HARD", "diff": "advanced", "exp_t": 120, "params": IRTItemParams(a=1.30, b=1.95, c=0.18), "imp": 2.0, "mastery": 0.30},
    ]

    def _arm_context_and_zpd(arm: dict) -> tuple[np.ndarray, float, float]:
        p = irt_3pl_probability(student_theta, arm["params"])
        info = fisher_information(student_theta, arm["params"])
        zpd_align = max(0.0, 1.0 - abs(p - ZPD_TARGET_PROB) * 1.5)
        x = np.array(
            [
                zpd_align,
                min(1.0, info / 1.5),
                1.0 - arm["mastery"],
                min(1.0, arm["imp"] / 5.5),
                0.85,
                0.25,
            ],
            dtype=np.float64,
        )
        return x, zpd_align, info

    oracle_zpd = max(_arm_context_and_zpd(a)[1] for a in candidate_arms)

    linucb_zpd_scores: list[float] = []
    linucb_info_scores: list[float] = []
    linucb_regret = 0.0

    random_zpd_scores: list[float] = []
    random_info_scores: list[float] = []
    random_regret = 0.0

    for _ in range(n_rounds):
        best_ucb = -1e9
        best_x = None
        best_zpd = 0.0
        best_info = 0.0
        for arm in candidate_arms:
            x, zpd_val, info_val = _arm_context_and_zpd(arm)
            w_hat = bandit.A_inv @ bandit.b
            ucb = float(x @ w_hat) + bandit.alpha * float(np.sqrt(max(1e-9, float(x @ bandit.A_inv @ x))))
            if ucb > best_ucb:
                best_ucb = ucb
                best_x = x
                best_zpd = zpd_val
                best_info = info_val

        reward = 0.65 * best_zpd + 0.35 * min(1.0, best_info)
        bandit.update(best_x, reward)
        linucb_zpd_scores.append(best_zpd)
        linucb_info_scores.append(best_info)
        linucb_regret += max(0.0, oracle_zpd - best_zpd)

        rand_arm = rng.choice(candidate_arms)
        _, r_zpd, r_info = _arm_context_and_zpd(rand_arm)
        random_zpd_scores.append(r_zpd)
        random_info_scores.append(r_info)
        random_regret += max(0.0, oracle_zpd - r_zpd)

    mean_linucb_zpd = round(float(np.mean(linucb_zpd_scores)), 4)
    mean_random_zpd = round(float(np.mean(random_zpd_scores)), 4)
    mean_linucb_info = round(float(np.mean(linucb_info_scores)), 4)
    mean_random_info = round(float(np.mean(random_info_scores)), 4)

    return {
        "n_rounds": n_rounds,
        "linucb_mean_zpd_alignment": mean_linucb_zpd,
        "random_mean_zpd_alignment": mean_random_zpd,
        "linucb_mean_fisher_info": mean_linucb_info,
        "random_mean_fisher_info": mean_random_info,
        "linucb_cumulative_regret": round(linucb_regret, 4),
        "random_cumulative_regret": round(random_regret, 4),
        "linucb_outperforms_random": bool(
            mean_linucb_zpd > mean_random_zpd and linucb_regret < random_regret
        ),
    }


def benchmark_engine_latencies(iterations: int = 50) -> dict:
    irt_times: list[float] = []
    bkt_times: list[float] = []
    bandit_times: list[float] = []
    rf_times: list[float] = []
    dina_times: list[float] = []
    rag_times: list[float] = []
    plan_times: list[float] = []
    fs_times: list[float] = []

    sample_item = get_item_params("NEET_2025_PHY_001", "medium", 90)
    sample_attempts = [
        {
            "question_id": "Q1",
            "question_text": "Calculate molarity in mol/L for 4.0 g NaOH in 500 mL.",
            "topic_id": "Mole Concept / Stoichiometry",
            "difficulty": "basic",
            "expected_time_seconds": 90,
            "time_taken_seconds": 45,
            "is_correct": True,
        }
    ]

    for i in range(iterations):
        t0 = time.perf_counter()
        update_student_theta_map(0.1, bool(i % 2 == 0), sample_item)
        irt_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        compute_bkt_update(0.45, bool(i % 2 == 0), "medium", 70, 90, 1, 3000, i + 1, elapsed_hours=12.0, streak=1)
        bkt_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        build_context_vector("NEET_2025_PHY_001", "medium", 90, 0.1, 0.45, 3.2, 0.85, None)
        bandit_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        COMPACT_RF_RUNTIME.predict_proba_single(
            subject="Physics",
            chapter="Electrostatics",
            topic_tag="Electric Field",
            question_type="Single_Choice_MCQ",
            ideal_time_sec=90,
            has_diagram=False,
        )
        rf_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        compute_dina_and_rt_profile(sample_attempts)
        dina_times.append((time.perf_counter() - t0) * 1000.0)

        t0 = time.perf_counter()
        FEATURE_STORE.put_student_features("HARNESS_BENCH", {"theta": 0.1})
        FEATURE_STORE.get_student_features("HARNESS_BENCH")
        fs_times.append((time.perf_counter() - t0) * 1000.0)
    FEATURE_STORE.evict_student("HARNESS_BENCH")

    for _ in range(15):
        t0 = time.perf_counter()
        generate_socratic_remediation(
            question_id="NEET_2025_PHY_001",
            topic_id="Ray Optics",
            question_text="Find the focal length of a concave mirror of radius 20 cm.",
            options=["-10 cm", "+10 cm", "-20 cm", "+20 cm"],
            correct_answer="a",
            selected_option="b",
        )
        rag_times.append((time.perf_counter() - t0) * 1000.0)

    for _ in range(10):
        t0 = time.perf_counter()
        compute_syllabus_coverage_and_improvement_plan(
            student_id="HARNESS_PLAN",
            global_theta=-0.1,
            topic_states=[{"topic_id": "Electrostatics", "level": "medium", "mastery_prob": 0.42}],
            attempt_records=sample_attempts,
        )
        plan_times.append((time.perf_counter() - t0) * 1000.0)

    return {
        "irt_3pl_map_update": _percentiles_ms(irt_times),
        "bkt_saint_ebbinghaus_update": _percentiles_ms(bkt_times),
        "linucb_context_scoring": _percentiles_ms(bandit_times),
        "compact_rf_300_tree_inference": _percentiles_ms(rf_times),
        "dina_8_skill_cognitive_diagnosis": _percentiles_ms(dina_times),
        "rag_socratic_remediation": _percentiles_ms(rag_times),
        "study_plan_coverage_prescription": _percentiles_ms(plan_times),
        "in_memory_feature_store": _percentiles_ms(fs_times),
        "all_slas_met": bool(
            _percentiles_ms(irt_times)["p95_ms"] < 5.0
            and _percentiles_ms(bkt_times)["p95_ms"] < 5.0
            and _percentiles_ms(rf_times)["p95_ms"] < 20.0
            and _percentiles_ms(rag_times)["p95_ms"] < 40.0
        ),
    }


def run_evaluation_harness(save_report: bool = True) -> dict:
    t0 = time.perf_counter()
    personas = simulate_student_personas()
    ablation = benchmark_linucb_vs_random_ablation(n_rounds=120)
    latencies = benchmark_engine_latencies(iterations=50)

    all_passed = (
        personas["strong_master"]["invariants_satisfied"]
        and personas["foundational_struggler"]["invariants_satisfied"]
        and personas["impulsive_guesser_telemetry_audit"]["invariants_satisfied"]
        and personas["spaced_repetition_learner"]["invariants_satisfied"]
        and ablation["linucb_outperforms_random"]
        and latencies["all_slas_met"]
    )

    report = {
        "harness_status": "PASSED" if all_passed else "FAILED",
        "execution_time_seconds": round(time.perf_counter() - t0, 3),
        "persona_simulations": personas,
        "linucb_vs_random_ablation": ablation,
        "latency_sla_benchmarks": latencies,
    }

    if save_report:
        with open(HARNESS_REPORT_PATH, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    if not all_passed:
        raise RuntimeError(f"Evaluation harness invariant failed: {json.dumps(report, indent=2)}")

    return report


if __name__ == "__main__":
    rep = run_evaluation_harness(save_report=True)
    print(json.dumps(rep, indent=2))
