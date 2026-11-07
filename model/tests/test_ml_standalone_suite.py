import ast
import csv
import json
import tokenize
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from model.corpus_distribution_and_readiness_audit import (
    format_distribution_markdown_table,
    run_corpus_distribution_and_readiness_audit,
)
from model.dina_cognitive_diagnosis import (
    COGNITIVE_SKILLS,
    compute_dina_and_rt_profile,
    extract_item_q_vector,
)
from model.evaluation_harness import run_evaluation_harness
from model.feature_store_and_drift import DriftAndCalibrationMonitor, InMemoryFeatureStore
from model.importance_calculator import compute_importance
from model.irt_3pl_model import (
    CALIBRATED_ITEMS,
    IRTItemParams,
    compute_mirt_3d_profile,
    fisher_information,
    irt_3pl_probability,
    predict_cold_start_difficulty_and_irt,
    theta_to_percentile,
    update_student_theta_map,
)
from model.knowledge_graph_dag import (
    DAG_NODES,
    compute_hierarchical_shrinkage_prior,
    compute_prerequisite_readiness,
    diagnose_prerequisite_bottlenecks,
    get_all_ancestors,
    propagate_dag_belief_update,
)
from model.linucb_bandit import (
    LinUCBStudentState,
    build_context_vector,
    record_bandit_feedback,
    reset_student_bandit,
)
from model.monte_carlo_conformal_neet import (
    _allocate_180_neet_questions,
    run_monte_carlo_conformal_neet_simulation,
)
from model.rag_socratic_engine import (
    _INDEX as RAG_INDEX,
    _detect_distractor_trap,
    generate_socratic_remediation,
)
from model.rf_compact_runtime import COMPACT_RF_RUNTIME
from model.state_machine.bkt_model import (
    compute_bkt_update,
    compute_ebbinghaus_forgetting_prior,
    compute_retention_urgency,
    resolve_level_with_hysteresis,
)
from model.study_plan_and_coverage_engine import (
    compute_syllabus_coverage_and_improvement_plan,
    format_study_plan_markdown_table,
)
from model.validate_question_bank import validate as validate_qbank
from model.validate_weightage_file import validate as validate_weightage
from model.verify_artifact_integrity import verify_all_ml_artifacts

MODEL_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = MODEL_DIR.parent


def ml_suite_01_question_bank_and_weightage_schema_validators():
    with open(PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json", encoding="utf-8") as f:
        qs = json.load(f)
    q_rep = validate_qbank(qs)
    assert len(qs) == 2400
    assert len(q_rep["issues"]) == 0

    with open(
        PROJECT_ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json",
        encoding="utf-8",
    ) as f:
        w_raw = json.load(f)
    w_rep = validate_weightage(w_raw.get("Unified_Topic_Weightage_And_Marks", w_raw))
    assert len(w_rep["issues"]) == 0


def ml_suite_02_exponential_decay_14_year_weightage_engine():
    scores = compute_importance(
        {"Optics": {2014: 8, 2025: 0}, "Genetics": {2014: 0, 2025: 8}},
        current_year=2026,
        decay=0.85,
    )
    assert scores["Genetics"] > scores["Optics"]
    assert abs(sum(scores.values()) - 100.0) < 0.05

    with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
        prod_imp = json.load(f)
    assert len(prod_imp) == 65
    assert abs(sum(prod_imp.values()) - 100.0) < 0.15


def ml_suite_03_random_forest_champion_and_compact_numpy_2400_parity():
    assert COMPACT_RF_RUNTIME.loaded is True
    df = pd.read_csv(PROJECT_ROOT / "data" / "training" / "question_features.csv")
    assert len(df) == 2400
    pipe = joblib.load(MODEL_DIR / "question_difficulty_model.joblib")
    classes = [str(c) for c in pipe.classes_]
    feature_cols = [
        "subject",
        "chapter",
        "topic_tag",
        "question_type",
        "ideal_time_sec",
        "has_diagram",
        "word_count",
        "char_length",
        "formula_density",
        "numerical_complexity",
        "negation_trap",
        "distractor_similarity",
        "marks",
        "negative_marks",
    ]
    sk_probs = pipe.predict_proba(df[feature_cols])
    for idx, row in enumerate(df.itertuples(index=False)):
        np_p = COMPACT_RF_RUNTIME.predict_proba_single(
            subject=str(row.subject),
            chapter=str(row.chapter),
            topic_tag=str(row.topic_tag),
            question_type=str(row.question_type),
            ideal_time_sec=float(row.ideal_time_sec),
            has_diagram=bool(row.has_diagram),
            word_count=float(row.word_count),
            char_length=float(row.char_length),
            formula_density=float(row.formula_density),
            numerical_complexity=float(row.numerical_complexity),
            negation_trap=float(row.negation_trap),
            distractor_similarity=float(row.distractor_similarity),
            marks=float(row.marks),
            negative_marks=float(row.negative_marks),
        )
        assert np_p is not None
        sk_row = {c: float(p) for c, p in zip(classes, sk_probs[idx])}
        assert max(sk_row, key=sk_row.get) == max(np_p, key=np_p.get)
        for c in classes:
            assert abs(sk_row[c] - np_p[c]) < 1e-3


def ml_suite_04_3pl_irt_fisher_map_and_3d_mirt_profiler():
    assert len(CALIBRATED_ITEMS) == 2400
    item = IRTItemParams(a=1.5, b=0.2, c=0.22)
    assert 0.22 <= irt_3pl_probability(-4.0, item) < irt_3pl_probability(2.5, item) <= 1.0
    assert fisher_information(0.3, item) > fisher_information(3.5, item)
    assert update_student_theta_map(0.0, True, item) > 0.0
    assert update_student_theta_map(0.0, False, item) < 0.0
    assert theta_to_percentile(1.5) > 90.0

    cold = predict_cold_start_difficulty_and_irt(
        "Calculate the electric field E = k*q/r^2 for +4 uC at 0.30 m.",
        ["4e5 N/C", "2e5 N/C", "1e5 N/C", "8e5 N/C"],
        has_diagram=False,
    )
    assert cold["rf_probabilities"] is not None
    assert cold["irt_estimator"] == "multi_output_ridge_npz"
    assert cold["predicted_difficulty"] in ("basic", "medium", "advanced")

    mirt = compute_mirt_3d_profile(
        [
            {
                "question_id": "Q1",
                "question_text": "Calculate the molarity of 2.0 mol in 0.5 L.",
                "question_type": "Numerical",
                "difficulty": "medium",
                "expected_time_seconds": 90,
                "is_correct": True,
            }
        ]
    )
    assert mirt["theta_numerical"] > 0.0


def ml_suite_05_bkt_saint_telemetry_and_ebbinghaus_forgetting():
    p_good, ci_good = compute_bkt_update(0.40, True, "medium", 55, 90, 0, 2000, 3, streak=2)
    p_guess, _ = compute_bkt_update(0.40, True, "medium", 4, 90, 4, 45000, 3, streak=0)
    assert p_good > p_guess + 0.05
    assert 0.0 < ci_good < 0.5

    decayed = compute_ebbinghaus_forgetting_prior(0.88, elapsed_hours=168.0, streak=0)
    assert 0.35 <= decayed < 0.88
    assert compute_retention_urgency(0.88, elapsed_hours=168.0, streak=0) > 0.25
    assert resolve_level_with_hysteresis("basic", 0.78, 0.20) in ("medium", "advanced")


def ml_suite_06_prerequisite_dag_acyclicity_shrinkage_and_propagation():
    from model.state_machine.bkt_model import get_chapter_bkt_prior

    assert len(DAG_NODES) == 65
    for node in DAG_NODES:
        assert node not in get_all_ancestors(node)
    prior = compute_hierarchical_shrinkage_prior(
        "Rotational Motion",
        {"Laws of Motion": 0.82, "Work, Energy & Power": 0.80, "Kinematics": 0.78},
    )
    assert prior > get_chapter_bkt_prior("Rotational Motion") + 0.20
    assert prior > 0.50
    assert compute_prerequisite_readiness("Rotational Motion", {"Laws of Motion": 0.20, "Work, Energy & Power": 0.20}) < 0.50
    bots = diagnose_prerequisite_bottlenecks({"Rotational Motion": 0.22, "Laws of Motion": 0.25})
    assert len(bots) >= 1
    boost = propagate_dag_belief_update("Rotational Motion", True, "advanced", {"Laws of Motion": 0.45})
    assert boost["Laws of Motion"] > 0.45


def ml_suite_07_disjoint_linucb_bandit_zpd_and_regret_bound():
    st = LinUCBStudentState(alpha=0.35)
    x = build_context_vector("NEET2020_PHY_144", "medium", 90, 0.1, 0.45, 3.0, 0.85, None)
    assert x.shape == (6,)
    u_before = float(np.sqrt(x @ st.A_inv @ x))
    st.update(x, 0.82)
    u_after = float(np.sqrt(x @ st.A_inv @ x))
    assert u_after < u_before

    try:
        rew = record_bandit_feedback(
            student_id="ML_SUITE_BANDIT_TMP",
            question_id="NEET2020_PHY_144",
            difficulty="medium",
            expected_time_seconds=90,
            student_theta=0.1,
            topic_mastery_prob=0.55,
            topic_importance=3.0,
            prereq_readiness=0.85,
            pace_flag=None,
            is_correct=True,
            time_taken_seconds=65,
            prior_mastery_prob=0.45,
        )
        assert 0.0 < rew <= 1.0
    finally:
        reset_student_bandit("ML_SUITE_BANDIT_TMP", remove=True)


def ml_suite_08_ncert_hybrid_tfidf_rag_and_scientific_trap_detector():
    assert len(RAG_INDEX.chunks) == 65
    assert "Sign Inversion Trap" in _detect_distractor_trap("-10 cm", "+10 cm")
    assert "SI Unit / Order-of-Magnitude Trap" in _detect_distractor_trap("3.2e-4 T", "3.2e-7 T")
    rem = generate_socratic_remediation(
        question_id="NEET_RAG_TEST",
        topic_id="Electrostatics",
        question_text="Two charges +q and -q separated by 2a form an electric dipole.",
        options=["-2kp/r^3", "+2kp/r^3", "kp/r^2", "0"],
        correct_answer="a",
        selected_option="b",
    )
    assert "Electrostatics" in rem["topic_id"]
    assert len(rem["socratic_guiding_questions"]) >= 1


def ml_suite_09_dina_8_skill_q_matrix_and_lognormal_response_time():
    assert len(COGNITIVE_SKILLS) == 8
    qv = extract_item_q_vector("Find pH and log concentration in mol/L for buffer equilibrium.", "Equilibrium", "medium")
    assert sum(qv.values()) >= 2
    prof = compute_dina_and_rt_profile(
        [
            {
                "question_id": "Q1",
                "question_text": "Calculate molarity in mol/L.",
                "topic_id": "Mole Concept / Stoichiometry",
                "difficulty": "basic",
                "expected_time_seconds": 90,
                "time_taken_seconds": 40,
                "is_correct": True,
            },
            {
                "question_id": "Q2",
                "question_text": "Which statement is NOT correct about lanthanoids?",
                "topic_id": "d- and f-Block Elements",
                "difficulty": "advanced",
                "expected_time_seconds": 90,
                "time_taken_seconds": 7,
                "is_correct": False,
            },
        ]
    )
    assert len(prof["skills"]) == 8
    assert prof["speed_ability_profile"]["behavior_classification"]["fluent_mastery"] == 1
    assert prof["speed_ability_profile"]["behavior_classification"]["rapid_guess"] == 1


def ml_suite_10_monte_carlo_conformal_180q_and_graph_laplacian_leverage():
    with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
        imp = json.load(f)
    alloc = _allocate_180_neet_questions(imp)
    assert sum(c for _, _, c in alloc) == 180
    sim = run_monte_carlo_conformal_neet_simulation(
        student_id="ML_SUITE_MC",
        global_theta=0.25,
        mastery_map={k: 0.48 for k in imp},
        theta_map={k: 0.25 for k in imp},
        ci_map={k: 0.28 for k in imp},
        topic_importance=imp,
        n_trials=300,
    )
    lo, hi = sim["conformal_90_interval"]
    assert lo <= sim["expected_score_attempt_all_720"] <= hi
    assert len(sim["high_yield_leverage_chapters"]) == 6


def ml_suite_11_corpus_65_topic_distribution_and_180q_readiness_audit():
    rep = run_corpus_distribution_and_readiness_audit(save_report=True)
    assert rep["audit_status"] == "PASSED"
    assert rep["corpus_totals"]["total_verified_questions"] == 2400
    assert rep["corpus_totals"]["total_canonical_chapters"] == 65
    assert rep["subject_difficulty_matrix"]["Physics"]["basic"] > 100
    assert rep["subject_difficulty_matrix"]["Physics"]["medium"] > 200
    assert rep["subject_difficulty_matrix"]["Physics"]["advanced"] > 100
    assert rep["gap_summary"]["topics_missing_basic_count"] == 0
    assert rep["gap_summary"]["topics_missing_medium_count"] == 0
    assert rep["gap_summary"]["topics_missing_advanced_count"] == 0
    assert rep["gap_summary"]["all_65_topics_have_all_three_tiers"] is True
    assert rep["gap_summary"]["adjacent_tier_fallback_resolves_100_percent_topics"] is True
    md = format_distribution_markdown_table(rep)
    assert "| **Physics** |" in md and "| **Total** |" in md


def ml_suite_12_syllabus_coverage_less_learned_and_improvement_plan_engine():
    plan = compute_syllabus_coverage_and_improvement_plan(
        student_id="ML_SUITE_PLAN",
        global_theta=-0.20,
        topic_states=[
            {
                "topic_id": "Rotational Motion",
                "level": "basic",
                "mastery_prob": 0.20,
                "irt_theta": -0.70,
                "pace_flag": "slow",
                "elapsed_hours": 10.0,
                "streak": 0,
            },
            {
                "topic_id": "Laws of Motion",
                "level": "basic",
                "mastery_prob": 0.26,
                "irt_theta": -0.50,
                "pace_flag": None,
                "elapsed_hours": 24.0,
                "streak": 0,
            },
            {
                "topic_id": "Units & Dimensions",
                "level": "basic",
                "mastery_prob": 0.22,
                "irt_theta": -0.55,
                "pace_flag": "slow",
                "elapsed_hours": 8.0,
                "streak": 0,
            },
            {
                "topic_id": "Electrostatics",
                "level": "advanced",
                "mastery_prob": 0.89,
                "irt_theta": 1.35,
                "pace_flag": "fast",
                "elapsed_hours": 170.0,
                "streak": 1,
            },
        ],
    )
    cov = plan["syllabus_coverage"]
    assert cov["total_chapters"] == 65
    assert cov["practiced_chapters"] == 4
    assert cov["less_learned_chapters"] >= 3
    assert len(plan["less_learned_lessons"]) >= 2
    assert len(plan["improvement_action_table"]) >= 5
    cats = {r["diagnostic_category"] for r in plan["improvement_action_table"]}
    assert "PREREQUISITE_BOTTLENECK" in cats
    assert "LESS_LEARNED_CRITICAL" in cats
    md = format_study_plan_markdown_table(plan)
    assert "Rotational Motion" in md and "PREREQUISITE_BOTTLENECK" in md


def ml_suite_13_feature_store_latency_and_psi_ece_brier_drift_monitor():
    fs = InMemoryFeatureStore()
    fs.put_student_features("S_ML", {"theta": 0.4})
    assert fs.get_student_features("S_ML")["theta"] == 0.4
    assert fs.get_latency_percentiles()["p95_ms"] < 5.0

    dm = DriftAndCalibrationMonitor()
    healthy = dm.evaluate_health(
        [{"predicted_prob": 0.72, "correct": True, "difficulty": "medium", "time_taken": 78} for _ in range(30)]
    )
    assert "effective_psi" in healthy and "brier_score" in healthy
    skewed = dm.evaluate_health(
        [{"predicted_prob": 0.92, "correct": False, "difficulty": "advanced", "time_taken": 160} for _ in range(35)]
    )
    assert skewed["drift_status"] == "DRIFT_WARNING"
    assert skewed["recalibration_required"] is True


def ml_suite_14_multi_model_tournament_and_sha256_registry_integrity():
    har = run_evaluation_harness(save_report=True)
    assert har["harness_status"] == "PASSED"
    assert "compact_rf_300_tree_inference" in har["latency_sla_benchmarks"]
    assert "study_plan_coverage_prescription" in har["latency_sla_benchmarks"]

    integrity = verify_all_ml_artifacts()
    assert integrity["integrity_status"] == "VERIFIED"
    assert integrity["verified_artifacts_count"] == 21


def ml_suite_15_clean_code_zero_comments_and_zero_seeded_attempts_gate():
    with open(PROJECT_ROOT / "data" / "training" / "student_attempts.csv", encoding="utf-8") as f:
        assert len(list(csv.DictReader(f))) == 0

    for py_file in MODEL_DIR.rglob("*.py"):
        with tokenize.open(py_file) as tf:
            for tok in tokenize.generate_tokens(tf.readline):
                assert tok.type != tokenize.COMMENT, f"Comment token in {py_file}:{tok.start[0]}"
        tree = ast.parse(py_file.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                assert ast.get_docstring(node) is None, f"Docstring in {py_file}"


def ml_suite_16_irt_tier_recalibration_audit():
    from model.propose_tier_recalibration import run_irt_tier_recalibration_audit

    rep = run_irt_tier_recalibration_audit(save_report=False)
    assert rep["source_corpus_mutated"] is False
    assert rep["total_audited_questions"] == 2400
    gap = rep["gap_resolution_summary"]
    assert gap["chapters_missing_basic_before"] == 0
    assert gap["chapters_resolvable_via_irt_anchors"] == 0
    assert gap["physics_chapters_missing_basic_before"] == 0
    assert gap["physics_chapters_resolvable_via_irt_anchors"] == 0
    assert gap["strict_global_basic_candidates_count"] > 200


def ml_suite_17_near_duplicate_pyq_cosine_similarity_audit():
    from model.question_similarity_audit import get_near_duplicate_siblings_map, run_question_similarity_audit

    rep = run_question_similarity_audit(save_report=False)
    assert rep["total_audited_questions"] == 2400
    assert rep["unique_effective_question_clusters"] > 2300
    sibs = get_near_duplicate_siblings_map()
    assert isinstance(sibs, dict)
    for qid, s_list in sibs.items():
        assert qid not in s_list
        for s_id in s_list:
            assert qid in sibs.get(s_id, [])


def ml_suite_18_temperature_probability_calibration_and_compression():
    from model.probability_calibrator import calibrate_rf_probabilities, get_calibration_metadata

    meta = get_calibration_metadata()
    assert 0.08 <= meta["optimal_temperature"] <= 3.0
    assert meta["post_calibration"]["multiclass_nll"] <= meta["pre_calibration"]["multiclass_nll"] + 1e-4
    comp = meta["serverless_compression_benchmark"]
    assert comp["compression_ratio_x"] >= 15.0
    cal_p, temp = calibrate_rf_probabilities({"basic": 0.05, "medium": 0.75, "advanced": 0.20})
    assert cal_p is not None and abs(sum(cal_p.values()) - 1.0) < 0.01
    assert temp == meta["optimal_temperature"]


def ml_suite_19_all_trained_auxiliary_models_and_chapter_datasets():
    from model.dina_cognitive_diagnosis import DINA_CALIBRATED_SKILLS
    from model.state_machine.bkt_model import CHAPTER_BKT_REGISTRY

    ch_df = pd.read_csv(PROJECT_ROOT / "data" / "training" / "chapter_wise_features.csv")
    assert ch_df.shape == (65, 18)
    assert int((ch_df["basic_pyqs"] == 0).sum()) == 0
    assert int((ch_df["medium_pyqs"] == 0).sum()) == 0
    assert int((ch_df["advanced_pyqs"] == 0).sum()) == 0

    with open(PROJECT_ROOT / "data" / "taxonomy" / "chapter_wise_neet_pyq_summary.json", encoding="utf-8") as f:
        ch_sum = json.load(f)
    assert ch_sum["total_canonical_chapters"] == 65
    assert ch_sum["total_verified_pyqs"] == 2400
    assert ch_sum["chapters_missing_basic"] == 0
    assert ch_sum["chapters_missing_medium"] == 0
    assert ch_sum["chapters_missing_advanced"] == 0

    assert len(CHAPTER_BKT_REGISTRY) == 65
    bkt_priors = {round(float(v["p_init"]), 4) for v in CHAPTER_BKT_REGISTRY.values()}
    assert len(bkt_priors) >= 15

    assert len(DINA_CALIBRATED_SKILLS) == 8
    cold_dina = compute_dina_and_rt_profile([])
    assert all(s["status"] == "prior_calibrated" for s in cold_dina["skills"])
    assert all(s["corpus_pyq_count"] > 0 for s in cold_dina["skills"])
    assert len({s["mastery_probability"] for s in cold_dina["skills"]}) == 8

    assert RAG_INDEX.svd_loaded is True
    assert (MODEL_DIR / "irt_cold_start_ridge.npz").exists()
    assert (MODEL_DIR / "linucb_prior_init.json").exists()
    assert (MODEL_DIR / "conformal_laplacian_model.json").exists()


if __name__ == "__main__":
    suites = [
        ml_suite_01_question_bank_and_weightage_schema_validators,
        ml_suite_02_exponential_decay_14_year_weightage_engine,
        ml_suite_03_random_forest_champion_and_compact_numpy_2400_parity,
        ml_suite_04_3pl_irt_fisher_map_and_3d_mirt_profiler,
        ml_suite_05_bkt_saint_telemetry_and_ebbinghaus_forgetting,
        ml_suite_06_prerequisite_dag_acyclicity_shrinkage_and_propagation,
        ml_suite_07_disjoint_linucb_bandit_zpd_and_regret_bound,
        ml_suite_08_ncert_hybrid_tfidf_rag_and_scientific_trap_detector,
        ml_suite_09_dina_8_skill_q_matrix_and_lognormal_response_time,
        ml_suite_10_monte_carlo_conformal_180q_and_graph_laplacian_leverage,
        ml_suite_11_corpus_65_topic_distribution_and_180q_readiness_audit,
        ml_suite_12_syllabus_coverage_less_learned_and_improvement_plan_engine,
        ml_suite_13_feature_store_latency_and_psi_ece_brier_drift_monitor,
        ml_suite_14_multi_model_tournament_and_sha256_registry_integrity,
        ml_suite_15_clean_code_zero_comments_and_zero_seeded_attempts_gate,
        ml_suite_16_irt_tier_recalibration_audit,
        ml_suite_17_near_duplicate_pyq_cosine_similarity_audit,
        ml_suite_18_temperature_probability_calibration_and_compression,
        ml_suite_19_all_trained_auxiliary_models_and_chapter_datasets,
    ]
    for fn in suites:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(suites)} STANDALONE PURE-ML & MLOPS SUITES PASSED (100%)!")
