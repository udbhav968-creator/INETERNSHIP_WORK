import ast
import csv
import json
import math
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
from model.quarantine_provenance_audit import audit_quarantine_expansion_readiness, get_quarantine_item_by_id
from model.response_data_pipeline import recalibrate_eligible_empirical_items
from model.rag_socratic_engine import query_socratic_qa_with_fallback

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


def ml_suite_03_random_forest_champion_and_compact_numpy_parity():
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
            assert abs(sk_row[c] - np_p[c]) < 1.5e-2


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
    rep = run_corpus_distribution_and_readiness_audit(save_report=False)
    assert rep["audit_status"] in ("PASSED", "WARNING")
    assert rep["corpus_totals"]["total_verified_questions"] == 2400
    assert rep["corpus_totals"]["total_canonical_chapters"] == 65
    assert rep["subject_difficulty_matrix"]["Physics"]["basic"] > 100
    assert rep["subject_difficulty_matrix"]["Physics"]["medium"] > 200
    assert rep["subject_difficulty_matrix"]["Physics"]["advanced"] > 100
    assert rep["gap_summary"]["topics_missing_basic_count"] == 3
    assert rep["gap_summary"]["topics_missing_medium_count"] == 1
    assert rep["gap_summary"]["topics_missing_advanced_count"] == 1
    assert rep["gap_summary"]["topics_with_any_zero_tier_count"] == 5
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
    har = run_evaluation_harness(save_report=False)
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
    assert gap["chapters_missing_basic_before"] == 3
    assert gap["chapters_resolvable_via_irt_anchors"] == 3
    assert gap["physics_chapters_missing_basic_before"] == 2
    assert gap["physics_chapters_resolvable_via_irt_anchors"] == 2
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
    assert int((ch_df["basic_pyqs"] == 0).sum()) == 3
    assert int((ch_df["medium_pyqs"] == 0).sum()) == 1
    assert int((ch_df["advanced_pyqs"] == 0).sum()) == 1

    with open(PROJECT_ROOT / "data" / "taxonomy" / "chapter_wise_neet_pyq_summary.json", encoding="utf-8") as f:
        ch_sum = json.load(f)
    assert ch_sum["total_canonical_chapters"] == 65
    assert ch_sum["total_verified_pyqs"] == 2400
    assert ch_sum["chapters_missing_basic"] == 3
    assert ch_sum["chapters_missing_medium"] == 1
    assert ch_sum["chapters_missing_advanced"] == 1

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


def ml_suite_20_pipeline_deep_invariants_and_statistical_drift_gate():
    from model.feature_store_and_drift import DriftAndCalibrationMonitor
    from model.irt_3pl_model import IRTItemParams, fisher_information, irt_3pl_probability
    from model.state_machine.bkt_model import compute_bkt_update

    item = IRTItemParams(a=1.65, b=0.45, c=0.20)
    p_low = irt_3pl_probability(-6.0, item)
    p_mid = irt_3pl_probability(0.45, item)
    p_high = irt_3pl_probability(6.0, item)
    assert 0.20 <= p_low <= 0.205
    assert p_low < p_mid < p_high <= 1.0
    assert fisher_information(0.45, item) > 0.0
    assert fisher_information(0.45, item) > fisher_information(5.0, item)

    p_streak, ci_streak = compute_bkt_update(0.90, True, "advanced", 45, 90, 0, 1000, 3, streak=25)
    assert 0.90 < p_streak < 1.0
    assert 0.0 < ci_streak < 0.25

    mon = DriftAndCalibrationMonitor()
    mon.log_prediction(0.75, 1, "medium", 82)
    mon.log_prediction(0.45, 0, "advanced", 115)
    mon.log_prediction(0.85, 1, "basic", 58)
    mon.log_prediction(0.25, 0, "advanced", 140)
    health = mon.evaluate_health()
    assert health["sample_size"] == 4
    assert health["wasserstein_response_time"] >= 0.0
    assert 0.0 <= health["ks_response_time_statistic"] <= 1.0
    assert 0.0 <= health["brier_score"] <= 1.0
    assert 0.0 <= health["expected_calibration_error"] <= 1.0


def ml_suite_21_cognitive_entropy_kl_information_and_sem_stopping_gate():
    from model.dina_cognitive_diagnosis import compute_dina_and_rt_profile
    from model.irt_3pl_model import IRTItemParams, compute_posterior_sem, kullback_leibler_item_information
    from model.rag_socratic_engine import classify_distractor_trap_taxonomy

    diag = compute_dina_and_rt_profile([])
    assert "map_attribute_vector" in diag
    assert len(diag["map_attribute_vector"]) == 8
    assert all(ch in ("0", "1") for ch in diag["map_attribute_vector"])
    assert 0.0 <= diag["cognitive_entropy_bits"] <= 8.0
    assert 0.0 <= diag["cognitive_profile_certainty_percent"] <= 100.0

    items_2 = [IRTItemParams(a=1.2, b=0.0, c=0.2), IRTItemParams(a=1.5, b=0.5, c=0.2)]
    sem_2 = compute_posterior_sem(0.0, items_2)
    items_10 = items_2 * 5
    sem_10 = compute_posterior_sem(0.0, items_10)
    assert sem_10["sem"] < sem_2["sem"]
    assert sem_10["test_information"] > sem_2["test_information"]
    assert sem_2["precision_status"] in ("EXPLORATORY", "MODERATE", "HIGH")
    assert sem_2["ci_95_bounds"][0] < 0.0 < sem_2["ci_95_bounds"][1]

    informative_item = IRTItemParams(a=2.0, b=0.2, c=0.2)
    uninformative_item = IRTItemParams(a=0.3, b=0.2, c=0.2)
    kl_info = kullback_leibler_item_information(0.2, informative_item)
    kl_uninfo = kullback_leibler_item_information(0.2, uninformative_item)
    assert kl_info > kl_uninfo >= 0.0

    assert classify_distractor_trap_taxonomy("-5.2 x 10^6", "5.2 x 10^6") == "SIGN_INVERSION"
    assert classify_distractor_trap_taxonomy("5.2 x 10^3", "5.2 x 10^6") == "ORDER_OF_MAGNITUDE_UNIT"
    assert classify_distractor_trap_taxonomy("0.5", "2.0") == "RECIPROCAL_INVERSION"
    assert classify_distractor_trap_taxonomy("Which statement is NOT correct", "Which statement is correct") == "NEGATION_LOGIC"


def ml_suite_22_national_rank_and_college_cutoff_simulation():
    from model.national_rank_simulator import (
        score_to_neet_percentile,
        score_to_all_india_rank,
        simulate_national_rank_and_college_cutoffs,
    )
    import numpy as np

    p_720 = score_to_neet_percentile(720)
    p_650 = score_to_neet_percentile(650)
    p_500 = score_to_neet_percentile(500)
    p_165 = score_to_neet_percentile(165)
    assert p_720 == 100.0
    assert p_650 >= 99.5
    assert p_650 > p_500 > p_165 > 0.0

    rk_720 = score_to_all_india_rank(720)
    rk_650 = score_to_all_india_rank(650)
    rk_165 = score_to_all_india_rank(165)
    assert rk_720 == 1
    assert 1 < rk_650 < 15000
    assert rk_165 > 1000000

    sim = simulate_national_rank_and_college_cutoffs(
        total_scores_trials=np.array([350, 360, 380, 400, 420], dtype=np.float64),
        conformal_score_interval=[320, 410],
        mean_score=380,
    )
    assert sim["total_national_candidates"] == 1999895
    assert sim["model_assumptions"]["score_to_percentile_curve"]["status"] == "approximate"
    assert sim["conformal_air_interval"][0] < sim["conformal_air_interval"][1]
    assert len(sim["college_admission_probabilities"]) >= 7
    assert any(col["college_id"] == "aiims_delhi" for col in sim["college_admission_probabilities"])


def ml_suite_23_online_streaming_calibration_and_continuous_retraining():
    from model.streaming_online_calibration import STREAMING_CALIBRATION_ENGINE
    from model.continuous_retraining_engine import evaluate_drift_and_retrain_if_needed

    new_a, new_b, new_c = STREAMING_CALIBRATION_ENGINE.update_item_parameters_online(
        question_id="test_q_stream",
        student_theta=0.0,
        is_correct=False,
        current_a=1.5,
        current_b=0.0,
        current_c=0.2,
    )
    assert new_b > 0.0
    assert 0.8 <= new_a <= 2.5

    trans = STREAMING_CALIBRATION_ENGINE.update_graph_edge_transfer(
        parent_topic="Mole Concept / Stoichiometry",
        child_topic="Biomolecules",
        parent_mastery=0.85,
        child_success=True,
    )
    assert 0.05 <= trans <= 0.95

    elbo = STREAMING_CALIBRATION_ENGINE.record_online_log_joint([0.8, 0.9], [0.5, 0.5])
    assert isinstance(elbo, float)
    assert abs(elbo - (math.log(0.8) + math.log(0.9) + 2 * math.log(0.5))) < 1e-3

    from model.continuous_retraining_engine import DRIFT_THRESHOLDS
    base_brier = DRIFT_THRESHOLDS["baseline_brier"]
    base_ece = DRIFT_THRESHOLDS["baseline_ece"]
    assert base_brier is not None and base_ece is not None
    assert DRIFT_THRESHOLDS["max_brier"] > base_brier

    drift_report = evaluate_drift_and_retrain_if_needed(
        current_psi=0.02,
        current_brier=base_brier,
        current_ece=base_ece,
        force_retrain=False,
        sample_size=500,
    )
    assert drift_report["drift_detected"] is False
    assert drift_report["retrain_triggered"] is False
    assert drift_report["action_taken"] == "NOOP_STABLE"

    small_report = evaluate_drift_and_retrain_if_needed(
        current_psi=0.9,
        current_brier=base_brier * 3,
        current_ece=base_ece * 3,
        force_retrain=False,
        sample_size=5,
    )
    assert small_report["action_taken"] == "INSUFFICIENT_DATA"
    assert small_report["retrain_triggered"] is False

    live_report = evaluate_drift_and_retrain_if_needed(force_retrain=False)
    assert live_report["metrics_source"] == "live_drift_monitor"
    assert live_report["retrain_triggered"] is False


def ml_suite_24_multilingual_hindi_glossary_and_socratic_audio_script():
    from model.multilingual_glossary import get_chapter_multilingual_metadata, translate_distractor_trap_to_hindi
    from model.socratic_audio_script_generator import generate_socratic_audio_script

    mole_meta = get_chapter_multilingual_metadata("Mole Concept / Stoichiometry")
    assert "मोल" in mole_meta["chapter_hi"]
    assert len(mole_meta["core_terms"]) > 0

    trap_hi = translate_distractor_trap_to_hindi("SIGN_INVERSION")
    assert "चिह्न" in trap_hi["label_hi"]

    audio = generate_socratic_audio_script(
        question_stem="Calculate the change in internal energy...",
        selected_option="-120 J",
        correct_option="+120 J",
        chapter_name="Mole Concept / Stoichiometry",
        distractor_trap="SIGN_INVERSION",
        language="en",
    )
    assert audio["duration_seconds_est"] >= 4.0
    assert "<speak>" in audio["ssml_payload"]
    assert "</speak>" in audio["ssml_payload"]




def ml_suite_25_section4_empirical_recalibration_and_quarantine_audit():
    recal = recalibrate_eligible_empirical_items()
    assert recal["status"] in ("SUCCESS", "ERROR")
    assert recal["threshold_responses"] == 30

    audit = audit_quarantine_expansion_readiness()
    assert audit["status"] == "SUCCESS"
    assert audit["total_quarantined_items"] == 815
    assert audit["ready_for_sme_review_count"] == 815
    assert audit["structurally_defective_count"] == 0
    assert audit["safe_to_merge_unconditionally"] is False
    assert (
        audit["provenance_policy"]
        == "QUARANTINED_ITEMS_REQUIRE_OFFICIAL_NTA_SESSION_SOURCE_OR_DOUBLE_BLIND_SME_SIGNOFF_BEFORE_MERGE"
    )

    item = get_quarantine_item_by_id(audit["sme_candidates_preview"][0])
    assert item is not None
    assert item["question_id"] == audit["sme_candidates_preview"][0]

    qa_res = query_socratic_qa_with_fallback(
        "Calculate molar mass of water",
        "What is atomic weight of H and O?",
        "Mole Concept / Stoichiometry",
    )
    assert qa_res["engine_mode"] == "NCERT_DETERMINISTIC_LSA_FALLBACK"
    assert "ncert_reference" in qa_res

def ml_suite_26_irt_equated_180q_initial_assessment_forms():
    from model.assessment_blueprint import (
        MAX_MARKS,
        SUBJECT_QUOTAS,
        build_equated_forms,
        form_index_for_student,
        load_forms,
        load_question_bank,
        score_initial_assessment,
    )

    bank = load_question_bank()
    out = build_equated_forms(bank=bank)
    rep = out["report"]
    assert rep["all_checks_passed"] is True, rep["checks"]
    assert rep["max_pairwise_overlap"] == 0
    assert rep["max_expected_marks_gap_across_theta"] <= 6.0
    assert rep["max_test_information_ratio_gap"] <= 0.05
    for f in rep["forms"]:
        assert f["n_items"] == 180
        assert f["by_subject"] == SUBJECT_QUOTAS
        assert f["sem_at_theta0"] <= 0.25

    stored = load_forms()
    assert stored["forms"] == out["forms"]["forms"]
    assert stored["max_marks"] == MAX_MARKS == 720
    sections = [q for q in stored["forms"][0]["question_ids"]]
    by_id = {q["question_id"]: q for q in bank}
    subj_seq = [by_id[qid]["subject"] for qid in sections]
    assert subj_seq == ["Physics"] * 45 + ["Chemistry"] * 45 + ["Biology"] * 90

    assert form_index_for_student("student-x") == form_index_for_student("student-x")
    assert {form_index_for_student(f"s{i}") for i in range(200)} == {0, 1, 2}

    ids = stored["forms"][0]["question_ids"]
    strong = score_initial_assessment([{"question_id": q, "is_correct": True} for q in ids], bank=bank)
    weak = score_initial_assessment([{"question_id": q, "is_correct": False} for q in ids], bank=bank)
    assert strong["marks"] == 720 and weak["marks"] == -180
    import random as _random
    from model.irt_3pl_model import D_SCALING as _D

    estimates = []
    for true_theta in (-1.5, 0.0, 1.0, 2.0):
        rng = _random.Random(7)
        sim = []
        for qid in ids:
            it = by_id[qid]
            p = it["c"] + (1.0 - it["c"]) / (1.0 + math.exp(-_D * it["a"] * (true_theta - it["b"])))
            sim.append({"question_id": qid, "is_correct": rng.random() < p})
        res = score_initial_assessment(sim, bank=bank)
        assert abs(res["theta_eap"] - true_theta) < 0.4, (true_theta, res["theta_eap"])
        assert res["theta_sem"] < 0.25
        assert sum(res["subject_marks"].values()) == res["marks"]
        estimates.append(res["theta_eap"])
    assert weak["theta_eap"] < estimates[0] < estimates[1] < estimates[2] < estimates[3] < strong["theta_eap"]


def ml_suite_27_sample_paper_ingestion_validation_and_cold_start_linking():
    import tempfile
    from model.chapter_resolver import ChapterResolver
    from model.sample_paper_ingestion import LINKING_PATH, ingest, load_accepted_sample_rows

    linking = json.loads(LINKING_PATH.read_text(encoding="utf-8"))
    assert linking["anchor_items"] == 2400
    assert linking["linked_b_rmse"] < linking["raw_b_rmse"]
    assert linking["linked_b_correlation"] > 0.9

    pyq = json.loads((PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json").read_text(encoding="utf-8"))
    resolver = ChapterResolver(pyq)
    assert resolver.label_route("Physics", "Motion in a Plane") == ("alias", "Kinematics")
    assert resolver.label_route("Biology", "Body Fluids and Circulation") == ("alias", "Circulation & Excretion")
    assert resolver.label_route("Chemistry", "Some Basic Concepts of Chemistry") == ("alias", "Mole Concept / Stoichiometry")
    assert resolver.label_route("Biology", "General Biology")[0] == "generic"
    assert resolver.label_route("Biology", "Motion in a Straight Line")[0] == "subject_conflict"
    resolver.fit([])
    for subj, m in resolver.metrics.items():
        if m["confidence_threshold"] < 1.0:
            assert m["holdout_precision_at_threshold"] >= 0.90

    rows = []
    for i, r in enumerate(pyq[:24]):
        x = dict(r)
        x["question_id"] = f"SP001_{i:03d}"
        x["paper_source"] = "Sample Paper 1"
        x["question_text"] = r["question_text"] + f" [sample variant {i}]"
        x["has_diagram"] = False
        x["image_asset"] = ""
        rows.append(x)
    rows[0]["chapter"] = rows[0]["chapter"].replace("&", "and")
    bad = [dict(rows[1], question_id="SP001_BAD1", verification_status="Pending", question_text="unverified item text"),
           dict(rows[2], question_id="SP001_BAD2", subject="Biology", chapter="Kinematics", question_text="conflicting subject text"),
           dict(rows[3], question_id="SP001_BAD3", has_diagram=True, image_asset="assets/diagrams/missing.png", question_text="missing diagram text"),
           dict(pyq[5], question_id="SP001_BAD4"),
           dict(rows[4], question_id="SP001_004"),
           dict(rows[6], question_id="SP001_BAD6", correct_option="e", question_text="bad option text")]
    with tempfile.TemporaryDirectory() as td:
        tdp = Path(td)
        (tdp / "assets" / "diagrams").mkdir(parents=True)
        (tdp / "assets" / "diagrams" / "ok.png").write_bytes(b"png")
        rows[7]["has_diagram"] = True
        rows[7]["image_asset"] = "assets/diagrams/ok.png"
        src = tdp / "samples.json"
        src.write_text(json.dumps(rows + bad), encoding="utf-8")
        kw = dict(sample_path=src, params_path=tdp / "p.json", report_path=tdp / "r.json", rejected_path=tdp / "x.json", assets_root=tdp, resolver=resolver)
        rep = ingest(**kw)
        assert rep["status"] == "ingested"
        assert rep["accepted"] == 24 and rep["rejected"] == 6
        reasons = rep["rejection_reasons"]
        for key in ("not_verified", "subject_chapter_conflict", "diagram_asset_missing", "duplicate_of_pyq", "duplicate_id", "bad_correct_option"):
            assert reasons.get(key, 0) >= 1, (key, reasons)
        params = json.loads((tdp / "p.json").read_text(encoding="utf-8"))["items"]
        assert params["SP001_000"]["chapter"] == pyq[0]["chapter"]
        for v in params.values():
            assert 0.5 <= v["a"] <= 2.5 and -3.0 <= v["b"] <= 3.0 and 0.10 <= v["c"] <= 0.30
        rep2 = ingest(**kw)
        assert rep2["cold_start_recomputed"] == 0 and rep2["cold_start_cached"] == 24
        accepted_rows, _ = load_accepted_sample_rows(src, tdp / "p.json")
        assert len(accepted_rows) == 24 and all(r["source_type"] == "sample_paper" for r in accepted_rows)
        missing = ingest(sample_path=tdp / "nope.json", report_path=tdp / "r2.json")
        assert missing["status"] == "dataset_not_present"

    real = json.loads((MODEL_DIR / "sample_paper_ingestion_report.json").read_text(encoding="utf-8"))
    if real.get("status") == "ingested":
        assert real["rows_in_file"] == real["accepted"] + real["rejected"]
        assert set(real["by_subject"]) == {"Physics", "Chemistry", "Biology"}
        sp = json.loads((MODEL_DIR / "sample_paper_irt_parameters.json").read_text(encoding="utf-8"))
        assert sp["item_count"] == real["accepted"]


def ml_suite_28_equated_unlimited_mock_180_generation():
    from model.mock_test_generator import generate_mock_180, reserved_initial_assessment_ids

    reserved = reserved_initial_assessment_ids()
    assert len(reserved) == 540
    seen: set[str] = set()
    for attempt in range(6):
        m = generate_mock_180("suite-student", attempt, seen, source="pyq")
        ids = m["question_ids"]
        assert len(ids) == 180 and len(set(ids)) == 180
        assert m["composition"]["by_subject"] == {"Physics": 45, "Chemistry": 45, "Biology": 90}
        assert not (set(ids) & reserved)
        assert m["equivalence_to_initial_assessment"]["within_tolerance"] is True
        if attempt == 0:
            assert m["repeated_items"] == 0
        seen |= set(ids)
    again = generate_mock_180("suite-student", 0, set(), source="pyq")["question_ids"]
    assert again == generate_mock_180("suite-student", 0, set(), source="pyq")["question_ids"]
    assert again != generate_mock_180("suite-student-2", 0, set(), source="pyq")["question_ids"]


def _simulate(ids, by_id, theta, seed, skip=0.0):
    import random as _random
    from model.irt_3pl_model import D_SCALING as _D

    rng = _random.Random(seed)
    out = []
    for qid in ids:
        it = by_id[qid]
        if rng.random() < skip:
            continue
        p = it["c"] + (1.0 - it["c"]) / (1.0 + math.exp(-_D * it["a"] * (theta - it["b"])))
        out.append({"question_id": qid, "is_correct": rng.random() < p, "time_taken_seconds": it["ideal_time_sec"] * rng.uniform(0.5, 1.6)})
    return out


def ml_suite_29_initial_assessment_profile_initialization():
    from model.assessment_blueprint import load_forms, load_question_bank
    from model.student_profile_initializer import initialize_student_profile

    by_id = {q["question_id"]: q for q in load_question_bank()}
    forms = load_forms()["forms"]
    profiles = []
    hits = 0
    trials = 0
    for k, theta in enumerate((-1.5, -0.5, 0.5, 1.5)):
        pr = initialize_student_profile(_simulate(forms[1]["question_ids"], by_id, theta, 11 + k))
        assert abs(pr["global_theta"] - theta) < 0.4
        assert len(pr["topic_states"]) == 65
        for s in pr["topic_states"]:
            assert s["level"] in ("basic", "medium", "advanced")
            if s["observed"]:
                assert s["spec_v1_initial_level"] == ("medium" if s["spec_v1_accuracy"] >= 0.7 else "basic")
                assert s["spec_v1_flag_for_extra_support"] == (s["spec_v1_accuracy"] < 0.4)
            assert 0.05 <= s["mastery_prob"] <= 0.95
            assert 0.0 < s["confidence_interval"] <= 0.45
        assert len(pr["priority_chapters"]) == 10
        profiles.append(pr)
        for rep in range(3):
            nxt = _simulate(forms[2]["question_ids"], by_id, theta, 100 + 7 * k + rep)
            marks = sum(4 if r["is_correct"] else -1 for r in nxt)
            lo, hi = pr["predicted_neet_score"]["interval_90"]
            hits += lo <= marks <= hi
            trials += 1
    means = [float(np.mean([s["mastery_prob"] for s in p["topic_states"]])) for p in profiles]
    assert means == sorted(means)
    adv = [sum(s["level"] == "advanced" for s in p["topic_states"]) for p in profiles]
    assert adv == sorted(adv)
    assert hits / trials >= 0.75


def ml_suite_30_test_analytics_and_equated_score_trend():
    from model.assessment_blueprint import load_forms, load_question_bank
    from model.test_analytics_engine import SKIP_THRESHOLD, analyze_test, score_trend

    assert abs(SKIP_THRESHOLD - 0.2) < 1e-9
    by_id = {q["question_id"]: q for q in load_question_bank()}
    forms = load_forms()["forms"]
    tests = []
    for k, theta in enumerate((-0.6, 0.0, 0.6)):
        ids = forms[k]["question_ids"]
        res = _simulate(ids, by_id, theta, 40 + k, skip=0.1)
        a = analyze_test(ids, res, test_type="mock_180")
        s = a["summary"]
        assert s["total"] == 180 and s["attempted"] == len(res)
        assert s["attempted"] + s["unattempted"] == 180
        assert s["marks"] == 4 * s["correct"] - s["incorrect"]
        assert sum(v["marks"] for v in a["by_subject"].values()) == s["marks"]
        assert sum(v["total"] for v in a["by_chapter"].values()) == 180
        assert len(a["questions"]) == 180
        assert a["negative_marking"]["avoidable_marks_lost"] <= a["negative_marking"]["marks_lost_to_wrong_answers"]
        tests.append({"theta": a["theta"], "theta_sem": a["theta_sem"], "marks": s["marks"], "test_type": "mock_180"})
    tr = score_trend(tests)
    assert tr["trend"]["direction"] == "improving"
    assert tr["trend"]["change_is_significant"] is True
    eq = [p["equated_marks"] for p in tr["points"]]
    assert eq == sorted(eq)
    assert score_trend(tests[:1])["trend"]["direction"] == "insufficient_tests"


def ml_suite_31_difficulty_leakage_audit_and_v2_candidate():
    rep = json.loads((MODEL_DIR / "difficulty_model_leakage_audit.json").read_text(encoding="utf-8"))
    assert rep["provenance"]["pyq_ideal_time_source"] == ["synthetic_tier_formula"]
    with_t = rep["pyq_with_ideal_time_5fold"]["accuracy"]
    without_t = rep["pyq_without_ideal_time_5fold"]["accuracy"]
    assert with_t - without_t > 0.1
    v2 = rep["sample_only_grouped_by_paper_5fold"]
    assert v2["accuracy"] > v2["majority_baseline"] + 0.15
    assert "ideal_time_sec" in rep["v2_candidate"]["excluded_features"]
    model = joblib.load(MODEL_DIR / rep["v2_candidate"]["artifact"])
    from model.difficulty_model_v2 import featurize

    pyq = json.loads((PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json").read_text(encoding="utf-8"))[:50]
    preds = model.predict(featurize(pyq))
    assert set(preds) <= {"basic", "medium", "advanced"} and len(preds) == 50


def ml_suite_32_unified_near_duplicate_guard_in_mocks():
    from model.mock_test_generator import generate_mock_180, reserved_initial_assessment_ids
    from model.unified_similarity_guard import UNIFIED_SIBLINGS_PATH, get_unified_siblings_map

    rep = json.loads(UNIFIED_SIBLINGS_PATH.read_text(encoding="utf-8"))
    sib = get_unified_siblings_map()
    for q, ss in list(sib.items())[:500]:
        for s in ss:
            assert q in sib.get(s, []), (q, s)
    base = reserved_initial_assessment_ids()
    full = reserved_initial_assessment_ids(include_siblings=True)
    assert base <= full
    seen: set[str] = set()
    for attempt in range(3):
        ids = generate_mock_180("guard-student", attempt, seen, source="unified")["question_ids"]
        s_ids = set(ids)
        assert not (s_ids & full)
        assert sum(1 for q in s_ids for s in sib.get(q, []) if s in s_ids) == 0
        seen |= s_ids
    assert rep["items_scanned"] >= 2400
    from model.ml_extension_registry import EXTENSION_ARTIFACTS, verify_extension_registry

    ext = verify_extension_registry()
    assert ext["status"] == "VERIFIED" and ext["extension_artifacts_verified"] == len(EXTENSION_ARTIFACTS)


def ml_suite_33_side_by_side_models_trained_on_unified_bank():
    from model.train_on_unified_bank import PRODUCTION_PATHS, REPORT_PATH, UNIFIED_PATHS

    rep = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    assert rep["training_corpus"]["pyq"] == 2400
    assert rep["training_corpus"]["sample_paper"] >= 9000
    assert rep["dina_8_skill_priors"]["items_unified"] == rep["training_corpus"]["total_items"]
    assert rep["bkt_chapter_priors"]["chapters"] == 65
    for k, p in UNIFIED_PATHS.items():
        assert p.exists(), p
        assert p != PRODUCTION_PATHS[k]
    bkt = json.loads(UNIFIED_PATHS["BKT_CHAPTER_JSON"].read_text(encoding="utf-8"))
    assert len(bkt["chapters"]) == 65
    for ch in bkt["chapters"].values():
        assert 0.0 < ch["p_init"] < 1.0
    prod = json.loads(PRODUCTION_PATHS["BKT_CHAPTER_JSON"].read_text(encoding="utf-8"))
    assert prod["weighted_global_prior"] == rep["bkt_chapter_priors"]["weighted_global_prior_production"]
    rag = rep["ncert_rag_index"]
    assert rag["heldout_chapter_retrieval_unified"]["eval_questions"] == rag["heldout_chapter_retrieval_production"]["eval_questions"] > 0
    assert rag["heldout_chapter_retrieval_production"]["top1_chapter_accuracy"] > 10 * rag["heldout_chapter_retrieval_production"]["random_baseline_top1"]


def ml_suite_36_honest_training_protocol_and_rag_vocabulary():
    from model.label_validity_audit import heuristic_vs_human_tiers
    from model.train_psychometric_and_rag_models import RAG_SVD_META_JSON

    rep = json.loads((PROJECT_ROOT / "model" / "validation_report.json").read_text(encoding="utf-8"))
    assert rep["evaluation_protocol"]["cross_validation"].startswith("StratifiedGroupKFold")
    assert rep["stratified_5fold_cross_validation"]["computed_on"] == "training_split_only"
    assert rep["temperature_calibration_summary"]["metrics_evaluated_on"] == "untouched_holdout"
    assert rep["target_validity"]["what_holdout_accuracy_measures"].startswith("agreement with the heuristic")
    hold = rep["random_forest_holdout_evaluation"]
    assert hold["training_rows"] + hold["testing_rows"] == 2400
    cal = json.loads((PROJECT_ROOT / "model" / "rf_temperature_calibration.json").read_text(encoding="utf-8"))
    if cal["temperature_decision"].startswith("rejected"):
        assert cal["optimal_temperature"] == 1.0
    meta = json.loads(RAG_SVD_META_JSON.read_text(encoding="utf-8"))
    vocab = meta["vocab"]
    assert len(vocab) == meta["vocab_size"] > 6000
    assert max(vocab) > "t", "vocabulary must not be truncated alphabetically"
    assert meta["heldout_chapter_retrieval"]["top1_chapter_accuracy"] >= 0.25
    human = heuristic_vs_human_tiers()
    assert human["available"] and human["cohens_kappa"] < 0.2
    harness = json.loads((PROJECT_ROOT / "model" / "harness_benchmark_report.json").read_text(encoding="utf-8"))
    cold = harness["linucb_cold_start_stochastic_ablation"]
    assert cold["linucb_outperforms_random"] and cold["reward_noise_sd"] > 0


def ml_suite_34_difficulty_v21_text_candidate_beats_v2_on_clean_rows():
    from model.difficulty_model_v21_text import V21_MODEL_PATH, V21_REPORT_PATH, featurize_with_text

    rep = json.loads(V21_REPORT_PATH.read_text(encoding="utf-8"))
    assert "GroupKFold" in rep["evaluation"]
    assert "ideal_time_sec" in rep["excluded_features"]
    clean = rep["near_duplicate_control"]
    v2c = clean["v2_structural_rf_on_clean_rows"]
    v21c = clean["v21_tfidf_logreg_on_clean_rows"]
    assert v21c["macro_f1"] > v2c["macro_f1"]
    assert v21c["accuracy"] > rep["majority_baseline"] + 0.20
    assert rep["v21_tfidf_logreg"]["ece_top_label"] < 0.05
    assert rep["decision"]["production_model_changed"] is False
    model = joblib.load(V21_MODEL_PATH)
    pyq = json.loads((PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json").read_text(encoding="utf-8"))[:40]
    proba = model.predict_proba(featurize_with_text(pyq))
    assert proba.shape == (40, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)


def ml_suite_35_honest_simulation_no_invented_penalties_and_sourced_facts():
    import inspect
    import model.monte_carlo_conformal_neet as mc
    from model.national_rank_simulator import MODEL_ASSUMPTIONS, NEET_YEAR_FACTS, QUALIFYING_CUTOFF_SOURCE
    from model.feature_store_and_drift import DriftAndCalibrationMonitor

    src = inspect.getsource(mc.run_monte_carlo_conformal_neet_simulation)
    assert "trap_penalty" not in src and "* 0.55" not in src and "rng.uniform" not in src
    assert mc.NEGATIVE_MARKING_BREAK_EVEN_PROB == 0.20
    out = mc.run_monte_carlo_conformal_neet_simulation(
        student_id="honesty_probe",
        global_theta=0.0,
        mastery_map={},
        theta_map={},
        ci_map={},
        topic_importance={t: 1.0 for t in DAG_NODES},
        n_trials=200,
    )
    assert out["interval_method"].startswith("monte_carlo_empirical_quantile")
    assert sum(out["item_parameter_sources"].values()) == 180
    assert out["negative_marking_optimizer"]["skip_decision_basis"] == "point_estimate_theta_not_oracle"
    assert out["cutoff_probabilities"]["sourced_qualifying_cutoff"]["score"] == QUALIFYING_CUTOFF_SOURCE["general_cutoff_score"]
    assert NEET_YEAR_FACTS[2026]["appeared"] == 1999895
    assert MODEL_ASSUMPTIONS["college_tier_thresholds"]["status"] == "approximate"
    empty = DriftAndCalibrationMonitor().evaluate_health()
    assert empty["metrics_available"] is False and empty["drift_status"] == "NO_ATTEMPTS_YET"


def ml_suite_37_student_response_training_pipeline():
    import tempfile
    from model.student_response_training import (
        bkt_one_step_predictions,
        data_readiness,
        fit_bkt_em,
        load_attempts,
        run_simulation_self_test,
        run_student_response_training,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        empty_csv = tmp / "empty.csv"
        empty_csv.write_text("student_id,question_id,paper_number,is_correct,timestamp\n", encoding="utf-8")
        skipped = run_student_response_training(csv_path=empty_csv, output_dir=tmp / "real")
        assert skipped["status"] == "SKIPPED_INSUFFICIENT_DATA"
        assert skipped["readiness"]["ready"] is False

        rep = run_simulation_self_test(tmp / "sim")
        assert rep["label"].startswith("SIMULATED")
        assert rep["parameter_recovery_correlation"]["b"] >= 0.9
        assert rep["parameter_recovery_correlation"]["a"] >= 0.4
        m = rep["held_out_metrics"]
        assert m["fitted_irt"]["auc"] > m["item_p_value"]["auc"] > m["global_rate"]["auc"]
        assert m["fitted_irt"]["log_loss"] < m["item_p_value"]["log_loss"]
        assert m["fitted_irt"]["ece"] < 0.05

        df, _ = load_attempts(tmp / "sim" / "simulated_attempts.csv")
        assert data_readiness(df)["students"] == 500

    rng = np.random.default_rng(3)
    true = {"p_init": 0.3, "p_transit": 0.12, "p_guess": 0.2, "p_slip": 0.08}
    seqs = []
    for _ in range(600):
        known = rng.random() < true["p_init"]
        s = []
        for _ in range(int(rng.integers(3, 12))):
            s.append(float(rng.random() < ((1 - true["p_slip"]) if known else true["p_guess"])))
            if not known and rng.random() < true["p_transit"]:
                known = True
        seqs.append(np.array(s))
    fit = fit_bkt_em(seqs[:480])
    for k in ("p_init", "p_guess", "p_slip"):
        assert abs(fit[k] - true[k]) < 0.06, (k, fit[k])
    assert abs(fit["p_transit"] - true["p_transit"]) < 0.06
    assert len(bkt_one_step_predictions(seqs[480:], fit)) == sum(len(s) for s in seqs[480:])


def ml_suite_38_train_serve_parity_deterministic_artifacts_and_honest_ui_claims():
    import re as _re
    import tempfile

    from model.deterministic_artifacts import npz_bytes, save_npz_deterministic
    from model.irt_3pl_model import rf_serving_status
    from model.knowledge_graph_dag import is_prerequisite_graph_acyclic
    from model.serving_runtime_parity import check_runtime_parity, requirements_pins, training_library_pins

    pins = training_library_pins()
    assert set(pins) == {"scikit-learn", "numpy", "pandas", "scipy", "joblib"}
    for req in ("backend/requirements.txt", "requirements.txt"):
        assert requirements_pins(PROJECT_ROOT / req) == pins, req

    py_version = (PROJECT_ROOT / ".python-version").read_text(encoding="utf-8").strip()
    dockerfile = (PROJECT_ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert f"FROM python:{py_version}-slim" in dockerfile
    assert "-c model/ml-training-constraints.txt" in dockerfile
    assert "model.serving_runtime_parity" in dockerfile
    for wf in (PROJECT_ROOT / ".github" / "workflows").glob("*.yml"):
        text = wf.read_text(encoding="utf-8")
        assert 'python-version: "3.11"' not in text, wf.name
        if "setup-python" in text:
            assert 'python-version-file: ".python-version"' in text, wf.name
            assert "ml-training-constraints.txt" in text, wf.name

    parity = check_runtime_parity(log_warning=False)
    assert parity["libraries_match"] is True, parity["mismatched_libraries"]
    if parity["python"]["match"] is not False:
        assert parity["all_match"] is True
    status = rf_serving_status()
    assert status["backend"] == "sklearn_joblib_pipeline" and status["error"] is None

    arrays = {"w": np.arange(12, dtype=np.float64).reshape(3, 4), "i": np.array([1, 2, 3], dtype=np.int16)}
    assert npz_bytes(arrays) == npz_bytes(arrays)
    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp) / "a.npz"
        assert save_npz_deterministic(target, **arrays) is True
        assert save_npz_deterministic(target, **arrays) is False
        loaded = np.load(target)
        assert all(np.array_equal(loaded[k], v) and loaded[k].dtype == v.dtype for k, v in arrays.items())
    for name in ("irt_cold_start_ridge.npz", "rf_difficulty_forest.npz", "ncert_rag_svd_index.npz"):
        data = np.load(MODEL_DIR / name)
        assert npz_bytes({k: data[k] for k in data.files}) == (MODEL_DIR / name).read_bytes(), name

    assert is_prerequisite_graph_acyclic() is True
    assert is_prerequisite_graph_acyclic({"A": {"prerequisites": ["B"]}, "B": {"prerequisites": ["A"]}}) is False

    vercel = json.loads((PROJECT_ROOT / "vercel.json").read_text(encoding="utf-8"))
    assert "KLASSED_JWT_SECRET" not in json.dumps(vercel)
    auth_src = (PROJECT_ROOT / "backend" / "app" / "auth.py").read_text(encoding="utf-8")
    assert "DEFAULT_JWT_SECRET" not in auth_src and "enterprise-hmac-secret" not in auth_src
    compose = (PROJECT_ROOT / "docker-compose.observability.yml").read_text(encoding="utf-8")
    assert "klassed_mlops" not in compose

    audit_view = (PROJECT_ROOT / "frontend" / "src" / "features" / "mlops" / "ModelAuditView.tsx").read_text(encoding="utf-8")
    dashboard = (PROJECT_ROOT / "frontend" / "src" / "features" / "dashboard" / "Dashboard.tsx").read_text(encoding="utf-8")
    assert "linucb_cold_start_stochastic_ablation" in audit_view
    assert "difficulty_v21_candidate" in audit_view
    assert not _re.search(r"\b815\b", audit_view)
    assert "Chi² Uniform" not in audit_view and "Passed · 0 Blocking Failures" not in audit_view
    assert "24-Lakh" not in dashboard and "Calibrated on official NTA" not in dashboard
    assert "Split conformal" not in dashboard and "Conformal Score Interval" not in dashboard


if __name__ == "__main__":
    suites = [
        ml_suite_01_question_bank_and_weightage_schema_validators,
        ml_suite_02_exponential_decay_14_year_weightage_engine,
        ml_suite_03_random_forest_champion_and_compact_numpy_parity,
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
        ml_suite_20_pipeline_deep_invariants_and_statistical_drift_gate,
        ml_suite_21_cognitive_entropy_kl_information_and_sem_stopping_gate,
        ml_suite_22_national_rank_and_college_cutoff_simulation,
        ml_suite_23_online_streaming_calibration_and_continuous_retraining,
        ml_suite_24_multilingual_hindi_glossary_and_socratic_audio_script,
        ml_suite_25_section4_empirical_recalibration_and_quarantine_audit,
        ml_suite_26_irt_equated_180q_initial_assessment_forms,
        ml_suite_27_sample_paper_ingestion_validation_and_cold_start_linking,
        ml_suite_28_equated_unlimited_mock_180_generation,
        ml_suite_29_initial_assessment_profile_initialization,
        ml_suite_30_test_analytics_and_equated_score_trend,
        ml_suite_31_difficulty_leakage_audit_and_v2_candidate,
        ml_suite_32_unified_near_duplicate_guard_in_mocks,
        ml_suite_33_side_by_side_models_trained_on_unified_bank,
        ml_suite_34_difficulty_v21_text_candidate_beats_v2_on_clean_rows,
        ml_suite_35_honest_simulation_no_invented_penalties_and_sourced_facts,
        ml_suite_36_honest_training_protocol_and_rag_vocabulary,
        ml_suite_37_student_response_training_pipeline,
        ml_suite_38_train_serve_parity_deterministic_artifacts_and_honest_ui_claims,
    ]
    for fn in suites:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(suites)} STANDALONE PURE-ML & MLOPS SUITES PASSED (100%)!")
