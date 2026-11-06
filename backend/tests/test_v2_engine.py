import concurrent.futures
import json
import math
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
MODEL_DIR = PROJECT_ROOT / "model"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR), str(MODEL_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.db_models import Student, StudentTopicState, TestAttempt, Question, Topic
from app.state_machine import process_answer
from app.time_tracking import update_pace
from app.paper_generator import generate_paper
from app.auth import create_access_token, verify_access_token, hash_password, verify_password

from model.importance_calculator import compute_importance
from model.validate_question_bank import validate as validate_qbank
from model.validate_weightage_file import validate as validate_weightage
from model.state_machine.bkt_model import (
    BKTParams,
    modulate_params_with_telemetry,
    compute_bkt_update,
    resolve_level_with_hysteresis,
)
from model.irt_3pl_model import (
    CALIBRATED_ITEMS,
    IRTItemParams,
    get_item_params,
    irt_3pl_probability,
    fisher_information,
    update_student_theta_map,
    theta_to_percentile,
)
from model.knowledge_graph_dag import (
    DAG_NODES,
    get_direct_prerequisites,
    get_all_ancestors,
    get_dependents,
    compute_prerequisite_readiness,
    compute_hierarchical_shrinkage_prior,
    diagnose_prerequisite_bottlenecks,
)
from model.linucb_bandit import (
    LinUCBStudentState,
    build_context_vector,
    score_candidate_ucb,
    record_bandit_feedback,
)
from model.rag_socratic_engine import generate_socratic_remediation, _INDEX as _RAG_INDEX
from model.feature_store_and_drift import InMemoryFeatureStore, DriftAndCalibrationMonitor


def test_01_random_forest_model_and_live_inference():
    metrics_path = MODEL_DIR / "question_difficulty_metrics.json"
    model_path = MODEL_DIR / "question_difficulty_model.joblib"
    csv_path = PROJECT_ROOT / "data" / "training" / "question_features.csv"

    assert metrics_path.exists() and model_path.exists() and csv_path.exists()
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    assert round(metrics["accuracy"], 4) == 0.9960
    assert metrics["training_rows"] == 1976 and metrics["testing_rows"] == 495

    pipeline = joblib.load(model_path)
    df = pd.read_csv(csv_path)
    features = metrics["features"]
    X = df[features]
    y = df["difficulty_level"]
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    preds = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test[:5])
    live_acc = (preds == y_test).mean()
    assert round(float(live_acc), 4) == 0.9960, f"Live RF accuracy mismatch: {live_acc}"
    assert probas.shape == (5, 3) and np.allclose(probas.sum(axis=1), 1.0)


def test_02_topic_importance_exponential_decay():
    synthetic_freq = {
        "Recent_Topic": {2026: 10},
        "Old_Topic": {2016: 10},
    }
    scores = compute_importance(synthetic_freq, current_year=2026, decay=0.9)
    assert round(sum(scores.values()), 1) == 100.0
    assert scores["Recent_Topic"] > scores["Old_Topic"] * 2.5

    with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
        prod_imp = json.load(f)
    assert len(prod_imp) == 65, f"Expected 65 topics in topic_importance.json, got {len(prod_imp)}"
    assert 99.5 <= sum(prod_imp.values()) <= 100.5


def test_03_question_bank_and_weightage_validators():
    with open(PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json", encoding="utf-8") as f:
        clean_qs = json.load(f)
    q_report = validate_qbank(clean_qs)
    assert q_report["total_questions"] == 2471
    assert q_report["issues"] == [], f"Unexpected question bank issues: {q_report['issues']}"


    broken_qs = [
        {
            "question_id": "BAD_1",
            "question_text": "Test?",
            "option_a": "Same",
            "option_b": "Same",
            "option_c": "Diff1",
            "option_d": "Diff2",
            "difficulty_level": "invalid_tier",
            "explanation": "Applying standard formulas and fundamental principles yields the correct outcome.",
        }
    ]
    bad_report = validate_qbank(broken_qs)
    assert len(bad_report["issues"]) >= 3

    with open(
        PROJECT_ROOT / "data" / "taxonomy" / "NEET_Unified_Weightage_And_Marks_2013_2026.json",
        encoding="utf-8",
    ) as f:
        w_data = json.load(f)
    units = w_data.get("Unified_Topic_Weightage_And_Marks", w_data)
    w_report = validate_weightage(units)
    assert w_report["total_units"] == 25 and w_report["issues"] == []


def test_04_bkt_and_saint_telemetry_modulator():
    base = BKTParams(p_init=0.35, p_transit=0.15, p_guess=0.22, p_slip=0.10)


    pt_fast, pg_fast, _ = modulate_params_with_telemetry(base, time_taken_seconds=5, expected_time_seconds=90)
    assert pg_fast > base.p_guess and pt_fast < base.p_transit


    pt_opt, _, _ = modulate_params_with_telemetry(base, time_taken_seconds=60, expected_time_seconds=90)
    assert pt_opt > base.p_transit


    _, pg_sw, ps_sw = modulate_params_with_telemetry(
        base, time_taken_seconds=60, expected_time_seconds=90, option_switches=3
    )
    assert pg_sw > base.p_guess and ps_sw > base.p_slip


    p_up, ci_1 = compute_bkt_update(0.40, is_correct=True, difficulty="medium", attempts_count=1)
    p_down, ci_20 = compute_bkt_update(0.40, is_correct=False, difficulty="medium", attempts_count=20)
    assert p_up > 0.40 > p_down
    assert ci_20 < ci_1


def test_05_state_machine_all_four_transitions():
    state = StudentTopicState(
        student_id="UNIT_S1",
        topic_id="Electrostatics",
        level="basic",
        streak=0,
        last_3_results=[],
        mastery_prob=0.35,
        confidence_interval=0.45,
    )

    process_answer(state, False, difficulty="basic", time_ratio=1.0)
    assert state.level == "basic"
    process_answer(state, True, difficulty="basic", time_ratio=0.9)
    process_answer(state, True, difficulty="basic", time_ratio=0.85)
    assert state.level == "medium"


    process_answer(state, False, difficulty="medium", time_ratio=1.3)
    process_answer(state, False, difficulty="medium", time_ratio=1.3)
    assert state.level == "basic"


    process_answer(state, True, difficulty="basic", time_ratio=0.8)
    process_answer(state, True, difficulty="basic", time_ratio=0.8)
    assert state.level == "medium"


    process_answer(state, True, difficulty="medium", time_ratio=0.85)
    process_answer(state, True, difficulty="medium", time_ratio=0.85)
    process_answer(state, True, difficulty="medium", time_ratio=0.85)
    assert state.level == "advanced"


    process_answer(state, False, difficulty="advanced", time_ratio=1.4)
    process_answer(state, False, difficulty="advanced", time_ratio=1.5)
    assert state.level == "medium"


def test_06_pace_tracking_invariants():
    state = StudentTopicState(
        student_id="UNIT_S1",
        topic_id="Thermodynamics",
        level="medium",
        attempts_counted=0,
        avg_time_ratio=None,
    )

    update_pace(state, is_correct=False, time_taken_seconds=200, difficulty="medium", expected_time_seconds=100)
    assert state.avg_time_ratio is None and state.attempts_counted == 0


    update_pace(state, is_correct=True, time_taken_seconds=80, difficulty="medium", expected_time_seconds=100)
    assert round(state.avg_time_ratio, 2) == 0.80 and state.pace_flag == "fast"


    update_pace(state, is_correct=True, time_taken_seconds=200, difficulty="medium", expected_time_seconds=100)
    assert round(state.avg_time_ratio, 2) == 1.40 and state.pace_flag == "slow"


def test_07_3pl_irt_mathematical_invariants():
    assert len(CALIBRATED_ITEMS) == 2471
    params = IRTItemParams(a=1.5, b=0.5, c=0.20)


    p_neg_inf = irt_3pl_probability(-20.0, params)
    p_pos_inf = irt_3pl_probability(20.0, params)
    p_mid = irt_3pl_probability(0.5, params)
    assert abs(p_neg_inf - 0.20) < 1e-4
    assert abs(p_pos_inf - 1.00) < 1e-4
    assert abs(p_mid - 0.60) < 1e-4


    info_near = fisher_information(0.6, params)
    info_tail = fisher_information(-1.5, params)
    assert info_near > info_tail > 0.0


    t_up = update_student_theta_map(0.0, is_correct=True, params=params)
    t_down = update_student_theta_map(0.0, is_correct=False, params=params)
    assert t_up > 0.0 > t_down
    assert theta_to_percentile(0.0) == 50.0
    assert theta_to_percentile(2.0) > 95.0


def test_08_prerequisite_dag_acyclicity_and_shrinkage():
    assert len(DAG_NODES) == 65


    for topic in DAG_NODES:
        ancestors = get_all_ancestors(topic)
        assert topic not in ancestors, f"Cycle detected in Prerequisite DAG involving {topic}"
        for prereq in get_direct_prerequisites(topic):
            assert prereq in DAG_NODES, f"Unknown prerequisite '{prereq}' referenced by '{topic}'"
            assert topic in get_dependents(prereq)

    mastery_map = {
        "Units & Dimensions": 0.80,
        "Kinematics": 0.75,
        "Laws of Motion": 0.70,
        "Work, Energy & Power": 0.20,
    }
    assert compute_prerequisite_readiness("Units & Dimensions", mastery_map) == 1.0
    assert round(compute_prerequisite_readiness("Work, Energy & Power", mastery_map), 4) == 0.7250

    eb_1 = compute_hierarchical_shrinkage_prior("Work, Energy & Power", mastery_map, attempts_count=1)
    eb_50 = compute_hierarchical_shrinkage_prior("Work, Energy & Power", mastery_map, attempts_count=50)
    assert eb_1 > 0.20 and abs(eb_50 - 0.20) < abs(eb_1 - 0.20)

    bottlenecks = diagnose_prerequisite_bottlenecks({"Kinematics": 0.25, "Laws of Motion": 0.22})
    assert len(bottlenecks) >= 1


def test_09_linucb_bandit_uncertainty_and_reward():
    x = build_context_vector(
        question_id="NEET_2024_PHY_1",
        difficulty="medium",
        expected_time_seconds=90,
        student_theta=0.0,
        topic_mastery_prob=0.45,
        topic_importance=3.0,
        prereq_readiness=0.85,
        pace_flag="fast",
    )
    assert x.shape == (6,) and np.all((x >= 0.0) & (x <= 1.0))

    student_id = "UNIT_BANDIT_STUDENT"
    from model.linucb_bandit import reset_student_bandit
    reset_student_bandit(student_id, remove=True)
    try:
        ucb1, exploit1, explore1 = score_candidate_ucb(student_id, x)
        assert abs(ucb1 - (exploit1 + explore1)) < 1e-6


        reward = record_bandit_feedback(
            student_id=student_id,
            question_id="NEET_2024_PHY_1",
            difficulty="medium",
            expected_time_seconds=90,
            student_theta=0.0,
            topic_mastery_prob=0.45,
            topic_importance=3.0,
            prereq_readiness=0.85,
            pace_flag="fast",
            is_correct=True,
            time_taken_seconds=70,
        )
        _, _, explore2 = score_candidate_ucb(student_id, x)
        assert 0.0 < reward <= 1.0
        assert explore2 < explore1, "LinUCB exploration bonus must shrink as context observations accumulate"
    finally:
        reset_student_bandit(student_id, remove=True)


def test_10_rag_socratic_engine_all_subjects():
    assert len(_RAG_INDEX.chunks) == 65
    for qid, chap in [
        ("NEET_2024_PHY_1", "Electrostatics"),
        ("NEET_2024_CHEM_1", "Mole Concept / Stoichiometry"),
        ("NEET_2024_BIO_1", "Cell: The Unit of Life"),
    ]:
        res = generate_socratic_remediation(
            question_id=qid,
            topic_id=chap,
            question_text=f"Conceptual question on {chap}",
            options=["Option A", "Option B", "Option C", "Option D"],
            correct_answer="a",
            selected_option="c",
        )
        assert res["topic_id"] == chap
        assert res["retrieval_confidence"] >= 0.82
        assert res["retrieval_latency_ms"] < 100.0
        assert len(res["socratic_guiding_questions"]) >= 1
        assert "Option (C" in res["misconception_diagnosis"] or "option (C" in res["misconception_diagnosis"]


def test_11_feature_store_and_drift_monitor():
    fs = InMemoryFeatureStore()
    assert fs.get_student_features("MISSING") is None
    fs.put_student_features("S1", {"mastery_prob": 0.72})
    assert fs.get_student_features("S1")["mastery_prob"] == 0.72
    lat = fs.get_latency_percentiles()
    assert lat["p50_ms"] < 5.0 and lat["cached_students"] == 1

    dm = DriftAndCalibrationMonitor()
    assert dm.compute_psi([0.3, 0.5, 0.2], [0.3, 0.5, 0.2]) == 0.0
    high_psi = dm.compute_psi([0.3, 0.5, 0.2], [0.8, 0.1, 0.1])
    assert high_psi > 0.20


    skewed = [{"predicted_prob": 0.90, "correct": 0, "difficulty": "advanced", "time_taken": 140} for _ in range(35)]
    health = dm.evaluate_health(skewed)
    assert health["recalibration_required"] is True
    assert health["drift_status"] == "DRIFT_WARNING"


def test_12_auth_bcrypt_and_jwt_security():
    hashed = hash_password("my_secret_pass")
    assert verify_password("my_secret_pass", hashed) is True
    assert verify_password("wrong_pass", hashed) is False
    assert verify_password("any", "malformed_hash") is False

    token = create_access_token("S100", "s100@klassed.in")
    assert verify_access_token(token)["sub"] == "S100"

    tampered = token[:-4] + "XXXX"
    assert verify_access_token(tampered) is None
    assert verify_access_token("not.a.jwt.token") is None


def test_13_database_counts_and_adaptive_paper_generator():
    db = SessionLocal()
    try:
        assert db.query(Topic).count() == 65
        assert db.query(Question).count() == 2471
        paper = generate_paper(db, "COLD_START_AUDIT", total_length=45)
        assert len(paper) >= 40
    finally:
        from model.linucb_bandit import reset_student_bandit
        reset_student_bandit("COLD_START_AUDIT", remove=True)
        db.close()


def test_14_all_11_endpoints_and_concurrent_stress():
    client = TestClient(app)
    test_email = "deep_suite_verifier@klassed.in"
    created_student_id = None

    try:
        db = SessionLocal()
        old = db.query(Student).filter_by(email=test_email).first()
        if old:
            db.query(TestAttempt).filter_by(student_id=old.student_id).delete()
            db.query(StudentTopicState).filter_by(student_id=old.student_id).delete()
            db.delete(old)
            db.commit()
        db.close()


        r_bad_pw = client.post("/auth/signup", json={"name": "X", "email": test_email, "password": "123"})
        assert r_bad_pw.status_code == 422


        r_signup = client.post(
            "/auth/signup",
            json={"name": "Deep Verifier", "email": test_email, "password": "strongpassword123"},
        )
        assert r_signup.status_code == 200
        created_student_id = r_signup.json()["student_id"]


        r_dup = client.post(
            "/auth/signup",
            json={"name": "Deep Verifier", "email": test_email, "password": "strongpassword123"},
        )
        assert r_dup.status_code == 409


        r_bad_login = client.post("/auth/login", json={"email": test_email, "password": "wrongpassword"})
        assert r_bad_login.status_code == 401


        r_login = client.post("/auth/login", json={"email": test_email, "password": "strongpassword123"})
        assert r_login.status_code == 200


        r_onboard = client.post(
            f"/students/{created_student_id}/onboarding",
            json={"class_level": "12", "target_neet_year": 2027},
        )
        assert r_onboard.status_code == 200 and r_onboard.json()["onboarded"] is True


        assert client.get(f"/students/{created_student_id}").status_code == 200


        r_paper = client.get(f"/students/{created_student_id}/next_paper")
        assert r_paper.status_code == 200
        paper_items = r_paper.json()
        assert len(paper_items) >= 40


        def _submit_one(idx: int):
            q = paper_items[idx]
            resp = client.post(
                "/submit_attempt",
                json={
                    "student_id": created_student_id,
                    "question_id": q["question_id"],
                    "paper_number": 1,
                    "selected_option": ["a", "b", "c", "d"][idx % 4],
                    "time_taken_seconds": 50 + idx * 5,
                    "option_switches": idx % 3,
                    "hesitation_ms": 3000 + idx * 200,
                    "dwell_time_ms": (50 + idx * 5) * 1000,
                },
            )
            return resp.status_code, resp.json()

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(_submit_one, i) for i in range(8)]
            results = [f.result() for f in futures]

        for code, body in results:
            assert code == 200, f"Concurrent submit failed: {code} {body}"
            assert "mastery_prob" in body and "irt_theta" in body


        r_state = client.get(f"/students/{created_student_id}/state")
        assert r_state.status_code == 200 and len(r_state.json()) >= 1


        r_rev = client.get(f"/students/{created_student_id}/papers/1/review")
        assert r_rev.status_code == 200 and r_rev.json()["total_questions"] == 8


        r_chap = client.get(f"/students/{created_student_id}/next_paper?topic_id=Wave%20Optics")
        assert r_chap.status_code == 200
        chap_items = r_chap.json()
        assert len(chap_items) >= 2 and all(q["topic_id"] == "Wave Optics" for q in chap_items)

        r_batch = client.post(
            "/submit_paper",
            json={
                "student_id": created_student_id,
                "paper_number": 1,
                "attempts": [
                    {
                        "student_id": created_student_id,
                        "question_id": chap_items[0]["question_id"],
                        "paper_number": 1,
                        "selected_option": "a",
                        "time_taken_seconds": 45,
                        "option_switches": 0,
                        "hesitation_ms": 1500,
                        "dwell_time_ms": 45000,
                    },
                    {
                        "student_id": created_student_id,
                        "question_id": chap_items[1]["question_id"],
                        "paper_number": 1,
                        "selected_option": "b",
                        "time_taken_seconds": 55,
                        "option_switches": 1,
                        "hesitation_ms": 2200,
                        "dwell_time_ms": 55000,
                    },
                ],
            },
        )
        assert r_batch.status_code == 200
        batch_body = r_batch.json()
        assert batch_body["paper_number"] == 2 and batch_body["total_questions"] == 2


        r_ana = client.get(f"/students/{created_student_id}/analytics")
        assert r_ana.status_code == 200 and r_ana.json()["total_attempts"] == 10


        r_ins = client.get(f"/students/{created_student_id}/v2_insights")
        assert r_ins.status_code == 200
        ins_body = r_ins.json()
        assert 120 <= ins_body["predicted_neet_score_720"] <= 720
        assert len(ins_body["recommended_unlock_topics"]) == 65
        assert "mirt_3d_profile" in ins_body and ins_body["mirt_3d_profile"] is not None
        assert "spaced_repetition_reviews" in ins_body


        r_soc = client.post(
            "/remediation/socratic_hint",
            json={"question_id": paper_items[0]["question_id"], "selected_option": "b"},
        )
        assert r_soc.status_code == 200 and len(r_soc.json()["socratic_hint"]) > 10

        r_cold = client.post(
            "/mlops/predict_cold_start_item",
            json={
                "question_text": "Which of the following is NOT true if a 5.0 kg block accelerates at 2.5 m/s^2 under F = m * a = 12.5 N?",
                "options": ["12.5 N", "-12.5 N", "6.25 N", "25.0 N"],
                "has_diagram": True,
            },
        )
        assert r_cold.status_code == 200
        assert r_cold.json()["predicted_difficulty"] in ("basic", "medium", "advanced")

        r_health = client.get("/mlops/health_and_drift")
        assert r_health.status_code == 200
        assert all(r_health.json()["models_loaded"].values())

    finally:
        if created_student_id:
            from model.linucb_bandit import reset_student_bandit
            reset_student_bandit(created_student_id, remove=True)
            db = SessionLocal()
            db.query(TestAttempt).filter_by(student_id=created_student_id).delete()
            db.query(StudentTopicState).filter_by(student_id=created_student_id).delete()
            db.query(Student).filter_by(student_id=created_student_id).delete()
            db.commit()
            db.close()


def test_15_v3_advanced_psychometric_and_nlp_upgrades():
    from model.state_machine.bkt_model import (
        compute_ebbinghaus_forgetting_prior,
        compute_retention_urgency,
    )
    from model.irt_3pl_model import (
        predict_cold_start_difficulty_and_irt,
        compute_mirt_3d_profile,
        recalibrate_item_parameters_online,
        get_item_params,
    )
    from model.knowledge_graph_dag import propagate_dag_belief_update
    from model.rag_socratic_engine import generate_socratic_remediation


    p_fresh = compute_ebbinghaus_forgetting_prior(0.85, elapsed_hours=0.1, streak=2)
    p_week_old = compute_ebbinghaus_forgetting_prior(0.85, elapsed_hours=168.0, streak=0)
    assert p_fresh == 0.85
    assert 0.35 <= p_week_old < p_fresh, f"Expected decay after 168h: {p_week_old} vs {p_fresh}"
    assert compute_retention_urgency(0.85, elapsed_hours=168.0, streak=0) > 0.25


    simple_pred = predict_cold_start_difficulty_and_irt(
        "What is the SI unit of electric current?",
        ["Ampere", "Volt", "Ohm", "Watt"],
        has_diagram=False,
    )
    complex_pred = predict_cold_start_difficulty_and_irt(
        "Which statement is NOT correct when calculating sin(30) = 0.5 for a 2.5e-3 mol/L solution at 298 K and 101.3 kPa with E = -13.6 / n^2 eV?",
        ["-3.40 eV", "+3.40 eV", "-1.51 eV", "+1.51 eV"],
        has_diagram=True,
    )
    assert complex_pred["complexity_score"] > simple_pred["complexity_score"]
    assert complex_pred["irt_params"]["b"] > simple_pred["irt_params"]["b"]


    mirt = compute_mirt_3d_profile([
        {"question_id": "Q1", "question_text": "Calculate the molarity of 2.5 mol in 0.5 L.", "question_type": "Numerical", "difficulty": "medium", "expected_time_seconds": 90, "is_correct": True},
        {"question_id": "Q2", "question_text": "Given F = q * E, find the electric field relation.", "question_type": "Direct Formula", "difficulty": "basic", "expected_time_seconds": 60, "is_correct": True},
        {"question_id": "Q3", "question_text": "Which organelle is known as the powerhouse of the cell?", "question_type": "Conceptual", "difficulty": "basic", "expected_time_seconds": 45, "is_correct": False},
    ])
    assert mirt["theta_numerical"] > 0.0
    assert mirt["theta_formula"] > 0.0
    assert mirt["theta_conceptual"] < 0.0


    before_b = get_item_params("NEET_2025_PHY_001").b
    updated_item = recalibrate_item_parameters_online("NEET_2025_PHY_001", student_theta=-1.0, is_correct=True, time_ratio=0.7)
    assert updated_item.b < before_b, "Item solved easily by low-theta student should become slightly easier (lower b_i)"
    CALIBRATED_ITEMS.pop("NEET_2025_PHY_001", None)


    dag_boosts = propagate_dag_belief_update(
        topic_id="Rotational Motion",
        is_correct=True,
        difficulty="advanced",
        student_mastery_map={"Laws of Motion": 0.45},
    )
    assert "Laws of Motion" in dag_boosts and dag_boosts["Laws of Motion"] > 0.45


    rem_sign = generate_socratic_remediation(
        question_id="NEET_SIGN_TEST",
        topic_id="Ray Optics",
        question_text="Find the focal length of a concave mirror of radius 20 cm.",
        options=["-10 cm", "+10 cm", "-20 cm", "+20 cm"],
        correct_answer="a",
        selected_option="b",
    )
    assert "Sign Inversion Trap" in rem_sign["misconception_diagnosis"]

    rem_factor2 = generate_socratic_remediation(
        question_id="NEET_FACTOR2_TEST",
        topic_id="Ray Optics",
        question_text="Find the focal length of a concave mirror of radius 20 cm.",
        options=["-10 cm", "+10 cm", "-20 cm", "+20 cm"],
        correct_answer="a",
        selected_option="c",
    )
    assert "Factor-of-2 Trap" in rem_factor2["misconception_diagnosis"]


def test_16_deep_training_and_cross_validation_pipeline():
    from model.training_and_validation_pipeline import run_deep_training_and_validation

    report = run_deep_training_and_validation(save_artifacts=True)
    assert report["pipeline_status"] == "PASSED"
    assert report["random_forest_holdout_evaluation"]["holdout_accuracy"] >= 0.990
    assert report["random_forest_holdout_evaluation"]["correct_predictions"] >= 491
    assert report["stratified_5fold_cross_validation"]["mean_cv_accuracy"] >= 0.980
    assert report["stratified_5fold_cross_validation"]["oof_multiclass_brier_score"] < 0.05
    assert report["irt_3pl_calibration_summary"]["calibrated_questions"] >= 2471
    assert (PROJECT_ROOT / "model" / "validation_report.json").exists()


def test_17_harness_engineering_multi_persona_and_bandit_regret():
    from model.evaluation_harness import run_evaluation_harness

    report = run_evaluation_harness(save_report=True)
    assert report["harness_status"] == "PASSED"
    assert report["persona_simulations"]["strong_master"]["invariants_satisfied"] is True
    assert report["persona_simulations"]["foundational_struggler"]["invariants_satisfied"] is True
    assert report["persona_simulations"]["impulsive_guesser_telemetry_audit"]["invariants_satisfied"] is True
    assert report["persona_simulations"]["spaced_repetition_learner"]["invariants_satisfied"] is True
    assert report["linucb_vs_random_ablation"]["linucb_outperforms_random"] is True
    assert report["latency_sla_benchmarks"]["all_slas_met"] is True
    assert (PROJECT_ROOT / "model" / "harness_benchmark_report.json").exists()


def test_18_cicd_pipeline_and_deployment_manifests():
    workflow_path = PROJECT_ROOT / ".github" / "workflows" / "ci_cd_mlops.yml"
    assert workflow_path.exists(), "Missing GitHub Actions CI/CD workflow"
    wf_text = workflow_path.read_text(encoding="utf-8")
    assert "model.training_and_validation_pipeline" in wf_text
    assert "model.evaluation_harness" in wf_text
    assert "backend/tests/test_v2_engine.py" in wf_text

    for manifest in ["Dockerfile", "vercel.json", "render.yaml", "railway.json", "api/index.py"]:
        assert (PROJECT_ROOT / manifest).exists(), f"Missing deployment manifest: {manifest}"


def test_19_serverless_numpy_tfidf_and_scientific_trap_detector():
    from model.rag_socratic_engine import _NumpyTfidfVectorizer, _detect_distractor_trap, _INDEX

    corpus = [
        f"{c['subject']} {c['chapter']} {c['core_concept']} {' '.join(c.get('formulas', []))}"
        for c in _INDEX.chunks
    ]
    vec = _NumpyTfidfVectorizer()
    mat = vec.fit_transform(corpus)
    assert mat.shape[0] == 65 and mat.shape[1] > 100

    row_norms = np.linalg.norm(mat, axis=1)
    assert np.allclose(row_norms, 1.0, atol=1e-5)

    q_vec = vec.transform(["Gauss law electric flux point charge Coulomb permitivity"])
    sims = (mat @ q_vec.T).flatten()
    best_chunk = _INDEX.chunks[int(np.argmax(sims))]
    assert best_chunk["chapter"] == "Electrostatics"


    assert "Sign Inversion Trap" in _detect_distractor_trap("−1.6e-19 C", "+1.6e-19 C")
    assert "SI Unit / Order-of-Magnitude Trap" in _detect_distractor_trap("2.5e-2 mol/L", "2.5e-5 mol/L")
    assert "Reciprocal Formula Trap" in _detect_distractor_trap("0.25 Hz", "4.0 Hz")


def test_20_all_65_chapters_end_to_end_curriculum_integrity():
    with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
        imp_topics = set(json.load(f).keys())
    dag_topics = set(DAG_NODES.keys())
    rag_topics = {c["chapter"] for c in _RAG_INDEX.chunks}

    assert len(imp_topics) == 65
    assert imp_topics == dag_topics == rag_topics, "Mismatch across importance, DAG, and RAG 65 chapters"

    db = SessionLocal()
    try:
        db_topics = {t.topic_id for t in db.query(Topic).all()}
        assert db_topics == dag_topics, "SQLite topics table does not match the 65 DAG chapters"


        for chap in sorted(dag_topics):
            drill = generate_paper(db, "SYLLABUS_AUDIT", topic_id=chap)
            assert len(drill) >= 2, f"Chapter drill for '{chap}' returned fewer than 2 items"
            assert all(q.topic_id == chap for q in drill), f"Topic mismatch in drill for '{chap}'"
    finally:
        from model.linucb_bandit import reset_student_bandit
        reset_student_bandit("SYLLABUS_AUDIT", remove=True)
        db.close()


def test_21_dina_8_skill_q_matrix_and_lognormal_rt_model():
    from model.dina_cognitive_diagnosis import (
        COGNITIVE_SKILLS,
        extract_item_q_vector,
        compute_dina_and_rt_profile,
    )

    assert len(COGNITIVE_SKILLS) == 8
    q_vec = extract_item_q_vector(
        "Calculate the pH and molarity in mol/L for a first-order reaction with half-life 20 s.",
        topic_id="Chemical Kinetics",
        difficulty="medium",
    )
    assert q_vec["dimensional_unit_analysis"] == 1
    assert q_vec["algebraic_numerical_computation"] == 1
    assert q_vec["log_exp_calculus_reasoning"] == 1

    attempts = [
        {
            "question_id": "Q1",
            "question_text": "Calculate the molarity in mol/L when 4.0 g of NaOH is dissolved in 500 mL.",
            "topic_id": "Mole Concept / Stoichiometry",
            "difficulty": "basic",
            "expected_time_seconds": 90,
            "time_taken_seconds": 42,
            "is_correct": True,
        },
        {
            "question_id": "Q2",
            "question_text": "Find the half-life and log rate constant for a first-order radioactive decay.",
            "topic_id": "Chemical Kinetics",
            "difficulty": "medium",
            "expected_time_seconds": 100,
            "time_taken_seconds": 135,
            "is_correct": False,
        },
        {
            "question_id": "Q3",
            "question_text": "Which of the following statements is incorrect regarding lanthanoid contraction?",
            "topic_id": "d- and f-Block Elements",
            "difficulty": "advanced",
            "expected_time_seconds": 90,
            "time_taken_seconds": 8,
            "is_correct": False,
        },
    ]
    profile = compute_dina_and_rt_profile(attempts)
    assert len(profile["skills"]) == 8
    skill_map = {s["skill_id"]: s for s in profile["skills"]}
    assert skill_map["dimensional_unit_analysis"]["mastery_probability"] > 0.50
    assert skill_map["log_exp_calculus_reasoning"]["mastery_probability"] < 0.50
    beh = profile["speed_ability_profile"]["behavior_classification"]
    assert beh["fluent_mastery"] >= 1
    assert beh["rapid_guess"] >= 1
    assert beh["conceptual_struggle"] >= 1


def test_22_monte_carlo_conformal_simulator_and_graph_laplacian_leverage():
    from model.monte_carlo_conformal_neet import (
        _allocate_180_neet_questions,
        run_monte_carlo_conformal_neet_simulation,
    )

    with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
        imp = json.load(f)

    alloc = _allocate_180_neet_questions(imp)
    assert len(alloc) == 65
    assert sum(cnt for _, _, cnt in alloc) == 180
    assert sum(cnt for _, subj, cnt in alloc if subj == "Physics") == 45
    assert sum(cnt for _, subj, cnt in alloc if subj == "Chemistry") == 45
    assert sum(cnt for _, subj, cnt in alloc if subj == "Biology") == 90

    sim_low = run_monte_carlo_conformal_neet_simulation(
        student_id="TEST_LOW",
        global_theta=-0.8,
        mastery_map={k: 0.22 for k in imp},
        theta_map={k: -0.8 for k in imp},
        ci_map={k: 0.35 for k in imp},
        topic_importance=imp,
        n_trials=500,
    )
    sim_high = run_monte_carlo_conformal_neet_simulation(
        student_id="TEST_HIGH",
        global_theta=1.4,
        mastery_map={k: 0.88 for k in imp},
        theta_map={k: 1.4 for k in imp},
        ci_map={k: 0.18 for k in imp},
        topic_importance=imp,
        n_trials=500,
    )
    assert sim_high["expected_score_attempt_all_720"] > sim_low["expected_score_attempt_all_720"] + 200
    low_90, high_90 = sim_low["conformal_90_interval"]
    assert low_90 <= sim_low["expected_score_attempt_all_720"] <= high_90
    assert sim_low["negative_marking_optimizer"]["expected_marks_gained_by_skipping"] > 0
    assert len(sim_low["high_yield_leverage_chapters"]) == 6
    assert sim_low["high_yield_leverage_chapters"][0]["total_expected_marks_gain"] > 0


def test_23_scientific_notation_traps_paper_history_and_remediation_urgency():
    from model.rag_socratic_engine import _detect_distractor_trap
    from model.state_machine.bkt_model import compute_retention_urgency

    assert "SI Unit / Order-of-Magnitude Trap" in _detect_distractor_trap("36 × 104 J", "36 × 107 J")
    assert "SI Unit / Order-of-Magnitude Trap" in _detect_distractor_trap("6.67 × 1015", "6.67 × 1011")
    assert "SI Unit / Order-of-Magnitude Trap" in _detect_distractor_trap("42 × 10–15", "42 × 10–18")
    assert "Factor-of-2 Trap" in _detect_distractor_trap("25 × 10–6 N", "50 × 10–6 N")
    assert "Sign Inversion Trap" in _detect_distractor_trap("-1.6 × 10⁻¹⁹ C", "1.6 × 10⁻¹⁹ C")

    assert compute_retention_urgency(0.29, elapsed_hours=0.1, streak=0) == 0.0
    assert compute_retention_urgency(0.29, elapsed_hours=43.3, streak=0) > 0.30

    client = TestClient(app)
    email = "deep_history_jwt_verifier@klassed.in"
    sid = None
    try:
        db = SessionLocal()
        old = db.query(Student).filter_by(email=email).first()
        if old:
            db.query(TestAttempt).filter_by(student_id=old.student_id).delete()
            db.query(StudentTopicState).filter_by(student_id=old.student_id).delete()
            db.delete(old)
            db.commit()
        db.close()

        r_up = client.post(
            "/auth/signup",
            json={"name": "History JWT Verifier", "email": email, "password": "strongpassword123"},
        )
        assert r_up.status_code == 200
        body = r_up.json()
        sid = body["student_id"]
        token = body["access_token"]
        assert token and len(token.split(".")) == 3
        headers = {"Authorization": f"Bearer {token}"}

        r_on = client.post(
            f"/students/{sid}/onboarding",
            json={"class_level": "12", "target_neet_year": 2027},
            headers=headers,
        )
        assert r_on.status_code == 200 and r_on.json()["onboarded"] is True

        r_qs = client.get(f"/students/{sid}/next_paper?topic_id=Electrostatics", headers=headers)
        assert r_qs.status_code == 200
        qs = r_qs.json()

        r_p1 = client.post(
            "/submit_paper",
            headers=headers,
            json={
                "student_id": sid,
                "paper_number": 1,
                "attempts": [
                    {
                        "student_id": sid,
                        "question_id": qs[0]["question_id"],
                        "paper_number": 1,
                        "selected_option": "a",
                        "time_taken_seconds": 40,
                    }
                ],
            },
        )
        assert r_p1.status_code == 200 and r_p1.json()["paper_number"] == 1

        r_p2 = client.post(
            "/submit_paper",
            headers=headers,
            json={
                "student_id": sid,
                "paper_number": 1,
                "attempts": [
                    {
                        "student_id": sid,
                        "question_id": qs[1]["question_id"],
                        "paper_number": 1,
                        "selected_option": "b",
                        "time_taken_seconds": 55,
                    }
                ],
            },
        )
        assert r_p2.status_code == 200 and r_p2.json()["paper_number"] == 2

        r_hist = client.get(f"/students/{sid}/papers/history", headers=headers)
        assert r_hist.status_code == 200
        hist = r_hist.json()
        assert len(hist) == 2
        assert [h["paper_number"] for h in hist] == [2, 1]
        assert all(h["total_questions"] == 1 for h in hist)

        import base64
        journal = [
            [1, qs[0]["question_id"], "a", 40, 0, 500, 40000],
            [2, qs[1]["question_id"], "b", 55, 1, 800, 55000],
        ]
        b64_journal = base64.urlsafe_b64encode(json.dumps(journal).encode("utf-8")).decode("ascii").rstrip("=")

        db = SessionLocal()
        db.query(TestAttempt).filter_by(student_id=sid).delete()
        db.query(StudentTopicState).filter_by(student_id=sid).delete()
        db.query(Student).filter_by(student_id=sid).delete()
        db.commit()
        db.close()

        onboarded_token = r_on.json()["access_token"]
        replay_headers = {
            "Authorization": f"Bearer {onboarded_token}",
            "X-Klassed-Journal": b64_journal,
        }
        r_snap = client.get(f"/students/{sid}/dashboard_snapshot", headers=replay_headers)
        assert r_snap.status_code == 200
        snap = r_snap.json()
        assert snap["analytics"]["total_attempts"] == 2
        assert len(snap["topics"]) == 1 and snap["topics"][0]["topic_id"] == "Electrostatics"
        assert len(snap["paper_history"]) == 2
        mirt = snap["v2_insights"]["mirt_3d_profile"]
        assert mirt["attempts_conceptual"] + mirt["attempts_formula"] + mirt["attempts_numerical"] == 2
    finally:
        if sid:
            from model.linucb_bandit import reset_student_bandit
            reset_student_bandit(sid, remove=True)
            db = SessionLocal()
            db.query(TestAttempt).filter_by(student_id=sid).delete()
            db.query(StudentTopicState).filter_by(student_id=sid).delete()
            db.query(Student).filter_by(student_id=sid).delete()
            db.commit()
            db.close()




def test_24_serverless_vault_relogin_compact_rf_parity_and_full_audit():
    import ast
    import base64
    import csv
    import tokenize
    from model.rf_compact_runtime import COMPACT_RF_RUNTIME

    assert COMPACT_RF_RUNTIME.loaded is True
    pipeline = joblib.load(MODEL_DIR / "question_difficulty_model.joblib")
    classes = [str(c) for c in pipeline.classes_]
    sample_rows = [
        ("Physics", "Electrostatics", "Electric Field", "Single_Choice_MCQ", 95, False),
        ("Chemistry", "Chemical & Ionic Equilibrium", "Ionic Equilibrium", "Assertion_Reason", 110, False),
        ("Biology", "Plant Kingdom", "Algae", "Diagram_Based", 45, True),
    ]
    for subj, chap, ttag, qtype, itime, diag in sample_rows:
        df_one = pd.DataFrame([{
            "subject": subj,
            "chapter": chap,
            "topic_tag": ttag,
            "question_type": qtype,
            "ideal_time_sec": itime,
            "has_diagram": int(diag),
            "marks": 4,
            "negative_marks": -1,
        }])
        sk_probs = {c: float(p) for c, p in zip(classes, pipeline.predict_proba(df_one)[0])}
        np_probs = COMPACT_RF_RUNTIME.predict_proba_single(
            subject=subj,
            chapter=chap,
            topic_tag=ttag,
            question_type=qtype,
            ideal_time_sec=itime,
            has_diagram=diag,
        )
        for c in classes:
            assert abs(sk_probs[c] - np_probs[c]) < 1e-3, f"RF parity drift on {c}: {sk_probs} vs {np_probs}"

    client = TestClient(app)
    r_audit = client.get("/mlops/audit_overview")
    assert r_audit.status_code == 200
    audit = r_audit.json()
    assert audit["corpus_integrity"]["verified_questions_in_db"] == 2471
    assert audit["corpus_integrity"]["canonical_chapters_in_db"] == 65
    assert audit["corpus_integrity"]["dag_acyclic"] is True
    assert audit["health_and_drift"]["models_loaded"]["rf_300_tree_compact_serverless_runtime"] is True
    assert audit["validation_report"]["pipeline_status"] == "PASSED"
    assert audit["harness_benchmark"]["harness_status"] == "PASSED"
    assert set(audit["chapters_by_subject"].keys()) >= {"Physics", "Chemistry", "Biology"}

    r_cal = client.post(
        "/mlops/predict_cold_start_item",
        json={
            "question_text": "Two point charges +4 uC and -9 uC are separated by 0.30 m. Calculate electric field E = k*q/r^2.",
            "options": ["5.2 x 10^6 N/C", "5.2 x 10^3 N/C", "2.0 x 10^6 N/C", "3.6 x 10^5 N/C"],
            "has_diagram": False,
            "subject": "Physics",
            "chapter": "Electrostatics",
            "topic_tag": "Electric Field",
        },
    )
    assert r_cal.status_code == 200
    cal = r_cal.json()
    assert cal["rf_probabilities"] is not None
    assert abs(sum(cal["rf_probabilities"].values()) - 1.0) < 0.01
    assert cal["mirt_dimension"] in ("conceptual", "formula", "numerical")
    assert len(cal["active_cognitive_skills"]) >= 2

    email = "vault_relogin_verifier@klassed.in"
    pw = "VaultSecurePass123!"
    sid = None
    try:
        db = SessionLocal()
        old = db.query(Student).filter_by(email=email).first()
        if old:
            db.query(TestAttempt).filter_by(student_id=old.student_id).delete()
            db.query(StudentTopicState).filter_by(student_id=old.student_id).delete()
            db.delete(old)
            db.commit()
        db.close()

        r_up = client.post("/auth/signup", json={"name": "Vault Verifier", "email": email, "password": pw})
        assert r_up.status_code == 200
        up_body = r_up.json()
        sid = up_body["student_id"]
        jwt_tok = up_body["access_token"]
        vault_tok = up_body["vault_token"]
        assert jwt_tok and vault_tok

        r_on = client.post(
            f"/students/{sid}/onboarding",
            json={"class_level": "dropper", "target_neet_year": 2027},
            headers={"Authorization": f"Bearer {jwt_tok}", "X-Klassed-Vault": vault_tok},
        )
        assert r_on.status_code == 200
        on_body = r_on.json()
        assert on_body["onboarded"] is True
        vault_tok = on_body["vault_token"]
        assert vault_tok

        qs = client.get(
            f"/students/{sid}/next_paper?topic_id=Electrostatics",
            headers={"Authorization": f"Bearer {on_body['access_token']}"},
        ).json()
        journal_b64 = base64.urlsafe_b64encode(
            json.dumps([[1, qs[0]["question_id"], "a", 42, 0, 400, 42000]]).encode("utf-8")
        ).decode("ascii").rstrip("=")

        db = SessionLocal()
        db.query(TestAttempt).filter_by(student_id=sid).delete()
        db.query(StudentTopicState).filter_by(student_id=sid).delete()
        db.query(Student).filter_by(student_id=sid).delete()
        db.commit()
        db.close()

        r_wrong_pw = client.post(
            "/auth/login",
            json={"email": email, "password": "WrongPassword999!"},
            headers={"X-Klassed-Vault": vault_tok},
        )
        assert r_wrong_pw.status_code == 401

        db = SessionLocal()
        db.query(Student).filter_by(student_id=sid).delete()
        db.commit()
        db.close()

        tampered_vault = vault_tok[:-4] + "ZZZZ"
        r_tampered = client.post(
            "/auth/login",
            json={"email": email, "password": pw},
            headers={"X-Klassed-Vault": tampered_vault},
        )
        assert r_tampered.status_code == 401

        r_relogin = client.post(
            "/auth/login",
            json={"email": email, "password": pw},
            headers={"X-Klassed-Vault": vault_tok, "X-Klassed-Journal": journal_b64},
        )
        assert r_relogin.status_code == 200
        re_body = r_relogin.json()
        assert re_body["student_id"] == sid
        assert re_body["onboarded"] is True
        assert re_body["class_level"] == "dropper"

        r_ana = client.get(
            f"/students/{sid}/analytics",
            headers={"Authorization": f"Bearer {re_body['access_token']}"},
        )
        assert r_ana.status_code == 200 and r_ana.json()["total_attempts"] == 1
    finally:
        if sid:
            from model.linucb_bandit import reset_student_bandit
            reset_student_bandit(sid, remove=True)
            db = SessionLocal()
            db.query(TestAttempt).filter_by(student_id=sid).delete()
            db.query(StudentTopicState).filter_by(student_id=sid).delete()
            db.query(Student).filter_by(student_id=sid).delete()
            db.commit()
            db.close()

    assert not (PROJECT_ROOT / "data" / "tagged_questions" / "biology_questions.TEMPLATE.json").exists()
    assert not (PROJECT_ROOT / "data" / "taxonomy" / "biology_taxonomy.TEMPLATE.json").exists()
    with open(PROJECT_ROOT / "data" / "training" / "student_attempts.csv", encoding="utf-8") as f:
        att_rows = list(csv.DictReader(f))
    assert len(att_rows) == 0, f"Unexpected residual rows in student_attempts.csv: {len(att_rows)}"

    py_roots = [PROJECT_ROOT / "model", PROJECT_ROOT / "backend", PROJECT_ROOT / "api"]
    for root in py_roots:
        for py_file in root.rglob("*.py"):
            with tokenize.open(py_file) as tf:
                for tok in tokenize.generate_tokens(tf.readline):
                    assert tok.type != tokenize.COMMENT, f"Comment token found in {py_file}:{tok.start[0]}"
            tree = ast.parse(py_file.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    assert ast.get_docstring(node) is None, f"Docstring found in {py_file}"

    line_marker = "/" + "/"
    block_marker = "/" + "*"
    for ts_file in (PROJECT_ROOT / "frontend" / "src").rglob("*"):
        if ts_file.suffix not in (".ts", ".tsx") or ts_file.name == "vite-env.d.ts":
            continue
        for idx, line in enumerate(ts_file.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            assert not stripped.startswith(line_marker), f"Line comment in {ts_file}:{idx}"
            assert block_marker not in stripped, f"Block comment in {ts_file}:{idx}"


if __name__ == "__main__":
    suites = [
        test_01_random_forest_model_and_live_inference,
        test_02_topic_importance_exponential_decay,
        test_03_question_bank_and_weightage_validators,
        test_04_bkt_and_saint_telemetry_modulator,
        test_05_state_machine_all_four_transitions,
        test_06_pace_tracking_invariants,
        test_07_3pl_irt_mathematical_invariants,
        test_08_prerequisite_dag_acyclicity_and_shrinkage,
        test_09_linucb_bandit_uncertainty_and_reward,
        test_10_rag_socratic_engine_all_subjects,
        test_11_feature_store_and_drift_monitor,
        test_12_auth_bcrypt_and_jwt_security,
        test_13_database_counts_and_adaptive_paper_generator,
        test_14_all_11_endpoints_and_concurrent_stress,
        test_15_v3_advanced_psychometric_and_nlp_upgrades,
        test_16_deep_training_and_cross_validation_pipeline,
        test_17_harness_engineering_multi_persona_and_bandit_regret,
        test_18_cicd_pipeline_and_deployment_manifests,
        test_19_serverless_numpy_tfidf_and_scientific_trap_detector,
        test_20_all_65_chapters_end_to_end_curriculum_integrity,
        test_21_dina_8_skill_q_matrix_and_lognormal_rt_model,
        test_22_monte_carlo_conformal_simulator_and_graph_laplacian_leverage,
        test_23_scientific_notation_traps_paper_history_and_remediation_urgency,
        test_24_serverless_vault_relogin_compact_rf_parity_and_full_audit,
    ]
    for fn in suites:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(suites)} DEEP UNIT, INTEGRATION, HARNESS, TRAINING & CI/CD SUITES PASSED (100%)!")
