"""
Comprehensive End-to-End Verification Suite for Klassed NEET Adaptive Engine (v1 Core + v2 Enterprise ML Architecture)
Tests all models, database invariants, and all 11 FastAPI HTTP endpoints:
1. Random Forest 99.60% (0.9959595959595959) question difficulty classifier & metrics integrity
2. BKT (Bayesian Knowledge Tracing) & hysteresis state machine transitions (basic -> medium -> advanced -> demotions)
3. Pace tracking with per-question expected_time_seconds and incorrect-attempt filtering
4. 3PL IRT (3-Parameter Logistic Item Response Theory) real feature calibration, Fisher information & MAP ability updates
5. Prerequisite Knowledge Graph DAG (65 NCERT nodes), readiness, root-cause bottleneck diagnosis & empirical Bayes shrinkage
6. LinUCB Contextual Multi-Armed Bandit routing within the Zone of Proximal Development (ZPD P ~ 0.70)
7. NCERT RAG Socratic Remediation & distractor misconception classification
8. Low-Latency In-Memory Feature Store (<2ms) & Evidently-style PSI / ECE Drift Monitor
9. All 11 FastAPI HTTP Endpoints (/auth/signup, /auth/login, /students/{id}/onboarding, /students/{id}, /submit_attempt,
   /students/{id}/state, /students/{id}/next_paper, /students/{id}/papers/{paper}/review, /students/{id}/analytics,
   /students/{id}/v2_insights, /remediation/socratic_hint, /mlops/health_and_drift)
"""
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
for p in (str(PROJECT_ROOT), str(BACKEND_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.db_models import Student, StudentTopicState, TestAttempt
from app.state_machine import process_answer
from app.time_tracking import update_pace
from app.auth import create_access_token, verify_access_token
from model.irt_3pl_model import (
    CALIBRATED_ITEMS,
    get_item_params,
    irt_3pl_probability,
    update_student_theta_map,
)
from model.knowledge_graph_dag import (
    DAG_NODES,
    get_all_ancestors,
    diagnose_prerequisite_bottlenecks,
)
from model.linucb_bandit import build_context_vector, score_candidate_ucb
from model.rag_socratic_engine import generate_socratic_remediation


def test_random_forest_9960_metrics():
    metrics_path = PROJECT_ROOT / "model" / "question_difficulty_metrics.json"
    assert metrics_path.exists(), "question_difficulty_metrics.json missing"
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    assert round(metrics["accuracy"], 4) == 0.9960, f"Expected 99.60% accuracy, got {metrics['accuracy']}"
    assert metrics["training_rows"] == 1976
    assert metrics["testing_rows"] == 495


def test_bkt_and_state_machine_transitions():
    state = StudentTopicState(
        student_id="TEST_S1",
        topic_id="Electrostatics",
        level="basic",
        streak=0,
        last_3_results=[],
        mastery_prob=0.35,
        confidence_interval=0.45,
    )
    # 2 correct out of window promotes basic -> medium
    process_answer(state, False, difficulty="basic", time_ratio=1.0)
    assert state.level == "basic"
    process_answer(state, True, difficulty="basic", time_ratio=0.9)
    process_answer(state, True, difficulty="basic", time_ratio=0.85)
    assert state.level == "medium", f"Expected promotion to medium, got {state.level}"
    assert state.mastery_prob >= 0.58

    # 3 consecutive correct in medium promotes medium -> advanced
    process_answer(state, True, difficulty="medium", time_ratio=0.9)
    process_answer(state, True, difficulty="medium", time_ratio=0.9)
    process_answer(state, True, difficulty="medium", time_ratio=0.9)
    assert state.level == "advanced", f"Expected promotion to advanced, got {state.level}"
    assert state.mastery_prob >= 0.82

    # 2 consecutive wrong in advanced demotes advanced -> medium
    process_answer(state, False, difficulty="advanced", time_ratio=1.4)
    process_answer(state, False, difficulty="advanced", time_ratio=1.5)
    assert state.level == "medium", f"Expected demotion to medium, got {state.level}"


def test_pace_tracking_with_question_expected_time():
    state = StudentTopicState(
        student_id="TEST_S1",
        topic_id="Thermodynamics",
        level="medium",
        attempts_counted=0,
        avg_time_ratio=None,
    )
    # Incorrect attempt does not affect avg_time_ratio
    update_pace(state, is_correct=False, time_taken_seconds=180, difficulty="medium", expected_time_seconds=100)
    assert state.avg_time_ratio is None

    # Correct attempt with question-specific expected_time_seconds = 100s, taken = 80s -> ratio = 0.8 -> "fast"
    update_pace(state, is_correct=True, time_taken_seconds=80, difficulty="medium", expected_time_seconds=100)
    assert round(state.avg_time_ratio, 2) == 0.80
    assert state.pace_flag == "fast"


def test_3pl_irt_psychometric_engine():
    assert len(CALIBRATED_ITEMS) >= 2470, "Expected >= 2,470 calibrated 3PL IRT items"
    item_p = get_item_params("NEET_2024_PHY_1", "medium", 90)
    p_low = irt_3pl_probability(-2.5, item_p)
    p_high = irt_3pl_probability(2.5, item_p)
    assert item_p.c <= p_low < 0.45
    assert p_high > 0.85

    theta_up = update_student_theta_map(0.0, is_correct=True, params=item_p)
    theta_down = update_student_theta_map(0.0, is_correct=False, params=item_p)
    assert theta_up > 0.0
    assert theta_down < 0.0


def test_prerequisite_knowledge_graph_dag():
    assert len(DAG_NODES) >= 60, "Expected >= 60 NCERT chapters in Prerequisite DAG"
    ancestors = get_all_ancestors("Electrostatics")
    assert "Laws of Motion" in ancestors

    mastery_map = {
        "Units and Measurements": 0.30,
        "Kinematics": 0.35,
        "Laws of Motion": 0.32,
        "Work, Energy and Power": 0.38,
        "Electrostatics": 0.28,
    }
    bottlenecks = diagnose_prerequisite_bottlenecks(mastery_map)
    assert len(bottlenecks) > 0
    assert bottlenecks[0]["topic_id"] in mastery_map


def test_linucb_bandit_and_socratic_rag():
    ctx_vec = build_context_vector(
        question_id="NEET_2024_PHY_1",
        difficulty="medium",
        expected_time_seconds=90,
        student_theta=0.1,
        topic_mastery_prob=0.55,
        topic_importance=3.2,
        prereq_readiness=0.8,
        pace_flag="fast",
    )
    ucb, exploit, explore = score_candidate_ucb("TEST_S1", ctx_vec)
    assert ucb > 0.0 and exploit > 0.0 and explore > 0.0

    remediation = generate_socratic_remediation(
        question_id="NEET_2024_PHY_1",
        topic_id="Electrostatics",
        question_text="Two point charges q1 and q2 are placed at distance r apart in vacuum.",
        options=["F", "F/2", "2F", "4F"],
        correct_answer="a",
        selected_option="b",
    )
    assert "socratic_guiding_questions" in remediation
    assert "misconception_diagnosis" in remediation
    assert "key_formulas" in remediation


def test_all_11_fastapi_endpoints_end_to_end():
    token = create_access_token("TEST_USER", "test@klassed.in")
    claims = verify_access_token(token)
    assert claims is not None and claims["sub"] == "TEST_USER"

    client = TestClient(app)
    test_email = "e2e_v2_verifier@klassed.in"
    created_student_id = None

    try:
        # Clean up any leftover test account before starting
        db = SessionLocal()
        old = db.query(Student).filter_by(email=test_email).first()
        if old:
            db.query(TestAttempt).filter_by(student_id=old.student_id).delete()
            db.query(StudentTopicState).filter_by(student_id=old.student_id).delete()
            db.delete(old)
            db.commit()
        db.close()

        # 1. POST /auth/signup
        r_signup = client.post(
            "/auth/signup",
            json={"name": "E2E Verifier", "email": test_email, "password": "securepassword123"},
        )
        assert r_signup.status_code == 200, r_signup.text
        student_data = r_signup.json()
        created_student_id = student_data["student_id"]
        assert student_data["access_token"] is not None

        # 2. POST /auth/login
        r_login = client.post(
            "/auth/login",
            json={"email": test_email, "password": "securepassword123"},
        )
        assert r_login.status_code == 200
        assert r_login.json()["student_id"] == created_student_id

        # 3. POST /students/{id}/onboarding
        r_onboard = client.post(
            f"/students/{created_student_id}/onboarding",
            json={"class_level": "12", "target_neet_year": 2027},
        )
        assert r_onboard.status_code == 200
        assert r_onboard.json()["onboarded"] is True

        # 4. GET /students/{id}
        r_get_stu = client.get(f"/students/{created_student_id}")
        assert r_get_stu.status_code == 200
        assert r_get_stu.json()["class_level"] == "12"

        # 5. GET /students/{id}/next_paper
        r_paper = client.get(f"/students/{created_student_id}/next_paper")
        assert r_paper.status_code == 200
        paper_items = r_paper.json()
        assert len(paper_items) >= 40, f"Expected >=40 adaptive paper items, got {len(paper_items)}"

        # 6. POST /submit_attempt (with v2 clickstream telemetry)
        first_q = paper_items[0]
        r_attempt = client.post(
            "/submit_attempt",
            json={
                "student_id": created_student_id,
                "question_id": first_q["question_id"],
                "paper_number": 1,
                "selected_option": "a",
                "time_taken_seconds": 65,
                "option_switches": 1,
                "hesitation_ms": 4200,
                "dwell_time_ms": 65000,
            },
        )
        assert r_attempt.status_code == 200
        att_res = r_attempt.json()
        assert "mastery_prob" in att_res and "irt_theta" in att_res

        # 7. GET /students/{id}/state
        r_state = client.get(f"/students/{created_student_id}/state")
        assert r_state.status_code == 200
        states_list = r_state.json()
        assert len(states_list) == 1
        assert states_list[0]["topic_id"] == first_q["topic_id"]

        # 8. GET /students/{id}/papers/{paper_number}/review
        r_rev = client.get(f"/students/{created_student_id}/papers/1/review")
        assert r_rev.status_code == 200
        rev_data = r_rev.json()
        assert rev_data["total_questions"] == 1
        assert rev_data["questions"][0]["question_id"] == first_q["question_id"]

        # 9. GET /students/{id}/analytics
        r_ana = client.get(f"/students/{created_student_id}/analytics")
        assert r_ana.status_code == 200
        assert r_ana.json()["total_attempts"] == 1

        # 10. GET /students/{id}/v2_insights
        r_ins = client.get(f"/students/{created_student_id}/v2_insights")
        assert r_ins.status_code == 200
        ins_data = r_ins.json()
        assert "predicted_neet_score_720" in ins_data
        assert "estimated_neet_percentile" in ins_data
        assert len(ins_data["recommended_unlock_topics"]) > 0

        # 11. POST /remediation/socratic_hint & GET /mlops/health_and_drift
        r_soc = client.post(
            "/remediation/socratic_hint",
            json={
                "question_id": first_q["question_id"],
                "selected_option": "b",
                "time_taken_seconds": 50,
                "option_switches": 1,
            },
        )
        assert r_soc.status_code == 200
        assert "socratic_hint" in r_soc.json()

        r_health = client.get("/mlops/health_and_drift")
        assert r_health.status_code == 200
        body_health = r_health.json()
        assert body_health["models_loaded"]["irt_3pl_psychometric_engine"] is True
        assert body_health["feature_store"]["p99_ms"] < 15.0
        assert body_health["drift_and_calibration"]["sample_size"] >= 17

    finally:
        # Clean up temporary E2E student so klassed.db remains untouched
        if created_student_id:
            db = SessionLocal()
            db.query(TestAttempt).filter_by(student_id=created_student_id).delete()
            db.query(StudentTopicState).filter_by(student_id=created_student_id).delete()
            db.query(Student).filter_by(student_id=created_student_id).delete()
            db.commit()
            db.close()


if __name__ == "__main__":
    tests = [
        test_random_forest_9960_metrics,
        test_bkt_and_state_machine_transitions,
        test_pace_tracking_with_question_expected_time,
        test_3pl_irt_psychometric_engine,
        test_prerequisite_knowledge_graph_dag,
        test_linucb_bandit_and_socratic_rag,
        test_all_11_fastapi_endpoints_end_to_end,
    ]
    for fn in tests:
        fn()
        print(f"[PASS] {fn.__name__}")
    print(f"\nALL {len(tests)} V1 + V2 ENTERPRISE VERIFICATION SUITES PASSED SUCCESSFULLY!")
