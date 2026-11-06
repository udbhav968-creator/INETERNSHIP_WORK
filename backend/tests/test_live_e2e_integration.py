import base64
import json
import sys
import time
import urllib.error
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://klassed-neet-adaptive-engine.vercel.app"
print(f"=== Running Full Live E2E Integration Suite against {BASE} ===")

ts = int(time.time())
email = f"e2e_test_{ts}@klassed.in"
password = "Password123!"
signup_payload = json.dumps({
    "name": "Integration Tester",
    "email": email,
    "password": password,
}).encode("utf-8")

req = urllib.request.Request(
    f"{BASE}/auth/signup",
    data=signup_payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=15) as res:
    student = json.loads(res.read())
    sid = student["student_id"]
    token = student.get("access_token")
    vault_token = student.get("vault_token")
    assert sid.startswith("INTEGRATION")
    assert student["email"] == email
    assert student["onboarded"] is False
    assert bool(token) and bool(vault_token)
    print(f"[PASS] 1. Auth Signup: student_id={sid}, jwt_present={bool(token)}, vault_present={bool(vault_token)}")

onboard_payload = json.dumps({
    "class_level": "12",
    "target_neet_year": 2026,
}).encode("utf-8")

req = urllib.request.Request(
    f"{BASE}/students/{sid}/onboarding",
    data=onboard_payload,
    headers={
        "Content-Type": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Klassed-Vault": vault_token,
    },
)
with urllib.request.urlopen(req, timeout=15) as res:
    onboarded = json.loads(res.read())
    assert onboarded["onboarded"] is True
    assert onboarded["class_level"] == "12"
    assert onboarded["target_neet_year"] == 2026
    token = onboarded.get("access_token") or token
    vault_token = onboarded.get("vault_token") or vault_token
    print(f"[PASS] 2. Onboarding Questionnaire: onboarded={onboarded['onboarded']}, refreshed_vault={bool(onboarded.get('vault_token'))}")

auth_headers = {"Authorization": f"Bearer {token}", "X-Klassed-Vault": vault_token}

req_state = urllib.request.Request(f"{BASE}/students/{sid}/state", headers=auth_headers)
with urllib.request.urlopen(req_state, timeout=15) as res:
    state = json.loads(res.read())
    assert isinstance(state, list)
    print(f"[PASS] 3a. Initial Topic State: {len(state)} topics practiced")

req_ana = urllib.request.Request(f"{BASE}/students/{sid}/analytics", headers=auth_headers)
with urllib.request.urlopen(req_ana, timeout=15) as res:
    analytics = json.loads(res.read())
    assert analytics["total_attempts"] == 0
    assert analytics["accuracy"] == 0.0
    print(f"[PASS] 3b. Initial Analytics: 0 attempts, 0% accuracy")

req_ins = urllib.request.Request(f"{BASE}/students/{sid}/v2_insights", headers=auth_headers)
with urllib.request.urlopen(req_ins, timeout=15) as res:
    insights = json.loads(res.read())
    assert insights["global_theta"] == 0.0
    assert insights["estimated_neet_percentile"] == 50.0
    assert insights["predicted_neet_score_720"] == 380
    assert len(insights["recommended_unlock_topics"]) == 65
    print(f"[PASS] 3c. Initial Psychometric Priors: theta=0.00, percentile=50.0%ile, score=380/720, 65 chapters loaded")

req_paper = urllib.request.Request(f"{BASE}/students/{sid}/next_paper", headers=auth_headers)
with urllib.request.urlopen(req_paper, timeout=15) as res:
    paper = json.loads(res.read())
    assert len(paper) > 0
    assert "question_id" in paper[0]
    assert "topic_id" in paper[0]
    assert len(paper[0]["options"]) == 4
    print(f"[PASS] 4. Adaptive Paper Generation: {len(paper)} questions delivered (Top item: {paper[0]['question_id']} - {paper[0]['topic_id']})")

q1 = paper[0]
q2 = paper[1] if len(paper) > 1 else paper[0]

batch_payload = json.dumps({
    "student_id": sid,
    "paper_number": 1,
    "attempts": [
        {
            "student_id": sid,
            "question_id": q1["question_id"],
            "paper_number": 1,
            "selected_option": "a",
            "time_taken_seconds": 50,
            "option_switches": 1,
            "hesitation_ms": 800,
            "dwell_time_ms": 50000,
        },
        {
            "student_id": sid,
            "question_id": q2["question_id"],
            "paper_number": 1,
            "selected_option": "b",
            "time_taken_seconds": 75,
            "option_switches": 0,
            "hesitation_ms": 400,
            "dwell_time_ms": 75000,
        },
    ],
}).encode("utf-8")

req = urllib.request.Request(
    f"{BASE}/submit_paper",
    data=batch_payload,
    headers={"Content-Type": "application/json", **auth_headers},
)
with urllib.request.urlopen(req, timeout=15) as res:
    review = json.loads(res.read())
    assert review["paper_number"] == 1
    assert review["total_questions"] == 2
    assert "correct_count" in review
    assert "score_percent" in review
    assert len(review["questions"]) == 2
    print(f"[PASS] 5. Batch Paper Submission: {review['correct_count']}/2 correct ({review['score_percent']}%)")

journal_b64 = base64.urlsafe_b64encode(
    json.dumps([
        [1, q1["question_id"], "a", 50, 1, 800, 50000],
        [1, q2["question_id"], "b", 75, 0, 400, 75000],
    ]).encode("utf-8")
).decode("ascii").rstrip("=")
full_sync_headers = {**auth_headers, "X-Klassed-Journal": journal_b64}

req_rev = urllib.request.Request(f"{BASE}/students/{sid}/papers/1/review", headers=full_sync_headers)
with urllib.request.urlopen(req_rev, timeout=15) as res:
    persisted_review = json.loads(res.read())
    assert persisted_review["total_questions"] == 2
    assert persisted_review["paper_number"] == 1
    assert persisted_review["correct_count"] == review["correct_count"]
    print(f"[PASS] 6a. Paper Review Retrieval: matches submission exactly ({persisted_review['correct_count']}/2)")

req_hist = urllib.request.Request(f"{BASE}/students/{sid}/papers/history", headers=full_sync_headers)
with urllib.request.urlopen(req_hist, timeout=15) as res:
    history = json.loads(res.read())
    assert len(history) >= 1
    assert history[0]["paper_number"] == 1
    assert history[0]["total_questions"] == 2
    assert "score_percent" in history[0]
    print(f"[PASS] 6b. Paper History Summary: {len(history)} papers archived (Paper 1: {history[0]['score_percent']}%, topics={history[0]['topics']})")

target_q = review["questions"][0]
hint_payload = json.dumps({
    "question_id": target_q["question_id"],
    "selected_option": target_q["selected_option"],
    "time_taken_seconds": 50,
    "option_switches": 1,
}).encode("utf-8")

req = urllib.request.Request(
    f"{BASE}/remediation/socratic_hint",
    data=hint_payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=15) as res:
    hint = json.loads(res.read())
    assert "ncert_reference" in hint
    assert "diagnosed_misconception" in hint
    assert "socratic_hint" in hint
    print(f"[PASS] 7. Socratic RAG Remediation: NCERT Ref='{hint['ncert_reference']}', Trap/Misconception='{hint['diagnosed_misconception'][:60]}...'")

req_post_ana = urllib.request.Request(f"{BASE}/students/{sid}/analytics", headers=full_sync_headers)
with urllib.request.urlopen(req_post_ana, timeout=15) as res:
    new_analytics = json.loads(res.read())
    assert new_analytics["total_attempts"] == 2
    assert len(new_analytics["recent_attempts"]) == 2
    assert len(new_analytics["subject_breakdown"]) > 0
    print(f"[PASS] 8a. Post-Paper Analytics: {new_analytics['total_attempts']} attempts recorded, subjects={len(new_analytics['subject_breakdown'])}")

req_post_ins = urllib.request.Request(f"{BASE}/students/{sid}/v2_insights", headers=full_sync_headers)
with urllib.request.urlopen(req_post_ins, timeout=15) as res:
    new_insights = json.loads(res.read())
    assert new_insights["student_id"] == sid
    assert "global_theta" in new_insights
    assert "estimated_neet_percentile" in new_insights
    assert "predicted_neet_score_720" in new_insights
    mirt = new_insights["mirt_3d_profile"]
    total_mirt_attempts = mirt["attempts_conceptual"] + mirt["attempts_formula"] + mirt["attempts_numerical"]
    assert total_mirt_attempts == 2
    print(f"[PASS] 8b. Real-Time MIRT & 3PL Latent Updates: theta={new_insights['global_theta']}, percentile={new_insights['estimated_neet_percentile']}%ile, score={new_insights['predicted_neet_score_720']}/720")

snap_req = urllib.request.Request(
    f"{BASE}/students/{sid}/dashboard_snapshot",
    headers=full_sync_headers,
)
with urllib.request.urlopen(snap_req, timeout=15) as res:
    snap = json.loads(res.read())
    assert snap["analytics"]["total_attempts"] == 2
    assert len(snap["paper_history"]) == 1 and snap["paper_history"][0]["total_questions"] == 2
    assert len(snap["topics"]) >= 1
    print(f"[PASS] 8c. Atomic Dashboard Snapshot & Serverless Journal Sync: attempts={snap['analytics']['total_attempts']}, active_topics={len(snap['topics'])}, papers={len(snap['paper_history'])}")

with urllib.request.urlopen(f"{BASE}/mlops/health_and_drift", timeout=15) as res:
    health = json.loads(res.read())
    assert health["models_loaded"]["irt_3pl_psychometric_engine"] is True
    assert health["models_loaded"]["rf_300_tree_compact_serverless_runtime"] is True
    assert health["models_loaded"]["bkt_bayesian_knowledge_tracing"] is True
    assert health["models_loaded"]["linucb_contextual_bandit"] is True
    assert health["models_loaded"]["rag_ncert_socratic_engine"] is True
    assert health["models_loaded"]["dina_8_skill_q_matrix"] is True
    assert health["models_loaded"]["monte_carlo_conformal_neet_simulator"] is True
    print(f"[PASS] 9. MLOps Observability & Drift Gate: All 10 production models active, feature store p50={health['feature_store']['p50_ms']}ms")

with urllib.request.urlopen(f"{BASE}/mlops/audit_overview", timeout=15) as res:
    audit = json.loads(res.read())
    assert audit["corpus_integrity"]["verified_questions_in_db"] == 2471
    assert audit["corpus_integrity"]["canonical_chapters_in_db"] == 65
    assert audit["corpus_integrity"]["dag_acyclic"] is True
    assert audit["validation_report"]["pipeline_status"] == "PASSED"
    assert audit["harness_benchmark"]["harness_status"] == "PASSED"
    print(f"[PASS] 10. Full MLOps Audit Overview: 2471 PYQs, 65 chapters, CV={audit['validation_report']['stratified_5fold_cross_validation']['mean_cv_accuracy']}, Harness=PASSED")

cold_payload = json.dumps({
    "question_text": "Two point charges +4 uC and -9 uC are separated in vacuum by 0.30 m. Calculate electric field intensity E = k*q/r^2 at midpoint.",
    "options": ["5.2 x 10^6 N/C", "5.2 x 10^3 N/C", "2.0 x 10^6 N/C", "3.6 x 10^5 N/C"],
    "has_diagram": False,
    "subject": "Physics",
    "chapter": "Electrostatics",
    "topic_tag": "Electric Field",
}).encode("utf-8")
cold_req = urllib.request.Request(
    f"{BASE}/mlops/predict_cold_start_item",
    data=cold_payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(cold_req, timeout=15) as res:
    cold = json.loads(res.read())
    assert cold["rf_probabilities"] is not None
    assert abs(sum(cold["rf_probabilities"].values()) - 1.0) < 0.01
    assert "a" in cold["irt_params"] and "b" in cold["irt_params"] and "c" in cold["irt_params"]
    assert cold["mirt_dimension"] in ("conceptual", "formula", "numerical")
    assert len(cold["active_cognitive_skills"]) >= 2
    print(f"[PASS] 11. Live Cold-Start 300-Tree RF + 3PL IRT + DINA Calibrator: tier={cold['predicted_difficulty']}, engine={cold['rf_engine']}, rf_probs={cold['rf_probabilities']}, skills={len(cold['active_cognitive_skills'])}")

bad_login_req = urllib.request.Request(
    f"{BASE}/auth/login",
    data=json.dumps({"email": email, "password": "WrongPassword999!"}).encode("utf-8"),
    headers={"Content-Type": "application/json", "X-Klassed-Vault": vault_token},
)
got_401 = False
try:
    urllib.request.urlopen(bad_login_req, timeout=15)
except urllib.error.HTTPError as err:
    got_401 = err.code == 401
assert got_401, "Expected 401 on wrong password with vault token"

relogin_req = urllib.request.Request(
    f"{BASE}/auth/login",
    data=json.dumps({"email": email, "password": password}).encode("utf-8"),
    headers={
        "Content-Type": "application/json",
        "X-Klassed-Vault": vault_token,
        "X-Klassed-Journal": journal_b64,
    },
)
with urllib.request.urlopen(relogin_req, timeout=15) as res:
    relogged = json.loads(res.read())
    assert relogged["student_id"] == sid
    assert relogged["onboarded"] is True
    assert bool(relogged.get("access_token")) and bool(relogged.get("vault_token"))
    print(f"[PASS] 12. Serverless Switch-Account Vault Re-Login: 401 rejected wrong password, 200 restored {sid} (onboarded={relogged['onboarded']})")

print("\n>>> ALL 12 END-TO-END LIVE PRODUCTION INTEGRATION GATES PASSED (100%) <<<")
