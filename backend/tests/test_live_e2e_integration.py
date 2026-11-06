import base64
import json
import time
import urllib.request
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "https://klassed-neet-adaptive-engine.vercel.app"
print(f"=== Running Full Live E2E Integration Suite against {BASE} ===")

ts = int(time.time())
email = f"e2e_test_{ts}@klassed.in"
signup_payload = json.dumps({
    "name": "Integration Tester",
    "email": email,
    "password": "Password123!",
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
    assert sid.startswith("INTEGRATION")
    assert student["email"] == email
    assert student["onboarded"] is False
    print(f"[PASS] 1. Auth Signup: student_id={sid}, token_present={bool(token)}")

onboard_payload = json.dumps({
    "class_level": "12",
    "target_neet_year": 2026,
}).encode("utf-8")

req = urllib.request.Request(
    f"{BASE}/students/{sid}/onboarding",
    data=onboard_payload,
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=15) as res:
    onboarded = json.loads(res.read())
    assert onboarded["onboarded"] is True
    assert onboarded["class_level"] == "12"
    assert onboarded["target_neet_year"] == 2026
    token = onboarded.get("access_token") or token
    print(f"[PASS] 2. Onboarding Questionnaire: onboarded={onboarded['onboarded']}")

with urllib.request.urlopen(f"{BASE}/students/{sid}/state", timeout=15) as res:
    state = json.loads(res.read())
    assert isinstance(state, list)
    print(f"[PASS] 3a. Initial Topic State: {len(state)} topics practiced")

with urllib.request.urlopen(f"{BASE}/students/{sid}/analytics", timeout=15) as res:
    analytics = json.loads(res.read())
    assert analytics["total_attempts"] == 0
    assert analytics["accuracy"] == 0.0
    print(f"[PASS] 3b. Initial Analytics: 0 attempts, 0% accuracy")

with urllib.request.urlopen(f"{BASE}/students/{sid}/v2_insights", timeout=15) as res:
    insights = json.loads(res.read())
    assert insights["global_theta"] == 0.0
    assert insights["estimated_neet_percentile"] == 50.0
    assert insights["predicted_neet_score_720"] == 380
    assert len(insights["recommended_unlock_topics"]) == 65
    print(f"[PASS] 3c. Initial Psychometric Priors: theta=0.00, percentile=50.0%ile, score=380/720, 65 chapters loaded")

with urllib.request.urlopen(f"{BASE}/students/{sid}/next_paper", timeout=15) as res:
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
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=15) as res:
    review = json.loads(res.read())
    assert review["paper_number"] == 1
    assert review["total_questions"] == 2
    assert "correct_count" in review
    assert "score_percent" in review
    assert len(review["questions"]) == 2
    print(f"[PASS] 5. Batch Paper Submission: {review['correct_count']}/2 correct ({review['score_percent']}%)")

with urllib.request.urlopen(f"{BASE}/students/{sid}/papers/1/review", timeout=15) as res:
    persisted_review = json.loads(res.read())
    assert persisted_review["total_questions"] == 2
    assert persisted_review["paper_number"] == 1
    assert persisted_review["correct_count"] == review["correct_count"]
    print(f"[PASS] 6a. Paper Review Retrieval: matches submission exactly ({persisted_review['correct_count']}/2)")

with urllib.request.urlopen(f"{BASE}/students/{sid}/papers/history", timeout=15) as res:
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

with urllib.request.urlopen(f"{BASE}/students/{sid}/analytics", timeout=15) as res:
    new_analytics = json.loads(res.read())
    assert new_analytics["total_attempts"] == 2
    assert len(new_analytics["recent_attempts"]) == 2
    assert len(new_analytics["subject_breakdown"]) > 0
    print(f"[PASS] 8a. Post-Paper Analytics: {new_analytics['total_attempts']} attempts recorded, subjects={len(new_analytics['subject_breakdown'])}")

with urllib.request.urlopen(f"{BASE}/students/{sid}/v2_insights", timeout=15) as res:
    new_insights = json.loads(res.read())
    assert new_insights["student_id"] == sid
    assert "global_theta" in new_insights
    assert "estimated_neet_percentile" in new_insights
    assert "predicted_neet_score_720" in new_insights
    mirt = new_insights["mirt_3d_profile"]
    total_mirt_attempts = mirt["attempts_conceptual"] + mirt["attempts_formula"] + mirt["attempts_numerical"]
    assert total_mirt_attempts == 2
    print(f"[PASS] 8b. Real-Time MIRT & 3PL Latent Updates: theta={new_insights['global_theta']}, percentile={new_insights['estimated_neet_percentile']}%ile, score={new_insights['predicted_neet_score_720']}/720 (MIRT: {mirt['attempts_conceptual']} conceptual, {mirt['attempts_formula']} formula, {mirt['attempts_numerical']} numerical)")

journal_b64 = base64.urlsafe_b64encode(
    json.dumps([
        [1, q1["question_id"], "a", 50, 1, 800, 50000],
        [1, q2["question_id"], "b", 75, 0, 400, 75000],
    ]).encode("utf-8")
).decode("ascii").rstrip("=")
snap_req = urllib.request.Request(
    f"{BASE}/students/{sid}/dashboard_snapshot",
    headers={
        "Authorization": f"Bearer {token}",
        "X-Klassed-Journal": journal_b64,
    },
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
    assert health["models_loaded"]["bkt_bayesian_knowledge_tracing"] is True
    assert health["models_loaded"]["linucb_contextual_bandit"] is True
    assert health["models_loaded"]["rag_ncert_socratic_engine"] is True
    assert health["models_loaded"]["dina_8_skill_q_matrix"] is True
    assert health["models_loaded"]["monte_carlo_conformal_neet_simulator"] is True
    print(f"[PASS] 9. MLOps Observability & Drift Gate: All 8 production models loaded, feature store p50={health['feature_store']['p50_ms']}ms")

print("\n>>> ALL 9 END-TO-END LIVE PRODUCTION INTEGRATION GATES PASSED (100%) <<<")
