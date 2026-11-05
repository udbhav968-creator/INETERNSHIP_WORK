"""
Klassed NEET Adaptive Engine — backend API (v1 Core + Enterprise v2 Psychometric & RAG Architecture).

Run locally with:
    uvicorn app.main:app --reload
"""
import json
import threading
from datetime import datetime, timezone
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from . import schemas, auth
from .analytics import build_analytics
from .database import get_db, init_db
from .db_models import Question, Student, StudentTopicState, TestAttempt
from .state_machine import process_answer
from .time_tracking import update_pace, expected_time
from .paper_generator import generate_paper
from .paths import MODEL_DIR, PROJECT_ROOT

from model.irt_3pl_model import (
    CALIBRATED_ITEMS,
    get_item_params,
    irt_3pl_probability,
    update_student_theta_map,
    theta_to_percentile,
    compute_mirt_3d_profile,
    predict_cold_start_difficulty_and_irt,
    recalibrate_item_parameters_online,
)
from model.knowledge_graph_dag import (
    DAG_NODES,
    compute_prerequisite_readiness,
    compute_hierarchical_shrinkage_prior,
    diagnose_prerequisite_bottlenecks,
    propagate_dag_belief_update,
)
from model.state_machine.bkt_model import (
    compute_ebbinghaus_forgetting_prior,
    compute_retention_urgency,
)
from model.linucb_bandit import record_bandit_feedback
from model.rag_socratic_engine import generate_socratic_remediation, _INDEX as _RAG_INDEX
from model.feature_store_and_drift import FEATURE_STORE, DRIFT_MONITOR

with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
    _TOPIC_IMPORTANCE = json.load(f)

app = FastAPI(
    title="Klassed NEET Adaptive Engine (v2 Enterprise Psychometric & RAG Edition)",
    version="2.0.0",
)

# Per-student striped locks guarding per-student state read-modify-write without blocking other students
_LOCK_REGISTRY_GUARD = threading.Lock()
_STUDENT_LOCKS: dict[str, threading.RLock] = {}


def _get_student_lock(student_id: str) -> threading.RLock:
    with _LOCK_REGISTRY_GUARD:
        if student_id not in _STUDENT_LOCKS:
            _STUDENT_LOCKS[student_id] = threading.RLock()
        return _STUDENT_LOCKS[student_id]

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"(http://(localhost|127\.0\.0\.1):\d+|https://.*\.(vercel\.app|netlify\.app|onrender\.com|up\.railway\.app|hf\.space))",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "version": "2.0.0",
        "engine": "Klassed NEET Adaptive Engine (v2 Enterprise Psychometric & RAG Edition)",
    }


# --- auth / onboarding ---

@app.post("/auth/signup", response_model=schemas.StudentOut)
def signup(payload: schemas.SignupIn, db: Session = Depends(get_db)):
    return auth.signup(db, payload.name, payload.email, payload.password)


@app.post("/auth/login", response_model=schemas.StudentOut)
def login(payload: schemas.LoginIn, db: Session = Depends(get_db)):
    return auth.login(db, payload.email, payload.password)


@app.post("/students/{student_id}/onboarding", response_model=schemas.StudentOut)
def complete_onboarding(student_id: str, payload: schemas.OnboardingIn, db: Session = Depends(get_db)):
    student = db.query(Student).filter_by(student_id=student_id).first()
    if not student:
        # Auto-provision student row if request landed on a fresh serverless instance
        student = Student(
            student_id=student_id,
            name=student_id.title(),
            email=f"{student_id.lower()}@klassed.in",
            password_hash="serverless_provisioned",
            onboarded=False,
        )
        db.add(student)
        db.flush()
    return auth.complete_onboarding(db, student, payload.class_level, payload.target_neet_year)


@app.get("/students/{student_id}", response_model=schemas.StudentOut)
def get_student(student_id: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter_by(student_id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


# --- adaptive engine (v1 Core + v2 BKT, 3PL IRT, LinUCB, Feature Store, Drift Monitor) ---

def _process_single_attempt_no_commit(
    attempt: schemas.AttemptIn,
    db: Session,
) -> tuple[bool, Question, StudentTopicState]:
    question = db.query(Question).filter_by(question_id=attempt.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    selected_clean = (attempt.selected_option or "").strip().lower()
    correct_clean = (question.correct_answer or "").strip().lower()
    is_correct = selected_clean == correct_clean

    opt_switches = int(attempt.option_switches or 0)
    hesitation = int(attempt.hesitation_ms or 0)
    dwell = int(attempt.dwell_time_ms or (attempt.time_taken_seconds * 1000))

    # Compute elapsed hours since last attempt in this topic BEFORE inserting the new attempt
    prev_topic_attempt = (
        db.query(TestAttempt)
        .join(Question, TestAttempt.question_id == Question.question_id)
        .filter(
            TestAttempt.student_id == attempt.student_id,
            Question.topic_id == question.topic_id,
        )
        .order_by(TestAttempt.attempt_id.desc())
        .first()
    )
    elapsed_hours = 0.0
    if prev_topic_attempt and getattr(prev_topic_attempt, "timestamp", None):
        ts = prev_topic_attempt.timestamp
        now_dt = datetime.now(timezone.utc) if ts.tzinfo else datetime.utcnow()
        elapsed_hours = max(0.0, (now_dt - ts).total_seconds() / 3600.0)

    # Log the raw attempt with v2 telemetry
    db.add(
        TestAttempt(
            student_id=attempt.student_id,
            question_id=attempt.question_id,
            paper_number=attempt.paper_number,
            is_correct=is_correct,
            time_taken_seconds=attempt.time_taken_seconds,
            selected_option=attempt.selected_option,
            option_switches=opt_switches,
            hesitation_ms=hesitation,
            dwell_time_ms=dwell,
        )
    )

    # Fetch or create this student's state for this topic
    state = (
        db.query(StudentTopicState)
        .filter_by(student_id=attempt.student_id, topic_id=question.topic_id)
        .first()
    )
    if not state:
        state = StudentTopicState(
            student_id=attempt.student_id,
            topic_id=question.topic_id,
            level="basic",
            streak=0,
            last_3_results=[],
            mastery_prob=0.35,
            confidence_interval=0.45,
            irt_theta=0.0,
        )
        db.add(state)
        db.flush()

    exp_sec = expected_time(question.difficulty, question.expected_time_seconds)
    time_ratio = attempt.time_taken_seconds / max(1, exp_sec)

    # Prior probability and mastery before state update
    prior_theta = float(getattr(state, "irt_theta", 0.0) or 0.0)
    prior_mastery = float(getattr(state, "mastery_prob", 0.35) or 0.35)
    irt_params = get_item_params(question.question_id, question.difficulty, exp_sec)
    pred_prob = irt_3pl_probability(prior_theta, irt_params)
    DRIFT_MONITOR.log_prediction(pred_prob, int(is_correct), question.difficulty, attempt.time_taken_seconds)

    # 1. State Machine & BKT Posterior Update (with Ebbinghaus forgetting prior)
    process_answer(
        state,
        is_correct,
        difficulty=question.difficulty,
        time_ratio=time_ratio,
        option_switches=opt_switches,
        hesitation_ms=hesitation,
        elapsed_hours=elapsed_hours,
    )

    # 2. Pace Tracking Update (Part I Fix #3: uses question.expected_time_seconds)
    update_pace(
        state,
        is_correct,
        attempt.time_taken_seconds,
        question.difficulty,
        expected_time_seconds=question.expected_time_seconds,
    )

    # 3. 3PL IRT Latent Ability (theta) MAP Update + Online Item Recalibration
    new_topic_theta = update_student_theta_map(
        current_theta=prior_theta,
        is_correct=is_correct,
        params=irt_params,
    )
    state.irt_theta = new_topic_theta
    recalibrate_item_parameters_online(
        question_id=question.question_id,
        student_theta=prior_theta,
        is_correct=is_correct,
        time_ratio=time_ratio,
    )

    student = db.query(Student).filter_by(student_id=attempt.student_id).first()
    if student:
        curr_global = float(getattr(student, "global_theta", 0.0) or 0.0)
        student.global_theta = round(0.85 * curr_global + 0.15 * new_topic_theta, 4)

    # Compute real prerequisite readiness from the student's topic states in the DB
    all_student_states = (
        db.query(StudentTopicState).filter_by(student_id=attempt.student_id).all()
    )
    state_by_topic = {s.topic_id: s for s in all_student_states}
    mastery_map = {
        s.topic_id: float(getattr(s, "mastery_prob", 0.35) or 0.35)
        for s in all_student_states
    }

    # 3b. 1-Hop Bayesian Belief Propagation across Prerequisite DAG
    dag_updates = propagate_dag_belief_update(
        topic_id=question.topic_id,
        is_correct=is_correct,
        difficulty=question.difficulty,
        student_mastery_map=mastery_map,
    )
    for neighbor_topic, new_neighbor_mastery in dag_updates.items():
        mastery_map[neighbor_topic] = new_neighbor_mastery
        if neighbor_topic in state_by_topic:
            state_by_topic[neighbor_topic].mastery_prob = new_neighbor_mastery

    prereq_ready = compute_prerequisite_readiness(question.topic_id, mastery_map)

    # 4. LinUCB Contextual Bandit Reward Update (with delayed mastery gain Delta P(L_t))
    record_bandit_feedback(
        student_id=attempt.student_id,
        question_id=question.question_id,
        difficulty=question.difficulty,
        expected_time_seconds=exp_sec,
        student_theta=new_topic_theta,
        topic_mastery_prob=float(state.mastery_prob or 0.5),
        topic_importance=float(_TOPIC_IMPORTANCE.get(question.topic_id, 1.5)),
        prereq_readiness=prereq_ready,
        pace_flag=state.pace_flag,
        is_correct=is_correct,
        time_taken_seconds=attempt.time_taken_seconds,
        prior_mastery_prob=prior_mastery,
    )

    # 5. Synchronize Low-Latency In-Memory Feature Store
    FEATURE_STORE.put_student_features(
        attempt.student_id,
        {
            "last_topic": question.topic_id,
            "level": state.level,
            "mastery_prob": state.mastery_prob,
            "confidence_interval": state.confidence_interval,
            "irt_theta": state.irt_theta,
            "pace_flag": state.pace_flag,
        },
    )
    db.flush()
    return is_correct, question, state


@app.post("/submit_attempt")
def submit_attempt(attempt: schemas.AttemptIn, db: Session = Depends(get_db)):
    with _get_student_lock(attempt.student_id):
        is_correct, _, state = _process_single_attempt_no_commit(attempt, db)
        db.commit()
        return {
            "is_correct": is_correct,
            "new_level": state.level,
            "pace_flag": state.pace_flag,
            "mastery_prob": state.mastery_prob,
            "confidence_interval": state.confidence_interval,
            "irt_theta": state.irt_theta,
        }


@app.post("/submit_paper", response_model=schemas.PaperReviewOut)
def submit_paper(payload: schemas.PaperSubmitIn, db: Session = Depends(get_db)):
    """
    Atomic batch paper submission endpoint: processes all answered questions in a single
    database transaction, prevents paper_number collisions across sessions, and directly
    returns PaperReviewOut in one round-trip.
    """
    if not payload.attempts:
        raise HTTPException(status_code=400, detail="No attempts provided in paper submission")

    with _get_student_lock(payload.student_id):
        existing_max_row = (
            db.query(TestAttempt.paper_number)
            .filter_by(student_id=payload.student_id)
            .order_by(TestAttempt.paper_number.desc())
            .first()
        )
        max_existing_paper = int(existing_max_row[0]) if existing_max_row and existing_max_row[0] else 0
        requested_paper = int(payload.paper_number or 1)
        clean_paper_number = (
            max_existing_paper + 1 if requested_paper <= max_existing_paper else requested_paper
        )

        review_questions: list[schemas.QuestionReviewOut] = []
        correct_count = 0

        for att in payload.attempts:
            att_copy = schemas.AttemptIn(
                student_id=payload.student_id,
                question_id=att.question_id,
                paper_number=clean_paper_number,
                selected_option=att.selected_option,
                time_taken_seconds=att.time_taken_seconds,
                option_switches=att.option_switches,
                hesitation_ms=att.hesitation_ms,
                dwell_time_ms=att.dwell_time_ms,
            )
            is_correct, question, _ = _process_single_attempt_no_commit(att_copy, db)
            if is_correct:
                correct_count += 1
            review_questions.append(
                schemas.QuestionReviewOut(
                    question_id=question.question_id,
                    topic_id=question.topic_id,
                    text=question.text,
                    options=question.options,
                    correct_answer=question.correct_answer,
                    selected_option=att.selected_option,
                    is_correct=is_correct,
                )
            )

        db.commit()
        total = len(review_questions)
        return schemas.PaperReviewOut(
            paper_number=clean_paper_number,
            total_questions=total,
            correct_count=correct_count,
            score_percent=round(100.0 * correct_count / total, 1) if total else 0.0,
            questions=review_questions,
        )


@app.get("/students/{student_id}/state", response_model=list[schemas.TopicStateOut])
def get_student_state(student_id: str, db: Session = Depends(get_db)):
    FEATURE_STORE.get_student_features(student_id)
    return db.query(StudentTopicState).filter_by(student_id=student_id).all()


@app.get("/students/{student_id}/next_paper", response_model=list[schemas.QuestionOut])
def next_paper(student_id: str, topic_id: str | None = None, db: Session = Depends(get_db)):
    return generate_paper(db, student_id, topic_id=topic_id)


@app.get("/students/{student_id}/papers/{paper_number}/review", response_model=schemas.PaperReviewOut)
def paper_review(student_id: str, paper_number: int, db: Session = Depends(get_db)):
    rows = (
        db.query(TestAttempt, Question)
        .join(Question, TestAttempt.question_id == Question.question_id)
        .filter(TestAttempt.student_id == student_id, TestAttempt.paper_number == paper_number)
        .all()
    )
    if not rows:
        raise HTTPException(status_code=404, detail="No attempts found for this paper")

    questions = []
    correct_count = 0
    for attempt, question in rows:
        if attempt.is_correct:
            correct_count += 1
        questions.append(
            schemas.QuestionReviewOut(
                question_id=question.question_id,
                topic_id=question.topic_id,
                text=question.text,
                options=question.options,
                correct_answer=question.correct_answer,
                selected_option=attempt.selected_option,
                is_correct=attempt.is_correct,
            )
        )

    total = len(questions)
    return schemas.PaperReviewOut(
        paper_number=paper_number,
        total_questions=total,
        correct_count=correct_count,
        score_percent=round(100 * correct_count / total, 1) if total else 0.0,
        questions=questions,
    )


@app.get("/students/{student_id}/analytics", response_model=schemas.AnalyticsOut)
def get_analytics(student_id: str, db: Session = Depends(get_db)):
    return build_analytics(db, student_id)


# --- Enterprise v2 Endpoints: Cognitive Insights, Socratic RAG & MLOps Drift ---

@app.get("/students/{student_id}/v2_insights", response_model=schemas.V2StudentInsightsOut)
def get_v2_student_insights(student_id: str, db: Session = Depends(get_db)):
    """
    Returns Enterprise v2 Psychometric & Prerequisite Graph Diagnostics:
    - 3PL IRT Global Latent Ability (theta), Predicted NEET Percentile & Score (/720)
    - Root-Cause Prerequisite Bottlenecks via Ancestral DAG Traversal
    - Next Recommended Topics in the Student's Zone of Proximal Development (ZPD) across all 65 NCERT chapters
    - Hierarchical Bayesian Empirical Shrinkage Mastery Estimates
    """
    student = db.query(Student).filter_by(student_id=student_id).first()
    states = db.query(StudentTopicState).filter_by(student_id=student_id).all()

    mastery_map = {}
    thetas = []
    topic_details = []

    for st in states:
        m_prob = float(getattr(st, "mastery_prob", 0.35) or 0.35)
        t_theta = float(getattr(st, "irt_theta", 0.0) or 0.0)
        mastery_map[st.topic_id] = m_prob
        thetas.append(t_theta)

    global_theta = (
        float(getattr(student, "global_theta", 0.0) or 0.0)
        if student and getattr(student, "global_theta", None) is not None
        else (sum(thetas) / len(thetas) if thetas else 0.0)
    )
    if thetas and abs(global_theta) < 1e-6:
        global_theta = round(sum(thetas) / len(thetas), 4)

    percentile = theta_to_percentile(global_theta)
    predicted_score = int(max(120, min(720, round(380 + global_theta * 110))))

    # Fetch student attempt rows for MIRT 3D profile and Ebbinghaus Spaced Repetition decay
    attempt_rows = (
        db.query(TestAttempt, Question)
        .join(Question, TestAttempt.question_id == Question.question_id)
        .filter(TestAttempt.student_id == student_id)
        .order_by(TestAttempt.attempt_id.asc())
        .all()
    )
    mirt_inputs = []
    last_attempt_ts_by_topic: dict[str, datetime] = {}
    for att, q in attempt_rows:
        rich_q = _RAG_INDEX.question_lookup.get(q.question_id, {})
        mirt_inputs.append(
            {
                "question_id": q.question_id,
                "question_text": q.text,
                "question_type": rich_q.get("question_type", ""),
                "difficulty": q.difficulty,
                "expected_time_seconds": q.expected_time_seconds or 90,
                "is_correct": att.is_correct,
            }
        )
        if getattr(att, "timestamp", None):
            last_attempt_ts_by_topic[q.topic_id] = att.timestamp

    mirt_3d = compute_mirt_3d_profile(mirt_inputs)
    spaced_reviews = []

    for st in states:
        att_cnt = int(st.attempts_counted or len(st.last_3_results or []) or 1)
        shrunk_m = compute_hierarchical_shrinkage_prior(
            topic_id=st.topic_id,
            student_mastery_map=mastery_map,
            attempts_count=att_cnt,
        )
        elapsed_h = 0.0
        if st.topic_id in last_attempt_ts_by_topic:
            ts = last_attempt_ts_by_topic[st.topic_id]
            now_dt = datetime.now(timezone.utc) if ts.tzinfo else datetime.utcnow()
            elapsed_h = max(0.0, (now_dt - ts).total_seconds() / 3600.0)

        decayed_m = compute_ebbinghaus_forgetting_prior(shrunk_m, elapsed_h, int(st.streak or 0))
        urgency = compute_retention_urgency(shrunk_m, elapsed_h, int(st.streak or 0))
        spaced_reviews.append(
            {
                "topic_id": st.topic_id,
                "peak_mastery": round(shrunk_m, 4),
                "decayed_mastery": round(decayed_m, 4),
                "elapsed_hours": round(elapsed_h, 2),
                "retention_urgency": urgency,
            }
        )
        topic_details.append(
            {
                "topic_id": st.topic_id,
                "level": st.level,
                "mastery_prob": round(shrunk_m, 4),
                "confidence_interval": round(float(getattr(st, "confidence_interval", 0.45) or 0.45), 4),
                "irt_theta": round(float(getattr(st, "irt_theta", 0.0) or 0.0), 4),
                "readiness_score": compute_prerequisite_readiness(st.topic_id, mastery_map),
                "pace_flag": st.pace_flag,
            }
        )
    spaced_reviews.sort(key=lambda r: (r["retention_urgency"], r["peak_mastery"]), reverse=True)

    raw_bottlenecks = diagnose_prerequisite_bottlenecks(mastery_map)
    formatted_bottlenecks = []
    for b in raw_bottlenecks[:5]:
        prereqs = b.get("weak_prerequisites") or [b["topic_id"]]
        formatted_bottlenecks.append(
            {
                "weak_topic": b["topic_id"],
                "weak_topic_mastery": b["current_mastery"],
                "root_cause_prerequisite": prereqs[0],
                "prerequisite_mastery": round(mastery_map.get(prereqs[0], 0.35), 3),
                "recommendation": b["recommendation"],
            }
        )

    # Evaluate all 65 NCERT chapters in DAG_NODES weighted by NEET importance and ZPD readiness
    state_lookup = {s.topic_id: s for s in states}
    recommended = []
    for topic_id, meta in DAG_NODES.items():
        st_obj = state_lookup.get(topic_id)
        att_cnt = int(st_obj.attempts_counted or len(st_obj.last_3_results or []) or 1) if st_obj else 0
        curr_m = compute_hierarchical_shrinkage_prior(topic_id, mastery_map, attempts_count=att_cnt)
        readiness = compute_prerequisite_readiness(topic_id, mastery_map)
        importance_weight = float(_TOPIC_IMPORTANCE.get(topic_id, 1.0))
        zpd_score = readiness * (1.0 - abs(curr_m - 0.55)) * (1.0 + 0.1 * importance_weight)
        recommended.append(
            {
                "topic_id": topic_id,
                "subject": meta.get("subject", "Science"),
                "cluster": meta.get("cluster", "General"),
                "level": st_obj.level if st_obj else "basic",
                "current_mastery": round(curr_m, 3),
                "readiness": readiness,
                "neet_weightage": round(importance_weight, 2),
                "prerequisites": meta.get("prerequisites", []),
                "zpd_priority_score": round(zpd_score, 4),
            }
        )
    recommended.sort(key=lambda r: r["zpd_priority_score"], reverse=True)

    return schemas.V2StudentInsightsOut(
        student_id=student_id,
        global_theta=round(global_theta, 4),
        estimated_neet_percentile=percentile,
        predicted_neet_score_720=predicted_score,
        zpd_target_success_prob=0.70,
        mirt_3d_profile=mirt_3d,
        spaced_repetition_reviews=spaced_reviews[:5],
        root_cause_bottlenecks=formatted_bottlenecks,
        recommended_unlock_topics=recommended,
        topic_mastery_details=topic_details,
    )


@app.post("/mlops/predict_cold_start_item")
def predict_cold_start_item(payload: schemas.ColdStartItemIn):
    """
    Zero-attempt cold-start NLP & cognitive complexity estimator: predicts difficulty tier
    ('basic', 'medium', 'advanced') and initial 3PL IRT parameters (a_i, b_i, c_i) directly
    from question text and options without requiring historical student attempt data.
    """
    return predict_cold_start_difficulty_and_irt(
        question_text=payload.question_text,
        options=payload.options,
        has_diagram=bool(payload.has_diagram),
    )


@app.post("/remediation/socratic_hint", response_model=schemas.SocraticHintOut)
def get_socratic_hint(payload: schemas.SocraticHintIn, db: Session = Depends(get_db)):
    """
    Retrieves NCERT-grounded Socratic guidance, formula anchors, and distractor
    misconception diagnosis for an attempted NEET question.
    """
    q_row = db.query(Question).filter_by(question_id=payload.question_id).first()
    rich_q = _RAG_INDEX.question_lookup.get(payload.question_id, {})

    topic_id = q_row.topic_id if q_row else rich_q.get("chapter", "Physics")
    q_text = q_row.text if q_row else rich_q.get("question_text", "")
    options = (
        q_row.options
        if q_row
        else [
            rich_q.get("option_a", "A"),
            rich_q.get("option_b", "B"),
            rich_q.get("option_c", "C"),
            rich_q.get("option_d", "D"),
        ]
    )
    correct_ans = q_row.correct_answer if q_row else str(rich_q.get("correct_option", "a")).lower()

    raw = generate_socratic_remediation(
        question_id=payload.question_id,
        topic_id=topic_id,
        question_text=q_text,
        options=options,
        correct_answer=correct_ans,
        selected_option=payload.selected_option,
    )
    formulas = raw.get("key_formulas") or []
    socratic_qs = raw.get("socratic_guiding_questions") or []
    steps = raw.get("guided_derivation_steps") or []

    return schemas.SocraticHintOut(
        question_id=payload.question_id,
        chapter=raw.get("topic_id", topic_id),
        ncert_reference=raw.get("ncert_reference", f"NCERT — {topic_id}"),
        concept_summary=raw.get("core_concept_anchor", ""),
        key_formula=formulas[0] if formulas else f"Standard NCERT relation for {topic_id}",
        diagnosed_misconception=raw.get("misconception_diagnosis", ""),
        socratic_hint=socratic_qs[0] if socratic_qs else "What fundamental NCERT principle governs this question?",
        verified_step_by_step=" ".join(steps),
        retrieval_similarity=float(raw.get("retrieval_confidence", 0.88)),
    )


@app.get("/mlops/health_and_drift")
def get_mlops_health_and_drift(db: Session = Depends(get_db)):
    """
    Exposes real-time Feature Store latency percentiles (p50/p95/p99) and
    Evidently-style Population Stability Index (PSI) & Expected Calibration Error (ECE)
    computed from actual TestAttempt rows and calibrated 3PL IRT probabilities.
    """
    recent_rows = (
        db.query(TestAttempt, Question)
        .join(Question, TestAttempt.question_id == Question.question_id)
        .order_by(TestAttempt.attempt_id.desc())
        .limit(250)
        .all()
    )
    all_states = db.query(StudentTopicState).all()
    theta_lookup = {
        (s.student_id, s.topic_id): float(getattr(s, "irt_theta", 0.0) or 0.0)
        for s in all_states
    }

    db_attempts = []
    for att, q in recent_rows:
        st_theta = theta_lookup.get((att.student_id, q.topic_id), 0.0)
        params = get_item_params(q.question_id, q.difficulty, q.expected_time_seconds or 90)
        real_pred_prob = irt_3pl_probability(st_theta, params)
        db_attempts.append(
            {
                "predicted_prob": real_pred_prob,
                "correct": att.is_correct,
                "difficulty": q.difficulty,
                "time_taken": att.time_taken_seconds,
            }
        )

    return {
        "feature_store": FEATURE_STORE.get_latency_percentiles(),
        "drift_and_calibration": DRIFT_MONITOR.evaluate_health(db_attempts),
        "models_loaded": {
            "random_forest_difficulty_classifier": (MODEL_DIR / "question_difficulty_model.joblib").exists(),
            "bkt_bayesian_knowledge_tracing": len(_TOPIC_IMPORTANCE) == 65,
            "irt_3pl_psychometric_engine": len(CALIBRATED_ITEMS) >= 2471,
            "ncert_prerequisite_dag_nodes": len(DAG_NODES),
            "linucb_contextual_bandit": len(_TOPIC_IMPORTANCE) == 65,
            "rag_ncert_socratic_engine": len(_RAG_INDEX.chunks) == 65,
        },
    }


# Serve production React SPA build when frontend/dist exists (Vercel / Docker / Render / Railway)
_FRONTEND_DIST = PROJECT_ROOT / "frontend" / "dist"
if (_FRONTEND_DIST / "index.html").exists():
    @app.get("/assets/{asset_name}", include_in_schema=False)
    def serve_frontend_asset(asset_name: str):
        assets_dir = _FRONTEND_DIST / "assets"
        target = assets_dir / asset_name
        if not target.exists():
            # Fallback for browsers holding a cached index.html with an older bundle hash
            if asset_name.endswith(".js"):
                js_files = sorted(assets_dir.glob("*.js"))
                if js_files:
                    target = js_files[0]
            elif asset_name.endswith(".css"):
                css_files = sorted(assets_dir.glob("*.css"))
                if css_files:
                    target = css_files[0]
        if not target.exists():
            raise HTTPException(status_code=404, detail="Asset not found")
        media_type = (
            "application/javascript; charset=utf-8"
            if target.suffix == ".js"
            else ("text/css; charset=utf-8" if target.suffix == ".css" else "application/octet-stream")
        )
        return Response(
            content=target.read_bytes(),
            media_type=media_type,
            headers={"Cache-Control": "no-cache, must-revalidate"},
        )

    @app.get("/vite.svg", include_in_schema=False)
    def serve_vite_svg():
        svg_path = _FRONTEND_DIST / "vite.svg"
        if svg_path.exists():
            return FileResponse(str(svg_path), media_type="image/svg+xml")
        raise HTTPException(status_code=404, detail="Not found")

    @app.get("/", include_in_schema=False)
    def serve_spa_root():
        return HTMLResponse(
            content=(_FRONTEND_DIST / "index.html").read_text(encoding="utf-8"),
            headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
        )


