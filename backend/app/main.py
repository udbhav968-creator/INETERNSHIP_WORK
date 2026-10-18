"""
Klassed NEET Adaptive Engine — backend API.

Run locally with:
    uvicorn app.main:app --reload

Endpoints:
    POST /auth/signup             -> create a real student account
    POST /auth/login              -> authenticate an existing account
    POST /students/{id}/onboarding -> record class + target NEET year
    POST /submit_attempt          -> score one answer, update state + pace
    GET  /students/{id}/state     -> current per-topic mastery + pace
    GET  /students/{id}/next_paper -> generate the next adaptive paper
    GET  /students/{id}/papers/{paper_number}/review
                                   -> score + correct answers for a
                                      paper the student has already
                                      fully attempted (post-exam only)

NOTE: this file assumes you may already have CORS / other routers wired
into your local main.py. Merge new sections in rather than overwriting
wholesale if it has diverged — see the handover notes.
"""
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session

from . import schemas, auth
from .analytics import build_analytics
from .database import get_db, init_db
from .db_models import Question, Student, StudentTopicState, TestAttempt
from .state_machine import process_answer
from .time_tracking import update_pace
from .paper_generator import generate_paper

app = FastAPI(title="Klassed NEET Adaptive Engine")

# Allow the Vite dev server (and any other local frontend origin) to call
# this API from the browser. Tighten allow_origins before shipping to prod.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup():
    init_db()


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
        raise HTTPException(status_code=404, detail="Student not found")
    return auth.complete_onboarding(db, student, payload.class_level, payload.target_neet_year)


@app.get("/students/{student_id}", response_model=schemas.StudentOut)
def get_student(student_id: str, db: Session = Depends(get_db)):
    student = db.query(Student).filter_by(student_id=student_id).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    return student


# --- adaptive engine (unchanged) ---

@app.post("/submit_attempt")
def submit_attempt(attempt: schemas.AttemptIn, db: Session = Depends(get_db)):
    question = db.query(Question).filter_by(question_id=attempt.question_id).first()
    if not question:
        raise HTTPException(status_code=404, detail="Question not found")

    is_correct = attempt.selected_option == question.correct_answer

    # log the raw attempt
    db.add(TestAttempt(
        student_id=attempt.student_id,
        question_id=attempt.question_id,
        paper_number=attempt.paper_number,
        is_correct=is_correct,
        time_taken_seconds=attempt.time_taken_seconds,
        selected_option=attempt.selected_option,
    ))

    # fetch or create this student's state for this topic
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
        )
        db.add(state)

    # two independent updates — accuracy drives level, time drives pace
    process_answer(state, is_correct)
    update_pace(state, is_correct, attempt.time_taken_seconds, question.difficulty)

    db.commit()
    return {"is_correct": is_correct, "new_level": state.level, "pace_flag": state.pace_flag}


@app.get("/students/{student_id}/state", response_model=list[schemas.TopicStateOut])
def get_student_state(student_id: str, db: Session = Depends(get_db)):
    return db.query(StudentTopicState).filter_by(student_id=student_id).all()


@app.get("/students/{student_id}/next_paper", response_model=list[schemas.QuestionOut])
def next_paper(student_id: str, db: Session = Depends(get_db)):
    return generate_paper(db, student_id)


@app.get("/students/{student_id}/papers/{paper_number}/review", response_model=schemas.PaperReviewOut)
def paper_review(student_id: str, paper_number: int, db: Session = Depends(get_db)):
    """
    Post-exam review only. This deliberately does NOT exist as a way to
    peek at answers mid-paper — it joins TestAttempt (what the student
    already submitted) with Question (to pull the correct_answer), so it
    only ever reveals the key for questions the student has already
    answered in that paper.
    """
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
        questions.append(schemas.QuestionReviewOut(
            question_id=question.question_id,
            topic_id=question.topic_id,
            text=question.text,
            options=question.options,
            correct_answer=question.correct_answer,
            selected_option=attempt.selected_option,
            is_correct=attempt.is_correct,
        ))

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
    """
    Thin wrapper around analytics.build_analytics — that module already
    computes real, database-derived stats from test_attempts; this just
    exposes it over HTTP. No new computation lives here.
    """
    return build_analytics(db, student_id)
