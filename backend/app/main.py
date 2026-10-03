"""
Klassed NEET Adaptive Engine — backend API.

Run locally with:
    uvicorn app.main:app --reload

Endpoints:
    POST /submit_attempt          -> score one answer, update state + pace
    GET  /students/{id}/state     -> current per-topic mastery + pace
    GET  /students/{id}/next_paper -> generate the next adaptive paper
"""
from fastapi import FastAPI, Depends, HTTPException
from sqlalchemy.orm import Session

from . import schemas
from .database import get_db, init_db
from .db_models import Question, StudentTopicState, TestAttempt
from .state_machine import process_answer
from .time_tracking import update_pace
from .paper_generator import generate_paper

app = FastAPI(title="Klassed NEET Adaptive Engine")


@app.on_event("startup")
def on_startup():
    init_db()


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
