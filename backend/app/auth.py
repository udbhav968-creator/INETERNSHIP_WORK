"""
Signup / login for real student accounts.

Password hashing uses bcrypt directly (no passlib — avoids the
passlib/bcrypt>=4.1 compatibility breakage). Student IDs are derived
from the student's name (e.g. "Parv Wadhwa" -> "PARVWADHWA"), with a
numeric suffix if that ID is already taken, so existing code that
keys everything off `student_id` (state machine, pace tracking,
paper generator) needs no changes.

This module owns only identity. Mastery state, pace tracking and
paper generation are untouched.
"""
import datetime
import re

import bcrypt
from fastapi import HTTPException
from sqlalchemy.orm import Session

from .db_models import Student

MIN_PASSWORD_LENGTH = 8
VALID_CLASS_LEVELS = ("11", "12", "dropper")


def _make_student_id(name: str, db: Session) -> str:
    base = re.sub(r"[^A-Za-z0-9]", "", name).upper()[:12] or "STUDENT"
    candidate = base
    suffix = 1
    while db.query(Student).filter_by(student_id=candidate).first():
        suffix += 1
        candidate = f"{base}{suffix}"
    return candidate


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8"), password_hash.encode("utf-8"))
    except ValueError:
        # malformed/legacy hash in the db — treat as "doesn't match" rather than 500ing
        return False


def signup(db: Session, name: str, email: str, password: str) -> Student:
    name = name.strip()
    email = email.strip().lower()

    if not name:
        raise HTTPException(status_code=422, detail="Name is required")
    if len(password) < MIN_PASSWORD_LENGTH:
        raise HTTPException(
            status_code=422,
            detail=f"Password must be at least {MIN_PASSWORD_LENGTH} characters",
        )
    if db.query(Student).filter_by(email=email).first():
        raise HTTPException(status_code=409, detail="An account with this email already exists")

    student = Student(
        student_id=_make_student_id(name, db),
        name=name,
        email=email,
        password_hash=hash_password(password),
        onboarded=False,
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return student


def login(db: Session, email: str, password: str) -> Student:
    email = email.strip().lower()
    student = db.query(Student).filter_by(email=email).first()
    if not student or not verify_password(password, student.password_hash):
        # deliberately identical message for "no such email" and "wrong password"
        # so the endpoint doesn't leak which emails are registered
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return student


def complete_onboarding(db: Session, student: Student, class_level: str, target_neet_year: int) -> Student:
    if class_level not in VALID_CLASS_LEVELS:
        raise HTTPException(
            status_code=422,
            detail=f"class_level must be one of {VALID_CLASS_LEVELS}",
        )
    current_year = datetime.date.today().year
    if target_neet_year < current_year or target_neet_year > current_year + 6:
        raise HTTPException(status_code=422, detail="target_neet_year looks invalid")

    student.class_level = class_level
    student.target_neet_year = target_neet_year
    student.onboarded = True
    db.commit()
    db.refresh(student)
    return student
