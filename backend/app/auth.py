"""
Signup / login for real student accounts + Enterprise v2 HMAC-SHA256 JWT Bearer Token Verification.

Password hashing uses bcrypt directly (no passlib — avoids the
passlib/bcrypt>=4.1 compatibility breakage). Student IDs are derived
from the student's name (e.g. "Parv Wadhwa" -> "PARVWADHWA"), with a
numeric suffix if that ID is already taken, so existing code that
keys everything off `student_id` (state machine, pace tracking,
paper generator) needs no changes.
"""
import base64
import datetime
import hashlib
import hmac
import json
import os
import re
import time
from typing import Optional

import bcrypt
from fastapi import HTTPException
from sqlalchemy.orm import Session

from .db_models import Student

MIN_PASSWORD_LENGTH = 8
VALID_CLASS_LEVELS = ("11", "12", "dropper")
JWT_SECRET_KEY = os.environ.get("KLASSED_JWT_SECRET", "klassed-neet-v2-enterprise-hmac-secret-2026")
JWT_EXPIRY_SECONDS = 86400 * 7  # 7 days


def create_access_token(student_id: str, email: str) -> str:
    """Generates an RFC 7519-compliant HMAC-SHA256 signed JWT token using standard library crypto."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": student_id,
        "email": email,
        "iat": int(time.time()),
        "exp": int(time.time()) + JWT_EXPIRY_SECONDS,
    }
    b64_header = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
    b64_payload = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
    signing_input = b64_header + b"." + b64_payload
    signature = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    b64_sig = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return (signing_input + b"." + b64_sig).decode("utf-8")


def verify_access_token(token: str) -> Optional[dict]:
    """Verifies an HMAC-SHA256 JWT token and returns the decoded claims if valid."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = f"{parts[0]}.{parts[1]}".encode("utf-8")
        expected_sig = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        b64_expected = base64.urlsafe_b64encode(expected_sig).rstrip(b"=").decode("utf-8")
        if not hmac.compare_digest(parts[2], b64_expected):
            return None
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


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
    student.access_token = create_access_token(student.student_id, student.email)
    return student


def login(db: Session, email: str, password: str) -> Student:
    email = email.strip().lower()
    student = db.query(Student).filter_by(email=email).first()
    if not student or not verify_password(password, student.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    student.access_token = create_access_token(student.student_id, student.email)
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
    student.access_token = create_access_token(student.student_id, student.email)
    return student
