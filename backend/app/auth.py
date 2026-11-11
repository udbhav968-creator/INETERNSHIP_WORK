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
from cryptography.fernet import Fernet
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session
from .db_models import Student

MIN_PASSWORD_LENGTH = 8
VALID_CLASS_LEVELS = ("11", "12", "dropper")
DEFAULT_JWT_SECRET = "klassed-neet-v2-enterprise-hmac-secret-2026"
_RAW_SECRET = os.environ.get("KLASSED_JWT_SECRET")
_IS_PROD = (
    os.environ.get("ENVIRONMENT", "").lower() in ("production", "prod")
    or os.environ.get("KLASSED_ENV", "").lower() in ("production", "prod")
    or os.environ.get("VERCEL_ENV", "").lower() == "production"
    or os.environ.get("NODE_ENV", "").lower() == "production"
    or bool(os.environ.get("RENDER"))
)

JWT_SECRET_KEY = _RAW_SECRET if _RAW_SECRET else DEFAULT_JWT_SECRET

JWT_EXPIRY_SECONDS = 86400 * 7
VAULT_EXPIRY_SECONDS = 86400 * 30

_VAULT_KEY = base64.urlsafe_b64encode(hashlib.sha256(JWT_SECRET_KEY.encode("utf-8")).digest())
_VAULT_CIPHER = Fernet(_VAULT_KEY)


def _sign_hmac_payload(header: dict, payload: dict) -> str:
    b64_header = base64.urlsafe_b64encode(json.dumps(header, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
    b64_payload = base64.urlsafe_b64encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")).rstrip(b"=")
    signing_input = b64_header + b"." + b64_payload
    signature = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
    b64_sig = base64.urlsafe_b64encode(signature).rstrip(b"=")
    return (signing_input + b"." + b64_sig).decode("utf-8")


def _verify_hmac_payload(token: str, expected_typ: str) -> Optional[dict]:
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = f"{parts[0]}.{parts[1]}".encode("utf-8")
        expected_sig = hmac.new(JWT_SECRET_KEY.encode("utf-8"), signing_input, hashlib.sha256).digest()
        b64_expected = base64.urlsafe_b64encode(expected_sig).rstrip(b"=").decode("utf-8")
        if not hmac.compare_digest(parts[2], b64_expected):
            return None
        padded_hdr = parts[0] + "=" * (-len(parts[0]) % 4)
        hdr = json.loads(base64.urlsafe_b64decode(padded_hdr.encode("utf-8")).decode("utf-8"))
        if hdr.get("typ") != expected_typ:
            return None
        padded = parts[1] + "=" * (-len(parts[1]) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded.encode("utf-8")).decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def create_access_token(
    student_id: str,
    email: str,
    name: Optional[str] = None,
    class_level: Optional[str] = None,
    target_neet_year: Optional[int] = None,
    onboarded: bool = False,
) -> str:
    now = int(time.time())
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {
        "sub": student_id,
        "email": email,
        "name": name or student_id,
        "class_level": class_level,
        "target_neet_year": target_neet_year,
        "onboarded": bool(onboarded),
        "iat": now,
        "exp": now + JWT_EXPIRY_SECONDS,
    }
    return _sign_hmac_payload(header, payload)


def verify_access_token(token: str) -> Optional[dict]:
    return _verify_hmac_payload(token, "JWT")


def create_vault_token(student: Student) -> str:
    now = int(time.time())
    payload = {
        "sub": student.student_id,
        "name": student.name,
        "email": student.email,
        "password_hash": student.password_hash,
        "class_level": student.class_level,
        "target_neet_year": student.target_neet_year,
        "onboarded": bool(student.onboarded),
        "iat": now,
        "exp": now + VAULT_EXPIRY_SECONDS,
    }
    raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    return _VAULT_CIPHER.encrypt(raw).decode("utf-8")


def verify_vault_token(token: str) -> Optional[dict]:
    try:
        raw = _VAULT_CIPHER.decrypt(str(token).strip().encode("utf-8"))
        payload = json.loads(raw.decode("utf-8"))
        if payload.get("exp", 0) < int(time.time()):
            return None
        return payload
    except Exception:
        return None


def _attach_tokens(student: Student) -> Student:
    student.access_token = create_access_token(
        student.student_id,
        student.email,
        name=student.name,
        class_level=student.class_level,
        target_neet_year=student.target_neet_year,
        onboarded=student.onboarded,
    )
    if str(student.password_hash or "").startswith("$2"):
        student.vault_token = create_vault_token(student)
    else:
        student.vault_token = None
    return student


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
        if not password or not password_hash:
            return False
        return bcrypt.checkpw(password.encode("utf-8"), str(password_hash).encode("utf-8"))
    except Exception:
        return False


def hydrate_student_from_vault(db: Session, vault_token: Optional[str]) -> Optional[Student]:
    if not vault_token:
        return None
    vault = verify_vault_token(vault_token.strip())
    if not vault:
        return None
    sid = str(vault.get("sub") or "").strip()
    email = str(vault.get("email") or "").strip().lower()
    pw_hash = str(vault.get("password_hash") or "")
    if not sid or not email or not pw_hash.startswith("$2"):
        return None

    student_by_email = db.query(Student).filter_by(email=email).first()
    student_by_id = db.query(Student).filter_by(student_id=sid).first()
    if student_by_email and student_by_id and student_by_email.student_id != student_by_id.student_id:
        return None
    student = student_by_email or student_by_id
    if student and (student.student_id != sid or student.email != email):
        return None
    if not student:
        student = Student(
            student_id=sid,
            name=str(vault.get("name") or sid),
            email=email,
            password_hash=pw_hash,
            class_level=vault.get("class_level"),
            target_neet_year=vault.get("target_neet_year"),
            onboarded=bool(vault.get("onboarded", False)),
        )
        db.add(student)
        db.commit()
        db.refresh(student)
    else:
        changed = False
        if not str(student.password_hash or "").startswith("$2"):
            student.password_hash = pw_hash
            changed = True
        if vault.get("onboarded") and not student.onboarded:
            student.class_level = vault.get("class_level")
            student.target_neet_year = vault.get("target_neet_year")
            student.onboarded = True
            changed = True
        if changed:
            db.commit()
            db.refresh(student)
    return student


def get_current_student_id(request: Request) -> str:
    auth_header = request.headers.get("authorization") or ""
    if not auth_header.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Authentication credentials were not provided")
    raw_token = auth_header[7:].strip()
    claims = verify_access_token(raw_token)
    if not claims or not claims.get("sub"):
        raise HTTPException(status_code=401, detail="Invalid or expired authentication credentials")
    return str(claims["sub"])


def verify_student_access(student_id: str, caller_id: str = Depends(get_current_student_id)) -> str:
    if str(caller_id) != str(student_id):
        raise HTTPException(status_code=403, detail="Forbidden: You cannot access another student's data")
    return str(caller_id)


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

    for _ in range(5):
        cand_id = _make_student_id(name, db)
        if db.query(Student).filter_by(student_id=cand_id).first():
            continue
        try:
            student = Student(
                student_id=cand_id,
                name=name,
                email=email,
                password_hash=hash_password(password),
                onboarded=False,
            )
            db.add(student)
            db.commit()
            db.refresh(student)
            return _attach_tokens(student)
        except Exception:
            db.rollback()
            if db.query(Student).filter_by(email=email).first():
                raise HTTPException(status_code=409, detail="An account with this email already exists")
            continue
    raise HTTPException(status_code=500, detail="Failed to allocate unique student identifier")


def login(db: Session, email: str, password: str, vault_token: Optional[str] = None) -> Student:
    email = (email or "").strip().lower()
    if not email or not password:
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    if vault_token:
        hydrated = hydrate_student_from_vault(db, vault_token)
        if hydrated and hydrated.email != email:
            hydrated = None
    student = db.query(Student).filter_by(email=email).first()
    if not student or not verify_password(password, student.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect email or password")
    return _attach_tokens(student)


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
    return _attach_tokens(student)
