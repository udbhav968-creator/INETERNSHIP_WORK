from pydantic import BaseModel, EmailStr
from typing import Optional


class AttemptIn(BaseModel):
    student_id: str
    question_id: str
    paper_number: int
    selected_option: str
    time_taken_seconds: int


class TopicStateOut(BaseModel):
    topic_id: str
    level: str
    pace_flag: Optional[str] = None
    avg_time_ratio: Optional[float] = None

    class Config:
        from_attributes = True


class QuestionOut(BaseModel):
    question_id: str
    topic_id: str
    difficulty: str
    text: str
    options: list[str]

    class Config:
        from_attributes = True


# --- auth / onboarding ---

class SignupIn(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class OnboardingIn(BaseModel):
    class_level: str   # "11" | "12" | "dropper"
    target_neet_year: int


class StudentOut(BaseModel):
    student_id: str
    name: str
    email: str
    class_level: Optional[str] = None
    target_neet_year: Optional[int] = None
    onboarded: bool

    class Config:
        from_attributes = True


# --- paper review (post-submission only — never exposes answers mid-exam) ---

class QuestionReviewOut(BaseModel):
    question_id: str
    topic_id: str
    text: str
    options: list[str]
    correct_answer: str
    selected_option: str
    is_correct: bool


class PaperReviewOut(BaseModel):
    paper_number: int
    total_questions: int
    correct_count: int
    score_percent: float
    questions: list[QuestionReviewOut]


# --- analytics (read-only, derived entirely from real test_attempts rows) ---

class SubjectBreakdownOut(BaseModel):
    subject: str
    attempts: int
    correct: int
    accuracy: float


class RecentAttemptOut(BaseModel):
    question_id: str
    topic_id: str
    is_correct: bool
    time_taken_seconds: int
    timestamp: str


class AnalyticsOut(BaseModel):
    total_attempts: int
    correct_attempts: int
    accuracy: float
    topics_practiced: int
    subject_breakdown: list[SubjectBreakdownOut]
    recent_attempts: list[RecentAttemptOut]
