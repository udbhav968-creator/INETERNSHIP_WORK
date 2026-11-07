from pydantic import BaseModel
from typing import Optional, Any

try:
    __import__("email_validator")
    from pydantic import EmailStr
except ImportError:
    EmailStr = str


class AttemptIn(BaseModel):
    student_id: str
    question_id: str
    paper_number: int
    selected_option: str
    time_taken_seconds: int
    option_switches: Optional[int] = 0
    hesitation_ms: Optional[int] = 0
    dwell_time_ms: Optional[int] = 0


class PaperSubmitIn(BaseModel):
    student_id: str
    paper_number: Optional[int] = None
    attempts: list[AttemptIn]


class TopicStateOut(BaseModel):
    topic_id: str
    level: str
    pace_flag: Optional[str] = None
    avg_time_ratio: Optional[float] = None
    mastery_prob: Optional[float] = 0.35
    confidence_interval: Optional[float] = 0.45
    irt_theta: Optional[float] = 0.0

    class Config:
        from_attributes = True


class QuestionOut(BaseModel):
    question_id: str
    topic_id: str
    difficulty: str
    text: str
    options: list[str]
    expected_time_seconds: Optional[int] = 90

    class Config:
        from_attributes = True


class SignupIn(BaseModel):
    name: str
    email: EmailStr
    password: str


class LoginIn(BaseModel):
    email: EmailStr
    password: str


class OnboardingIn(BaseModel):
    class_level: str
    target_neet_year: int


class StudentOut(BaseModel):
    student_id: str
    name: str
    email: str
    class_level: Optional[str] = None
    target_neet_year: Optional[int] = None
    onboarded: bool
    global_theta: Optional[float] = 0.0
    access_token: Optional[str] = None
    vault_token: Optional[str] = None

    class Config:
        from_attributes = True


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


class PaperHistorySummaryOut(BaseModel):
    paper_number: int
    total_questions: int
    correct_count: int
    score_percent: float
    total_time_seconds: int
    timestamp: str
    topics: list[str]


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


class SocraticHintIn(BaseModel):
    question_id: str
    selected_option: str
    time_taken_seconds: Optional[int] = 60
    option_switches: Optional[int] = 0


class SocraticHintOut(BaseModel):
    question_id: str
    chapter: str
    ncert_reference: str
    concept_summary: str
    key_formula: str
    diagnosed_misconception: str
    socratic_hint: str
    verified_step_by_step: str
    retrieval_similarity: float


class ColdStartItemIn(BaseModel):
    question_text: str
    options: Optional[list[str]] = None
    has_diagram: Optional[bool] = False
    subject: Optional[str] = "Physics"
    chapter: Optional[str] = "Laws of Motion"
    topic_tag: Optional[str] = "Newton's Laws"
    question_type: Optional[str] = None
    ideal_time_sec: Optional[int] = None


class V2StudentInsightsOut(BaseModel):
    student_id: str
    global_theta: float
    estimated_neet_percentile: float
    predicted_neet_score_720: int
    zpd_target_success_prob: float
    mirt_3d_profile: Optional[dict[str, Any]] = None
    dina_cognitive_profile: Optional[dict[str, Any]] = None
    monte_carlo_conformal_simulator: Optional[dict[str, Any]] = None
    spaced_repetition_reviews: Optional[list[dict[str, Any]]] = None
    root_cause_bottlenecks: list[dict[str, Any]]
    recommended_unlock_topics: list[dict[str, Any]]
    topic_mastery_details: list[dict[str, Any]]


class DashboardSnapshotOut(BaseModel):
    topics: list[TopicStateOut]
    analytics: AnalyticsOut
    v2_insights: V2StudentInsightsOut
    paper_history: list[PaperHistorySummaryOut]
    study_plan: Optional[dict[str, Any]] = None
