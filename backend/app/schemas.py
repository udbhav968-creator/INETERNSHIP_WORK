from pydantic import BaseModel, field_validator
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
    time_taken_seconds: Optional[float] = 60.0
    option_switches: Optional[int] = 0
    hesitation_ms: Optional[int] = 0
    dwell_time_ms: Optional[int] = 0

    @field_validator("selected_option")
    @classmethod
    def validate_selected_option(cls, v: str) -> str:
        clean = (v or "").strip().lower()
        if clean in ("0", "1", "2", "3"):
            clean = ("a", "b", "c", "d")[int(clean)]
        if clean not in ("a", "b", "c", "d"):
            raise ValueError("selected_option must be one of: a, b, c, d")
        return clean

    @field_validator("time_taken_seconds")
    @classmethod
    def validate_time_taken_seconds(cls, v: Optional[float]) -> float:
        if v is None:
            return 60.0
        val = float(v)
        if val < 0.5 or val > 3600.0:
            raise ValueError("time_taken_seconds must be between 0.5 and 3600.0")
        return round(val, 2)


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
    time_taken_seconds: Optional[float] = 60.0
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
    distractor_trap_taxonomy: Optional[str] = "CONCEPTUAL_SUBSTITUTION"
    audio_socratic_script: Optional[dict[str, Any]] = None
    hindi_glossary: Optional[dict[str, Any]] = None


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
    global_theta: Optional[float] = 0.0
    estimated_neet_percentile: Optional[float] = None
    predicted_neet_score_720: Optional[int] = None
    zpd_target_success_prob: float
    mirt_3d_profile: Optional[dict[str, Any]] = None
    dina_cognitive_profile: Optional[dict[str, Any]] = None
    monte_carlo_conformal_simulator: Optional[dict[str, Any]] = None
    spaced_repetition_reviews: Optional[list[dict[str, Any]]] = None
    root_cause_bottlenecks: list[dict[str, Any]]
    recommended_unlock_topics: list[dict[str, Any]]
    topic_mastery_details: list[dict[str, Any]]


class CATAssessmentOut(BaseModel):
    student_id: str
    items_administered: int
    latent_theta: float
    sem_theta: float
    confidence_interval_95: list[float]
    cumulative_fisher_information: float
    target_sem_threshold: float
    stopping_rule_met: bool
    predicted_neet_score: Optional[int] = None
    assessment_status: str


class RevisionDayPlanOut(BaseModel):
    day: int
    date_label: str
    chapter: str
    subject: str
    action_type: str
    target_questions: int
    urgency_score: float
    potential_marks_gain: float
    reason: str


class RevisionCalendarOut(BaseModel):
    student_id: str
    generated_at: str
    total_days: int
    daily_schedule: list[RevisionDayPlanOut]


class FatigueAnalysisOut(BaseModel):
    student_id: str
    total_attempts_analyzed: int
    fatigue_index: float
    latency_trend: str
    mean_early_latency_sec: float
    mean_late_latency_sec: float
    early_accuracy_pct: float
    late_accuracy_pct: float
    accuracy_drop_pct: float
    rapid_guess_count: int
    pacing_coaching_advice: str


class DashboardSnapshotOut(BaseModel):
    topics: list[TopicStateOut]
    analytics: AnalyticsOut
    v2_insights: V2StudentInsightsOut
    paper_history: list[PaperHistorySummaryOut]
    study_plan: Optional[dict[str, Any]] = None
    cat_assessment: Optional[CATAssessmentOut] = None
    fatigue_analysis: Optional[FatigueAnalysisOut] = None
    revision_calendar: Optional[RevisionCalendarOut] = None
