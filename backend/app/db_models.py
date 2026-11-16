import datetime
from sqlalchemy import Column, String, Integer, Boolean, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"
    student_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    class_level = Column(String, nullable=True)
    target_neet_year = Column(Integer, nullable=True)
    onboarded = Column(Boolean, default=False, nullable=False)
    global_theta = Column(Float, default=0.0, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


class Topic(Base):
    __tablename__ = "topics"
    topic_id = Column(String, primary_key=True)
    subject = Column(String, nullable=False)
    name = Column(String, nullable=False)


class Question(Base):
    __tablename__ = "questions"
    question_id = Column(String, primary_key=True)
    topic_id = Column(String, ForeignKey("topics.topic_id"), nullable=False)
    difficulty = Column(String, nullable=False)
    expected_time_seconds = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    options = Column(JSON, nullable=False)
    correct_answer = Column(String, nullable=False)
    wrong_option_tags = Column(JSON, nullable=True)
    years_asked = Column(JSON, nullable=True)

    @property
    def paper_source(self) -> str:
        qid = self.question_id or ""
        if "2026" in qid:
            return "NEET 2026 Re-NEET"
        for y in range(2025, 2012, -1):
            if str(y) in qid:
                return f"NEET {y}"
        if self.years_asked and isinstance(self.years_asked, list) and len(self.years_asked) > 0:
            return f"NEET {self.years_asked[0]}"
        return "Official NEET PYQ"


class StudentTopicState(Base):
    __tablename__ = "student_topic_state"
    student_id = Column(String, ForeignKey("students.student_id"), primary_key=True)
    topic_id = Column(String, ForeignKey("topics.topic_id"), primary_key=True)
    level = Column(String, default="basic")
    streak = Column(Integer, default=0)
    last_3_results = Column(JSON, default=list)
    avg_time_ratio = Column(Float, nullable=True)
    attempts_counted = Column(Integer, default=0)
    pace_flag = Column(String, nullable=True)
    mastery_prob = Column(Float, default=0.35, nullable=False)
    confidence_interval = Column(Float, default=0.45, nullable=False)
    irt_theta = Column(Float, default=0.0, nullable=False)


class TestAttempt(Base):
    __tablename__ = "test_attempts"
    attempt_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.student_id"), nullable=False)
    question_id = Column(String, ForeignKey("questions.question_id"), nullable=False)
    paper_number = Column(Integer, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    time_taken_seconds = Column(Float, nullable=False)
    selected_option = Column(String, nullable=False)
    option_switches = Column(Integer, default=0, nullable=False)
    hesitation_ms = Column(Integer, default=0, nullable=False)
    dwell_time_ms = Column(Integer, default=0, nullable=False)
    timestamp = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))


class SelectionDecision(Base):
    __tablename__ = "selection_decisions"
    decision_id = Column(Integer, primary_key=True, autoincrement=True)
    request_id = Column(String, nullable=False, index=True)
    student_id = Column(String, nullable=False, index=True)
    topic_id = Column(String, nullable=True)
    question_id = Column(String, nullable=False, index=True)
    slot = Column(Integer, nullable=False)
    policy = Column(String, nullable=False)
    arm = Column(String, nullable=False)
    epsilon = Column(Float, default=0.0, nullable=False)
    propensity = Column(Float, nullable=False)
    chosen_score = Column(Float, nullable=True)
    candidate_ids = Column(JSON, nullable=True)
    candidate_scores = Column(JSON, nullable=True)
    candidate_count = Column(Integer, nullable=False)
    reward = Column(Float, nullable=True)
    attempt_id = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
    rewarded_at = Column(DateTime, nullable=True)


class ScorePredictionSnapshot(Base):
    __tablename__ = "score_prediction_snapshots"
    snapshot_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, nullable=False, index=True)
    snapshot_date = Column(String, nullable=False)
    expected_score = Column(Float, nullable=False)
    low_90 = Column(Float, nullable=False)
    high_90 = Column(Float, nullable=False)
    global_theta = Column(Float, nullable=True)
    attempts_so_far = Column(Integer, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.datetime.now(datetime.timezone.utc))
