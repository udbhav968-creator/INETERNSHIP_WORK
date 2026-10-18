"""
SQLAlchemy models — mirrors docs/spec.md Section 5 exactly.
Using SQLite for local dev (zero setup). Swap DATABASE_URL in database.py
for Postgres later without touching these models.
"""
from sqlalchemy import Column, String, Integer, Boolean, Float, JSON, ForeignKey, DateTime
from sqlalchemy.orm import declarative_base
import datetime

Base = declarative_base()


class Student(Base):
    __tablename__ = "students"
    student_id = Column(String, primary_key=True)
    name = Column(String, nullable=False)

    # --- auth ---
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)

    # --- onboarding ---
    class_level = Column(String, nullable=True)          # "11" | "12" | "dropper"
    target_neet_year = Column(Integer, nullable=True)
    onboarded = Column(Boolean, default=False, nullable=False)

    created_at = Column(DateTime, default=datetime.datetime.utcnow)


class Topic(Base):
    __tablename__ = "topics"
    topic_id = Column(String, primary_key=True)
    subject = Column(String, nullable=False)
    name = Column(String, nullable=False)


class Question(Base):
    __tablename__ = "questions"
    question_id = Column(String, primary_key=True)
    topic_id = Column(String, ForeignKey("topics.topic_id"), nullable=False)
    difficulty = Column(String, nullable=False)  # "basic" | "medium" | "advanced"
    expected_time_seconds = Column(Integer, nullable=False)
    text = Column(String, nullable=False)
    options = Column(JSON, nullable=False)              # list[str]
    correct_answer = Column(String, nullable=False)
    wrong_option_tags = Column(JSON, nullable=True)      # {option: reason}
    years_asked = Column(JSON, nullable=True)            # list[int]


class StudentTopicState(Base):
    __tablename__ = "student_topic_state"
    student_id = Column(String, ForeignKey("students.student_id"), primary_key=True)
    topic_id = Column(String, ForeignKey("topics.topic_id"), primary_key=True)
    level = Column(String, default="basic")              # "basic" | "medium" | "advanced"
    streak = Column(Integer, default=0)
    last_3_results = Column(JSON, default=list)           # list[bool], most recent last
    avg_time_ratio = Column(Float, nullable=True)
    attempts_counted = Column(Integer, default=0)          # correct-only attempts counted into avg_time_ratio
    pace_flag = Column(String, nullable=True)              # "fast" | "slow" | None


class TestAttempt(Base):
    __tablename__ = "test_attempts"
    attempt_id = Column(Integer, primary_key=True, autoincrement=True)
    student_id = Column(String, ForeignKey("students.student_id"), nullable=False)
    question_id = Column(String, ForeignKey("questions.question_id"), nullable=False)
    paper_number = Column(Integer, nullable=False)
    is_correct = Column(Boolean, nullable=False)
    time_taken_seconds = Column(Integer, nullable=False)
    selected_option = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
