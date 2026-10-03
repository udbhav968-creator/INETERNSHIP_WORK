from pydantic import BaseModel
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
