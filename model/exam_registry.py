from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class ExamSection:
    subject: str
    questions: int
    compulsory: int


@dataclass(frozen=True)
class ExamSpec:
    exam_id: str
    title: str
    sections: tuple[ExamSection, ...]
    marks_correct: float
    marks_wrong: float
    marks_unattempted: float
    duration_minutes: int
    question_bank: str
    taxonomy: str
    status: str
    sources: tuple[str, ...] = field(default_factory=tuple)

    @property
    def total_questions(self) -> int:
        return sum(s.questions for s in self.sections)

    @property
    def max_score(self) -> float:
        return self.marks_correct * sum(s.compulsory for s in self.sections)

    @property
    def subject_quotas(self) -> dict[str, int]:
        return {s.subject: s.questions for s in self.sections}

    @property
    def guess_break_even_probability(self) -> float:
        gain, loss = self.marks_correct - self.marks_unattempted, self.marks_unattempted - self.marks_wrong
        return loss / (gain + loss) if gain + loss > 0 else 0.0

    def expected_score(self, p_correct: np.ndarray, attempt: np.ndarray | None = None) -> float:
        p = np.asarray(p_correct, dtype=float)
        att = np.ones_like(p, dtype=bool) if attempt is None else np.asarray(attempt, dtype=bool)
        per_item = np.where(att, p * self.marks_correct + (1 - p) * self.marks_wrong, self.marks_unattempted)
        return float(per_item.sum())

    def optimal_attempt_mask(self, p_correct: np.ndarray) -> np.ndarray:
        return np.asarray(p_correct, dtype=float) >= self.guess_break_even_probability


NEET_UG = ExamSpec(
    exam_id="neet_ug",
    title="NEET-UG",
    sections=(ExamSection("Physics", 45, 45), ExamSection("Chemistry", 45, 45), ExamSection("Biology", 90, 90)),
    marks_correct=4.0,
    marks_wrong=-1.0,
    marks_unattempted=0.0,
    duration_minutes=180,
    question_bank="data/tagged_questions/neet_questions_clean.json",
    taxonomy="data/taxonomy/NEET_Unified_Weightage_And_Marks_2013_2026.json",
    status="configured",
    sources=("NTA NEET-UG information bulletin (180 compulsory questions, +4/-1, 3 hours)",),
)

EXAMS: dict[str, ExamSpec] = {NEET_UG.exam_id: NEET_UG}

ONBOARDING_CHECKLIST = (
    "Exam pattern (sections, compulsory counts, marking, duration) taken from the official bulletin and cited in sources",
    "Question bank with verified provenance and chapter tags, passing model.validate_question_bank",
    "Chapter taxonomy and prerequisite graph for the exam",
    "Weightage history for topic importance",
    "Initial IRT priors (cold-start ridge) re-fitted on the new bank",
    "Assessment forms equated with model.assessment_blueprint and passing all equivalence checks",
    "Score-to-rank curve sourced per year, marked approximate where it is",
)


def get_exam(exam_id: str) -> ExamSpec:
    if exam_id not in EXAMS:
        raise KeyError(f"exam '{exam_id}' is not configured; follow ONBOARDING_CHECKLIST to add it")
    return EXAMS[exam_id]


def consistency_with_engine() -> dict:
    from . import assessment_blueprint as bp
    from .monte_carlo_conformal_neet import NEGATIVE_MARKING_BREAK_EVEN_PROB

    spec = NEET_UG
    checks = {
        "subject_quotas": spec.subject_quotas == dict(bp.SUBJECT_QUOTAS),
        "marks_correct": spec.marks_correct == bp.MARKS_CORRECT,
        "marks_wrong": spec.marks_wrong == bp.MARKS_WRONG,
        "max_score": spec.max_score == bp.MAX_MARKS,
        "duration": spec.duration_minutes == bp.DURATION_MINUTES,
        "skip_break_even": abs(spec.guess_break_even_probability - NEGATIVE_MARKING_BREAK_EVEN_PROB) < 1e-9,
    }
    return {"exam": spec.exam_id, "checks": checks, "consistent": all(checks.values())}
