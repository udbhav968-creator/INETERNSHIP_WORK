from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
REPORT_PATH = MODEL_DIR / "explanation_quality_report.json"
MIN_CHARS = 25
MAX_CHARS = 1800
MIN_SPECIFIC_OVERLAP = 0.08
NUMERIC_TOLERANCE = 0.02

BOILERPLATE_PATTERNS = (
    r"analy[sz]e the fundamental principles",
    r"apply the governing scientific relationships",
    r"mathematically and conceptually verified",
    r"identify the core concepts of",
    r"is verified as the correct answer",
    r"evaluate the given statements/data",
    r"analy[sz]e the chemical species and reaction conditions",
    r"comparison with standard iupac / ncert chemical behavio",
    r"identify the given physical parameters in the problem",
    r"yields the final result corresponding to option",
)
_STOP = set(
    "the a an of to in is are and or for on with by as at be this that from it its which what when where how option options correct answer given "
    "following statement statements step true false not no all none above both only can will was were has have than then also into".split()
)
_WRONG_CLAIM = re.compile(r"option\s*\(?([a-d])\)?\s*(?:is|are)\s*(?:the\s*)?(?:correct|right|answer)", re.I)
_NUM = re.compile(r"-?\d+(?:\.\d+)?")


def _tokens(text: str) -> set[str]:
    return {t for t in re.findall(r"[a-z][a-z0-9]+", str(text).lower()) if t not in _STOP and len(t) > 2}


def _numbers(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(str(text)):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def verify_explanation(question: dict, explanation: str, reference_text: str = "") -> dict:
    text = str(explanation or "").strip()
    correct = str(question.get("correct_option", "")).strip().lower()
    issues, warnings = [], []
    if len(text) < MIN_CHARS:
        issues.append("too_short")
    if len(text) > MAX_CHARS:
        issues.append("too_long")
    for m in _WRONG_CLAIM.finditer(text):
        if correct and m.group(1).lower() != correct:
            issues.append(f"claims_option_{m.group(1).lower()}_but_key_is_{correct}")
    if any(re.search(p, text, re.I) for p in BOILERPLATE_PATTERNS):
        issues.append("boilerplate_template")
    q_tokens = _tokens(question.get("question_text", "")) | set().union(*[_tokens(question.get(f"option_{k}", "")) for k in "abcd"])
    e_tokens = _tokens(text)
    overlap = len(q_tokens & e_tokens) / max(1, len(e_tokens))
    if e_tokens and overlap < MIN_SPECIFIC_OVERLAP:
        warnings.append("low_overlap_with_question")
    correct_text = str(question.get(f"option_{correct}", ""))
    key_nums, expl_nums = _numbers(correct_text), _numbers(text)
    if key_nums and expl_nums:
        target = key_nums[0]
        if not any(abs(n - target) <= max(1e-9, abs(target) * NUMERIC_TOLERANCE) for n in expl_nums):
            warnings.append("answer_number_not_found_in_explanation")
    grounding = None
    if reference_text:
        r_tokens = _tokens(reference_text)
        grounding = round(len(r_tokens & e_tokens) / max(1, len(e_tokens)), 3)
        if grounding < MIN_SPECIFIC_OVERLAP:
            warnings.append("weak_grounding_in_reference")
    return {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "question_overlap": round(overlap, 3),
        "grounding_overlap": grounding,
    }


def build_grounded_prompt(question: dict, reference_text: str) -> str:
    opts = "\n".join(f"({k}) {question.get(f'option_{k}', '')}" for k in "abcd")
    return (
        "Explain to a NEET student why the keyed answer is correct and why the others are not. "
        "Use only the facts in the reference. Do not change the answer key.\n\n"
        f"Question: {question.get('question_text', '')}\n{opts}\n"
        f"Keyed answer: ({question.get('correct_option', '')})\n\nReference (NCERT): {reference_text}\n"
    )


def explain_with_guardrails(question: dict, reference_text: str, generate: Callable[[str], str] | None = None, attempts: int = 2) -> dict:
    tried = []
    if generate is not None:
        prompt = build_grounded_prompt(question, reference_text)
        for _ in range(attempts):
            candidate = generate(prompt)
            check = verify_explanation(question, candidate, reference_text)
            tried.append(check)
            if check["passed"] and "weak_grounding_in_reference" not in check["warnings"]:
                return {"text": candidate, "source": "generated_and_verified", "checks": tried}
    stored = str(question.get("explanation", "")).strip()
    check = verify_explanation(question, stored, reference_text)
    tried.append(check)
    if check["passed"]:
        return {"text": stored, "source": "stored", "checks": tried}
    return {"text": None, "source": "none_passed", "checks": tried}


def audit_stored_explanations(save: bool = True) -> dict:
    qs = json.loads(CLEAN_QS_PATH.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    failing = []
    for q in qs:
        res = verify_explanation(q, q.get("explanation", ""))
        for k in res["issues"] + res["warnings"]:
            counts[k] = counts.get(k, 0) + 1
        if not res["passed"]:
            failing.append({"question_id": q["question_id"], "chapter": q.get("chapter"), "issues": res["issues"]})
    report = {
        "what": "automatic checks on the stored explanation of every PYQ (the same checks any AI-written explanation must pass)",
        "questions": len(qs),
        "passed": len(qs) - len(failing),
        "failed": len(failing),
        "issue_counts": dict(sorted(counts.items(), key=lambda kv: -kv[1])),
        "failing_by_chapter": _by_chapter(failing),
        "failing_questions": failing,
        "action": "content team rewrites failing explanations; warnings are for spot checks, not rejection",
    }
    if save:
        REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _by_chapter(failing: list[dict]) -> dict[str, int]:
    out: dict[str, int] = {}
    for f in failing:
        out[f["chapter"]] = out.get(f["chapter"], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    rep = audit_stored_explanations()
    print(json.dumps({k: rep[k] for k in ("questions", "passed", "failed", "issue_counts")}, indent=2))
