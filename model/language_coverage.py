from __future__ import annotations

import json
import re
from pathlib import Path

from .multilingual_glossary import CHAPTER_HINDI_DICTIONARY

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
TRANSLATIONS_DIR = PROJECT_ROOT / "data" / "translations"
REPORT_PATH = MODEL_DIR / "language_coverage_report.json"

SCRIPTS = {
    "hi": ("Hindi", r"[ऀ-ॿ]"),
    "mr": ("Marathi", r"[ऀ-ॿ]"),
    "bn": ("Bengali", r"[ঀ-৿]"),
    "as": ("Assamese", r"[ঀ-৿]"),
    "gu": ("Gujarati", r"[઀-૿]"),
    "pa": ("Punjabi", r"[਀-੿]"),
    "or": ("Odia", r"[଀-୿]"),
    "ta": ("Tamil", r"[஀-௿]"),
    "te": ("Telugu", r"[ఀ-౿]"),
    "kn": ("Kannada", r"[ಀ-೿]"),
    "ml": ("Malayalam", r"[ഀ-ൿ]"),
    "ur": ("Urdu", r"[؀-ۿ]"),
}
REQUIRED_FIELDS = ("question_id", "question_text", "option_a", "option_b", "option_c", "option_d", "translator", "reviewer", "status")
_NUM = re.compile(r"\d+(?:\.\d+)?")
_FORMULA = re.compile(r"\b(?:[A-Z][a-z]?\d*){2,}\b|\b[A-Z][a-z]?\d+\b")


def validate_translation(source: dict, target: dict, lang: str) -> list[str]:
    issues = [f"missing_{f}" for f in REQUIRED_FIELDS if not str(target.get(f, "")).strip()]
    script = SCRIPTS.get(lang, (None, None))[1]
    fields = ["question_text", "option_a", "option_b", "option_c", "option_d", "explanation"]
    for f in fields:
        src, tgt = str(source.get(f, "") or ""), str(target.get(f, "") or "")
        if not src or not tgt:
            continue
        if sorted(_NUM.findall(src)) != sorted(_NUM.findall(tgt)):
            issues.append(f"{f}:numbers_changed")
        if set(_FORMULA.findall(src)) - set(_FORMULA.findall(tgt)):
            issues.append(f"{f}:formula_or_symbol_lost")
    if script and not re.search(script, str(target.get("question_text", ""))):
        issues.append("question_text_not_in_target_script")
    if str(target.get("status", "")) not in ("draft", "reviewed", "approved"):
        issues.append("status_must_be_draft_reviewed_or_approved")
    if str(target.get("status")) == "approved" and str(target.get("translator", "")).strip() == str(target.get("reviewer", "")).strip():
        issues.append("reviewer_must_differ_from_translator")
    return issues


def _load_language(lang: str) -> list[dict]:
    path = TRANSLATIONS_DIR / lang / "questions.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_language_coverage(save: bool = True) -> dict:
    qs = json.loads(CLEAN_QS_PATH.read_text(encoding="utf-8"))
    by_id = {q["question_id"]: q for q in qs}
    chapters = sorted({q["chapter"] for q in qs})
    languages = {}
    for lang, (name, _) in SCRIPTS.items():
        rows = _load_language(lang)
        approved, failing = 0, []
        for r in rows:
            src = by_id.get(str(r.get("question_id")))
            if src is None:
                failing.append({"question_id": r.get("question_id"), "issues": ["unknown_question_id"]})
                continue
            issues = validate_translation(src, r, lang)
            if issues:
                failing.append({"question_id": r["question_id"], "issues": issues})
            elif r.get("status") == "approved":
                approved += 1
        languages[lang] = {
            "language": name,
            "questions_translated": len(rows),
            "questions_approved_and_valid": approved,
            "question_coverage": round(approved / len(qs), 4),
            "failing_checks": failing[:200],
            "chapter_names_and_glossary": (len([c for c in chapters if c in CHAPTER_HINDI_DICTIONARY]) if lang == "hi" else 0),
        }
    report = {
        "what": "how much of the question bank is available in each Indian language, with automatic checks on every translation",
        "english_questions": len(qs),
        "chapters": len(chapters),
        "translation_file_format": "data/translations/<lang>/questions.jsonl, one JSON object per line with " + ", ".join(REQUIRED_FIELDS) + " and optional explanation",
        "checks": "numbers unchanged, chemical formulas and symbols kept, target script used, status draft/reviewed/approved, approved items reviewed by a second person",
        "languages": languages,
        "status": "NOT_STARTED" if not any(v["questions_translated"] for v in languages.values()) else "IN_PROGRESS",
    }
    if save:
        REPORT_PATH.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    rep = run_language_coverage()
    print(json.dumps({k: {"translated": v["questions_translated"], "approved": v["questions_approved_and_valid"], "chapter_glossary": v["chapter_names_and_glossary"]} for k, v in rep["languages"].items()}, indent=1))
    print(rep["status"])
