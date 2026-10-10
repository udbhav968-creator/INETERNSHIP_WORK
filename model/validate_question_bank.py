"""
Question bank quality validator.

Run this on ANY tagged batch before trusting it — from this intern or
anyone else. Catches the exact problems found in the first submission:
boilerplate explanations, blanket fake verification, duplicate options,
and non-standard difficulty labels.

Usage:
    python validate_question_bank.py path/to/file.json
"""
import json
import sys
from collections import Counter

BOILERPLATE_PATTERNS = [
    "Applying standard formulas and fundamental principles",
    "Comprehensive step-by-step solution",
    "Step-by-step solution for NEET",
    "yields the correct outcome",
]

ALLOWED_DIFFICULTIES = {"basic", "medium", "advanced"}


def validate(questions: list[dict]) -> dict:
    n = len(questions)
    report = {"total_questions": n, "issues": []}

    boilerplate = sum(
        1 for q in questions
        if any(p in (q.get("explanation") or "") for p in BOILERPLATE_PATTERNS)
    )
    if boilerplate:
        report["issues"].append(
            f"{boilerplate}/{n} ({boilerplate/n*100:.1f}%) explanations look templated/boilerplate, not real content."
        )

    reviewer_notes = Counter(q.get("reviewer_notes") for q in questions if "reviewer_notes" in q)
    if reviewer_notes and reviewer_notes.most_common(1)[0][1] > n * 0.5:
        note, count = reviewer_notes.most_common(1)[0]
        report["issues"].append(
            f"{count}/{n} questions share the identical reviewer note ('{note}') — likely not individually reviewed."
        )

    dup_options = 0
    for q in questions:
        opts = q.get("options")
        if isinstance(opts, dict):
            opts = list(opts.values())
        elif isinstance(opts, list):
            pass
        else:
            opts = [q.get(f"option_{c}") for c in "abcd" if q.get(f"option_{c}") is not None]
        if opts and len(set(opts)) < len(opts):
            dup_options += 1
    if dup_options:
        report["issues"].append(
            f"{dup_options}/{n} ({dup_options/n*100:.1f}%) questions have duplicate answer options — structurally broken MCQs."
        )

    difficulties = Counter(str(q.get("difficulty", q.get("difficulty_level", ""))).lower() for q in questions)
    bad_labels = {d: c for d, c in difficulties.items() if d not in ALLOWED_DIFFICULTIES}
    if bad_labels:
        report["issues"].append(
            f"Non-standard difficulty labels found (need exactly basic/medium/advanced): {bad_labels}"
        )

    # duplicate question_text across supposedly distinct entries (catches fake "variants")
    texts = Counter(q.get("question_text", q.get("text", "")).split("for ", 1)[-1] for q in questions)
    near_dupes = sum(c for c in texts.values() if c > 1)
    if near_dupes:
        report["issues"].append(
            f"{near_dupes} questions share text with at least one other entry after stripping 'Variant N for' prefixes — possible fake duplicated variants."
        )

    report["verdict"] = "REJECT — do not use as production data" if report["issues"] else "No red flags found — still spot-check manually before trusting fully."
    return report


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_question_bank.py <path-to-json>")
        sys.exit(1)

    path = sys.argv[1]

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    # handle both a flat list and the nested {"Curated_Sample_Papers": [...]} shape
    if isinstance(data, dict):
        for key in ("Curated_Sample_Papers", "questions", "Adaptive_Item_Bank"):
            if key in data:
                print(f"\n=== Validating '{key}' ({len(data[key])} entries) ===")
                report = validate(data[key])
                print(json.dumps(report, indent=2))
    else:
        report = validate(data)
        print(json.dumps(report, indent=2))
