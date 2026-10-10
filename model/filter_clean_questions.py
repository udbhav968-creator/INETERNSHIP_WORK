"""
Splits a question bank into a clean subset (safe to load now) and a
quarantined subset (needs fixing) — so you don't have to wait for a 100%
perfect resubmission before using the data that's already good.

Usage:
    python filter_clean_questions.py path/to/question_bank.json
Outputs, next to the input file:
    <name>.clean.json       - safe to load into the DB now
    <name>.quarantine.json  - send back to the intern with specifics
"""
import json
import re
import sys
from pathlib import Path

GENERIC_CHAPTER_PATTERN = re.compile(r"^General (Biology|Chemistry|Physics)$|Core Concepts$")
GLUED_WORDS_PATTERN = re.compile(r"[a-zA-Z]{18,}")


def is_clean(q: dict) -> bool:
    opts = [q.get("option_a"), q.get("option_b"), q.get("option_c"), q.get("option_d")]
    if len(set(opts)) < 4:
        return False
    chapter = q.get("chapter", "")
    topic_tag = q.get("topic_tag", "")
    if GENERIC_CHAPTER_PATTERN.search(f"{chapter} {topic_tag}"):
        return False
    if GLUED_WORDS_PATTERN.search(q.get("question_text", "")):
        return False
    return True


def main(path: str):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    clean = [q for q in data if is_clean(q)]
    quarantine = [q for q in data if not is_clean(q)]

    p = Path(path)
    clean_path = p.with_suffix("").with_suffix(".clean.json")
    quarantine_path = p.with_suffix("").with_suffix(".quarantine.json")

    with open(clean_path, "w") as f:
        json.dump(clean, f, indent=2)
    with open(quarantine_path, "w") as f:
        json.dump(quarantine, f, indent=2)

    print(f"Clean: {len(clean)} questions -> {clean_path}")
    print(f"Quarantined: {len(quarantine)} questions -> {quarantine_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python filter_clean_questions.py <path-to-question-bank.json>")
        sys.exit(1)
    main(sys.argv[1])
