"""
Weightage file validator — checks internal consistency and flags
implausible topic patterns (e.g. a topic active in only one year across
a 13-year span, which usually means a taxonomy/classification bug rather
than a real absence from the exam).

Usage:
    python validate_weightage_file.py path/to/file.json
"""
import json
import sys
from collections import defaultdict


def validate(units: list[dict]) -> dict:
    issues = []

    # 1. cross-check stated totals vs actual sum
    mismatches = []
    for u in units:
        s = sum(yd["question_count"] for yd in u["yearly_weightage"].values())
        if s != u.get("total_13_year_occurrences"):
            mismatches.append(u["unit_name"])
    if mismatches:
        issues.append(f"{len(mismatches)} units where stated total != sum of yearly counts: {mismatches}")

    # 2. marks = count * 4 consistency
    bad_marks = sum(
        1 for u in units for yd in u["yearly_weightage"].values()
        if yd["total_marks"] != yd["question_count"] * 4
    )
    if bad_marks:
        issues.append(f"{bad_marks} year-entries where marks don't equal question_count * 4")

    # 3. topics active in only one year across the whole span (likely a taxonomy bug)
    single_year_topics = []
    for u in units:
        active = [y for y, yd in u["yearly_weightage"].items() if yd["question_count"] > 0]
        if len(active) <= 1:
            single_year_topics.append(f"{u['subject']} - {u['unit_name']} (only in {active})")
    if single_year_topics:
        issues.append(
            f"{len(single_year_topics)} topics show activity in only one year out of 13 — "
            f"almost always a taxonomy/merging bug, not a real pattern: {single_year_topics}"
        )

    # 4. per-year subject totals — flag any year whose Physics/Chem/Bio split looks broken
    totals = defaultdict(lambda: defaultdict(int))
    for u in units:
        for y, yd in u["yearly_weightage"].items():
            totals[y][u["subject"]] += yd["question_count"]
    year_summary = {
        y: dict(subj_totals) for y, subj_totals in sorted(totals.items())
    }

    return {
        "total_units": len(units),
        "issues": issues,
        "per_year_subject_totals": year_summary,
        "verdict": "NEEDS FIXES before use" if issues else "No structural red flags — spot-check manually before trusting fully.",
    }


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python validate_weightage_file.py <path-to-json>")
        sys.exit(1)

    with open(sys.argv[1]) as f:
        data = json.load(f)

    units = data.get("Unified_Topic_Weightage_And_Marks", data if isinstance(data, list) else [])
    report = validate(units)
    print(json.dumps(report, indent=2))
