import json
import sys
from collections import defaultdict


def validate(units: list[dict]) -> dict:
    issues = []

    mismatches = []
    for u in units:
        s = sum(yd["question_count"] for yd in u["yearly_weightage"].values())
        if s != u.get("total_13_year_occurrences"):
            mismatches.append(u["unit_name"])
    if mismatches:
        issues.append(f"{len(mismatches)} units where stated total != sum of yearly counts: {mismatches}")

    bad_marks = sum(
        1 for u in units for yd in u["yearly_weightage"].values()
        if yd["total_marks"] != yd["question_count"] * 4
    )
    if bad_marks:
        issues.append(f"{bad_marks} year-entries where marks don't equal question_count * 4")

    single_year_topics = []
    for u in units:
        active = [y for y, yd in u["yearly_weightage"].items() if yd["question_count"] > 0]
        if len(active) <= 1:
            single_year_topics.append(f"{u['subject']} - {u['unit_name']} (only in {active})")
    if single_year_topics:
        issues.append(
            f"{len(single_year_topics)} topics show activity in only one year out of 13: {single_year_topics}"
        )

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
        "verdict": "NEEDS FIXES before use" if issues else "Valid",
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python validate_weightage_file.py path/to/file.json")
        sys.exit(1)
    with open(sys.argv[1], encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "Unified_Topic_Weightage_And_Marks" in data:
        data = data["Unified_Topic_Weightage_And_Marks"]
    result = validate(data)
    print(json.dumps(result, indent=2))
    sys.exit(1 if result["issues"] else 0)
