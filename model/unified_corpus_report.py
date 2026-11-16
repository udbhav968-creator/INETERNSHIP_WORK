import json
from collections import Counter
from pathlib import Path

from .assessment_blueprint import SUBJECT_QUOTAS, TIER_ORDER, blueprint_targets, load_question_bank
from .mock_test_generator import mock_capacity, reserved_initial_assessment_ids

MODEL_DIR = Path(__file__).resolve().parent
REPORT_PATH = MODEL_DIR / "unified_corpus_readiness_report.json"


def build_report(save: bool = True) -> dict:
    bank = load_question_bank(source="unified")
    targets = blueprint_targets(bank)
    reserved = reserved_initial_assessment_ids(include_siblings=True)
    cells = Counter((q["unit"], q["tier"], q["source_type"]) for q in bank)
    chapters = []
    for unit, quota in sorted(targets["chapter_quota"].items()):
        row = {"subject": unit.split("::")[0], "chapter": unit.split("::")[1], "quota_per_180q_paper": quota}
        for t in TIER_ORDER:
            row[f"{t}_pyq"] = cells.get((unit, t, "pyq"), 0)
            row[f"{t}_sample"] = cells.get((unit, t, "sample_paper"), 0)
        avail = sum(1 for q in bank if q["unit"] == unit and q["question_id"] not in reserved)
        row["available_for_mocks"] = avail
        row["fresh_mocks_supported"] = avail // quota if quota else None
        chapters.append(row)
    chapters.sort(key=lambda r: (r["fresh_mocks_supported"] if r["fresh_mocks_supported"] is not None else 10 ** 6))
    report = {
        "bank_size": len(bank),
        "by_source": dict(Counter(q["source_type"] for q in bank)),
        "by_subject_tier": {
            s: {t: sum(1 for q in bank if q["subject"] == s and q["tier"] == t) for t in TIER_ORDER} for s in SUBJECT_QUOTAS
        },
        "reserved_for_initial_assessment_incl_near_twins": len(reserved),
        "mock_capacity": mock_capacity("unified"),
        "weakest_chapters_for_mock_capacity": chapters[:12],
        "chapters": chapters,
    }
    if save:
        REPORT_PATH.write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


if __name__ == "__main__":
    r = build_report()
    print({k: r[k] for k in ("bank_size", "by_source", "by_subject_tier", "reserved_for_initial_assessment_incl_near_twins", "mock_capacity")})
    for c in r["weakest_chapters_for_mock_capacity"]:
        print(c["subject"], c["chapter"], "quota", c["quota_per_180q_paper"], "avail", c["available_for_mocks"], "fresh mocks", c["fresh_mocks_supported"])
