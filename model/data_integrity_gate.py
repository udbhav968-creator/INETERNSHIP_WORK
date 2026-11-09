from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .build_verified_corpus import is_official_pyq

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CLEAN_QBANK_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
INTEGRITY_REPORT_PATH = MODEL_DIR / "data_integrity_report.json"
TIER_AUDIT_MARKDOWN_PATH = PROJECT_ROOT / "docs" / "QUESTION_BANK_TIER_AUDIT.md"

TIERS = ("basic", "medium", "advanced")
CHI_SQUARE_CRITICAL_DF3_P001 = 16.266
MIN_GROUP_SIZE_FOR_KEY_TEST = 100


def _chi_square_uniform(counts: Counter) -> tuple[float, int]:
    n = sum(counts.values())
    if n == 0:
        return 0.0, 0
    expected = n / 4.0
    stat = sum((counts.get(k, 0) - expected) ** 2 / expected for k in ("a", "b", "c", "d"))
    return stat, n


def _key_distribution(group: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(q.get("correct_option", "")).strip().lower() for q in group)
    stat, n = _chi_square_uniform(counts)
    return {
        "n": n,
        "counts": {k: counts.get(k, 0) for k in ("a", "b", "c", "d")},
        "percent": {k: round(100.0 * counts.get(k, 0) / n, 2) if n else 0.0 for k in ("a", "b", "c", "d")},
        "chi_square_df3": round(stat, 3),
        "skewed_p_lt_0_001": bool(n >= MIN_GROUP_SIZE_FOR_KEY_TEST and stat > CHI_SQUARE_CRITICAL_DF3_P001),
    }


def _tier_matrix(questions: list[dict[str, Any]], tier_key: str) -> dict[str, dict[str, Counter]]:
    matrix: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    for q in questions:
        tier = str(q.get(tier_key, "")).strip().lower()
        if tier in TIERS:
            matrix[str(q["subject"]).strip()][str(q["chapter"]).strip()][tier] += 1
    return matrix


def _zero_cells(matrix: dict[str, dict[str, Counter]]) -> list[dict[str, str]]:
    cells: list[dict[str, str]] = []
    for subject, chapters in matrix.items():
        for chapter, counter in chapters.items():
            for tier in TIERS:
                if counter.get(tier, 0) == 0:
                    cells.append({"subject": subject, "chapter": chapter, "tier": tier})
    return sorted(cells, key=lambda c: (c["subject"], c["chapter"], c["tier"]))


def run_data_integrity_gate(
    questions: list[dict[str, Any]] | None = None,
    save_report: bool = True,
) -> dict[str, Any]:
    if questions is None:
        questions = json.loads(CLEAN_QBANK_PATH.read_text(encoding="utf-8"))

    blocking: list[str] = []
    warnings: list[str] = []

    non_official = [q for q in questions if not is_official_pyq(q)]
    if non_official:
        blocking.append(
            f"{len(non_official)} question(s) lack verifiable official exam provenance (paper_source / question_id)."
        )

    overall_key = _key_distribution(questions)
    key_groups: list[dict[str, Any]] = []
    if overall_key["skewed_p_lt_0_001"]:
        blocking.append(
            f"Answer-key position is skewed overall (chi-square {overall_key['chi_square_df3']} > {CHI_SQUARE_CRITICAL_DF3_P001})."
        )
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for q in questions:
        grouped[f"source::{q.get('paper_source', 'unknown')}"].append(q)
        grouped[f"subject::{q.get('subject', 'unknown')}"].append(q)
    for name, group in sorted(grouped.items()):
        if len(group) < MIN_GROUP_SIZE_FOR_KEY_TEST:
            continue
        dist = _key_distribution(group)
        dist["group"] = name
        key_groups.append(dist)
        if dist["skewed_p_lt_0_001"]:
            blocking.append(f"Answer-key position is skewed within {name} (chi-square {dist['chi_square_df3']}).")

    ids = [str(q.get("question_id", "")).strip() for q in questions]
    dup_ids = sorted(k for k, v in Counter(ids).items() if v > 1)
    bad_structure: list[str] = []
    for q in questions:
        opts = [str(q.get(k, "")).strip() for k in ("option_a", "option_b", "option_c", "option_d")]
        if any(not o for o in opts) or len(set(o.lower() for o in opts)) < 4:
            bad_structure.append(str(q.get("question_id")))
        elif str(q.get("correct_option", "")).strip().lower() not in ("a", "b", "c", "d"):
            bad_structure.append(str(q.get("question_id")))
    if dup_ids:
        blocking.append(f"{len(dup_ids)} duplicate question_id value(s).")
    if bad_structure:
        blocking.append(f"{len(bad_structure)} question(s) have missing/duplicate options or an invalid answer key.")

    current_matrix = _tier_matrix(questions, "difficulty_level")
    current_zero = _zero_cells(current_matrix)
    has_original = any("original_difficulty_level" in q for q in questions)
    original_zero: list[dict[str, str]] = []
    original_subject_totals: dict[str, dict[str, int]] = {}
    if has_original:
        original_matrix = _tier_matrix(questions, "original_difficulty_level")
        original_zero = _zero_cells(original_matrix)
        for subject, chapters in original_matrix.items():
            tot = Counter()
            for counter in chapters.values():
                tot.update(counter)
            original_subject_totals[subject] = {t: int(tot.get(t, 0)) for t in TIERS}
        phys_basic = original_subject_totals.get("Physics", {}).get("basic", 0)
        if phys_basic == 0:
            warnings.append(
                "Physics has 0 Basic questions under the ORIGINAL source tags; current tiers are a heuristic relabelling."
            )
    if current_zero:
        warnings.append(f"{len(current_zero)} (chapter, tier) cell(s) have zero questions under current tiers.")
    if original_zero:
        warnings.append(f"{len(original_zero)} (chapter, tier) cell(s) have zero questions under original source tags.")

    heuristic_tiers = sum(1 for q in questions if q.get("difficulty_source") == "chapter_text_complexity_quantile_heuristic")
    synthetic_time = sum(1 for q in questions if q.get("ideal_time_source") == "synthetic_tier_formula")
    if heuristic_tiers:
        warnings.append(
            f"{heuristic_tiers}/{len(questions)} difficulty tiers are heuristic (within-chapter text-complexity quantiles), not empirical."
        )
    if synthetic_time:
        warnings.append(
            f"{synthetic_time}/{len(questions)} ideal_time_sec values are synthetic (tier formula), not measured solve times."
        )

    subject_totals: dict[str, dict[str, int]] = {}
    for subject, chapters in current_matrix.items():
        tot = Counter()
        for counter in chapters.values():
            tot.update(counter)
        subject_totals[subject] = {t: int(tot.get(t, 0)) for t in TIERS}

    chapter_counts = {
        subject: {chapter: int(sum(c.values())) for chapter, c in sorted(chapters.items())}
        for subject, chapters in current_matrix.items()
    }
    below_40 = sorted(
        (
            {"subject": subject, "chapter": chapter, "count": count}
            for subject, chapters in chapter_counts.items()
            for chapter, count in chapters.items()
            if count < 40
        ),
        key=lambda r: (r["count"], r["chapter"]),
    )

    report = {
        "status": "FAILED" if blocking else "PASSED",
        "blocking_failures": blocking,
        "warnings": warnings,
        "totals": {
            "questions": len(questions),
            "chapters": len({str(q["chapter"]).strip() for q in questions}),
            "official_pyq_provenance_verified": len(questions) - len(non_official),
        },
        "answer_key": {"overall": overall_key, "groups": key_groups},
        "structure": {
            "duplicate_ids": dup_ids[:20],
            "invalid_questions": bad_structure[:20],
            "invalid_count": len(bad_structure),
        },
        "tier_coverage": {
            "current_tiers": {
                "subject_totals": subject_totals,
                "zero_cells": current_zero,
                "zero_cell_count": len(current_zero),
                "chapters_below_40_questions": below_40,
                "chapters_below_40_count": len(below_40),
            },
            "original_source_tags": {
                "available": has_original,
                "subject_totals": original_subject_totals,
                "zero_cells": original_zero,
                "zero_cell_count": len(original_zero),
            },
        },
        "label_provenance": {
            "difficulty_tiers_heuristic": heuristic_tiers,
            "ideal_time_synthetic": synthetic_time,
        },
    }
    if save_report:
        INTEGRITY_REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def render_tier_audit_markdown(questions: list[dict[str, Any]] | None = None) -> str:
    if questions is None:
        questions = json.loads(CLEAN_QBANK_PATH.read_text(encoding="utf-8"))
    cur = _tier_matrix(questions, "difficulty_level")
    has_orig = any("original_difficulty_level" in q for q in questions)
    orig = _tier_matrix(questions, "original_difficulty_level") if has_orig else {}

    lines = [
        "# Question Bank Tier Audit (official PYQs only)",
        "",
        "Generated by `python -m model.data_integrity_gate --markdown`. Every zero is flagged with `ZERO`.",
        "",
        "- **Current tiers** are the engine's tiers. They are a heuristic (within-chapter text-complexity quantiles), not measured difficulty.",
        "- **Original tags** are the tiers present in the source data before that relabelling.",
        "",
    ]
    for subject in ("Physics", "Chemistry", "Biology"):
        chapters = sorted(cur.get(subject, {}).keys())
        lines += [
            f"## {subject}",
            "",
            "| Chapter | Basic | Medium | Advanced | Total | Orig Basic | Orig Medium | Orig Advanced | Flags |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
        ]
        for ch in chapters:
            c = cur[subject][ch]
            o = orig.get(subject, {}).get(ch, Counter())
            flags = [f"ZERO current {t}" for t in TIERS if c.get(t, 0) == 0]
            if has_orig:
                flags += [f"ZERO original {t}" for t in TIERS if o.get(t, 0) == 0]
            total = sum(c.values())
            if total < 40:
                flags.append(f"only {total} questions")
            lines.append(
                f"| {ch} | {c.get('basic', 0)} | {c.get('medium', 0)} | {c.get('advanced', 0)} | {total} | "
                f"{o.get('basic', 0)} | {o.get('medium', 0)} | {o.get('advanced', 0)} | {'; '.join(flags) or '-'} |"
            )
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    rep = run_data_integrity_gate(save_report=True)
    if "--markdown" in sys.argv:
        TIER_AUDIT_MARKDOWN_PATH.parent.mkdir(parents=True, exist_ok=True)
        TIER_AUDIT_MARKDOWN_PATH.write_text(render_tier_audit_markdown() + "\n", encoding="utf-8")
    print(json.dumps({k: rep[k] for k in ("status", "blocking_failures", "warnings", "totals")}, indent=2))
    sys.exit(1 if rep["status"] == "FAILED" else 0)
