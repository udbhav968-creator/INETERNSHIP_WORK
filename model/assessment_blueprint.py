import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .irt_3pl_model import D_SCALING, get_item_params, theta_to_percentile
from .question_similarity_audit import get_near_duplicate_siblings_map

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
QUESTION_BANK_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"
IMPORTANCE_PATH = MODEL_DIR / "topic_importance.json"
FORMS_PATH = MODEL_DIR / "assessment_forms.json"
REPORT_PATH = MODEL_DIR / "assessment_equivalence_report.json"

BLUEPRINT_VERSION = "neet180-irt-equated-v1"
SUBJECT_QUOTAS = {"Physics": 45, "Chemistry": 45, "Biology": 90}
SUBJECT_ORDER = ("Physics", "Chemistry", "Biology")
TIER_ORDER = ("basic", "medium", "advanced")
TIER_SHARE = {"basic": 0.25, "medium": 0.50, "advanced": 0.25}
DEFAULT_FORM_COUNT = 3
MARKS_CORRECT = 4
MARKS_WRONG = -1
MAX_MARKS = MARKS_CORRECT * sum(SUBJECT_QUOTAS.values())
DURATION_MINUTES = 180
THETA_GRID = np.round(np.arange(-3.0, 3.01, 0.5), 2)
TCC_TOLERANCE_MARKS = 6.0
TIF_TOLERANCE_RATIO = 0.05
MEAN_B_TOLERANCE = 0.05
SWAP_ITERATIONS = 6000


def _stable_seed(*parts: object) -> int:
    return int(hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16], 16)


def _largest_remainder(total: int, weights: dict[str, float], floor_each: int = 0) -> dict[str, int]:
    keys = sorted(weights)
    alloc = {k: floor_each for k in keys}
    left = total - floor_each * len(keys)
    wsum = sum(max(0.0, weights[k]) for k in keys) or float(len(keys))
    raw = {k: max(0.0, weights[k]) / wsum * left for k in keys}
    for k in keys:
        alloc[k] += int(raw[k])
    rest = total - sum(alloc.values())
    for k in sorted(keys, key=lambda k: (raw[k] - int(raw[k]), weights[k], k), reverse=True)[:rest]:
        alloc[k] += 1
    return alloc


def _bank_item(r: dict, tier: str, params: dict, source_type: str) -> dict:
    return {
        "question_id": r["question_id"],
        "subject": r["subject"],
        "chapter": r["chapter"],
        "unit": f"{r['subject']}::{r['chapter']}",
        "tier": tier,
        "a": float(params["a"]),
        "b": float(params["b"]),
        "c": float(params["c"]),
        "ideal_time_sec": int(r.get("ideal_time_sec") or 90),
        "source_type": source_type,
    }


def load_question_bank(path: Path = QUESTION_BANK_PATH, source: str = "pyq") -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    bank = []
    for r in rows:
        tier = str(r.get("difficulty_level") or "medium").strip().lower()
        if tier not in TIER_ORDER or r.get("subject") not in SUBJECT_QUOTAS:
            continue
        p = get_item_params(r["question_id"], tier, int(r.get("ideal_time_sec") or 90))
        bank.append(_bank_item(r, tier, {"a": p.a, "b": p.b, "c": p.c}, "pyq"))
    if source in ("unified", "sample_paper"):
        from .sample_paper_ingestion import load_accepted_sample_rows

        sample_rows, params = load_accepted_sample_rows()
        sample = [_bank_item(r, r["difficulty_level"], params[r["question_id"]], "sample_paper") for r in sample_rows]
        bank = sample if source == "sample_paper" else bank + sample
    bank.sort(key=lambda q: q["question_id"])
    return bank


def _item_curves(item: dict) -> tuple[np.ndarray, np.ndarray]:
    z = np.clip(D_SCALING * item["a"] * (THETA_GRID - item["b"]), -30.0, 30.0)
    p_star = 1.0 / (1.0 + np.exp(-z))
    p = item["c"] + (1.0 - item["c"]) * p_star
    q = 1.0 - p
    info = (D_SCALING ** 2) * (item["a"] ** 2) * (q / np.maximum(p, 1e-9)) * ((p - item["c"]) / (1.0 - item["c"])) ** 2
    expected_marks = MARKS_CORRECT * p + MARKS_WRONG * q
    return expected_marks, info


def blueprint_targets(bank: list[dict]) -> dict:
    with open(IMPORTANCE_PATH, encoding="utf-8") as f:
        importance = json.load(f)
    chapters_by_subject: dict[str, set[str]] = defaultdict(set)
    chapter_name = {}
    for q in bank:
        chapters_by_subject[q["subject"]].add(q["unit"])
        chapter_name[q["unit"]] = q["chapter"]
    chapter_quota: dict[str, int] = {}
    for subject, quota in SUBJECT_QUOTAS.items():
        chs = sorted(chapters_by_subject.get(subject, set()))
        weights = {ch: float(importance.get(chapter_name[ch], 1.0)) for ch in chs}
        chapter_quota.update(_largest_remainder(quota, weights, floor_each=1 if quota >= len(chs) else 0))
    tier_quota = {s: _largest_remainder(n, dict(TIER_SHARE)) for s, n in SUBJECT_QUOTAS.items()}
    return {"chapter_quota": chapter_quota, "subject_tier_quota": tier_quota}


def _greedy_form(
    bank: list[dict],
    targets: dict,
    rng: random.Random,
    blocked: set[str],
    siblings: dict[str, list[str]],
) -> list[dict]:
    pools: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for q in bank:
        pools[(q["unit"], q["tier"])].append(q)
    by_subject_chapter: dict[str, list[str]] = defaultdict(list)
    for ch, n in targets["chapter_quota"].items():
        if n > 0:
            by_subject_chapter[ch.split("::", 1)[0]].append(ch)
    local_block = set(blocked)

    def take(ch: str, tier: str) -> dict | None:
        cands = [q for q in pools.get((ch, tier), []) if q["question_id"] not in local_block]
        if not cands:
            return None
        q = cands[rng.randrange(len(cands))]
        local_block.add(q["question_id"])
        local_block.update(siblings.get(q["question_id"], []))
        return q

    form: list[dict] = []
    for subject in SUBJECT_ORDER:
        remaining = dict(targets["subject_tier_quota"][subject])
        chapters = sorted(by_subject_chapter[subject], key=lambda ch: (-targets["chapter_quota"][ch], ch))
        for ch in chapters:
            for _ in range(targets["chapter_quota"][ch]):
                order = sorted(TIER_ORDER, key=lambda t: (remaining[t], -TIER_ORDER.index(t)), reverse=True)
                picked = None
                for tier in order:
                    if remaining[tier] > 0:
                        picked = take(ch, tier)
                        if picked:
                            remaining[tier] -= 1
                            break
                if picked is None:
                    for tier in order:
                        picked = take(ch, tier)
                        if picked:
                            break
                if picked is not None:
                    form.append(picked)
    return form


def _curves(form: list[dict], cache: dict[str, tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    tcc = np.zeros_like(THETA_GRID, dtype=float)
    tif = np.zeros_like(THETA_GRID, dtype=float)
    for q in form:
        e, i = cache[q["question_id"]]
        tcc += e
        tif += i
    return tcc, tif


def _loss(tcc: np.ndarray, tif: np.ndarray, target_tcc: np.ndarray, target_tif: np.ndarray) -> float:
    return float(np.sum(((tcc - target_tcc) / TCC_TOLERANCE_MARKS) ** 2) + np.sum(((tif - target_tif) / (TIF_TOLERANCE_RATIO * target_tif)) ** 2))


def _equate(
    form: list[dict],
    bank: list[dict],
    used_elsewhere: set[str],
    target_tcc: np.ndarray,
    target_tif: np.ndarray,
    cache: dict,
    siblings: dict[str, list[str]],
    rng: random.Random,
) -> list[dict]:
    form = list(form)
    by_cell: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for q in bank:
        by_cell[(q["unit"], q["tier"])].append(q)
    in_form = {q["question_id"] for q in form}

    def blocked_ids() -> set[str]:
        out = set(used_elsewhere) | in_form
        for qid in in_form:
            out.update(siblings.get(qid, []))
        return out

    tcc, tif = _curves(form, cache)
    best = _loss(tcc, tif, target_tcc, target_tif)
    blocked = blocked_ids()
    for _ in range(SWAP_ITERATIONS):
        idx = rng.randrange(len(form))
        old = form[idx]
        cands = [q for q in by_cell[(old["unit"], old["tier"])] if q["question_id"] not in blocked]
        if not cands:
            continue
        new = cands[rng.randrange(len(cands))]
        e_old, i_old = cache[old["question_id"]]
        e_new, i_new = cache[new["question_id"]]
        tcc2, tif2 = tcc - e_old + e_new, tif - i_old + i_new
        cand_loss = _loss(tcc2, tif2, target_tcc, target_tif)
        if cand_loss < best:
            form[idx] = new
            in_form.discard(old["question_id"])
            in_form.add(new["question_id"])
            tcc, tif, best = tcc2, tif2, cand_loss
            blocked = blocked_ids()
    return form


def build_equated_forms(n_forms: int = DEFAULT_FORM_COUNT, bank: list[dict] | None = None) -> dict:
    bank = bank if bank is not None else load_question_bank()
    targets = blueprint_targets(bank)
    siblings = get_near_duplicate_siblings_map()
    cache = {q["question_id"]: _item_curves(q) for q in bank}
    rng = random.Random(_stable_seed(BLUEPRINT_VERSION, n_forms))

    forms: list[list[dict]] = []
    used: set[str] = set()
    for k in range(n_forms):
        f = _greedy_form(bank, targets, random.Random(_stable_seed(BLUEPRINT_VERSION, "form", k)), used, siblings)
        forms.append(f)
        used.update(q["question_id"] for q in f)

    chapter_tier_counts = [Counter((q["unit"], q["tier"]) for q in f) for f in forms]
    reference = min(range(n_forms), key=lambda k: sum(abs(chapter_tier_counts[k][key] - chapter_tier_counts[0][key]) for key in chapter_tier_counts[0]))
    bank_tcc = np.mean([cache[q["question_id"]][0] for q in bank], axis=0) * sum(SUBJECT_QUOTAS.values())
    bank_tif = np.mean([cache[q["question_id"]][1] for q in bank], axis=0) * sum(SUBJECT_QUOTAS.values())
    ref_tcc, ref_tif = _curves(forms[reference], cache)
    target_tcc = 0.5 * (ref_tcc + bank_tcc)
    target_tif = 0.5 * (ref_tif + bank_tif)

    for sweep in range(2):
        for k in range(n_forms):
            others = set()
            for j, g in enumerate(forms):
                if j != k:
                    others.update(q["question_id"] for q in g)
            forms[k] = _equate(forms[k], bank, others, target_tcc, target_tif, cache, siblings, rng)
        stacked = [_curves(f, cache) for f in forms]
        target_tcc = np.mean([s[0] for s in stacked], axis=0)
        target_tif = np.mean([s[1] for s in stacked], axis=0)

    ordered_forms = []
    for k, f in enumerate(forms):
        section_rng = random.Random(_stable_seed(BLUEPRINT_VERSION, "order", k))
        ordered = []
        for subject in SUBJECT_ORDER:
            sec = sorted([q for q in f if q["subject"] == subject], key=lambda q: q["question_id"])
            section_rng.shuffle(sec)
            ordered.extend(sec)
        ordered_forms.append(ordered)

    report = equivalence_report(ordered_forms, cache, targets)
    payload = {
        "blueprint_version": BLUEPRINT_VERSION,
        "duration_minutes": DURATION_MINUTES,
        "total_questions": sum(SUBJECT_QUOTAS.values()),
        "max_marks": MAX_MARKS,
        "marking": {"correct": MARKS_CORRECT, "wrong": MARKS_WRONG, "unattempted": 0},
        "subject_quotas": SUBJECT_QUOTAS,
        "section_order": list(SUBJECT_ORDER),
        "forms": [
            {"form_index": k, "question_ids": [q["question_id"] for q in f]}
            for k, f in enumerate(ordered_forms)
        ],
    }
    return {"forms": payload, "report": report}


def _form_stats(form: list[dict], cache: dict) -> dict:
    tcc, tif = _curves(form, cache)
    b = np.array([q["b"] for q in form])
    return {
        "n_items": len(form),
        "by_subject": dict(Counter(q["subject"] for q in form)),
        "by_tier": dict(Counter(q["tier"] for q in form)),
        "by_subject_tier": {s: dict(Counter(q["tier"] for q in form if q["subject"] == s)) for s in SUBJECT_ORDER},
        "chapters_covered": len({q["unit"] for q in form}),
        "mean_b": round(float(b.mean()), 4),
        "sd_b": round(float(b.std()), 4),
        "mean_a": round(float(np.mean([q["a"] for q in form])), 4),
        "expected_marks_by_theta": {f"{t:+.1f}": round(float(v), 1) for t, v in zip(THETA_GRID, tcc)},
        "test_information_by_theta": {f"{t:+.1f}": round(float(v), 2) for t, v in zip(THETA_GRID, tif)},
        "sem_at_theta0": round(1.0 / math.sqrt(float(tif[list(THETA_GRID).index(0.0)])), 4),
    }


def equivalence_report(forms: list[list[dict]], cache: dict, targets: dict) -> dict:
    stats = [_form_stats(f, cache) for f in forms]
    curves = [_curves(f, cache) for f in forms]
    tccs = np.array([c[0] for c in curves])
    tifs = np.array([c[1] for c in curves])
    max_tcc_gap = float(np.max(tccs.max(axis=0) - tccs.min(axis=0)))
    max_tif_ratio_gap = float(np.max((tifs.max(axis=0) - tifs.min(axis=0)) / tifs.mean(axis=0)))
    mean_b_gap = float(max(s["mean_b"] for s in stats) - min(s["mean_b"] for s in stats))
    ids = [{q["question_id"] for q in f} for f in forms]
    overlap = max((len(ids[i] & ids[j]) for i in range(len(ids)) for j in range(i + 1, len(ids))), default=0)
    checks = {
        "every_form_has_180_items": all(s["n_items"] == 180 for s in stats),
        "subject_quotas_exact": all(s["by_subject"] == SUBJECT_QUOTAS for s in stats),
        "all_chapters_covered": all(s["chapters_covered"] == len(targets["chapter_quota"]) for s in stats),
        "forms_disjoint": overlap == 0,
        "expected_marks_gap_within_tolerance": max_tcc_gap <= TCC_TOLERANCE_MARKS,
        "test_information_gap_within_tolerance": max_tif_ratio_gap <= TIF_TOLERANCE_RATIO,
        "mean_difficulty_gap_within_tolerance": mean_b_gap <= MEAN_B_TOLERANCE,
    }
    tier_dev = {
        k: {
            subj: {t: s["by_subject_tier"][subj].get(t, 0) - targets["subject_tier_quota"][subj][t] for t in TIER_ORDER}
            for subj in SUBJECT_ORDER
        }
        for k, s in enumerate(stats)
    }
    return {
        "blueprint_version": BLUEPRINT_VERSION,
        "tier_label_deviation_by_form": tier_dev,
        "method": "Stratified chapter x tier blueprint, then 3PL IRT equating by item swaps that match test characteristic and test information curves across forms",
        "targets": {
            "subject_quotas": SUBJECT_QUOTAS,
            "difficulty_share": TIER_SHARE,
            "subject_tier_quota": targets["subject_tier_quota"],
            "chapter_quota": targets["chapter_quota"],
        },
        "tolerances": {
            "expected_marks_gap": TCC_TOLERANCE_MARKS,
            "test_information_ratio_gap": TIF_TOLERANCE_RATIO,
            "mean_b_gap": MEAN_B_TOLERANCE,
        },
        "max_expected_marks_gap_across_theta": round(max_tcc_gap, 2),
        "max_test_information_ratio_gap": round(max_tif_ratio_gap, 4),
        "mean_b_gap": round(mean_b_gap, 4),
        "max_pairwise_overlap": overlap,
        "checks": checks,
        "all_checks_passed": all(checks.values()),
        "forms": stats,
    }


def load_forms(path: Path = FORMS_PATH) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def form_index_for_student(student_id: str, n_forms: int = DEFAULT_FORM_COUNT) -> int:
    return _stable_seed(BLUEPRINT_VERSION, "assign", student_id) % max(1, n_forms)


def score_initial_assessment(responses: list[dict], bank: list[dict] | None = None) -> dict:
    bank = bank if bank is not None else load_question_bank()
    by_id = {q["question_id"]: q for q in bank}
    grid = np.linspace(-4.0, 4.0, 161)
    log_post = -0.5 * grid ** 2
    chapter_obs: dict[str, list[int]] = defaultdict(list)
    subject_marks: dict[str, int] = defaultdict(int)
    correct = wrong = 0
    for r in responses:
        q = by_id.get(r.get("question_id"))
        if q is None or r.get("is_correct") is None:
            continue
        y = 1 if r["is_correct"] else 0
        z = np.clip(D_SCALING * q["a"] * (grid - q["b"]), -30.0, 30.0)
        p = q["c"] + (1.0 - q["c"]) / (1.0 + np.exp(-z))
        log_post += y * np.log(np.maximum(p, 1e-12)) + (1 - y) * np.log(np.maximum(1.0 - p, 1e-12))
        chapter_obs[q["unit"]].append(y)
        subject_marks[q["subject"]] += MARKS_CORRECT if y else MARKS_WRONG
        correct += y
        wrong += 1 - y
    post = np.exp(log_post - log_post.max())
    post /= post.sum()
    theta = float(np.sum(grid * post))
    sem = float(math.sqrt(max(1e-9, np.sum((grid - theta) ** 2 * post))))
    return {
        "theta_eap": round(theta, 4),
        "theta_sem": round(sem, 4),
        "percentile": theta_to_percentile(theta),
        "attempted": correct + wrong,
        "correct": correct,
        "marks": correct * MARKS_CORRECT + wrong * MARKS_WRONG,
        "subject_marks": dict(subject_marks),
        "chapter_accuracy": {ch: round(sum(v) / len(v), 4) for ch, v in sorted(chapter_obs.items())},
    }


def main() -> dict:
    out = build_equated_forms()
    with open(FORMS_PATH, "w", encoding="utf-8") as f:
        json.dump(out["forms"], f, indent=2)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(out["report"], f, indent=2)
    rep = out["report"]
    print(f"forms={len(out['forms']['forms'])} all_checks_passed={rep['all_checks_passed']}")
    print(f"max_expected_marks_gap={rep['max_expected_marks_gap_across_theta']} max_tif_ratio_gap={rep['max_test_information_ratio_gap']} mean_b_gap={rep['mean_b_gap']} overlap={rep['max_pairwise_overlap']}")
    for k, v in rep["checks"].items():
        print(f"  {k}: {v}")
    return out


if __name__ == "__main__":
    main()
