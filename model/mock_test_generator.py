import random
from collections import Counter, defaultdict

import numpy as np

from .assessment_blueprint import (
    BLUEPRINT_VERSION,
    DURATION_MINUTES,
    MAX_MARKS,
    SUBJECT_ORDER,
    SUBJECT_QUOTAS,
    TCC_TOLERANCE_MARKS,
    TIF_TOLERANCE_RATIO,
    _curves,
    _greedy_form,
    _item_curves,
    _stable_seed,
    blueprint_targets,
    load_forms,
    load_question_bank,
)
from .unified_similarity_guard import get_unified_siblings_map

MOCK_VERSION = f"{BLUEPRINT_VERSION}-mock"
REPEAT_PENALTY = 4.0
TIER_PENALTY = 0.5
CHAPTER_PENALTY = 3.0
SAME_CHAPTER_PROPOSAL = 0.7
OPTIMIZER_ITERATIONS = 24000
_BANK_CACHE: dict[str, tuple[list[dict], dict, dict]] = {}


def _bank_bundle(source: str) -> tuple[list[dict], dict, dict]:
    if source not in _BANK_CACHE:
        bank = load_question_bank(source=source)
        cache = {q["question_id"]: _item_curves(q) for q in bank}
        _BANK_CACHE[source] = (bank, cache, blueprint_targets(bank))
    return _BANK_CACHE[source]


def reserved_initial_assessment_ids(include_siblings: bool = False) -> set[str]:
    try:
        ids = {qid for f in load_forms()["forms"] for qid in f["question_ids"]}
    except FileNotFoundError:
        return set()
    if include_siblings:
        sib = get_unified_siblings_map()
        ids |= {s for q in list(ids) for s in sib.get(q, [])}
    return ids


def reference_curves(cache: dict, bank_by_id: dict) -> tuple[np.ndarray, np.ndarray]:
    forms = load_forms()["forms"]
    curves = [_curves([bank_by_id[q] for q in f["question_ids"] if q in bank_by_id], cache) for f in forms]
    return np.mean([c[0] for c in curves], axis=0), np.mean([c[1] for c in curves], axis=0)


def mock_capacity(source: str = "unified", reserve_initial_forms: bool = True) -> dict:
    bank, _, targets = _bank_bundle(source)
    reserved = reserved_initial_assessment_ids(include_siblings=True) if reserve_initial_forms else set()
    avail_unit = Counter(q["unit"] for q in bank if q["question_id"] not in reserved)
    avail_tier = Counter((q["subject"], q["tier"]) for q in bank if q["question_id"] not in reserved)
    per_unit = {u: avail_unit.get(u, 0) // n for u, n in targets["chapter_quota"].items() if n > 0}
    per_tier = {
        f"{s}::{t}": avail_tier.get((s, t), 0) // n
        for s, tq in targets["subject_tier_quota"].items()
        for t, n in tq.items()
        if n > 0
    }
    bottleneck_unit = min(per_unit, key=per_unit.get)
    bottleneck_tier = min(per_tier, key=per_tier.get)
    return {
        "source": source,
        "pool_size": len(bank) - len(reserved & {q["question_id"] for q in bank}),
        "non_repeating_mocks_by_chapter_quota": per_unit[bottleneck_unit],
        "non_repeating_mocks_by_tier_quota": per_tier[bottleneck_tier],
        "non_repeating_mocks_guaranteed": min(per_unit[bottleneck_unit], per_tier[bottleneck_tier]),
        "bottleneck_chapter": bottleneck_unit,
        "bottleneck_tier_cell": bottleneck_tier,
    }


def _fill_deficits(form: list[dict], bank: list[dict], targets: dict, allowed: set[str], blocked: set[str], rng: random.Random) -> list[dict]:
    have = Counter(q["unit"] for q in form)
    by_unit = defaultdict(list)
    for q in bank:
        if q["question_id"] in allowed and q["question_id"] not in blocked:
            by_unit[q["unit"]].append(q)
    out = list(form)
    taken = {q["question_id"] for q in out}
    for unit, quota in sorted(targets["chapter_quota"].items()):
        need = quota - have.get(unit, 0)
        cands = [q for q in by_unit.get(unit, []) if q["question_id"] not in taken]
        rng.shuffle(cands)
        for q in cands[:max(0, need)]:
            out.append(q)
            taken.add(q["question_id"])
    for subject, quota in SUBJECT_QUOTAS.items():
        cur = sum(1 for q in out if q["subject"] == subject)
        if cur < quota:
            extra = [q for u, qs in by_unit.items() if u.startswith(subject + "::") for q in qs if q["question_id"] not in taken]
            rng.shuffle(extra)
            for q in extra[: quota - cur]:
                out.append(q)
                taken.add(q["question_id"])
    return out


def _curve_loss(tcc: np.ndarray, tif: np.ndarray, target_tcc: np.ndarray, target_tif: np.ndarray) -> float:
    return float(
        np.sum(((tcc - target_tcc) / TCC_TOLERANCE_MARKS) ** 2)
        + np.sum(((tif - target_tif) / (TIF_TOLERANCE_RATIO * target_tif)) ** 2)
    )


def _optimize(
    form: list[dict],
    bank: list[dict],
    cache: dict,
    targets: dict,
    target_tcc: np.ndarray,
    target_tif: np.ndarray,
    seen: set[str],
    reserved: set[str],
    siblings: dict[str, list[str]],
    rng: random.Random,
) -> list[dict]:
    form = list(form)
    by_unit: dict[str, list[dict]] = defaultdict(list)
    by_subject: dict[str, list[dict]] = defaultdict(list)
    for q in bank:
        if q["question_id"] not in reserved:
            by_unit[q["unit"]].append(q)
            by_subject[q["subject"]].append(q)
    tier_target = {(s, t): n for s, tq in targets["subject_tier_quota"].items() for t, n in tq.items()}
    tier_count = Counter((q["subject"], q["tier"]) for q in form)
    chapter_target = dict(targets["chapter_quota"])
    chapter_count = Counter(q["unit"] for q in form)

    def chapter_dev(counter: Counter) -> int:
        return sum((counter.get(u, 0) - n) ** 2 for u, n in chapter_target.items())

    in_form = {q["question_id"] for q in form}
    sib_block = Counter(sib for q in form for sib in siblings.get(q["question_id"], []))
    repeats = sum(1 for q in form if q["question_id"] in seen)
    tcc, tif = _curves(form, cache)

    def tier_dev(counter: Counter) -> int:
        return sum(abs(counter.get(k, 0) - v) for k, v in tier_target.items())

    def total(tcc_, tif_, rep_, tc_, cc_) -> float:
        return (
            _curve_loss(tcc_, tif_, target_tcc, target_tif)
            + REPEAT_PENALTY * rep_
            + TIER_PENALTY * tier_dev(tc_)
            + CHAPTER_PENALTY * chapter_dev(cc_)
        )

    best = total(tcc, tif, repeats, tier_count, chapter_count)
    for _ in range(OPTIMIZER_ITERATIONS):
        idx = rng.randrange(len(form))
        old = form[idx]
        pool = by_unit.get(old["unit"], []) if rng.random() < SAME_CHAPTER_PROPOSAL else by_subject.get(old["subject"], [])
        if len(pool) < 2:
            continue
        new = pool[rng.randrange(len(pool))]
        nid = new["question_id"]
        if nid in in_form or sib_block.get(nid, 0) > 0:
            continue
        e_o, i_o = cache[old["question_id"]]
        e_n, i_n = cache[nid]
        tcc2, tif2 = tcc - e_o + e_n, tif - i_o + i_n
        rep2 = repeats - (old["question_id"] in seen) + (nid in seen)
        tc2 = tier_count.copy()
        tc2[(old["subject"], old["tier"])] -= 1
        tc2[(new["subject"], new["tier"])] += 1
        cc2 = chapter_count
        if new["unit"] != old["unit"]:
            cc2 = chapter_count.copy()
            cc2[old["unit"]] -= 1
            cc2[new["unit"]] += 1
        cand = total(tcc2, tif2, rep2, tc2, cc2)
        if cand < best:
            form[idx] = new
            in_form.discard(old["question_id"])
            in_form.add(nid)
            for sib in siblings.get(old["question_id"], []):
                sib_block[sib] -= 1
            for sib in siblings.get(nid, []):
                sib_block[sib] += 1
            tcc, tif, repeats, tier_count, chapter_count, best = tcc2, tif2, rep2, tc2, cc2, cand
    return form


def generate_mock_180(
    student_id: str,
    attempt_index: int,
    seen_question_ids: set[str] | None = None,
    source: str = "unified",
    reserve_initial_forms: bool = True,
) -> dict:
    bank, cache, targets = _bank_bundle(source)
    by_id = {q["question_id"]: q for q in bank}
    siblings = get_unified_siblings_map()
    seen = set(seen_question_ids or set())
    seen |= {s for q in list(seen) for s in siblings.get(q, [])}
    reserved = reserved_initial_assessment_ids(include_siblings=True) if reserve_initial_forms else set()
    seed = _stable_seed(MOCK_VERSION, source, student_id, attempt_index)
    rng = random.Random(seed)

    blocked_fresh = seen | reserved
    form = _greedy_form(bank, targets, random.Random(seed), blocked_fresh, siblings)
    if len(form) < sum(SUBJECT_QUOTAS.values()):
        form = _fill_deficits(form, bank, targets, allowed=seen - reserved, blocked=set(), rng=rng)
    if len(form) < sum(SUBJECT_QUOTAS.values()):
        form = _fill_deficits(form, bank, targets, allowed=set(by_id), blocked=set(), rng=rng)

    target_tcc, target_tif = reference_curves(cache, by_id)
    form = _optimize(form, bank, cache, targets, target_tcc, target_tif, seen, reserved, siblings, rng)

    ordered = []
    for subject in SUBJECT_ORDER:
        sec = sorted([q for q in form if q["subject"] == subject], key=lambda q: q["question_id"])
        rng.shuffle(sec)
        ordered.extend(sec[: SUBJECT_QUOTAS[subject]])
    tcc, tif = _curves(ordered, cache)
    return {
        "test_type": "mock_180",
        "mock_version": MOCK_VERSION,
        "student_id": student_id,
        "attempt_index": attempt_index,
        "source": source,
        "duration_minutes": DURATION_MINUTES,
        "max_marks": MAX_MARKS,
        "question_ids": [q["question_id"] for q in ordered],
        "composition": {
            "by_subject": dict(Counter(q["subject"] for q in ordered)),
            "by_difficulty": dict(Counter(q["tier"] for q in ordered)),
            "by_source": dict(Counter(q["source_type"] for q in ordered)),
            "chapters_covered": len({q["unit"] for q in ordered}),
        },
        "repeated_items": sum(1 for q in ordered if q["question_id"] in seen),
        "chapter_weightage_deviation": {
            "chapters_off_target": sum(
                1 for u, n in targets["chapter_quota"].items() if Counter(q["unit"] for q in ordered).get(u, 0) != n
            ),
            "max_abs_deviation": max(
                abs(Counter(q["unit"] for q in ordered).get(u, 0) - n) for u, n in targets["chapter_quota"].items()
            ),
        },
        "equivalence_to_initial_assessment": {
            "max_expected_marks_gap": round(float(np.max(np.abs(tcc - target_tcc))), 2),
            "max_test_information_ratio_gap": round(float(np.max(np.abs(tif - target_tif) / target_tif)), 4),
            "within_tolerance": bool(
                np.max(np.abs(tcc - target_tcc)) <= TCC_TOLERANCE_MARKS
                and np.max(np.abs(tif - target_tif) / target_tif) <= TIF_TOLERANCE_RATIO
            ),
        },
    }
