import json
from pathlib import Path

from .dina_cognitive_diagnosis import compute_dina_and_rt_profile
from .knowledge_graph_dag import (
    DAG_NODES,
    compute_hierarchical_shrinkage_prior,
    compute_prerequisite_readiness,
    get_dependents,
    get_direct_prerequisites,
)
from .monte_carlo_conformal_neet import run_monte_carlo_conformal_neet_simulation
from .state_machine.bkt_model import (
    compute_ebbinghaus_forgetting_prior,
    compute_retention_urgency,
    get_chapter_bkt_prior,
)

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
_CHAPTER_SUMMARY_PATH = PROJECT_ROOT / "data" / "taxonomy" / "chapter_wise_neet_pyq_summary.json"

with open(MODEL_DIR / "topic_importance.json", encoding="utf-8") as f:
    _TOPIC_IMPORTANCE: dict[str, float] = json.load(f)


def _load_chapter_pyq_counts() -> dict[str, int]:
    if _CHAPTER_SUMMARY_PATH.exists():
        try:
            payload = json.loads(_CHAPTER_SUMMARY_PATH.read_text(encoding="utf-8"))
            raw_ch = payload.get("chapters", {})
            if isinstance(raw_ch, dict):
                return {
                    str(c_name): int(info.get("total_pyqs", 40))
                    for c_name, info in raw_ch.items()
                    if isinstance(info, dict)
                }
            elif isinstance(raw_ch, list):
                return {
                    str(row["chapter"]): int(row["total_pyqs"])
                    for row in raw_ch
                    if "chapter" in row and "total_pyqs" in row
                }
        except Exception:
            pass
    return {}


_CHAPTER_PYQ_COUNTS: dict[str, int] = _load_chapter_pyq_counts()


def compute_syllabus_coverage_and_improvement_plan(
    student_id: str,
    global_theta: float = 0.0,
    topic_states: list[dict] | None = None,
    attempt_records: list[dict] | None = None,
    top_k_actions: int = 12,
) -> dict:
    states = topic_states or []
    attempts = attempt_records or []

    practiced_set: set[str] = set()
    raw_mastery_map: dict[str, float] = {}
    decayed_mastery_map: dict[str, float] = {}
    theta_map: dict[str, float] = {}
    ci_map: dict[str, float] = {}
    level_map: dict[str, str] = {}
    pace_map: dict[str, str | None] = {}
    urgency_map: dict[str, float] = {}
    elapsed_map: dict[str, float] = {}

    for st in states:
        tid = str(st["topic_id"])
        practiced_set.add(tid)
        ch_prior = get_chapter_bkt_prior(tid)
        m_prob = float(st.get("mastery_prob") if st.get("mastery_prob") is not None else ch_prior)
        elapsed_h = float(st.get("elapsed_hours", 0.0) or 0.0)
        streak = int(st.get("streak", 0) or 0)
        decayed_m = compute_ebbinghaus_forgetting_prior(
            m_prob, elapsed_hours=elapsed_h, streak=streak, topic_id=tid
        )
        urg = compute_retention_urgency(
            m_prob, elapsed_hours=elapsed_h, streak=streak, topic_id=tid
        )

        raw_mastery_map[tid] = m_prob
        decayed_mastery_map[tid] = decayed_m
        theta_map[tid] = float(st.get("irt_theta", global_theta) or global_theta)
        ci_map[tid] = float(st.get("confidence_interval", 0.35) or 0.35)
        level_map[tid] = str(st.get("level") or "basic")
        pace_map[tid] = st.get("pace_flag")
        urgency_map[tid] = urg
        elapsed_map[tid] = elapsed_h

    effective_mastery_all: dict[str, float] = {}
    for tid in DAG_NODES:
        if tid in decayed_mastery_map:
            effective_mastery_all[tid] = decayed_mastery_map[tid]
        else:
            effective_mastery_all[tid] = compute_hierarchical_shrinkage_prior(tid, decayed_mastery_map)

    mc_sim = run_monte_carlo_conformal_neet_simulation(
        student_id=student_id,
        global_theta=global_theta,
        mastery_map=effective_mastery_all,
        theta_map=theta_map,
        ci_map=ci_map,
        topic_importance=_TOPIC_IMPORTANCE,
        n_trials=300,
    )
    leverage_lookup = {
        row["topic_id"]: row for row in mc_sim.get("high_yield_leverage_chapters", [])
    }

    dina_profile = compute_dina_and_rt_profile(attempts)
    weakest_skill = dina_profile["primary_cognitive_bottleneck"]

    all_chapters = sorted(DAG_NODES.keys())
    total_chapters = len(all_chapters)

    mastered_list: list[str] = []
    developing_list: list[str] = []
    less_learned_rows: list[dict] = []
    untouched_list: list[str] = []
    decay_alert_list: list[str] = []

    subject_buckets: dict[str, dict] = {
        s: {
            "total_chapters": 0,
            "practiced_chapters": 0,
            "mastered_chapters": 0,
            "developing_chapters": 0,
            "less_learned_chapters": 0,
            "untouched_chapters": 0,
            "mastery_values": [],
        }
        for s in ("Physics", "Chemistry", "Biology")
    }

    action_candidates: list[dict] = []

    for tid in all_chapters:
        meta = DAG_NODES[tid]
        subj = str(meta.get("subject", "Physics"))
        cluster = str(meta.get("cluster", "General"))
        imp = float(_TOPIC_IMPORTANCE.get(tid, 1.0))
        is_practiced = tid in practiced_set
        eff_m = effective_mastery_all[tid]
        peak_m = raw_mastery_map.get(tid, eff_m)
        urg = urgency_map.get(tid, 0.0)
        prereqs = get_direct_prerequisites(tid)
        weak_prereqs = [p for p in prereqs if effective_mastery_all.get(p, get_chapter_bkt_prior(p)) < 0.45]
        dependents = get_dependents(tid)
        readiness = compute_prerequisite_readiness(tid, effective_mastery_all)
        avail_pyqs = int(_CHAPTER_PYQ_COUNTS.get(tid, 15))

        sb = subject_buckets[subj]
        sb["total_chapters"] += 1
        sb["mastery_values"].append(eff_m)

        if not is_practiced:
            untouched_list.append(tid)
            sb["untouched_chapters"] += 1
            lesson_status = "UNTOUCHED"
        else:
            sb["practiced_chapters"] += 1
            if eff_m >= 0.75:
                mastered_list.append(tid)
                sb["mastered_chapters"] += 1
                lesson_status = "MASTERED"
            elif eff_m >= 0.45:
                developing_list.append(tid)
                sb["developing_chapters"] += 1
                lesson_status = "DEVELOPING"
            else:
                sb["less_learned_chapters"] += 1
                lesson_status = "LESS_LEARNED"
                less_learned_rows.append(
                    {
                        "topic_id": tid,
                        "subject": subj,
                        "cluster": cluster,
                        "level": level_map.get(tid, "basic"),
                        "pace_flag": pace_map.get(tid),
                        "peak_mastery_percent": round(peak_m * 100.0, 1),
                        "decayed_mastery_percent": round(eff_m * 100.0, 1),
                        "neet_weightage_percent": round(imp, 2),
                        "blocking_prerequisites": weak_prereqs,
                        "downstream_unlocked_count": len(dependents),
                        "deficit_severity": round((1.0 - eff_m) * imp, 3),
                    }
                )

            if urg >= 0.25:
                decay_alert_list.append(tid)

        lev_entry = leverage_lookup.get(tid)
        est_marks_gain = (
            float(lev_entry["total_expected_marks_gain"])
            if lev_entry
            else round((0.88 - min(0.88, eff_m)) * (imp / 100.0) * 720.0 * (1.0 + 0.06 * len(dependents)), 2)
        )

        if is_practiced and eff_m < 0.45 and weak_prereqs:
            diag_cat = "PREREQUISITE_BOTTLENECK"
            priority = (1.0 - eff_m) * imp * 2.4 + 0.6 * len(dependents)
            directive = (
                f"Remediate foundational prerequisite(s) {', '.join(weak_prereqs)} first, then re-drill {tid}."
            )
            drill_q = min(12, avail_pyqs)
        elif is_practiced and eff_m < 0.45:
            diag_cat = "LESS_LEARNED_CRITICAL"
            pace_note = " with untimed step-by-step derivation" if pace_map.get(tid) == "slow" else ""
            priority = (1.0 - eff_m) * imp * 2.1 + 0.45 * len(dependents)
            directive = (
                f"Drill {tid} ({level_map.get(tid, 'basic')} tier){pace_note}; focus on {weakest_skill['name']}."
            )
            drill_q = min(15 if pace_map.get(tid) == "slow" else 12, avail_pyqs)
        elif is_practiced and urg >= 0.25:
            diag_cat = "SPACED_REPETITION_DECAY"
            priority = urg * imp * 1.85 + (peak_m - eff_m) * 4.0
            directive = (
                f"Memory retention decayed from {round(peak_m * 100)}% to {round(eff_m * 100)}% over "
                f"{round(elapsed_map.get(tid, 0.0), 1)}h; run spaced-repetition review."
            )
            drill_q = min(10, avail_pyqs)
        elif eff_m < 0.75:
            diag_cat = "HIGH_YIELD_UNLOCK"
            priority = (1.0 - eff_m) * imp * readiness * (1.0 + 0.08 * len(dependents))
            directive = (
                f"High-yield ZPD chapter (readiness {round(readiness * 100)}%, unlocks {len(dependents)} downstream, "
                f"{avail_pyqs} verified PYQs); complete adaptive chapter drill."
            )
            drill_q = min(15, avail_pyqs)
        else:
            continue

        action_candidates.append(
            {
                "topic_id": tid,
                "subject": subj,
                "cluster": cluster,
                "lesson_status": lesson_status,
                "diagnostic_category": diag_cat,
                "current_mastery_percent": round(peak_m * 100.0, 1),
                "decayed_mastery_percent": round(eff_m * 100.0, 1),
                "readiness_percent": round(readiness * 100.0, 1),
                "neet_weightage_percent": round(imp, 2),
                "expected_marks_gain_720": round(max(0.5, est_marks_gain), 1),
                "blocking_prerequisites": weak_prereqs,
                "recommended_start_tier": (
                    level_map[tid]
                    if tid in level_map
                    else ("advanced" if eff_m >= 0.70 else ("medium" if eff_m >= 0.45 else "basic"))
                ),
                "recommended_drill_questions": max(3, drill_q),
                "available_chapter_pyqs": avail_pyqs,
                "remediation_priority_score": round(priority, 4),
                "what_to_improve": directive,
            }
        )

    less_learned_rows.sort(key=lambda r: r["deficit_severity"], reverse=True)
    action_candidates.sort(key=lambda r: r["remediation_priority_score"], reverse=True)
    ranked_actions = [
        {"rank": idx + 1, **row} for idx, row in enumerate(action_candidates[:top_k_actions])
    ]

    latest_by_qid: dict[str, dict] = {}
    for att in attempts:
        qid = str(att.get("question_id") or "")
        if qid:
            latest_by_qid[qid] = att

    mistake_retest_queue: list[dict] = []
    for qid, att in latest_by_qid.items():
        if bool(att.get("is_correct", False)):
            continue
        tid = str(att.get("topic_id") or "")
        subj = str((DAG_NODES.get(tid) or {}).get("subject", "Physics"))
        raw_t = att.get("time_taken_seconds")
        t_taken = float(raw_t) if raw_t is not None else 60.0
        raw_exp = att.get("expected_time_seconds")
        exp_t = float(raw_exp) if raw_exp is not None else 90.0
        m_cur = effective_mastery_all.get(tid, get_chapter_bkt_prior(tid))
        interval_h = 24 if m_cur < 0.45 else (72 if m_cur < 0.70 else 168)
        error_mode = (
            "impulsive_guess"
            if t_taken <= max(10.0, 0.25 * exp_t)
            else ("slow_struggle" if t_taken >= 1.35 * exp_t else "conceptual_or_trap_error")
        )
        mistake_retest_queue.append(
            {
                "question_id": qid,
                "topic_id": tid,
                "subject": subj,
                "difficulty": str(att.get("difficulty") or "medium"),
                "time_taken_seconds": round(t_taken, 2),
                "error_mode": error_mode,
                "ebbinghaus_retest_hours": interval_h,
                "chapter_mastery_percent": round(m_cur * 100.0, 1),
            }
        )
    mistake_retest_queue.sort(key=lambda r: (r["ebbinghaus_retest_hours"], r["chapter_mastery_percent"]))

    subject_coverage_out: dict[str, dict] = {}
    for subj, sb in subject_buckets.items():
        tot_c = max(1, sb["total_chapters"])
        subject_coverage_out[subj] = {
            "total_chapters": sb["total_chapters"],
            "practiced_chapters": sb["practiced_chapters"],
            "coverage_percent": round(100.0 * sb["practiced_chapters"] / tot_c, 1),
            "mastered_chapters": sb["mastered_chapters"],
            "developing_chapters": sb["developing_chapters"],
            "less_learned_chapters": sb["less_learned_chapters"],
            "untouched_chapters": sb["untouched_chapters"],
            "mean_mastery_percent": round(100.0 * sum(sb["mastery_values"]) / len(sb["mastery_values"]), 1),
        }

    return {
        "student_id": student_id,
        "syllabus_coverage": {
            "total_chapters": total_chapters,
            "practiced_chapters": len(practiced_set),
            "untouched_chapters": len(untouched_list),
            "coverage_percent": round(100.0 * len(practiced_set) / total_chapters, 1),
            "mastered_chapters": len(mastered_list),
            "developing_chapters": len(developing_list),
            "less_learned_chapters": len(less_learned_rows),
            "retention_decay_alerts": len(decay_alert_list),
            "subject_coverage": subject_coverage_out,
        },
        "primary_cognitive_skill_to_improve": weakest_skill,
        "projected_neet_score_interval_90": mc_sim["conformal_90_interval"],
        "optimal_skip_policy": mc_sim["negative_marking_optimizer"],
        "less_learned_lessons": less_learned_rows,
        "improvement_action_table": ranked_actions,
        "mistake_retest_queue": mistake_retest_queue[:15],
    }


def format_study_plan_markdown_table(plan: dict) -> str:
    cov = plan["syllabus_coverage"]
    lines = [
        f"### Syllabus Coverage Summary ({cov['practiced_chapters']} / {cov['total_chapters']} Chapters = {cov['coverage_percent']}%)",
        "",
        "| Subject | Practiced / Total | Coverage % | Mastered (>=75%) | Developing (45-74%) | Less-Learned (<45%) | Mean Mastery |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]
    for subj, sc in cov["subject_coverage"].items():
        lines.append(
            f"| **{subj}** | {sc['practiced_chapters']} / {sc['total_chapters']} | {sc['coverage_percent']}% | "
            f"{sc['mastered_chapters']} | {sc['developing_chapters']} | {sc['less_learned_chapters']} | "
            f"{sc['mean_mastery_percent']}% |"
        )
    lines.extend(
        [
            "",
            "### Top Ranked Improvement & Remediation Plan",
            "",
            "| Rank | Chapter | Subject | Diagnosis | Mastery | +NEET Marks | Drill Qs | Exact Improvement Action |",
            "| :---: | :--- | :--- | :--- | :---: | :---: | :---: | :--- |",
        ]
    )
    for row in plan["improvement_action_table"]:
        lines.append(
            f"| {row['rank']} | **{row['topic_id']}** | {row['subject']} | `{row['diagnostic_category']}` | "
            f"{row['decayed_mastery_percent']}% | +{row['expected_marks_gain_720']} | "
            f"{row['recommended_drill_questions']} ({row['recommended_start_tier']}) | {row['what_to_improve']} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    sample_plan = compute_syllabus_coverage_and_improvement_plan(
        student_id="DEMO_DIAGNOSTIC",
        global_theta=-0.15,
        topic_states=[
            {"topic_id": "Rotational Motion", "level": "basic", "mastery_prob": 0.22, "irt_theta": -0.65, "pace_flag": "slow", "elapsed_hours": 14.0, "streak": 0},
            {"topic_id": "Laws of Motion", "level": "basic", "mastery_prob": 0.28, "irt_theta": -0.45, "pace_flag": None, "elapsed_hours": 30.0, "streak": 0},
            {"topic_id": "Electrostatics", "level": "advanced", "mastery_prob": 0.88, "irt_theta": 1.25, "pace_flag": "fast", "elapsed_hours": 160.0, "streak": 2},
            {"topic_id": "Chemical Kinetics", "level": "medium", "mastery_prob": 0.58, "irt_theta": 0.20, "pace_flag": None, "elapsed_hours": 6.0, "streak": 1},
        ],
    )
    print(format_study_plan_markdown_table(sample_plan))
