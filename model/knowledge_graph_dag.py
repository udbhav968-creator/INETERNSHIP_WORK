import json
from collections import deque
from pathlib import Path

from .state_machine.bkt_model import get_chapter_bkt_prior

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
_DAG_PATH = PROJECT_ROOT / "data" / "taxonomy" / "prerequisite_dag.json"

with open(_DAG_PATH, encoding="utf-8") as f:
    _RAW_DAG = json.load(f)

DAG_NODES: dict[str, dict] = _RAW_DAG.get("nodes", {})

_DEPENDENTS: dict[str, list[str]] = {topic: [] for topic in DAG_NODES}
_CLUSTERS: dict[str, list[str]] = {}

for topic, meta in DAG_NODES.items():
    cluster = meta.get("cluster", "General")
    _CLUSTERS.setdefault(cluster, []).append(topic)
    for prereq in meta.get("prerequisites", []):
        if prereq in _DEPENDENTS:
            _DEPENDENTS[prereq].append(topic)


def get_direct_prerequisites(topic_id: str) -> list[str]:
    node = DAG_NODES.get(topic_id)
    if not node:
        return []
    return list(node.get("prerequisites", []))


def get_all_ancestors(topic_id: str) -> list[str]:
    visited: list[str] = []
    queue = deque(get_direct_prerequisites(topic_id))
    seen = set(queue)
    while queue:
        curr = queue.popleft()
        visited.append(curr)
        for parent in get_direct_prerequisites(curr):
            if parent not in seen:
                seen.add(parent)
                queue.append(parent)
    return visited


def get_dependents(topic_id: str) -> list[str]:
    return list(_DEPENDENTS.get(topic_id, []))


def compute_prerequisite_readiness(
    topic_id: str, student_mastery_map: dict[str, float], default_prior: float | None = None
) -> float:
    prereqs = get_direct_prerequisites(topic_id)
    if not prereqs:
        return 1.0
    scores = [
        student_mastery_map.get(p, default_prior if default_prior is not None else get_chapter_bkt_prior(p))
        for p in prereqs
    ]
    return round(sum(scores) / len(scores), 4)


def compute_hierarchical_shrinkage_prior(
    topic_id: str,
    student_mastery_map: dict[str, float],
    attempts_count: int = 0,
    global_prior: float | None = None,
    shrinkage_strength: float = 2.0,
) -> float:
    base_prior = global_prior if global_prior is not None else get_chapter_bkt_prior(topic_id)
    node = DAG_NODES.get(topic_id, {})
    prereqs = node.get("prerequisites", [])
    cluster = node.get("cluster")
    siblings = _CLUSTERS.get(cluster, []) if cluster else []

    related_scores = [
        student_mastery_map[t]
        for t in set(prereqs + siblings)
        if t in student_mastery_map and t != topic_id
    ]
    m = len(related_scores)
    if m == 0:
        cluster_prior = base_prior
    else:
        sample_mean = sum(related_scores) / m
        cluster_prior = (m * sample_mean + shrinkage_strength * base_prior) / (m + shrinkage_strength)

    if topic_id not in student_mastery_map:
        return round(max(0.05, min(0.95, cluster_prior)), 4)

    observed = float(student_mastery_map[topic_id])
    n_obs = max(1, int(attempts_count)) if attempts_count > 0 else 1
    shrunk = (n_obs * observed + shrinkage_strength * cluster_prior) / (n_obs + shrinkage_strength)
    return round(max(0.05, min(0.95, shrunk)), 4)


def diagnose_prerequisite_bottlenecks(
    student_mastery_map: dict[str, float], threshold: float = 0.45
) -> list[dict]:
    bottlenecks: list[dict] = []

    for topic_id, mastery in student_mastery_map.items():
        prereqs = get_direct_prerequisites(topic_id)
        weak_prereqs = [
            p for p in prereqs if student_mastery_map.get(p, get_chapter_bkt_prior(p)) < threshold
        ]
        dependents = get_dependents(topic_id)

        if mastery < threshold and (weak_prereqs or dependents):
            bottlenecks.append({
                "topic_id": topic_id,
                "subject": DAG_NODES.get(topic_id, {}).get("subject", "Unknown"),
                "cluster": DAG_NODES.get(topic_id, {}).get("cluster", "General"),
                "current_mastery": round(mastery, 3),
                "weak_prerequisites": weak_prereqs,
                "blocked_downstream_topics": dependents,
                "recommendation": (
                    f"Remediate prerequisite(s) {', '.join(weak_prereqs)} first before drilling {topic_id}."
                    if weak_prereqs
                    else f"Foundational bottleneck: strengthening {topic_id} unlocks {len(dependents)} downstream chapter(s)."
                ),
            })

    bottlenecks.sort(
        key=lambda b: (len(b["blocked_downstream_topics"]), 1.0 - b["current_mastery"]),
        reverse=True,
    )
    return bottlenecks


def propagate_dag_belief_update(
    topic_id: str,
    is_correct: bool,
    difficulty: str,
    student_mastery_map: dict[str, float],
) -> dict[str, float]:
    updates: dict[str, float] = {}
    diff = (difficulty or "medium").lower()

    if is_correct and diff in ("medium", "advanced"):
        boost_rate = 0.08 if diff == "advanced" else 0.05
        for prereq in get_direct_prerequisites(topic_id):
            curr = float(student_mastery_map.get(prereq, get_chapter_bkt_prior(prereq)))
            if curr < 0.88:
                new_m = round(min(0.92, curr + boost_rate * (1.0 - curr)), 4)
                updates[prereq] = new_m

    elif not is_correct and diff == "basic":
        for dep in get_dependents(topic_id):
            if dep in student_mastery_map:
                curr = float(student_mastery_map[dep])
                new_m = round(max(0.12, curr * 0.95), 4)
                updates[dep] = new_m

    return updates
