import json
import math
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict, Any, List, Optional

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
QUESTIONS_PATH = PROJECT_ROOT / "data" / "tagged_questions" / "neet_questions_clean.json"


def _compute_real_corpus_baselines() -> tuple[dict[str, float], list[float]]:
    diff_counts = {"basic": 0, "medium": 0, "advanced": 0}
    time_counts = [0, 0, 0]
    if QUESTIONS_PATH.exists():
        with open(QUESTIONS_PATH, encoding="utf-8") as f:
            questions = json.load(f)
        total = max(1, len(questions))
        for q in questions:
            d = str(q.get("difficulty_level", "medium")).lower()
            if d in ("basic", "easy", "base"):
                diff_counts["basic"] += 1
            elif d in ("advanced", "hard"):
                diff_counts["advanced"] += 1
            else:
                diff_counts["medium"] += 1

            t = int(q.get("ideal_time_sec") or 85)
            if t < 75:
                time_counts[0] += 1
            elif t <= 95:
                time_counts[1] += 1
            else:
                time_counts[2] += 1
        return (
            {k: round(v / total, 4) for k, v in diff_counts.items()},
            [round(c / total, 4) for c in time_counts],
        )
    return ({"basic": 0.3448, "medium": 0.4974, "advanced": 0.1578}, [0.30, 0.50, 0.20])


class InMemoryFeatureStore:
    def __init__(self, max_latency_samples: int = 1000):
        self._lock = threading.RLock()
        self._student_cache: Dict[str, Dict[str, Any]] = {}
        self._latencies_ms: deque = deque(maxlen=max_latency_samples)
        self._hits: int = 0
        self._misses: int = 0

    def get_student_features(self, student_id: str) -> Optional[Dict[str, Any]]:
        t0 = time.perf_counter()
        with self._lock:
            record = self._student_cache.get(student_id)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._latencies_ms.append(elapsed_ms)
            if record is not None:
                self._hits += 1
            else:
                self._misses += 1
            return dict(record) if record is not None else None

    def put_student_features(self, student_id: str, features: Dict[str, Any]) -> float:
        t0 = time.perf_counter()
        with self._lock:
            self._student_cache[student_id] = {
                **features,
                "updated_at_epoch": time.time(),
            }
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self._latencies_ms.append(elapsed_ms)
            return elapsed_ms

    def evict_student(self, student_id: str) -> None:
        with self._lock:
            self._student_cache.pop(student_id, None)

    def get_latency_percentiles(self) -> Dict[str, float]:
        with self._lock:
            if not self._latencies_ms:
                t0 = time.perf_counter()
                _ = self._student_cache.get("__probe__")
                probe_ms = (time.perf_counter() - t0) * 1000.0
                self._latencies_ms.append(probe_ms)

            sorted_lat = sorted(self._latencies_ms)
            n = len(sorted_lat)
            p50 = sorted_lat[min(n - 1, int(0.50 * n))]
            p95 = sorted_lat[min(n - 1, int(0.95 * n))]
            p99 = sorted_lat[min(n - 1, int(0.99 * n))]
            total_queries = self._hits + self._misses
            hit_rate = (self._hits / total_queries) if total_queries > 0 else 1.0
            cached_cnt = len(self._student_cache)
        return {
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95, 4),
            "p99_ms": round(p99, 4),
            "cache_hit_rate": round(hit_rate, 4),
            "cached_students": cached_cnt,
        }


class DriftAndCalibrationMonitor:
    def __init__(self):
        self._lock = threading.RLock()
        self.baseline_difficulty_dist, self.baseline_time_buckets = _compute_real_corpus_baselines()
        self.predictions_log: deque = deque(maxlen=5000)

    def log_prediction(self, predicted_prob: float, actual_correct: int, difficulty: str, time_taken: int):
        with self._lock:
            self.predictions_log.append(
                {
                    "prob": max(0.001, min(0.999, float(predicted_prob))),
                    "correct": 1 if actual_correct else 0,
                    "difficulty": (difficulty or "medium").lower(),
                    "time_taken": int(time_taken),
                }
            )

    @staticmethod
    def compute_psi(expected_props: List[float], actual_props: List[float]) -> float:
        psi_val = 0.0
        for exp_p, act_p in zip(expected_props, actual_props):
            e = max(1e-4, float(exp_p))
            a = max(1e-4, float(act_p))
            psi_val += (a - e) * math.log(a / e)
        return round(psi_val, 5)

    def evaluate_health(self, db_attempts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        with self._lock:
            records = list(self.predictions_log)
        if db_attempts:
            for att in db_attempts:
                records.append(
                    {
                        "prob": max(0.001, min(0.999, float(att["predicted_prob"]))),
                        "correct": 1 if att.get("correct") else 0,
                        "difficulty": str(att.get("difficulty", "medium")).lower(),
                        "time_taken": int(att.get("time_taken", 60)),
                    }
                )

        if not records:
            return {
                "psi_difficulty": 0.0,
                "psi_response_time": 0.0,
                "effective_psi": 0.0,
                "brier_score": 0.0,
                "expected_calibration_error": 0.0,
                "drift_status": "NO_ATTEMPTS_YET",
                "recalibration_required": False,
                "sample_size": 0,
            }

        n = len(records)
        diff_counts = {"basic": 0, "medium": 0, "advanced": 0}
        time_counts = [0, 0, 0]
        brier_sum = 0.0

        bins = [{"count": 0, "prob_sum": 0.0, "acc_sum": 0.0} for _ in range(5)]

        for r in records:
            d = r["difficulty"]
            if d in ("easy", "basic"):
                diff_counts["basic"] += 1
            elif d in ("hard", "advanced"):
                diff_counts["advanced"] += 1
            else:
                diff_counts["medium"] += 1

            t = r["time_taken"]
            if t < 75:
                time_counts[0] += 1
            elif t <= 95:
                time_counts[1] += 1
            else:
                time_counts[2] += 1

            p = r["prob"]
            y = r["correct"]
            brier_sum += (p - y) ** 2

            bin_idx = min(4, int(p * 5))
            bins[bin_idx]["count"] += 1
            bins[bin_idx]["prob_sum"] += p
            bins[bin_idx]["acc_sum"] += y

        actual_diff_props = [
            diff_counts["basic"] / n,
            diff_counts["medium"] / n,
            diff_counts["advanced"] / n,
        ]
        expected_diff_props = [
            self.baseline_difficulty_dist["basic"],
            self.baseline_difficulty_dist["medium"],
            self.baseline_difficulty_dist["advanced"],
        ]
        psi_diff = self.compute_psi(expected_diff_props, actual_diff_props)

        actual_time_props = [c / n for c in time_counts]
        psi_time = self.compute_psi(self.baseline_time_buckets, actual_time_props)

        brier_score = round(brier_sum / n, 4)

        ece = 0.0
        for b in bins:
            if b["count"] > 0:
                avg_p = b["prob_sum"] / b["count"]
                avg_acc = b["acc_sum"] / b["count"]
                ece += (b["count"] / n) * abs(avg_p - avg_acc)
        ece = round(ece, 4)

        effective_psi = psi_diff if n >= 30 else round(psi_diff * (n / 30.0), 4)
        recal_needed = effective_psi > 0.20 or (n >= 30 and ece > 0.08)

        return {
            "psi_difficulty": psi_diff,
            "psi_response_time": psi_time,
            "effective_psi": effective_psi,
            "brier_score": brier_score,
            "expected_calibration_error": ece,
            "drift_status": "DRIFT_WARNING" if recal_needed else "HEALTHY",
            "recalibration_required": recal_needed,
            "sample_size": n,
        }


FEATURE_STORE = InMemoryFeatureStore()
DRIFT_MONITOR = DriftAndCalibrationMonitor()
