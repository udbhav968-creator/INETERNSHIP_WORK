from __future__ import annotations

import time
from collections import defaultdict
from typing import Any
import numpy as np

_LATENCY_WINDOWS: dict[str, list[float]] = defaultdict(list)
_MAX_WINDOW_SIZE = 500
_HARD_SLA_MS: dict[str, float] = {
    "irt_3pl_map": 10.0,
    "bkt_update": 10.0,
    "linucb_scoring": 10.0,
    "rf_inference": 50.0,
    "dina_diagnosis": 25.0,
    "rag_remediation": 50.0,
    "study_plan": 100.0,
}


class LatencyWatchdog:
    def __init__(self, subsystem: str) -> None:
        self.subsystem = subsystem
        self.start_t = 0.0

    def __enter__(self) -> LatencyWatchdog:
        self.start_t = time.perf_counter()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        elapsed_ms = (time.perf_counter() - self.start_t) * 1000.0
        buf = _LATENCY_WINDOWS[self.subsystem]
        buf.append(elapsed_ms)
        if len(buf) > _MAX_WINDOW_SIZE:
            buf.pop(0)


def get_subsystem_telemetry_metrics() -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    sla_breaches = 0

    for sub, vals in _LATENCY_WINDOWS.items():
        if not vals:
            continue
        p50 = float(np.percentile(vals, 50))
        p95 = float(np.percentile(vals, 95))
        p99 = float(np.percentile(vals, 99))
        limit = _HARD_SLA_MS.get(sub, 50.0)
        is_breach = p95 > limit
        if is_breach:
            sla_breaches += 1

        metrics[sub] = {
            "samples": len(vals),
            "p50_ms": round(p50, 4),
            "p95_ms": round(p95, 4),
            "p99_ms": round(p99, 4),
            "sla_limit_ms": limit,
            "sla_status": "BREACHED" if is_breach else "HEALTHY",
        }

    return {
        "subsystems_monitored": len(metrics),
        "total_sla_breaches": sla_breaches,
        "engine_health": "OPTIMAL" if sla_breaches == 0 else "DEGRADED",
        "telemetry_metrics": metrics,
    }
