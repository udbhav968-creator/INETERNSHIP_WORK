from __future__ import annotations

import os


def behavioural_telemetry_enabled() -> bool:
    return os.environ.get("KLASSED_BEHAVIOURAL_TELEMETRY", "on").strip().lower() not in ("0", "off", "false", "no")


def minimise_behavioural_signals(option_switches: int, hesitation_ms: int, dwell_time_ms: int, time_taken_seconds: float) -> tuple[int, int, int]:
    if behavioural_telemetry_enabled():
        return int(option_switches), int(hesitation_ms), int(dwell_time_ms)
    return 0, 0, int(round(float(time_taken_seconds) * 1000.0))


def privacy_status() -> dict:
    return {
        "behavioural_telemetry": "on" if behavioural_telemetry_enabled() else "off",
        "controls": "KLASSED_BEHAVIOURAL_TELEMETRY=off stops storing answer switches and hesitation; time per question is still stored because scoring needs it",
        "pending_decision": "legal review under the DPDP Act and Rules 2025 (verifiable parental consent for under-18s; limits on behavioural monitoring of children)",
    }
