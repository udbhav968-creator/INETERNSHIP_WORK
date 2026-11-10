from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, text
from .database import engine, _IS_SQLITE, _DEFAULT_DB_PATH

TABLE_ORDER = [
    "topics",
    "questions",
    "students",
    "student_topic_state",
    "test_attempts",
    "papers",
    "fatigue_logs",
    "socratic_conversations",
    "ncert_alignment_manifest",
]


def get_database_telemetry() -> dict[str, Any]:
    t0 = time.perf_counter()
    dialect_name = engine.dialect.name
    pool_class = engine.pool.__class__.__name__
    is_connected = False
    error_msg = None

    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            is_connected = True
    except Exception as exc:
        error_msg = str(exc)

    latency_ms = round((time.perf_counter() - t0) * 1000.0, 3)

    return {
        "status": "HEALTHY" if is_connected else "DEGRADED",
        "dialect": dialect_name,
        "is_sqlite": _IS_SQLITE,
        "pool_class": pool_class,
        "pool_size": getattr(engine.pool, "size", lambda: 0)(),
        "checked_out_connections": getattr(engine.pool, "checkedout", lambda: 0)(),
        "overflow": getattr(engine.pool, "overflow", lambda: 0)(),
        "ping_latency_ms": latency_ms,
        "connection_error": error_msg,
    }


def check_postgres_connection(db_url: str | None = None) -> dict[str, Any]:
    target_url = db_url or os.environ.get("DATABASE_URL") or os.environ.get("KLASSED_DB_URL")
    if not target_url:
        return {
            "status": "SKIPPED",
            "detail": "No PostgreSQL connection URL provided or configured in environment",
            "configured": False,
        }

    if target_url.startswith("postgres://"):
        target_url = target_url.replace("postgres://", "postgresql://", 1)

    t0 = time.perf_counter()
    try:
        temp_engine = create_engine(target_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
        with temp_engine.connect() as conn:
            ver_row = conn.execute(text("SELECT version()")).fetchone()
            ver_str = str(ver_row[0]) if ver_row else "Unknown"
        temp_engine.dispose()
        elapsed = round((time.perf_counter() - t0) * 1000.0, 3)
        return {
            "status": "CONNECTED",
            "dialect": "postgresql",
            "server_version": ver_str,
            "latency_ms": elapsed,
            "configured": True,
        }
    except Exception as exc:
        return {
            "status": "CONNECTION_FAILED",
            "detail": str(exc),
            "configured": True,
        }


def migrate_sqlite_to_postgres(
    sqlite_path: Path | None = None,
    pg_url: str | None = None,
    batch_size: int = 500,
) -> dict[str, Any]:
    src_path = sqlite_path or Path(_DEFAULT_DB_PATH)
    if not src_path.exists():
        return {
            "status": "ERROR",
            "detail": f"Source SQLite database does not exist at {src_path}",
            "migrated_tables": {},
        }

    target_url = pg_url or os.environ.get("DATABASE_URL") or os.environ.get("KLASSED_DB_URL")
    if not target_url or target_url.startswith("sqlite"):
        return {
            "status": "ERROR",
            "detail": "Target PostgreSQL URL must be provided and start with postgresql://",
            "migrated_tables": {},
        }

    if target_url.startswith("postgres://"):
        target_url = target_url.replace("postgres://", "postgresql://", 1)

    t0 = time.perf_counter()
    pg_engine = create_engine(target_url, pool_pre_ping=True)
    counts: dict[str, int] = {}

    try:
        s_conn = sqlite3.connect(str(src_path))
        s_conn.row_factory = sqlite3.Row
        s_cur = s_conn.cursor()

        with pg_engine.begin() as p_conn:
            for tbl in TABLE_ORDER:
                s_cur.execute(f"SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name=?", (tbl,))
                if s_cur.fetchone()[0] == 0:
                    continue

                s_cur.execute(f"SELECT * FROM {tbl}")
                rows = s_cur.fetchall()
                if not rows:
                    counts[tbl] = 0
                    continue

                col_names = list(rows[0].keys())
                cols_sql = ", ".join(col_names)
                params_sql = ", ".join(f":{c}" for c in col_names)
                insert_stmt = text(f"INSERT INTO {tbl} ({cols_sql}) VALUES ({params_sql}) ON CONFLICT DO NOTHING")

                migrated_this_tbl = 0
                for i in range(0, len(rows), batch_size):
                    batch = [dict(r) for r in rows[i : i + batch_size]]
                    p_conn.execute(insert_stmt, batch)
                    migrated_this_tbl += len(batch)
                counts[tbl] = migrated_this_tbl

        s_conn.close()
    except Exception as exc:
        pg_engine.dispose()
        return {
            "status": "FAILED",
            "detail": str(exc),
            "migrated_tables": counts,
        }

    pg_engine.dispose()
    elapsed = round(time.perf_counter() - t0, 3)
    return {
        "status": "SUCCESS",
        "migrated_tables": counts,
        "total_records_migrated": sum(counts.values()),
        "elapsed_seconds": elapsed,
    }
