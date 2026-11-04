import os
import shutil
import tempfile
from pathlib import Path
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from .db_models import Base
from .paths import PROJECT_ROOT

# Anchor SQLite DB path to PROJECT_ROOT/backend/klassed.db so launching uvicorn
# from either the repo root or backend/ always resolves to the exact same database.
# In read-only serverless environments (e.g. Vercel), copy the pre-seeded SQLite DB to /tmp.
_REPO_DB_PATH = PROJECT_ROOT / "backend" / "klassed.db"


def _resolve_sqlite_path() -> str:
    if os.environ.get("VERCEL") or not os.access(_REPO_DB_PATH.parent, os.W_OK):
        tmp_db = Path(tempfile.gettempdir()) / "klassed.db"
        if not tmp_db.exists() and _REPO_DB_PATH.exists():
            shutil.copy2(_REPO_DB_PATH, tmp_db)
        return tmp_db.as_posix()
    return _REPO_DB_PATH.as_posix()


_DEFAULT_DB_PATH = _resolve_sqlite_path()
DATABASE_URL = os.environ.get("KLASSED_DB_URL", f"sqlite:///{_DEFAULT_DB_PATH}")

_IS_SQLITE = DATABASE_URL.startswith("sqlite")
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False, "timeout": 30} if _IS_SQLITE else {},
    pool_pre_ping=True,
)

if _IS_SQLITE:
    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=5000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Call this once on startup to create all tables, v2 columns, and composite high-concurrency indexes."""
    Base.metadata.create_all(bind=engine)
    _migrate_v2_columns()
    _create_composite_indexes()


def _migrate_v2_columns():
    """Ensure existing SQLite databases gain v2 columns without data loss."""
    migrations = [
        ("students", "global_theta", "FLOAT NOT NULL DEFAULT 0.0"),
        ("student_topic_state", "mastery_prob", "FLOAT NOT NULL DEFAULT 0.35"),
        ("student_topic_state", "confidence_interval", "FLOAT NOT NULL DEFAULT 0.45"),
        ("student_topic_state", "irt_theta", "FLOAT NOT NULL DEFAULT 0.0"),
        ("test_attempts", "option_switches", "INTEGER NOT NULL DEFAULT 0"),
        ("test_attempts", "hesitation_ms", "INTEGER NOT NULL DEFAULT 0"),
        ("test_attempts", "dwell_time_ms", "INTEGER NOT NULL DEFAULT 0"),
    ]
    with engine.begin() as conn:
        for table_name, col_name, col_def in migrations:
            try:
                rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                existing_cols = {r[1] for r in rows}
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"))
            except Exception:
                pass


def _create_composite_indexes():
    """Creates composite B-tree indexes for sub-millisecond adaptive paper generation and analytics."""
    index_statements = [
        "CREATE INDEX IF NOT EXISTS idx_questions_topic_diff ON questions (topic_id, difficulty)",
        "CREATE INDEX IF NOT EXISTS idx_attempts_student_paper ON test_attempts (student_id, paper_number)",
        "CREATE INDEX IF NOT EXISTS idx_attempts_student_ts ON test_attempts (student_id, timestamp DESC)",
        "CREATE INDEX IF NOT EXISTS idx_state_student_topic ON student_topic_state (student_id, topic_id)",
    ]
    with engine.begin() as conn:
        for stmt in index_statements:
            try:
                conn.execute(text(stmt))
            except Exception:
                pass


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure schema, v2 column migrations, and composite indexes are applied immediately on import
init_db()
