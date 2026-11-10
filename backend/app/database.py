import os
import shutil
import tempfile
from pathlib import Path
from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker
from .db_models import Base
from .paths import PROJECT_ROOT

_REPO_DB_PATH = PROJECT_ROOT / "backend" / "klassed.db"


def _resolve_sqlite_path() -> str:
    if os.environ.get("VERCEL") or not os.access(_REPO_DB_PATH.parent, os.W_OK):
        tmp_db = Path(tempfile.gettempdir()) / "klassed.db"
        if not tmp_db.exists() and _REPO_DB_PATH.exists():
            shutil.copy2(_REPO_DB_PATH, tmp_db)
        return tmp_db.as_posix()
    return _REPO_DB_PATH.as_posix()


_DEFAULT_DB_PATH = _resolve_sqlite_path()
DATABASE_URL = os.environ.get("DATABASE_URL") or os.environ.get("KLASSED_DB_URL") or f"sqlite:///{_DEFAULT_DB_PATH}"
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

_IS_SQLITE = DATABASE_URL.startswith("sqlite")
engine_kwargs = {"pool_pre_ping": True}
if _IS_SQLITE:
    engine_kwargs["connect_args"] = {"check_same_thread": False, "timeout": 30}
else:
    engine_kwargs["pool_size"] = int(os.environ.get("DB_POOL_SIZE", 20))
    engine_kwargs["max_overflow"] = int(os.environ.get("DB_MAX_OVERFLOW", 10))
    engine_kwargs["pool_recycle"] = 1800

engine = create_engine(DATABASE_URL, **engine_kwargs)

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
    Base.metadata.create_all(bind=engine)
    _migrate_v2_columns()
    _create_composite_indexes()


def _migrate_v2_columns():
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
        is_sqlite = engine.dialect.name == "sqlite"
        for table_name, col_name, col_def in migrations:
            try:
                if is_sqlite:
                    rows = conn.execute(text(f"PRAGMA table_info({table_name})")).fetchall()
                    existing_cols = {r[1] for r in rows}
                else:
                    rows = conn.execute(
                        text("SELECT column_name FROM information_schema.columns WHERE table_name = :t"),
                        {"t": table_name},
                    ).fetchall()
                    existing_cols = {r[0] for r in rows}
                if col_name not in existing_cols:
                    conn.execute(text(f"ALTER TABLE {table_name} ADD COLUMN {col_name} {col_def}"))
            except Exception:
                pass


def _create_composite_indexes():
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


init_db()
