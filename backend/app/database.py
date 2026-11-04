import os
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from .db_models import Base
from .paths import PROJECT_ROOT

# Anchor SQLite DB path to PROJECT_ROOT/backend/klassed.db so launching uvicorn
# from either the repo root or backend/ always resolves to the exact same database.
_DEFAULT_DB_PATH = (PROJECT_ROOT / "backend" / "klassed.db").as_posix()
DATABASE_URL = os.environ.get("KLASSED_DB_URL", f"sqlite:///{_DEFAULT_DB_PATH}")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Call this once on startup to create all tables and apply non-destructive v2 column migrations."""
    Base.metadata.create_all(bind=engine)
    _migrate_v2_columns()


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


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Ensure schema and v2 non-destructive column migrations are applied immediately on import
init_db()
