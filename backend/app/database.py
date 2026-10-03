from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .db_models import Base

# SQLite for local dev — zero setup needed. Swap this string for a Postgres
# URL later (e.g. "postgresql://user:pass@host/dbname") without touching
# any model or logic code.
DATABASE_URL = "sqlite:///./klassed.db"

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Call this once on startup to create all tables."""
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
