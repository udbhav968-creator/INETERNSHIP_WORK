"""
Single source of truth for project paths. Import PROJECT_ROOT / DATA_DIR /
MODEL_DIR from here instead of using relative open() calls anywhere else —
those break the moment uvicorn is launched from a different working
directory (e.g. from the repo root vs. from inside backend/).
"""
from pathlib import Path

# backend/app/paths.py -> parents[2] is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "model"
