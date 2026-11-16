from __future__ import annotations

import importlib
import logging
import platform
import re
from pathlib import Path

LOGGER = logging.getLogger("klassed.ml.parity")

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
CONSTRAINTS_FILE = MODEL_DIR / "ml-training-constraints.txt"
PYTHON_VERSION_FILE = PROJECT_ROOT / ".python-version"

_IMPORT_NAMES = {"scikit-learn": "sklearn", "numpy": "numpy", "pandas": "pandas", "scipy": "scipy", "joblib": "joblib"}

_WARNED = False


def training_library_pins(path: Path = CONSTRAINTS_FILE) -> dict[str, str]:
    pins: dict[str, str] = {}
    if not path.exists():
        return pins
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        m = re.fullmatch(r"([A-Za-z0-9_.\-]+)\s*==\s*([A-Za-z0-9_.\-+]+)", line)
        if m:
            pins[m.group(1).lower()] = m.group(2)
    return pins


def training_python_version(path: Path = PYTHON_VERSION_FILE) -> str | None:
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return value or None


def _installed_version(dist: str) -> str | None:
    module = _IMPORT_NAMES.get(dist, dist.replace("-", "_"))
    try:
        return str(importlib.import_module(module).__version__)
    except Exception:
        return None


def check_runtime_parity(log_warning: bool = True) -> dict:
    global _WARNED
    pins = training_library_pins()
    libraries = {}
    for dist, trained in sorted(pins.items()):
        serving = _installed_version(dist)
        libraries[dist] = {"trained_with": trained, "serving_with": serving, "match": serving == trained}

    py_train = training_python_version()
    py_serve = platform.python_version()
    python_match = None if py_train is None else (py_serve == py_train or py_serve.startswith(py_train + "."))

    mismatched = sorted(d for d, v in libraries.items() if not v["match"])
    libraries_match = bool(pins) and not mismatched
    report = {
        "source_of_truth": "model/ml-training-constraints.txt and .python-version",
        "python": {"trained_with": py_train, "serving_with": py_serve, "match": python_match},
        "libraries": libraries,
        "mismatched_libraries": mismatched,
        "libraries_match": libraries_match,
        "all_match": libraries_match and python_match is not False,
    }
    if log_warning and not report["all_match"] and not _WARNED:
        _WARNED = True
        LOGGER.warning(
            "Serving environment differs from training pins (python %s vs %s; mismatched: %s). "
            "Install with: pip install -r backend/requirements.txt -c model/ml-training-constraints.txt",
            py_serve,
            py_train,
            ", ".join(mismatched) or "none",
        )
    return report


def requirements_pins(path: Path) -> dict[str, str]:
    return training_library_pins(path)


if __name__ == "__main__":
    import json
    import sys

    strict = "--strict" in sys.argv[1:]
    rep = check_runtime_parity(log_warning=False)
    print(json.dumps(rep, indent=2))
    if not rep["libraries_match"]:
        sys.exit(1)
    if rep["python"]["match"] is False:
        print(f"Python {rep['python']['serving_with']} differs from .python-version {rep['python']['trained_with']}", file=sys.stderr)
        sys.exit(1 if strict else 0)
    sys.exit(0)
