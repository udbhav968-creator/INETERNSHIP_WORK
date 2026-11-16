from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
RELEASES_PATH = MODEL_DIR / "releases.json"
CORE_REGISTRY = MODEL_DIR / "model_registry_manifest.json"
EXTENSION_REGISTRY = MODEL_DIR / "ml_extension_registry.json"
PIPELINE_RUN = MODEL_DIR / "ml_pipeline_run.json"
STAGES = ("candidate", "staging", "production", "archived")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _registry_paths() -> list[str]:
    paths: set[str] = set()
    for reg in (CORE_REGISTRY, EXTENSION_REGISTRY):
        if not reg.exists():
            continue
        data = json.loads(reg.read_text(encoding="utf-8"))
        arts = data.get("artifacts") or data.get("files") or {}
        items = arts.items() if isinstance(arts, dict) else [((a.get("path") or a.get("file")), a) for a in arts]
        for key, meta in items:
            p = meta.get("path", key) if isinstance(meta, dict) else key
            if not p:
                continue
            rel = p if (PROJECT_ROOT / p).exists() else f"model/{Path(p).name}"
            if (PROJECT_ROOT / rel).exists():
                paths.add(rel)
    return sorted(paths)


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else None
    except Exception:
        return None


def load_releases(path: Path = RELEASES_PATH) -> dict:
    if not path.exists():
        return {"schema": 1, "releases": []}
    return json.loads(path.read_text(encoding="utf-8"))


def save_releases(data: dict, path: Path = RELEASES_PATH) -> None:
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def _find(data: dict, version: str) -> dict:
    for r in data["releases"]:
        if r["version"] == version:
            return r
    raise KeyError(f"release {version} not found")


def current_release(data: dict, stage: str = "production") -> dict | None:
    rel = [r for r in data["releases"] if r["stage"] == stage]
    return rel[-1] if rel else None


def create_release(notes: str = "", data: dict | None = None) -> dict:
    data = data if data is not None else load_releases()
    artifacts = {p: _sha256(PROJECT_ROOT / p) for p in _registry_paths()}
    digest = hashlib.sha256(json.dumps(artifacts, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    metrics = json.loads(PIPELINE_RUN.read_text(encoding="utf-8")).get("metrics", {}) if PIPELINE_RUN.exists() else {}
    for r in data["releases"]:
        if r["artifact_digest"] == digest:
            return r
    rel = {
        "version": f"r{len(data['releases']) + 1:04d}-{digest[:7]}",
        "artifact_digest": digest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": _git("rev-parse", "HEAD"),
        "stage": "candidate",
        "canary_percent": 0,
        "artifacts": artifacts,
        "metrics": metrics,
        "notes": notes,
        "history": [],
    }
    data["releases"].append(rel)
    return rel


def promote(data: dict, version: str, stage: str, canary_percent: int = 0, reviewer: str = "") -> dict:
    if stage not in STAGES:
        raise ValueError(f"stage must be one of {STAGES}")
    rel = _find(data, version)
    if stage == "production":
        if rel["stage"] != "staging":
            raise ValueError("only a staging release can be promoted to production")
        if not reviewer:
            raise ValueError("production promotion needs a named reviewer")
        for other in data["releases"]:
            if other is not rel and other["stage"] == "production":
                other["stage"] = "archived"
                other["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": f"superseded by {version}"})
    rel["stage"] = stage
    rel["canary_percent"] = int(max(0, min(100, canary_percent))) if stage == "staging" else (100 if stage == "production" else 0)
    rel["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": f"-> {stage}", "reviewer": reviewer, "canary_percent": rel["canary_percent"]})
    return rel


def rollback(data: dict, reviewer: str) -> tuple[dict, dict]:
    prod = current_release(data, "production")
    archived = [r for r in data["releases"] if r["stage"] == "archived" and r is not prod]
    if prod is None or not archived:
        raise ValueError("nothing to roll back to")
    target = archived[-1]
    prod["stage"] = "archived"
    prod["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": f"rolled back to {target['version']}", "reviewer": reviewer})
    target["stage"] = "production"
    target["canary_percent"] = 100
    target["history"].append({"at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "event": "restored to production by rollback", "reviewer": reviewer})
    return prod, target


def restore_command(rel: dict) -> list[str]:
    if not rel.get("git_commit"):
        return []
    return ["git", "checkout", str(rel["git_commit"]), "--", *sorted(rel["artifacts"])]


def verify_against(rel: dict) -> dict:
    mismatched = [p for p, h in rel["artifacts"].items() if not (PROJECT_ROOT / p).exists() or _sha256(PROJECT_ROOT / p) != h]
    return {"version": rel["version"], "artifacts": len(rel["artifacts"]), "mismatched": mismatched, "matches": not mismatched}


def serves_candidate(student_id: str, canary_percent: int, salt: str = "klassed-canary") -> bool:
    digest = hashlib.sha256(f"{salt}:{student_id}".encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % 100) < int(canary_percent)


def main() -> int:
    ap = argparse.ArgumentParser(description="Model releases: create, promote, roll back, verify")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create")
    c.add_argument("--notes", default="")
    p = sub.add_parser("promote")
    p.add_argument("version")
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--canary", type=int, default=0)
    p.add_argument("--reviewer", default="")
    r = sub.add_parser("rollback")
    r.add_argument("--reviewer", required=True)
    r.add_argument("--apply", action="store_true")
    sub.add_parser("list")
    sub.add_parser("verify")
    args = ap.parse_args()
    data = load_releases()
    if args.cmd == "create":
        rel = create_release(args.notes, data)
        save_releases(data)
        print(json.dumps({k: rel[k] for k in ("version", "stage", "git_commit")}, indent=2))
    elif args.cmd == "promote":
        rel = promote(data, args.version, args.stage, args.canary, args.reviewer)
        save_releases(data)
        print(json.dumps({k: rel[k] for k in ("version", "stage", "canary_percent")}, indent=2))
    elif args.cmd == "rollback":
        old, new = rollback(data, args.reviewer)
        save_releases(data)
        cmd = restore_command(new)
        print(json.dumps({"rolled_back": old["version"], "now_production": new["version"], "restore": " ".join(cmd)}, indent=2))
        if args.apply and cmd:
            subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    elif args.cmd == "list":
        for rel in data["releases"]:
            print(f"{rel['version']:22s} {rel['stage']:10s} canary={rel['canary_percent']:3d}% commit={str(rel.get('git_commit'))[:7]} {rel['created_at_utc']}")
    elif args.cmd == "verify":
        prod = current_release(data, "production")
        if prod is None:
            print("no production release recorded")
            return 1
        rep = verify_against(prod)
        print(json.dumps(rep, indent=2))
        return 0 if rep["matches"] else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
