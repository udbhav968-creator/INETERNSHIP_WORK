import json
from pathlib import Path

from .verify_artifact_integrity import _artifact_canonical_size_and_sha256

MODEL_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODEL_DIR.parent
EXTENSION_REGISTRY_FILE = MODEL_DIR / "ml_extension_registry.json"

EXTENSION_ARTIFACTS = [
    "model/assessment_forms.json",
    "model/cold_start_irt_linking.json",
    "model/sample_paper_irt_parameters.json",
    "model/unified_near_duplicate_map.json",
    "model/question_difficulty_model_v2_candidate.joblib",
    "data/tagged_questions/neet_samples_001_to_101_verified.json",
    "model/bkt_chapter_parameters_unified.json",
    "model/dina_calibrated_parameters_unified.json",
    "model/ncert_rag_svd_index_unified.npz",
    "model/ncert_rag_svd_meta_unified.json",
    "model/linucb_prior_init_unified.json",
    "model/conformal_laplacian_model_unified.json",
]


def build_extension_registry(save: bool = True) -> dict:
    artifacts = {}
    for rel in EXTENSION_ARTIFACTS:
        p = PROJECT_ROOT / rel
        size, digest = _artifact_canonical_size_and_sha256(p) if p.exists() else (0, "")
        artifacts[rel] = {"exists": p.exists(), "size_bytes": size, "sha256": digest}
    manifest = {"registry": "ml-extension-v1", "artifacts": artifacts}
    if save:
        EXTENSION_REGISTRY_FILE.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def verify_extension_registry() -> dict:
    manifest = json.loads(EXTENSION_REGISTRY_FILE.read_text(encoding="utf-8"))
    verified = 0
    for rel, meta in manifest["artifacts"].items():
        p = PROJECT_ROOT / rel
        if not p.exists():
            raise RuntimeError(f"Extension artifact missing: {rel}")
        size, digest = _artifact_canonical_size_and_sha256(p)
        if size != int(meta["size_bytes"]) or digest != meta["sha256"]:
            raise RuntimeError(f"Extension artifact checksum mismatch: {rel}")
        verified += 1
    return {"extension_artifacts_verified": verified, "status": "VERIFIED"}


if __name__ == "__main__":
    build_extension_registry()
    print(verify_extension_registry())
