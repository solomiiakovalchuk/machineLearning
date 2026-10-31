"""Build practical07/fixed/ from the frozen lab06 handoff WITHOUT touching the model.

Copies pipeline.joblib byte for byte, records its SHA-256, derives data_schema.json v2
(named features, ranges, domain check, policies) from the lab06 schema, writes minimal requirements.
"""
import json
import shutil
from pathlib import Path

from audit_core import sha256_file

ROOT = Path(__file__).resolve().parent
FIXED = ROOT / "fixed"

MIN_REQUIREMENTS = """# Inference-only requirements (the model is a plain sklearn Pipeline).
# Versions are the ones recorded in lab06/handoff/environment.json.
numpy==2.2.3
scipy==1.15.3
scikit-learn==1.9.0
joblib==1.6.0
threadpoolctl==3.6.0
"""


def build_fixed(handoff: Path, cfg: dict) -> dict:
    FIXED.mkdir(exist_ok=True)
    src = handoff / "pipeline.joblib"
    dst = FIXED / "pipeline.joblib"
    shutil.copyfile(src, dst)
    digest = sha256_file(dst)
    (FIXED / "pipeline.sha256").write_text(f"{digest}  pipeline.joblib\n", encoding="utf-8")
    old = json.loads((handoff / "data_schema.json").read_text(encoding="utf-8"))
    env = json.loads((handoff / "environment.json").read_text(encoding="utf-8"))
    new = {
        "schema_version": "2.0",
        "derived_from": "lab06/handoff/data_schema.json",
        "model_file": "pipeline.joblib",
        "model_sha256_file": "pipeline.sha256",
        "model_version": "lab06@" + cfg["object"]["frozen_commit"][:7] + ":automl_seed42_trial10",
        "sklearn_version": env["sklearn"],
        "positive_class": old["positive_class"],
        "threshold": cfg["threshold"],
        "features": old["features"],
        "record_format": "mapping feature_name -> number; key order is irrelevant",
        "unknown_fields": "reject",
        "missing_fields": "reject",
        "range_policy": "warn (accepted_with_warning); probability is not calibrated outside the train range",
        "domain_checks": {"area_radius_rel_tol": cfg["domain"]["area_radius_rel_tol"],
                          "rule": "|area_mean / (pi * radius_mean^2) - 1| <= tol, otherwise warning"},
        "max_batch": cfg["domain"]["max_batch"],
        "constraints": "finite, non-negative, numeric (bool and str rejected)",
    }
    (FIXED / "data_schema.json").write_text(json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(handoff / "inference_example.json", FIXED / "inference_example.json")
    (FIXED / "requirements.txt").write_text(MIN_REQUIREMENTS, encoding="utf-8")
    return {"model_sha256": digest, "identical_to_handoff": digest == sha256_file(src)}
