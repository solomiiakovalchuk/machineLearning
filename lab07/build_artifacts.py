"""Accept the lab06 handoff and build every lab07 artifact (idempotent).

Usage (from lab07/):  python build_artifacts.py [--handoff ../lab06/handoff]
Stages: accept handoff -> copy/convert artifacts -> metadata -> rebuild check (reports/reproducibility.json)
-> project_manifest.json (always LAST: it hashes the files written before it).
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402

from ml_project.integrity import sha256_file  # noqa: E402

HANDOFF_COMMIT = "81445e2"
REQUIRED = ["pipeline.joblib", "decision_record.json", "decision.sha256", "config.json", "data_schema.json",
            "split_ids.csv", "inference_example.json", "test_metrics.json", "MODEL_CARD.md",
            "requirements.txt", "environment.json", "data_provenance.json", "inference.py"]
# Expected by the lab07 methodology (task 1) but not delivered by the handoff.
EXPECTED_BUT_ABSENT = {"model checksum (sha256 of pipeline.joblib)": "handoff only hashes the decision record"}


def dump_json(path: Path, obj) -> None:
    path.write_bytes((json.dumps(obj, indent=2, ensure_ascii=False) + "\n").encode("utf-8"))


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


MANIFEST_FILES = ["models/pipeline.joblib", "models/model_metadata.json", "configs/project_config.json",
                  "configs/schema.json", "configs/frozen_decision.json", "configs/inference_example.json",
                  "data/split_ids.csv", "pyproject.toml", "requirements.txt", "build_artifacts.py",
                  "reports/reproducibility.json", "reports/handoff_acceptance.json", "reports/model_card.md"]


def write_manifest() -> None:
    """Hash every key file as it is on disk NOW; must run after all other writes."""
    files = list(MANIFEST_FILES)
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "src" / "ml_project").glob("*.py"))
    files += sorted(p.relative_to(ROOT).as_posix() for p in (ROOT / "tests").glob("*.py"))
    present = [f for f in files if (ROOT / f).is_file()]
    absent = [f for f in files if f not in present]
    dump_json(ROOT / "project_manifest.json", {
        "manifest_version": "1.0", "model_version": "1.0.0", "schema_version": "1.0",
        "algorithm": "sha256", "not_covered": ["project_manifest.json itself", "README.md", "AI_USAGE.md",
                                               "reports/test_results.txt", "notebook/", "figures/", "dist/"],
        "files": {f: sha256_file(ROOT / f) for f in present}, "optional_absent_at_build": absent})


def accept_handoff(handoff: Path):
    """Check presence, internal consistency and independent evidence; never repair silently."""
    report = {"handoff_commit": HANDOFF_COMMIT, "files": {}, "checks": {}, "gaps": [], "contradictions": []}
    for name in REQUIRED + ["cv_metrics.csv"]:
        p = handoff / name
        report["files"][name] = {"present": p.is_file(), "sha256": sha256_file(p) if p.is_file() else None}
    missing = [n for n in REQUIRED if not report["files"][n]["present"]]
    if missing:
        raise SystemExit(f"handoff incomplete, missing: {missing}")
    for what, why in EXPECTED_BUT_ABSENT.items():
        report["gaps"].append(f"{what}: absent ({why})")

    model_sha = report["files"]["pipeline.joblib"]["sha256"]
    decision_sha = report["files"]["decision_record.json"]["sha256"]
    recorded = (handoff / "decision.sha256").read_text().strip()
    report["checks"]["decision_sha_matches_decision_sha256_file"] = recorded == decision_sha
    tm = read_json(handoff / "test_metrics.json")
    report["checks"]["test_metrics_decision_sha_matches"] = tm["decision_sha256"] == decision_sha
    decision = read_json(handoff / "decision_record.json")
    report["checks"]["decision_has_no_test_metrics"] = decision["test_metrics_present"] is False
    report["checks"]["decision_config_sha_matches_handoff_config"] = (
        decision["config_sha256"] == report["files"]["config.json"]["sha256"])

    # Independent evidence for the model bytes: the lab06 results artifact (same repo, same commit).
    lab06_model = handoff.parent / "results" / "models" / "automl.joblib"
    if lab06_model.is_file():
        report["checks"]["model_bytes_equal_lab06_results_automl"] = sha256_file(lab06_model) == model_sha
    else:
        report["checks"]["model_bytes_equal_lab06_results_automl"] = None
        report["gaps"].append("lab06/results/models/automl.joblib not available: no independent model evidence")

    # Split ids and schema ranges vs the claims made in test_metrics / data_schema.
    split = pd.read_csv(handoff / "split_ids.csv")
    counts = split["part"].value_counts().to_dict()
    report["split_counts"] = {k: int(v) for k, v in counts.items()}
    report["checks"]["split_ids_unique_and_complete"] = bool(split["id"].is_unique and sorted(split["id"]) == list(range(569)))
    report["checks"]["test_count_matches_test_metrics"] = counts.get("test") == tm["n"]
    from sklearn.datasets import load_breast_cancer
    raw = load_breast_cancer()
    y = (raw.target == 0).astype(int)  # lab06 experiment.load_data: y=1 is malignant
    test_ids = split.loc[split.part == "test", "id"].to_numpy()
    train_ids = split.loc[split.part == "train", "id"].to_numpy()
    report["checks"]["test_positive_count_matches_test_metrics"] = int(y[test_ids].sum()) == tm["positive_n"]
    schema = read_json(handoff / "data_schema.json")
    X = raw.data
    smin = np.array([f["min"] for f in schema["features"]])
    smax = np.array([f["max"] for f in schema["features"]])
    report["checks"]["schema_range_equals_train_min_max"] = bool(
        np.allclose(smin, X[train_ids].min(axis=0), rtol=1e-9) and np.allclose(smax, X[train_ids].max(axis=0), rtol=1e-9))
    report["checks"]["schema_range_equals_all_data_min_max"] = bool(
        np.allclose(smin, X.min(axis=0), rtol=1e-9) and np.allclose(smax, X.max(axis=0), rtol=1e-9))

    # Model evidence by behaviour: reproduce the handoff control probabilities.
    model = joblib.load(handoff / "pipeline.joblib")  # bytes already tied to lab06/results above
    example = read_json(handoff / "inference_example.json")
    p = model.predict_proba(np.array(example["input"], float))[:, 1]
    report["control_reproduction_max_abs_diff"] = float(np.max(np.abs(p - np.array(example["expected_probability"]))))
    report["model_params"] = model.steps[-1][1].get_params()
    return report, model, decision, tm, schema, example


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--handoff", default=str(ROOT.parent / "lab06" / "handoff"))
    args = ap.parse_args()
    handoff = Path(args.handoff).resolve()

    report, model, decision, tm, schema, example = accept_handoff(handoff)

    # 1. copy / convert artifacts
    shutil.copyfile(handoff / "pipeline.joblib", ROOT / "models" / "pipeline.joblib")
    shutil.copyfile(handoff / "split_ids.csv", ROOT / "data" / "split_ids.csv")
    shutil.copyfile(handoff / "decision_record.json", ROOT / "configs" / "frozen_decision.json")
    names = [f["name"] for f in schema["features"]]
    dump_json(ROOT / "configs" / "schema.json", {
        "schema_version": "1.0",
        "positive_class": "malignant", "negative_class": "benign",
        "target_mapping": "y=1 is malignant; sklearn load_breast_cancer target==0 (lab06 experiment.load_data)",
        "order_required": True, "finite_required": True, "non_negative": True,
        "out_of_range_policy": "warning", "missing_policy": "reject", "extra_policy": "reject",
        "source": "lab06/handoff/data_schema.json",
        "features": [{"name": f["name"], "dtype": f["dtype"], "required": f["required"],
                      "min": f["min"], "max": f["max"]} for f in schema["features"]]})
    probs, thr = example["expected_probability"], example["threshold"]
    records = [dict(zip(names, row)) for row in example["input"]]
    dump_json(ROOT / "configs" / "inference_example.json", {
        "description": "Control requests from lab06 handoff/inference_example.json, converted to named features.",
        "source": "lab06/handoff/inference_example.json",
        "record": records[0], "records": records,
        "expected_probability": probs,
        "expected_predicted_class": ["malignant" if p >= thr else "benign" for p in probs],
        "threshold": thr})

    # 2. metadata
    env = read_json(handoff / "environment.json")
    mp = report["model_params"]
    dump_json(ROOT / "models" / "model_metadata.json", {
        "model_version": "1.0.0", "schema_version": "1.0",
        "family": decision["family"], "candidate_id": decision["candidate_id"], "params": decision["params"],
        "fixed_model_params": {"max_iter": mp["max_iter"], "random_state": mp["random_state"],
                               "solver": mp["solver"], "penalty": mp.get("penalty", "l2")},
        "pipeline": "SimpleImputer(median, keep_empty_features=True) -> StandardScaler -> LogisticRegression",
        "positive_class": "malignant", "negative_class": "benign", "threshold": thr,
        "model_sha256": sha256_file(ROOT / "models" / "pipeline.joblib"),
        "schema_sha256": sha256_file(ROOT / "configs" / "schema.json"),
        "frozen_decision_sha256": sha256_file(ROOT / "configs" / "frozen_decision.json"),
        "split_ids_sha256": sha256_file(ROOT / "data" / "split_ids.csv"),
        "data_reference": {"raw_x_target_sha256": read_json(handoff / "data_provenance.json")["raw_x_target_sha256"],
                           "target_definition": "y = (load_breast_cancer().target == 0), 1 = malignant"},
        "reference_test_metrics": {"source": "lab06/handoff/test_metrics.json", "ap": tm["ap"], "n": tm["n"],
                                   "positive_n": tm["positive_n"], "threshold": tm["threshold"],
                                   "confusion_matrix": tm["confusion_matrix"]},
        "environment": {"python": platform.python_version(), "scikit-learn": sklearn.__version__, "numpy": np.__version__,
                        "scipy": scipy.__version__, "joblib": joblib.__version__},
        "trained_in_lab06_environment": {k: env[k] for k in ("python", "sklearn", "numpy", "scipy")},
        "source_handoff": {"commit": HANDOFF_COMMIT, "directory": "lab06/handoff"},
        "intended_use": "Educational prototype: rank FNA records by probability of malignant for expert review.",
        "limitations": decision["limitations"] + ["not a diagnostic tool; no calibration or drift monitoring"],
    })
    dump_json(ROOT / "reports" / "handoff_acceptance.json", report)
    print("handoff accepted; control diff", report["control_reproduction_max_abs_diff"])

    # 3. rebuild check: refit the frozen spec into a temp file and compare with the shipped model
    from ml_project.train import rebuild_model
    with tempfile.TemporaryDirectory() as tmp:
        repro = rebuild_model(ROOT, Path(tmp) / "rebuilt.joblib")
    repro["environment"] = {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__, "joblib": joblib.__version__}
    dump_json(ROOT / "reports" / "reproducibility.json", repro)
    print("rebuild passed:", repro["passed"], "max |dp| test:", repro["max_abs_diff_saved_vs_rebuilt_test"])
    if not repro["passed"]:
        raise SystemExit("rebuild did not reproduce the saved model; manifest not written")

    # 4. manifest last
    write_manifest()
    print("manifest written")


if __name__ == "__main__":
    main()
