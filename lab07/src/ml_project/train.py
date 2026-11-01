"""Controlled rebuild of the frozen model: same train ids, family, params, preprocessing, target mapping.

No search is performed here. Test ids are used only to recompute the already known reference metric.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

from .config import load_project_config
from .errors import RebuildError
from .integrity import sha256_file
from .predictor import ProjectPredictor

SUPPORTED_FAMILY = "lrl2"


def raw_data_sha256(raw) -> str:
    """Same definition as lab06 data_provenance.json: sha256(data bytes + target bytes)."""
    h = hashlib.sha256()
    h.update(np.ascontiguousarray(raw.data).tobytes())
    h.update(np.ascontiguousarray(raw.target).tobytes())
    return h.hexdigest()


def build_pipeline(params: dict, fixed: dict, random_state: int):
    """Frozen pipeline: median imputer -> StandardScaler -> L2 logistic regression."""
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    allowed = {"C", "class_weight"}
    if set(params) - allowed:
        raise RebuildError(f"unsupported frozen params: {sorted(set(params) - allowed)}")
    model = LogisticRegression(max_iter=fixed["max_iter"], random_state=random_state, **params)
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()), ("model", model)])


def load_frozen_inputs(project_root):
    """Load data + ids + frozen decision, checking hashes and the target mapping. Returns a dict."""
    from sklearn.datasets import load_breast_cancer

    config = load_project_config(Path(project_root) / "configs" / "project_config.json")
    paths = config.paths
    metadata = json.loads(paths["model_metadata"].read_text(encoding="utf-8"))
    for key, field in (("frozen_decision", "frozen_decision_sha256"), ("split_ids", "split_ids_sha256")):
        if sha256_file(paths[key]) != metadata[field]:
            raise RebuildError(f"{key} hash differs from model_metadata.json")
    decision = json.loads(paths["frozen_decision"].read_text(encoding="utf-8"))
    if decision["family"] != SUPPORTED_FAMILY:
        raise RebuildError(f"frozen family {decision['family']!r} is not supported; rebuild never searches")

    raw = load_breast_cancer()
    if raw_data_sha256(raw) != metadata["data_reference"]["raw_x_target_sha256"]:
        raise RebuildError("raw dataset differs from the one used in lab06")
    y = (np.asarray(raw.target) == 0).astype(int)  # 1 = malignant, as in lab06 experiment.load_data
    X = np.asarray(raw.data, dtype=float)

    import csv
    with open(paths["split_ids"], newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    train_ids = np.array([int(r["id"]) for r in rows if r["part"] == "train"])
    test_ids = np.array([int(r["id"]) for r in rows if r["part"] == "test"])
    if set(train_ids) & set(test_ids) or len(train_ids) + len(test_ids) != len(X):
        raise RebuildError("split ids overlap or do not cover the dataset")
    return {"config": config, "metadata": metadata, "decision": decision, "X": X, "y": y,
            "train_ids": train_ids, "test_ids": test_ids}


def rebuild_model(project_root, output_path) -> dict:
    """Refit the frozen pipeline on the frozen train ids, save it to output_path and compare with the saved model.

    The shipped model file is never overwritten. Returns a JSON-serialisable report.
    """
    import joblib
    from sklearn.metrics import average_precision_score, confusion_matrix

    data = load_frozen_inputs(project_root)
    config, metadata, decision = data["config"], data["metadata"], data["decision"]
    output_path = Path(output_path).resolve()
    if output_path == config.paths["model"]:
        raise RebuildError("output_path must differ from the frozen model path")

    X, y, tr, te = data["X"], data["y"], data["train_ids"], data["test_ids"]
    pipeline = build_pipeline(decision["params"], metadata["fixed_model_params"], config.random_state)
    pipeline.fit(X[tr], y[tr])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output_path)

    saved = ProjectPredictor(config)  # hash-verified shipped model
    p_saved_test = saved.model_probabilities(X[te])
    p_new_test = pipeline.predict_proba(X[te])[:, 1]
    p_saved_all = saved.model_probabilities(X)
    p_new_all = pipeline.predict_proba(X)[:, 1]
    example = json.loads(config.paths["inference_example"].read_text(encoding="utf-8"))
    names = saved.describe()["feature_names"]
    control = np.array([[r[n] for n in names] for r in example["records"]])
    p_saved_ctrl = saved.model_probabilities(control)
    p_new_ctrl = pipeline.predict_proba(control)[:, 1]

    ref = metadata["reference_test_metrics"]
    ap_new = float(average_precision_score(y[te], p_new_test))
    ap_saved = float(average_precision_score(y[te], p_saved_test))
    cm = confusion_matrix(y[te], p_new_test >= config.threshold, labels=[0, 1]).tolist()
    saved_lr, new_lr = saved.model_steps()[-1], pipeline.steps[-1][1]
    report = {
        "family": decision["family"], "candidate_id": decision["candidate_id"], "params": decision["params"],
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "train_test_disjoint": not (set(tr) & set(te)),
        "train_ids_sha256": hashlib.sha256(",".join(map(str, tr.tolist())).encode()).hexdigest(),
        "rebuilt_sha256": sha256_file(output_path),
        "saved_sha256": metadata["model_sha256"],
        "byte_identical_file": sha256_file(output_path) == metadata["model_sha256"],
        "tolerances": {"probability_abs": config.probability_tolerance, "metric_abs": config.metric_tolerance},
        "control_records": {"saved": p_saved_ctrl.tolist(), "rebuilt": p_new_ctrl.tolist(),
                            "expected": example["expected_probability"],
                            "max_abs_diff_saved_vs_rebuilt": float(np.max(np.abs(p_saved_ctrl - p_new_ctrl))),
                            "max_abs_diff_rebuilt_vs_expected": float(np.max(np.abs(p_new_ctrl - np.array(example["expected_probability"]))))},
        "max_abs_diff_saved_vs_rebuilt_test": float(np.max(np.abs(p_saved_test - p_new_test))),
        "max_abs_diff_saved_vs_rebuilt_all569": float(np.max(np.abs(p_saved_all - p_new_all))),
        "max_abs_diff_coef": float(np.max(np.abs(saved_lr.coef_ - new_lr.coef_))),
        "abs_diff_intercept": float(np.max(np.abs(saved_lr.intercept_ - new_lr.intercept_))),
        "test_ap": {"reference_lab06": ref["ap"], "saved": ap_saved, "rebuilt": ap_new,
                    "abs_diff_rebuilt_vs_reference": abs(ap_new - ref["ap"])},
        "test_confusion_matrix": {"reference_lab06": ref["confusion_matrix"], "rebuilt": cm},
    }
    report["passed"] = bool(
        report["train_test_disjoint"]
        and report["control_records"]["max_abs_diff_saved_vs_rebuilt"] <= config.probability_tolerance
        and report["max_abs_diff_saved_vs_rebuilt_test"] <= config.probability_tolerance
        and report["test_ap"]["abs_diff_rebuilt_vs_reference"] <= config.metric_tolerance
        and cm == ref["confusion_matrix"])
    return report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild the frozen model (no model selection)")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--output", required=True, help="where to write the rebuilt model (not the shipped one)")
    args = parser.parse_args(argv)
    try:
        report = rebuild_model(args.project_root, args.output)
    except Exception as exc:  # controlled message + non-zero exit
        print(f"error[{type(exc).__name__}]: {exc}", file=sys.stderr)
        return getattr(exc, "exit_code", 1)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 9


if __name__ == "__main__":
    sys.exit(main())
