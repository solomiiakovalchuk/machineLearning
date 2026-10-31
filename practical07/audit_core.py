"""Small, independently testable audit functions for practical work 7.

Convention: y == 1 means malignant (positive class). NOTE: sklearn's load_breast_cancer
encodes malignant as 0, so the audit target is 1 - bunch.target (see `audit_target`).
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixed"))
from serving import validate_record, record_to_vector, ACCEPTED, REJECTED, WARNING  # noqa: E402,F401

SEVERITIES = ("critical", "high", "medium", "low")
CLAIM_VERDICTS = ("підтверджено", "підтверджено частково", "спростовано", "неможливо перевірити")
AUDIT_VERDICTS = ("прийнято", "прийнято з обмеженнями", "потребує доопрацювання", "відхилено")
METRIC_KEYS = ("ap", "roc_auc", "f1", "precision", "recall", "balanced_accuracy", "brier", "log_loss")


def audit_target(sklearn_target) -> np.ndarray:
    """Malignant = 1. sklearn encodes malignant as 0 and benign as 1."""
    return (np.asarray(sklearn_target) == 0).astype(int)


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- data / leakage
def detect_split_overlap(train_ids, test_ids) -> set:
    """O_split = |I_train ∩ I_test| as the set of shared ids."""
    return set(np.asarray(train_ids).tolist()) & set(np.asarray(test_ids).tolist())


def detect_duplicate_rows(X) -> int:
    """Number of rows that exactly repeat an earlier row."""
    return int(pd.DataFrame(np.asarray(X)).duplicated().sum())


def min_cross_distance(X_train, X_test) -> np.ndarray:
    """Distance from every test row to its nearest train row in train-standardised space."""
    Xtr, Xte = np.asarray(X_train, float), np.asarray(X_test, float)
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sd = np.where(sd == 0, 1.0, sd)
    a, b = (Xtr - mu) / sd, (Xte - mu) / sd
    d2 = (b ** 2).sum(1)[:, None] + (a ** 2).sum(1)[None, :] - 2 * b @ a.T
    return np.sqrt(np.clip(d2.min(1), 0, None))


def feature_target_correlations(X, y, names=None, threshold=0.9) -> pd.DataFrame:
    """r_j = corr(X_j, y). High |r| is a SIGNAL for manual review, not proof of leakage."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    names = list(names) if names is not None else [f"x{j}" for j in range(X.shape[1])]
    r = np.array([np.corrcoef(X[:, j], y)[0, 1] if X[:, j].std() > 0 else np.nan for j in range(X.shape[1])])
    df = pd.DataFrame({"feature": names, "r": r, "abs_r": np.abs(r)})
    df["signal"] = df["abs_r"] >= threshold
    return df.sort_values("abs_r", ascending=False, ignore_index=True)


def pipeline_fits_inside(model) -> dict:
    """Static check: preprocessing steps are part of the Pipeline object itself."""
    steps = getattr(model, "steps", None)
    if not steps:
        return {"is_pipeline": False, "steps": [], "preprocessing_inside": False}
    names = [type(s).__name__ for _, s in steps]
    pre = [n for n in names[:-1]]
    return {"is_pipeline": True, "steps": names, "preprocessing_inside": len(pre) > 0}


# ---------------------------------------------------------------- metrics
def evaluate_model(model, X, y, threshold=0.5) -> dict:
    """Independent recomputation of metrics from predict_proba[:, 1]."""
    y = np.asarray(y).astype(int)
    p = model.predict_proba(np.asarray(X, float))[:, 1]
    return metrics_from_proba(y, p, threshold)


def metrics_from_proba(y, p, threshold=0.5) -> dict:
    y = np.asarray(y).astype(int)
    p = np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        "n": int(len(y)), "positive_n": int(y.sum()), "threshold": float(threshold),
        "ap": float(average_precision_score(y, p)),
        "roc_auc": float(roc_auc_score(y, p)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1])),
        "confusion_matrix": cm.tolist(),
    }


def compare_metrics(reported: dict, reproduced: dict, tol=1e-9, keys=METRIC_KEYS) -> pd.DataFrame:
    """m = |m_reported - m_reproduced| per metric plus the confusion matrix equality."""
    rows = []
    for k in keys:
        if k in reported and k in reproduced:
            d = abs(float(reported[k]) - float(reproduced[k]))
            rows.append({"metric": k, "reported": reported[k], "reproduced": reproduced[k],
                         "abs_delta": d, "within_tol": bool(d <= tol)})
    if "confusion_matrix" in reported and "confusion_matrix" in reproduced:
        same = np.array_equal(np.asarray(reported["confusion_matrix"]), np.asarray(reproduced["confusion_matrix"]))
        rows.append({"metric": "confusion_matrix", "reported": str(reported["confusion_matrix"]),
                     "reproduced": str(reproduced["confusion_matrix"]),
                     "abs_delta": 0.0 if same else float("nan"), "within_tol": bool(same)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- uncertainty / stability
def summarize(values) -> dict:
    v = np.asarray(values, float)
    return {"n": int(len(v)), "mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
            "min": float(v.min()), "max": float(v.max())}


def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=42) -> tuple:
    """Percentile bootstrap interval of the MEAN of `values`."""
    v = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(1)
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def bootstrap_metric_ci(y, p, metric, n_boot=2000, alpha=0.05, seed=42, threshold=0.5):
    """Percentile CI of a test metric by resampling OBJECTS (stratification not enforced)."""
    y, p = np.asarray(y).astype(int), np.asarray(p, float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        vals.append(metrics_from_proba(y[i], p[i], threshold)[metric])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


def prediction_disagreement(labels_a, labels_b) -> float:
    """D = (1/n) * sum I[y_a != y_b]."""
    a, b = np.asarray(labels_a), np.asarray(labels_b)
    if a.shape != b.shape:
        raise ValueError("label vectors must have equal shape")
    return float(np.mean(a != b))


def mean_abs_proba_diff(p_a, p_b) -> float:
    """Δp = (1/n) * sum |p_a - p_b|."""
    a, b = np.asarray(p_a, float), np.asarray(p_b, float)
    if a.shape != b.shape:
        raise ValueError("probability vectors must have equal shape")
    return float(np.mean(np.abs(a - b)))


def ranking_stability(importances, top_k=5) -> dict:
    """importances: (runs, features). Mean pairwise Spearman rho and mean pairwise top-k Jaccard."""
    M = np.asarray(importances, float)
    ranks = pd.DataFrame(M.T).rank(ascending=False).to_numpy().T
    rho, jac = [], []
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            rho.append(np.corrcoef(ranks[i], ranks[j])[0, 1])
            a, b = set(np.argsort(-M[i])[:top_k]), set(np.argsort(-M[j])[:top_k])
            jac.append(len(a & b) / len(a | b))
    return {"mean_spearman": float(np.mean(rho)), "mean_topk_jaccard": float(np.mean(jac)), "pairs": len(rho)}


# ---------------------------------------------------------------- error analysis
def select_error_objects(ids, y, p, threshold=0.5, n_near=5, n_conf_err=0) -> pd.DataFrame:
    """All errors + `n_near` objects closest to the threshold (+ most confident errors if requested)."""
    ids, y, p = np.asarray(ids), np.asarray(y).astype(int), np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    df = pd.DataFrame({"id": ids, "y": y, "p": p, "pred": pred})
    df["is_error"] = df.y != df.pred
    df["margin"] = (df.p - threshold).abs()
    df["conf_in_wrong"] = np.where(df.is_error, (df.p - df.y).abs(), np.nan)
    parts = []
    err = df[df.is_error].copy()
    err["category"] = "error"
    parts.append(err)
    if n_conf_err:
        top = err.sort_values("conf_in_wrong", ascending=False).head(n_conf_err).copy()
        top["category"] = "confident_error"
        parts.append(top)
    near = df.sort_values("margin").head(n_near).copy()
    near["category"] = "near_threshold"
    parts.append(near)
    out = pd.concat(parts, ignore_index=True)
    return out


# ---------------------------------------------------------------- claims, findings, verdict
_OPS = {
    "le": lambda a, b: a <= b, "ge": lambda a, b: a >= b, "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b, "gt": lambda a, b: a > b, "true": lambda a, b: bool(a) is True,
    "false": lambda a, b: bool(a) is False,
}


def verify_claim(claim: dict, evidence: dict) -> dict:
    """Evaluate every acceptance criterion of a claim against measured evidence.

    claim["criteria"] = [{"name", "key", "op", "value"}]; evidence maps key -> measured value.
    A criterion whose key is absent (or None) makes the claim unverifiable only if NO criterion
    could be evaluated; otherwise the missing one counts as failed.
    """
    crit = claim.get("criteria") or []
    if not crit:
        raise ValueError("a claim needs at least one acceptance criterion")
    results = []
    for c in crit:
        if c["op"] not in _OPS:
            raise ValueError(f"unknown operator {c['op']}")
        measured = evidence.get(c["key"])
        if measured is None or (isinstance(measured, float) and np.isnan(measured)):
            results.append({"name": c["name"], "measured": None, "passed": False, "evaluated": False})
        else:
            results.append({"name": c["name"], "measured": measured, "passed": bool(_OPS[c["op"]](measured, c.get("value"))),
                            "evaluated": True})
    n_eval = sum(r["evaluated"] for r in results)
    n_pass = sum(r["passed"] for r in results)
    if n_eval == 0:
        verdict = "неможливо перевірити"
    elif n_pass == len(results):
        verdict = "підтверджено"
    elif n_pass == 0:
        verdict = "спростовано"
    else:
        verdict = "підтверджено частково"
    return {"id": claim["id"], "verdict": verdict, "criteria_results": results, "passed": n_pass, "total": len(results)}


def build_finding(problem: str, evidence: str, severity: str, recommendation: str = "", status: str = "open",
                  fid: str | None = None) -> dict:
    if severity not in SEVERITIES:
        raise ValueError(f"severity must be one of {SEVERITIES}")
    if not str(problem).strip() or not str(evidence).strip():
        raise ValueError("a finding needs a problem and evidence")
    if status not in ("open", "fixed", "accepted_risk", "documented"):
        raise ValueError("unknown finding status")
    return {"id": fid, "problem": problem, "evidence": evidence, "severity": severity,
            "recommendation": recommendation, "status": status}


def decide_verdict(findings, rule=None) -> str:
    """Verdict from the OPEN findings only: critical -> відхилено, high -> доопрацювання, any other -> з обмеженнями."""
    open_f = [f for f in findings if f["status"] in ("open", "accepted_risk", "documented")]
    sev = {f["severity"] for f in open_f}
    if "critical" in sev:
        return "відхилено"
    if "high" in sev:
        return "потребує доопрацювання"
    if sev:
        return "прийнято з обмеженнями"
    return "прийнято"


def file_manifest(paths, root) -> dict:
    root = Path(root)
    out = {}
    for p in paths:
        p = Path(p)
        key = p.relative_to(root).as_posix() if p.is_absolute() else p.as_posix()
        out[key] = {"sha256": sha256_file(p if p.is_absolute() else root / p),
                    "bytes": (p if p.is_absolute() else root / p).stat().st_size}
    return out


def inventory(root, required) -> pd.DataFrame:
    root = Path(root)
    rows = []
    for name in required:
        f = root / name
        rows.append({"file": name, "present": f.is_file(), "bytes": f.stat().st_size if f.is_file() else 0})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- git provenance (read-only)
def git_text(repo, *args):
    """Run a read-only git command; None when git fails."""
    try:
        p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=60)
    except Exception:
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def worktree_matches_commit(repo, commit, relpath):
    """Does the working-tree file have the same git blob as `commit:relpath`?"""
    blob = git_text(repo, "rev-parse", f"{commit}:{relpath}")
    now = git_text(repo, "hash-object", str(Path(repo) / relpath))
    if blob is None or now is None:
        return None
    return blob == now


def commits_adding(repo, relpaths):
    """Short hashes of the commits that first added each path."""
    out = set()
    for rel in relpaths:
        log = git_text(repo, "log", "--diff-filter=A", "--format=%h", "--", rel)
        if log:
            out.add(log.splitlines()[-1])
    return out
