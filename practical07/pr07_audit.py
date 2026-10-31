"""One-command audit of the frozen lab06 solution:  python pr07_audit.py

Reads lab06 (never writes there), writes results/, figures/, artifacts/ and builds fixed/.
Protocol parameters come from configs/audit_config.json (frozen before the first run).
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import audit_core as A
from build_fixed import build_fixed

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
CFG = json.loads((ROOT / "configs" / "audit_config.json").read_text(encoding="utf-8"))
OBJ = REPO / CFG["object"]["dir"]
H = OBJ / "handoff"
RES, FIG, ART, FIXED = ROOT / "results", ROOT / "figures", ROOT / "artifacts", ROOT / "fixed"
SEED, THR = CFG["seed"], CFG["threshold"]
TOL = CFG["tolerances"]
BLUE, GRAY, RED, GREEN, ORANGE = "#2b6cb0", "#8a8f98", "#c0392b", "#2f855a", "#dd6b20"


def _json_default(o):
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not serialisable: {type(o)}")


def jdump(obj, path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")


def git(*args):
    try:
        p = subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, timeout=60)
        return p.stdout.strip() if p.returncode == 0 else None
    except Exception:
        return None


def sk_name(n: str) -> str:
    """sklearn feature name -> lab06 schema name ('mean radius' -> 'radius_mean')."""
    parts = n.split()
    if parts[0] == "mean":
        return "_".join(parts[1:]) + "_mean"
    if parts[0] == "worst":
        return "_".join(parts[1:]) + "_worst"
    if parts[-1] == "error":
        return "_".join(parts[:-1]) + "_se"
    raise ValueError(n)


def make_pipeline(C, class_weight=None):
    return Pipeline([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                     ("scaler", StandardScaler()),
                     ("model", LogisticRegression(C=C, class_weight=class_weight, max_iter=5000, random_state=SEED))])


def load_legacy_predict():
    spec = importlib.util.spec_from_file_location("lab06_inference", H / "inference.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.predict


# ====================================================================== stages
def stage_frozen():
    """Task 1: fix the version of the audited object."""
    commit = CFG["object"]["frozen_commit"]
    head = git("rev-parse", "HEAD")
    exists = git("cat-file", "-t", commit) == "commit"
    rows = []
    for name in CFG["object"]["handoff_files"]:
        f = H / name
        rel = f"{CFG['object']['dir']}/handoff/{name}"
        same = A.worktree_matches_commit(REPO, commit, rel) if exists else None
        first = git("log", "--diff-filter=A", "--format=%h %aI", "--", rel)
        rows.append({"file": name, "sha256": A.sha256_file(f) if f.is_file() else None,
                     "bytes": f.stat().st_size if f.is_file() else 0,
                     "worktree_vs_commit": same,
                     "added_in_commit": (first.splitlines()[-1] if first else None)})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "frozen_object.csv", index=False)
    status = git("status", "--porcelain", "--", CFG["object"]["dir"])
    info = {"frozen_commit": commit, "commit_exists": exists, "repo_head_at_audit": head,
            "lab06_working_tree_clean": (status == "") if status is not None else None,
            "all_blobs_match_frozen_commit": bool(df.worktree_vs_commit.isin(["identical", "line_endings_only"]).all()),
            "files_identical_bytes": int((df.worktree_vs_commit == "identical").sum()),
            "files_line_endings_only": int((df.worktree_vs_commit == "line_endings_only").sum()),
            "files_different": int((df.worktree_vs_commit == "different").sum()),
            "n_files": len(df)}
    key = [f"{CFG['object']['dir']}/handoff/{n}" for n in ("decision.sha256", "decision_record.json", "pipeline.joblib", "test_metrics.json")]
    info["handoff_files_added_in_n_commits"] = len(A.commits_adding(REPO, key))
    return df, info


def stage_inventory():
    """Task 3: completeness table."""
    rows = []
    def add(item, path, present_note=""):
        p = REPO / path
        rows.append({"item": item, "path": path, "present": p.exists(), "note": present_note})
    add("дані: snapshot", "lab06/data/README.md", "дані не копіюються; sklearn.load_breast_cancer + SHA-256 у data_provenance.json")
    add("split ids", "lab06/handoff/split_ids.csv")
    add("код навчання", "lab06/experiment.py")
    add("код запуску", "lab06/run_experiment.py")
    add("конфігурація", "lab06/handoff/config.json")
    add("залежності", "lab06/handoff/requirements.txt")
    add("модель", "lab06/handoff/pipeline.joblib")
    add("схема входу", "lab06/handoff/data_schema.json")
    add("контрольний приклад", "lab06/handoff/inference_example.json")
    add("inference", "lab06/handoff/inference.py")
    add("заявлені результати", "lab06/handoff/test_metrics.json")
    add("рішення + хеш", "lab06/handoff/decision_record.json")
    add("README з командами", "lab06/README.md")
    add("журнал ШІ", "lab06/AI_USAGE.md")
    add("model card", "lab06/handoff/MODEL_CARD.md")
    add("тести", "lab06/tests/test_experiment.py")
    df = pd.DataFrame(rows)
    df.to_csv(RES / "inventory.csv", index=False)
    return df


def stage_data():
    """Task 6: data audit."""
    d = load_breast_cancer()
    X, y_sk = d.data, d.target
    y = A.audit_target(y_sk)
    names = [sk_name(n) for n in d.feature_names]
    split = pd.read_csv(H / "split_ids.csv")
    tr_ids = split[split.part == "train"].id.to_numpy()
    te_ids = split[split.part == "test"].id.to_numpy()
    schema = json.loads((H / "data_schema.json").read_text(encoding="utf-8"))
    schema_names = [f["name"] for f in schema["features"]]
    prov = json.loads((H / "data_provenance.json").read_text(encoding="utf-8"))
    import hashlib
    raw_hash = hashlib.sha256(np.ascontiguousarray(d.data).tobytes() + np.ascontiguousarray(d.target).tobytes()).hexdigest()
    Xtr, Xte, ytr, yte = X[tr_ids], X[te_ids], y[tr_ids], y[te_ids]
    smin = np.array([f["min"] for f in schema["features"]])
    smax = np.array([f["max"] for f in schema["features"]])
    lab05 = pd.read_csv(REPO / "lab05" / "results" / "split_ids.csv")
    l5_train = set(lab05[lab05.part == "train"].id)
    facts = {
        "n": int(len(X)), "n_features": int(X.shape[1]), "positives": int(y.sum()), "negatives": int((1 - y).sum()),
        "sklearn_target_malignant_code": 0,
        "train_n": int(len(tr_ids)), "test_n": int(len(te_ids)),
        "train_pos": int(ytr.sum()), "test_pos": int(yte.sum()),
        "train_pos_rate": float(ytr.mean()), "test_pos_rate": float(yte.mean()),
        "nan_count": int(np.isnan(X).sum()), "duplicate_rows_all": A.detect_duplicate_rows(X),
        "constant_features": int((X.std(0) == 0).sum()), "negative_values": int((X < 0).sum()),
        "ids_unique": bool(len(set(tr_ids)) == len(tr_ids) and len(set(te_ids)) == len(te_ids)),
        "ids_cover_dataset": bool(set(tr_ids) | set(te_ids) == set(range(len(X)))),
        "split_overlap": len(A.detect_split_overlap(tr_ids, te_ids)),
        "feature_names_match_schema": names == schema_names,
        "schema_range_equals_train_range": bool(np.allclose(smin, Xtr.min(0), rtol=0, atol=1e-12) and np.allclose(smax, Xtr.max(0), rtol=0, atol=1e-12)),
        "schema_range_equals_full_range": bool(np.allclose(smin, X.min(0), rtol=0, atol=1e-12) and np.allclose(smax, X.max(0), rtol=0, atol=1e-12)),
        "test_outside_train_range_cells": int(((Xte < smin) | (Xte > smax)).sum()),
        "raw_sha256_matches_provenance": raw_hash == prov["raw_x_target_sha256"],
        "raw_sha256": raw_hash,
        "lab06_test_in_lab05_train": int(len(set(te_ids) & l5_train)),
        "lab05_overlap_matches_provenance": int(len(set(te_ids) & l5_train)) == prov["lab05_split_overlap"]["lab06_test_ids_in_lab05_train"],
        "dup_train_test_cross": int(len(set(map(tuple, np.round(Xtr, 12))) & set(map(tuple, np.round(Xte, 12))))),
        "group_or_patient_id_available": False,
        "time_column_available": False,
    }
    dist = A.min_cross_distance(Xtr, Xte)
    facts["min_test_to_train_distance"] = float(dist.min())
    facts["n_near_duplicate_test"] = int((dist < CFG["leakage"]["near_duplicate_distance"]).sum())
    jdump(facts, RES / "data_audit.json")
    return dict(X=X, y=y, names=names, tr_ids=tr_ids, te_ids=te_ids, Xtr=Xtr, Xte=Xte, ytr=ytr, yte=yte, schema=schema, facts=facts)


def stage_leakage(D, frozen):
    """Task 7: leakage scan."""
    corr = A.feature_target_correlations(D["Xtr"], D["ytr"], D["names"], CFG["leakage"]["corr_signal_threshold"])
    corr.to_csv(RES / "leakage_correlations.csv", index=False)
    scaler = frozen.named_steps["scaler"]
    mean_train, mean_all = D["Xtr"].mean(0), D["X"].mean(0)
    d_train = float(np.abs(scaler.mean_ - mean_train).max())
    d_all = float(np.abs(scaler.mean_ - mean_all).max())
    code = (OBJ / "run_experiment.py").read_text(encoding="utf-8").splitlines()
    first_test = next(i for i, l in enumerate(code) if "Xall[test_ids]" in l and not l.strip().startswith("#"))
    freeze = next(i for i, l in enumerate(code) if "decision.sha256" in l and "write_text" in l)
    cfgj = json.loads((H / "config.json").read_text(encoding="utf-8"))
    pf = A.pipeline_fits_inside(frozen)
    dec_hash_ok = A.sha256_file(H / "decision_record.json") == (H / "decision.sha256").read_text().strip()
    all_hashes = " ".join(p.read_text(errors="ignore") for p in H.iterdir() if p.suffix in (".json", ".md", ".sha256", ".txt", ".csv"))
    model_hash = A.sha256_file(H / "pipeline.joblib")
    checks = pd.DataFrame([
        {"check": "id overlap train/test", "result": D["facts"]["split_overlap"], "expected": 0, "ok": D["facts"]["split_overlap"] == 0},
        {"check": "exact duplicate rows (all data)", "result": D["facts"]["duplicate_rows_all"], "expected": 0, "ok": D["facts"]["duplicate_rows_all"] == 0},
        {"check": "identical train/test feature rows", "result": D["facts"]["dup_train_test_cross"], "expected": 0, "ok": D["facts"]["dup_train_test_cross"] == 0},
        {"check": "max |r| with target (signal only)", "result": round(float(corr.abs_r.max()), 4), "expected": f"< {CFG['leakage']['corr_signal_threshold']}", "ok": bool(corr.abs_r.max() < CFG["leakage"]["corr_signal_threshold"])},
        {"check": "test rows closer than near-duplicate distance to a train row", "result": D["facts"]["n_near_duplicate_test"], "expected": 0, "ok": D["facts"]["n_near_duplicate_test"] == 0},
        {"check": "preprocessing is inside the Pipeline object", "result": "->".join(pf["steps"]), "expected": "imputer, scaler before model", "ok": pf["preprocessing_inside"]},
        {"check": "scaler mean = train mean (max |delta|)", "result": f"{d_train:.2e}", "expected": "<= 1e-9", "ok": d_train <= TOL["coefficient"]},
        {"check": "scaler mean differs from full-data mean (max |delta|)", "result": f"{d_all:.3e}", "expected": "> 1e-6", "ok": d_all > 1e-6},
        {"check": "scaler n_samples_seen_ equals train size", "result": int(scaler.n_samples_seen_), "expected": D["facts"]["train_n"], "ok": int(scaler.n_samples_seen_) == D["facts"]["train_n"]},
        {"check": "config flags test_used_for_search/selection are false", "result": f"{cfgj['test_used_for_search']}/{cfgj['test_used_for_selection']}", "expected": "False/False", "ok": (cfgj["test_used_for_search"] is False and cfgj["test_used_for_selection"] is False)},
        {"check": "run_experiment.py: decision hash written before first Xall[test_ids]", "result": f"line {freeze + 1} < line {first_test + 1}", "expected": "freeze < test", "ok": freeze < first_test},
        {"check": "decision.sha256 matches decision_record.json", "result": dec_hash_ok, "expected": True, "ok": bool(dec_hash_ok)},
        {"check": "pipeline.joblib SHA-256 is recorded anywhere in the handoff", "result": model_hash in all_hashes, "expected": True, "ok": model_hash in all_hashes},
        {"check": "group/patient id or time column available", "result": False, "expected": "needed to exclude dependence", "ok": False},
    ])
    checks.to_csv(RES / "leakage_checks.csv", index=False)
    ev = {"scaler_mean_delta_train": d_train, "scaler_mean_delta_all": d_all, "decision_before_test_in_code": freeze < first_test,
          "decision_hash_valid": bool(dec_hash_ok), "model_hash_recorded": model_hash in all_hashes,
          "flags_false": cfgj["test_used_for_search"] is False and cfgj["test_used_for_selection"] is False,
          "max_abs_corr": float(corr.abs_r.max()), "top_corr_feature": corr.feature[0], "n_corr_signals": int(corr.signal.sum()),
          "preprocessing_inside": pf["preprocessing_inside"], "pipeline_steps": pf["steps"], "model_sha256": model_hash}
    return corr, checks, ev


def stage_dataflow(frozen_ev):
    rows = [
        ("fit (imputer, scaler, LR)", "train, 455 записів", "так", "scaler.mean_ == середнє train (Δ=%.1e)" % frozen_ev["scaler_mean_delta_train"]),
        ("selection (family, C)", "вкладена CV на train: 5 outer × 3 inner", "так", "decision_record.json; правило відтворено з fold_metrics.csv"),
        ("threshold tuning", "не виконувалось; фіксований 0.5", "ні", "config.json: threshold=0.5"),
        ("interpretation (permutation importance)", "outer-validation частини train", "так", "lab06 README §8; test не використано"),
        ("freeze рішення", "decision_record.json + decision.sha256", "—", "хеш лише рішення, не моделі"),
        ("test (вибрана модель)", "114 записів, один раз для вибраної моделі", "оцінка", "test_metrics.json"),
        ("test (другий фіналіст)", "114 записів після freeze, AUDIT_ONLY", "оцінка", "test_audit_other_finalist.json"),
        ("аудит ПР7: відтворення, stability, помилки", "test + train (лише читання/оцінка)", "оцінка", "жодне рішення про модель не змінюється"),
    ]
    df = pd.DataFrame(rows, columns=["stage", "data", "used_for_decision", "evidence"])
    df.to_csv(RES / "dataflow.csv", index=False)
    return df


def stage_metrics(D, frozen):
    """Tasks 9, 10: independent metrics, dummy baseline, reproduction delta."""
    cfgj = json.loads((H / "config.json").read_text(encoding="utf-8"))
    rec = json.loads((H / "decision_record.json").read_text(encoding="utf-8"))
    Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
    refit = make_pipeline(rec["params"]["C"], rec["params"]["class_weight"]).fit(Xtr, ytr)
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    rows = []
    for name, m, X, y, split in (("dummy_prior", dummy, Xte, yte, "test"), ("frozen_joblib", frozen, Xte, yte, "test"),
                                 ("refit_on_train", refit, Xte, yte, "test"), ("dummy_prior", dummy, Xtr, ytr, "train_resubstitution"),
                                 ("frozen_joblib", frozen, Xtr, ytr, "train_resubstitution")):
        mm = A.evaluate_model(m, X, y, THR)
        rows.append({"model": name, "split": split, **{k: mm[k] for k in ("n", "positive_n", *A.METRIC_KEYS)},
                     "tn": mm["confusion_matrix"][0][0], "fp": mm["confusion_matrix"][0][1],
                     "fn": mm["confusion_matrix"][1][0], "tp": mm["confusion_matrix"][1][1]})
    ind = pd.DataFrame(rows)
    ind.to_csv(RES / "independent_metrics.csv", index=False)
    reported = json.loads((H / "test_metrics.json").read_text(encoding="utf-8"))
    mine = A.evaluate_model(frozen, Xte, yte, THR)
    rep = A.compare_metrics(reported, mine, TOL["metric"])
    rep.to_csv(RES / "reproduction_delta.csv", index=False)
    p_frozen = frozen.predict_proba(Xte)[:, 1]
    saved = pd.read_csv(OBJ / "results" / "test_predictions.csv").set_index("id").loc[D["te_ids"]]
    dp_saved = float(np.abs(saved.probability.to_numpy() - p_frozen).max())
    p_refit = refit.predict_proba(Xte)[:, 1]
    coef_delta = float(np.abs(refit.named_steps["model"].coef_ - frozen.named_steps["model"].coef_).max())
    legacy = load_legacy_predict()
    ex = json.loads((H / "inference_example.json").read_text(encoding="utf-8"))
    dp_example = float(np.abs(legacy(frozen, ex["input"]) - np.array(ex["expected_probability"])).max())
    # lab06 decision rule recomputed from its own fold table
    fm = pd.read_csv(OBJ / "results" / "fold_metrics.csv")
    a = fm[fm.branch == "automl"].sort_values("fold").ap.to_numpy()
    m_ = fm[fm.branch == "manual"].sort_values("fold").ap.to_numpy()
    dbar = float((a - m_).mean())
    eng = pd.read_csv(OBJ / "results" / "engineering.csv").set_index("branch")
    rule_ok = abs(dbar) <= rec["epsilon"] and eng.loc["automl", "size_bytes"] < eng.loc["manual", "size_bytes"] and rec["selected_branch"] == "automl" and abs(dbar - rec["paired_difference"]["mean_difference"]) <= TOL["metric"]
    ev = {
        "max_metric_delta": float(rep[rep.metric != "confusion_matrix"].abs_delta.max()),
        "cm_equal": bool(rep[rep.metric == "confusion_matrix"].within_tol.iloc[0]),
        "all_metrics_within_tol": bool(rep.within_tol.all()),
        "max_proba_delta_vs_saved": dp_saved, "max_proba_delta_vs_example": dp_example,
        "coef_delta_refit_vs_joblib": coef_delta, "max_proba_delta_refit_vs_joblib": float(np.abs(p_refit - p_frozen).max()),
        "decision_rule_reproduced": bool(rule_ok), "mean_paired_diff_recomputed": dbar,
        "size_automl": int(eng.loc["automl", "size_bytes"]), "size_manual": int(eng.loc["manual", "size_bytes"]),
        "reported_cv_ap_mean": rec["cv_ap_mean"], "reported_cv_ap_std": rec["cv_ap_std"],
        "dummy_test_ap": float(ind[(ind.model == "dummy_prior") & (ind.split == "test")].ap.iloc[0]),
        "frozen_test_ap": mine["ap"], "frozen_test_roc_auc": mine["roc_auc"], "frozen_test_cm": mine["confusion_matrix"],
        "frozen_test_recall": mine["recall"], "frozen_test_precision": mine["precision"],
        "other_finalist_audit_exists": (OBJ / "results" / "test_audit_other_finalist.json").is_file(),
        "n_models_scored_on_test": 1 + int((OBJ / "results" / "test_audit_other_finalist.json").is_file()),
        "config_threshold": cfgj["threshold"], "C": rec["params"]["C"],
    }
    return refit, dummy, ind, rep, p_frozen, ev


def stage_repeated_cv(D, refit, dummy):
    """Task 10: repeated 5x3 CV on train only for the frozen config (and dummy)."""
    cv = RepeatedStratifiedKFold(n_splits=CFG["repeated_cv"]["n_splits"], n_repeats=CFG["repeated_cv"]["n_repeats"], random_state=SEED)
    Xtr, ytr = D["Xtr"], D["ytr"]
    rows, oof_sum, oof_cnt = [], np.zeros(len(ytr)), np.zeros(len(ytr))
    ctrl_rows = []
    ctrl = make_pipeline(1.0)
    for k, (a, b) in enumerate(cv.split(Xtr, ytr)):
        rep_, fold = divmod(k, CFG["repeated_cv"]["n_splits"])
        for name, base in (("frozen_config", refit), ("dummy_prior", dummy), ("control_C1.0", ctrl)):
            m = clone(base).fit(Xtr[a], ytr[a])
            p = m.predict_proba(Xtr[b])[:, 1]
            mm = A.metrics_from_proba(ytr[b], p, THR)
            rows.append({"model": name, "repeat": rep_, "fold": fold, **{k_: mm[k_] for k_ in A.METRIC_KEYS}})
            if name == "frozen_config":
                oof_sum[b] += p
                oof_cnt[b] += 1
    folds = pd.DataFrame(rows)
    folds.to_csv(RES / "repeated_cv_folds.csv", index=False)
    summ = []
    for name, g in folds.groupby("model", sort=False):
        for met in ("ap", "roc_auc", "balanced_accuracy", "f1", "recall", "precision", "brier"):
            s = A.summarize(g[met])
            lo, hi = A.bootstrap_ci(g[met], CFG["bootstrap"]["n_boot"], CFG["bootstrap"]["alpha"], SEED)
            summ.append({"model": name, "metric": met, **s, "ci_low": lo, "ci_high": hi})
    summ = pd.DataFrame(summ)
    summ.to_csv(RES / "repeated_cv_summary.csv", index=False)
    oof = oof_sum / oof_cnt
    a_f = folds[folds.model == "frozen_config"].ap.to_numpy()
    a_c = folds[folds.model == "control_C1.0"].ap.to_numpy()
    diff = a_f - a_c
    dlo, dhi = A.bootstrap_ci(diff, CFG["bootstrap"]["n_boot"], CFG["bootstrap"]["alpha"], SEED)
    ev = {"paired_diff_C01_minus_C1_mean": float(diff.mean()), "paired_diff_ci_low": dlo, "paired_diff_ci_high": dhi,
          "paired_diff_within_eps": bool(abs(diff.mean()) <= CFG["equivalence_epsilon"] and dlo >= -CFG["equivalence_epsilon"] and dhi <= CFG["equivalence_epsilon"]),
          "cv_ap_mean": float(a_f.mean()), "cv_ap_std": float(a_f.std(ddof=1)), "cv_ap_min": float(a_f.min()), "cv_ap_max": float(a_f.max()),
          "oof_mean_p_min": float(oof.min())}
    return folds, summ, oof, ev


def stage_stability(D, frozen, refit):
    """Task 12: bootstrap retrains, disagreement, probabilities, importance ranking."""
    Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
    rng = np.random.default_rng(SEED)
    ref_p = frozen.predict_proba(Xte)[:, 1]
    ref_lab = (ref_p >= THR).astype(int)
    S = CFG["stability"]
    runs, imps, plist = [], [], []
    for k in range(S["n_retrains"]):
        idx = rng.integers(0, len(ytr), len(ytr))
        oob = np.setdiff1d(np.arange(len(ytr)), idx)
        m = clone(refit).fit(Xtr[idx], ytr[idx])
        p = m.predict_proba(Xte)[:, 1]
        lab = (p >= THR).astype(int)
        mm = A.metrics_from_proba(yte, p, THR)
        runs.append({"run": k, "n_unique_train": int(len(np.unique(idx))), "n_oob": int(len(oob)),
                     "ap": mm["ap"], "roc_auc": mm["roc_auc"], "recall": mm["recall"], "precision": mm["precision"],
                     "balanced_accuracy": mm["balanced_accuracy"], "fn": mm["confusion_matrix"][1][0], "fp": mm["confusion_matrix"][0][1],
                     "disagreement_D": A.prediction_disagreement(ref_lab, lab), "mean_abs_dp": A.mean_abs_proba_diff(ref_p, p),
                     "max_abs_dp": float(np.abs(ref_p - p).max())})
        pi = permutation_importance(m, Xtr[oob], ytr[oob], scoring="average_precision", n_repeats=S["permutation_repeats"], random_state=SEED + k)
        imps.append(pi.importances_mean)
        plist.append(p)
    runs = pd.DataFrame(runs)
    runs.to_csv(RES / "stability_runs.csv", index=False)
    P = np.array(plist)
    flips = (((P >= THR).astype(int)) != ref_lab[None, :]).mean(0)
    objs = pd.DataFrame({"id": D["te_ids"], "y": yte, "p_ref": ref_p, "p_mean": P.mean(0), "p_std": P.std(0, ddof=1),
                         "p_min": P.min(0), "p_max": P.max(0), "flip_rate": flips})
    objs.to_csv(RES / "stability_objects.csv", index=False)
    I = np.array(imps)
    pd.DataFrame(I, columns=D["names"]).assign(run=range(len(I))).to_csv(RES / "importance_runs.csv", index=False)
    ranks = pd.DataFrame(I).rank(axis=1, ascending=False).to_numpy()
    summ = pd.DataFrame({"feature": D["names"], "mean_drop_ap": I.mean(0), "std": I.std(0, ddof=1), "mean_rank": ranks.mean(0),
                         "freq_top1": (ranks == 1).mean(0), "freq_topk": (ranks <= S["top_k"]).mean(0)}).sort_values("mean_drop_ap", ascending=False, ignore_index=True)
    summ.to_csv(RES / "importance_summary.csv", index=False)
    rs = A.ranking_stability(I, S["top_k"])
    ev = {"n_retrains": S["n_retrains"], "D_mean": float(runs.disagreement_D.mean()), "D_max": float(runs.disagreement_D.max()),
          "D_min": float(runs.disagreement_D.min()), "dp_mean": float(runs.mean_abs_dp.mean()), "dp_max_obj": float(runs.max_abs_dp.max()),
          "ap_mean": float(runs.ap.mean()), "ap_std": float(runs.ap.std(ddof=1)), "ap_min": float(runs.ap.min()), "ap_max": float(runs.ap.max()),
          "recall_min": float(runs.recall.min()), "recall_max": float(runs.recall.max()),
          "n_objects_ever_flipped": int((flips > 0).sum()), "max_flip_rate": float(flips.max()),
          "top_feature": summ.feature[0], "top_feature_freq_top1": float(summ.freq_top1[0]),
          "texture_worst_freq_top1": float(summ.set_index("feature").loc["texture_worst", "freq_top1"]),
          "texture_worst_mean_rank": float(summ.set_index("feature").loc["texture_worst", "mean_rank"]),
          "top_features": summ.feature.head(5).tolist(), **{f"rank_{k}": v for k, v in rs.items()}}
    return runs, objs, summ, I, ev


def stage_errors(D, frozen, p_test, oof, objs, ctrl_model):
    """Task 13: >= 10 error / suspicious objects."""
    Xtr, ytr, Xte, yte = D["Xtr"], D["ytr"], D["Xte"], D["yte"]
    E = CFG["error_analysis"]
    t = A.select_error_objects(D["te_ids"], yte, p_test, THR, n_near=E["n_near_threshold"])
    t["set"] = "test"
    unst = objs.sort_values(["flip_rate", "p_std"], ascending=False).head(E["n_unstable"])[["id"]].copy()
    ut = pd.DataFrame({"id": unst.id.to_numpy()}).merge(pd.DataFrame({"id": D["te_ids"], "y": yte, "p": p_test}), on="id")
    ut["pred"] = (ut.p >= THR).astype(int); ut["is_error"] = ut.y != ut.pred
    ut["margin"] = (ut.p - THR).abs(); ut["conf_in_wrong"] = np.where(ut.is_error, (ut.p - ut.y).abs(), np.nan)
    ut["category"] = "unstable_under_retrain"; ut["set"] = "test"
    o = A.select_error_objects(D["tr_ids"], ytr, oof, THR, n_near=0, n_conf_err=0)
    o = o.sort_values("conf_in_wrong", ascending=False).head(E["n_confident_oof"]).copy()
    o["category"] = "confident_error_cv_oof"; o["set"] = "train_oof"
    allr = pd.concat([t, ut, o], ignore_index=True)
    # merge categories of the same object
    allr = allr.groupby(["set", "id"], as_index=False).agg(y=("y", "first"), p=("p", "first"), pred=("pred", "first"), is_error=("is_error", "first"),
                                                         margin=("margin", "first"), category=("category", lambda s: "+".join(dict.fromkeys(s))))
    smin = np.array([f["min"] for f in D["schema"]["features"]]); smax = np.array([f["max"] for f in D["schema"]["features"]])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    n_oor, maxz, geom, dummy_wrong, ctrl_wrong, pmean, pstd = [], [], [], [], [], [], []
    for _, r in allr.iterrows():
        x = D["X"][int(r.id)]
        n_oor.append(int(((x < smin) | (x > smax)).sum()))
        maxz.append(float(np.abs((x - mu) / sd).max()))
        geom.append(float(x[3] / (np.pi * x[0] ** 2)))
        dummy_wrong.append(bool(r.y != 0))  # prior baseline predicts the majority class (benign)
        ctrl_wrong.append(bool(int(ctrl_model.predict_proba(x[None, :])[0, 1] >= THR) != r.y))
    allr["n_features_outside_train_range"] = n_oor
    allr["max_abs_z"] = maxz
    allr["area_radius_ratio"] = geom
    allr["baseline_wrong"] = dummy_wrong
    allr["control_C1_wrong"] = ctrl_wrong
    allr["type"] = np.where(~allr.is_error, "correct", np.where(allr.y == 1, "FN", "FP"))
    allr = allr.merge(objs[["id", "p_std", "flip_rate"]].rename(columns={"p_std": "retrain_p_std", "flip_rate": "retrain_flip_rate"}), on="id", how="left")
    allr.loc[allr["set"] == "train_oof", ["retrain_p_std", "retrain_flip_rate"]] = np.nan
    allr.to_csv(RES / "error_objects.csv", index=False)
    te_err = allr[(allr["set"] == "test") & allr.is_error]
    ev = {"n_objects": int(len(allr)), "n_test_errors": int(len(te_err)), "test_error_ids": te_err.id.astype(int).tolist(),
          "test_error_p": te_err.p.round(4).tolist(), "test_error_types": te_err.type.tolist(),
          "n_test_errors_shared_with_baseline": int(te_err.baseline_wrong.sum()),
          "n_test_errors_shared_with_C1": int(te_err.control_C1_wrong.sum()),
          "n_oof_errors_total": int(((oof >= THR).astype(int) != ytr).sum()),
          "n_oof_fn": int(((oof < THR) & (ytr == 1)).sum()), "n_oof_fp": int(((oof >= THR) & (ytr == 0)).sum()),
          "confusion_matches_errors": None}
    return allr, ev


def stage_negative(frozen, D):
    """Task 15: negative tests for the original serving code and for the hardened one."""
    import joblib as jl
    sys.path.insert(0, str(FIXED))
    import serving as S
    legacy = load_legacy_predict()
    schema2 = json.loads((FIXED / "data_schema.json").read_text(encoding="utf-8"))
    names = [f["name"] for f in schema2["features"]]
    ex = json.loads((H / "inference_example.json").read_text(encoding="utf-8"))
    base_vec = list(ex["input"][0])
    base_rec = dict(zip(names, base_vec))
    p0 = float(legacy(frozen, [base_vec])[0])
    ia, ir = names.index("area_mean"), names.index("radius_mean")

    def legacy_call(batch):
        try:
            p = legacy(frozen, batch)
            return "accepted", [float(v) for v in p], ""
        except Exception as e:  # noqa: BLE001 - we record any rejection
            return "rejected", None, f"{type(e).__name__}: {str(e)[:70]}"

    def hard_call(recs):
        try:
            res = S.predict_records(frozen, recs, schema2)
        except Exception as e:  # noqa: BLE001
            return "rejected", None, f"{type(e).__name__}: {str(e)[:70]}"
        r = res[0]
        return r["status"], ([r["probability"]] if r["probability"] is not None else None), "; ".join(r["messages"])[:90]

    sc = []
    def add(name, expected, legacy_batch, hard_recs, same_p_expected=False):
        ls, lp, lm = legacy_call(legacy_batch)
        hs, hp, hm = hard_call(hard_recs)
        def ok(status, p):
            if status != expected:
                return False
            if same_p_expected:
                return p is not None and abs(p[0] - p0) <= TOL["probability"]
            return True
        sc.append({"scenario": name, "expected": expected, "legacy_status": ls, "legacy_p": (lp[0] if lp else None), "legacy_message": lm, "legacy_ok": ok(ls, lp),
                   "hardened_status": hs, "hardened_p": (hp[0] if hp else None), "hardened_message": hm, "hardened_ok": ok(hs, hp)})

    add("baseline_valid_record", "accepted", [base_vec], [base_rec], True)
    add("missing_feature", "rejected", [base_vec[:-1]], [{k: v for k, v in base_rec.items() if k != "fractal_dimension_worst"}])
    add("extra_feature", "rejected", [base_vec + [1.0]], [{**base_rec, "unknown_field": 1.0}])
    add("wrong_type_text", "rejected", [base_vec[:3] + ["abc"] + base_vec[4:]], [{**base_rec, "area_mean": "abc"}])
    add("numeric_string", "rejected", [[str(v) for v in base_vec]], [{k: str(v) for k, v in base_rec.items()}])
    add("none_value", "rejected", [base_vec[:3] + [None] + base_vec[4:]], [{**base_rec, "area_mean": None}])
    add("nan_value", "rejected", [base_vec[:3] + [float("nan")] + base_vec[4:]], [{**base_rec, "area_mean": float("nan")}])
    add("inf_value", "rejected", [base_vec[:3] + [float("inf")] + base_vec[4:]], [{**base_rec, "area_mean": float("inf")}])
    swapped = list(base_vec); swapped[0], swapped[1] = swapped[1], swapped[0]
    shuffled = dict(reversed(list(base_rec.items())))
    # hardened: same record, other key order -> must give the SAME probability; legacy: client sends columns in another order
    add("reordered_fields", "accepted", [swapped], [shuffled], True)
    oor = dict(base_rec); oor["radius_mean"] = base_rec["radius_mean"] * 10
    oor_vec = list(base_vec); oor_vec[ir] *= 10
    add("out_of_range_x10_radius", "accepted_with_warning", [oor_vec], [oor])
    neg = dict(base_rec); neg["radius_mean"] = -5.0
    neg_vec = list(base_vec); neg_vec[ir] = -5.0
    add("negative_radius", "rejected", [neg_vec], [neg])
    inc = dict(base_rec); inc["radius_mean"] = 10.0; inc["area_mean"] = 2000.0
    inc_vec = list(base_vec); inc_vec[ir] = 10.0; inc_vec[ia] = 2000.0
    add("domain_area_inconsistent_with_radius", "accepted_with_warning", [inc_vec], [inc])
    add("empty_batch", "rejected", [], [])
    big = CFG["domain"]["max_batch"] + 1
    add("oversized_batch", "rejected", [base_vec] * big, [base_rec] * big)
    # model substitution: valid joblib, different model
    tmp = Path(tempfile.mkdtemp())
    try:
        sub = make_pipeline(10.0).fit(D["Xtr"], D["ytr"])
        jl.dump(sub, tmp / "pipeline.joblib", compress=3)
        try:
            m2 = jl.load(tmp / "pipeline.joblib")
            leg = "accepted"; lp2 = float(legacy(m2, [base_vec])[0]); lm2 = "joblib.load has no integrity check"
        except Exception as e:  # noqa: BLE001
            leg, lp2, lm2 = "rejected", None, str(e)[:70]
        try:
            S.load_verified_model(tmp / "pipeline.joblib", FIXED / "pipeline.sha256"); hs, hm = "accepted", ""
        except S.IntegrityError as e:
            hs, hm = "rejected", f"IntegrityError: {str(e)[:60]}"
        sc.append({"scenario": "substituted_model_file", "expected": "rejected", "legacy_status": leg, "legacy_p": lp2, "legacy_message": lm2,
                   "legacy_ok": leg == "rejected", "hardened_status": hs, "hardened_p": None, "hardened_message": hm, "hardened_ok": hs == "rejected"})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    df = pd.DataFrame(sc)
    df.to_csv(RES / "negative_tests.csv", index=False)
    reorder = df[df.scenario == "reordered_fields"].iloc[0]
    oorr = df[df.scenario == "out_of_range_x10_radius"].iloc[0]
    incr = df[df.scenario == "domain_area_inconsistent_with_radius"].iloc[0]
    invalid_names = ["missing_feature", "extra_feature", "wrong_type_text", "nan_value", "inf_value", "negative_radius", "empty_batch"]
    ev = {"n_scenarios": int(len(df)), "legacy_ok": int(df.legacy_ok.sum()), "hardened_ok": int(df.hardened_ok.sum()),
          "legacy_rejects_invalid_values": bool(df[df.scenario.isin(invalid_names)].legacy_ok.all()),
          "legacy_detects_reorder": bool(reorder.legacy_ok), "legacy_reorder_p": reorder.legacy_p, "base_p": p0,
          "legacy_warns_out_of_range": bool(oorr.legacy_ok), "legacy_oor_p": oorr.legacy_p,
          "legacy_flags_domain_inconsistency": bool(incr.legacy_ok), "legacy_domain_p": incr.legacy_p,
          "legacy_accepts_numeric_string": bool(df[df.scenario == "numeric_string"].legacy_status.iloc[0] == "accepted"),
          "legacy_detects_model_substitution": bool(df[df.scenario == "substituted_model_file"].legacy_ok.iloc[0])}
    return df, ev


def stage_selection_control(D):
    """Task 11: control experiments for AI/AutoML recommendations (train only)."""
    SC = CFG["selection_control"]
    Xtr, ytr = D["Xtr"], D["ytr"]
    rows = []
    from sklearn.metrics import average_precision_score
    for s in range(SC["n_seeds"]):
        seed = SC["seed_start"] + s
        skf = StratifiedKFold(n_splits=SC["inner_folds"], shuffle=True, random_state=seed)
        scores = {C: [] for C in SC["C_grid"]}
        for a, b in skf.split(Xtr, ytr):
            for C in SC["C_grid"]:
                m = make_pipeline(C).fit(Xtr[a], ytr[a])
                scores[C].append(average_precision_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))
        mean = {C: float(np.mean(v)) for C, v in scores.items()}
        best = max(mean, key=mean.get)
        rows.append({"seed": seed, "best_C": best, **{f"ap_C{C}": mean[C] for C in SC["C_grid"]}})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "selection_control.csv", index=False)
    freq = df.best_C.value_counts(normalize=True).to_dict()
    ev = {"freq_best_C": {str(k): float(v) for k, v in freq.items()},
          "freq_C01": float(freq.get(0.1, 0.0)), "freq_C1": float(freq.get(1.0, 0.0)),
          "mean_ap_by_C": {str(C): float(df[f"ap_C{C}"].mean()) for C in SC["C_grid"]},
          "range_ap_C01_C1": float(max(abs(df["ap_C0.1"] - df["ap_C1.0"]))),
          "n_distinct_best_C": int(df.best_C.nunique())}
    return df, ev


def stage_fixed(frozen, D, p_test):
    """Task 17: evaluate the hardened package on the same test set; predictions must be identical."""
    info = build_fixed(H, CFG)
    sys.path.insert(0, str(FIXED))
    import importlib
    S = importlib.import_module("serving")
    model = S.load_verified_model(FIXED / "pipeline.joblib", FIXED / "pipeline.sha256")
    schema2 = json.loads((FIXED / "data_schema.json").read_text(encoding="utf-8"))
    names = [f["name"] for f in schema2["features"]]
    rng = np.random.default_rng(SEED)
    recs = []
    for x in D["Xte"]:
        order = rng.permutation(len(names))  # random key order for every record
        recs.append({names[j]: float(x[j]) for j in order})
    res = S.predict_records(model, recs, schema2, THR)
    p_h = np.array([r["probability"] for r in res], float)
    lab_h = np.array([r["label"] for r in res])
    lab_o = (p_test >= THR).astype(int)
    mh = A.metrics_from_proba(D["yte"], p_h, THR)
    mo = A.metrics_from_proba(D["yte"], p_test, THR)
    ev = {**info, "n_records": len(res), "statuses": pd.Series([r["status"] for r in res]).value_counts().to_dict(),
          "max_abs_dp_hardened_vs_original": float(np.abs(p_h - p_test).max()),
          "labels_identical": bool(np.array_equal(lab_h, lab_o)),
          "metrics_identical": bool(all(mh[k] == mo[k] for k in A.METRIC_KEYS) and mh["confusion_matrix"] == mo["confusion_matrix"]),
          "hardened_ap": mh["ap"], "hardened_roc_auc": mh["roc_auc"], "hardened_cm": mh["confusion_matrix"],
          "sha256_file": (FIXED / "pipeline.sha256").read_text().split()[0]}
    jdump(ev, RES / "fixed_evaluation.json")
    return ev


def stage_clean_run():
    p = RES / "clean_run.json"
    if not p.is_file():
        return {"executed": False}
    j = json.loads(p.read_text(encoding="utf-8"))
    by = {a["name"][0]: a for a in j["attempts"]}
    return {"executed": True, "A_success": by["A"]["success"], "A_pip_exit": by["A"]["pip_exit"], "A_run_exit": by["A"]["run_exit"],
            "A_pip_s": by["A"]["pip_s"], "A_n_installed": by["A"]["n_installed"], "A_core": by["A"]["installed_core"],
            "B_success": by["B"]["success"], "B_pip_exit": by["B"]["pip_exit"], "B_pip_s": by["B"]["pip_s"], "B_n_installed": by["B"]["n_installed"],
            "C_success": by["C"]["success"], "C_pip_s": by["C"]["pip_s"], "C_n_installed": by["C"]["n_installed"],
            "python": by["A"]["python"], "A_path_len": by["A"]["venv_path_length"], "B_path_len": by["B"]["venv_path_length"]}


def object_graph_modules(model):
    mods, seen = set(), set()
    def walk(o, depth=0):
        if id(o) in seen or depth > 4:
            return
        seen.add(id(o))
        mods.add(type(o).__module__.split(".")[0])
        if isinstance(o, (list, tuple)):
            for v in o: walk(v, depth + 1)
        elif hasattr(o, "__dict__"):
            for v in vars(o).values(): walk(v, depth + 1)
    walk(model)
    return sorted(mods - {"builtins"})


# ====================================================================== claims / findings
def build_claims(F):
    ct = CFG["claim_thresholds"]
    C = []
    def claim(cid, text, source, check, criteria, evidence_keys, level):
        C.append({"id": cid, "claim": text, "source": source, "independent_check": check, "criteria": criteria, "level": level})
    claim("C-01", "На закритому test (114 записів) модель має AP=0.9966, ROC-AUC=0.9977, CM=[[72,0],[2,40]] за порогу 0.5.",
          "lab06/handoff/test_metrics.json; MODEL_CARD.md",
          "Перерахунок метрик із joblib на test-ids за split_ids.csv; порівняння з файлом; перевірка збережених ймовірностей",
          [{"name": "max |Δметрики| ≤ 1e-9", "key": "max_metric_delta", "op": "le", "value": TOL["metric"]},
           {"name": "матриця помилок збігається", "key": "cm_equal", "op": "true"},
           {"name": "max |Δp| до test_predictions.csv ≤ 1e-12", "key": "max_proba_delta_vs_saved", "op": "le", "value": TOL["probability"]}], None, 3)
    claim("C-02", "Test не використовувався для пошуку чи вибору; рішення заморожене (хеш) до відкриття test.",
          "handoff/config.json; decision_record.json; MODEL_CARD.md",
          "Прапорці конфігурації; відтворення правила вибору з fold_metrics.csv; scaler.mean_ = середнє train; порядок рядків у run_experiment.py; історія git",
          [{"name": "прапорці test_used_* = false", "key": "flags_false", "op": "true"},
           {"name": "правило вибору відтворено", "key": "decision_rule_reproduced", "op": "true"},
           {"name": "scaler навчено на train (Δ ≤ 1e-9)", "key": "scaler_mean_delta_train", "op": "le", "value": TOL["coefficient"]},
           {"name": "коефіцієнти після refit на train збігаються (Δ ≤ 1e-9)", "key": "coef_delta_refit_vs_joblib", "op": "le", "value": TOL["coefficient"]},
           {"name": "freeze передує test у коді", "key": "decision_before_test_in_code", "op": "true"},
           {"name": "час створення рішення доведено незалежно від файлів (різні коміти)", "key": "timing_independently_provable", "op": "true"}], None, 3)
    claim("C-03", "Модель узагальнює на нові дані: CV AP 0.9942 ± 0.0058, test AP 0.9966, значно вище baseline.",
          "lab06/README.md §6, §10; MODEL_CARD.md",
          "Повторна CV 5×3 лише на train; DummyClassifier; leakage scan; перетин із попередніми роботами курсу",
          [{"name": f"CV AP ≥ baseline AP + {ct['min_ap_margin_over_baseline']}", "key": "cv_margin_over_baseline", "op": "ge", "value": ct["min_ap_margin_over_baseline"]},
           {"name": "|повторна CV AP − заявлена| ≤ ε", "key": "cv_ap_diff_to_reported", "op": "le", "value": CFG["equivalence_epsilon"]},
           {"name": "test AP не нижче min повторної CV", "key": "test_ap_minus_cv_min", "op": "ge", "value": 0.0},
           {"name": "target-проксі відсутні (|r|<0.9)", "key": "n_corr_signals", "op": "eq", "value": 0},
           {"name": "test незалежний від попередніх робіт курсу (перетин 0)", "key": "lab06_test_in_lab05_train", "op": "eq", "value": 0}], None, 3)
    claim("C-04", "AutoML-кандидат (C=0.1) і Manual (C=1.0) практично еквівалентні: |ΔAP| < ε=0.01.",
          "lab06/handoff/decision_record.json (reason); README §6, §9",
          "Парна різниця AP на повторній CV 5×3 train з bootstrap-інтервалом",
          [{"name": "|середня ΔAP| ≤ ε", "key": "paired_diff_abs", "op": "le", "value": CFG["equivalence_epsilon"]},
           {"name": "інтервал ΔAP повністю в [−ε, ε]", "key": "paired_diff_within_eps", "op": "true"}], None, 4)
    claim("C-05", "Пакет передачі повний і його цілісність забезпечена SHA-256 рішення.",
          "lab06/README.md §10; MODEL_CARD.md",
          "Інвентаризація файлів; перевірка decision.sha256; пошук хеш-суми pipeline.joblib у пакеті",
          [{"name": "усі 16 елементів інвентаризації присутні", "key": "inventory_complete", "op": "true"},
           {"name": "decision.sha256 збігається з рішенням", "key": "decision_hash_valid", "op": "true"},
           {"name": "SHA-256 pipeline.joblib зафіксовано в пакеті", "key": "model_hash_recorded", "op": "true"}], None, 2)
    claim("C-06", "inference.py перевіряє контракт входу: 30 ознак у порядку схеми, скінченні, невід'ємні.",
          "lab06/handoff/inference.py; data_schema.json; MODEL_CARD.md",
          "Негативні сценарії (13) для оригінального predict() у контрольному експерименті",
          [{"name": "некоректні значення/форма відхиляються", "key": "legacy_rejects_invalid_values", "op": "true"},
           {"name": "інший порядок полів виявляється", "key": "legacy_detects_reorder", "op": "true"},
           {"name": "значення поза train range супроводжується warning", "key": "legacy_warns_out_of_range", "op": "true"},
           {"name": "предметна неузгодженість (площа/радіус) виявляється", "key": "legacy_flags_domain_inconsistency", "op": "true"}], None, 4)
    claim("C-07", "Inference потребує лише sklearn/numpy/joblib; flaml/xgboost/lightgbm потрібні тільки для пошуку, а requirements.txt встановлюється.",
          "lab06/README.md §10; handoff/requirements.txt",
          "Скан класів у об'єкті моделі; чисті venv: повний requirements (довгий і короткий шлях) і мінімальний",
          [{"name": "об'єкт моделі використовує лише sklearn/numpy/scipy", "key": "graph_only_sklearn", "op": "true"},
           {"name": "повний requirements.txt встановлено без помилок", "key": "clean_full_install_ok", "op": "true"},
           {"name": "inference працює у venv лише з мінімальними залежностями", "key": "clean_minimal_ok", "op": "true"}], None, 3)
    claim("C-08", "Дані: 569 записів, 212 malignant, split 455/114 stratified, без дублікатів/пропусків; схема містить train-діапазони; хеш сирих даних відтворюється.",
          "lab06/handoff/data_provenance.json; data_schema.json; README §2",
          "Перерахунок із load_breast_cancer; зіставлення з split_ids.csv і схемою",
          [{"name": "n=569, positives=212", "key": "data_counts_ok", "op": "true"},
           {"name": "перетин ID = 0 і ID покривають набір", "key": "ids_ok", "op": "true"},
           {"name": "дублікатів і NaN немає", "key": "no_dup_nan", "op": "true"},
           {"name": "min/max схеми = діапазон train", "key": "schema_range_equals_train_range", "op": "true"},
           {"name": "SHA-256 сирих даних збігається", "key": "raw_sha256_matches_provenance", "op": "true"}], None, 3)
    claim("C-09", "Єдина узгоджена вершина важливості ознак — texture_worst (permutation importance).",
          "lab06/README.md §8", "Permutation importance фінальної конфігурації на OOB-вибірках 20 bootstrap-перенавчань (train)",
          [{"name": f"texture_worst — top-1 принаймні у {int(ct['top1_frequency']*100)}% повторів", "key": "texture_worst_freq_top1", "op": "ge", "value": ct["top1_frequency"]},
           {"name": f"середній top-5 Jaccard ≥ {ct['topk_jaccard']}", "key": "rank_mean_topk_jaccard", "op": "ge", "value": ct["topk_jaccard"]}], None, 4)
    claim("C-10", "Test використано один раз.",
          "lab06/README.md §10 («Test (один раз, поріг 0.5)»)",
          "Підрахунок моделей, оцінених на test, за артефактами results/",
          [{"name": "на test оцінено не більше однієї моделі", "key": "n_models_scored_on_test", "op": "le", "value": 1},
           {"name": "додаткову оцінку розкрито й позначено AUDIT_ONLY", "key": "second_evaluation_disclosed", "op": "true"}], None, 2)
    return C


def build_findings(F):
    out = []
    def add(cond, *a, **k):
        if cond:
            out.append(A.build_finding(*a, **k))
    add(not F["model_hash_recorded"],
        "Немає контрольної суми pipeline.joblib; decision.sha256 охоплює лише рішення, тому підміну або пошкодження моделі виявити неможливо.",
        f"Пошук SHA-256 {F['model_sha256'][:12]}… у всіх текстових файлах handoff: не знайдено. У сценарії підміни файлу на іншу валідну модель (C=10) оригінальний joblib.load приймає її без помилки ({F['legacy_substitution']}); joblib/pickle виконує код під час завантаження.",
        "high", "Записати SHA-256 моделі й перевіряти байти ДО joblib.load; прив'язати хеш моделі до decision_record.", status="fixed")
    add(not F["legacy_detects_reorder"],
        "Inference приймає лише позиційний масив без імен ознак: переставлені колонки не виявляються й дають інший прогноз без попередження.",
        f"Обмін місцями radius_mean і texture_mean: p={F['legacy_reorder_p']:.4f} замість {F['base_p']:.4f} (статус accepted). Схема має order_required=true, але порядок не перевіряється.",
        "high", "Приймати записи з іменами ознак, відхиляти відсутні/зайві поля, упорядковувати за схемою.", status="fixed")
    add(not F["legacy_warns_out_of_range"] or not F["legacy_flags_domain_inconsistency"],
        "Немає попередження про значення поза train-діапазоном і предметну неузгодженість (площа/радіус); схема містить min/max, але вони не застосовуються.",
        f"radius_mean×10: p={F['legacy_oor_p']:.4f}, accepted без попередження; radius_mean=10 при area_mean=2000 (відношення до π·r² ≈ {2000/(np.pi*100):.1f}): p={F['legacy_domain_p']:.4f}, accepted. MODEL_CARD сам радить стежити за дрейфом.",
        "medium", "Повертати accepted_with_warning за виходу за діапазон train і за порушення |area/(π r²) − 1| ≤ 0.15; явно описати політику.", status="fixed")
    cr = F["clean_run"]
    if cr.get("executed"):
        add((not cr["A_success"]) or (not cr["B_success"]),
            "Повний requirements.txt із handoff не встановлюється чисто: він вимагає flaml, xgboost, lightgbm і jupyter, хоча inference потребує лише sklearn-стеку; на довгому шляху встановлення падає.",
            f"Чисте venv: спроба A (довгий шлях, {cr['A_path_len']} символів) pip exit={cr['A_pip_exit']}, але inference у такому частково встановленому venv завершився exit={cr['A_run_exit']} (numpy/sklearn були встановлені раніше за збій); спроба B (короткий шлях, {cr['B_path_len']}) pip exit={cr['B_pip_exit']}; мінімальний набір (C) success={cr['C_success']}, {cr['C_n_installed']} пакетів проти {cr['B_n_installed']}.",
            "medium", "Розділити залежності: мінімальний inference-requirements і окремий файл для відтворення пошуку.", status="fixed")
        add(cr["A_success"] and cr["B_success"],
            "requirements.txt для inference надлишковий (flaml, xgboost, lightgbm, jupyter).",
            f"Встановлення успішне, але {cr['B_n_installed']} пакетів проти {cr['C_n_installed']} у мінімальному наборі.", "low",
            "Окремий мінімальний requirements для inference.", status="fixed")
    add(F["lab06_test_in_lab05_train"] > 0,
        "Test ЛР6 не є «свіжим» щодо попередніх робіт курсу: більшість test-записів були у train ЛР5; набір публічний і багаторазово використаний.",
        f"{F['lab06_test_in_lab05_train']} із {F['test_n']} test-ID ЛР6 були у train ЛР5 (перераховано за results/split_ids.csv ЛР5, лише ID; збігається з data_provenance.json). Модель ЛР5 була без вчителя і не використовувала мітки, тож пряме зараження не доведено, але незалежність test обмежена.",
        "medium", "Для заяв про узагальнення потрібен зовнішній набір; у висновках позначати test як внутрішню перевірку.", status="accepted_risk")
    add(F["test_recall_ci_width"] > 0.1,
        "Test малий (42 malignant): метрики за порогом 0.5 мають широкий інтервал, а 2 пропущені malignant становлять 4.8% recall.",
        f"Bootstrap 95% для recall: [{F['recall_ci'][0]:.3f}, {F['recall_ci'][1]:.3f}]; AP: [{F['ap_ci'][0]:.4f}, {F['ap_ci'][1]:.4f}]. FN: ID {F['test_error_ids']} з p={F['test_error_p']}. Пороги 0.5 не підбирався під ціну FN.",
        "medium", "Не трактувати test як доказ рівня recall; для застосувань із дорогим FN окремо обирати поріг на CV, не на test.", status="accepted_risk")
    add(not F["group_or_patient_id_available"],
        "Немає ідентифікатора пацієнта/групи та часу: залежність між записами одного пацієнта виключити неможливо, тож оцінка узагальнення може бути завищена.",
        "sklearn.load_breast_cancer не містить patient id; split випадковий stratified. Перевірено лише дублікати та близькість test→train (мін. відстань %.3f у стандартизованому просторі)." % F["min_test_to_train_distance"],
        "medium", "Описати як обмеження; для реального застосування потрібен груповий split.", status="accepted_risk")
    add(F["timing_independently_provable"] is False,
        "Порядок «freeze → test» доведено лише побічно: рішення, модель і test-метрики потрапили в один коміт, а хеш підтверджує незмінність байтів, не час.",
        f"git: файли handoff додано у {F['n_add_commits']} коміт(и); статичний порядок рядків у run_experiment.py правильний, scaler.mean_ збігається з train (Δ={F['scaler_mean_delta_train']:.1e}).",
        "low", "Для майбутніх робіт фіксувати хеш рішення в окремому, раніше створеному коміті або журналі.", status="documented")
    add(F["n_models_scored_on_test"] > 1,
        "Заява «test використано один раз» неточна: на test також оцінено другого фіналіста (AUDIT_ONLY) після freeze.",
        "results/test_audit_other_finalist.json (AUDIT_ONLY=true): Manual C=1.0, AP=0.9943. Вибір не змінювався, README розкриває це в §10.",
        "low", "Формулювати: «для вибраної моделі один раз; другий фіналіст — лише аудит».", status="documented")
    add(F["freq_C01"] < 1.0,
        f"Вибір C=0.1 є практично довільним: на 20 seeds внутрішньої 3-fold CV (train) C=0.1 обирається у {F['freq_C01']*100:.0f}% випадків, C=1.0 — у {F['freq_C1']*100:.0f}%.",
        f"selection_control.csv: частоти найкращого C = {F['freq_best_C']}; різниця AP між C=0.1 і C=1.0 на повторній CV: {F['paired_diff_mean']:+.4f} (інтервал [{F['paired_diff_lo']:+.4f}, {F['paired_diff_hi']:+.4f}]) — в межах ε. Це підтверджує чесне формулювання lab06 про довільність tie-break.",
        "low", "Не інтерпретувати C=0.1 як оптимальний; якість еквівалентна C=1.0.", status="documented")
    for i, f in enumerate(out, 1):
        f["id"] = f"F-{i:02d}"
    return out


# ====================================================================== figures
def fig_save(fig, name):
    fig.tight_layout()
    fig.savefig(FIG / name, dpi=140)
    plt.close(fig)


def make_figures(D, corr, ind, folds, summ, objs, runs, imp_summ, errs, neg, claims_df, find_df, sel, p_test, F):
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    # 1 correlations
    top = corr.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top.feature, top.abs_r, color=BLUE)
    ax.axvline(CFG["leakage"]["corr_signal_threshold"], color=RED, ls="--", label=f"поріг сигналу |r|={CFG['leakage']['corr_signal_threshold']}")
    ax.set_xlabel("|corr(ознака, target)| на train"); ax.set_title("Leakage scan: 12 найсильніших кореляцій з target")
    ax.set_xlim(0, 1); ax.legend(loc="lower right"); fig_save(fig, "01_leakage_correlations.png")
    # 2 metrics vs baseline
    t = ind[ind.split == "test"].set_index("model")
    mets = ["ap", "roc_auc", "balanced_accuracy", "recall", "precision"]
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    w = 0.38; x = np.arange(len(mets))
    ax.bar(x - w / 2, [t.loc["dummy_prior", m] for m in mets], w, color=GRAY, label="DummyClassifier (prior)")
    ax.bar(x + w / 2, [t.loc["frozen_joblib", m] for m in mets], w, color=BLUE, label="Заморожена модель (joblib)")
    for i, m in enumerate(mets):
        ax.text(i + w / 2, t.loc["frozen_joblib", m] + 0.01, f"{t.loc['frozen_joblib', m]:.3f}", ha="center", fontsize=8)
        ax.text(i - w / 2, t.loc["dummy_prior", m] + 0.01, f"{t.loc['dummy_prior', m]:.3f}", ha="center", fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels(mets); ax.set_ylim(0, 1.12); ax.set_ylabel("значення метрики на test")
    ax.set_title("Незалежно перераховані метрики: baseline проти моделі"); ax.legend(loc="lower left"); fig_save(fig, "02_metrics_vs_baseline.png")
    # 3 repeated cv
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for k, (name, col) in enumerate((("frozen_config", BLUE), ("control_C1.0", ORANGE))):
        v = folds[folds.model == name].ap.to_numpy()
        ax.scatter(np.full_like(v, k) + np.random.default_rng(1).uniform(-0.12, 0.12, len(v)), v, color=col, alpha=0.7, label=f"{name}: 15 fold-оцінок")
        s = summ[(summ.model == name) & (summ.metric == "ap")].iloc[0]
        ax.errorbar([k], [s["mean"]], yerr=[[s["mean"] - s.ci_low], [s.ci_high - s["mean"]]], color="black", capsize=6, lw=2)
    ax.axhline(F["dummy_test_ap"], color=GRAY, ls="--", label=f"dummy AP={F['dummy_test_ap']:.3f}")
    ax.axhline(F["frozen_test_ap"], color=GREEN, ls=":", label=f"test AP={F['frozen_test_ap']:.4f}")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["frozen (C=0.1)", "контроль (C=1.0)"]); ax.set_ylabel("AP на валідаційних folds")
    ax.set_title("Repeated CV 5×3 (train): розподіл AP та bootstrap-інтервал середнього"); ax.legend(fontsize=8, loc="center right")
    ax.set_ylim(0.3, 1.02); fig_save(fig, "03_repeated_cv.png")
    # 4 confusion
    fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
    for ax, name, title in zip(axs, ("dummy_prior", "frozen_joblib"), ("Dummy (prior)", "Заморожена модель")):
        r = t.loc[name]; cm = np.array([[r.tn, r.fp], [r.fn, r.tp]])
        ax.imshow(cm, cmap="Blues")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["pred benign", "pred malignant"]); ax.set_yticks([0, 1]); ax.set_yticklabels(["true benign", "true malignant"]); ax.set_title(title)
    fig.suptitle("Confusion matrix на test (поріг 0.5)"); fig_save(fig, "04_confusion_matrices.png")
    # 5 stability
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    axs[0].plot(runs.run, runs.ap, "o-", color=BLUE, label="AP на test"); axs[0].plot(runs.run, runs.recall, "s--", color=ORANGE, label="recall на test")
    axs[0].axhline(F["frozen_test_ap"], color=GREEN, ls=":", label="еталон: заморожена модель (AP)")
    axs[0].set_xlabel("номер bootstrap-перенавчання"); axs[0].set_ylabel("метрика"); axs[0].set_title("Метрики між перенавчаннями"); axs[0].legend(fontsize=8)
    axs[1].bar(runs.run, runs.disagreement_D, color=BLUE, label="D (частка розбіжних міток)")
    ax2 = axs[1].twinx(); ax2.plot(runs.run, runs.mean_abs_dp, "o-", color=RED, label="середній |Δp|"); ax2.spines["right"].set_visible(True)
    axs[1].set_xlabel("номер bootstrap-перенавчання"); axs[1].set_ylabel("D"); ax2.set_ylabel("середній |Δp|"); axs[1].set_title("Розбіжність прогнозів з еталоном")
    h1, l1 = axs[1].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels(); axs[1].legend(h1 + h2, l1 + l2, fontsize=8, loc="upper right")
    fig_save(fig, "05_stability.png")
    # 6 importance
    top = imp_summ.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.barh(top.feature, top.mean_drop_ap, xerr=top["std"], color=BLUE, capsize=3)
    ax.set_xlabel("середнє падіння AP після перестановки (OOB, ± std за 20 перенавчаннями)"); ax.set_title("Permutation importance: top-10 і розкид")
    fig_save(fig, "06_permutation_importance.png")
    # 7 error analysis
    o = np.argsort(p_test)
    fig, ax = plt.subplots(figsize=(8, 4.3))
    yy = D["yte"][o]; pp = p_test[o]; ids = D["te_ids"][o]
    ax.scatter(np.arange(len(pp))[yy == 0], pp[yy == 0], color=BLUE, s=14, label="benign (y=0)")
    ax.scatter(np.arange(len(pp))[yy == 1], pp[yy == 1], color=ORANGE, s=14, label="malignant (y=1)")
    ax.axhline(THR, color="black", ls="--", label="поріг 0.5")
    err = (pp >= THR).astype(int) != yy
    ax.scatter(np.arange(len(pp))[err], pp[err], facecolors="none", edgecolors=RED, s=120, lw=2, label="помилка")
    for k in np.where(err)[0]:
        ax.annotate(f"ID {ids[k]}", (k, pp[k]), textcoords="offset points", xytext=(-10, 10), color=RED, fontsize=8)
    ax.set_xlabel("test-записи, відсортовані за ймовірністю"); ax.set_ylabel("P(malignant)"); ax.set_title("Аналіз помилок: ймовірності на test та пропущені malignant"); ax.legend(fontsize=8)
    fig_save(fig, "07_error_analysis.png")
    # 8 negative tests
    fig, ax = plt.subplots(figsize=(8, 5.2))
    cols = {"accepted": "#9ae6b4", "accepted_with_warning": "#faf089", "rejected": "#fc8181"}
    for i, r in neg.iterrows():
        for j, (lab, st, ok) in enumerate((("оригінал", r.legacy_status, r.legacy_ok), ("виправлена версія", r.hardened_status, r.hardened_ok))):
            ax.add_patch(plt.Rectangle((j, -i - 1), 1, 1, color=cols.get(st, "white"), ec="white"))
            ax.text(j + 0.5, -i - 0.5, f"{st}{'' if ok else '  ✗'}", ha="center", va="center", fontsize=8)
    ax.set_xlim(0, 2); ax.set_ylim(-len(neg), 0); ax.set_xticks([0.5, 1.5]); ax.set_xticklabels(["оригінал (lab06)", "виправлена версія"]); ax.xaxis.tick_top()
    ax.set_yticks([-i - 0.5 for i in range(len(neg))]); ax.set_yticklabels(neg.scenario, fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title("Негативні сценарії: ✗ — поведінка відрізняється від очікуваної", pad=28); fig_save(fig, "08_negative_tests.png")
    # 9 claims & findings
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.8))
    vc = claims_df.verdict.value_counts().reindex(A.CLAIM_VERDICTS, fill_value=0)
    axs[0].bar(range(4), vc.values, color=[GREEN, ORANGE, RED, GRAY]); axs[0].set_xticks(range(4)); axs[0].set_xticklabels(["підтв.", "частково", "спрост.", "не перев."], fontsize=8)
    axs[0].set_ylabel("кількість тверджень"); axs[0].set_title("Вердикти реєстру тверджень")
    sv = find_df.severity.value_counts().reindex(A.SEVERITIES, fill_value=0)
    axs[1].bar(range(4), sv.values, color=["#742a2a", RED, ORANGE, GRAY]); axs[1].set_xticks(range(4)); axs[1].set_xticklabels(A.SEVERITIES)
    axs[1].set_ylabel("кількість знахідок"); axs[1].set_title("Знахідки за критичністю")
    fig_save(fig, "09_claims_findings.png")
    # 10 selection control
    fig, ax = plt.subplots(figsize=(7, 4))
    for C, col in zip(CFG["selection_control"]["C_grid"], (GRAY, BLUE, ORANGE, GREEN)):
        ax.plot(sel.seed, sel[f"ap_C{C}"], "o-", color=col, ms=3, label=f"C={C}")
    ax.set_xlabel("seed внутрішньої 3-fold CV"); ax.set_ylabel("середній AP на folds (train)"); ax.set_title("Контроль вибору C: різниці менші за шум між seeds"); ax.legend(fontsize=8, ncol=2)
    fig_save(fig, "10_selection_control.png")


# ====================================================================== main
def main():
    t0 = time.time()
    for d in (RES, FIG, ART):
        d.mkdir(exist_ok=True)
    fro_df, fro = stage_frozen()
    inv = stage_inventory()
    D = stage_data()
    frozen = joblib.load(H / "pipeline.joblib")
    corr, lk_checks, lk = stage_leakage(D, frozen)
    dataflow = stage_dataflow(lk)
    refit, dummy, ind, rep, p_test, mt = stage_metrics(D, frozen)
    ctrl_model = make_pipeline(1.0).fit(D["Xtr"], D["ytr"])
    folds, summ, oof, rcv = stage_repeated_cv(D, refit, dummy)
    runs, objs, imp_summ, I, st = stage_stability(D, frozen, refit)
    errs, er = stage_errors(D, frozen, p_test, oof, objs, ctrl_model)
    stage_fixed_info = stage_fixed(frozen, D, p_test)
    neg, ng = stage_negative(frozen, D)
    sel, sc = stage_selection_control(D)
    cr = stage_clean_run()
    ci_rec = A.bootstrap_metric_ci(D["yte"], p_test, "recall", CFG["bootstrap"]["n_boot_test"], seed=SEED)
    ci_ap = A.bootstrap_metric_ci(D["yte"], p_test, "ap", CFG["bootstrap"]["n_boot_test"], seed=SEED)
    ci_auc = A.bootstrap_metric_ci(D["yte"], p_test, "roc_auc", CFG["bootstrap"]["n_boot_test"], seed=SEED)
    jdump({"recall": ci_rec, "ap": ci_ap, "roc_auc": ci_auc, "n_boot": CFG["bootstrap"]["n_boot_test"]}, RES / "test_bootstrap_ci.json")
    graph = object_graph_modules(frozen)
    # ---- facts for claims and findings
    cm = mt["frozen_test_cm"]
    er["confusion_matches_errors"] = bool(er["n_test_errors"] == cm[0][1] + cm[1][0])
    F = {**D["facts"], **lk, **mt, **rcv, **st, **er, **ng, **sc, **{"fixed_" + k: v for k, v in stage_fixed_info.items()}}
    F.update({
        "inventory_complete": bool(inv.present.all()), "graph_modules": graph, "graph_only_sklearn": set(graph) <= {"sklearn", "numpy", "scipy", "joblib"},
        "clean_run": cr,
        "clean_full_install_ok": (cr.get("A_success") and cr.get("B_success")) if cr.get("executed") else None,
        "clean_minimal_ok": cr.get("C_success") if cr.get("executed") else None,
        "timing_independently_provable": fro["handoff_files_added_in_n_commits"] > 1,
        "n_add_commits": fro["handoff_files_added_in_n_commits"],
        "cv_margin_over_baseline": rcv["cv_ap_mean"] - mt["dummy_test_ap"],
        "cv_ap_diff_to_reported": abs(rcv["cv_ap_mean"] - mt["reported_cv_ap_mean"]),
        "test_ap_minus_cv_min": mt["frozen_test_ap"] - rcv["cv_ap_min"],
        "paired_diff_abs": abs(rcv["paired_diff_C01_minus_C1_mean"]),
        "paired_diff_mean": rcv["paired_diff_C01_minus_C1_mean"], "paired_diff_lo": rcv["paired_diff_ci_low"], "paired_diff_hi": rcv["paired_diff_ci_high"],
        "data_counts_ok": D["facts"]["n"] == 569 and D["facts"]["positives"] == 212,
        "ids_ok": D["facts"]["split_overlap"] == 0 and D["facts"]["ids_cover_dataset"] and D["facts"]["ids_unique"],
        "no_dup_nan": D["facts"]["duplicate_rows_all"] == 0 and D["facts"]["nan_count"] == 0,
        "second_evaluation_disclosed": "AUDIT_ONLY" in (OBJ / "results" / "test_audit_other_finalist.json").read_text(),
        "test_recall_ci_width": ci_rec[1] - ci_rec[0], "recall_ci": ci_rec, "ap_ci": ci_ap,
        "legacy_substitution": "p=%.4f" % neg[neg.scenario == "substituted_model_file"].legacy_p.iloc[0],
    })
    # ---- claims
    claims = build_claims(F)
    crow = []
    for c in claims:
        v = A.verify_claim(c, F)
        evtxt = "; ".join(f"{r['name']}: {'ДА' if r['passed'] else 'НІ'}" + (f" ({r['measured']:.4g})" if isinstance(r["measured"], (int, float)) and not isinstance(r["measured"], bool) else "") for r in v["criteria_results"])
        crow.append({"id": c["id"], "claim": c["claim"], "source": c["source"], "independent_check": c["independent_check"],
                     "evidence_level": c["level"], "acceptance_criteria": " | ".join(x["name"] for x in c["criteria"]),
                     "evidence": evtxt, "criteria_passed": f"{v['passed']}/{v['total']}", "verdict": v["verdict"]})
    claims_df = pd.DataFrame(crow)
    claims_df.to_csv(RES / "claims.csv", index=False)
    # ---- findings
    findings = build_findings(F)
    find_df = pd.DataFrame(findings)[["id", "severity", "problem", "evidence", "recommendation", "status"]]
    find_df.to_csv(RES / "findings.csv", index=False)
    verdict_orig = A.decide_verdict([{**f, "status": "open"} for f in findings])
    verdict_fixed = A.decide_verdict(findings)
    verdict = {"verdict": verdict_orig, "verdict_scope": "початкове рішення lab06 (до виправлень)",
               "verdict_after_fixes": verdict_fixed,
               "verdict_options": list(A.AUDIT_VERDICTS),
               "rule": CFG["verdict_rule"],
               "frozen_commit": CFG["object"]["frozen_commit"], "claims": claims_df.verdict.value_counts().to_dict(),
               "findings_by_severity": find_df.severity.value_counts().to_dict(),
               "findings_by_status": find_df.status.value_counts().to_dict(),
               "basis": {"claims_ids": claims_df.id.tolist(), "finding_ids": find_df.id.tolist(),
                         "blocking_findings": [f["id"] for f in findings if f["severity"] in ("critical", "high")]},
               "intended_use": "навчальне ранжування записів FNA для експерта; не діагностика",
               "reproduced_test": {"ap": mt["frozen_test_ap"], "roc_auc": mt["frozen_test_roc_auc"], "confusion_matrix": mt["frozen_test_cm"]}}
    jdump(verdict, RES / "verdict.json")
    # ---- AI recommendation checks (task 11)
    ai = pd.DataFrame([
        {"id": "R-1", "recommendation": "Tie-break за розміром серіалізованого pipeline вибирає AutoML (lab06 AI_USAGE §4)", "risk": "Вибір за 3 байти може приховати реальну різницю якості",
         "control_test": "Парна різниця AP C=0.1 і C=1.0 на repeated CV 5×3 (train)",
         "actual_result": f"розміри {F['size_automl']} vs {F['size_manual']} байт; ΔAP={F['paired_diff_mean']:+.4f}, інтервал [{F['paired_diff_lo']:+.4f}, {F['paired_diff_hi']:+.4f}]",
         "decision": "прийнято: різниця в межах ε, але вибір довільний"},
        {"id": "R-2", "recommendation": "Стабільність пошуку: три seeds дають те саме сімейство lrl2 (AI_USAGE §3)", "risk": "Стабільне сімейство не означає стабільного C",
         "control_test": f"{CFG['selection_control']['n_seeds']} seeds внутрішньої 3-fold CV на train за сіткою C",
         "actual_result": f"частоти найкращого C: {F['freq_best_C']}; різних значень: {F['n_distinct_best_C']}",
         "decision": "змінено: сімейство стабільне, але C — ні (знахідка про довільність вибору C)"},
        {"id": "R-3", "recommendation": "Хеш рішення + одноразовий запуск гарантують, що test не впливав на вибір (AI_USAGE §2)", "risk": "Хеш доводить байти, не час і не зв'язок із моделлю",
         "control_test": "scaler.mean_ vs середнє train/усіх даних; refit на train → коефіцієнти; відтворення правила вибору; порядок у коді",
         "actual_result": f"Δscaler(train)={F['scaler_mean_delta_train']:.1e}, Δscaler(all)={F['scaler_mean_delta_all']:.1e}; Δcoef={F['coef_delta_refit_vs_joblib']:.1e}; правило відтворено={F['decision_rule_reproduced']}; handoff у {F['n_add_commits']} коміті",
         "decision": "прийнято частково: модель відповідає train-only протоколу; час не доведено"},
        {"id": "R-4", "recommendation": "Закріпити у requirements flaml/xgboost/lightgbm, бо FLAML їх імпортує; inference потребує лише sklearn (AI_USAGE §6, README §10)", "risk": "Надлишкові залежності ламають чисте встановлення",
         "control_test": "Чисті venv: повний і мінімальний requirements; скан об'єкта моделі",
         "actual_result": (f"об'єкт моделі: {graph}; A pip exit={cr.get('A_pip_exit')}, B pip exit={cr.get('B_pip_exit')}, мінімальний набір success={cr.get('C_success')}" if cr.get("executed") else "clean_run.json відсутній"),
         "decision": "змінено: для inference достатньо мінімального набору"},
    ])
    ai.to_csv(RES / "ai_recommendation_checks.csv", index=False)
    # ---- figures
    make_figures(D, corr, ind, folds, summ, objs, runs, imp_summ, errs, neg, claims_df, find_df, sel, p_test, F)
    # ---- summary & manifest
    summary = {k: F[k] for k in ("max_metric_delta", "cm_equal", "dummy_test_ap", "frozen_test_ap", "frozen_test_roc_auc", "frozen_test_cm", "cv_ap_mean", "cv_ap_std",
                                  "cv_ap_min", "cv_ap_max", "D_mean", "D_max", "dp_mean", "n_objects", "n_test_errors", "test_error_ids", "model_sha256",
                                  "legacy_ok", "hardened_ok", "n_scenarios")}
    summary.update(elapsed_s=round(time.time() - t0, 1), python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, frozen_info=fro,
                   fixed_evaluation=stage_fixed_info, clean_run=cr, graph_modules=graph, recall_ci=ci_rec, ap_ci=ci_ap, roc_auc_ci=ci_auc,
                   importance_top=st["top_features"], texture_worst_freq_top1=st["texture_worst_freq_top1"], rank_stability={k: v for k, v in F.items() if k.startswith("rank_")},
                   selection_control=sc, repeated_cv_paired=rcv)
    jdump(summary, RES / "summary.json")
    files = sorted([p for d in (RES, FIG) for p in d.glob("*") if p.is_file()]) + [ROOT / "audit_core.py", ROOT / "pr07_audit.py", ROOT / "build_fixed.py", ROOT / "clean_run.py",
             ROOT / "configs" / "audit_config.json"] + sorted(p for p in FIXED.glob("*") if p.is_file())
    man = {"generated_for": "practical07", "files": A.file_manifest(files, ROOT),
           "frozen_lab06_handoff": {r.file: {"sha256": r.sha256, "bytes": int(r.bytes)} for r in fro_df.itertuples()},
           "frozen_commit": CFG["object"]["frozen_commit"]}
    jdump(man, ART / "file_manifest.json")
    print(f"audit finished in {time.time() - t0:.1f}s; claims={claims_df.verdict.value_counts().to_dict()}; findings={find_df.severity.value_counts().to_dict()}; verdict={verdict_orig}")


if __name__ == "__main__":
    main()
