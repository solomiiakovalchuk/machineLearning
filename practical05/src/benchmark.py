"""Вимірювання якості, часу, розміру й пропускної здатності кандидатів.

Це "дорога" частина (run_benchmark.py). Результат - таблиця benchmark_results.csv,
яку далі читає model_selector без жодного коду конкретних моделей.
"""
from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
import scipy
import sklearn
import threadpoolctl
from sklearn.base import clone
from sklearn.metrics import (balanced_accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from threadpoolctl import threadpool_limits

from src.candidates import (CANDIDATES, CV_SPLITS, RANDOM_STATE, THRESHOLD,
                            describe_candidates, load_data, make_cv)

BATCH_SIZES = (1, 256)
WARMUPS = 7
REPEATS = 45
N_FITS = 3

METRICS = ["f1", "recall", "precision", "roc_auc", "balanced_accuracy", "train_f1"]

RESULT_COLUMNS = [
    "model", "cv_f1_mean", "cv_f1_std", "cv_recall_mean", "cv_recall_std",
    "cv_precision_mean", "cv_roc_auc_mean", "cv_roc_auc_std",
    "cv_balanced_accuracy_mean", "train_f1_mean", "fit_time_s",
    "latency_p50_ms_b1", "latency_p95_ms_b1", "latency_p50_ms_b256", "latency_p95_ms_b256",
    "throughput_samples_s", "model_size_mb",
]


def evaluate_cv(pipeline, X: pd.DataFrame, y: pd.Series, cv, threshold: float = THRESHOLD
                ) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """K-fold оцінка: метрики на валідації + train_f1; повертає (по фолдах, зведення)."""
    rows = []
    for fold, (tr, va) in enumerate(cv.split(X, y), start=1):
        model = clone(pipeline).fit(X.iloc[tr], y.iloc[tr])
        p_val = model.predict_proba(X.iloc[va])[:, 1]
        y_val = y.iloc[va].to_numpy()
        pred = (p_val >= threshold).astype(int)
        p_tr = model.predict_proba(X.iloc[tr])[:, 1]
        rows.append({
            "fold": fold,
            "f1": f1_score(y_val, pred),
            "recall": recall_score(y_val, pred),
            "precision": precision_score(y_val, pred),
            "roc_auc": roc_auc_score(y_val, p_val),
            "balanced_accuracy": balanced_accuracy_score(y_val, pred),
            "train_f1": f1_score(y.iloc[tr].to_numpy(), (p_tr >= threshold).astype(int)),
        })
    fold_df = pd.DataFrame(rows)
    summary: Dict[str, float] = {}
    for m in METRICS:
        prefix = "" if m == "train_f1" else "cv_"        # train_f1 - не валідаційна метрика
        summary[f"{prefix}{m}_mean"] = float(fold_df[m].mean())
        summary[f"{prefix}{m}_std"] = float(fold_df[m].std(ddof=1))
    return fold_df, summary


def profile_prediction(model, batch, warmups: int = WARMUPS, repeats: int = REPEATS) -> np.ndarray:
    """Час одного виклику predict_proba (мс): warmups прогрівів + repeats вимірів."""
    for _ in range(warmups):
        model.predict_proba(batch)
    times = np.empty(repeats)
    for i in range(repeats):
        start = time.perf_counter_ns()
        model.predict_proba(batch)
        times[i] = (time.perf_counter_ns() - start) / 1e6
    return times


def measure_size(model, path) -> float:
    """Розмір усього pipeline (препроцесор + класифікатор) у файлі joblib, МБ."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path, compress=3)
    return os.path.getsize(path) / 1024 ** 2


def _cpu_name() -> str:
    if sys.platform == "win32":
        try:
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True, text=True, timeout=20).stdout.strip()
            if out:
                return out.splitlines()[0].strip()
        except Exception:
            pass
    return platform.processor() or "unknown"


def run_benchmark(out_dir) -> pd.DataFrame:
    """Повний benchmark; пише результати в out_dir (results/)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data = load_data()
    X, y = data.X_train, data.y_train       # тест тут не використовується
    cv = make_cv()

    summaries: Dict[str, dict] = {}
    fold_frames = []
    fitted: Dict[str, object] = {}

    with threadpool_limits(limits=1):
        for name, factory in CANDIDATES.items():
            print(f"[{name}] CV ...", flush=True)
            fold_df, summ = evaluate_cv(factory(), X, y, cv, THRESHOLD)
            fold_df.insert(0, "model", name)
            fold_frames.append(fold_df)

            fit_times = []
            for _ in range(N_FITS):
                model = factory()
                s = time.perf_counter()
                model.fit(X, y)
                fit_times.append(time.perf_counter() - s)
            summ["fit_time_s"] = float(np.median(fit_times))
            summ["model_size_mb"] = measure_size(model, models_dir / f"{name}.joblib")
            summaries[name] = summ
            fitted[name] = model

        # профілювання: однаковий пакет для всіх моделей, прогрів + серія вимірів
        batches = {1: X.iloc[[0]], 256: X.sample(256, random_state=RANDOM_STATE)}
        sample_rows = []
        for name, model in fitted.items():
            print(f"[{name}] профілювання ...", flush=True)
            for bs, batch in batches.items():
                ms = profile_prediction(model, batch, WARMUPS, REPEATS)
                sample_rows.append(pd.DataFrame({
                    "model": name, "batch_size": bs,
                    "repeat": np.arange(1, len(ms) + 1), "ms": ms}))
    samples = pd.concat(sample_rows, ignore_index=True)

    rows = []
    for name, summ in summaries.items():
        row = {"model": name, **summ}
        for bs in BATCH_SIZES:
            ms = samples[(samples["model"] == name) & (samples["batch_size"] == bs)]["ms"].to_numpy()
            row[f"latency_p50_ms_b{bs}"] = float(np.percentile(ms, 50))
            row[f"latency_p95_ms_b{bs}"] = float(np.percentile(ms, 95))
        row["throughput_samples_s"] = 256 / (row["latency_p50_ms_b256"] / 1000)
        rows.append(row)
    results = pd.DataFrame(rows)[RESULT_COLUMNS]

    results.to_csv(out_dir / "benchmark_results.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(out_dir / "fold_results.csv", index=False)
    samples.to_csv(out_dir / "latency_samples.csv", index=False)

    runtime = time.perf_counter() - t0
    meta = {
        "runtime_s": round(runtime, 1),
        "versions": {
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
            "joblib": joblib.__version__, "threadpoolctl": threadpoolctl.__version__,
        },
        "platform": platform.platform(),
        "cpu": _cpu_name(),
        "cpu_count": os.cpu_count(),
        "n_threads": 1,
        "batch_sizes": list(BATCH_SIZES),
        "warmups": WARMUPS, "repeats": REPEATS, "n_fits": N_FITS,
        "cv": {"type": "StratifiedKFold", "n_splits": CV_SPLITS, "shuffle": True,
               "random_state": RANDOM_STATE},
        "threshold": THRESHOLD,
        "split": {"train": int(len(data.X_train)), "test": int(len(data.X_test)),
                  "test_used_in_benchmark": False},
        "size_serialization": "joblib.dump(pipeline, compress=3)",
        "candidates": describe_candidates(),
    }
    (out_dir / "experiment_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Готово за {runtime:.1f} с -> {out_dir}")
    return results
