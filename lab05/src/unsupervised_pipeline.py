"""Чисті функції LR5: дані, препроцесинг, PCA, K-Means, контрольовані аномалії, детектори.

Мітка діагнозу використовується ЛИШЕ для стратифікації розбиття та зовнішнього аудиту
після фіксації K. Вона відокремлена у DataBundle (y_train/y_test) і не потрапляє
в жоден крок препроцесингу, PCA, кластеризації чи детекції.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import skew
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    adjusted_rand_score,
    average_precision_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    f1_score,
    homogeneity_score,
    normalized_mutual_info_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, MinMaxScaler, RobustScaler, StandardScaler

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[1] / "configs" / "experiment_config.json"
BUNDLE_VERSION = "1.0.0"
DETECTORS = ("if30", "ifpca", "rec", "combined")
SCALERS = {"standard": StandardScaler, "minmax": MinMaxScaler, "robust": RobustScaler}
BASE_FEATURES = ("radius", "texture", "perimeter", "area", "smoothness",
                 "compactness", "concavity", "concave_points", "symmetry", "fractal_dimension")
AGGREGATES = ("mean", "se", "worst")
ANOMALY_KINDS = ("calibration_scale", "geometry_break", "worst_mean_swap", "texture_noise")
NOISE_BASES = ("texture", "smoothness", "symmetry", "fractal_dimension")


# =====================================================================
# Конфігурація
# =====================================================================
def _is_int(v) -> bool:
    return isinstance(v, (int, np.integer)) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, bool)


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ValueError(msg)


def _section(config: dict, name: str) -> dict:
    _require(isinstance(config, dict), "config must be a dict")
    _require(name in config and isinstance(config[name], dict), f"config['{name}'] must be an object")
    return config[name]


def validate_config(config: dict, n_test: Optional[int] = None) -> dict:
    """Перевіряє конфігурацію; кидає ValueError з поясненням. Повертає config."""
    split = _section(config, "split")
    ts = split.get("test_size")
    _require(_is_num(ts) and 0 < ts < 1, f"split.test_size must be in (0, 1), got {ts!r}")
    _require(_is_int(split.get("random_state")), "split.random_state must be an int")

    prep = _section(config, "preprocessing")
    _require(prep.get("scaler") in SCALERS,
             f"preprocessing.scaler must be one of {sorted(SCALERS)}, got {prep.get('scaler')!r}")
    _require(isinstance(prep.get("log1p"), bool), "preprocessing.log1p must be a bool")

    pca = _section(config, "pca")
    thr = pca.get("variance_thresholds")
    _require(isinstance(thr, list) and len(thr) > 0, "pca.variance_thresholds must be a non-empty list")
    for t in thr:
        _require(_is_num(t) and 0 < t <= 1, f"pca.variance_thresholds must be in (0, 1], got {t!r}")
    _require(len(set(thr)) == len(thr), "pca.variance_thresholds must be unique")
    _require(pca.get("working_threshold") in thr,
             f"pca.working_threshold={pca.get('working_threshold')!r} must be in variance_thresholds {thr}")

    km = _section(config, "kmeans")
    ks = km.get("k_values")
    _require(isinstance(ks, list) and all(_is_int(k) and k >= 2 for k in ks),
             "kmeans.k_values must be a list of ints >= 2")
    _require(len(set(ks)) == len(ks), "kmeans.k_values must be unique")
    _require(len(ks) >= 6, f"kmeans.k_values needs >= 6 values, got {len(ks)}")
    seeds = km.get("seeds")
    _require(isinstance(seeds, list) and all(_is_int(s) for s in seeds), "kmeans.seeds must be a list of ints")
    _require(len(set(seeds)) >= 5, f"kmeans.seeds needs >= 5 unique seeds, got {len(set(seeds))}")
    _require(_is_int(km.get("n_init")) and km["n_init"] >= 1, "kmeans.n_init must be an int >= 1")
    ref = km.get("reference_seed", 42)
    _require(ref in seeds, f"kmeans.reference_seed={ref} must be one of kmeans.seeds")
    pt = km.get("practical_tie")
    _require(_is_num(pt) and pt >= 0, "kmeans.practical_tie must be a number >= 0")

    st = _section(config, "stability")
    sf = st.get("subsample_fraction")
    _require(_is_num(sf) and 0 < sf <= 1, f"stability.subsample_fraction must be in (0, 1], got {sf!r}")
    _require(_is_int(st.get("n_subsample_runs")) and st["n_subsample_runs"] >= 2,
             "stability.n_subsample_runs must be an int >= 2")
    ut = st.get("unstable_threshold")
    _require(_is_num(ut) and 0 < ut <= 1, f"stability.unstable_threshold must be in (0, 1], got {ut!r}")

    an = _section(config, "anomalies")
    _require(_is_int(an.get("seed")), "anomalies.seed must be an int")
    cpt = an.get("count_per_type")
    _require(_is_int(cpt) and cpt >= 1, "anomalies.count_per_type must be an int >= 1")
    kinds = an.get("kinds")
    _require(isinstance(kinds, dict) and len(kinds) > 0, "anomalies.kinds must be a non-empty object")
    for kind, params in kinds.items():
        _require(kind in ANOMALY_KINDS, f"unknown anomaly kind {kind!r}; allowed: {list(ANOMALY_KINDS)}")
        _require(isinstance(params, dict), f"anomalies.kinds.{kind} must be an object")
    if "calibration_scale" in kinds:
        f = kinds["calibration_scale"].get("factor")
        _require(_is_num(f) and f > 0, "calibration_scale.factor must be > 0")
    if "geometry_break" in kinds:
        f = kinds["geometry_break"].get("area_factor")
        _require(_is_num(f) and f > 0, "geometry_break.area_factor must be > 0")
    if "texture_noise" in kinds:
        s = kinds["texture_noise"].get("sigma")
        _require(_is_num(s) and s >= 0, "texture_noise.sigma must be >= 0")
    if n_test is not None:
        total = cpt * len(kinds)
        _require(total < n_test,
                 f"count_per_type*len(kinds)={total} must be < n_test={n_test}")

    iso = _section(config, "isolation_forest")
    _require(_is_int(iso.get("n_estimators")) and iso["n_estimators"] >= 1,
             "isolation_forest.n_estimators must be an int >= 1")
    _require(_is_int(iso.get("random_state")), "isolation_forest.random_state must be an int")

    q = config.get("threshold_quantile")
    _require(_is_num(q) and 0 < q < 1, f"threshold_quantile must be in (0, 1), got {q!r}")
    _require(config.get("combined_rule") == "max_percentile",
             f"combined_rule must be 'max_percentile', got {config.get('combined_rule')!r}")
    return config


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict:
    """Читає JSON-конфігурацію та валідує її."""
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    return validate_config(cfg)


# =====================================================================
# Дані та розбиття
# =====================================================================
def to_snake(col: str) -> str:
    """'mean concave points' -> 'concave_points_mean', 'radius error' -> 'radius_se'."""
    parts = col.split()
    if parts[0] in ("mean", "worst"):
        return "_".join(parts[1:]) + "_" + parts[0]
    if parts[-1] == "error":
        return "_".join(parts[:-1]) + "_se"
    return col


@dataclass
class DataBundle:
    """Розбиття даних. y_* тримаються окремо й використовуються лише для аудиту."""
    X_train: pd.DataFrame          # сирі ознаки, індекс = id об'єкта
    X_test: pd.DataFrame
    y_train: pd.Series             # AUDIT ONLY: 0 = malignant, 1 = benign
    y_test: pd.Series              # AUDIT ONLY
    id_train: np.ndarray
    id_test: np.ndarray
    feature_names: List[str] = field(default_factory=list)


def check_split(id_train: Sequence[int], id_test: Sequence[int], n_total: int) -> None:
    """id train/test не перетинаються і разом дають усі 0..n_total-1."""
    a, b = set(map(int, id_train)), set(map(int, id_test))
    if len(a) != len(id_train) or len(b) != len(id_test):
        raise ValueError("duplicate ids inside a split part")
    if a & b:
        raise ValueError(f"train/test ids overlap: {sorted(a & b)[:5]}...")
    if (a | b) != set(range(n_total)):
        raise ValueError("train and test ids together must cover all objects exactly once")


def load_dataset(config: dict) -> DataBundle:
    """Завантажує Breast Cancer, перейменовує колонки, розбиває за id (мітка лише для stratify)."""
    raw = load_breast_cancer(as_frame=True)
    X = raw.data.copy()
    X.columns = [to_snake(c) for c in X.columns]
    y = pd.Series(raw.target.values, index=np.arange(len(X)), name="target")
    X.index = np.arange(len(X))
    ids = np.arange(len(X))
    sp = config["split"]
    strat = y.values if sp.get("stratify_by_label_for_split_only", True) else None
    id_tr, id_te = train_test_split(ids, test_size=sp["test_size"], stratify=strat,
                                    random_state=sp["random_state"])
    check_split(id_tr, id_te, len(X))
    return DataBundle(
        X_train=X.loc[id_tr].copy(), X_test=X.loc[id_te].copy(),
        y_train=y.loc[id_tr].copy(), y_test=y.loc[id_te].copy(),
        id_train=id_tr, id_test=id_te, feature_names=list(X.columns),
    )


# =====================================================================
# Препроцесинг (fit лише на train)
# =====================================================================
def _log1p_checked(X):
    """log1p з перевіркою на від'ємні значення (інакше NaN із RuntimeWarning)."""
    X = np.asarray(X, dtype=float)
    if (X < 0).any():
        raise ValueError("log1p preprocessing requires non-negative features; found negative values")
    return np.log1p(X)


def fit_preprocessor(X_train: pd.DataFrame, config: dict) -> Pipeline:
    """Pipeline(log1p -> scaler), навчений лише на train.

    check_inverse=False: внутрішня перевірка sklearn викликає transform на масиві й дає
    UserWarning про імена ознак; коректність expm1(log1p(x)) = x перевіряється в тестах.
    """
    prep = config["preprocessing"]
    if prep["log1p"]:
        first = FunctionTransformer(_log1p_checked, inverse_func=np.expm1, validate=True,
                                    check_inverse=False, feature_names_out="one-to-one")
    else:
        first = FunctionTransformer(validate=True, check_inverse=False, feature_names_out="one-to-one")
    pipe = Pipeline([("log1p", first), ("scaler", SCALERS[prep["scaler"]]())])
    return pipe.fit(X_train)


def preprocessing_check(X_train: pd.DataFrame) -> pd.DataFrame:
    """Асиметрія до/після log1p та кількість нулів (train)."""
    raw_sk = skew(X_train.values, axis=0)
    log_sk = skew(np.log1p(X_train.values), axis=0)
    return pd.DataFrame({
        "feature": X_train.columns,
        "skew_raw_train": raw_sk,
        "skew_log1p_train": log_sk,
        "n_zeros": (X_train.values == 0).sum(axis=0),
    })


# =====================================================================
# PCA
# =====================================================================
def fit_pca(X_scaled: np.ndarray, n_components: int) -> PCA:
    """PCA з повним SVD (детермінований)."""
    return PCA(n_components=n_components, svd_solver="full", random_state=42).fit(X_scaled)


def choose_n_components(evr, threshold: float) -> int:
    """Найменше q, для якого кумулятивна частка дисперсії >= threshold (допуск 1e-12)."""
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float, np.floating, np.integer)):
        raise ValueError(f"threshold must be a number, got {threshold!r}")
    if not (0 < threshold <= 1):
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")
    arr = np.asarray(evr, dtype=float)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("evr must be a non-empty 1-D array")
    if not np.all(np.isfinite(arr)) or (arr < 0).any():
        raise ValueError("evr must be finite and non-negative")
    if arr.sum() > 1 + 1e-9:
        raise ValueError(f"evr sums to {arr.sum():.6f} > 1")
    cum = np.cumsum(arr)
    idx = int(np.searchsorted(cum, threshold - 1e-12, side="left"))
    if idx >= len(cum):
        raise ValueError(f"evr cumulative sum {cum[-1]:.6f} never reaches threshold {threshold}")
    return idx + 1


def reconstruction_errors(pca: PCA, X_scaled: np.ndarray) -> np.ndarray:
    """Середня квадратична помилка відновлення по ознаках для кожного об'єкта."""
    X = np.asarray(X_scaled, dtype=float)
    rec = pca.inverse_transform(pca.transform(X))
    return ((X - rec) ** 2).mean(axis=1)


def reconstruction_error_curve(pca_full: PCA, X_scaled: np.ndarray) -> pd.DataFrame:
    """Середня та p95 помилка відновлення для q = 1..n_components (з одного повного PCA)."""
    Xc = np.asarray(X_scaled, dtype=float) - pca_full.mean_
    V = pca_full.components_
    rows = []
    for q in range(1, V.shape[0] + 1):
        rec = (Xc @ V[:q].T) @ V[:q]
        e = ((Xc - rec) ** 2).mean(axis=1)
        rows.append({"q": q, "mse_mean": e.mean(), "mse_p95": np.percentile(e, 95)})
    return pd.DataFrame(rows)


def pca_dimension_comparison(X_scaled: np.ndarray, pca_full: PCA, thresholds: Sequence[float],
                             n_init: int, seed: int, n_fit_repeats: int = 5) -> pd.DataFrame:
    """Для кожного порога: q, cum. дисперсія, помилка відновлення, silhouette K=2, час підгонки PCA."""
    evr = pca_full.explained_variance_ratio_
    cum = np.cumsum(evr)
    rows = []
    for th in thresholds:
        q = choose_n_components(evr, th)
        times = []
        for _ in range(n_fit_repeats):
            t0 = time.perf_counter()
            pq = fit_pca(X_scaled, q)
            times.append(time.perf_counter() - t0)
        e = reconstruction_errors(pq, X_scaled)
        Z = pq.transform(X_scaled)
        lab = KMeans(n_clusters=2, n_init=n_init, random_state=seed).fit_predict(Z)
        rows.append({"threshold": th, "q": q, "cumulative_variance": cum[q - 1],
                     "recon_mse_mean": e.mean(), "recon_mse_p95": np.percentile(e, 95),
                     "silhouette_k2": silhouette_score(Z, lab), "pca_fit_time_s": float(np.mean(times))})
    return pd.DataFrame(rows)


# =====================================================================
# Кластеризація
# =====================================================================
def evaluate_clustering(X: np.ndarray, k_values: Sequence[int], seeds: Sequence[int], n_init: int):
    """K-Means для кожної пари (K, seed). Повертає (таблиця метрик, {(K, seed): labels})."""
    rows, labels = [], {}
    for k in k_values:
        for s in seeds:
            t0 = time.perf_counter()
            km = KMeans(n_clusters=k, n_init=n_init, random_state=s).fit(X)
            dt = time.perf_counter() - t0
            lab = km.labels_
            labels[(k, s)] = lab
            rows.append({"K": k, "seed": s, "inertia": km.inertia_,
                         "silhouette": silhouette_score(X, lab),
                         "davies_bouldin": davies_bouldin_score(X, lab),
                         "calinski_harabasz": calinski_harabasz_score(X, lab),
                         "fit_time_s": dt})
    return pd.DataFrame(rows), labels


def summarize_clustering(runs: pd.DataFrame, labels: dict, seeds: Sequence[int]) -> pd.DataFrame:
    """Середні/std метрик по seed і середні попарні ARI/NMI між seed для кожного K."""
    rows = []
    for k, g in runs.groupby("K", sort=True):
        row = {"K": int(k)}
        for m in ("inertia", "silhouette", "davies_bouldin", "calinski_harabasz", "fit_time_s"):
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_std"] = g[m].std(ddof=0)
        aris = [adjusted_rand_score(labels[(k, a)], labels[(k, b)]) for a, b in combinations(seeds, 2)]
        nmis = [normalized_mutual_info_score(labels[(k, a)], labels[(k, b)]) for a, b in combinations(seeds, 2)]
        row.update(ari_mean=float(np.mean(aris)), ari_std=float(np.std(aris)),
                   nmi_mean=float(np.mean(nmis)), nmi_std=float(np.std(nmis)))
        rows.append(row)
    return pd.DataFrame(rows)


def select_k(summary: pd.DataFrame, practical_tie: float) -> dict:
    """Максимум середнього silhouette; якщо різниця < practical_tie — вищий ARI, потім менший K."""
    best = summary["silhouette_mean"].max()
    cand = summary[(best - summary["silhouette_mean"]) < practical_tie].copy()
    cand["_ari_key"] = -cand["ari_mean"].round(10)
    cand = cand.sort_values(["_ari_key", "K"])
    chosen = cand.iloc[0]
    top = summary.sort_values("silhouette_mean", ascending=False).iloc[0]
    return {"selected_k": int(chosen["K"]),
            "silhouette_mean": float(chosen["silhouette_mean"]),
            "ari_mean": float(chosen["ari_mean"]),
            "best_silhouette_k": int(top["K"]),
            "best_silhouette": float(best),
            "tie_candidates": [int(k) for k in cand["K"]],
            "practical_tie": practical_tie}


def fit_reference_kmeans(X: np.ndarray, k: int, seed: int, n_init: int) -> KMeans:
    """Еталонна модель K-Means (повторює прогін evaluate_clustering для (k, seed))."""
    return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(X)


def align_cluster_labels(reference, candidate) -> np.ndarray:
    """Угорське узгодження міток candidate з reference за матрицею спряженості."""
    ref = np.asarray(reference)
    cand = np.asarray(candidate)
    if ref.shape != cand.shape or ref.ndim != 1:
        raise ValueError("reference and candidate must be 1-D arrays of the same length")
    ref_l, cand_l = np.unique(ref), np.unique(cand)
    cont = np.zeros((len(cand_l), len(ref_l)), dtype=int)
    ci = {c: i for i, c in enumerate(cand_l)}
    ri = {r: i for i, r in enumerate(ref_l)}
    for c, r in zip(cand, ref):
        cont[ci[c], ri[r]] += 1
    rows, cols = linear_sum_assignment(-cont)
    mapping = {cand_l[i]: ref_l[j] for i, j in zip(rows, cols)}
    nxt = int(max(ref_l.max(), cand_l.max())) + 1
    for c in cand_l:                      # зайві мітки candidate (K_cand > K_ref)
        if c not in mapping:
            mapping[c] = nxt
            nxt += 1
    return np.array([mapping[c] for c in cand])


def compute_object_stability(reference, runs: Sequence[np.ndarray]) -> np.ndarray:
    """Частка прогонів, де узгоджена мітка об'єкта збігається з еталонною."""
    if len(runs) == 0:
        raise ValueError("runs must be non-empty")
    ref = np.asarray(reference)
    same = [align_cluster_labels(ref, r) == ref for r in runs]
    return np.mean(same, axis=0)


def measure_cluster_stability(Z: np.ndarray, k: int, reference_model: KMeans, seed_labels: Sequence[np.ndarray],
                              fraction: float, n_runs: int, n_init: int, base_seed: int = 1000) -> dict:
    """(a) попарні ARI/NMI між seed на повних даних; (б) об'єктна стабільність за підвибірками."""
    ref = reference_model.labels_
    n = len(Z)
    m = max(k, int(round(fraction * n)))
    runs = []
    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        idx = rng.choice(n, size=m, replace=False)
        km = KMeans(n_clusters=k, n_init=n_init, random_state=base_seed + r).fit(Z[idx])
        runs.append(km.predict(Z))
    stab = compute_object_stability(ref, runs)
    aligned = [align_cluster_labels(ref, r) for r in runs]
    return {
        "seed_ari_mean": float(np.mean([adjusted_rand_score(a, b) for a, b in combinations(seed_labels, 2)])),
        "seed_nmi_mean": float(np.mean([normalized_mutual_info_score(a, b) for a, b in combinations(seed_labels, 2)])),
        "run_ari_vs_reference_mean": float(np.mean([adjusted_rand_score(ref, a) for a in aligned])),
        "object_stability": stab,
        "n_runs": n_runs,
        "subsample_size": m,
    }


def object_stability_table(ids, Z: np.ndarray, reference_model: KMeans, stability: np.ndarray) -> pd.DataFrame:
    """id, еталонний кластер, стабільність, відстань до свого центроїда та запас до найближчого іншого."""
    d = reference_model.transform(Z)
    lab = reference_model.labels_
    own = d[np.arange(len(Z)), lab]
    other = np.where(np.eye(d.shape[1], dtype=bool)[lab], np.inf, d).min(axis=1)
    return pd.DataFrame({"id": np.asarray(ids), "reference_cluster": lab, "stability": stability,
                         "dist_own_centroid": own, "margin_to_other": other - own})


def external_audit(y_true, labels) -> dict:
    """Зовнішні метрики (ЛИШЕ аудит після фіксації K): ARI, NMI, homogeneity, completeness."""
    return {"ari": adjusted_rand_score(y_true, labels),
            "nmi": normalized_mutual_info_score(y_true, labels),
            "homogeneity": homogeneity_score(y_true, labels),
            "completeness": completeness_score(y_true, labels)}


def cluster_composition(labels, y_true) -> pd.DataFrame:
    """Таблиця кластер x діагноз (аудит)."""
    lab = np.asarray(labels)
    y = np.asarray(y_true)
    rows = []
    for c in np.unique(lab):
        m = lab == c
        mal, ben = int((y[m] == 0).sum()), int((y[m] == 1).sum())
        rows.append({"cluster": int(c), "malignant": mal, "benign": ben, "size": mal + ben,
                     "malignant_share": mal / (mal + ben)})
    return pd.DataFrame(rows)


# =====================================================================
# Контрольовані аномалії
# =====================================================================
@dataclass
class AnomalyBatch:
    X: pd.DataFrame            # модифіковані сирі ознаки (індекс = id)
    is_anomaly: np.ndarray     # bool
    kind: np.ndarray           # 'clean' або назва типу
    params_table: pd.DataFrame  # id, kind, params (json)


def _cols(names: Sequence[str], columns: Sequence[str]) -> List[str]:
    missing = [c for c in names if c not in columns]
    if missing:
        raise ValueError(f"missing columns for anomaly injection: {missing}")
    return list(names)


def inject_controlled_anomalies(X_test_raw: pd.DataFrame, ids, config: dict) -> AnomalyBatch:
    """Вносить 4 типи аномалій у сирі значення test (до log1p/scaler). Вхід не змінюється."""
    an = config["anomalies"]
    kinds = an["kinds"]
    for k in kinds:
        if k not in ANOMALY_KINDS:
            raise ValueError(f"unknown anomaly kind {k!r}; allowed: {list(ANOMALY_KINDS)}")
    ids = np.asarray(ids)
    if len(ids) != len(X_test_raw):
        raise ValueError("ids and X_test_raw must have the same length")
    cpt = an["count_per_type"]
    total = cpt * len(kinds)
    if total >= len(ids):
        raise ValueError(f"too many anomalies: {total} requested for {len(ids)} test objects")
    columns = list(X_test_raw.columns)
    rng = np.random.default_rng(an["seed"])
    chosen = rng.choice(np.sort(ids), size=total, replace=False)
    pos_of = {int(i): p for p, i in enumerate(ids)}
    arr = X_test_raw.to_numpy(dtype=float, copy=True)
    kind = np.full(len(ids), "clean", dtype="<U32")
    ci = {c: i for i, c in enumerate(columns)}
    params_rows = []
    for b, (name, p) in enumerate(kinds.items()):
        block = chosen[b * cpt:(b + 1) * cpt]
        pos = np.array([pos_of[int(i)] for i in block])
        kind[pos] = name
        if name == "calibration_scale":
            f = float(p["factor"])
            lin = _cols([f"{b_}_{a}" for b_ in ("radius", "perimeter") for a in AGGREGATES], columns)
            area = _cols([f"area_{a}" for a in AGGREGATES], columns)
            arr[np.ix_(pos, [ci[c] for c in lin])] *= f
            arr[np.ix_(pos, [ci[c] for c in area])] *= f * f
            pj = {"factor": f}
        elif name == "geometry_break":
            f = float(p["area_factor"])
            area = _cols(["area_mean", "area_worst"], columns)
            arr[np.ix_(pos, [ci[c] for c in area])] *= f
            pj = {"area_factor": f}
        elif name == "worst_mean_swap":
            for base in BASE_FEATURES:
                im, iw = ci[_cols([f"{base}_mean"], columns)[0]], ci[_cols([f"{base}_worst"], columns)[0]]
                tmp = arr[pos, im].copy()
                arr[pos, im] = arr[pos, iw]
                arr[pos, iw] = tmp
            pj = {}
        else:  # texture_noise
            sigma = float(p["sigma"])
            noisy = _cols([f"{b_}_{a}" for b_ in NOISE_BASES for a in AGGREGATES], columns)
            idx = [ci[c] for c in noisy]
            noise = np.exp(rng.normal(0.0, sigma, size=(len(pos), len(idx))))
            arr[np.ix_(pos, idx)] = arr[np.ix_(pos, idx)] * noise
            pj = {"sigma": sigma, "n_cells": int(noise.size)}
        for i in block:
            params_rows.append({"id": int(i), "kind": name, "params": json.dumps(pj)})
    X_new = pd.DataFrame(arr, index=X_test_raw.index, columns=columns)
    pt = pd.DataFrame(params_rows)
    return AnomalyBatch(X=X_new, is_anomaly=kind != "clean", kind=kind, params_table=pt)


# =====================================================================
# Детектори аномалій
# =====================================================================
def empirical_percentile(sorted_train: np.ndarray, s) -> np.ndarray:
    """Емпіричний перцентиль відносно відсортованих train-скорів: searchsorted(right)/n."""
    return np.searchsorted(sorted_train, np.asarray(s), side="right") / len(sorted_train)


@dataclass
class DetectorBundle:
    pca: PCA
    if30: IsolationForest
    ifpca: IsolationForest
    train_scores: Dict[str, np.ndarray]    # по DETECTORS
    sorted_train: Dict[str, np.ndarray]    # відсортовані ifpca та rec (для перцентилів)
    thresholds: Dict[str, float]
    quantile: float


def _raw_scores(bundle: DetectorBundle, X_scaled: np.ndarray, Z: np.ndarray) -> Dict[str, np.ndarray]:
    s_if30 = -bundle.if30.score_samples(X_scaled)
    s_ifpca = -bundle.ifpca.score_samples(Z)
    s_rec = reconstruction_errors(bundle.pca, X_scaled)
    u_if = empirical_percentile(bundle.sorted_train["ifpca"], s_ifpca)
    u_rec = empirical_percentile(bundle.sorted_train["rec"], s_rec)
    return {"if30": s_if30, "ifpca": s_ifpca, "rec": s_rec, "combined": np.maximum(u_if, u_rec)}


def fit_anomaly_detectors(X_train_scaled: np.ndarray, Z_train: np.ndarray, config: dict,
                          pca: PCA) -> DetectorBundle:
    """IsolationForest у 30-вимірному та PCA-просторі + поріг як квантиль train-скорів.

    Відхилення від чернетки: pca передається явно (потрібен для rec-скору нових даних).
    """
    X = np.asarray(X_train_scaled, dtype=float)
    Z = np.asarray(Z_train, dtype=float)
    if X.ndim != 2 or Z.ndim != 2 or len(X) != len(Z):
        raise ValueError("X_train_scaled and Z_train must be 2-D with the same number of rows")
    if Z.shape[1] != pca.n_components_:
        raise ValueError(f"Z_train has {Z.shape[1]} columns, pca has {pca.n_components_} components")
    iso = config["isolation_forest"]

    def make():
        return IsolationForest(n_estimators=iso["n_estimators"], max_samples=iso.get("max_samples", "auto"),
                               contamination="auto", random_state=iso["random_state"])

    b = DetectorBundle(pca=pca, if30=make().fit(X), ifpca=make().fit(Z), train_scores={},
                       sorted_train={}, thresholds={}, quantile=config["threshold_quantile"])
    s_if30 = -b.if30.score_samples(X)
    s_ifpca = -b.ifpca.score_samples(Z)
    s_rec = reconstruction_errors(pca, X)
    b.sorted_train = {"ifpca": np.sort(s_ifpca), "rec": np.sort(s_rec)}
    comb = np.maximum(empirical_percentile(b.sorted_train["ifpca"], s_ifpca),
                      empirical_percentile(b.sorted_train["rec"], s_rec))
    b.train_scores = {"if30": s_if30, "ifpca": s_ifpca, "rec": s_rec, "combined": comb}
    b.thresholds = {d: float(np.quantile(v, b.quantile)) for d, v in b.train_scores.items()}
    return b


def score_anomalies(bundle: DetectorBundle, X_scaled: np.ndarray, Z: np.ndarray, ids=None) -> pd.DataFrame:
    """Скори та прапорці (score > поріг) для чотирьох детекторів."""
    X = np.asarray(X_scaled, dtype=float)
    Zc = np.asarray(Z, dtype=float)
    if len(X) != len(Zc):
        raise ValueError("X_scaled and Z must have the same number of rows")
    if not (np.isfinite(X).all() and np.isfinite(Zc).all()):
        raise ValueError("inputs contain NaN or inf")
    sc = _raw_scores(bundle, X, Zc)
    df = pd.DataFrame({"id": np.arange(len(X)) if ids is None else np.asarray(ids)})
    for d in DETECTORS:
        df[f"score_{d}"] = sc[d]
    for d in DETECTORS:
        df[f"flag_{d}"] = sc[d] > bundle.thresholds[d]
    return df


def precision_at_k(y, scores, k: int) -> float:
    """Частка аномалій серед k об'єктів із найбільшим score.

    Якщо на межі top-k кілька однакових балів, береться очікуване значення за
    випадкового вибору серед них: результат не залежить від порядку рядків.
    """
    y = np.asarray(y).astype(float)
    s = np.asarray(scores, dtype=float)
    if not _is_int(k) or not 1 <= k <= len(s):
        raise ValueError(f"k must be an int in [1, {len(s)}], got {k!r}")
    kth = np.sort(s)[::-1][k - 1]
    above, tied = s > kth, s == kth
    need = k - int(above.sum())
    return float((y[above].sum() + need * y[tied].mean()) / k)


def evaluate_anomalies(scores_df: pd.DataFrame, y_controlled) -> pd.DataFrame:
    """Метрики кожного детектора проти контрольованих аномалій (y=1 - аномалія)."""
    y = np.asarray(y_controlled).astype(int)
    k = int(y.sum())
    if len(y) != len(scores_df):
        raise ValueError("y_controlled length must match scores_df")
    if k == 0 or k == len(y):
        raise ValueError("y_controlled must contain both anomalies and clean objects")
    rows = []
    for d in DETECTORS:
        s = scores_df[f"score_{d}"].to_numpy()
        flag = scores_df[f"flag_{d}"].to_numpy().astype(int)
        tp = int(((flag == 1) & (y == 1)).sum())
        fp = int(((flag == 1) & (y == 0)).sum())
        fn = int(((flag == 0) & (y == 1)).sum())
        tn = int(((flag == 0) & (y == 0)).sum())
        rows.append({"detector": d, "roc_auc": roc_auc_score(y, s), "average_precision": average_precision_score(y, s),
                     "precision": precision_score(y, flag, zero_division=0),
                     "recall": recall_score(y, flag, zero_division=0),
                     "f1": f1_score(y, flag, zero_division=0),
                     "precision_at_k": precision_at_k(y, s, k), "k": k,
                     "n_flagged": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn})
    return pd.DataFrame(rows)


def recall_by_kind(scores_df: pd.DataFrame, kind) -> pd.DataFrame:
    """Частка виявлених аномалій за типом (рядки - типи, колонки - детектори)."""
    kind = np.asarray(kind)
    rows = []
    for kd in [k for k in dict.fromkeys(kind) if k != "clean"]:
        m = kind == kd
        row = {"kind": kd, "n": int(m.sum())}
        for d in DETECTORS:
            row[f"recall_{d}"] = float(scores_df[f"flag_{d}"].to_numpy()[m].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def top_k_jaccard(a, b, k: int) -> float:
    """Jaccard між множинами індексів top-k за двома векторами скорів."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != b.shape or a.ndim != 1:
        raise ValueError("a and b must be 1-D arrays of the same length")
    if not _is_int(k) or not 1 <= k <= len(a):
        raise ValueError(f"k must be an int in [1, {len(a)}], got {k!r}")
    sa = set(np.argsort(-a, kind="stable")[:k].tolist())
    sb = set(np.argsort(-b, kind="stable")[:k].tolist())
    return len(sa & sb) / len(sa | sb)


def if_disagreement(flag30, flagpca, is_anomaly) -> pd.DataFrame:
    """Збіг рішень IF30 та IFPCA: обидва / лише 30 / лише PCA / жоден (серед аномалій і чистих)."""
    f30, fp, an = (np.asarray(x).astype(bool) for x in (flag30, flagpca, is_anomaly))
    rows = []
    for grp, m in (("anomalies", an), ("clean", ~an)):
        rows.append({"group": grp, "n": int(m.sum()),
                     "both": int((f30 & fp & m).sum()), "only_if30": int((f30 & ~fp & m).sum()),
                     "only_ifpca": int((~f30 & fp & m).sum()), "none": int((~f30 & ~fp & m).sum())})
    return pd.DataFrame(rows)


def posthoc_q_sensitivity(X_train_scaled, X_test_scaled, is_anomaly, q_by_threshold: Dict[float, int],
                          quantile: float) -> pd.DataFrame:
    """POST-HOC діагностика (не впливає на жоден вибір): rec-детектор для інших q."""
    y = np.asarray(is_anomaly).astype(int)
    rows = []
    for th, q in q_by_threshold.items():
        p = fit_pca(X_train_scaled, q)
        s_tr, s_te = reconstruction_errors(p, X_train_scaled), reconstruction_errors(p, X_test_scaled)
        flag = s_te > np.quantile(s_tr, quantile)
        rows.append({"variance_threshold": th, "q": q, "roc_auc": roc_auc_score(y, s_te),
                     "average_precision": average_precision_score(y, s_te),
                     "recall": recall_score(y, flag, zero_division=0),
                     "precision": precision_score(y, flag, zero_division=0), "n_flagged": int(flag.sum())})
    return pd.DataFrame(rows)


def false_positive_table(scores_df, X_scaled, pca: PCA, feature_names, is_anomaly, y_audit) -> pd.DataFrame:
    """Чисті test-об'єкти, позначені rec/combined: діагноз (лише аудит) та 3 ознаки з найбільшим |залишком|."""
    X = np.asarray(X_scaled, dtype=float)
    resid = X - pca.inverse_transform(pca.transform(X))
    an = np.asarray(is_anomaly).astype(bool)
    flag = (scores_df["flag_rec"] | scores_df["flag_combined"]).to_numpy()
    y = np.asarray(y_audit)
    rows = []
    for i in np.where(flag & ~an)[0]:
        top = np.argsort(-np.abs(resid[i]))[:3]
        row = {"id": int(scores_df["id"].iloc[i]),
               "flag_rec": bool(scores_df["flag_rec"].iloc[i]),
               "flag_combined": bool(scores_df["flag_combined"].iloc[i]),
               "score_rec": float(scores_df["score_rec"].iloc[i]),
               "diagnosis_audit_only": "malignant" if y[i] == 0 else "benign"}
        for j, t in enumerate(top, 1):
            row[f"feature_{j}"] = feature_names[t]
            row[f"residual_{j}"] = float(resid[i, t])
        rows.append(row)
    return pd.DataFrame(rows)


def false_negative_table(scores_df, kind, thresholds: Dict[str, float]) -> pd.DataFrame:
    """Пропущені аномалії: детектор, тип, скор та поріг."""
    kind = np.asarray(kind)
    rows = []
    for d in DETECTORS:
        miss = np.where((kind != "clean") & ~scores_df[f"flag_{d}"].to_numpy())[0]
        for i in miss:
            s = float(scores_df[f"score_{d}"].iloc[i])
            rows.append({"detector": d, "id": int(scores_df["id"].iloc[i]), "kind": str(kind[i]),
                         "score": s, "threshold": thresholds[d], "score_over_threshold": s / thresholds[d]})
    return pd.DataFrame(rows, columns=["detector", "id", "kind", "score", "threshold", "score_over_threshold"])


# =====================================================================
# Артефакти та інференс
# =====================================================================
def make_bundle(config: dict, preprocessor: Pipeline, pca: PCA, kmeans_scaled30: KMeans, kmeans_pca: KMeans,
                detectors: DetectorBundle, feature_names: Sequence[str], selected_k: dict) -> dict:
    """Збирає словник артефактів для joblib."""
    return {"config": copy.deepcopy(config), "preprocessor": preprocessor, "pca": pca,
            "kmeans_scaled30": kmeans_scaled30, "kmeans_pca": kmeans_pca,
            "if30": detectors.if30, "ifpca": detectors.ifpca,
            "train_scores": detectors.train_scores, "thresholds": detectors.thresholds,
            "detectors": detectors, "selected_k": dict(selected_k),
            "feature_names": list(feature_names), "version": BUNDLE_VERSION}


def save_artifacts(bundle: dict, path: str | Path) -> Path:
    import joblib
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, path)
    return path


def load_bundle(path: str | Path) -> dict:
    import joblib
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"bundle not found: {path}")
    bundle = joblib.load(path)
    for key in ("config", "preprocessor", "pca", "kmeans_scaled30", "kmeans_pca", "detectors",
                "feature_names", "version"):
        if key not in bundle:
            raise ValueError(f"bundle is missing key {key!r}")
    return bundle


def predict_new(bundle: dict, X_raw: pd.DataFrame) -> pd.DataFrame:
    """Кластери (обидва простори) та аномальні скори/прапорці для нового пакета сирих даних."""
    if not isinstance(X_raw, pd.DataFrame):
        raise TypeError("X_raw must be a pandas DataFrame")
    names = bundle["feature_names"]
    missing = [c for c in names if c not in X_raw.columns]
    if missing:
        raise ValueError(f"missing feature columns: {missing[:5]}")
    X = X_raw[names]
    if X.isna().any().any():
        raise ValueError("input contains NaN")
    Xs = bundle["preprocessor"].transform(X)
    Z = bundle["pca"].transform(Xs)
    out = score_anomalies(bundle["detectors"], Xs, Z, ids=X_raw.index.to_numpy())
    out.insert(1, "cluster_scaled30", bundle["kmeans_scaled30"].predict(Xs))
    out.insert(2, "cluster_pca", bundle["kmeans_pca"].predict(Z))
    return out
