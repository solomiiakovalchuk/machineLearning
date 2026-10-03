"""LR5: PCA, K-Means та пошук аномалій на Breast Cancer Wisconsin.

Запуск: python lr05_experiment.py [--config configs/experiment_config.json]
Створює results/*.csv|json, figures/*.png, artifacts/lr05_bundle.joblib, results/run_log.txt.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import scipy
import sklearn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src import plotting as P  # noqa: E402
from src import unsupervised_pipeline as U  # noqa: E402

RESULTS, FIGURES, ARTIFACTS = ROOT / "results", ROOT / "figures", ROOT / "artifacts"


class Log:
    """Дублює повідомлення в консоль і в run_log.txt."""

    def __init__(self):
        self.lines = []

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        self.lines.append(msg)


def _df_str(df: pd.DataFrame, digits: int = 4) -> str:
    return df.round(digits).to_string(index=False)


def run_experiment(config_path: Path) -> dict:
    t_start = time.perf_counter()
    log = Log()
    for d in (RESULTS, FIGURES, ARTIFACTS):
        d.mkdir(parents=True, exist_ok=True)

    cfg = U.load_config(config_path)
    log(f"== {cfg['experiment']} v{cfg['version']} ==")

    # ---------------- дані та розбиття ----------------
    data = U.load_dataset(cfg)
    U.validate_config(cfg, n_test=len(data.id_test))
    feats = data.feature_names
    pd.concat([pd.DataFrame({"id": data.id_train, "part": "train"}),
               pd.DataFrame({"id": data.id_test, "part": "test"})]).sort_values("id") \
        .to_csv(RESULTS / "split_ids.csv", index=False)
    log(f"train={len(data.id_train)} test={len(data.id_test)} features={len(feats)}")

    # ---------------- препроцесинг (fit лише на train) ----------------
    prep = U.fit_preprocessor(data.X_train, cfg)
    Xs_tr = prep.transform(data.X_train)
    Xs_te_clean = prep.transform(data.X_test)
    chk = U.preprocessing_check(data.X_train)
    chk.to_csv(RESULTS / "preprocessing_check.csv", index=False)
    log(f"skew |>1|: raw={(chk.skew_raw_train.abs() > 1).sum()} log1p={(chk.skew_log1p_train.abs() > 1).sum()} "
        f"(mean |skew| {chk.skew_raw_train.abs().mean():.2f} -> {chk.skew_log1p_train.abs().mean():.2f})")

    # ---------------- PCA ----------------
    pca_full = U.fit_pca(Xs_tr, Xs_tr.shape[1])
    evr = pca_full.explained_variance_ratio_
    pd.DataFrame({"component": np.arange(1, len(evr) + 1), "explained_variance_ratio": evr,
                  "cumulative_variance": np.cumsum(evr)}).to_csv(RESULTS / "pca_variance.csv", index=False)
    thresholds = cfg["pca"]["variance_thresholds"]
    q_by_th = {th: U.choose_n_components(evr, th) for th in thresholds}
    q = q_by_th[cfg["pca"]["working_threshold"]]
    space_pca = f"pca{q}"
    space_full = "scaled30"
    km_cfg = cfg["kmeans"]
    ref_seed = km_cfg.get("reference_seed", 42)
    dim_cmp = U.pca_dimension_comparison(Xs_tr, pca_full, thresholds, km_cfg["n_init"], ref_seed)
    dim_cmp.to_csv(RESULTS / "pca_dimension_comparison.csv", index=False)
    log("\n[PCA] q per threshold: " + ", ".join(f"{th:.2f}->q={qq}" for th, qq in q_by_th.items()) +
        f"; working q={q}")
    log(_df_str(dim_cmp))
    pca = U.fit_pca(Xs_tr, q)
    Z_tr = pca.transform(Xs_tr)
    Z_te_clean = pca.transform(Xs_te_clean)
    load = pd.DataFrame(pca_full.components_[:3].T, index=feats, columns=["PC1", "PC2", "PC3"])
    load.rename_axis("feature").to_csv(RESULTS / "pca_loadings.csv")
    curve = U.reconstruction_error_curve(pca_full, Xs_tr)

    # ---------------- кластеризація ----------------
    spaces = {space_full: Xs_tr, space_pca: Z_tr}
    runs_all, summaries, labels_by_space, selected, ref_models = [], {}, {}, {}, {}
    for sp, X in spaces.items():
        runs, labels = U.evaluate_clustering(X, km_cfg["k_values"], km_cfg["seeds"], km_cfg["n_init"])
        summ = U.summarize_clustering(runs, labels, km_cfg["seeds"])
        runs.insert(0, "space", sp)
        summ.insert(0, "space", sp)
        runs_all.append(runs)
        summaries[sp] = summ
        labels_by_space[sp] = labels
        selected[sp] = U.select_k(summ, km_cfg["practical_tie"])
        ref_models[sp] = U.fit_reference_kmeans(X, selected[sp]["selected_k"], ref_seed, km_cfg["n_init"])
        if not np.array_equal(ref_models[sp].labels_, labels[(selected[sp]["selected_k"], ref_seed)]):
            raise RuntimeError("reference K-Means differs from the grid run with the same seed")
    pd.concat(runs_all).to_csv(RESULTS / "clustering_runs.csv", index=False)
    pd.concat(summaries.values()).to_csv(RESULTS / "clustering_summary.csv", index=False)
    (RESULTS / "selected_k.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    log("\n[K-Means] selected K: " + "; ".join(f"{sp}: K={s['selected_k']} (sil={s['silhouette_mean']:.4f}, "
                                              f"ARI={s['ari_mean']:.4f})" for sp, s in selected.items()))
    for sp in spaces:
        s = summaries[sp]
        log(f"-- {sp}")
        log(_df_str(s[["K", "silhouette_mean", "silhouette_std", "davies_bouldin_mean",
                       "calinski_harabasz_mean", "ari_mean", "nmi_mean"]]))

    # ---------------- стабільність ----------------
    st_cfg = cfg["stability"]
    stab, stab_info = {}, {}
    for sp, X in spaces.items():
        k = selected[sp]["selected_k"]
        seed_labels = [labels_by_space[sp][(k, s)] for s in km_cfg["seeds"]]
        info = U.measure_cluster_stability(X, k, ref_models[sp], seed_labels, st_cfg["subsample_fraction"],
                                           st_cfg["n_subsample_runs"], km_cfg["n_init"])
        tab = U.object_stability_table(data.id_train, X, ref_models[sp], info["object_stability"])
        tab.to_csv(RESULTS / f"object_stability_{sp}.csv", index=False)
        stab[sp] = info["object_stability"]
        n_unst = int((stab[sp] < st_cfg["unstable_threshold"]).sum())
        stab_info[sp] = {"K": k, "seed_ari_mean": info["seed_ari_mean"], "seed_nmi_mean": info["seed_nmi_mean"],
                         "run_ari_vs_reference_mean": info["run_ari_vs_reference_mean"],
                         "mean_object_stability": float(stab[sp].mean()),
                         "n_below_threshold": n_unst, "n_below_1": int((stab[sp] < 1).sum()),
                         "n_objects": len(stab[sp]), "subsample_size": info["subsample_size"]}
        log(f"\n[stability {sp}] K={k}: seed ARI={info['seed_ari_mean']:.4f} NMI={info['seed_nmi_mean']:.4f}; "
            f"subsample runs ARI vs ref={info['run_ari_vs_reference_mean']:.4f}; mean stability="
            f"{stab[sp].mean():.4f}; objects < {st_cfg['unstable_threshold']}: {n_unst}; objects < 1: "
            f"{stab_info[sp]['n_below_1']}")
        log("10 least stable objects:")
        log(_df_str(tab.sort_values(["stability", "margin_to_other"]).head(10)))
    (RESULTS / "stability_summary.json").write_text(json.dumps(stab_info, indent=2), encoding="utf-8")

    # ---------------- зовнішній аудит (мітки відкриваються ТІЛЬКИ тут, після фіксації K) ----------------
    audit_rows, comp_rows, comp_train = [], [], {}
    for sp, Xtr_sp, Xte_sp in ((space_full, Xs_tr, Xs_te_clean), (space_pca, Z_tr, Z_te_clean)):
        rm = ref_models[sp]
        for part, lab, y in (("train", rm.labels_, data.y_train.values),
                             ("test_clean", rm.predict(Xte_sp), data.y_test.values)):
            audit_rows.append({"space": sp, "part": part, "K": rm.n_clusters, "n": len(y),
                               **U.external_audit(y, lab)})
            comp = U.cluster_composition(lab, y)
            comp.insert(0, "part", part)
            comp.insert(0, "space", sp)
            comp_rows.append(comp)
            if part == "train":
                comp_train[sp] = comp
    audit = pd.DataFrame(audit_rows)
    audit.to_csv(RESULTS / "external_audit.csv", index=False)
    comp_all = pd.concat(comp_rows)
    comp_all.to_csv(RESULTS / "cluster_composition.csv", index=False)
    log("\n[external audit]")
    log(_df_str(audit))
    log(_df_str(comp_all))

    # ---------------- контрольовані аномалії (лише test) ----------------
    batch = U.inject_controlled_anomalies(data.X_test, data.id_test, cfg)
    batch.params_table.to_csv(RESULTS / "anomaly_injections.csv", index=False)
    Xs_te = prep.transform(batch.X)
    Z_te = pca.transform(Xs_te)
    n_anom = int(batch.is_anomaly.sum())
    log(f"\n[anomalies] {n_anom} injected of {len(batch.kind)} test objects: "
        f"{pd.Series(batch.kind).value_counts().to_dict()}")

    det = U.fit_anomaly_detectors(Xs_tr, Z_tr, cfg, pca)
    sc_tr = U.score_anomalies(det, Xs_tr, Z_tr, ids=data.id_train)
    sc_te = U.score_anomalies(det, Xs_te, Z_te, ids=data.id_test)
    sc_te_clean = U.score_anomalies(det, Xs_te_clean, Z_te_clean, ids=data.id_test)
    metrics = U.evaluate_anomalies(sc_te, batch.is_anomaly)
    metrics["threshold"] = [det.thresholds[d] for d in metrics["detector"]]
    metrics.to_csv(RESULTS / "anomaly_metrics.csv", index=False)
    rbk = U.recall_by_kind(sc_te, batch.kind)
    rbk.to_csv(RESULTS / "anomaly_recall_by_kind.csv", index=False)
    clean_m = ~batch.is_anomaly
    fc = pd.DataFrame([{
        "detector": d, "threshold": det.thresholds[d],
        "train_flagged": int(sc_tr[f"flag_{d}"].sum()), "train_n": len(sc_tr),
        "test_clean_flagged": int(sc_te[f"flag_{d}"][clean_m].sum()), "test_clean_n": int(clean_m.sum()),
        "test_anomalies_flagged": int(sc_te[f"flag_{d}"][~clean_m].sum()), "test_anomalies_n": n_anom,
        "test_unmodified_flagged": int(sc_te_clean[f"flag_{d}"].sum()), "test_unmodified_n": len(sc_te_clean),
    } for d in U.DETECTORS])
    fc.to_csv(RESULTS / "flagged_counts.csv", index=False)
    log("\n[anomaly metrics]")
    log(_df_str(metrics))
    log("[recall by kind]")
    log(_df_str(rbk))
    log("[flagged counts]")
    log(_df_str(fc))

    k_top = n_anom
    jac = {"top_k_jaccard_if30_vs_ifpca": U.top_k_jaccard(sc_te["score_if30"], sc_te["score_ifpca"], k_top),
           "k": k_top}
    dis = U.if_disagreement(sc_te["flag_if30"], sc_te["flag_ifpca"], batch.is_anomaly)
    dis.to_csv(RESULTS / "if_disagreement.csv", index=False)
    log(f"\n[IF30 vs IFPCA] top-{k_top} Jaccard = {jac['top_k_jaccard_if30_vs_ifpca']:.3f}")
    log(_df_str(dis))

    ph = U.posthoc_q_sensitivity(Xs_tr, Xs_te, batch.is_anomaly, q_by_th, cfg["threshold_quantile"])
    ph.to_csv(RESULTS / "posthoc_q_sensitivity.csv", index=False)
    log("[POST-HOC q sensitivity, not used for any choice]")
    log(_df_str(ph))

    fpt = U.false_positive_table(sc_te, Xs_te, pca, feats, batch.is_anomaly, data.y_test.values)
    fpt.to_csv(RESULTS / "false_positives.csv", index=False)
    fnt = U.false_negative_table(sc_te, batch.kind, det.thresholds)
    fnt.to_csv(RESULTS / "false_negatives.csv", index=False)
    log(f"\n[FP] clean test objects flagged by rec or combined: {len(fpt)} "
        f"(malignant by audit: {int((fpt.diagnosis_audit_only == 'malignant').sum()) if len(fpt) else 0})")
    log(f"[FN] missed anomalies per detector: {fnt.groupby('detector').size().to_dict()}")

    # ---------------- артефакти ----------------
    bundle = U.make_bundle(cfg, prep, pca, ref_models[space_full], ref_models[space_pca], det, feats,
                           {sp: s["selected_k"] for sp, s in selected.items()})
    U.save_artifacts(bundle, ARTIFACTS / "lr05_bundle.joblib")
    chk_load = U.predict_new(U.load_bundle(ARTIFACTS / "lr05_bundle.joblib"), data.X_test.head(5))
    if not np.isfinite(chk_load.drop(columns=["flag_if30", "flag_ifpca", "flag_rec", "flag_combined"]).to_numpy(float)).all():
        raise RuntimeError("predict_new returned non-finite values")

    # ---------------- рисунки ----------------
    P.fig_scree(evr, thresholds, q_by_th, FIGURES / "01_scree_cumulative.png")
    P.fig_recon_vs_q(curve, q_by_th, FIGURES / "02_reconstruction_error_vs_q.png")
    P.fig_loadings(pca_full.components_, feats, FIGURES / "03_pca_loadings.png")
    P.fig_pca_2d(Z_tr, ref_models[space_pca].labels_, ref_models[space_pca].cluster_centers_, evr,
                 selected[space_pca]["selected_k"], FIGURES / "04_pca_2d_projection.png")
    P.fig_clustering_metrics(summaries, {sp: s["selected_k"] for sp, s in selected.items()},
                             FIGURES / "05_clustering_metrics.png")
    P.fig_stability(stab, st_cfg["unstable_threshold"], FIGURES / "06_object_stability.png")
    P.fig_composition(comp_train, FIGURES / "07_cluster_composition.png")
    P.fig_rec_distributions(sc_tr["score_rec"].to_numpy(), sc_te_clean["score_rec"].to_numpy(), batch.kind,
                            sc_te["score_rec"].to_numpy(), det.thresholds["rec"],
                            FIGURES / "08_reconstruction_error_distributions.png")
    P.fig_pr_curves(sc_te, batch.is_anomaly, FIGURES / "09_anomaly_pr_curves.png")
    P.fig_if_scatter(sc_te, batch.kind, det.thresholds["if30"], det.thresholds["ifpca"],
                     FIGURES / "10_if_score_scaled_vs_pca.png")
    top = np.lexsort((-sc_te["score_rec"].to_numpy(), -sc_te["score_combined"].to_numpy()))[:6]
    P.fig_top_profiles(top, Xs_te, feats, data.id_test, batch.kind, sc_te["score_combined"].to_numpy(),
                       FIGURES / "11_top_anomalies_profiles.png")

    runtime = time.perf_counter() - t_start
    log(f"\nRuntime: {runtime:.1f} s")
    return {"cfg": cfg, "log": log, "runtime": runtime, "q_by_th": q_by_th, "selected": selected,
            "stab_info": stab_info, "audit": audit, "metrics": metrics, "jaccard": jac}


def write_log_and_manifest(res: dict, warn_msgs: list) -> None:
    head = [f"python: {sys.version.split()[0]} | sklearn {sklearn.__version__} | numpy {np.__version__} | "
            f"pandas {pd.__version__} | scipy {scipy.__version__} | matplotlib {matplotlib.__version__} | "
            f"joblib {joblib.__version__}",
            f"platform: {platform.platform()}", f"runtime_s: {res['runtime']:.1f}",
            f"warnings captured during run: {len(warn_msgs)}"] + [f"  - {m}" for m in warn_msgs]
    key = [f"q per threshold: {res['q_by_th']}",
           f"selected K: { {sp: s['selected_k'] for sp, s in res['selected'].items()} }",
           f"jaccard: {res['jaccard']}", "-" * 60, "FULL CONSOLE OUTPUT:"]
    (RESULTS / "run_log.txt").write_text("\n".join(head + key + res["log"].lines) + "\n", encoding="utf-8")
    files = []
    for d in (RESULTS, FIGURES, ARTIFACTS):
        for p in sorted(d.glob("*")):
            if p.is_file() and p.name != "manifest.json":
                files.append({"path": p.relative_to(ROOT).as_posix(), "size_bytes": p.stat().st_size})
    (RESULTS / "manifest.json").write_text(json.dumps({"files": files}, indent=2), encoding="utf-8")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(U.DEFAULT_CONFIG_PATH), help="шлях до JSON-конфігурації")
    args = ap.parse_args(argv)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = run_experiment(Path(args.config))
    msgs = [f"{w.category.__name__}: {w.message}" for w in caught]
    write_log_and_manifest(res, msgs)
    for m in msgs:
        print("WARNING:", m)
    print(f"warnings: {len(msgs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
