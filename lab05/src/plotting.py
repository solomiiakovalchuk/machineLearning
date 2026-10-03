"""Побудова рисунків LR5 (PNG, dpi 120). Кожна функція приймає готові таблиці/масиви."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score, precision_recall_curve  # noqa: E402

DPI = 120
KIND_COLORS = {"clean": "#9aa0a6", "calibration_scale": "#d95f02", "geometry_break": "#7570b3",
               "worst_mean_swap": "#1b9e77", "texture_noise": "#e7298a"}
DET_LABELS = {"if30": "IF (30-вим.)", "ifpca": "IF (PCA)", "rec": "Помилка відновлення PCA",
              "combined": "Комбінований (max перцентиль)"}


def _save(fig, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)


def fig_scree(evr, thresholds, q_by_threshold, path):
    cum = np.cumsum(evr)
    x = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x, evr, color="#4c78a8", label="Частка дисперсії компоненти")
    ax.set_xlabel("Номер головної компоненти")
    ax.set_ylabel("Частка поясненої дисперсії")
    ax2 = ax.twinx()
    ax2.plot(x, cum, "o-", color="#e45756", ms=4, label="Кумулятивна частка")
    for th in thresholds:
        q = q_by_threshold[th]
        ax2.axhline(th, ls="--", lw=0.8, color="#555555")
        ax2.plot([q], [cum[q - 1]], "s", color="black", ms=8)
        ax2.annotate(f"{th:.2f}: q={q}", (q, cum[q - 1]), textcoords="offset points", xytext=(8, -14), fontsize=9)
    ax2.set_ylabel("Кумулятивна дисперсія")
    ax2.set_ylim(0, 1.02)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="center right")
    ax.set_title("Scree-графік PCA (train, 30 ознак) та пороги дисперсії")
    _save(fig, path)


def fig_recon_vs_q(curve: pd.DataFrame, q_by_threshold, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(curve["q"], curve["mse_mean"], "o-", label="Середня помилка")
    ax.plot(curve["q"], curve["mse_p95"], "s-", label="95-й перцентиль")
    for th, q in q_by_threshold.items():
        ax.axvline(q, ls=":", color="gray")
        ax.text(q, ax.get_ylim()[1] * 0.9, f" {th:.2f}\n q={q}", fontsize=8, va="top")
    ax.set_xlabel("Кількість компонент q")
    ax.set_ylabel("MSE відновлення (стандартизований простір)")
    ax.set_title("Помилка відновлення train залежно від q")
    ax.legend()
    _save(fig, path)


def fig_loadings(components, feature_names, path):
    L = np.asarray(components)[:3].T
    fig, ax = plt.subplots(figsize=(6, 10))
    im = ax.imshow(L, cmap="RdBu_r", vmin=-0.45, vmax=0.45, aspect="auto")
    ax.set_xticks(range(3), ["PC1", "PC2", "PC3"])
    ax.set_yticks(range(len(feature_names)), feature_names, fontsize=8)
    for i in range(L.shape[0]):
        for j in range(3):
            ax.text(j, i, f"{L[i, j]:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="Навантаження")
    ax.set_xlabel("Головна компонента")
    ax.set_ylabel("Ознака")
    ax.set_title("Навантаження перших трьох компонент")
    _save(fig, path)


def fig_pca_2d(Z, labels, centers, evr, k, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for c in np.unique(labels):
        m = labels == c
        ax.scatter(Z[m, 0], Z[m, 1], s=14, color=cmap(int(c)), alpha=0.7, label=f"Кластер {c} (n={int(m.sum())})")
    ax.scatter(centers[:, 0], centers[:, 1], marker="X", s=200, c="black", edgecolors="white", label="Центроїди")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Train у площині PC1/PC2, K-Means K={k} у PCA-просторі\n"
                 f"(PC1+PC2 = {100 * (evr[0] + evr[1]):.1f}% дисперсії)")
    ax.legend()
    _save(fig, path)


def fig_clustering_metrics(summaries: dict, selected: dict, path):
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    spec = [("silhouette", "Silhouette (вище краще)", "silhouette_mean", "silhouette_std"),
            ("davies_bouldin", "Davies-Bouldin (нижче краще)", "davies_bouldin_mean", "davies_bouldin_std"),
            ("calinski_harabasz", "Calinski-Harabasz (вище краще)", "calinski_harabasz_mean", "calinski_harabasz_std"),
            ("ari", "Середній попарний ARI між seed", "ari_mean", "ari_std")]
    for ax, (_, title, m, s) in zip(axes.ravel(), spec):
        for space, df in summaries.items():
            ax.errorbar(df["K"], df[m], yerr=df[s], marker="o", capsize=3, label=space)
            ax.axvline(selected[space], ls=":", lw=0.8, color="gray")
        ax.set_title(title)
        ax.set_xlabel("K")
        ax.set_ylabel(title.split(" (")[0])
        ax.legend()
    fig.suptitle("Метрики K-Means: scaled30 vs PCA (середнє та std по seed)")
    _save(fig, path)


def fig_stability(stab_by_space: dict, threshold, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(0, 1, 21)
    for space, s in stab_by_space.items():
        ax.hist(s, bins=bins, alpha=0.55, label=f"{space} (n={len(s)})")
    ax.axvline(threshold, color="red", ls="--", label=f"Поріг нестабільності {threshold}")
    ax.set_yscale("log")
    ax.set_xlabel("Об'єктна стабільність (частка прогонів із тією ж міткою)")
    ax.set_ylabel("Кількість об'єктів (лог. шкала)")
    ax.set_title("Об'єктна стабільність кластерів (20 підвибірок)")
    ax.legend()
    _save(fig, path)


def fig_composition(comp_by_space: dict, path):
    fig, axes = plt.subplots(1, len(comp_by_space), figsize=(5.5 * len(comp_by_space), 5), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (space, comp) in zip(axes, comp_by_space.items()):
        x = np.arange(len(comp))
        ax.bar(x, comp["benign"], color="#4c78a8", label="benign")
        ax.bar(x, comp["malignant"], bottom=comp["benign"], color="#e45756", label="malignant")
        ax.set_xticks(x, [str(c) for c in comp["cluster"]])
        for xi, (b, m) in enumerate(zip(comp["benign"], comp["malignant"])):
            ax.text(xi, b + m + 3, f"{m / (b + m):.0%} зл.", ha="center", fontsize=9)
        ax.set_xlabel("Кластер")
        ax.set_title(space)
        ax.legend()
    axes[0].set_ylabel("Кількість об'єктів (train)")
    fig.suptitle("Склад кластерів за діагнозом (лише зовнішній аудит)")
    _save(fig, path)


def fig_rec_distributions(train_s, test_clean_s, test_kind, test_s, thr, path):
    groups = [("train", train_s), ("test (чисті)", test_clean_s)]
    for kd in [k for k in dict.fromkeys(test_kind) if k != "clean"]:
        groups.append((kd, test_s[test_kind == kd]))
    fig, ax = plt.subplots(figsize=(9, 5.5))
    data = [g[1] for g in groups]
    ax.boxplot(data, orientation="horizontal", tick_labels=[g[0] for g in groups], showfliers=False)
    rng = np.random.default_rng(0)
    for i, (name, v) in enumerate(groups, 1):
        col = KIND_COLORS.get(name, "#9aa0a6")
        ax.scatter(v, i + rng.uniform(-0.18, 0.18, len(v)), s=10, color=col, alpha=0.6, zorder=3)
    ax.axvline(thr, color="red", ls="--", label=f"Поріг (95% train) = {thr:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("Помилка відновлення PCA (MSE, лог. шкала)")
    ax.set_ylabel("Група")
    ax.set_title("Розподіл помилки відновлення: train / чисті test / аномалії")
    ax.legend()
    _save(fig, path)


def fig_pr_curves(scores_df, y, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    y = np.asarray(y).astype(int)
    for d, lab in DET_LABELS.items():
        s = scores_df[f"score_{d}"].to_numpy()
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{lab} (AP={average_precision_score(y, s):.2f})")
    ax.axhline(y.mean(), color="gray", ls=":", label=f"Випадковий ({y.mean():.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_ylim(0, 1.02)
    ax.set_title("PR-криві детекторів на test з контрольованими аномаліями")
    ax.legend(loc="lower left", fontsize=9)
    _save(fig, path)


def fig_if_scatter(scores_df, kind, thr30, thrpca, path):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    for kd in ["clean"] + [k for k in dict.fromkeys(kind) if k != "clean"]:
        m = np.asarray(kind) == kd
        ax.scatter(scores_df["score_if30"][m], scores_df["score_ifpca"][m], s=40 if kd != "clean" else 18,
                   color=KIND_COLORS.get(kd, "k"), alpha=0.85 if kd != "clean" else 0.5,
                   edgecolors="k" if kd != "clean" else "none", linewidths=0.4,
                   label="чисті" if kd == "clean" else kd)
    ax.axvline(thr30, color="red", ls="--", lw=0.9, label="поріг IF30")
    ax.axhline(thrpca, color="blue", ls="--", lw=0.9, label="поріг IF-PCA")
    ax.set_xlabel("Скор IF у 30-вимірному просторі (вище = аномальніше)")
    ax.set_ylabel("Скор IF у PCA-просторі")
    ax.set_title("Isolation Forest: scaled30 vs PCA (test)")
    ax.legend(fontsize=8)
    _save(fig, path)


def fig_top_profiles(top_idx, X_scaled, feature_names, ids, kinds, scores, path):
    n = len(top_idx)
    fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 11), sharex=True)
    for ax, i in zip(axes.ravel(), top_idx):
        z = X_scaled[i]
        ax.barh(range(len(z)), z, color=np.where(z >= 0, "#e45756", "#4c78a8"))
        ax.set_yticks(range(len(z)), feature_names, fontsize=6)
        ax.invert_yaxis()
        ax.axvline(0, color="k", lw=0.6)
        lab = "чистий" if kinds[i] == "clean" else kinds[i]
        ax.set_title(f"id={ids[i]} [{lab}]\nскор={scores[i]:.3f}", fontsize=9)
        ax.set_xlabel("z-значення ознаки")
    axes.ravel()[0].set_ylabel("Ознака")
    fig.suptitle("Профілі топ-6 об'єктів за комбінованим скором (test)")
    _save(fig, path)
