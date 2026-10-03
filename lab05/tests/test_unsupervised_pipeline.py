"""Модульні тести LR5 (unittest): python -m unittest discover -s tests -v"""
import copy
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import unsupervised_pipeline as U  # noqa: E402

BUNDLE_PATH = ROOT / "artifacts" / "lr05_bundle.joblib"
FEATURES = [f"{b}_{a}" for a in U.AGGREGATES for b in U.BASE_FEATURES]


def base_config():
    cfg = U.load_config()
    cfg["isolation_forest"]["n_estimators"] = 30     # швидко для тестів
    return cfg


def synthetic_raw(n=60, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.uniform(0.5, 10.0, size=(n, len(FEATURES))), columns=FEATURES,
                        index=np.arange(n) + 100)


class TestChooseNComponents(unittest.TestCase):
    def test_exact_boundary(self):
        evr = [0.5, 0.3, 0.2]
        self.assertEqual(U.choose_n_components(evr, 0.5), 1)     # рівно на порозі
        self.assertEqual(U.choose_n_components(evr, 0.51), 2)
        self.assertEqual(U.choose_n_components(evr, 0.8), 2)     # 0.5+0.3 з похибкою float
        self.assertEqual(U.choose_n_components(evr, 0.81), 3)

    def test_threshold_one_with_float_noise(self):
        evr = [0.1] * 10                                          # cumsum[-1] = 0.9999999999999999
        self.assertEqual(U.choose_n_components(evr, 1.0), 10)

    def test_invalid_threshold(self):
        for bad in (0, -0.1, 1.01, float("nan"), True, "0.9"):
            with self.assertRaises(ValueError, msg=f"threshold={bad!r}"):
                U.choose_n_components([0.6, 0.4], bad)

    def test_invalid_evr(self):
        with self.assertRaises(ValueError):
            U.choose_n_components([[0.5, 0.5]], 0.9)             # не 1-D
        with self.assertRaises(ValueError):
            U.choose_n_components([0.9, -0.1], 0.5)              # від'ємне
        with self.assertRaises(ValueError):
            U.choose_n_components([0.7, 0.7], 0.5)               # сума > 1
        with self.assertRaises(ValueError):
            U.choose_n_components([], 0.5)                       # порожній
        with self.assertRaises(ValueError):
            U.choose_n_components([0.2, 0.2], 0.9)               # не досягає порога


class TestPCAAndPreprocessing(unittest.TestCase):
    def test_transform_inverse_shapes(self):
        X = np.random.default_rng(1).normal(size=(40, 8))
        p = U.fit_pca(X, 3)
        Z = p.transform(X)
        self.assertEqual(Z.shape, (40, 3))
        self.assertEqual(p.inverse_transform(Z).shape, X.shape)
        e = U.reconstruction_errors(p, X)
        self.assertEqual(e.shape, (40,))
        self.assertTrue((e >= 0).all())

    def test_full_pca_reconstruction_is_exact(self):
        X = np.random.default_rng(2).normal(size=(30, 5))
        e = U.reconstruction_errors(U.fit_pca(X, 5), X)
        self.assertLess(e.max(), 1e-20)

    def test_curve_matches_pca_reconstruction(self):
        X = np.random.default_rng(3).normal(size=(50, 6))
        curve = U.reconstruction_error_curve(U.fit_pca(X, 6), X)
        direct = U.reconstruction_errors(U.fit_pca(X, 3), X)
        self.assertAlmostEqual(curve.loc[curve.q == 3, "mse_mean"].item(), direct.mean(), places=10)

    def test_preprocessor_fitted_on_train_only(self):
        tr, te = synthetic_raw(60, 1), synthetic_raw(20, 2) * 5      # test з іншим розподілом
        pipe = U.fit_preprocessor(tr, base_config())
        scaler = pipe.named_steps["scaler"]
        np.testing.assert_allclose(scaler.mean_, np.log1p(tr.values).mean(axis=0))
        before = scaler.mean_.copy()
        out = pipe.transform(te)                                      # transform не змінює параметри
        np.testing.assert_array_equal(scaler.mean_, before)
        self.assertEqual(out.shape, te.shape)
        self.assertGreater(np.abs(out.mean(axis=0)).max(), 0.1)       # test НЕ центрований за своїм середнім

    def test_log1p_inverse_roundtrip_and_negative_values(self):
        tr = synthetic_raw(30)
        pipe = U.fit_preprocessor(tr, base_config())
        back = pipe.named_steps["log1p"].inverse_transform(np.log1p(tr.values))
        np.testing.assert_allclose(back, tr.values)
        bad = tr.copy()
        bad.iloc[0, 0] = -0.5
        with self.assertRaises(ValueError):
            U.fit_preprocessor(bad, base_config())


class TestSplit(unittest.TestCase):
    def test_ids_disjoint_and_complete(self):
        d = U.load_dataset(base_config())
        self.assertEqual(len(d.id_train) + len(d.id_test), 569)
        self.assertFalse(set(d.id_train) & set(d.id_test))
        self.assertEqual(set(d.id_train) | set(d.id_test), set(range(569)))
        self.assertEqual(len(d.id_test), 114)
        self.assertNotIn("target", d.X_train.columns)               # мітка не в ознаках
        self.assertEqual(list(d.X_train.index), list(d.y_train.index))
        self.assertIn("concave_points_worst", d.feature_names)

    def test_check_split_errors(self):
        with self.assertRaises(ValueError):
            U.check_split([0, 1, 2], [2, 3], 4)                       # перетин
        with self.assertRaises(ValueError):
            U.check_split([0, 1], [3], 4)                             # не повне покриття


class TestClustering(unittest.TestCase):
    def test_ari_permutation_invariant(self):
        a = np.array([0, 0, 1, 1, 2, 2, 2])
        perm = np.array([2, 0, 1])[a]
        self.assertAlmostEqual(adjusted_rand_score(a, perm), 1.0)

    def test_align_recovers_permutation(self):
        ref = np.array([0, 0, 1, 1, 2, 2, 2, 0])
        cand = np.array([2, 0, 1])[ref]                               # переставлені мітки
        np.testing.assert_array_equal(U.align_cluster_labels(ref, cand), ref)

    def test_align_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            U.align_cluster_labels([0, 1], [0, 1, 1])

    def test_object_stability(self):
        ref = np.array([0, 0, 1, 1, 0, 1])
        self.assertTrue((U.compute_object_stability(ref, [ref, ref, ref]) == 1).all())
        swapped = 1 - ref                                             # той самий поділ, інші назви
        self.assertTrue((U.compute_object_stability(ref, [ref, swapped]) == 1).all())
        diff = ref.copy()
        diff[0] = 1
        st = U.compute_object_stability(ref, [ref, diff])
        self.assertEqual(st[0], 0.5)
        self.assertEqual(st[1:].min(), 1.0)
        with self.assertRaises(ValueError):
            U.compute_object_stability(ref, [])

    def test_select_k_rule(self):
        s = pd.DataFrame({"K": [2, 3, 4], "silhouette_mean": [0.40, 0.395, 0.30], "ari_mean": [0.80, 0.95, 1.0]})
        self.assertEqual(U.select_k(s, 0.01)["selected_k"], 3)        # практична рівність -> вищий ARI
        self.assertEqual(U.select_k(s, 0.001)["selected_k"], 2)       # без рівності -> max silhouette
        s["ari_mean"] = 1.0
        self.assertEqual(U.select_k(s, 0.01)["selected_k"], 2)        # рівний ARI -> менший K

    def test_evaluate_clustering_shapes(self):
        rng = np.random.default_rng(0)
        X = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(6, 1, (30, 3))])
        runs, labels = U.evaluate_clustering(X, [2, 3], [1, 2, 3], n_init=2)
        self.assertEqual(len(runs), 6)
        self.assertEqual(len(labels), 6)
        summ = U.summarize_clustering(runs, labels, [1, 2, 3])
        self.assertAlmostEqual(summ.loc[summ.K == 2, "ari_mean"].item(), 1.0)
        self.assertEqual(U.select_k(summ, 0.01)["selected_k"], 2)


class TestAnomalyInjection(unittest.TestCase):
    def setUp(self):
        self.cfg = base_config()
        self.cfg["anomalies"]["count_per_type"] = 3
        self.X = synthetic_raw(60)
        self.ids = self.X.index.to_numpy()

    def test_deterministic_and_seed_sensitive(self):
        a = U.inject_controlled_anomalies(self.X, self.ids, self.cfg)
        b = U.inject_controlled_anomalies(self.X, self.ids, self.cfg)
        pd.testing.assert_frame_equal(a.X, b.X)
        np.testing.assert_array_equal(a.kind, b.kind)
        cfg2 = copy.deepcopy(self.cfg)
        cfg2["anomalies"]["seed"] = 7
        c = U.inject_controlled_anomalies(self.X, self.ids, cfg2)
        self.assertFalse(np.array_equal(a.kind, c.kind))

    def test_input_not_modified_and_counts(self):
        before = self.X.copy()
        a = U.inject_controlled_anomalies(self.X, self.ids, self.cfg)
        pd.testing.assert_frame_equal(self.X, before)
        self.assertEqual(int(a.is_anomaly.sum()), 12)
        self.assertEqual(pd.Series(a.kind[a.is_anomaly]).value_counts().tolist(), [3, 3, 3, 3])
        self.assertEqual(len(a.params_table), 12)
        self.assertNotIn("diagnosis", " ".join(a.params_table.columns))
        changed = (a.X.values != self.X.values).any(axis=1)
        np.testing.assert_array_equal(changed, a.is_anomaly)          # змінені лише обрані рядки

    def test_effects_per_kind(self):
        a = U.inject_controlled_anomalies(self.X, self.ids, self.cfg)
        f = self.cfg["anomalies"]["kinds"]["calibration_scale"]["factor"]
        m = a.kind == "calibration_scale"
        np.testing.assert_allclose(a.X.loc[m, "radius_mean"], self.X.loc[m, "radius_mean"] * f)
        np.testing.assert_allclose(a.X.loc[m, "area_worst"], self.X.loc[m, "area_worst"] * f * f)
        m = a.kind == "geometry_break"
        np.testing.assert_allclose(a.X.loc[m, "area_mean"], self.X.loc[m, "area_mean"] * 0.4)
        np.testing.assert_allclose(a.X.loc[m, "radius_mean"], self.X.loc[m, "radius_mean"])
        m = a.kind == "worst_mean_swap"
        np.testing.assert_allclose(a.X.loc[m, "texture_mean"], self.X.loc[m, "texture_worst"])
        np.testing.assert_allclose(a.X.loc[m, "texture_worst"], self.X.loc[m, "texture_mean"])
        m = a.kind == "texture_noise"
        np.testing.assert_allclose(a.X.loc[m, "radius_mean"], self.X.loc[m, "radius_mean"])
        self.assertFalse(np.allclose(a.X.loc[m, "texture_mean"], self.X.loc[m, "texture_mean"]))

    def test_too_many_anomalies(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["anomalies"]["count_per_type"] = 15                       # 60 >= n=60
        with self.assertRaises(ValueError):
            U.inject_controlled_anomalies(self.X, self.ids, cfg)

    def test_unknown_kind_and_missing_columns(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["anomalies"]["kinds"]["bogus"] = {}
        with self.assertRaises(ValueError):
            U.inject_controlled_anomalies(self.X, self.ids, cfg)
        with self.assertRaises(ValueError):
            U.inject_controlled_anomalies(self.X.drop(columns=["area_mean"]), self.ids, self.cfg)


class TestDetectors(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(0)
        cls.cfg = base_config()
        cls.Xtr = rng.normal(size=(200, 8))
        cls.pca = U.fit_pca(cls.Xtr, 3)
        cls.Ztr = cls.pca.transform(cls.Xtr)
        cls.det = U.fit_anomaly_detectors(cls.Xtr, cls.Ztr, cls.cfg, cls.pca)
        Xte = rng.normal(size=(40, 8))
        Xte[:8] *= 6                                                  # явні викиди
        cls.Xte = Xte
        cls.scores = U.score_anomalies(cls.det, Xte, cls.pca.transform(Xte))
        cls.y = np.r_[np.ones(8), np.zeros(32)].astype(bool)

    def test_score_columns_and_percentile_range(self):
        for d in U.DETECTORS:
            self.assertIn(f"score_{d}", self.scores)
            self.assertIn(f"flag_{d}", self.scores)
        c = self.scores["score_combined"]
        self.assertTrue(((c >= 0) & (c <= 1)).all())
        tr = self.det.train_scores["combined"]
        self.assertTrue(((tr >= 0) & (tr <= 1)).all())

    def test_threshold_is_train_quantile(self):
        for d in U.DETECTORS:
            self.assertAlmostEqual(self.det.thresholds[d], np.quantile(self.det.train_scores[d], 0.95))
        share = (self.det.train_scores["rec"] > self.det.thresholds["rec"]).mean()
        self.assertLessEqual(share, 0.06)

    def test_empirical_percentile(self):
        s = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_allclose(U.empirical_percentile(s, [0.5, 2.0, 10.0]), [0.0, 0.5, 1.0])

    def test_metric_ranges(self):
        m = U.evaluate_anomalies(self.scores, self.y)
        self.assertEqual(list(m.detector), list(U.DETECTORS))
        for col in ("roc_auc", "average_precision", "precision", "recall", "f1", "precision_at_k"):
            self.assertTrue(((m[col] >= 0) & (m[col] <= 1)).all(), col)
        self.assertTrue((m.tp + m.fp + m.fn + m.tn == len(self.y)).all())
        self.assertGreater(m.set_index("detector").loc["rec", "roc_auc"], 0.9)  # викиди 6 sigma

    def test_evaluate_requires_both_classes(self):
        with self.assertRaises(ValueError):
            U.evaluate_anomalies(self.scores, np.zeros(len(self.scores)))

    def test_score_rejects_nan(self):
        bad = self.Xte.copy()
        bad[0, 0] = np.nan
        with self.assertRaises(ValueError):
            U.score_anomalies(self.det, bad, np.zeros((len(bad), 3)))

    def test_recall_by_kind_and_disagreement(self):
        kind = np.array(["a"] * 4 + ["b"] * 4 + ["clean"] * 32)
        r = U.recall_by_kind(self.scores, kind)
        self.assertEqual(list(r.kind), ["a", "b"])
        t = U.if_disagreement([1, 1, 0, 0], [1, 0, 1, 0], [1, 1, 1, 0])
        self.assertEqual(t.loc[t.group == "anomalies", ["both", "only_if30", "only_ifpca", "none"]].values.tolist(),
                         [[1, 1, 1, 0]])
        self.assertEqual(int(t.loc[t.group == "clean", "none"].item()), 1)


class TestJaccard(unittest.TestCase):
    def test_known_value(self):
        a, b = [5, 4, 3, 2, 1], [1, 2, 3, 4, 5]
        self.assertEqual(U.top_k_jaccard(a, b, 2), 0.0)               # {0,1} vs {3,4}
        self.assertAlmostEqual(U.top_k_jaccard(a, b, 3), 1 / 5)       # {0,1,2} vs {2,3,4}
        self.assertEqual(U.top_k_jaccard(a, a, 4), 1.0)

    def test_invalid_k(self):
        for k in (0, 6, 2.5):
            with self.assertRaises(ValueError):
                U.top_k_jaccard([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], k)


class TestConfigValidation(unittest.TestCase):
    def test_valid_config(self):
        U.validate_config(U.load_config(), n_test=114)

    def _bad(self, mutate, n_test=None):
        cfg = U.load_config()
        mutate(cfg)
        with self.assertRaises(ValueError):
            U.validate_config(cfg, n_test=n_test)

    def test_bad_working_threshold(self):
        self._bad(lambda c: c["pca"].update(working_threshold=0.85))

    def test_too_few_k(self):
        self._bad(lambda c: c["kmeans"].update(k_values=[2, 3, 4]))

    def test_other_errors(self):
        self._bad(lambda c: c["kmeans"].update(k_values=[2, 2, 3, 4, 5, 6, 7]))
        self._bad(lambda c: c["kmeans"].update(seeds=[1, 2, 3]))
        self._bad(lambda c: c["split"].update(test_size=1.0))
        self._bad(lambda c: c["stability"].update(subsample_fraction=0))
        self._bad(lambda c: c["pca"].update(variance_thresholds=[0.8, 1.2], working_threshold=0.8))
        self._bad(lambda c: c["preprocessing"].update(scaler="zscore"))
        self._bad(lambda c: c.update(threshold_quantile=1.0))
        self._bad(lambda c: c["anomalies"]["kinds"].update(bogus={}))
        self._bad(lambda c: c["anomalies"].update(count_per_type=30), n_test=114)   # 120 >= 114


class TestArtifacts(unittest.TestCase):
    @unittest.skipUnless(BUNDLE_PATH.exists(), "артефакти ще не створені (запустіть lr05_experiment.py)")
    def test_bundle_loads_and_predicts(self):
        bundle = U.load_bundle(BUNDLE_PATH)
        for key in ("config", "preprocessor", "pca", "kmeans_scaled30", "kmeans_pca", "if30", "ifpca",
                    "train_scores", "thresholds", "feature_names", "version"):
            self.assertIn(key, bundle)
        data = U.load_dataset(bundle["config"])
        new = data.X_test.head(5)
        out = U.predict_new(bundle, new)
        self.assertEqual(len(out), 5)
        num = out[[c for c in out.columns if c.startswith("score_")]].to_numpy()
        self.assertEqual(num.shape, (5, 4))
        self.assertTrue(np.isfinite(num).all())
        self.assertTrue(out["cluster_pca"].isin(range(bundle["selected_k"]["pca6"])).all())
        with self.assertRaises(ValueError):
            U.predict_new(bundle, new.drop(columns=["radius_mean"]))
        with self.assertRaises(FileNotFoundError):
            U.load_bundle(ROOT / "artifacts" / "no_such.joblib")


if __name__ == "__main__":
    unittest.main()
