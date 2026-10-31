import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import audit_core as A  # noqa: E402
sys.path.insert(0, str(ROOT / "fixed"))
import serving as S  # noqa: E402


def toy_schema(tol=0.15):
    feats = [{"name": "radius_mean", "min": 5.0, "max": 30.0}, {"name": "area_mean", "min": 100.0, "max": 3000.0},
             {"name": "texture_mean", "min": 1.0, "max": 40.0}]
    return {"features": feats, "domain_checks": {"area_radius_rel_tol": tol}, "max_batch": 3}


def good(**kw):
    r = {"radius_mean": 10.0, "area_mean": float(np.pi * 100), "texture_mean": 20.0}
    r.update(kw)
    return r


class FakeModel:
    def __init__(self, p):
        self.p = np.asarray(p, float)

    def predict_proba(self, X):
        return np.column_stack([1 - self.p, self.p])


class RowModel:
    """p = texture_mean / 40, so every row gets its own probability."""

    def predict_proba(self, X):
        p = np.asarray(X)[:, 2] / 40.0
        return np.column_stack([1 - p, p])


class TestData(unittest.TestCase):
    def test_overlap(self):
        self.assertEqual(A.detect_split_overlap([1, 2, 3], [3, 4]), {3})
        self.assertEqual(A.detect_split_overlap([1, 2], [3, 4]), set())

    def test_duplicates(self):
        self.assertEqual(A.detect_duplicate_rows([[1, 2], [1, 2], [3, 4]]), 1)

    def test_target_polarity_malignant_is_one(self):
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        y = A.audit_target(d.target)
        self.assertEqual(int(y.sum()), 212)
        self.assertEqual(d.target_names[0], "malignant")
        self.assertTrue(((d.target == 0) == (y == 1)).all())

    def test_correlation_signal(self):
        rng = np.random.default_rng(0)
        y = rng.integers(0, 2, 200)
        X = np.column_stack([y * 1.0, rng.normal(size=200), np.ones(200)])
        df = A.feature_target_correlations(X, y, ["proxy", "noise", "const"], 0.9).set_index("feature")
        self.assertEqual(df.index[0], "proxy")
        self.assertTrue(bool(df.loc["proxy", "signal"]))
        self.assertFalse(bool(df.loc["noise", "signal"]))
        self.assertTrue(np.isnan(df.loc["const", "r"]))

    def test_cross_distance_zero_for_identical_row(self):
        Xtr = np.array([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
        self.assertAlmostEqual(float(A.min_cross_distance(Xtr, Xtr[[1]])[0]), 0.0, places=6)

    def test_pipeline_check(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        self.assertTrue(A.pipeline_fits_inside(make_pipeline(StandardScaler(), LogisticRegression()))["preprocessing_inside"])
        self.assertFalse(A.pipeline_fits_inside(LogisticRegression())["is_pipeline"])


class TestMetrics(unittest.TestCase):
    def test_confusion_and_ap(self):
        m = A.metrics_from_proba([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9], 0.5)
        self.assertEqual(m["confusion_matrix"], [[1, 1], [1, 1]])
        self.assertAlmostEqual(m["ap"], 0.5 * 1.0 + 0.5 * (2 / 3), places=12)
        self.assertAlmostEqual(m["roc_auc"], 0.75, places=12)

    def test_evaluate_model_uses_positive_column(self):
        m = A.evaluate_model(FakeModel([0.9, 0.1, 0.8]), np.zeros((3, 1)), [1, 0, 1], 0.5)
        self.assertEqual(m["confusion_matrix"], [[1, 0], [0, 2]])
        self.assertEqual(m["recall"], 1.0)

    def test_compare_metrics_tolerance(self):
        rep = {"ap": 0.9, "confusion_matrix": [[1, 0], [0, 1]]}
        ok = A.compare_metrics(rep, {"ap": 0.9 + 1e-12, "confusion_matrix": [[1, 0], [0, 1]]}, 1e-9, keys=("ap",))
        self.assertTrue(ok.within_tol.all())
        bad = A.compare_metrics(rep, {"ap": 0.9 + 1e-6, "confusion_matrix": [[1, 0], [1, 0]]}, 1e-9, keys=("ap",))
        self.assertFalse(bad.within_tol.any())

    def test_summarize_and_bootstrap(self):
        v = [0.9, 0.95, 1.0, 0.97]
        s = A.summarize(v)
        self.assertAlmostEqual(s["std"], float(np.std(v, ddof=1)))
        lo, hi = A.bootstrap_ci(v, 2000, 0.05, 1)
        self.assertLessEqual(lo, s["mean"])
        self.assertGreaterEqual(hi, s["mean"])
        self.assertEqual((lo, hi), A.bootstrap_ci(v, 2000, 0.05, 1))
        self.assertEqual(A.bootstrap_ci([0.5, 0.5, 0.5], 100, 0.05, 1), (0.5, 0.5))

    def test_bootstrap_metric_ci_is_clipped_to_unit_interval(self):
        """Regression: on the real lab06 test split the AP interval had an upper bound of 1.0000000000000002."""
        import joblib
        import pandas as pd
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        y = A.audit_target(d.target)
        split = pd.read_csv(ROOT.parent / "lab06" / "handoff" / "split_ids.csv")
        te = split[split.part == "test"].id.to_numpy()
        model = joblib.load(ROOT.parent / "lab06" / "handoff" / "pipeline.joblib")
        p = model.predict_proba(d.data[te])[:, 1]
        for metric in ("ap", "roc_auc", "recall"):
            lo, hi = A.bootstrap_metric_ci(y[te], p, metric, n_boot=400, seed=42)
            self.assertLessEqual(hi, 1.0, metric)
            self.assertGreaterEqual(lo, 0.0, metric)

    def test_disagreement_and_dp(self):
        self.assertEqual(A.prediction_disagreement([0, 1, 1, 0], [0, 1, 0, 1]), 0.5)
        self.assertAlmostEqual(A.mean_abs_proba_diff([0.1, 0.5], [0.3, 0.4]), 0.15)
        with self.assertRaises(ValueError):
            A.prediction_disagreement([0, 1], [0])

    def test_ranking_stability(self):
        M = np.tile(np.arange(10.0), (4, 1))
        r = A.ranking_stability(M, 3)
        self.assertAlmostEqual(r["mean_spearman"], 1.0)
        self.assertAlmostEqual(r["mean_topk_jaccard"], 1.0)
        M2 = np.array([np.arange(10.0), np.arange(10.0)[::-1]])
        self.assertAlmostEqual(A.ranking_stability(M2, 3)["mean_spearman"], -1.0)

    def test_error_objects(self):
        ids = np.arange(6)
        y = [1, 1, 0, 0, 1, 0]
        p = [0.2, 0.9, 0.6, 0.1, 0.55, 0.45]
        df = A.select_error_objects(ids, y, p, 0.5, n_near=2, n_conf_err=1)
        self.assertEqual(sorted(df[df.category == "error"].id), [0, 2])
        self.assertEqual(set(df[df.category == "near_threshold"].id), {4, 5})
        self.assertEqual(df[df.category == "confident_error"].id.tolist(), [0])


class TestClaimsFindings(unittest.TestCase):
    claim = {"id": "C-X", "criteria": [{"name": "a", "key": "a", "op": "le", "value": 1.0}, {"name": "b", "key": "b", "op": "true"}]}

    def test_verdicts(self):
        self.assertEqual(A.verify_claim(self.claim, {"a": 0.5, "b": True})["verdict"], "підтверджено")
        self.assertEqual(A.verify_claim(self.claim, {"a": 0.5, "b": False})["verdict"], "підтверджено частково")
        self.assertEqual(A.verify_claim(self.claim, {"a": 2.0, "b": False})["verdict"], "спростовано")
        self.assertEqual(A.verify_claim(self.claim, {})["verdict"], "неможливо перевірити")

    def test_claim_needs_criteria(self):
        with self.assertRaises(ValueError):
            A.verify_claim({"id": "x", "criteria": []}, {})

    def test_finding_validation(self):
        f = A.build_finding("p", "e", "high", "r", fid="F-1")
        self.assertEqual(f["status"], "open")
        with self.assertRaises(ValueError):
            A.build_finding("p", "e", "urgent")
        with self.assertRaises(ValueError):
            A.build_finding("p", "", "low")

    def test_verdict_rule(self):
        def mk(sev, st="open"):
            return {"severity": sev, "status": st}
        self.assertEqual(A.decide_verdict([mk("critical")]), "відхилено")
        self.assertEqual(A.decide_verdict([mk("high"), mk("low")]), "потребує доопрацювання")
        self.assertEqual(A.decide_verdict([mk("high", "fixed"), mk("medium", "accepted_risk")]), "прийнято з обмеженнями")
        self.assertEqual(A.decide_verdict([mk("high", "fixed")]), "прийнято")


class TestServing(unittest.TestCase):
    def test_valid_and_reordered(self):
        sc = toy_schema()
        self.assertEqual(S.validate_record(good(), sc), ("accepted", []))
        shuffled = dict(reversed(list(good().items())))
        self.assertEqual(S.validate_record(shuffled, sc)[0], "accepted")
        self.assertEqual(S.record_to_vector(shuffled, sc), S.record_to_vector(good(), sc))

    def test_rejections(self):
        sc = toy_schema()
        r = good()
        del r["texture_mean"]
        st, msgs = S.validate_record(r, sc)
        self.assertEqual(st, "rejected")
        self.assertIn("texture_mean", msgs[0])
        self.assertEqual(S.validate_record(good(extra=1.0), sc)[0], "rejected")
        for bad in ("12", None, True, [1.0], float("nan"), float("inf"), -1.0):
            self.assertEqual(S.validate_record(good(texture_mean=bad), sc)[0], "rejected", bad)
        self.assertEqual(S.validate_record([1, 2, 3], sc)[0], "rejected")

    def test_numpy_scalars_accepted(self):
        rec = {k: np.float64(v) for k, v in good().items()}
        self.assertEqual(S.validate_record(rec, toy_schema())[0], "accepted")

    def test_warnings(self):
        sc = toy_schema()
        st, msg = S.validate_record(good(radius_mean=100.0, area_mean=float(np.pi * 1e4)), sc)
        self.assertEqual(st, "accepted_with_warning")
        self.assertTrue(any("outside train range" in m for m in msg))
        st, msg = S.validate_record(good(area_mean=2000.0), sc)
        self.assertEqual(st, "accepted_with_warning")
        self.assertTrue(any("area_mean/(pi" in m for m in msg))

    def test_predict_records_batch(self):
        sc = toy_schema()
        broken = good()
        del broken["area_mean"]
        out = S.predict_records(RowModel(), [good(texture_mean=30.0), broken, good(texture_mean=10.0)], sc)
        self.assertEqual([o["status"] for o in out], ["accepted", "rejected", "accepted"])
        self.assertIsNone(out[1]["probability"])
        self.assertEqual((out[0]["label"], out[2]["label"]), (1, 0))
        with self.assertRaises(ValueError):
            S.predict_records(RowModel(), [], sc)
        with self.assertRaises(ValueError):
            S.predict_records(RowModel(), [good()] * 4, sc)

    def test_checksum_gate(self):
        import joblib
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            joblib.dump({"a": 1}, t / "m.joblib")
            (t / "m.sha256").write_text(A.sha256_file(t / "m.joblib") + "  m.joblib\n")
            self.assertEqual(S.load_verified_model(t / "m.joblib", t / "m.sha256"), {"a": 1})
            joblib.dump({"a": 2}, t / "m.joblib")
            with self.assertRaises(S.IntegrityError):
                S.load_verified_model(t / "m.joblib", t / "m.sha256")
            (t / "m.sha256").write_text("zz")
            with self.assertRaises(S.IntegrityError):
                S.load_verified_model(t / "m.joblib", t / "m.sha256")


class TestManifest(unittest.TestCase):
    def test_sha256_and_manifest(self):
        with tempfile.TemporaryDirectory() as t:
            p = Path(t) / "a.txt"
            p.write_bytes(b"abc")
            self.assertEqual(A.sha256_file(p), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")
            self.assertEqual(A.file_manifest([p], t)["a.txt"]["bytes"], 3)

    def test_inventory(self):
        with tempfile.TemporaryDirectory() as t:
            (Path(t) / "a").write_text("x")
            self.assertEqual(A.inventory(t, ["a", "b"]).present.tolist(), [True, False])


if __name__ == "__main__":
    unittest.main()
