"""Integration tests: completeness and consistency of the audit artifacts (run pr07_audit.py first)."""
import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
import audit_core as A  # noqa: E402

R, F = ROOT / "results", ROOT / "figures"
H = REPO / "lab06" / "handoff"
CFG = json.loads((ROOT / "configs" / "audit_config.json").read_text(encoding="utf-8"))


def csv(name):
    return pd.read_csv(R / name)


class TestRequiredArtifacts(unittest.TestCase):
    def test_files_exist(self):
        for name in ("claims.csv", "findings.csv", "verdict.json", "independent_metrics.csv", "reproduction_delta.csv", "repeated_cv_summary.csv",
                     "error_objects.csv", "negative_tests.csv", "stability_runs.csv", "leakage_checks.csv", "dataflow.csv", "inventory.csv",
                     "ai_recommendation_checks.csv", "fixed_evaluation.json", "summary.json"):
            self.assertTrue((R / name).is_file(), name)
        self.assertTrue((ROOT / "artifacts" / "file_manifest.json").is_file())
        self.assertGreaterEqual(len(list(F.glob("*.png"))), 8)

    def test_claims_registry(self):
        c = csv("claims.csv")
        self.assertGreaterEqual(len(c), 5)
        for col in ("id", "claim", "source", "independent_check", "acceptance_criteria", "evidence", "verdict"):
            self.assertFalse(c[col].isna().any(), col)
        self.assertTrue(c.verdict.isin(A.CLAIM_VERDICTS).all())
        self.assertEqual(c.id.nunique(), len(c))

    def test_findings_registry(self):
        f = csv("findings.csv")
        self.assertGreaterEqual(len(f), 3)
        for col in ("id", "severity", "problem", "evidence", "recommendation", "status"):
            self.assertFalse(f[col].isna().any(), col)
        self.assertTrue(f.severity.isin(A.SEVERITIES).all())
        self.assertEqual(f.id.nunique(), len(f))

    def test_verdict_machine_readable_and_consistent(self):
        v = json.loads((R / "verdict.json").read_text(encoding="utf-8"))
        self.assertIn(v["verdict"], A.AUDIT_VERDICTS)
        self.assertIn(v["verdict_after_fixes"], A.AUDIT_VERDICTS)
        f = csv("findings.csv")
        recs = f.to_dict("records")
        self.assertEqual(v["verdict"], A.decide_verdict([{**r, "status": "open"} for r in recs]))
        self.assertEqual(v["verdict_after_fixes"], A.decide_verdict(recs))
        self.assertEqual(v["claims"], csv("claims.csv").verdict.value_counts().to_dict())
        self.assertEqual(v["findings_by_severity"], f.severity.value_counts().to_dict())
        self.assertEqual(set(v["basis"]["finding_ids"]), set(f.id))

    def test_manifest_hashes_match_files(self):
        m = json.loads((ROOT / "artifacts" / "file_manifest.json").read_text(encoding="utf-8"))
        self.assertGreater(len(m["files"]), 20)
        for rel, info in m["files"].items():
            self.assertEqual(A.sha256_file(ROOT / rel), info["sha256"], rel)
        for name, info in m["frozen_lab06_handoff"].items():
            self.assertEqual(A.sha256_file(H / name), info["sha256"], name)


class TestFrozenObject(unittest.TestCase):
    def test_frozen_files_match_commit(self):
        f = csv("frozen_object.csv")
        self.assertEqual(len(f), len(CFG["object"]["handoff_files"]))
        self.assertFalse((f.worktree_vs_commit == "different").any())
        self.assertFalse(f.worktree_vs_commit.isna().any())
        s = json.loads((R / "summary.json").read_text(encoding="utf-8"))["frozen_info"]
        self.assertTrue(s["all_blobs_match_frozen_commit"])
        self.assertEqual(s["frozen_commit"], CFG["object"]["frozen_commit"])

    def test_decision_hash_file_equals_decision_bytes(self):
        self.assertEqual(A.sha256_file(H / "decision_record.json"), (H / "decision.sha256").read_text().strip())


class TestMetricsArtifacts(unittest.TestCase):
    def test_reproduction_within_tolerance(self):
        d = csv("reproduction_delta.csv")
        self.assertTrue(d.within_tol.all())
        self.assertLessEqual(d[d.metric != "confusion_matrix"].abs_delta.max(), CFG["tolerances"]["metric"])

    def test_control_prediction_from_saved_model(self):
        ex = json.loads((H / "inference_example.json").read_text(encoding="utf-8"))
        p = joblib.load(H / "pipeline.joblib").predict_proba(np.asarray(ex["input"], float))[:, 1]
        np.testing.assert_allclose(p, ex["expected_probability"], rtol=1e-10, atol=1e-12)

    def test_independent_metrics_vs_reported(self):
        m = csv("independent_metrics.csv")
        row = m[(m.model == "frozen_joblib") & (m.split == "test")].iloc[0]
        rep = json.loads((H / "test_metrics.json").read_text(encoding="utf-8"))
        self.assertAlmostEqual(row.ap, rep["ap"], places=12)
        self.assertEqual([[row.tn, row.fp], [row.fn, row.tp]], rep["confusion_matrix"])
        dummy = m[(m.model == "dummy_prior") & (m.split == "test")].iloc[0]
        self.assertAlmostEqual(dummy.roc_auc, 0.5)
        self.assertLess(dummy.ap, 0.5)

    def test_repeated_cv_shape_and_intervals(self):
        f = csv("repeated_cv_folds.csv")
        self.assertEqual(len(f[f.model == "frozen_config"]), CFG["repeated_cv"]["n_splits"] * CFG["repeated_cv"]["n_repeats"])
        s = csv("repeated_cv_summary.csv")
        s = s[(s.model == "frozen_config")]
        self.assertTrue(((s.ci_low <= s["mean"]) & (s["mean"] <= s.ci_high)).all())
        self.assertTrue(((s["min"] <= s["mean"]) & (s["mean"] <= s["max"])).all())

    def test_summary_matches_fold_table(self):
        f = csv("repeated_cv_folds.csv")
        s = csv("repeated_cv_summary.csv")
        for model in f.model.unique():
            a = f[f.model == model].ap
            r = s[(s.model == model) & (s.metric == "ap")].iloc[0]
            self.assertAlmostEqual(r["mean"], a.mean(), places=12)
            self.assertAlmostEqual(r["std"], a.std(ddof=1), places=12)


class TestSplitAndLeakage(unittest.TestCase):
    def test_split_ids(self):
        s = pd.read_csv(H / "split_ids.csv")
        tr, te = s[s.part == "train"].id, s[s.part == "test"].id
        self.assertEqual((len(tr), len(te)), (455, 114))
        self.assertEqual(A.detect_split_overlap(tr, te), set())
        self.assertEqual(set(tr) | set(te), set(range(569)))

    def test_leakage_checks_recorded(self):
        c = csv("leakage_checks.csv").set_index("check")
        self.assertTrue(bool(c.loc["id overlap train/test", "ok"]))
        self.assertFalse(bool(c.loc["pipeline.joblib SHA-256 is recorded anywhere in the handoff", "ok"]))

    def test_data_audit_json(self):
        d = json.loads((R / "data_audit.json").read_text(encoding="utf-8"))
        self.assertEqual((d["n"], d["positives"], d["train_n"], d["test_n"]), (569, 212, 455, 114))
        self.assertTrue(d["feature_names_match_schema"])
        self.assertTrue(d["raw_sha256_matches_provenance"])


class TestErrorsStabilityNegative(unittest.TestCase):
    def test_error_objects_consistent_with_confusion_matrix(self):
        e = csv("error_objects.csv")
        self.assertGreaterEqual(len(e), 10)
        t = e[(e["set"] == "test") & e.is_error]
        rep = json.loads((H / "test_metrics.json").read_text(encoding="utf-8"))
        cm = rep["confusion_matrix"]
        self.assertEqual(len(t), cm[0][1] + cm[1][0])
        self.assertEqual(int((t.type == "FN").sum()), cm[1][0])
        self.assertEqual(int((t.type == "FP").sum()), cm[0][1])
        self.assertGreaterEqual(int(e.category.str.contains("near_threshold").sum()), CFG["error_analysis"]["n_near_threshold"])
        self.assertTrue(e.category.str.contains("confident_error").any())

    def test_stability_runs(self):
        s = csv("stability_runs.csv")
        self.assertGreaterEqual(len(s), 10)
        self.assertTrue(s.disagreement_D.between(0, 1).all())
        self.assertTrue((s.mean_abs_dp >= 0).all())
        imp = csv("importance_summary.csv")
        self.assertEqual(len(imp), 30)

    def test_negative_tests_cover_required_cases(self):
        n = csv("negative_tests.csv")
        need = {"missing_feature", "extra_feature", "wrong_type_text", "nan_value", "inf_value", "reordered_fields", "out_of_range_x10_radius",
                "domain_area_inconsistent_with_radius"}
        self.assertTrue(need <= set(n.scenario))
        self.assertTrue(n.hardened_ok.all())
        self.assertFalse(n.legacy_ok.all())

    def test_fixed_version_does_not_change_predictions(self):
        e = json.loads((R / "fixed_evaluation.json").read_text(encoding="utf-8"))
        self.assertEqual(e["max_abs_dp_hardened_vs_original"], 0.0)
        self.assertTrue(e["labels_identical"])
        self.assertTrue(e["metrics_identical"])
        self.assertTrue(e["identical_to_handoff"])
        self.assertEqual(A.sha256_file(ROOT / "fixed" / "pipeline.joblib"), A.sha256_file(H / "pipeline.joblib"))
        self.assertEqual((ROOT / "fixed" / "pipeline.sha256").read_text().split()[0], A.sha256_file(H / "pipeline.joblib"))

    def test_unit_interval_metrics_in_ci_file(self):
        ci = json.loads((R / "test_bootstrap_ci.json").read_text(encoding="utf-8"))
        for k in ("ap", "roc_auc", "recall"):
            self.assertLessEqual(ci[k][1], 1.0, k)
            self.assertLessEqual(ci[k][0], ci[k][1])

    def test_selection_finding_wording_matches_interval(self):
        """Regression: the finding must not call a difference 'noise' when its bootstrap interval excludes zero."""
        s = json.loads((R / "summary.json").read_text(encoding="utf-8"))["repeated_cv_paired"]
        f = csv("findings.csv")
        row = f[f.problem.str.contains("практично довільний")].iloc[0]
        excludes_zero = s["paired_diff_ci_high"] < 0 or s["paired_diff_ci_low"] > 0
        if excludes_zero:
            self.assertIn("систематично", row.evidence)
            self.assertNotIn("в межах шуму", row.evidence)
            self.assertNotIn("підтверджує довільність", row.evidence)

    def test_ai_checks_at_least_three(self):
        self.assertGreaterEqual(len(csv("ai_recommendation_checks.csv")), 3)


if __name__ == "__main__":
    unittest.main()
