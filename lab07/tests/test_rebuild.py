import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from support import CONFIG_PATH, PROJECT, make_project_copy, read_json

from ml_project import train
from ml_project.config import load_project_config
from ml_project.errors import RebuildError


class RebuildTests(unittest.TestCase):
    """Fits the frozen spec once (about a second) and checks every reproducibility claim."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="lab07_rebuild_"))
        cls.out = cls.tmp / "rebuilt.joblib"
        cls.config = load_project_config(CONFIG_PATH)
        cls.report = train.rebuild_model(PROJECT, cls.out)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_rebuild_passed_and_wrote_file(self):
        self.assertTrue(self.report["passed"])
        self.assertTrue(self.out.is_file())

    def test_uses_frozen_ids_only(self):
        self.assertEqual((self.report["n_train"], self.report["n_test"]), (455, 114))
        self.assertTrue(self.report["train_test_disjoint"])

    def test_probabilities_equal_saved_within_tolerance(self):
        tol = self.config.probability_tolerance
        self.assertLessEqual(self.report["control_records"]["max_abs_diff_saved_vs_rebuilt"], tol)
        self.assertLessEqual(self.report["max_abs_diff_saved_vs_rebuilt_test"], tol)
        self.assertLessEqual(self.report["max_abs_diff_saved_vs_rebuilt_all569"], tol)
        self.assertLessEqual(self.report["max_abs_diff_coef"], tol)

    def test_control_probabilities_equal_handoff_expectation(self):
        ctrl = self.report["control_records"]
        self.assertLessEqual(ctrl["max_abs_diff_rebuilt_vs_expected"], self.config.probability_tolerance)

    def test_reference_test_metric_reproduced(self):
        ap = self.report["test_ap"]
        self.assertLessEqual(ap["abs_diff_rebuilt_vs_reference"], self.config.metric_tolerance)
        self.assertAlmostEqual(ap["rebuilt"], 0.9965986394557824, places=12)
        self.assertEqual(self.report["test_confusion_matrix"]["rebuilt"], [[72, 0], [2, 40]])

    def test_report_is_json_serialisable(self):
        json.dumps(self.report)

    def test_shipped_model_is_not_overwritten(self):
        with self.assertRaisesRegex(RebuildError, "differ"):
            train.rebuild_model(PROJECT, self.config.paths["model"])

    def test_no_search_machinery_is_used(self):
        self.assertNotIn("flaml", sys.modules)
        from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
        with mock.patch.object(GridSearchCV, "fit", side_effect=AssertionError("search")), \
                mock.patch.object(RandomizedSearchCV, "fit", side_effect=AssertionError("search")):
            train.rebuild_model(PROJECT, self.tmp / "again.joblib")
        self.assertNotIn("flaml", sys.modules)

    def test_wrong_target_mapping_would_be_detected(self):
        """Sensitivity check: fitting with the flipped label (benign=1) must NOT satisfy the reference."""
        from sklearn.metrics import average_precision_score
        data = train.load_frozen_inputs(PROJECT)
        X, y, tr, te = data["X"], data["y"], data["train_ids"], data["test_ids"]
        meta = data["metadata"]
        flipped = train.build_pipeline(data["decision"]["params"], meta["fixed_model_params"], 42).fit(X[tr], 1 - y[tr])
        ap = average_precision_score(y[te], flipped.predict_proba(X[te])[:, 1])
        self.assertGreater(abs(ap - meta["reference_test_metrics"]["ap"]), 0.1)

    def test_target_is_malignant_is_one(self):
        data = train.load_frozen_inputs(PROJECT)
        self.assertEqual(int(data["y"].sum()), 212)  # 212 malignant in the dataset provenance
        self.assertEqual(int(data["y"][data["test_ids"]].sum()), data["metadata"]["reference_test_metrics"]["positive_n"])


class RebuildGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = make_project_copy()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_modified_split_ids_rejected(self):
        path = self.tmp / "data" / "split_ids.csv"
        path.write_text(path.read_text().replace(",train", ",test", 1), encoding="utf-8")
        with self.assertRaisesRegex(RebuildError, "split_ids"):
            train.rebuild_model(self.tmp, self.tmp / "x.joblib")

    def test_other_family_is_refused_not_searched(self):
        meta_path = self.tmp / "models" / "model_metadata.json"
        dec_path = self.tmp / "configs" / "frozen_decision.json"
        dec = read_json(dec_path)
        dec["family"] = "rf"
        dec_path.write_text(json.dumps(dec), encoding="utf-8")
        import hashlib
        meta = read_json(meta_path)
        meta["frozen_decision_sha256"] = hashlib.sha256(dec_path.read_bytes()).hexdigest()
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        with self.assertRaisesRegex(RebuildError, "never searches"):
            train.rebuild_model(self.tmp, self.tmp / "x.joblib")


if __name__ == "__main__":
    unittest.main()
