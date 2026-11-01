"""Contract tests over the shipped artifacts: metadata, manifest, handoff consistency."""
import json
import shutil
import sys
import unittest

import numpy as np

from support import PROJECT, make_project_copy, read_json

from ml_project.errors import ArtifactIntegrityError
from ml_project.integrity import sha256_file, verify_manifest

LAB06_HANDOFF = PROJECT.parent / "lab06" / "handoff"


class MetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.meta = read_json(PROJECT / "models" / "model_metadata.json")

    def test_hashes_match_files(self):
        self.assertEqual(self.meta["model_sha256"], sha256_file(PROJECT / "models" / "pipeline.joblib"))
        self.assertEqual(self.meta["schema_sha256"], sha256_file(PROJECT / "configs" / "schema.json"))
        self.assertEqual(self.meta["frozen_decision_sha256"], sha256_file(PROJECT / "configs" / "frozen_decision.json"))
        self.assertEqual(self.meta["split_ids_sha256"], sha256_file(PROJECT / "data" / "split_ids.csv"))

    def test_required_fields(self):
        for key in ("model_version", "schema_version", "family", "candidate_id", "intended_use", "limitations",
                    "environment", "threshold", "positive_class"):
            self.assertIn(key, self.meta)
        for lib in ("python", "scikit-learn", "numpy", "joblib"):
            self.assertIn(lib, self.meta["environment"])

    def test_frozen_decision_is_the_lab06_record(self):
        self.assertEqual(self.meta["frozen_decision_sha256"],
                         "62caba19558d196bdefb4b1b459d8d6ba1809f5d1f0aa9b7450401380a5a584c")
        decision = read_json(PROJECT / "configs" / "frozen_decision.json")
        self.assertEqual((decision["family"], decision["candidate_id"]), ("lrl2", "automl_seed42_trial10"))
        self.assertFalse(decision["test_metrics_present"])


class SchemaTests(unittest.TestCase):
    def test_schema_and_example_are_consistent(self):
        schema = read_json(PROJECT / "configs" / "schema.json")
        example = read_json(PROJECT / "configs" / "inference_example.json")
        names = [f["name"] for f in schema["features"]]
        self.assertEqual(len(names), 30)
        for rec in example["records"]:
            self.assertEqual(set(rec), set(names))
        self.assertEqual(example["record"], example["records"][0])
        self.assertEqual(len(example["expected_probability"]), len(example["records"]))

    @unittest.skipUnless(LAB06_HANDOFF.is_dir(), "lab06 handoff not next to lab07 (e.g. extracted archive)")
    def test_feature_order_equals_lab06_handoff(self):
        lab06 = [f["name"] for f in read_json(LAB06_HANDOFF / "data_schema.json")["features"]]
        mine = [f["name"] for f in read_json(PROJECT / "configs" / "schema.json")["features"]]
        self.assertEqual(lab06, mine)

    @unittest.skipUnless(LAB06_HANDOFF.is_dir(), "lab06 handoff not next to lab07")
    def test_model_and_ids_equal_lab06_handoff(self):
        self.assertEqual(sha256_file(LAB06_HANDOFF / "pipeline.joblib"), sha256_file(PROJECT / "models" / "pipeline.joblib"))
        self.assertEqual(sha256_file(LAB06_HANDOFF / "split_ids.csv"), sha256_file(PROJECT / "data" / "split_ids.csv"))
        self.assertEqual(sha256_file(LAB06_HANDOFF / "decision_record.json"),
                         sha256_file(PROJECT / "configs" / "frozen_decision.json"))

    def test_schema_ranges_equal_train_part_only(self):
        from sklearn.datasets import load_breast_cancer
        import csv
        X = load_breast_cancer().data
        with open(PROJECT / "data" / "split_ids.csv", newline="") as fh:
            train = [int(r["id"]) for r in csv.DictReader(fh) if r["part"] == "train"]
        schema = read_json(PROJECT / "configs" / "schema.json")
        lo = np.array([f["min"] for f in schema["features"]])
        hi = np.array([f["max"] for f in schema["features"]])
        np.testing.assert_allclose(lo, X[train].min(axis=0), rtol=1e-9)
        np.testing.assert_allclose(hi, X[train].max(axis=0), rtol=1e-9)

    def test_handoff_acceptance_report_has_no_failed_hard_check(self):
        report = read_json(PROJECT / "reports" / "handoff_acceptance.json")
        soft = {"schema_range_equals_all_data_min_max"}  # false by design: ranges are train-only
        failed = [k for k, v in report["checks"].items() if v is not True and k not in soft]
        self.assertEqual(failed, [])
        self.assertTrue(report["gaps"])  # the missing model checksum is recorded, not hidden


class ReproducibilityReportTests(unittest.TestCase):
    def test_saved_report_is_a_pass_with_stated_tolerance(self):
        rep = read_json(PROJECT / "reports" / "reproducibility.json")
        cfg = read_json(PROJECT / "configs" / "project_config.json")
        self.assertTrue(rep["passed"])
        self.assertEqual(rep["tolerances"]["probability_abs"], cfg["numeric_tolerance"]["probability_abs"])
        self.assertLessEqual(rep["max_abs_diff_saved_vs_rebuilt_test"], rep["tolerances"]["probability_abs"])


class ManifestTests(unittest.TestCase):
    def test_manifest_matches_files(self):
        result = verify_manifest(PROJECT)
        self.assertGreaterEqual(result["checked"], 20)

    def test_manifest_lists_required_files(self):
        files = read_json(PROJECT / "project_manifest.json")["files"]
        for rel in ("models/pipeline.joblib", "configs/schema.json", "configs/project_config.json",
                    "src/ml_project/predictor.py", "configs/frozen_decision.json", "models/model_metadata.json"):
            self.assertIn(rel, files)
        self.assertEqual(files["models/pipeline.joblib"], sha256_file(PROJECT / "models" / "pipeline.joblib"))
        self.assertEqual(files["configs/frozen_decision.json"], sha256_file(PROJECT / "configs" / "frozen_decision.json"))

    def test_manifest_detects_tampering_and_missing_file(self):
        tmp = make_project_copy()
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        manifest = read_json(PROJECT / "project_manifest.json")
        for rel in manifest["files"]:  # copy every listed file so only the tamper differs
            dst = tmp / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copyfile(PROJECT / rel, dst)
        self.assertEqual(verify_manifest(tmp)["checked"], len(manifest["files"]))
        target = tmp / "src" / "ml_project" / "predictor.py"
        target.write_bytes(target.read_bytes() + b"# edited\n")
        with self.assertRaisesRegex(ArtifactIntegrityError, "predictor.py"):
            verify_manifest(tmp)
        target.unlink()
        with self.assertRaisesRegex(ArtifactIntegrityError, "missing"):
            verify_manifest(tmp)


if __name__ == "__main__":
    unittest.main()
