import json
import pickle
import shutil
import unittest
from unittest import mock

from support import CONFIG_PATH, control_example, make_project_copy, read_json

from ml_project.config import load_project_config
from ml_project.errors import ArtifactIntegrityError, BatchSizeError, ConfigError, InputValidationError
from ml_project.predictor import ProjectPredictor, sha256_file

RESPONSE_KEYS = {"probability", "predicted_class", "threshold", "positive_class",
                 "model_version", "schema_version", "warnings"}


class PredictorIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_project_config(CONFIG_PATH)
        cls.predictor = ProjectPredictor(cls.config)
        cls.example = control_example()

    def test_control_prediction_regression(self):
        tol = self.config.probability_tolerance
        for record, expected, cls_name in zip(self.example["records"], self.example["expected_probability"],
                                              self.example["expected_predicted_class"]):
            out = self.predictor.predict_record(record)
            self.assertLessEqual(abs(out["probability"] - expected), tol)
            self.assertEqual(out["predicted_class"], cls_name)

    def test_response_contract(self):
        out = self.predictor.predict_record(self.example["record"])
        self.assertEqual(set(out), RESPONSE_KEYS)
        self.assertIsInstance(out["probability"], float)
        self.assertEqual(out["positive_class"], "malignant")
        self.assertEqual(out["threshold"], 0.5)
        self.assertEqual(out["warnings"], [])
        text = json.dumps(out)
        self.assertNotIn("joblib", text.lower())
        self.assertNotIn(":\\", text)

    def test_json_field_order_is_irrelevant(self):
        record = self.example["record"]
        reversed_record = dict(reversed(list(record.items())))
        sorted_record = dict(sorted(record.items()))
        base = self.predictor.predict_record(record)["probability"]
        self.assertEqual(self.predictor.predict_record(reversed_record)["probability"], base)
        self.assertEqual(self.predictor.predict_record(sorted_record)["probability"], base)

    def test_single_and_batch_agree(self):
        batch = self.predictor.predict_batch(self.example["records"])
        self.assertEqual(len(batch), 3)
        for record, from_batch in zip(self.example["records"], batch):
            single = self.predictor.predict_record(record)
            self.assertLessEqual(abs(single["probability"] - from_batch["probability"]), self.config.probability_tolerance)

    def test_out_of_range_warns_but_predicts(self):
        record = dict(self.example["record"], radius_mean=900.0)
        out = self.predictor.predict_record(record)
        self.assertEqual(len(out["warnings"]), 1)
        self.assertIn("radius_mean", out["warnings"][0])
        self.assertTrue(0.0 <= out["probability"] <= 1.0)

    def test_invalid_input_blocked(self):
        record = dict(self.example["record"])
        del record["radius_mean"]
        with self.assertRaises(InputValidationError):
            self.predictor.predict_record(record)
        with self.assertRaises(BatchSizeError):
            self.predictor.predict_batch([self.example["record"]] * (self.config.max_batch_size + 1))

    def test_describe(self):
        d = self.predictor.describe()
        self.assertEqual(d["n_features"], 30)
        self.assertEqual(len(d["feature_names"]), 30)
        self.assertEqual(d["positive_class"], "malignant")
        self.assertEqual(d["family"], "lrl2")
        self.assertEqual(d["candidate_id"], "automl_seed42_trial10")
        self.assertEqual(d["model_sha256"], sha256_file(self.config.paths["model"]))
        self.assertNotIn("/", json.dumps(d).replace("n/a", ""))  # no filesystem paths
        json.dumps(d)


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_project_copy()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.config_path = self.tmp / "configs" / "project_config.json"

    def load(self):
        return ProjectPredictor(load_project_config(self.config_path))

    def test_untouched_copy_loads(self):
        self.assertEqual(self.load().describe()["n_features"], 30)

    def test_tampered_model_rejected_before_unpickling(self):
        marker = self.tmp / "pwned.txt"

        class Evil:
            def __reduce__(self):
                return (open, (str(marker), "w"))

        (self.tmp / "models" / "pipeline.joblib").write_bytes(pickle.dumps(Evil()))
        with mock.patch("joblib.load") as joblib_load:
            with self.assertRaisesRegex(ArtifactIntegrityError, "model hash"):
                self.load()
            joblib_load.assert_not_called()
        # also without mocking: really unpickling the payload would create the marker file
        with self.assertRaises(ArtifactIntegrityError):
            self.load()
        self.assertFalse(marker.exists())

    def test_one_flipped_byte_in_model_rejected(self):
        path = self.tmp / "models" / "pipeline.joblib"
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 0x01
        path.write_bytes(bytes(data))
        with self.assertRaises(ArtifactIntegrityError):
            self.load()

    def test_tampered_schema_rejected(self):
        path = self.tmp / "configs" / "schema.json"
        path.write_bytes(path.read_bytes().replace(b'"radius_mean"', b'"radius_meen"', 1))
        with self.assertRaisesRegex(ArtifactIntegrityError, "schema"):
            self.load()

    def test_tampered_frozen_decision_rejected(self):
        path = self.tmp / "configs" / "frozen_decision.json"
        path.write_bytes(path.read_bytes().replace(b'"C": 0.1', b'"C": 0.2', 1))
        with self.assertRaisesRegex(ArtifactIntegrityError, "frozen_decision"):
            self.load()

    def test_threshold_change_in_config_is_not_a_silent_tweak(self):
        raw = read_json(self.config_path)
        raw["threshold"] = 0.3
        self.config_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "threshold"):
            self.load()

    def test_version_mismatch_between_config_and_metadata(self):
        raw = read_json(self.config_path)
        raw["model_version"] = "9.9.9"
        self.config_path.write_text(json.dumps(raw), encoding="utf-8")
        with self.assertRaisesRegex(ConfigError, "model_version"):
            self.load()

    def test_missing_model_file_is_controlled_error(self):
        (self.tmp / "models" / "pipeline.joblib").unlink()
        with self.assertRaises(ArtifactIntegrityError):
            self.load()


class HashTests(unittest.TestCase):
    def test_sha256_file_known_value(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "x.bin"
            p.write_bytes(b"abc")
            self.assertEqual(sha256_file(p), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
