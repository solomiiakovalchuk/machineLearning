import json
import os
import shutil
import subprocess
import sys
import unittest

from support import PROJECT, control_example, make_project_copy, schema_feature_names

CONFIG = str(PROJECT / "configs" / "project_config.json")


def run_cli(*args, config=CONFIG, cwd=None):
    env = dict(os.environ, PYTHONPATH=str(PROJECT / "src"), PYTHONIOENCODING="utf-8")
    return subprocess.run([sys.executable, "-m", "ml_project.cli", "--config", config, *args],
                          capture_output=True, text=True, env=env, cwd=cwd, timeout=120)


class CliTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_project_copy()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.example = control_example()

    def write(self, name, text):
        p = self.tmp / name
        p.write_text(text, encoding="utf-8")
        return str(p)

    def test_metadata_smoke(self):
        r = run_cli("metadata")
        self.assertEqual(r.returncode, 0, r.stderr)
        d = json.loads(r.stdout)
        self.assertEqual(d["n_features"], 30)
        self.assertEqual(r.stderr, "")

    def test_predict_control_example_file(self):
        r = run_cli("predict", str(PROJECT / "configs" / "inference_example.json"))
        self.assertEqual(r.returncode, 0, r.stderr)
        out = json.loads(r.stdout)
        self.assertAlmostEqual(out["probability"], self.example["expected_probability"][0], places=12)
        self.assertEqual(out["predicted_class"], "malignant")

    def test_predict_bare_record_and_validate(self):
        path = self.write("rec.json", json.dumps(self.example["record"]))
        self.assertEqual(run_cli("predict", path).returncode, 0)
        r = run_cli("validate", path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout), {"valid": True, "records": 1, "warnings": [[]]})

    def test_predict_batch_json_list_and_csv(self):
        names = schema_feature_names()
        path = self.write("batch.json", json.dumps(self.example["records"]))
        r = run_cli("predict-batch", path)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(json.loads(r.stdout)["count"], 3)
        rows = [",".join(names)] + [",".join(repr(rec[n]) for n in names) for rec in self.example["records"]]
        csv_path = self.write("batch.csv", "\n".join(rows) + "\n")
        rc = run_cli("predict-batch", csv_path)
        self.assertEqual(rc.returncode, 0, rc.stderr)
        for got, want in zip(json.loads(rc.stdout)["results"], self.example["expected_probability"]):
            self.assertAlmostEqual(got["probability"], want, places=12)

    def test_error_exit_codes_are_distinct_and_messages_on_stderr(self):
        rec = self.example["record"]
        missing = {k: v for k, v in rec.items() if k != "area_mean"}
        cases = {
            "input_validation": (self.write("a.json", json.dumps(missing)), "predict", 5),
            "nan_literal": (self.write("b.json", json.dumps(rec).replace(str(rec["area_mean"]), "NaN")), "predict", 5),
            "bad_json": (self.write("c.json", "{oops"), "predict", 4),
            "missing_file": (str(self.tmp / "nope.json"), "predict", 4),
            "single_to_batch": (self.write("d.json", json.dumps(rec)), "predict-batch", 5),
            "list_to_predict": (self.write("e.json", json.dumps([rec])), "predict", 5),
        }
        for label, (path, cmd, code) in cases.items():
            with self.subTest(label):
                r = run_cli(cmd, path)
                self.assertEqual(r.returncode, code, r.stderr)
                self.assertEqual(r.stdout, "")
                self.assertTrue(r.stderr.startswith("error["), r.stderr)
                self.assertNotIn("Traceback", r.stderr)

    def test_batch_too_large_has_own_exit_code(self):
        cfg = json.loads((self.tmp / "configs" / "project_config.json").read_text())
        cfg["max_batch_size"] = 2
        config_path = self.write("configs/project_config.json", json.dumps(cfg))
        path = self.write("big.json", json.dumps(self.example["records"]))
        r = run_cli("predict-batch", path, config=config_path)
        self.assertEqual(r.returncode, 6)
        self.assertIn("max_batch_size", r.stderr)

    def test_bad_config_exit_code(self):
        r = run_cli("metadata", config=str(self.tmp / "configs" / "missing.json"))
        self.assertEqual(r.returncode, 3)
        cfg = json.loads((self.tmp / "configs" / "project_config.json").read_text())
        cfg["paths"]["model"] = "../evil.joblib"
        config_path = self.write("configs/project_config.json", json.dumps(cfg))
        r = run_cli("metadata", config=config_path)
        self.assertEqual(r.returncode, 3)
        self.assertIn("'..'", r.stderr)

    def test_tampered_model_exit_code(self):
        path = self.tmp / "models" / "pipeline.joblib"
        path.write_bytes(path.read_bytes() + b"\x00")
        r = run_cli("metadata", config=str(self.tmp / "configs" / "project_config.json"))
        self.assertEqual(r.returncode, 7)
        self.assertIn("model hash", r.stderr)

    def test_usage_error_exit_code(self):
        self.assertEqual(run_cli("no-such-command").returncode, 2)

    def test_csv_problems(self):
        names = schema_feature_names()
        rec = self.example["record"]
        header = ",".join(names)
        row = ",".join(repr(rec[n]) for n in names)
        bad_cell = row.replace(repr(rec["radius_mean"]), "abc", 1)
        cases = {"non_numeric": (header + "\n" + bad_cell + "\n", 5),
                 "short_row": (header + "\n" + ",".join(row.split(",")[:-1]) + "\n", 5),
                 "empty_cell": (header + "\n" + "," + ",".join(row.split(",")[1:]) + "\n", 5),
                 "nan_cell": (header + "\n" + row.replace(repr(rec["radius_mean"]), "nan", 1) + "\n", 5),
                 "empty_file": ("", 5)}
        for label, (text, code) in cases.items():
            with self.subTest(label):
                r = run_cli("predict-batch", self.write(label + ".csv", text))
                self.assertEqual(r.returncode, code, r.stderr)

    def test_duplicate_json_keys_are_rejected_not_last_wins(self):
        # json.loads keeps the LAST duplicate silently: the first (invalid) radius_mean would vanish.
        rec = self.example["record"]
        body = json.dumps(rec)
        dup = body.replace("{", '{"radius_mean": 1000.0, ', 1)
        r = run_cli("predict", self.write("dup.json", dup))
        self.assertEqual(r.returncode, 4, r.stdout + r.stderr)
        self.assertIn("duplicate key", r.stderr)
        self.assertEqual(r.stdout, "")

    def test_config_with_duplicate_key_is_rejected(self):
        text = (self.tmp / "configs" / "project_config.json").read_text()
        path = self.write("configs/project_config.json", text.replace('"threshold": 0.5,', '"threshold": 0.5, "threshold": 0.5,'))
        r = run_cli("metadata", config=path)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("duplicate key", r.stderr)

    def test_verify_command(self):
        r = run_cli("verify")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertTrue(json.loads(r.stdout)["manifest_ok"])

    def test_validate_reports_out_of_range_warnings(self):
        rec = dict(self.example["record"], radius_mean=900.0)
        r = run_cli("validate", self.write("w.json", json.dumps(rec)))
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(len(json.loads(r.stdout)["warnings"][0]), 1)


if __name__ == "__main__":
    unittest.main()
