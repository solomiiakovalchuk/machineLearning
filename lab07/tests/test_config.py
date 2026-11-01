import json
import shutil
import unittest

from support import CONFIG_PATH, PROJECT, make_project_copy, read_json

from ml_project.config import ProjectConfig, load_project_config
from ml_project.errors import ConfigError


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = make_project_copy()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)
        self.path = self.tmp / "configs" / "project_config.json"
        self.raw = read_json(CONFIG_PATH)

    def write(self, raw):
        self.path.write_text(json.dumps(raw), encoding="utf-8")
        return load_project_config(self.path)

    def test_real_config_loads_and_is_immutable(self):
        cfg = load_project_config(CONFIG_PATH)
        self.assertIsInstance(cfg, ProjectConfig)
        self.assertEqual(cfg.project_root, PROJECT.resolve())
        self.assertEqual(cfg.threshold, 0.5)
        self.assertEqual(cfg.probability_tolerance, 1e-12)
        with self.assertRaises(Exception):
            cfg.threshold = 0.9
        with self.assertRaises(TypeError):
            cfg.paths["model"] = PROJECT

    def test_paths_are_inside_project(self):
        cfg = load_project_config(CONFIG_PATH)
        for p in cfg.paths.values():
            self.assertTrue(p.is_relative_to(cfg.project_root))

    def test_missing_required_field(self):
        del self.raw["threshold"]
        with self.assertRaisesRegex(ConfigError, "threshold"):
            self.write(self.raw)

    def test_unknown_field_rejected(self):
        self.raw["password"] = "x"
        with self.assertRaisesRegex(ConfigError, "unknown"):
            self.write(self.raw)

    def test_bad_values(self):
        cases = [("threshold", 0.0), ("threshold", 1.0), ("threshold", 1.5), ("threshold", True),
                 ("threshold", "0.5"), ("threshold", float("nan")),
                 ("max_batch_size", 0), ("max_batch_size", -1), ("max_batch_size", 2.5),
                 ("max_batch_size", True), ("max_batch_size", 10**9),
                 ("positive_class", ""), ("random_state", -1), ("model_version", 1)]
        for key, value in cases:
            with self.subTest(key=key, value=value):
                raw = dict(self.raw)
                raw[key] = value
                with self.assertRaises(ConfigError):
                    self.write(raw)

    def test_same_positive_and_negative_class(self):
        self.raw["negative_class"] = self.raw["positive_class"]
        with self.assertRaises(ConfigError):
            self.write(self.raw)

    def test_bad_tolerance(self):
        for tol in ({"probability_abs": 0.1, "metric_abs": 1e-9}, {"probability_abs": 1e-12},
                    {"probability_abs": -1e-12, "metric_abs": 1e-9}):
            with self.subTest(tol=tol):
                raw = dict(self.raw, numeric_tolerance=tol)
                with self.assertRaises(ConfigError):
                    self.write(raw)

    def test_path_traversal_and_absolute_paths_rejected(self):
        bad = ["../outside.joblib", "models/../../outside.joblib", "/etc/passwd", "C:\\Windows\\x",
               "C:/Windows/x", "..\\x", "\\\\server\\share\\x", "", 5]
        for value in bad:
            with self.subTest(path=value):
                raw = json.loads(json.dumps(self.raw))
                raw["paths"]["model"] = value
                with self.assertRaises(ConfigError):
                    self.write(raw)

    def test_symlink_escape_rejected(self):
        outside = self.tmp.parent / "lab07_outside_dir"
        outside.mkdir(exist_ok=True)
        self.addCleanup(shutil.rmtree, outside, ignore_errors=True)
        link = self.tmp / "escape"
        try:
            link.symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks not permitted on this system")
        raw = json.loads(json.dumps(self.raw))
        raw["paths"]["model"] = "escape/model.joblib"
        with self.assertRaisesRegex(ConfigError, "outside"):
            self.write(raw)

    def test_unreadable_and_malformed_files(self):
        with self.assertRaises(ConfigError):
            load_project_config(self.tmp / "configs" / "nope.json")
        self.path.write_text("{not json", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_project_config(self.path)
        self.path.write_text("[]", encoding="utf-8")
        with self.assertRaises(ConfigError):
            load_project_config(self.path)


if __name__ == "__main__":
    unittest.main()
