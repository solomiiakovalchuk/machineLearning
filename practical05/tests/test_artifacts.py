"""Цілісність артефактів після реального запуску run_benchmark.py + select_model.py."""
import json
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.candidates import CANDIDATES, load_data  # noqa: E402

RESULTS = ROOT / "results"
SCENARIOS = ("realtime_workstation", "batch_second_reading")


def _need(path: Path) -> Path:
    if not path.exists():
        raise unittest.SkipTest(
            f"немає {path.relative_to(ROOT)}: спочатку запустіть run_benchmark.py і select_model.py")
    return path


class BenchmarkResultsTests(unittest.TestCase):
    def test_six_unique_rows_without_nan(self):
        df = pd.read_csv(_need(RESULTS / "benchmark_results.csv"))
        self.assertEqual(len(df), 6)
        self.assertEqual(df["model"].nunique(), 6)
        self.assertEqual(set(df["model"]), set(CANDIDATES))
        self.assertFalse(df.isna().any().any(), "у benchmark_results.csv є NaN")
        numeric = df.drop(columns="model")
        self.assertTrue(np.isfinite(numeric.to_numpy(dtype=float)).all())

    def test_saved_models_exist_for_all_candidates(self):
        for name in CANDIDATES:
            self.assertTrue(_need(RESULTS / "models" / f"{name}.joblib").stat().st_size > 0)


class SelectionTests(unittest.TestCase):
    def test_primary_equals_first_ranking_row(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                sel = json.loads(_need(RESULTS / f"selection_{scenario}.json").read_text(encoding="utf-8"))
                ranking = pd.read_csv(_need(RESULTS / f"ranking_{scenario}.csv"))
                self.assertEqual(sel["scenario"], scenario)
                self.assertEqual(sel["primary"]["model"], ranking.loc[0, "model"])
                self.assertAlmostEqual(sel["primary"]["score"], ranking.loc[0, "score"], places=9)
                if sel["backup"] is not None:
                    self.assertEqual(sel["backup"]["model"], ranking.loc[1, "model"])

    def test_ranking_and_rejected_partition_all_models(self):
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                ranking = pd.read_csv(_need(RESULTS / f"ranking_{scenario}.csv"))
                rejected = pd.read_csv(_need(RESULTS / f"rejected_{scenario}.csv"))
                self.assertEqual(set(ranking["model"]) | set(rejected["model"]), set(CANDIDATES))
                self.assertTrue(set(ranking["model"]).isdisjoint(rejected["model"]))

    def test_selected_pipeline_loads_and_predicts_on_raw_rows(self):
        raw5 = load_data().X_train.iloc[:5]                      # сирі (необроблені) рядки
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                sel = json.loads(_need(RESULTS / f"selection_{scenario}.json").read_text(encoding="utf-8"))
                selected = joblib.load(_need(RESULTS / f"selected_{scenario}.joblib"))
                stored = joblib.load(_need(RESULTS / "models" / f"{sel['primary']['model']}.joblib"))
                proba = selected.predict_proba(raw5)
                self.assertEqual(proba.shape, (5, 2))
                self.assertTrue(((proba >= 0.0) & (proba <= 1.0)).all())
                np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)
                np.testing.assert_allclose(proba, stored.predict_proba(raw5), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
