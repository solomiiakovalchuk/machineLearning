"""Модульні тести model_selector на малих ручних таблицях (без sklearn і без benchmark)."""
import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model_selector import (NoEligibleModelError, check_constraints,  # noqa: E402
                                normalize_value, rank_models, reweight,
                                save_selection, sensitivity_analysis,
                                validate_config)


def make_config():
    return {
        "scenario": "toy",
        "version": "0.1",
        "description": "тестовий сценарій",
        "constraints": {
            "recall": {"min": 0.9},
            "latency": {"max": 10.0},
        },
        "criteria": {
            "f1": {"direction": "maximize", "weight": 0.6, "worst": 0.5, "best": 1.0},
            "latency": {"direction": "minimize", "weight": 0.4, "worst": 10.0, "best": 0.0},
        },
        "unknown_top_level_key": "ігнорується",
    }


def make_results():
    return pd.DataFrame({
        "model": ["a", "b", "c"],
        "f1": [0.9, 0.8, 0.7],
        "recall": [0.95, 0.92, 0.97],
        "latency": [2.0, 1.0, 4.0],
    })


class NormalizeTests(unittest.TestCase):
    def test_maximize_normalization(self):
        spec = {"direction": "maximize", "worst": 0.5, "best": 1.0}
        self.assertAlmostEqual(normalize_value(0.5, spec), 0.0)
        self.assertAlmostEqual(normalize_value(0.75, spec), 0.5)
        self.assertAlmostEqual(normalize_value(1.0, spec), 1.0)

    def test_minimize_normalization(self):
        spec = {"direction": "minimize", "worst": 10.0, "best": 0.0}
        self.assertAlmostEqual(normalize_value(10.0, spec), 0.0)
        self.assertAlmostEqual(normalize_value(2.5, spec), 0.75)
        self.assertAlmostEqual(normalize_value(0.0, spec), 1.0)

    def test_clipping_outside_anchors(self):
        mx = {"direction": "maximize", "worst": 0.5, "best": 1.0}
        mn = {"direction": "minimize", "worst": 10.0, "best": 1.0}
        self.assertEqual(normalize_value(0.2, mx), 0.0)
        self.assertEqual(normalize_value(1.3, mx), 1.0)
        self.assertEqual(normalize_value(50.0, mn), 0.0)       # гірше за worst
        self.assertEqual(normalize_value(0.1, mn), 1.0)        # краще за best

    def test_unknown_direction(self):
        with self.assertRaisesRegex(ValueError, "direction"):
            normalize_value(1.0, {"direction": "upward", "worst": 0.0, "best": 1.0})
        cfg = make_config()
        cfg["criteria"]["f1"]["direction"] = "upward"
        with self.assertRaisesRegex(ValueError, "direction"):
            validate_config(cfg)

    def test_anchors_contradict_direction(self):
        with self.assertRaisesRegex(ValueError, "best > worst"):
            normalize_value(0.7, {"direction": "maximize", "worst": 1.0, "best": 0.5})
        with self.assertRaisesRegex(ValueError, "best < worst"):
            normalize_value(0.7, {"direction": "minimize", "worst": 0.5, "best": 1.0})
        cfg = make_config()
        cfg["criteria"]["f1"].update(worst=1.0, best=0.5)
        with self.assertRaisesRegex(ValueError, "best > worst"):
            validate_config(cfg)


class ConstraintTests(unittest.TestCase):
    def test_bounds_are_inclusive(self):
        ok, why = check_constraints({"recall": 0.9, "latency": 10.0}, make_config()["constraints"])
        self.assertTrue(ok)
        self.assertEqual(why, [])

    def test_all_violations_are_listed(self):
        ok, why = check_constraints({"recall": 0.85, "latency": 12.5}, make_config()["constraints"])
        self.assertFalse(ok)
        self.assertEqual(len(why), 2)
        self.assertIn("recall=0.85 < min 0.9", why[0])
        self.assertIn("latency=12.5 > max 10.0", why[1])

    def test_ineligible_model_removed_with_all_reasons(self):
        res = make_results()
        res.loc[res["model"] == "c", ["recall", "latency"]] = [0.5, 30.0]
        ranking, rejected = rank_models(res, make_config())
        self.assertNotIn("c", ranking["model"].tolist())
        row = rejected.set_index("model").loc["c"]
        self.assertEqual(row["n_violations"], 2)
        self.assertIn("recall=", row["violations"])
        self.assertIn("latency=", row["violations"])


class ValidationTests(unittest.TestCase):
    def test_wrong_weight_sum(self):
        cfg = make_config()
        cfg["criteria"]["f1"]["weight"] = 0.7          # сума 1.1
        with self.assertRaisesRegex(ValueError, "сума ваг"):
            validate_config(cfg)
        with self.assertRaisesRegex(ValueError, "сума ваг"):
            rank_models(make_results(), cfg)

    def test_missing_metric_column(self):
        res = make_results().drop(columns="latency")
        with self.assertRaisesRegex(ValueError, "latency"):
            rank_models(res, make_config())

    def test_nan_metric(self):
        res = make_results()
        res.loc[1, "f1"] = np.nan
        with self.assertRaisesRegex(ValueError, "f1.*'b'|'b'.*f1"):
            rank_models(res, make_config())

    def test_non_numeric_metric(self):
        res = make_results().astype({"recall": object})
        res.loc[0, "recall"] = "high"
        with self.assertRaisesRegex(ValueError, "не числова"):
            rank_models(res, make_config())

    def test_duplicate_model_names(self):
        res = pd.concat([make_results(), make_results().iloc[[0]]], ignore_index=True)
        with self.assertRaisesRegex(ValueError, "дублікати"):
            rank_models(res, make_config())


class RankingTests(unittest.TestCase):
    def test_ranking_order_and_score(self):
        ranking, rejected = rank_models(make_results(), make_config())
        self.assertEqual(len(rejected), 0)
        # a: 0.6*0.8 + 0.4*0.8 = 0.8 ; b: 0.6*0.6 + 0.4*0.9 = 0.72 ; c: 0.6*0.4 + 0.4*0.6 = 0.48
        self.assertEqual(ranking["model"].tolist(), ["a", "b", "c"])
        self.assertAlmostEqual(ranking.loc[0, "score"], 0.8)
        self.assertEqual(ranking["rank"].tolist(), [1, 2, 3])

    def test_no_eligible_models(self):
        res = make_results()
        res["recall"] = 0.1
        with self.assertRaises(NoEligibleModelError) as ctx:
            rank_models(res, make_config())
        self.assertIsInstance(ctx.exception, ValueError)
        self.assertEqual(set(ctx.exception.violations), {"a", "b", "c"})
        self.assertIn("recall", str(ctx.exception))

    def test_identical_scores_tie_break_by_name(self):
        res = pd.DataFrame({
            "model": ["zeta", "alpha", "mid"],
            "f1": [0.8, 0.8, 0.8],
            "recall": [0.95, 0.95, 0.95],
            "latency": [2.0, 2.0, 2.0],
        })
        for _ in range(3):                                   # порядок рядків не впливає
            res = res.sample(frac=1.0, random_state=_)
            ranking, _rej = rank_models(res, make_config())
            self.assertEqual(ranking["model"].tolist(), ["alpha", "mid", "zeta"])

    def test_reweight_keeps_sum_one(self):
        cfg = make_config()
        for w in (0.0, 0.2, 0.6, 1.0):
            new = reweight(cfg, "f1", w)
            total = sum(s["weight"] for s in new["criteria"].values())
            self.assertAlmostEqual(total, 1.0, places=12)
            self.assertAlmostEqual(new["criteria"]["f1"]["weight"], w)
            validate_config(new)
        self.assertEqual(cfg["criteria"]["f1"]["weight"], 0.6)   # оригінал не змінено

    def test_reweight_scales_others_proportionally(self):
        cfg = make_config()
        cfg["criteria"]["recall"] = {"direction": "maximize", "weight": 0.0,
                                     "worst": 0.9, "best": 1.0}
        cfg["criteria"]["f1"]["weight"], cfg["criteria"]["latency"]["weight"] = 0.5, 0.5
        new = reweight(cfg, "f1", 0.8)
        self.assertAlmostEqual(new["criteria"]["latency"]["weight"], 0.2)
        self.assertAlmostEqual(new["criteria"]["recall"]["weight"], 0.0)

    def test_sensitivity_analysis_changes_winner(self):
        # a сильніший за f1, b швидший: при вазі f1 = 0 перемагає b, при 1 - a
        sens = sensitivity_analysis(make_results(), make_config(), "f1", [0.0, 1.0])
        winners = sens[sens["winner"]].set_index("weight")["model"].to_dict()
        self.assertEqual(winners, {0.0: "b", 1.0: "a"})


class SelectionFileTests(unittest.TestCase):
    def test_json_consistent_with_first_ranking_row(self):
        res = make_results()
        res.loc[2, "recall"] = 0.1                           # c відхилено
        cfg = make_config()
        ranking, rejected = rank_models(res, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            sel = save_selection(ranking, rejected, cfg, tmp, models_dir=None)
            data = json.loads((Path(tmp) / "selection_toy.json").read_text(encoding="utf-8"))
            self.assertEqual(data, sel)
            self.assertEqual(data["primary"]["model"], ranking.loc[0, "model"])
            self.assertAlmostEqual(data["primary"]["score"], ranking.loc[0, "score"])
            self.assertEqual(data["backup"]["model"], ranking.loc[1, "model"])
            self.assertAlmostEqual(data["score_gap"], ranking.loc[0, "score"] - ranking.loc[1, "score"])
            self.assertEqual([r["model"] for r in data["rejected"]], ["c"])
            self.assertTrue((Path(tmp) / "MODEL_SELECTION_toy.md").exists())
            csv_rank = pd.read_csv(Path(tmp) / "ranking_toy.csv")
            self.assertEqual(csv_rank.loc[0, "model"], data["primary"]["model"])

    def test_single_eligible_model_has_no_backup(self):
        res = make_results()
        res.loc[1:, "recall"] = 0.1
        cfg = make_config()
        ranking, rejected = rank_models(res, cfg)
        with tempfile.TemporaryDirectory() as tmp:
            data = save_selection(ranking, rejected, copy.deepcopy(cfg), tmp)
        self.assertEqual(data["primary"]["model"], "a")
        self.assertIsNone(data["backup"])
        self.assertIsNone(data["score_gap"])


if __name__ == "__main__":
    unittest.main()
