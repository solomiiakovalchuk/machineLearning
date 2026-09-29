"""Модульні тести політики. Запуск (з каталогу policy):

    python -m unittest -v test_decision_policy.py
"""
import json
import unittest
from pathlib import Path

from decision_policy import (
    ACTION_BIOPSY,
    ACTION_REVIEW,
    ACTION_STANDARD,
    DEFAULT_CONFIG_PATH,
    DecisionPolicy,
    capacity_k,
    select_top_k,
)

LOW, HIGH = 0.30, 0.70   # тестові пороги (не пов'язані з вибраною політикою)


class TestZones(unittest.TestCase):
    def setUp(self):
        self.policy = DecisionPolicy(threshold_low=LOW, threshold_high=HIGH)

    def test_1_below_low(self):
        self.assertEqual(self.policy.decide(0.10), ACTION_STANDARD)

    def test_2_equal_low_goes_to_review(self):
        self.assertEqual(self.policy.decide(LOW), ACTION_REVIEW)

    def test_3_between_thresholds(self):
        self.assertEqual(self.policy.decide(0.50), ACTION_REVIEW)

    def test_4_equal_high_goes_to_biopsy(self):
        self.assertEqual(self.policy.decide(HIGH), ACTION_BIOPSY)

    def test_5_above_high(self):
        self.assertEqual(self.policy.decide(0.95), ACTION_BIOPSY)

    def test_just_below_thresholds(self):
        self.assertEqual(self.policy.decide(0.2999999), ACTION_STANDARD)
        self.assertEqual(self.policy.decide(0.6999999), ACTION_REVIEW)

    def test_range_edges_are_valid(self):
        self.assertEqual(self.policy.decide(0.0), ACTION_STANDARD)
        self.assertEqual(self.policy.decide(1.0), ACTION_BIOPSY)

    def test_6_score_out_of_range(self):
        for bad in (-0.01, 1.01, float("nan"), float("inf")):
            with self.subTest(score=bad):
                with self.assertRaises(ValueError):
                    self.policy.decide(bad)

    def test_6_score_wrong_type(self):
        for bad in ("0.5", None, True):
            with self.subTest(score=bad):
                with self.assertRaises(TypeError):
                    self.policy.decide(bad)

    def test_7_low_greater_than_high_rejected(self):
        with self.assertRaises(ValueError):
            DecisionPolicy(threshold_low=0.8, threshold_high=0.2)

    def test_7_thresholds_out_of_range_rejected(self):
        with self.assertRaises(ValueError):
            DecisionPolicy(threshold_low=-0.1, threshold_high=0.5)
        with self.assertRaises(ValueError):
            DecisionPolicy(threshold_low=0.1, threshold_high=1.5)

    def test_single_threshold_has_empty_review_zone(self):
        single = DecisionPolicy(threshold_low=0.4, threshold_high=0.4)
        self.assertEqual(single.decide(0.3999), ACTION_STANDARD)
        self.assertEqual(single.decide(0.4), ACTION_BIOPSY)
        self.assertNotIn(ACTION_REVIEW, single.decide_batch([i / 100 for i in range(101)]))


class TestBatch(unittest.TestCase):
    def setUp(self):
        self.policy = DecisionPolicy(threshold_low=LOW, threshold_high=HIGH)

    def test_8_batch_preserves_order(self):
        scores = [0.95, 0.10, 0.50, 0.30, 0.70, 0.0, 1.0]
        expected = [ACTION_BIOPSY, ACTION_STANDARD, ACTION_REVIEW, ACTION_REVIEW,
                    ACTION_BIOPSY, ACTION_STANDARD, ACTION_BIOPSY]
        self.assertEqual(self.policy.decide_batch(scores), expected)
        # результат партії = поелементні рішення в тому ж порядку
        self.assertEqual(self.policy.decide_batch(scores), [self.policy.decide(s) for s in scores])

    def test_8_empty_batch(self):
        self.assertEqual(self.policy.decide_batch([]), [])

    def test_batch_rejects_bad_element(self):
        with self.assertRaises(ValueError):
            self.policy.decide_batch([0.1, 1.5, 0.2])


class TestTopK(unittest.TestCase):
    def test_capacity_k(self):
        self.assertEqual(capacity_k(455, 0.40), 182)
        self.assertEqual(capacity_k(114, 0.40), 45)
        self.assertEqual(capacity_k(5, 0.40), 2)
        self.assertEqual(capacity_k(0, 0.40), 0)
        with self.assertRaises(ValueError):
            capacity_k(10, 1.5)

    def test_empty_batch(self):
        self.assertEqual(select_top_k([], 3), [])

    def test_batch_smaller_than_k(self):
        self.assertEqual(select_top_k([0.2, 0.9], 5), [1, 0])

    def test_k_zero_and_negative(self):
        self.assertEqual(select_top_k([0.2, 0.9], 0), [])
        with self.assertRaises(ValueError):
            select_top_k([0.2, 0.9], -1)

    def test_selects_highest_scores(self):
        self.assertEqual(select_top_k([0.1, 0.8, 0.3, 0.9, 0.5], 2), [3, 1])

    def test_equal_scores_tie_by_object_order(self):
        # усі бали рівні -> беруться перші k за порядком надходження
        self.assertEqual(select_top_k([0.5] * 6, 3), [0, 1, 2])
        # ties на межі квоти: бал 0.7 мають індекси 1, 3, 4 -> при k=2 беруться 2 і 1
        self.assertEqual(select_top_k([0.2, 0.7, 0.9, 0.7, 0.7], 2), [2, 1])
        self.assertEqual(select_top_k([0.2, 0.7, 0.7, 0.7, 0.1], 2), [1, 2])

    def test_reproducible_tie_order(self):
        scores = [0.4, 0.6, 0.6, 0.6, 0.4, 0.6]
        first = select_top_k(scores, 3)
        for _ in range(5):
            self.assertEqual(select_top_k(scores, 3), first)
        self.assertEqual(first, [1, 2, 3])


class TestQuota(unittest.TestCase):
    def setUp(self):
        self.policy = DecisionPolicy(threshold_low=0.5, threshold_high=0.5)

    def test_no_overflow_keeps_decisions(self):
        scores = [0.9, 0.1, 0.2, 0.8, 0.1]       # 2 біопсії, k = floor(0.4*5) = 2
        self.assertEqual(self.policy.decide_batch_with_quota(scores, 0.4),
                         self.policy.decide_batch(scores))

    def test_overflow_demotes_lowest_scores_with_stable_ties(self):
        scores = [0.6, 0.9, 0.6, 0.6, 0.1]       # 4 біопсії, k = 2: лишаються 0.9 і перше 0.6
        actions = self.policy.decide_batch_with_quota(scores, 0.4)
        self.assertEqual(actions, [ACTION_BIOPSY, ACTION_BIOPSY, ACTION_REVIEW,
                                   ACTION_REVIEW, ACTION_STANDARD])

    def test_empty_batch_with_quota(self):
        self.assertEqual(self.policy.decide_batch_with_quota([], 0.4), [])


class TestConfig(unittest.TestCase):
    @unittest.skipUnless(DEFAULT_CONFIG_PATH.exists(), "policy_config.json ще не створено")
    def test_policy_from_config_matches_json(self):
        cfg = json.loads(Path(DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))
        for name, params in cfg["policy_variants"].items():
            policy = DecisionPolicy.from_config(DEFAULT_CONFIG_PATH, variant=name)
            self.assertEqual(policy.threshold_low, params["threshold_low"])
            self.assertEqual(policy.threshold_high, params["threshold_high"])
            # правило score >= threshold_high -> дія високої зони
            self.assertEqual(policy.decide(params["threshold_high"]), cfg["actions"]["high"])


if __name__ == "__main__":
    unittest.main()
