#!/usr/bin/env python3
"""Offset bootstrap, the spelling projection, and the second sentence."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common as base
import run_publish as publish


class IntervalTest(unittest.TestCase):
    def test_recentered_interval_covers_the_point_the_raw_one_misses(self):
        draws = np.zeros(80)
        report = publish.ci(draws, point=4.0)
        self.assertEqual(report["percentile_ci95"], [0.0, 0.0])
        low, high = report["ci95"]
        self.assertLessEqual(low, 4.0)
        self.assertGreaterEqual(high, 4.0)
        self.assertTrue(report["ci_above_one"])
        self.assertFalse(report["ci_below_one"])


class ProjectionTest(unittest.TestCase):
    def test_a_point_inside_the_language_span_has_ratio_zero(self):
        means = np.eye(6, 8)
        english_mean = means.mean(axis=0)
        ratio, basis, _origin = publish.ratio_from_means(means, english_mean)
        self.assertAlmostEqual(ratio, 0.0, places=8)
        self.assertLessEqual(basis.shape[1], 5)

    def test_a_new_axis_is_outside_the_span(self):
        means = np.eye(6, 8)
        english_mean = np.zeros(8)
        english_mean[7] = 10.0
        ratio, _basis, _origin = publish.ratio_from_means(means, english_mean)
        self.assertGreater(ratio, 1.0)

    def test_ridge_removes_a_spelling_direction(self):
        rng = np.random.default_rng(0)
        gram = rng.normal(size=(30, 4))
        weights = rng.normal(size=(4, 3))
        features = gram @ weights
        train = np.zeros(30, dtype=bool)
        train[:20] = True
        residual = publish.project_out(features, gram, train)
        self.assertLess(np.linalg.norm(residual), 0.25 * np.linalg.norm(features))

    def test_pairwise_language_distances_match_themselves(self):
        means = np.random.default_rng(0).normal(size=(6, 8))
        distances, pairs = publish.pairwise(means)
        self.assertEqual(len(pairs), 15)
        report = publish.correlation(distances, distances)
        self.assertAlmostEqual(report["spearman"], 1.0)
        self.assertAlmostEqual(report["pearson"], 1.0)


class SentenceAndTableTest(unittest.TestCase):
    def test_second_sentence_moves_the_name_span(self):
        rows = [{"name": "Amy", "language_code": "control_common", "split": "test"}]
        retargeted = publish.retarget(rows, publish.ALT_TEMPLATE)[0]
        self.assertEqual(retargeted["prompt"], "The person Amy is called")
        self.assertEqual(
            retargeted["prompt"][
                retargeted["name_char_start"] : retargeted["name_char_end"]
            ],
            "Amy",
        )

    def test_shuffle_keeps_the_letters_and_the_label(self):
        rows = [
            {
                "name": "Mugisha",
                "language_code": "kinyarwanda",
                "include_in_language_probe": 1,
                "split": "train",
            }
        ]
        shuffled = publish.shuffled_rows(rows, seed=base.SEED)[0]
        self.assertEqual(sorted(shuffled["name"]), sorted("Mugisha"))
        self.assertEqual(shuffled["language_code"], "kinyarwanda")
        self.assertIn(shuffled["name"], shuffled["prompt"])

    def test_prepared_table_drops_the_double_listed_names(self):
        rows, languages, labels, train, names = publish.prepared(base.load_rows(0))
        self.assertEqual(len(languages), 6)
        self.assertEqual(int((labels >= 0).sum()), 596)
        self.assertEqual(int((labels < 0).sum()), 600)
        self.assertEqual(len(rows), len(names))
        self.assertEqual(int(train.sum()) + int((~train).sum()), len(rows))
        self.assertTrue(
            all(
                row["include_in_language_probe"] == 1 or row["is_east_african"] == 0
                for row in rows
            )
        )
        self.assertNotIn(
            "Mugisha",
            [row["name"] for row in rows if row["include_in_language_probe"] == 0],
        )


if __name__ == "__main__":
    unittest.main()
