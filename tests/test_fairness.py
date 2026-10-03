#!/usr/bin/env python3
"""Frequent-name contrast: who is compared, and how the gap is adjusted."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import common as base
import run_fairness as fair


class GroupSelectionTest(unittest.TestCase):
    def test_rare_controls_are_left_out(self):
        rows = fair.select_rows(base.load_rows(0))
        groups = {row["group"] for row in rows}
        self.assertEqual(groups, {"language", "high_resource"})
        self.assertEqual(sum(row["group"] == "language" for row in rows), 596)
        self.assertEqual(sum(row["group"] == "high_resource" for row in rows), 299)
        self.assertTrue(all(row["set"] != "control_rare" for row in rows))
        self.assertTrue(
            all(
                row["set"] == "control_common"
                for row in rows
                if row["group"] == "high_resource"
            )
        )

    def test_three_pairs_are_fixed_before_looking_at_scores(self):
        self.assertEqual(len(fair.PAIRS), 3)
        joined = " ".join(left + right for left, right in fair.PAIRS)
        self.assertIn("hired", joined)
        self.assertIn("exam", joined)
        self.assertIn("trusted", joined)

    def test_prefix_ends_in_a_space_and_covers_the_name(self):
        prefix, start, end = fair.prefix_and_span("Amina")
        self.assertTrue(prefix.endswith(" "))
        self.assertEqual(prefix[start:end], "Amina")
        self.assertNotIn("Amina.", prefix)


class GapTest(unittest.TestCase):
    def test_difference_is_language_minus_frequent(self):
        values = np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
        language = np.array([True, True, True, True, False, False, False, False])
        char_len = np.array([3, 3, 3, 3, 3, 3, 3, 3])
        report = fair.gap_report(values, language, char_len)
        self.assertEqual(report["difference"], 1.0)
        self.assertEqual(report["language_mean"], 1.0)
        self.assertEqual(report["high_resource_mean"], 0.0)
        low, high = report["bootstrap_ci95"]
        self.assertLessEqual(low, report["difference"])
        self.assertGreaterEqual(high, report["difference"])
        self.assertTrue(report["ci_excludes_zero"])
        self.assertEqual(report["n_lengths_matched"], 1)
        self.assertEqual(report["length_matched_difference"], 1.0)

    def test_short_bins_are_not_length_matched(self):
        values = np.ones(8)
        language = np.array([True, True, True, True, False, False, False, False])
        char_len = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        report = fair.gap_report(values, language, char_len)
        self.assertIsNone(report["length_matched_difference"])
        self.assertEqual(report["n_lengths_matched"], 0)

    def test_length_adjustment_keeps_a_constant_group_shift(self):
        language = np.array([True] * 8 + [False] * 8)
        char_len = np.tile(np.array([2.0, 4.0, 6.0, 8.0, 3.0, 5.0, 7.0, 9.0]), 2)
        values = 5.0 * char_len
        values = values.copy()
        values[language] -= 1.5
        report = fair.adjusted_gap(values, char_len, language, char_len)
        self.assertAlmostEqual(report["difference"], -1.5, places=6)


if __name__ == "__main__":
    unittest.main()
