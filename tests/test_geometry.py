#!/usr/bin/env python3
"""English offset: on the language plane, and off it."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_english_subspace as english


def _split(labels):
    train = np.zeros(len(labels), dtype=bool)
    for label in set(labels.tolist()):
        index = np.flatnonzero(labels == label)
        train[index[: len(index) // 2]] = True
    return train, ~train


def _table(english_vector, n_per=6, hidden=8):
    rows = []
    labels = []
    for language in range(6):
        for _ in range(n_per):
            vector = np.zeros(hidden)
            vector[language] = 1.0
            rows.append(vector)
            labels.append(language)
    for _ in range(n_per):
        rows.append(np.asarray(english_vector, dtype=np.float64))
        labels.append(-1)
    labels = np.asarray(labels)
    train, test = _split(labels)
    languages = [f"l{index}" for index in range(6)]
    return np.stack(rows), labels, train, test, languages


class CentroidBasisTest(unittest.TestCase):
    def test_six_means_span_at_most_five_directions(self):
        means = np.random.default_rng(0).normal(size=(6, 32))
        basis, dimension = english.centroid_basis(means)
        self.assertLessEqual(dimension, 5)
        self.assertEqual(basis.shape, (32, dimension))
        self.assertTrue(np.allclose(basis.T @ basis, np.eye(dimension), atol=1e-8))

    def test_direction_inside_and_outside_the_basis(self):
        basis = np.eye(4, 2)
        self.assertAlmostEqual(english.direction_in_subspace(basis[:, 0], basis), 1.0)
        outside = np.array([0.0, 0.0, 1.0, 0.0])
        self.assertAlmostEqual(english.direction_in_subspace(outside, basis), 0.0)


class OffsetTest(unittest.TestCase):
    def test_controls_on_the_language_mean_have_no_offset(self):
        english_vector = np.zeros(8)
        english_vector[:6] = 1.0 / 6.0
        features, labels, train, test, languages = _table(english_vector)
        report = english.geometry(
            features, labels, train, test, languages, binary_probe=False
        )
        self.assertAlmostEqual(
            report["english_orthogonal_over_language_gap"], 0.0, places=8
        )
        self.assertAlmostEqual(
            report["english_test_orthogonal_over_language_test"], 0.0, places=8
        )

    def test_controls_on_a_new_axis_sit_off_the_plane(self):
        english_vector = np.zeros(8)
        english_vector[7] = 10.0
        features, labels, train, test, languages = _table(english_vector)
        report = english.geometry(
            features, labels, train, test, languages, binary_probe=False
        )
        self.assertGreater(report["english_orthogonal_over_language_gap"], 1.0)
        self.assertGreater(report["english_test_orthogonal_over_language_test"], 1.0)
        self.assertEqual(report["dimension"], 5)

    def test_six_way_probe_separates_the_plane(self):
        english_vector = np.zeros(8)
        english_vector[:6] = 1.0 / 6.0
        features, labels, train, test, languages = _table(english_vector)
        groups = np.asarray(
            ["english" if label < 0 else languages[label] for label in labels]
        )
        report = english.analyze_site(features, labels, groups, train, test, languages)
        assigned = report["english_assigned_by_six_way_probe"]
        self.assertEqual(set(assigned), set(languages))
        self.assertEqual(sum(assigned.values()), int((test & (labels < 0)).sum()))
        self.assertGreater(report["seven_way_accuracy"], 0.9)
        self.assertIn("english", report["seven_way_recall"])


if __name__ == "__main__":
    unittest.main()
