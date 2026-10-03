#!/usr/bin/env python3
"""Where the English control names sit relative to the six-language subspace.

The subspace is the span of the six training language centroids. English is
the pooled Latin-script controls, frequent and uncommon together. Labels use
-1 for those controls. A small orthogonal distance means the control centroid
lies in the same subspace as the six languages. The ratio reported in the
paper divides that distance by the average gap between language centroids, so
1 means "as far off the plane as the languages are from each other."

The 2D scatter is the in-plane view. The separation the paper reports is the
component that scatter leaves out.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_language_mech as mech
import run_paper_strength as paper
import common as base

OUT = base.RESULTS / "english_subspace"


def centroid_basis(means):
    """Orthonormal basis of the centered class means.

    Six points in a high-dimensional residual span at most five dimensions
    after centering. Singular values below 1e-4 of the largest are dropped.
    """
    centered = means - means.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0:
        basis = np.zeros((means.shape[1], 1), dtype=np.float64)
        return basis, 1
    keep = max(1, int(np.sum(singular > 1e-4 * singular[0])))
    basis = vt[:keep].T
    orthogonal, _ = np.linalg.qr(basis)
    return orthogonal[:, :keep], keep


def unit(vector):
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


def direction_in_subspace(direction, basis):
    """Fraction of a unit vector that lies inside `basis`. Zero means a new axis."""
    vector = unit(direction)
    return float(np.linalg.norm(basis.T @ vector))


def fit_predict(train_x, train_y, test_x):
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_x)
    clf = LogisticRegression(max_iter=500, tol=1e-3, random_state=base.SEED)
    clf.fit(train_s, train_y)
    pred = clf.predict(scaler.transform(test_x))
    proba = clf.predict_proba(scaler.transform(test_x))
    order = np.argsort(clf.classes_)
    proba = proba[:, order]
    return pred, proba, clf.coef_ / scaler.scale_


def language_frame(features, labels, train, languages):
    lang_train = train & (labels >= 0)
    train_rows = []
    for index in range(len(languages)):
        chosen = lang_train & (labels == index)
        train_rows.append(features[chosen].mean(axis=0))
    train_means = np.stack(train_rows)
    basis, dimension = centroid_basis(train_means)
    origin = train_means.mean(axis=0)
    return train_means, basis, dimension, origin


def geometry(features, labels, train, test, languages, binary_probe=True):
    """How far English sits from the span of the six training language centroids.

    `english_orthogonal_over_language_gap` is the paper's offset. The in-plane
    ratio is the part a 2D plot can show. The cloud ratio compares mean
    orthogonal distance of English test points with language test points.
    """
    lang_test = test & (labels >= 0)
    eng_train = train & (labels < 0)
    eng_test = test & (labels < 0)
    train_means, basis, dimension, origin = language_frame(
        features, labels, train, languages
    )

    def ortho_norm(rows):
        centered = rows - origin
        inplane = (centered @ basis) @ basis.T
        return np.linalg.norm(centered - inplane, axis=1)

    gaps = []
    centered_means = train_means - origin
    for i in range(len(languages)):
        for j in range(i + 1, len(languages)):
            gaps.append(np.linalg.norm(centered_means[i] - centered_means[j]))
    gap = float(np.mean(gaps))
    english_offset = features[eng_train].mean(axis=0) - origin
    inplane = (english_offset @ basis) @ basis.T
    english_ortho = float(np.linalg.norm(english_offset - inplane))
    english_inplane = float(np.linalg.norm(inplane))
    nearest = int(np.argmin(np.linalg.norm(centered_means - inplane, axis=1)))
    language_point = (
        float(ortho_norm(features[lang_test]).mean()) if np.any(lang_test) else None
    )
    english_point = (
        float(ortho_norm(features[eng_test]).mean()) if np.any(eng_test) else None
    )

    report = {
        "dimension": dimension,
        "language_centroid_gap": gap,
        "english_orthogonal_distance": english_ortho,
        "english_inplane_distance": english_inplane,
        "english_orthogonal_over_language_gap": english_ortho / gap if gap else None,
        "english_inplane_over_language_gap": english_inplane / gap if gap else None,
        "language_test_orthogonal_distance": language_point,
        "english_test_orthogonal_distance": english_point,
        "english_test_orthogonal_over_language_test": (
            english_point / language_point if language_point else None
        ),
        "nearest_language_to_english_centroid": languages[nearest],
    }
    if binary_probe:
        binary = (labels >= 0).astype(np.int32)
        _, _, binary_raw = fit_predict(features[train], binary[train], features[test])
        report["binary_direction_in_language_subspace"] = direction_in_subspace(
            binary_raw.reshape(-1), basis
        )
    return report


def analyze_site(features, labels, groups, train, test, languages):
    lang_train = train & (labels >= 0)
    lang_test = test & (labels >= 0)
    eng_test = test & (labels < 0)
    train_means, basis, _, origin = language_frame(features, labels, train, languages)
    report = geometry(features, labels, train, test, languages)

    six_pred, six_proba, _ = fit_predict(
        features[lang_train], labels[lang_train], features[lang_test | eng_test]
    )
    scored = lang_test | eng_test
    test_groups = groups[scored]
    english_mask = test_groups == "english"
    language_mask = ~english_mask
    assigned = six_pred[english_mask]
    assignment = {
        languages[index]: int(np.sum(assigned == index))
        for index in range(len(languages))
    }
    english_confidence = (
        float(six_proba[english_mask].max(axis=1).mean())
        if np.any(english_mask)
        else None
    )
    language_confidence = (
        float(six_proba[language_mask].max(axis=1).mean())
        if np.any(language_mask)
        else None
    )

    seven = labels.copy()
    seven[labels < 0] = len(languages)
    seven_pred, _, seven_raw = fit_predict(
        features[train], seven[train], features[test]
    )
    seven_labels = [name for name in languages] + ["english"]
    seven_recall = {}
    truth = seven[test]
    for index, name in enumerate(seven_labels):
        mask = truth == index
        if np.any(mask):
            seven_recall[name] = float(np.mean(seven_pred[mask] == index))
    _, seven_singular, seven_vt = np.linalg.svd(seven_raw, full_matrices=False)
    seven_keep = max(1, int(np.sum(seven_singular > 1e-4 * seven_singular[0])))
    seven_basis = seven_vt[:seven_keep].T
    seven_q, _ = np.linalg.qr(seven_basis)
    seven_basis = seven_q[:, :seven_keep]
    overlap = seven_basis.T @ basis
    captured = float(np.sum(overlap**2) / seven_basis.shape[1])

    # The scatter uses the first two centroid directions. Off-plane distance
    # is invisible in these coordinates.
    plane = basis[:, : min(2, basis.shape[1])]
    plotted = test
    coords = (features[plotted] - origin) @ plane
    report.update(
        {
            "seven_way_accuracy": float(np.mean(seven_pred == truth)),
            "seven_way_recall": seven_recall,
            "seven_way_dimension": int(seven_basis.shape[1]),
            "seven_way_energy_inside_language_subspace": captured,
            "english_assigned_by_six_way_probe": assignment,
            "english_mean_confidence": english_confidence,
            "language_mean_confidence": language_confidence,
            "projection": {
                "x": coords[:, 0].astype(float).tolist(),
                "y": (
                    coords[:, 1].astype(float).tolist()
                    if coords.shape[1] > 1
                    else [0.0] * len(coords)
                ),
                "group": groups[plotted].tolist(),
            },
        }
    )
    return report


def run_model(model_name, rows, device, batch_size, token) -> dict:
    print(f"\n=== {model_name} ===", flush=True)
    languages = sorted(
        {row["language_code"] for row in rows if row["include_in_language_probe"] == 1}
    )
    language_id = {code: index for index, code in enumerate(languages)}
    labels = np.asarray(
        [
            (
                language_id[row["language_code"]]
                if row["include_in_language_probe"] == 1
                else -1
            )
            for row in rows
        ]
    )
    groups = np.asarray(
        [
            row["language_code"] if row["include_in_language_probe"] == 1 else "english"
            for row in rows
        ]
    )
    # Drop the four double-listed names that are in neither probe.
    keep = np.asarray(
        [
            row["include_in_language_probe"] == 1 or row["is_east_african"] == 0
            for row in rows
        ]
    )
    rows = [row for row, flag in zip(rows, keep) if flag]
    labels = labels[keep]
    groups = groups[keep]
    train = np.asarray([row["split"] == "train" for row in rows])
    test = ~train

    model, tokenizer = paper.load_model(model_name, token, device, "float32")
    print(" collecting activations", flush=True)
    clean = mech.collect_clean(model, tokenizer, rows, device, batch_size)
    result = {
        "model": model_name,
        "languages": languages,
        "n_english_train": int(((labels < 0) & train).sum()),
        "n_english_test": int(((labels < 0) & test).sum()),
        "n_language_test": int(((labels >= 0) & test).sum()),
    }
    for site in ("last", "mean"):
        bank = clean[site]
        best_layer = 0
        best_score = -1.0
        lang_train = train & (labels >= 0)
        by_layer = []
        for index, features in enumerate(bank):
            # Choose the layer on training accuracy. Test accuracy is stored
            # but does not pick the layer, so the reported offset is not tuned
            # on the names it is scored on.
            pred, _, _ = fit_predict(
                features[lang_train], labels[lang_train], features[lang_train]
            )
            score = float(np.mean(pred == labels[lang_train]))
            if score > best_score:
                best_layer, best_score = index, score
            layer_geometry = geometry(features, labels, train, test, languages)
            layer_geometry["layer"] = index
            layer_geometry["train_accuracy"] = score
            by_layer.append(layer_geometry)
        print(f"  {site} layer {best_layer}", flush=True)
        report = analyze_site(bank[best_layer], labels, groups, train, test, languages)
        report["layer"] = best_layer
        report["train_accuracy"] = best_score
        report["by_layer"] = by_layer
        result[site] = report
        ratio = report["english_orthogonal_over_language_gap"]
        inside = report["binary_direction_in_language_subspace"]
        cloud = report["english_test_orthogonal_over_language_test"]
        print(
            f"    english offset / language gap {ratio:.3f}, "
            f"english cloud / language cloud {cloud:.3f}, "
            f"binary direction inside subspace {inside:.3f}",
            flush=True,
        )

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--models",
        nargs="*",
        default=[
            "distilgpt2",
            "HuggingFaceTB/SmolLM2-360M",
            "Qwen/Qwen3-0.6B",
            "Qwen/Qwen2.5-0.5B",
        ],
    )
    args = parser.parse_args()
    token = base.load_project_env()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load_rows(0)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device {device} rows {len(rows)}", flush=True)
    for model_name in args.models:
        path = OUT / f"{base.slug(model_name)}.json"
        try:
            summary = run_model(model_name, rows, device, args.batch_size, token)
        except Exception as exc:
            message = base.redact(str(exc))
            print(f"FAILED {model_name}: {message}", flush=True)
            summary = {"model": model_name, "error": message}
        path.write_text(json.dumps(summary, indent=2, default=mech.json_ready) + "\n")
        print(f" wrote {path}", flush=True)


if __name__ == "__main__":
    main()
