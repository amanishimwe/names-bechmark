#!/usr/bin/env python3
"""Neuron-subspace audit of the East African names table.

Fits held-out linear probes on the residual stream and on MLP neurons at the
last piece of the name, then ablates those directions and compares the change
in place probability with a random subspace of the same size.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import re
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _preload_env() -> None:
    """Load .env before Hugging Face reads download timeouts at import."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "600")
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "60")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


_preload_env()

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = ROOT / "data" / "names.csv"
CONFIG = json.loads((ROOT / "data" / "config.json").read_text())
RESULTS = ROOT / "results"
SEED = int(CONFIG["seed"])
PLACES = list(CONFIG["place_vocabulary"])
# Two Qwen sizes from different generations, plus two other ungated families
# whose residual stream and MLP down-projection can be hooked directly.
# Cached models first, then the two Qwen checkpoints. Downloads use HF_TOKEN from .env.
MODELS = [
    "HuggingFaceTB/SmolLM2-360M",
    "distilgpt2",
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen2.5-0.5B",
]


TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]+")


def redact(text: str) -> str:
    return TOKEN_RE.sub("hf_[redacted]", text)


def load_project_env() -> str | None:
    """Load KEY=VALUE pairs from the project .env without printing secrets."""
    path = ROOT / ".env"
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        os.environ["HF_TOKEN"] = token
        os.environ["HUGGING_FACE_HUB_TOKEN"] = token
    return token or None


def load_rows(limit: int) -> list[dict]:
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["is_east_african"] = int(row["is_east_african"])
        row["include_in_language_probe"] = int(row["include_in_language_probe"])
        row["name_char_start"] = int(row["name_char_start"])
        row["name_char_end"] = int(row["name_char_end"])
        row["places"] = [p for p in row["place_labels"].split("|") if p]
    if limit <= 0:
        return rows
    rng = random.Random(SEED)
    kept = []
    buckets: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        buckets.setdefault((row["split"], row["set"]), []).append(row)
    for group in buckets.values():
        rng.shuffle(group)
        kept.extend(group[:limit])
    return kept


def name_positions(offset_rows, starts, ends) -> tuple[list[int], list[int]]:
    positions, counts = [], []
    for offsets, start, end in zip(offset_rows, starts, ends):
        hits = []
        for index, pair in enumerate(offsets):
            left, right = int(pair[0]), int(pair[1])
            if left == right:
                continue
            if right > start and left < end:
                hits.append(index)
        if not hits:
            raise RuntimeError(f"name span [{start}, {end}) matched no tokens")
        positions.append(hits[-1])
        counts.append(len(hits))
    return positions, counts


def as_offset_rows(mapping):
    if torch.is_tensor(mapping):
        return mapping.tolist()
    return [item.tolist() if torch.is_tensor(item) else item for item in mapping]


def encode_batch(tokenizer, rows):
    prompts = [row["prompt"] for row in rows]
    encoded = tokenizer(
        prompts,
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    positions, counts = name_positions(
        as_offset_rows(encoded.pop("offset_mapping")),
        [row["name_char_start"] for row in rows],
        [row["name_char_end"] for row in rows],
    )
    return encoded, positions, counts


def layer_modules(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(f"unsupported architecture: {type(model).__name__}")


def neuron_module(layer):
    mlp = layer.mlp
    for name in ("down_proj", "dense_4h_to_h", "c_proj"):
        module = getattr(mlp, name, None)
        if module is not None:
            return module
    raise RuntimeError(f"no MLP down projection on {type(layer).__name__}")


def collect(model, tokenizer, rows, device, batch_size: int):
    layers = layer_modules(model)
    n_layers = len(layers)
    residual = [[] for _ in range(n_layers)]
    neurons = [[] for _ in range(n_layers)]
    logits = []
    token_counts = []
    hooks = []

    def make_neuron_hook(layer_index, positions):
        state = {"fired": False}

        def hook(module, inputs):
            if state["fired"]:
                return
            state["fired"] = True
            activation = inputs[0]
            if activation.ndim != 3:
                raise RuntimeError(
                    f"expected (batch, sequence, neurons), got {tuple(activation.shape)}"
                )
            batch = torch.arange(activation.size(0), device=activation.device)
            neurons[layer_index].append(
                activation[batch, positions].detach().to("cpu", torch.float32)
            )

        return hook

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded, positions, counts = encode_batch(tokenizer, batch)
        token_counts.extend(counts)
        pos = torch.tensor(positions, device=device)
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
            if torch.is_tensor(value)
        }
        for index, layer in enumerate(layers):
            hooks.append(
                neuron_module(layer).register_forward_pre_hook(make_neuron_hook(index, pos))
            )
        with torch.no_grad():
            output = model(**encoded, output_hidden_states=True, use_cache=False)
        for hook in hooks:
            hook.remove()
        hooks.clear()
        hidden = output.hidden_states
        if len(hidden) != n_layers + 1:
            raise RuntimeError(
                f"expected {n_layers + 1} hidden states, got {len(hidden)}"
            )
        batch_index = torch.arange(pos.size(0), device=device)
        for index in range(n_layers):
            # hidden[0] is the embedding stream. hidden[i + 1] is layer i.
            residual[index].append(
                hidden[index + 1][batch_index, pos].detach().to("cpu", torch.float32)
            )
        last = encoded["attention_mask"].sum(dim=1) - 1
        logits.append(output.logits[batch_index, last].detach().to("cpu", torch.float32))
        print(f"  collected {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    packed = {
        "residual": [torch.cat(parts, dim=0).numpy() for parts in residual],
        "neurons": [torch.cat(parts, dim=0).numpy() for parts in neurons],
        "logits": torch.cat(logits, dim=0).numpy(),
        "n_tokens": np.asarray(token_counts, dtype=np.int32),
    }
    if packed["residual"][0].shape[0] != len(rows) or packed["neurons"][0].shape[0] != len(rows):
        raise RuntimeError(
            "activation rows "
            f"{packed['residual'][0].shape[0]}/{packed['neurons'][0].shape[0]} "
            f"do not match prompts {len(rows)}"
        )
    return packed


def place_token_ids(tokenizer) -> dict[str, int]:
    mapping = {}
    for place in PLACES:
        pieces = tokenizer.encode(" " + place, add_special_tokens=False)
        if not pieces:
            pieces = tokenizer.encode(place, add_special_tokens=False)
        mapping[place] = pieces[0]
        print(
            f"  place {place}: {tokenizer.convert_ids_to_tokens(pieces)!r}",
            flush=True,
        )
    return mapping


def fit_probe(train_x, train_y, test_x, test_y):
    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_x)
    test_s = scaler.transform(test_x)
    clf = LogisticRegression(max_iter=500, tol=1e-3)
    clf.fit(train_s, train_y)
    pred = clf.predict(test_s)
    accuracy = float(np.mean(pred == test_y))
    raw = clf.coef_ / scaler.scale_
    return accuracy, raw, pred


def shuffled_accuracy(train_x, train_y, test_x, test_y, repeats: int = 5) -> float:
    rng = np.random.default_rng(SEED)
    scores = []
    for _ in range(repeats):
        mixed = rng.permutation(train_y)
        accuracy, _, _ = fit_probe(train_x, mixed, test_x, test_y)
        scores.append(accuracy)
    return float(np.mean(scores))


def token_accuracy(train_n, train_y, test_n, test_y) -> float:
    clf = LogisticRegression(max_iter=1000)
    clf.fit(train_n.reshape(-1, 1), train_y)
    return float(clf.score(test_n.reshape(-1, 1), test_y))


def subspace_dimension(train_x, train_y, test_x, test_y, full_accuracy, raw) -> int:
    if full_accuracy <= 0:
        return 0
    _, _, vt = np.linalg.svd(raw, full_matrices=False)
    target = 0.9 * full_accuracy
    best = int(vt.shape[0])
    for k in range(1, vt.shape[0] + 1):
        basis = vt[:k].T
        accuracy, _, _ = fit_probe(train_x @ basis, train_y, test_x @ basis, test_y)
        if accuracy + 1e-9 >= target:
            return k
        best = k
    return best


def top_basis(raw: np.ndarray, k: int) -> np.ndarray:
    """Orthonormal directions in activation space, largest probe singular vectors first."""
    k = max(1, min(int(k), raw.shape[0], raw.shape[1]))
    _, _, vt = np.linalg.svd(raw, full_matrices=False)
    basis = vt[:k].T
    q, _ = np.linalg.qr(basis)
    return np.ascontiguousarray(q[:, :k], dtype=np.float32)


def subset_score(pred, y_test, mask_in_test) -> float | None:
    if not np.any(mask_in_test):
        return None
    return float(np.mean(pred[mask_in_test] == y_test[mask_in_test]))


def evaluate_split(
    features, labels, train_mask, test_mask, repeats: int, one_token=None, with_baselines: bool = True
):
    y_train = labels[train_mask]
    y_test = labels[test_mask]
    if len(np.unique(y_train)) < 2 or len(y_test) == 0:
        return None
    x_train = features[train_mask]
    x_test = features[test_mask]
    accuracy, raw, pred = fit_probe(x_train, y_train, x_test, y_test)
    result = {"accuracy": accuracy}
    if one_token is not None:
        in_test = one_token[test_mask]
        result["accuracy_one_token"] = subset_score(pred, y_test, in_test)
        result["accuracy_multi_token"] = subset_score(pred, y_test, ~in_test)
    if not with_baselines:
        return result
    dimension = subspace_dimension(x_train, y_train, x_test, y_test, accuracy, raw)
    result["dimension"] = dimension
    result["direction"] = top_basis(raw, max(dimension, 1))
    result["shuffled_accuracy"] = (
        None
        if repeats <= 0
        else shuffled_accuracy(x_train, y_train, x_test, y_test, repeats=repeats)
    )
    return result


def place_scores(logits: np.ndarray, place_ids: dict[str, int], rows: list[dict]):
    mass, correct, canada = [], [], []
    winners = []
    grouped: dict[str, dict[str, list]] = {}
    for vector, row in zip(logits, rows):
        if not row["places"]:
            continue
        probs = torch.softmax(torch.tensor(vector), dim=-1)
        named = {place: float(probs[index]) for place, index in place_ids.items()}
        winner = max(named, key=named.get)
        winners.append(winner)
        hit = int(winner in row["places"])
        accepted = sum(named[place] for place in row["places"])
        correct.append(hit)
        mass.append(accepted)
        canada.append(named["Canada"])
        bucket = grouped.setdefault(row["language_code"], {"correct": [], "mass": []})
        bucket["correct"].append(hit)
        bucket["mass"].append(accepted)
    if not mass:
        return {"n": 0}
    return {
        "n": len(mass),
        "place_accuracy": float(np.mean(correct)),
        "acceptable_mass": float(np.mean(mass)),
        "canada_mass": float(np.mean(canada)),
        "winners": {place: int(winners.count(place)) for place in PLACES},
        "by_language": {
            code: {
                "n": len(bucket["correct"]),
                "place_accuracy": float(np.mean(bucket["correct"])),
                "acceptable_mass": float(np.mean(bucket["mass"])),
            }
            for code, bucket in sorted(grouped.items())
        },
    }


def ablate(model, tokenizer, rows, layer_index, basis, place_ids, device, batch_size: int, site: str):
    basis_t = torch.tensor(np.ascontiguousarray(basis), dtype=torch.float32, device=device)
    layer = layer_modules(model)[layer_index]
    state = {"positions": None}

    def project_out(hidden):
        hidden = hidden.clone()
        batch = torch.arange(hidden.size(0), device=hidden.device)
        pos = state["positions"]
        vector = hidden[batch, pos].to(basis_t.dtype)
        hidden[batch, pos] = vector - (vector @ basis_t @ basis_t.T)
        return hidden

    def residual_hook(module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        rest = output[1:] if isinstance(output, tuple) else None
        if not torch.is_tensor(hidden) or hidden.ndim != 3:
            raise RuntimeError(f"layer output is not a residual tensor: {type(output)}")
        hidden = project_out(hidden)
        if rest is None:
            return hidden
        return (hidden, *rest)

    def neuron_hook(module, inputs):
        activation = inputs[0]
        if activation.ndim != 3:
            raise RuntimeError(f"expected neuron activation, got {tuple(activation.shape)}")
        return (project_out(activation),) + tuple(inputs[1:])

    if site == "residual":
        handle = layer.register_forward_hook(residual_hook)
    elif site == "neurons":
        handle = neuron_module(layer).register_forward_pre_hook(neuron_hook)
    else:
        raise RuntimeError(site)
    pieces = []
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded, positions, _ = encode_batch(tokenizer, batch)
            state["positions"] = torch.tensor(positions, device=device)
            encoded = {
                key: value.to(device)
                for key, value in encoded.items()
                if torch.is_tensor(value)
            }
            with torch.no_grad():
                output = model(**encoded, use_cache=False)
            last = encoded["attention_mask"].sum(dim=1) - 1
            batch_index = torch.arange(last.size(0), device=device)
            pieces.append(output.logits[batch_index, last].detach().to("cpu", torch.float32))
    finally:
        handle.remove()
    logits = torch.cat(pieces, dim=0).numpy()
    return place_scores(logits, place_ids, rows)


def random_basis(width: int, k: int, draw: int) -> np.ndarray:
    rng = np.random.default_rng(SEED + draw)
    matrix = rng.normal(size=(width, k))
    q, _ = np.linalg.qr(matrix)
    return q[:, :k]


def slug(model_name: str) -> str:
    return model_name.replace("/", "__")


def public_probe(result, token_baseline):
    if result is None:
        return {
            "accuracy": None,
            "shuffled_accuracy": None,
            "token_count_accuracy": token_baseline,
            "dimension": None,
        }
    row = {
        "accuracy": result["accuracy"],
        "shuffled_accuracy": result.get("shuffled_accuracy"),
        "token_count_accuracy": token_baseline,
        "dimension": result.get("dimension"),
    }
    if "accuracy_one_token" in result:
        row["accuracy_one_token"] = result["accuracy_one_token"]
        row["accuracy_multi_token"] = result["accuracy_multi_token"]
    return row


def run_model(
    model_name: str, rows: list[dict], device, batch_size: int, repeats: int, token: str | None
) -> dict:
    print(f"\n=== {model_name} ===", flush=True)
    hub = {"token": token} if token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_name, **hub)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    try:
        model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, **hub)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32, **hub)
    model.to(device)
    model.eval()
    place_ids = place_token_ids(tokenizer)
    print(" collecting activations", flush=True)
    cache = collect(model, tokenizer, rows, device, batch_size)
    train = np.asarray([row["split"] == "train" for row in rows])
    test = ~train
    binary = np.asarray([row["is_east_african"] for row in rows])
    language_rows = np.asarray([row["include_in_language_probe"] == 1 for row in rows])
    languages = sorted({row["language_code"] for row in rows if row["include_in_language_probe"] == 1})
    language_id = {code: index for index, code in enumerate(languages)}
    language = np.asarray(
        [language_id.get(row["language_code"], -1) for row in rows]
    )
    n_tokens = cache["n_tokens"]
    one_token = n_tokens == 1
    token_binary = token_accuracy(n_tokens[train], binary[train], n_tokens[test], binary[test])
    token_language = None
    if len(np.unique(language[train & language_rows])) >= 2:
        token_language = token_accuracy(
            n_tokens[train & language_rows],
            language[train & language_rows],
            n_tokens[test & language_rows],
            language[test & language_rows],
        )

    def curve(site: str):
        bank = cache[site]
        binary_curve = []
        language_curve = []
        best_binary_layer = None
        best_binary_accuracy = -1.0
        best_language_layer = None
        best_language_accuracy = -1.0
        for index, features in enumerate(bank):
            binary_result = evaluate_split(
                features,
                binary,
                train,
                test,
                repeats,
                one_token=one_token,
                with_baselines=False,
            )
            language_result = evaluate_split(
                features,
                language,
                train & language_rows,
                test & language_rows,
                repeats,
                with_baselines=False,
            )
            binary_row = public_probe(binary_result, token_binary)
            binary_row["layer"] = index
            language_row = public_probe(language_result, token_language)
            language_row["layer"] = index
            binary_curve.append(binary_row)
            language_curve.append(language_row)
            if binary_result is not None and binary_result["accuracy"] > best_binary_accuracy:
                best_binary_accuracy = binary_result["accuracy"]
                best_binary_layer = index
            if language_result is not None and language_result["accuracy"] > best_language_accuracy:
                best_language_accuracy = language_result["accuracy"]
                best_language_layer = index
            binary_text = (
                "n/a"
                if binary_row["accuracy"] is None
                else f"{binary_row['accuracy']:.3f}"
            )
            language_text = (
                ""
                if language_row["accuracy"] is None
                else f" language {language_row['accuracy']:.3f}"
            )
            print(f"  {site} layer {index}: binary {binary_text}{language_text}", flush=True)

        def finish(layer_index, labels, train_mask, test_mask, curve_rows, tokens):
            if layer_index is None:
                return None
            print(f"  {site} baselines at layer {layer_index}", flush=True)
            result = evaluate_split(
                bank[layer_index],
                labels,
                train_mask,
                test_mask,
                repeats,
                one_token=tokens,
            )
            if result is None:
                return None
            result["layer"] = layer_index
            curve_rows[layer_index].update(public_probe(result, curve_rows[layer_index]["token_count_accuracy"]))
            curve_rows[layer_index]["layer"] = layer_index
            return result

        early_binary = finish(0, binary, train, test, binary_curve, one_token)
        best_binary = (
            early_binary
            if best_binary_layer in (None, 0)
            else finish(best_binary_layer, binary, train, test, binary_curve, one_token)
        )
        best_language = finish(
            best_language_layer,
            language,
            train & language_rows,
            test & language_rows,
            language_curve,
            None,
        )
        return binary_curve, language_curve, best_binary, best_language, early_binary

    print(" probing residual stream", flush=True)
    residual_binary, residual_language, best_binary, best_language, early_binary = curve("residual")
    print(" probing MLP neurons", flush=True)
    neuron_binary, neuron_language, best_neuron_binary, best_neuron_language, early_neuron = curve(
        "neurons"
    )

    east_index = [i for i, row in enumerate(rows) if test[i] and row["places"]]
    east_test = [rows[i] for i in east_index]
    clean = place_scores(cache["logits"][east_index], place_ids, east_test)
    ablation = []
    targets = []
    for site, name, result, kind in (
        ("residual", "early", early_binary, "binary"),
        ("residual", "best_binary", best_binary, "binary"),
        ("residual", "best_language", best_language, "language"),
        ("neurons", "early", early_neuron, "binary"),
        ("neurons", "best_binary", best_neuron_binary, "binary"),
        ("neurons", "best_language", best_neuron_language, "language"),
    ):
        if result is None:
            continue
        targets.append((site, name, result["layer"], result["direction"], kind))
    seen = set()
    for site, name, layer_index, basis, kind in targets:
        key = (site, layer_index, kind, int(basis.shape[1]))
        if key in seen:
            continue
        seen.add(key)
        print(
            f" ablating {site} {name} layer {layer_index} k={basis.shape[1]}",
            flush=True,
        )
        true_scores = ablate(
            model, tokenizer, east_test, layer_index, basis, place_ids, device, batch_size, site
        )
        random_mass, random_acc = [], []
        for draw in range(5):
            guess = random_basis(basis.shape[0], basis.shape[1], draw)
            scored = ablate(
                model,
                tokenizer,
                east_test,
                layer_index,
                guess,
                place_ids,
                device,
                batch_size,
                site,
            )
            random_mass.append(scored["acceptable_mass"])
            random_acc.append(scored["place_accuracy"])
        ablation.append(
            {
                "site": site,
                "name": name,
                "layer": layer_index,
                "kind": kind,
                "k": int(basis.shape[1]),
                "clean_place_accuracy": clean["place_accuracy"],
                "clean_acceptable_mass": clean["acceptable_mass"],
                "ablated_place_accuracy": true_scores["place_accuracy"],
                "ablated_acceptable_mass": true_scores["acceptable_mass"],
                "ablated_canada_mass": true_scores["canada_mass"],
                "random_place_accuracy": float(np.mean(random_acc)),
                "random_acceptable_mass": float(np.mean(random_mass)),
                "ablated_by_language": true_scores.get("by_language"),
            }
        )

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    values, counts = np.unique(n_tokens, return_counts=True)
    return {
        "model": model_name,
        "n": len(rows),
        "n_layers": len(residual_binary),
        "languages": languages,
        "n_tokens": {
            "histogram": {str(int(value)): int(count) for value, count in zip(values, counts)},
            "east_african_mean": float(n_tokens[binary == 1].mean()) if np.any(binary == 1) else None,
            "control_mean": float(n_tokens[binary == 0].mean()) if np.any(binary == 0) else None,
        },
        "token_count_baseline": {"binary": token_binary, "language": token_language},
        "clean_place": clean,
        "residual_binary": residual_binary,
        "residual_language": residual_language,
        "neuron_binary": neuron_binary,
        "neuron_language": neuron_language,
        "ablation": ablation,
        "best_residual_binary_layer": None if best_binary is None else best_binary["layer"],
        "best_residual_language_layer": None if best_language is None else best_language["layer"],
        "best_neuron_binary_layer": None if best_neuron_binary is None else best_neuron_binary["layer"],
        "best_neuron_language_layer": None
        if best_neuron_language is None
        else best_neuron_language["layer"],
    }


def json_ready(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="Rows per split and set. 0 means all.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--shuffle-repeats", type=int, default=5)
    parser.add_argument("--models", nargs="*", default=MODELS)
    args = parser.parse_args()
    token = load_project_env()
    RESULTS.mkdir(exist_ok=True)
    rows = load_rows(args.limit)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"device {device} rows {len(rows)} hf_token {'yes' if token else 'no'}",
        flush=True,
    )
    summaries = []
    for model_name in args.models:
        try:
            summary = run_model(
                model_name, rows, device, args.batch_size, args.shuffle_repeats, token
            )
        except Exception as exc:
            message = redact(str(exc))
            print(redact("".join(traceback.format_exception(exc))), flush=True)
            print(f"FAILED {model_name}: {message}", flush=True)
            summary = {"model": model_name, "error": message}
        path = RESULTS / f"{slug(model_name)}.json"
        path.write_text(json.dumps(summary, indent=2, default=json_ready) + "\n")
        summaries.append({"model": model_name, "path": str(path), "error": summary.get("error")})
        print(f" wrote {path}", flush=True)
    (RESULTS / "index.json").write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
