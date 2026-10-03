#!/usr/bin/env python3
"""Language-focused checks on the name residual.

Four measurements, all scored by the six-way language label:

1. Project out a direction and fit the language probe again.
2. Patch the residual at the name from one language into another.
3. Separate the attention write from the MLP write.
4. Compare token position and character n-gram baselines.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_subspace as base

OUT = base.RESULTS / "language_mech"


class Probe:
    def fit(self, train_x, train_y):
        self.scaler = StandardScaler()
        train_s = self.scaler.fit_transform(train_x)
        self.clf = LogisticRegression(max_iter=500, tol=1e-3)
        self.clf.fit(train_s, train_y)
        self.raw = self.clf.coef_ / self.scaler.scale_
        return self

    def predict(self, features):
        return self.clf.predict(self.scaler.transform(features))

    def accuracy(self, features, labels):
        if len(labels) == 0:
            return None
        return float(np.mean(self.predict(features) == labels))


def json_ready(value):
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def recalls(pred, truth, languages) -> dict:
    scores = {}
    for index, name in enumerate(languages):
        mask = truth == index
        if np.any(mask):
            scores[name] = float(np.mean(pred[mask] == index))
    return scores


def row_basis(raw: np.ndarray) -> tuple[np.ndarray, int]:
    """Orthonormal basis of the probe's row space, estimated on the training fit."""
    _, singular, vt = np.linalg.svd(raw, full_matrices=False)
    if singular.size == 0 or singular[0] <= 0:
        basis = np.zeros((raw.shape[1], 1), dtype=np.float32)
        return basis, 1
    keep = max(1, int(np.sum(singular > 1e-4 * singular[0])))
    basis = vt[:keep].T
    orthogonal, _ = np.linalg.qr(basis)
    return np.ascontiguousarray(orthogonal[:, :keep], dtype=np.float32), keep


def name_hits(offset_rows, starts, ends) -> list[list[int]]:
    hits_all = []
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
        hits_all.append(hits)
    return hits_all


def prepare_batch(tokenizer, rows, device):
    encoded = tokenizer(
        [row["prompt"] for row in rows],
        return_offsets_mapping=True,
        padding=True,
        return_tensors="pt",
    )
    hits = name_hits(
        base.as_offset_rows(encoded.pop("offset_mapping")),
        [row["name_char_start"] for row in rows],
        [row["name_char_end"] for row in rows],
    )
    encoded = {key: value.to(device) for key, value in encoded.items() if torch.is_tensor(value)}
    return encoded, hits


def first_tensor(output):
    if torch.is_tensor(output):
        return output
    if isinstance(output, tuple):
        for item in output:
            if torch.is_tensor(item) and item.ndim == 3:
                return item
    raise RuntimeError(f"no (batch, sequence, hidden) tensor in {type(output).__name__}")


def replace_tensor(output, updated):
    if torch.is_tensor(output):
        return updated
    items = list(output)
    for index, item in enumerate(items):
        if torch.is_tensor(item) and item.ndim == 3:
            items[index] = updated
            return tuple(items)
    raise RuntimeError("no residual tensor to replace")


def component_modules(layer):
    attention = getattr(layer, "self_attn", None)
    if attention is None:
        attention = getattr(layer, "attn", None)
    if attention is None or not hasattr(layer, "mlp"):
        raise RuntimeError(f"cannot split {type(layer).__name__} into attention and MLP")
    return attention, layer.mlp


def gather_last(hidden, hits):
    positions = torch.tensor([item[-1] for item in hits], device=hidden.device)
    batch = torch.arange(hidden.size(0), device=hidden.device)
    return hidden[batch, positions]


def gather_spans(hidden, hits):
    last, first, mean = [], [], []
    for index, tokens in enumerate(hits):
        piece = hidden[index, tokens]
        last.append(piece[-1])
        first.append(piece[0])
        mean.append(piece.mean(dim=0))
    return torch.stack(last), torch.stack(first), torch.stack(mean)


def stack_layers(parts):
    return [torch.cat(layer_parts, dim=0).numpy() for layer_parts in parts]


def load_model(model_name, token, device):
    hub = {"token": token} if token else {}
    tokenizer = base.AutoTokenizer.from_pretrained(model_name, **hub)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    try:
        model = base.AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32, **hub)
    except TypeError:
        model = base.AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, **hub
        )
    model.to(device)
    model.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    return model, tokenizer


def collect_clean(model, tokenizer, rows, device, batch_size: int):
    layers = base.layer_modules(model)
    n_layers = len(layers)
    last_parts = [[] for _ in range(n_layers)]
    first_parts = [[] for _ in range(n_layers)]
    mean_parts = [[] for _ in range(n_layers)]
    block_parts = [[] for _ in range(n_layers)]
    attn_parts = [[] for _ in range(n_layers)]
    mlp_parts = [[] for _ in range(n_layers)]
    counts = []
    stash = {"hits": None, "attn": {}, "mlp": {}, "errors": [], "check": True}
    handles = []

    def make_hook(kind, layer_index):
        module_list = attn_parts if kind == "attn" else mlp_parts

        def hook(module, inputs, output):
            tensor = first_tensor(output)
            module_list[layer_index].append(
                gather_last(tensor, stash["hits"]).detach().to("cpu", torch.float32)
            )
            if stash["check"]:
                stash[kind][layer_index] = tensor.detach().clone()

        return hook

    def make_block_hook(layer_index):
        def hook(module, inputs, output):
            tensor = first_tensor(output)
            last, first, mean = gather_spans(tensor, stash["hits"])
            last_parts[layer_index].append(last.detach().to("cpu", torch.float32))
            first_parts[layer_index].append(first.detach().to("cpu", torch.float32))
            mean_parts[layer_index].append(mean.detach().to("cpu", torch.float32))
            block_parts[layer_index].append(last_parts[layer_index][-1])
            if stash["check"] and layer_index in stash["attn"] and layer_index in stash["mlp"]:
                total = inputs[0] + stash["attn"][layer_index] + stash["mlp"][layer_index]
                stash["errors"].append((total - tensor).abs().max().item())

        return hook

    for index, layer in enumerate(layers):
        attention, mlp = component_modules(layer)
        handles.append(attention.register_forward_hook(make_hook("attn", index)))
        handles.append(mlp.register_forward_hook(make_hook("mlp", index)))
        handles.append(layer.register_forward_hook(make_block_hook(index)))
    decomp_error = None
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded, hits = prepare_batch(tokenizer, batch, device)
            stash["hits"] = hits
            stash["attn"] = {}
            stash["mlp"] = {}
            with torch.no_grad():
                output = model(**encoded, output_hidden_states=True, use_cache=False)
            hidden = output.hidden_states
            if len(hidden) != n_layers + 1:
                raise RuntimeError(f"expected {n_layers + 1} hidden states, got {len(hidden)}")
            if stash["check"]:
                if len(stash["errors"]) != n_layers:
                    raise RuntimeError(
                        f"decomposition check saw {len(stash['errors'])} layers, expected {n_layers}"
                    )
                decomp_error = float(max(stash["errors"]))
                print(f"  residual = input + attention + MLP, max error {decomp_error:.3e}", flush=True)
                if decomp_error > 1e-2:
                    raise RuntimeError(
                        f"attention/MLP outputs do not add up to the residual ({decomp_error:.3e})"
                    )
                stash["check"] = False
            counts.extend(len(item) for item in hits)
            print(f"  collected {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    packed = {
        "last": stack_layers(last_parts),
        "first": stack_layers(first_parts),
        "mean": stack_layers(mean_parts),
        "block": stack_layers(block_parts),
        "attn": stack_layers(attn_parts),
        "mlp": stack_layers(mlp_parts),
        "n_tokens": np.asarray(counts, dtype=np.int32),
        "decomp_error": decomp_error,
    }
    if packed["last"][0].shape[0] != len(rows) or packed["attn"][0].shape[0] != len(rows):
        raise RuntimeError(
            "activation rows "
            f"{packed['last'][0].shape[0]}/{packed['attn'][0].shape[0]} "
            f"do not match prompts {len(rows)}"
        )
    return packed


def project_hook(basis, stash):
    def hook(module, inputs, output):
        tensor = first_tensor(output)
        updated = tensor.clone()
        positions = torch.tensor([item[-1] for item in stash["hits"]], device=updated.device)
        batch = torch.arange(updated.size(0), device=updated.device)
        vector = updated[batch, positions].to(basis.dtype)
        updated[batch, positions] = (vector - vector @ basis @ basis.T).to(updated.dtype)
        return replace_tensor(output, updated)

    return hook


def forward_last(model, tokenizer, rows, device, batch_size, layer_index, basis, site, source=None):
    """Residual at the last name piece. Optionally project out `basis` or patch `source`."""
    layers = base.layer_modules(model)
    stash = {"hits": None}
    handles = []
    if basis is not None:
        module = layers[layer_index]
        if site == "residual":
            target = module
        elif site == "attn":
            target = component_modules(module)[0]
        elif site == "mlp":
            target = component_modules(module)[1]
        else:
            raise RuntimeError(site)
        basis_t = torch.tensor(np.ascontiguousarray(basis), dtype=torch.float32, device=device)
        handles.append(target.register_forward_hook(project_hook(basis_t, stash)))
    if source is not None:
        source_t = torch.tensor(np.ascontiguousarray(source), dtype=torch.float32, device=device)
        cursor = {"start": 0}

        def patch(module, inputs, output):
            tensor = first_tensor(output)
            updated = tensor.clone()
            positions = torch.tensor([item[-1] for item in stash["hits"]], device=updated.device)
            batch = torch.arange(updated.size(0), device=updated.device)
            stop = cursor["start"] + updated.size(0)
            updated[batch, positions] = source_t[cursor["start"] : stop].to(updated.dtype)
            cursor["start"] = stop
            return replace_tensor(output, updated)

        handles.append(layers[layer_index].register_forward_hook(patch))
    parts = [[] for _ in layers]

    def make_record(index):
        def hook(module, inputs, output):
            tensor = first_tensor(output)
            parts[index].append(gather_last(tensor, stash["hits"]).detach().to("cpu", torch.float32))

        return hook

    for index, layer in enumerate(layers):
        handles.append(layer.register_forward_hook(make_record(index)))
    try:
        for start in range(0, len(rows), batch_size):
            batch = rows[start : start + batch_size]
            encoded, hits = prepare_batch(tokenizer, batch, device)
            stash["hits"] = hits
            with torch.no_grad():
                model(**encoded, use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    return stack_layers(parts)


def fit_on(features, labels, train, test) -> Probe:
    return Probe().fit(features[train], labels[train])


def scored(probe: Probe, features, labels, mask, languages=None):
    pred = probe.predict(features[mask])
    truth = labels[mask]
    row = {"accuracy": float(np.mean(pred == truth))}
    if languages is not None:
        row["recall"] = recalls(pred, truth, languages)
    return row


def refit_score(features, labels, train, test, languages=None):
    probe = fit_on(features, labels, train, test)
    return scored(probe, features, labels, test, languages)


def length_bins(pred, truth, n_tokens, mask):
    bins = {}
    tokens = n_tokens[mask]
    for count in sorted(set(tokens.tolist())):
        chosen = tokens == count
        if int(chosen.sum()) < 5:
            continue
        bins[str(int(count))] = {
            "n": int(chosen.sum()),
            "accuracy": float(np.mean(pred[chosen] == truth[chosen])),
        }
    return bins


def char_accuracy(names, labels, train, test, clip: str) -> dict:
    def clip_name(name: str) -> str:
        if clip == "first3":
            return name[:3]
        if clip == "last3":
            return name[-3:]
        return name

    vectorizer = CountVectorizer(analyzer="char", ngram_range=(2, 3), min_df=1)
    train_names = [clip_name(names[index]) for index in np.flatnonzero(train)]
    test_names = [clip_name(names[index]) for index in np.flatnonzero(test)]
    train_x = vectorizer.fit_transform(train_names)
    test_x = vectorizer.transform(test_names)
    clf = LogisticRegression(max_iter=500, tol=1e-3)
    clf.fit(train_x, labels[train])
    pred = clf.predict(test_x)
    truth = labels[test]
    return {
        "accuracy": float(np.mean(pred == truth)),
        "n_features": int(train_x.shape[1]),
    }


def layer_curve(bank, labels, train, test):
    curve = []
    probes = []
    best_layer, best_accuracy = None, -1.0
    for index, features in enumerate(bank):
        probe = fit_on(features, labels, train, test)
        accuracy = probe.accuracy(features[test], labels[test])
        curve.append(accuracy)
        probes.append(probe)
        if accuracy > best_accuracy:
            best_layer, best_accuracy = index, accuracy
    return curve, probes, best_layer


def intervention_scores(features, residual_probe, labels, train, test, languages, binary, binary_probe, binary_train, binary_test):
    language = scored(residual_probe, features, labels, test, languages)
    refit = refit_score(features, labels, train, test, languages)
    binary_original = binary_probe.accuracy(features[binary_test], binary[binary_test])
    binary_refit = refit_score(features, binary, binary_train, binary_test)["accuracy"]
    return {
        "language_original": language["accuracy"],
        "language_original_recall": language["recall"],
        "language_refit": refit["accuracy"],
        "language_refit_recall": refit["recall"],
        "binary_original": binary_original,
        "binary_refit": binary_refit,
    }


def mean_rows(rows: list[dict]) -> dict:
    keys = [key for key, value in rows[0].items() if isinstance(value, float)]
    summary = {key: float(np.mean([row[key] for row in rows])) for key in keys}
    summary["draws"] = len(rows)
    return summary


def make_pairs(language, n_tokens, pool):
    rng = np.random.default_rng(base.SEED)
    pairs = []
    matched = 0
    for dest in pool:
        same_length = [
            int(src)
            for src in pool
            if language[src] != language[dest] and n_tokens[src] == n_tokens[dest]
        ]
        chosen = same_length or [
            int(src) for src in pool if language[src] != language[dest]
        ]
        src = int(rng.choice(chosen))
        matched += int(bool(same_length))
        pairs.append((src, int(dest)))
    return pairs, matched


def run_model(model_name, rows, device, batch_size, token, experiments: set[str]) -> dict:
    print(f"\n=== {model_name} ===", flush=True)
    languages = sorted({row["language_code"] for row in rows if row["include_in_language_probe"] == 1})
    language_id = {code: index for index, code in enumerate(languages)}
    language = np.asarray([language_id.get(row["language_code"], -1) for row in rows])
    binary = np.asarray([row["is_east_african"] for row in rows])
    train = np.asarray([row["split"] == "train" for row in rows])
    test = ~train
    language_rows = language >= 0
    lang_train = train & language_rows
    lang_test = test & language_rows
    if len(np.unique(language[lang_train])) < 2 or not np.any(lang_test):
        raise RuntimeError("language split does not have train and test classes")

    model, tokenizer = load_model(model_name, token, device)
    print(" collecting activations", flush=True)
    clean = collect_clean(model, tokenizer, rows, device, batch_size)
    n_tokens = clean["n_tokens"]
    majority = int(np.bincount(language[lang_train]).argmax())
    majority_accuracy = float(np.mean(language[lang_test] == majority))
    token_baseline = base.token_accuracy(
        n_tokens[lang_train], language[lang_train], n_tokens[lang_test], language[lang_test]
    )
    names = [row["name"] for row in rows]
    orthography = {
        clip: char_accuracy(names, language, lang_train, lang_test, clip)
        for clip in ("full", "first3", "last3")
    }
    print(
        "  char n-gram "
        + " ".join(f"{clip} {orthography[clip]['accuracy']:.3f}" for clip in orthography),
        flush=True,
    )

    print(" probing residual positions", flush=True)
    position_curves = {}
    position_probes = {}
    position_best = {}
    for site in ("last", "first", "mean"):
        curve, probes, best_layer = layer_curve(clean[site], language, lang_train, lang_test)
        position_curves[site] = curve
        position_probes[site] = probes
        position_best[site] = best_layer
        print(f"  {site} best layer {best_layer} {curve[best_layer]:.3f}", flush=True)
    best_layer = position_best["last"]
    residual_probe = position_probes["last"][best_layer]
    basis, k = row_basis(residual_probe.raw)
    clean_pred = residual_probe.predict(clean["last"][best_layer][lang_test])
    clean_truth = language[lang_test]
    clean_recall = recalls(clean_pred, clean_truth, languages)

    print(" probing attention and MLP writes", flush=True)
    component_curves = {}
    component_probes = {}
    component_best = {}
    for site in ("attn", "mlp"):
        curve, probes, layer_index = layer_curve(clean[site], language, lang_train, lang_test)
        component_curves[site] = curve
        component_probes[site] = probes
        component_best[site] = layer_index
        print(f"  {site} best layer {layer_index} {curve[layer_index]:.3f}", flush=True)

    binary_probe = fit_on(clean["last"][best_layer], binary, train, test)
    binary_clean = binary_probe.accuracy(clean["last"][best_layer][test], binary[test])
    final_layer = len(clean["last"]) - 1
    final_probe = position_probes["last"][final_layer]

    result = {
        "model": model_name,
        "n": len(rows),
        "n_layers": len(clean["last"]),
        "languages": languages,
        "n_language_train": int(lang_train.sum()),
        "n_language_test": int(lang_test.sum()),
        "majority_baseline": majority_accuracy,
        "token_count_baseline": token_baseline,
        "decomp_error": clean["decomp_error"],
        "orthography": orthography,
        "positions": {
            site: {
                "best_layer": position_best[site],
                "accuracy": position_curves[site][position_best[site]],
                "by_layer": position_curves[site],
            }
            for site in ("last", "first", "mean")
        },
        "clean_language": {
            "layer": best_layer,
            "accuracy": position_curves["last"][best_layer],
            "dimension": k,
            "recall": clean_recall,
            "by_token_count": length_bins(clean_pred, clean_truth, n_tokens, lang_test),
            "final_layer": final_layer,
            "final_accuracy": position_curves["last"][final_layer],
        },
        "components": {
            site: {
                "best_layer": component_best[site],
                "accuracy": component_curves[site][component_best[site]],
                "by_layer": component_curves[site],
            }
            for site in ("attn", "mlp")
        },
        "clean_binary_accuracy": binary_clean,
    }

    def score_bank(bank):
        return intervention_scores(
            bank[best_layer],
            residual_probe,
            language,
            lang_train,
            lang_test,
            languages,
            binary,
            binary_probe,
            train,
            test,
        ) | {
            "final_language_original": final_probe.accuracy(bank[final_layer][lang_test], language[lang_test]),
            "final_language_refit": refit_score(bank[final_layer], language, lang_train, lang_test)["accuracy"],
        }

    if "ablation" in experiments:
        print(f" ablating language subspace at layer {best_layer} k={k}", flush=True)
        ablated = forward_last(
            model, tokenizer, rows, device, batch_size, best_layer, basis, "residual"
        )
        language_removed = score_bank(ablated)
        binary_basis, binary_k = row_basis(binary_probe.raw)
        print(f" ablating binary direction at layer {best_layer} k={binary_k}", flush=True)
        binary_ablated = forward_last(
            model, tokenizer, rows, device, batch_size, best_layer, binary_basis, "residual"
        )
        binary_removed = score_bank(binary_ablated)
        random_scores = []
        for draw in range(5):
            print(f" ablating random subspace draw {draw + 1}/5", flush=True)
            guess = base.random_basis(basis.shape[0], basis.shape[1], draw)
            random_bank = forward_last(
                model, tokenizer, rows, device, batch_size, best_layer, guess, "residual"
            )
            random_scores.append(score_bank(random_bank))
        result["ablation"] = {
            "layer": best_layer,
            "k": k,
            "binary_k": binary_k,
            "language_subspace": language_removed,
            "binary_direction": binary_removed,
            "random_subspace": mean_rows(random_scores),
        }
        print(
            "  refit after language removal "
            f"{language_removed['language_refit']:.3f}, after binary removal "
            f"{binary_removed['language_refit']:.3f}, after random "
            f"{result['ablation']['random_subspace']['language_refit']:.3f}",
            flush=True,
        )

    if "components" in experiments:
        causal = {}
        layers_to_test = sorted({best_layer, component_best["attn"], component_best["mlp"]})
        for layer_index in layers_to_test:
            causal[str(layer_index)] = {}
            layer_probe = position_probes["last"][layer_index]
            layer_binary = fit_on(clean["last"][layer_index], binary, train, test)
            for site in ("attn", "mlp"):
                print(f" removing {site} language write at layer {layer_index}", flush=True)
                site_basis, site_k = row_basis(component_probes[site][layer_index].raw)
                bank = forward_last(
                    model, tokenizer, rows, device, batch_size, layer_index, site_basis, site
                )
                causal[str(layer_index)][site] = {
                    "k": site_k,
                    "language_original": layer_probe.accuracy(bank[layer_index][lang_test], language[lang_test]),
                    "language_refit": refit_score(bank[layer_index], language, lang_train, lang_test)["accuracy"],
                    "final_language_refit": refit_score(bank[final_layer], language, lang_train, lang_test)["accuracy"],
                    "binary_original": layer_binary.accuracy(bank[layer_index][test], binary[test]),
                }
        result["component_ablation"] = causal

    if "patching" in experiments:
        pool = np.flatnonzero(lang_test)
        pairs, matched = make_pairs(language, n_tokens, pool)
        dest_rows = [rows[dest] for _, dest in pairs]
        print(f" patching {len(pairs)} pairs ({matched} length-matched)", flush=True)
        readouts = sorted({best_layer, final_layer})
        by_readout = {
            str(readout): []
            for readout in readouts
        }
        ceilings = {}
        for readout in readouts:
            probe = position_probes["last"][readout]
            src_index = np.asarray([src for src, _ in pairs])
            dest_index = np.asarray([dest for _, dest in pairs])
            src_pred = probe.predict(clean["last"][readout][src_index])
            dest_pred = probe.predict(clean["last"][readout][dest_index])
            ceilings[str(readout)] = {
                "source_ceiling": float(np.mean(src_pred == language[src_index])),
                "clean_dest_match": float(np.mean(dest_pred == language[dest_index])),
            }
        for layer_index in range(len(clean["last"])):
            source = np.stack([clean["block"][layer_index][src] for src, _ in pairs])
            patched = forward_last(
                model,
                tokenizer,
                dest_rows,
                device,
                batch_size,
                layer_index,
                None,
                "residual",
                source=source,
            )
            for readout in readouts:
                probe = position_probes["last"][readout]
                pred = probe.predict(patched[readout])
                src_labels = language[[src for src, _ in pairs]]
                dest_labels = language[[dest for _, dest in pairs]]
                by_readout[str(readout)].append(
                    {
                        "layer": layer_index,
                        "direct_copy": layer_index == readout,
                        "source_match": float(np.mean(pred == src_labels)),
                        "dest_match": float(np.mean(pred == dest_labels)),
                    }
                )
            src_match = by_readout[str(final_layer)][-1]["source_match"]
            dst_match = by_readout[str(final_layer)][-1]["dest_match"]
            print(
                f"  patch layer {layer_index}: final source {src_match:.3f} dest {dst_match:.3f}",
                flush=True,
            )
        for readout, curve in by_readout.items():
            transferable = [row for row in curve if not row["direct_copy"] and row["layer"] <= int(readout)]
            if transferable:
                peak = max(transferable, key=lambda row: row["source_match"])
                ceilings[readout]["best_transfer_layer"] = peak["layer"]
                ceilings[readout]["best_transfer_source_match"] = peak["source_match"]
                ceilings[readout]["best_transfer_dest_match"] = peak["dest_match"]
        result["patching"] = {
            "n_pairs": len(pairs),
            "n_length_matched": matched,
            "by_readout": ceilings,
            "curves": by_readout,
        }

    del model
    gc.collect()
    if device.type == "mps":
        torch.mps.empty_cache()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--models", nargs="*", default=["distilgpt2"])
    parser.add_argument(
        "--experiments",
        nargs="*",
        default=["ablation", "components", "patching"],
        choices=["ablation", "components", "patching"],
    )
    args = parser.parse_args()
    token = base.load_project_env()
    OUT.mkdir(parents=True, exist_ok=True)
    rows = base.load_rows(args.limit)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(
        f"device {device} rows {len(rows)} experiments {' '.join(args.experiments)}",
        flush=True,
    )
    for model_name in args.models:
        path = OUT / f"{base.slug(model_name)}.json"
        try:
            summary = run_model(
                model_name, rows, device, args.batch_size, token, set(args.experiments)
            )
        except Exception as exc:
            message = base.redact(str(exc))
            print(f"FAILED {model_name}: {message}", flush=True)
            summary = {"model": model_name, "error": message}
        path.write_text(json.dumps(summary, indent=2, default=json_ready) + "\n")
        print(f" wrote {path}", flush=True)


if __name__ == "__main__":
    main()
