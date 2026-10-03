"""Shared paths, environment, and model helpers for the name experiments.

Every experiment script imports this module. It loads `.env` before
`transformers` is imported, so a Hugging Face token and the download timeouts
are already set when a model is fetched. Activations elsewhere are read from
block-output hooks, not from `output.hidden_states[-1]`, which is the residual
after the final layer norm.
"""

from __future__ import annotations

import csv
import json
import os
import random
import re
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

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DATA = ROOT / "data" / "names.csv"
CONFIG = json.loads((ROOT / "data" / "config.json").read_text())
RESULTS = ROOT / "results"
SEED = int(CONFIG["seed"])
TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]+")


def redact(text: str) -> str:
    """Strip a Hugging Face token before an error is printed or stored."""
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
    """Read the name table. `limit` keeps that many rows from each split and set.

    A non-positive limit returns the full 1,200-row table. The four Mugisha and
    Rukundo rows stay in the file; callers drop them from the six-way probe.
    """
    with DATA.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["is_east_african"] = int(row["is_east_african"])
        row["include_in_language_probe"] = int(row["include_in_language_probe"])
        row["name_char_start"] = int(row["name_char_start"])
        row["name_char_end"] = int(row["name_char_end"])
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


def as_offset_rows(mapping):
    """Character offsets for each token, one list of (start, end) pairs per row."""
    if torch.is_tensor(mapping):
        return mapping.tolist()
    return [item.tolist() if torch.is_tensor(item) else item for item in mapping]


def layer_modules(model):
    """Transformer blocks, in order. Qwen uses `model.layers`; GPT-2 uses `transformer.h`."""
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers
    if hasattr(model, "gpt_neox"):
        return model.gpt_neox.layers
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h
    raise RuntimeError(f"unsupported architecture: {type(model).__name__}")


def slug(model_name: str) -> str:
    """Turn `org/name` into a filename. Result files are `{slug}.json`."""
    return model_name.replace("/", "__")
