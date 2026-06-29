"""Mirra-hosted benchmark datasets, downloaded + cached on first use.

    from mirra.benchmarks import load_radiology

    corpus, dataset = load_radiology()      # downloads once, caches in ~/.cache/mirra
    my_retriever.index(corpus)              # index Mirra's synthetic corpus
    report = evaluate(my_agent, dataset)    # score against the benchmark labels

`corpus` is a list of {id, text, ...} you load into YOUR retriever; `dataset`
is a Mirra Dataset of labeled queries. The data is 100% synthetic (no PHI).

Override the host with the MIRRA_BENCHMARK_URL env var (e.g. a GCS bucket).
"""

from __future__ import annotations

import json
import os
import urllib.request
from pathlib import Path

from .dataset import Dataset

DEFAULT_BASE_URL = os.getenv(
    "MIRRA_BENCHMARK_URL",
    "https://raw.githubusercontent.com/mirrahealth/mirra-sdk/main/benchmarks",
)


def _cache_dir() -> Path:
    d = Path(os.getenv("MIRRA_CACHE_DIR", str(Path.home() / ".cache" / "mirra")))
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(rel_path: str, base_url: str) -> Path:
    """Download base_url/rel_path into the cache (once) and return the local path."""
    local = _cache_dir() / rel_path.replace("/", "__")
    if local.exists() and local.stat().st_size > 0:
        return local
    url = f"{base_url.rstrip('/')}/{rel_path}"
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            data = resp.read()
    except Exception as e:
        raise RuntimeError(
            f"Could not download benchmark from {url} ({e}). "
            f"If the host is private, set MIRRA_BENCHMARK_URL to a reachable location."
        ) from e
    local.write_bytes(data)
    return local


def _read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_radiology(
    version: str = "v1",
    base_url: str = DEFAULT_BASE_URL,
    force: bool = False,
) -> tuple[list[dict], Dataset]:
    """Return (corpus, dataset) for the synthetic radiology benchmark.

    corpus : list of {id, text, modality, region, ...} — index these into your retriever.
    dataset: Mirra Dataset of labeled queries (query, relevant_ids, expected_answer).
    """
    folder = f"radiology-{version}"
    if force:
        for name in ("corpus.jsonl", "queries.jsonl"):
            p = _cache_dir() / f"{folder}/{name}".replace("/", "__")
            if p.exists():
                p.unlink()
    corpus = _read_jsonl(_fetch(f"{folder}/corpus.jsonl", base_url))
    queries = _read_jsonl(_fetch(f"{folder}/queries.jsonl", base_url))
    return corpus, Dataset.from_dicts(queries)
