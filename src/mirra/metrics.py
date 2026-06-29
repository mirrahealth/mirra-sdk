"""Label-based retrieval metrics — computed exactly from ground-truth ids."""

from __future__ import annotations

import math


def hit_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    return 1.0 if any(d in rel for d in retrieved[:k]) else 0.0


def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    if k <= 0:
        return 0.0
    rel = set(relevant)
    return sum(1 for d in retrieved[:k] if d in rel) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    rel = set(relevant)
    if not rel:
        return 0.0
    return sum(1 for d in retrieved[:k] if d in rel) / len(rel)


def mrr(retrieved: list[str], relevant: list[str]) -> float:
    """Reciprocal rank of the first relevant document."""
    rel = set(relevant)
    for i, d in enumerate(retrieved, start=1):
        if d in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Binary-relevance nDCG@k."""
    rel = set(relevant)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, d in enumerate(retrieved[:k], start=1)
        if d in rel
    )
    ideal_hits = min(len(rel), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0
