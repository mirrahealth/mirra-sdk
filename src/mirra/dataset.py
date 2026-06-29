"""Evaluation dataset — a list of queries, optionally labeled.

A case is *labeled* when it carries `relevant_ids` (ground-truth relevant
document ids) and/or an `expected_answer` (reference answer). Labels unlock
exact retrieval metrics and answer correctness; unlabeled cases fall back to
the LLM judge (the "hybrid" model).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Iterable, Iterator, Optional


@dataclass
class Case:
    query: str
    relevant_ids: list[str] = field(default_factory=list)  # ground-truth relevant doc ids
    expected_answer: Optional[str] = None                  # reference answer
    meta: dict = field(default_factory=dict)

    @property
    def labeled_retrieval(self) -> bool:
        return bool(self.relevant_ids)

    @property
    def labeled_answer(self) -> bool:
        return self.expected_answer is not None


class Dataset:
    def __init__(self, cases: list[Case]):
        self.cases = cases

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self) -> Iterator[Case]:
        return iter(self.cases)

    # ── loaders ────────────────────────────────────────────────
    @classmethod
    def from_dicts(cls, rows: Iterable[dict]) -> "Dataset":
        return cls([
            Case(
                query=r["query"],
                relevant_ids=list(r.get("relevant_ids", []) or []),
                expected_answer=r.get("expected_answer"),
                meta=r.get("meta", {}) or {},
            )
            for r in rows
        ])

    @classmethod
    def from_jsonl(cls, path: str) -> "Dataset":
        with open(path, "r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
        return cls.from_dicts(rows)
