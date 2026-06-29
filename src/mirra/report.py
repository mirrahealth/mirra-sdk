"""Per-case and aggregate evaluation results."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from statistics import mean
from typing import Optional


@dataclass
class CaseResult:
    query: str
    answer: Optional[str]
    context_ids: list[str]
    relevant_ids: list[str]
    metrics: dict           # all per-case metrics actually computed
    pillars: dict           # {retrieval, faithfulness, answer}
    overall: float          # weighted Mirra Retrieval Score for this case
    labeled: bool           # whether ground-truth labels were used

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "context_ids": self.context_ids,
            "relevant_ids": self.relevant_ids,
            "metrics": {k: round(v, 4) for k, v in self.metrics.items()},
            "pillars": {k: round(v, 4) for k, v in self.pillars.items() if v is not None},
            "overall": round(self.overall, 4),
            "labeled": self.labeled,
        }


@dataclass
class Report:
    cases: list[CaseResult]
    k: int
    summary: dict = field(default_factory=dict)

    # ── aggregation ────────────────────────────────────────────
    @classmethod
    def build(cls, cases: list[CaseResult], k: int) -> "Report":
        def avg(key: str) -> Optional[float]:
            vals = [c.metrics[key] for c in cases if key in c.metrics]
            return round(mean(vals), 4) if vals else None

        def avg_pillar(key: str) -> Optional[float]:
            vals = [c.pillars[key] for c in cases if c.pillars.get(key) is not None]
            return round(mean(vals), 4) if vals else None

        metric_keys = sorted({m for c in cases for m in c.metrics})
        summary = {
            "n": len(cases),
            "labeled": sum(1 for c in cases if c.labeled),
            "overall": round(mean([c.overall for c in cases]), 4) if cases else 0.0,
            "pillars": {
                "retrieval": avg_pillar("retrieval"),
                "faithfulness": avg_pillar("faithfulness"),
                "answer": avg_pillar("answer"),
            },
            "metrics": {key: avg(key) for key in metric_keys},
        }
        return cls(cases=cases, k=k, summary=summary)

    # ── output ─────────────────────────────────────────────────
    def to_dict(self) -> dict:
        return {"k": self.k, "summary": self.summary, "cases": [c.to_dict() for c in self.cases]}

    def to_json(self, path: Optional[str] = None, indent: int = 2) -> Optional[str]:
        data = json.dumps(self.to_dict(), indent=indent)
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(data)
            return None
        return data

    def summary_str(self) -> str:
        s = self.summary
        lines = [
            "─" * 52,
            f"  Mirra Retrieval Eval · {s['n']} queries (k={self.k})",
            f"  {s['labeled']}/{s['n']} labeled · {s['n'] - s['labeled']} judge-scored",
            "─" * 52,
            f"  MIRRA RETRIEVAL SCORE      {s['overall']:.3f}",
            "",
        ]
        for name, val in s["pillars"].items():
            if val is not None:
                bar = "█" * int(val * 20) + "░" * (20 - int(val * 20))
                lines.append(f"  {name:<14} {bar} {val:.3f}")
        lines.append("")
        for name, val in s["metrics"].items():
            if val is not None:
                lines.append(f"    {name:<20} {val:.3f}")
        lines.append("─" * 52)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary_str()
