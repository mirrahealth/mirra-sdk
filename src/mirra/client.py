"""The Mirra client — wrap a retrieval/RAG agent and get scores inline.

Two entry points:
  • Mirra().evaluate(agent_fn, dataset)  → run the whole dataset, return a Report
  • Mirra().score(query, answer=..., contexts=...)  → score one case inline
"""

from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Callable, Optional, Union

from .dataset import Dataset
from .judges import Judge
from .metrics import hit_at_k, mrr, ndcg_at_k, precision_at_k, recall_at_k
from .report import CaseResult, Report

# Mirra Retrieval Score weighting (renormalized over whichever pillars exist)
PILLAR_WEIGHTS = {"retrieval": 0.4, "faithfulness": 0.3, "answer": 0.3}
# the hosted Mirra app (override with MIRRA_BASE_URL, e.g. http://localhost:3000)
DEFAULT_BASE_URL = os.getenv(
    "MIRRA_BASE_URL", "https://mirra-eval-leqtfzljxa-uc.a.run.app"
)


def _overall(pillars: dict) -> float:
    present = {k: v for k, v in pillars.items() if v is not None}
    if not present:
        return 0.0
    total_w = sum(PILLAR_WEIGHTS[k] for k in present)
    return sum(PILLAR_WEIGHTS[k] * v for k, v in present.items()) / total_w


def _normalize_output(out: Any) -> tuple[Optional[str], list[str], list[str]]:
    """Accept dict / object / str from the agent; return (answer, contexts, context_ids)."""
    if out is None:
        return None, [], []
    if isinstance(out, str):
        return out, [], []

    get = (lambda k: out.get(k)) if isinstance(out, dict) else (lambda k: getattr(out, k, None))
    answer = get("answer")
    raw_contexts = get("contexts") or []
    context_ids = list(get("context_ids") or [])

    contexts: list[str] = []
    for c in raw_contexts:
        if isinstance(c, dict):
            contexts.append(str(c.get("text") or c.get("content") or ""))
            if not context_ids and (c.get("id") is not None):
                context_ids.append(str(c["id"]))
        else:
            contexts.append(str(c))
    return (str(answer) if answer is not None else None, contexts, context_ids)


class Mirra:
    def __init__(
        self,
        api_key: Optional[str] = None,
        judge: Union[Judge, dict, None] = None,
        k: int = 5,
        base_url: str = DEFAULT_BASE_URL,
    ):
        self.api_key = api_key or os.getenv("MIRRA_API_KEY")
        self.k = k
        self.base_url = base_url.rstrip("/")
        if isinstance(judge, Judge):
            self.judge = judge
        elif isinstance(judge, dict):
            self.judge = Judge(**judge)
        else:
            self.judge = Judge()  # auto-detect from env

    # ── score one case (inline) ────────────────────────────────
    def score(
        self,
        query: str,
        *,
        answer: Optional[str] = None,
        contexts: Optional[list[str]] = None,
        context_ids: Optional[list[str]] = None,
        relevant_ids: Optional[list[str]] = None,
        expected_answer: Optional[str] = None,
        k: Optional[int] = None,
    ) -> CaseResult:
        k = k or self.k
        contexts = contexts or []
        context_ids = context_ids or []
        relevant_ids = relevant_ids or []
        metrics: dict = {}
        labeled = bool(relevant_ids) or (expected_answer is not None)

        # ── retrieval pillar ──
        retrieval_pillar: Optional[float] = None
        if relevant_ids and context_ids:
            metrics["precision@k"] = precision_at_k(context_ids, relevant_ids, k)
            metrics["recall@k"] = recall_at_k(context_ids, relevant_ids, k)
            metrics["mrr"] = mrr(context_ids, relevant_ids)
            metrics["ndcg@k"] = ndcg_at_k(context_ids, relevant_ids, k)
            metrics["hit@k"] = hit_at_k(context_ids, relevant_ids, k)
            retrieval_pillar = metrics["ndcg@k"]
        if self.judge.available and contexts:
            metrics["context_relevance"] = self.judge.context_relevance(query, contexts)
            if retrieval_pillar is None:  # no labels → judge carries the pillar
                retrieval_pillar = metrics["context_relevance"]

        # ── generation pillars (RAG) ──
        faithfulness_pillar: Optional[float] = None
        answer_pillar: Optional[float] = None
        if answer is not None and self.judge.available:
            if contexts:
                metrics["faithfulness"] = self.judge.faithfulness(answer, contexts)
                faithfulness_pillar = metrics["faithfulness"]
            if expected_answer is not None:
                metrics["answer_correctness"] = self.judge.answer_correctness(answer, expected_answer)
                answer_pillar = metrics["answer_correctness"]
            else:
                metrics["answer_relevance"] = self.judge.answer_relevance(query, answer)
                answer_pillar = metrics["answer_relevance"]

        pillars = {
            "retrieval": retrieval_pillar,
            "faithfulness": faithfulness_pillar,
            "answer": answer_pillar,
        }
        return CaseResult(
            query=query,
            answer=answer,
            context_ids=context_ids,
            relevant_ids=relevant_ids,
            metrics=metrics,
            pillars=pillars,
            overall=_overall(pillars),
            labeled=labeled,
        )

    # ── run the whole dataset ──────────────────────────────────
    def evaluate(
        self,
        agent_fn: Callable[[str], Any],
        dataset: Union[Dataset, list],
        *,
        k: Optional[int] = None,
        name: Optional[str] = None,
        on_case: Optional[Callable[[CaseResult], None]] = None,
        progress: bool = True,
    ) -> Report:
        if isinstance(dataset, list):
            dataset = Dataset.from_dicts(dataset) if dataset and isinstance(dataset[0], dict) else Dataset(dataset)
        k = k or self.k
        results: list[CaseResult] = []
        total = len(dataset)
        for i, case in enumerate(dataset, start=1):
            answer, contexts, context_ids = _normalize_output(agent_fn(case.query))
            cr = self.score(
                case.query,
                answer=answer,
                contexts=contexts,
                context_ids=context_ids,
                relevant_ids=case.relevant_ids,
                expected_answer=case.expected_answer,
                k=k,
            )
            results.append(cr)
            if on_case:
                on_case(cr)
            if progress:
                print(f"  [{i}/{total}] {case.query[:48]:<48}  score={cr.overall:.3f}", flush=True)

        report = Report.build(results, k)
        if self.api_key:
            self._sync(report, name)
        return report

    # ── optional: push the report to the Mirra dashboard ───────
    def _sync(self, report: Report, name: Optional[str] = None) -> None:
        try:
            body = json.dumps(
                {"kind": "retrieval", "name": name or "Retrieval eval", "report": report.to_dict()}
            ).encode()
            req = urllib.request.Request(
                f"{self.base_url}/api/sdk-runs",
                data=body,
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode())
                if data.get("url"):
                    print(f"  ↳ synced to Mirra: {data['url']}")
        except Exception as e:  # never let sync break a local eval
            print(f"  ↳ dashboard sync skipped ({e})")


def evaluate(agent_fn: Callable[[str], Any], dataset, **kwargs) -> Report:
    """Convenience: one-shot evaluation with a default client."""
    client_kwargs = {key: kwargs.pop(key) for key in ("api_key", "judge", "base_url") if key in kwargs}
    return Mirra(**client_kwargs).evaluate(agent_fn, dataset, **kwargs)
