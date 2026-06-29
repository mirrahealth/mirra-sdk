"""Mirra — evaluate retrieval & RAG agents inline.

    from mirra import evaluate, Dataset

    def my_agent(query: str) -> dict:
        docs = retriever.search(query)
        return {
            "answer": llm.answer(query, docs),
            "contexts": [d.text for d in docs],
            "context_ids": [d.id for d in docs],
        }

    report = evaluate(my_agent, Dataset.from_jsonl("eval.jsonl"), k=5)
    print(report)                 # aggregate scores
    report.cases[0].metrics       # per-query scores, returned inline
"""

from .client import Mirra, evaluate
from .dataset import Case, Dataset
from .judges import Judge
from .report import CaseResult, Report

__version__ = "0.1.0"
__all__ = [
    "Mirra",
    "evaluate",
    "Dataset",
    "Case",
    "Report",
    "CaseResult",
    "Judge",
]
