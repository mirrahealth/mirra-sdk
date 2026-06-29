"""LLM-as-judge for reference-free (and reference-based) quality scoring.

Pluggable backend: OpenAI, Anthropic, or a custom callable. Auto-detects a
provider from the environment if not specified. Each judge returns a 0..1
score. Used for the "hybrid" fallback when labels are absent, and for RAG
quality (faithfulness, answer relevance) which has no label equivalent.
"""

from __future__ import annotations

import json
import os
import re
from typing import Callable, Optional


class Judge:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        fn: Optional[Callable[[str, str], str]] = None,
    ):
        self.fn = fn
        self._client = None
        self.api_key = api_key
        if fn is not None:
            self.provider = "custom"
            self.model = "custom"
            return
        if provider is None:
            if os.getenv("OPENAI_API_KEY"):
                provider = "openai"
            elif os.getenv("ANTHROPIC_API_KEY"):
                provider = "anthropic"
        self.provider = provider
        self.model = model or (
            "gpt-4o-mini" if provider == "openai" else
            "claude-haiku-4-5" if provider == "anthropic" else None
        )

    @property
    def available(self) -> bool:
        return self.provider is not None

    # ── backend ────────────────────────────────────────────────
    def _complete(self, system: str, user: str) -> str:
        if self.fn is not None:
            return self.fn(system, user)
        if self.provider == "openai":
            if self._client is None:
                from openai import OpenAI
                self._client = OpenAI(api_key=self.api_key)
            r = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                max_tokens=200,
            )
            return r.choices[0].message.content or "{}"
        if self.provider == "anthropic":
            if self._client is None:
                from anthropic import Anthropic
                self._client = Anthropic(api_key=self.api_key)
            r = self._client.messages.create(
                model=self.model,
                max_tokens=200,
                system=system + ' Respond ONLY with JSON {"score": <0..1>}.',
                messages=[{"role": "user", "content": user}],
            )
            return "".join(b.text for b in r.content if b.type == "text")
        raise RuntimeError(
            "No judge provider available. Set OPENAI_API_KEY or ANTHROPIC_API_KEY, "
            "pass Judge(provider=..., api_key=...), or supply a custom fn."
        )

    def _score(self, system: str, user: str) -> float:
        text = self._complete(system, user)
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0) if m else text)
            return max(0.0, min(1.0, float(obj.get("score", 0.0))))
        except Exception:
            return 0.0

    # ── individual judges (each returns 0..1) ──────────────────
    def context_relevance(self, query: str, contexts: list[str]) -> float:
        if not contexts:
            return 0.0
        return self._score(
            "You grade retrieval quality. Given a query and the retrieved contexts, "
            "rate 0..1 how relevant and sufficient the contexts are for answering the "
            'query. Respond JSON {"score": <0..1>}.',
            json.dumps({"query": query, "contexts": contexts})[:8000],
        )

    def faithfulness(self, answer: str, contexts: list[str]) -> float:
        return self._score(
            "You grade groundedness. Rate 0..1 how fully the answer is supported by the "
            "contexts: 1 = every claim is grounded in the contexts, 0 = unsupported or "
            'hallucinated. Respond JSON {"score": <0..1>}.',
            json.dumps({"answer": answer, "contexts": contexts})[:8000],
        )

    def answer_relevance(self, query: str, answer: str) -> float:
        return self._score(
            "Rate 0..1 how directly and completely the answer addresses the query. "
            'Respond JSON {"score": <0..1>}.',
            json.dumps({"query": query, "answer": answer})[:8000],
        )

    def answer_correctness(self, answer: str, expected: str) -> float:
        return self._score(
            "Rate 0..1 how correct the answer is compared to the reference answer "
            "(semantic correctness, not wording). 1 = fully correct, 0 = wrong. "
            'Respond JSON {"score": <0..1>}.',
            json.dumps({"answer": answer, "reference": expected})[:8000],
        )
