# Mirra — retrieval & RAG agent evaluation

Evaluate your **retrieval / RAG agent** and get scores back **inline**. You run
your agent in your own environment; Mirra wraps the run and returns retrieval
metrics + RAG quality per query — no endpoint to expose, no data leaves your
infra (unless you opt into dashboard sync).

```python
from mirra import evaluate, Dataset

def my_agent(query: str) -> dict:
    docs = retriever.search(query)
    return {
        "answer":      llm.answer(query, docs),     # for RAG (optional)
        "contexts":    [d.text for d in docs],       # retrieved text
        "context_ids": [d.id for d in docs],         # retrieved ids (for label metrics)
    }

report = evaluate(my_agent, Dataset.from_jsonl("eval.jsonl"), k=5)
print(report)                      # aggregate scores
report.cases[0].metrics            # per-query scores, returned inline
report.to_json("report.json")
```

## Install

```bash
pip install mirra-eval               # core (label-based retrieval metrics)
pip install "mirra-eval[openai]"     # + LLM judge via OpenAI
pip install "mirra-eval[anthropic]"  # + LLM judge via Anthropic
```

The distribution is `mirra-eval`; the import name is `mirra` (`from mirra import evaluate`).

## Your agent

`agent_fn(query: str)` returns the retrieval result. Accepted shapes: a dict (or
object) with any of —

| field | type | used for |
|---|---|---|
| `contexts` | `list[str]` (or `[{text, id}]`) | faithfulness, judge-based retrieval relevance |
| `context_ids` | `list[str]` | label-based retrieval metrics (precision/recall/MRR/nDCG) |
| `answer` | `str` | RAG quality (faithfulness, answer correctness/relevance) |

A pure retriever returns just `contexts`/`context_ids`; a RAG agent adds `answer`.

## Dataset

A list of cases, or JSONL. Labels are **optional** — that's the hybrid model:

```jsonl
{"query": "first line treatment for type 2 diabetes", "relevant_ids": ["d1"], "expected_answer": "Metformin."}
{"query": "what diagnoses diabetes", "relevant_ids": ["d3"]}
{"query": "a live production query with no labels"}
```

- `relevant_ids` present → exact retrieval metrics.
- `expected_answer` present → answer correctness vs reference.
- neither → the **LLM judge** scores relevance / faithfulness / answer quality.

## What's measured

**Retrieval (label-based, exact)** — `precision@k`, `recall@k`, `hit@k`, `mrr`, `ndcg@k`.

**RAG quality (LLM judge, reference-free)** —
- `context_relevance` — are retrieved contexts relevant & sufficient?
- `faithfulness` — is the answer grounded in the contexts (vs hallucinated)?
- `answer_correctness` (vs `expected_answer`) or `answer_relevance` (reference-free).

**Mirra Retrieval Score** — weighted across the pillars that apply:
`retrieval 0.4 · faithfulness 0.3 · answer 0.3` (renormalized over present pillars).

## Test on a Mirra benchmark (synthetic radiology)

Don't have a labeled eval set? Use a **Mirra-hosted synthetic benchmark**. It's
a corpus of fictional radiology reports + labeled queries — 100% synthetic, no
PHI. Downloaded and cached on first use.

```python
from mirra import evaluate
from mirra.benchmarks import load_radiology

corpus, dataset = load_radiology()     # ~/.cache/mirra (downloads once)

my_retriever.index(corpus)             # index Mirra's corpus into YOUR retriever
report = evaluate(my_agent, dataset, k=5)
print(report)
```

- `corpus` — `list[{id, text, modality, region, ...}]`; load these into your retriever.
- `dataset` — labeled queries (each query's relevant report id is correct by construction).
- Override the host with `MIRRA_BENCHMARK_URL` (e.g. a GCS bucket).

> Synthetic and **not clinically validated** — for benchmarking retrieval
> quality, not clinical use.

## The LLM judge

Auto-detects a provider from the environment (`OPENAI_API_KEY`, then
`ANTHROPIC_API_KEY`). Override or plug in your own:

```python
from mirra import Mirra, Judge

m = Mirra(judge=Judge(provider="anthropic", model="claude-haiku-4-5"))
m = Mirra(judge=Judge(fn=my_callable))   # fully custom / offline judge
```

Without a judge configured, Mirra still runs and reports the exact retrieval
metrics; the judge-only pillars are simply omitted.

## Score one case inline

```python
from mirra import Mirra
m = Mirra(k=5)
result = m.score(
    "can ibuprofen affect blood pressure meds?",
    answer="Yes — ibuprofen can raise BP and blunt ACE inhibitors.",
    contexts=["Ibuprofen can raise blood pressure and blunt ACE inhibitors."],
    context_ids=["d4"], relevant_ids=["d4"],
)
result.metrics      # {'precision@k':…, 'ndcg@k':…, 'faithfulness':…, …}
result.overall      # the case's Mirra Retrieval Score
```

## Optional: sync to the Mirra dashboard

```python
m = Mirra(api_key="mk_…")   # posts the report to your Mirra workspace, returns a URL
```

Omit `api_key` to run fully local — nothing is sent anywhere.

---

© Mirra Health · the RL environment for healthcare AI agents
