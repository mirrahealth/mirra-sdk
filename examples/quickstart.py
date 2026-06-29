"""Quickstart — evaluate a tiny retrieval+answer agent with Mirra.

Runs fully offline (label-based retrieval metrics). If OPENAI_API_KEY or
ANTHROPIC_API_KEY is set, the LLM judge also scores faithfulness / answer
quality (the "hybrid" model).

    python examples/quickstart.py
"""

from mirra import Dataset, evaluate

# ── a toy corpus + keyword retriever standing in for the client's agent ──
CORPUS = {
    "d1": "Metformin is first-line therapy for type 2 diabetes.",
    "d2": "Lisinopril is an ACE inhibitor used to treat hypertension.",
    "d3": "HbA1c above 6.5% is diagnostic for diabetes.",
    "d4": "Ibuprofen can raise blood pressure and blunt ACE inhibitors.",
    "d5": "Atorvastatin lowers LDL cholesterol.",
}


def my_agent(query: str) -> dict:
    q = set(query.lower().split())
    ranked = sorted(
        CORPUS.items(),
        key=lambda kv: len(q & set(kv[1].lower().split())),
        reverse=True,
    )
    top = ranked[:3]
    return {
        "answer": " ".join(text for _, text in top[:1]),
        "contexts": [text for _, text in top],
        "context_ids": [cid for cid, _ in top],
    }


# ── a small labeled eval set (queries + ground-truth relevant ids) ──
dataset = Dataset.from_dicts([
    {"query": "first line treatment for type 2 diabetes",
     "relevant_ids": ["d1"], "expected_answer": "Metformin."},
    {"query": "can ibuprofen affect blood pressure medication",
     "relevant_ids": ["d4"], "expected_answer": "Yes, ibuprofen can raise BP and blunt ACE inhibitors."},
    {"query": "what diagnoses diabetes",
     "relevant_ids": ["d3"]},  # no expected_answer → judge scores answer relevance
])

report = evaluate(my_agent, dataset, k=3)
print()
print(report)
report.to_json("report.json")
print("\nFull per-case results written to report.json")
