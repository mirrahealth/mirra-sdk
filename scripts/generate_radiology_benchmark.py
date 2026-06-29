"""Generate a synthetic radiology retrieval benchmark with OpenAI.

Produces a corpus of fictional radiology reports + queries whose ground-truth
relevant report is correct *by construction* (each query is written from a
specific report). Output: corpus.jsonl, queries.jsonl, manifest.json.

  OPENAI_API_KEY=... python scripts/generate_radiology_benchmark.py \
      --reports-per-study 10 --queries 30 --out benchmarks/radiology-v1

Synthetic and NOT clinically validated — for benchmarking retrieval, not care.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone

from openai import OpenAI

MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

STUDIES = [
    ("CXR", "chest radiograph (CXR)", "Chest"),
    ("CT_CHEST", "CT of the chest", "Chest"),
    ("MRI_BRAIN", "MRI of the brain", "Neuro"),
    ("US_ABDOMEN", "abdominal ultrasound", "Abdomen"),
    ("XR_EXTREMITY", "extremity radiograph", "Musculoskeletal"),
    ("MRI_SPINE", "MRI of the lumbar spine", "Spine"),
]

REPORT_SYS = (
    "You generate SYNTHETIC, fictional radiology reports for software testing. "
    "Never use real patient data. Reports must be realistic in structure and "
    "terminology, vary between normal and abnormal findings, and be concise. "
    'Return strict JSON: {"reports": [{"technique": str, "findings": str, "impression": str}, ...]}.'
)

QUERY_SYS = (
    "You write retrieval queries for a radiology report search benchmark. For "
    "each report given (with its id), write ONE clinical question that this "
    "specific report answers — specific enough to distinguish it from similar "
    "studies — plus a one-sentence expected answer drawn from its impression. "
    'Return strict JSON: {"queries": [{"id": str, "query": str, "expected_answer": str}, ...]}.'
)


def _json(client: OpenAI, system: str, user: str, max_tokens: int = 1800) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.7,
        max_tokens=max_tokens,
    )
    return json.loads(resp.choices[0].message.content or "{}")


def gen_reports(client: OpenAI, study_label: str, n: int) -> list[dict]:
    out = _json(client, REPORT_SYS, f"Generate {n} distinct synthetic {study_label} reports.")
    return out.get("reports", [])[:n]


def gen_queries(client: OpenAI, reports: list[dict]) -> list[dict]:
    payload = [{"id": r["id"], "impression": r.get("impression", ""), "report": r["text"][:400]} for r in reports]
    out = _json(client, QUERY_SYS, "Reports:\n" + json.dumps(payload))
    return out.get("queries", [])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-per-study", type=int, default=10)
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--out", default="benchmarks/radiology-v1")
    ap.add_argument("--batch", type=int, default=5, help="queries per LLM call")
    args = ap.parse_args()

    client = OpenAI()
    os.makedirs(args.out, exist_ok=True)

    # ── corpus ──
    corpus: list[dict] = []
    n = 0
    for code, label, region in STUDIES:
        reports = gen_reports(client, label, args.reports_per_study)
        for r in reports:
            n += 1
            rid = f"r{n:04d}"
            text = f"TECHNIQUE: {r.get('technique','')}\nFINDINGS: {r.get('findings','')}\nIMPRESSION: {r.get('impression','')}".strip()
            corpus.append({
                "id": rid, "text": text, "study_type": code,
                "modality": label, "region": region, "impression": r.get("impression", ""),
            })
        print(f"  {code:<14} {len(reports)} reports  (corpus={n})", flush=True)

    # ── queries (labels correct by construction) ──
    import random
    rng = random.Random(7)
    chosen = rng.sample(corpus, min(args.queries, len(corpus)))
    queries: list[dict] = []
    for i in range(0, len(chosen), args.batch):
        batch = chosen[i : i + args.batch]
        for q in gen_queries(client, batch):
            rid = q.get("id")
            if any(c["id"] == rid for c in batch):
                queries.append({
                    "query": q["query"],
                    "relevant_ids": [rid],
                    "expected_answer": q.get("expected_answer", ""),
                })
        print(f"  queries {min(i + args.batch, len(chosen))}/{len(chosen)}", flush=True)

    # ── write ──
    with open(os.path.join(args.out, "corpus.jsonl"), "w") as f:
        for c in corpus:
            f.write(json.dumps(c) + "\n")
    with open(os.path.join(args.out, "queries.jsonl"), "w") as f:
        for q in queries:
            f.write(json.dumps(q) + "\n")
    manifest = {
        "name": "radiology", "version": os.path.basename(args.out).replace("radiology-", ""),
        "corpus": len(corpus), "queries": len(queries), "model": MODEL,
        "studies": [s[0] for s in STUDIES],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "synthetic": True, "clinically_validated": False,
    }
    with open(os.path.join(args.out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ {len(corpus)} reports · {len(queries)} queries → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
