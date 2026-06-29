"""Generate a synthetic radiology retrieval benchmark with OpenAI.

Produces a corpus of fictional radiology reports + queries whose ground-truth
relevant report is correct *by construction* (each query is written from a
specific report). Chunked + parallelized so it scales to hundreds of reports.

  OPENAI_API_KEY=... python scripts/generate_radiology_benchmark.py \
      --reports-per-study 100 --queries 200 --out benchmarks/radiology-v1

Output: corpus.jsonl, queries.jsonl, manifest.json.
Synthetic and NOT clinically validated — for benchmarking retrieval, not care.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    "terminology. Vary widely across the batch: mix normal and abnormal, vary "
    "anatomy, pathology, severity, and patient context. Keep each concise. "
    'Return strict JSON: {"reports": [{"technique": str, "findings": str, "impression": str}, ...]}.'
)

QUERY_SYS = (
    "You write retrieval queries for a radiology report search benchmark. For "
    "each report given (with its id), write ONE clinical question that this "
    "specific report answers — specific enough to distinguish it from similar "
    "studies — plus a one-sentence expected answer drawn from its impression. "
    'Return strict JSON: {"queries": [{"id": str, "query": str, "expected_answer": str}, ...]}.'
)


def _json(client: OpenAI, system: str, user: str, max_tokens: int = 2400) -> dict:
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.8,
        max_tokens=max_tokens,
    )
    try:
        return json.loads(resp.choices[0].message.content or "{}")
    except Exception:
        return {}


def _report_chunk(client: OpenAI, label: str, m: int) -> list[dict]:
    out = _json(client, REPORT_SYS, f"Generate {m} distinct synthetic {label} reports.")
    return [r for r in out.get("reports", []) if r.get("findings") and r.get("impression")]


def _query_chunk(client: OpenAI, reports: list[dict]) -> list[dict]:
    payload = [{"id": r["id"], "impression": r["impression"], "report": r["text"][:400]} for r in reports]
    out = _json(client, QUERY_SYS, "Reports:\n" + json.dumps(payload))
    ids = {r["id"] for r in reports}
    return [q for q in out.get("queries", []) if q.get("id") in ids and q.get("query")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-per-study", type=int, default=100)
    ap.add_argument("--queries", type=int, default=200)
    ap.add_argument("--out", default="benchmarks/radiology-v1")
    ap.add_argument("--report-batch", type=int, default=6)
    ap.add_argument("--query-batch", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    client = OpenAI()
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(7)

    # ── corpus: parallelize all report chunks across all studies ──
    tasks = []  # (code, label, region, size)
    for code, label, region in STUDIES:
        rem = args.reports_per_study
        while rem > 0:
            tasks.append((code, label, region, min(args.report_batch, rem)))
            rem -= args.report_batch

    corpus: list[dict] = []
    seen: set[str] = set()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_report_chunk, client, t[1], t[3]): t for t in tasks}
        for fut in as_completed(futs):
            code, label, region, _ = futs[fut]
            for r in fut.result():
                text = f"TECHNIQUE: {r.get('technique','')}\nFINDINGS: {r['findings']}\nIMPRESSION: {r['impression']}".strip()
                key = text.lower()[:160]
                if key in seen:
                    continue  # dedup near-identical reports
                seen.add(key)
                corpus.append({"text": text, "study_type": code, "modality": label,
                               "region": region, "impression": r["impression"]})
            done += 1
            print(f"  reports {done}/{len(tasks)} chunks · corpus={len(corpus)}", end="\r", flush=True)
    rng.shuffle(corpus)
    for i, c in enumerate(corpus, 1):
        c["id"] = f"r{i:05d}"
    print(f"\n  corpus: {len(corpus)} unique reports")

    # ── queries: derive from sampled reports (labels correct by construction) ──
    chosen = rng.sample(corpus, min(args.queries, len(corpus)))
    qbatches = [chosen[i:i + args.query_batch] for i in range(0, len(chosen), args.query_batch)]
    queries: list[dict] = []
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(_query_chunk, client, b) for b in qbatches]
        for fut in as_completed(futs):
            for q in fut.result():
                queries.append({"query": q["query"], "relevant_ids": [q["id"]],
                                "expected_answer": q.get("expected_answer", "")})
            done += 1
            print(f"  queries {done}/{len(qbatches)} chunks · queries={len(queries)}", end="\r", flush=True)
    print(f"\n  queries: {len(queries)}")

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
