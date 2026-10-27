"""
Retrieval evaluation for the SEMO RAG assistant.

Measures whether the correct semo.edu source page lands in the retrieved passages.
This tests RETRIEVAL only — it makes no Anthropic API calls, so it costs nothing to run.

Metrics:
  Recall@1  — correct page is the single top-ranked passage
  Recall@3  — correct page appears in the 3 passages handed to Claude  <-- the number that matters
  MRR       — mean reciprocal rank of the correct page
  Latency   — wall-clock retrieval time per question

Recall@3 is the headline: those 3 passages are the ONLY context the model may use.
If the right page is not in there, the model cannot possibly answer correctly.

Usage:  .venv/bin/python eval/run_eval.py
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.rag.hybrid_retriever import build_hybrid_retriever  # noqa: E402

GROUND_TRUTH = Path(__file__).parent / "ground_truth.json"


def normalize(url: str) -> str:
    """semo.edu/admissions and semo.edu/admissions/index are the same page."""
    u = url.strip().lower().rstrip("/")
    for suffix in ("/index",):
        if u.endswith(suffix):
            u = u[: -len(suffix)]
    return u.replace("https://", "").replace("http://", "")


def bm25_is_live() -> bool:
    """The hybrid retriever silently degrades to semantic-only if Elasticsearch is down.
    We must know which system we actually measured."""
    try:
        from app.rag.bm25_retriever import bm25_search
        return len(bm25_search("admission requirements", top_k=3)) > 0
    except Exception:
        return False


def main() -> None:
    cases = json.loads(GROUND_TRUTH.read_text())
    retriever = build_hybrid_retriever()

    hybrid = bm25_is_live()
    mode = "HYBRID (BM25 + dense)" if hybrid else "SEMANTIC-ONLY (Elasticsearch unavailable)"
    print(f"\nRetrieval mode measured: {mode}")
    print(f"Ground-truth questions:  {len(cases)}\n")
    print(f"{'#':>3}  {'@1':^3} {'@3':^3}  {'ms':>6}  question")
    print("-" * 78)

    hits1, hits3, rr, latencies, misses = 0, 0, [], [], []

    for i, case in enumerate(cases, 1):
        q, expected = case["question"], normalize(case["expected_source"])
        # Some questions are answered equally well by another page (verified by reading it)
        accepted = {expected, *(normalize(u) for u in case.get("also_accept", []))}

        t0 = time.perf_counter()
        docs = retriever.invoke(q)
        ms = (time.perf_counter() - t0) * 1000
        latencies.append(ms)

        ranked = [normalize(d.metadata.get("source", "")) for d in docs]

        rank = next((j for j, s in enumerate(ranked, 1) if s in accepted), None)
        top1 = rank == 1
        top3 = rank is not None and rank <= 3

        hits1 += top1
        hits3 += top3
        rr.append(1.0 / rank if rank else 0.0)

        if not top3:
            misses.append({"question": q, "expected": expected, "got": ranked})

        print(f"{i:>3}  {'✓' if top1 else '·':^3} {'✓' if top3 else '✗':^3}  {ms:>6.0f}  {q[:52]}")

    n = len(cases)
    print("-" * 78)
    print(f"\n{'RESULTS':<28} (n={n}, {mode})")
    print(f"{'  Recall@1':<28} {hits1}/{n}  = {hits1/n:.0%}")
    print(f"{'  Recall@3  (headline)':<28} {hits3}/{n}  = {hits3/n:.0%}")
    print(f"{'  MRR':<28} {statistics.mean(rr):.3f}")
    print(f"{'  Median retrieval latency':<28} {statistics.median(latencies):.0f} ms")
    print(f"{'  p95 retrieval latency':<28} {sorted(latencies)[int(0.95*n)-1]:.0f} ms")

    if misses:
        print(f"\nMISSES ({len(misses)}) — correct page absent from top 3:")
        for m in misses:
            print(f"  Q: {m['question']}")
            print(f"     wanted: {m['expected']}")
            print(f"     got:    {m['got'][:3]}")

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps({
        "n": n,
        "retrieval_mode": mode,
        "bm25_active": hybrid,
        "recall_at_1": hits1 / n,
        "recall_at_3": hits3 / n,
        "mrr": statistics.mean(rr),
        "median_latency_ms": statistics.median(latencies),
        "p95_latency_ms": sorted(latencies)[int(0.95 * n) - 1],
        "misses": misses,
    }, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
