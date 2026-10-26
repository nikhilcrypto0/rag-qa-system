"""
PROXY experiment: what would the hybrid retriever score if BM25 were actually running?

WHY THIS EXISTS
---------------
The production retriever is designed as hybrid (BM25 keyword + dense semantic), but BM25
runs on Elasticsearch, and when ES is unavailable the exception is swallowed and the system
silently degrades to semantic-only. The measured baseline (77% Recall@3) is therefore a
SEMANTIC-ONLY number.

Its failures were concentrated on exact proper nouns ("Higher Learning Commission",
"Handshake", "Intensive English Program") — exactly what keyword search is good at.
This script tests that hypothesis.

HONESTY BOUNDARY
----------------
This substitutes an in-process BM25 (rank_bm25) for Elasticsearch's BM25 over the SAME
1,164 chunks, then runs the SAME dedup + cross-encoder rerank as production. Both implement
Okapi BM25, so this is a faithful proxy for the architecture — but it is NOT the production
Elasticsearch path.

=> Quote 77% as the system's accuracy. Quote THIS number only as "what turning BM25 on
   should recover", clearly labelled as a proxy experiment.
"""
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rank_bm25 import BM25Okapi  # noqa: E402

from app.config import get_settings  # noqa: E402
from app.rag.reranker import rerank  # noqa: E402
from app.rag.vector_store import get_vector_store  # noqa: E402
from eval.run_eval import normalize  # noqa: E402


def tokenize(text: str) -> list:
    return re.findall(r"[a-z0-9]+", text.lower())


def main() -> None:
    settings = get_settings()
    cases = json.loads((Path(__file__).parent / "ground_truth.json").read_text())

    store = get_vector_store()
    all_docs = list(store.docstore._dict.values())
    print(f"Corpus: {len(all_docs)} chunks (identical to production index)")

    bm25 = BM25Okapi([tokenize(d.page_content) for d in all_docs])

    top_k, top_n = settings.top_k_retrieval, settings.top_k_rerank
    dense = store.as_retriever(search_type="similarity", search_kwargs={"k": top_k})

    hits1, hits3, rr, lat, misses = 0, 0, [], [], []

    print(f"\n{'#':>3}  {'@1':^3} {'@3':^3}  {'ms':>6}  question")
    print("-" * 78)

    for i, case in enumerate(cases, 1):
        q, expected = case["question"], normalize(case["expected_source"])

        t0 = time.perf_counter()
        # 1. BM25 sparse (in-process stand-in for Elasticsearch)
        scores = bm25.get_scores(tokenize(q))
        sparse = [all_docs[j] for j in sorted(range(len(scores)), key=lambda x: -scores[x])[:top_k]]
        # 2. Dense semantic
        dense_docs = dense.invoke(q)
        # 3. Merge + dedup (same shallow content hash as production)
        seen, cands = set(), []
        for d in sparse + dense_docs:
            k = hash(d.page_content[:200])
            if k not in seen:
                seen.add(k)
                cands.append(d)
        # 4. Same cross-encoder rerank as production
        ranked = [d for d, _ in rerank(q, cands, top_n=top_n)]
        ms = (time.perf_counter() - t0) * 1000
        lat.append(ms)

        srcs = [normalize(d.metadata.get("source", "")) for d in ranked]
        rank = next((j for j, s in enumerate(srcs, 1) if s == expected), None)
        hits1 += rank == 1
        hits3 += rank is not None and rank <= 3
        rr.append(1.0 / rank if rank else 0.0)
        if not (rank and rank <= 3):
            misses.append({"question": q, "expected": expected, "got": srcs})

        print(f"{i:>3}  {'✓' if rank == 1 else '·':^3} {'✓' if rank and rank <= 3 else '✗':^3}  {ms:>6.0f}  {q[:52]}")

    n = len(cases)
    print("-" * 78)
    print(f"\nPROXY HYBRID RESULTS (n={n})   [in-process BM25 + dense + rerank]")
    print(f"{'  Recall@1':<28} {hits1}/{n}  = {hits1/n:.0%}")
    print(f"{'  Recall@3':<28} {hits3}/{n}  = {hits3/n:.0%}")
    print(f"{'  MRR':<28} {statistics.mean(rr):.3f}")
    print(f"{'  Median latency':<28} {statistics.median(lat):.0f} ms")

    if misses:
        print(f"\nSTILL MISSING ({len(misses)}):")
        for m in misses:
            print(f"  Q: {m['question']}\n     wanted: {m['expected']}\n     got:    {m['got']}")

    (Path(__file__).parent / "results_hybrid_proxy.json").write_text(json.dumps({
        "n": n,
        "mode": "PROXY hybrid (in-process BM25, not Elasticsearch)",
        "recall_at_1": hits1 / n,
        "recall_at_3": hits3 / n,
        "mrr": statistics.mean(rr),
        "median_latency_ms": statistics.median(lat),
        "misses": misses,
    }, indent=2))


if __name__ == "__main__":
    main()
