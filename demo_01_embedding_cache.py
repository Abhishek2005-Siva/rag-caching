"""Step 1 -- add an embedding cache and measure the speedup.

Run:  python3 demo_01_embedding_cache.py

We fire a stream of queries where some repeat (very common in real traffic --
FAQs, retries, popular questions). Without the cache, every query pays the full
embedding cost. With it, repeats are nearly free.
"""
import time

from rag.corpus import DOCUMENTS
from rag.embedder import HashingEmbedder
from rag.cache import CachedEmbedder
from rag.hybrid import HybridRetriever

# A realistic-ish stream: 5 distinct questions, but several are repeated.
QUERY_STREAM = [
    "how do I reset my password",
    "how do I export a note to pdf",
    "how do I reset my password",      # repeat
    "how much storage do I get",
    "how do I export a note to pdf",   # repeat
    "how do I reset my password",      # repeat
    "how do I share a note with someone",
    "how much storage do I get",       # repeat
]


def run(retriever, label):
    start = time.perf_counter()
    for q in QUERY_STREAM:
        retriever.retrieve(q, top_k=3)
    elapsed = time.perf_counter() - start
    print(f"{label:24} {elapsed*1000:7.1f} ms  for {len(QUERY_STREAM)} queries")
    return elapsed


def main():
    # --- Without the embedding cache ---
    plain = HashingEmbedder()
    r1 = HybridRetriever(DOCUMENTS, plain)
    plain.calls = 0  # reset after indexing so we count only query-time work
    t_plain = run(r1, "No cache")
    print(f"{'':24} -> {plain.calls} embedding model calls\n")

    # --- With the embedding cache ---
    cached = CachedEmbedder(HashingEmbedder())
    r2 = HybridRetriever(DOCUMENTS, cached)
    cached.inner.calls = 0
    cached.hits = cached.misses = 0
    # Note: the docs were embedded during indexing; those got cached too, but we
    # reset counters so we measure query-time behavior.
    t_cached = run(r2, "With embedding cache")
    print(f"{'':24} -> {cached.calls} embedding model calls")
    print(f"{'':24}    {cached.stats()}")

    print(f"\nSpeedup on this stream: {t_plain / t_cached:.1f}x")
    print("The 3 repeated queries never touched the embedding model the 2nd+ time.")


if __name__ == "__main__":
    main()
