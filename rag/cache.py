"""Caching layers for the RAG. We add these one technique at a time.

TECHNIQUE 1 -- EMBEDDING CACHE (this file, for now)
---------------------------------------------------
Embedding the SAME text always produces the SAME vector, so recomputing it is
pure waste. An embedding cache is a dict keyed by a hash of the text: on a hit
we skip the (slow, paid) model call entirely.

This is an EXACT-match cache -- the text must be byte-for-byte identical. It
wraps any object with an `.embed(text)` method, so `HybridRetriever` doesn't
even know it's there. That "wrap, don't rewrite" shape is how we'll add every
later cache too.
"""
import hashlib

from .embedder import cosine


class CachedEmbedder:
    def __init__(self, inner):
        self.inner = inner            # the real embedder we're wrapping
        self.store: dict[str, list[float]] = {}
        self.hits = 0
        self.misses = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def embed(self, text: str) -> list[float]:
        key = self._key(text)
        cached = self.store.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        vec = self.inner.embed(text)
        self.store[key] = vec
        return vec

    @property
    def calls(self) -> int:
        # Pass through the real number of model calls, for reporting.
        return self.inner.calls

    def stats(self) -> str:
        total = self.hits + self.misses
        rate = (self.hits / total * 100) if total else 0
        return (f"embedding cache: {self.hits} hits / {self.misses} misses "
                f"({rate:.0f}% hit rate), {len(self.store)} vectors stored")


class SemanticCache:
    """TECHNIQUE 3 -- SEMANTIC (approximate) CACHE.

    The embedding cache above only fires on byte-identical text. Real users
    ask the same question in different words -- "how do I reset my password"
    vs "how can I change my password" -- and an exact-match cache treats those
    as two unrelated keys, even though the correct cached answer is the same.

    A semantic cache fixes that by keying on MEANING instead of text: embed
    the incoming query, compare it against every previously-cached query's
    embedding with cosine similarity, and if the best match clears a
    `threshold`, reuse its cached value instead of recomputing anything
    downstream (retrieval + generation, not just the embedding).

    This is a real tradeoff, not a free upgrade over exact-match:
      - threshold too LOW  -> false positives: a genuinely different question
        ("how much storage do I get" vs "how do I increase my storage") gets
        served someone else's cached answer. Wrong and hard to notice.
      - threshold too HIGH -> false negatives: it degrades into the exact-match
        cache, rarely firing on real paraphrases, and you get little benefit.
      - There is no universally correct number. Tune it against a labeled set
        of "should match" / "should not match" query pairs from real traffic.

    Storage here is a plain list scanned linearly (O(n) per lookup) -- fine
    for a demo with dozens of entries. A production semantic cache uses an
    approximate-nearest-neighbor index (FAISS, HNSW, a vector DB) so lookup
    stays fast as the cache grows to thousands/millions of entries.
    """

    def __init__(self, threshold: float = 0.90):
        self.threshold = threshold
        self.entries: list[dict] = []  # [{"vector", "text", "value"}]
        self.exact_hits = 0     # best match's text is identical to the query
        self.semantic_hits = 0  # best match clears the threshold but text differs
        self.misses = 0

    def lookup(self, query_vec: list[float], query_text: str | None = None):
        """Return (value, similarity, matched_text) on a hit, else (None, best_sim, None)."""
        best_entry, best_sim = None, -1.0
        for entry in self.entries:
            sim = cosine(query_vec, entry["vector"])
            if sim > best_sim:
                best_entry, best_sim = entry, sim

        if best_entry is not None and best_sim >= self.threshold:
            if query_text is not None and query_text == best_entry["text"]:
                self.exact_hits += 1
            else:
                self.semantic_hits += 1
            return best_entry["value"], best_sim, best_entry["text"]

        self.misses += 1
        return None, max(best_sim, 0.0), None

    def put(self, query_vec: list[float], query_text: str, value):
        self.entries.append({"vector": query_vec, "text": query_text, "value": value})

    def stats(self) -> str:
        total = self.exact_hits + self.semantic_hits + self.misses
        rate = ((self.exact_hits + self.semantic_hits) / total * 100) if total else 0
        return (f"semantic cache: {self.exact_hits} exact hits, {self.semantic_hits} "
                f"semantic hits, {self.misses} misses ({rate:.0f}% hit rate), "
                f"{len(self.entries)} entries, threshold={self.threshold}")
