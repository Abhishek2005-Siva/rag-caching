"""A local, dependency-free embedder so the whole project runs with no API key.

WHY A FAKE EMBEDDER?
--------------------
A real RAG turns text into a vector using a model (OpenAI's text-embedding-3,
a sentence-transformers model, etc.). That call costs money and/or ~10-100 ms.
For *learning caching*, what matters is that embedding is (a) deterministic for
the same input and (b) slow enough that caching produces a visible speedup.

`HashingEmbedder` hashes each word into one of `dim` buckets and counts them --
a "bag of hashed words" vector. Two texts that share words end up with similar
vectors, which is enough to demonstrate retrieval and, later, semantic caching.
The `simulated_latency` sleep stands in for a real model/API round-trip.

To go real later, swap this class for one that calls an embedding API -- every
other file only depends on the `.embed(text) -> list[float]` interface.
"""
import hashlib
import math
import time

from .tokenize import tokenize


class HashingEmbedder:
    def __init__(self, dim: int = 256, simulated_latency: float = 0.04):
        self.dim = dim
        self.simulated_latency = simulated_latency
        self.calls = 0  # how many times we actually computed an embedding

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        if self.simulated_latency:
            time.sleep(self.simulated_latency)  # pretend we called a slow model

        vec = [0.0] * self.dim
        for token in tokenize(text):
            bucket = int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim
            vec[bucket] += 1.0

        # L2-normalize so cosine similarity behaves well.
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # inputs are already normalized
