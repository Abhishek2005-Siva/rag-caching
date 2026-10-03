"""Sparse retrieval with BM25 -- keyword matching, implemented from scratch.

BM25 scores a document for a query by, roughly:
  - rewarding documents that contain the query's words (term frequency),
  - down-weighting words that appear in many documents (inverse doc frequency),
  - and normalizing for document length so long docs don't win by default.

It has no idea that "log in" and "sign in" mean the same thing -- that is
exactly the gap the dense retriever fills. Hybrid = use both.
"""
import math
from collections import Counter

from .tokenize import tokenize


class BM25:
    def __init__(self, documents: list[dict], k1: float = 1.5, b: float = 0.75):
        # k1 controls how fast term-frequency saturates; b controls how much
        # document length is penalized. 1.5 / 0.75 are the standard defaults.
        self.k1 = k1
        self.b = b
        self.doc_ids = [d["id"] for d in documents]
        doc_tokens = [tokenize(d["text"]) for d in documents]
        self.doc_len = [len(t) for t in doc_tokens]
        self.N = len(documents)
        self.avgdl = sum(self.doc_len) / self.N
        self.tf = [Counter(toks) for toks in doc_tokens]

        # Document frequency: in how many docs does each term appear?
        df = Counter()
        for toks in doc_tokens:
            for term in set(toks):
                df[term] += 1
        self.idf = {
            term: math.log(1 + (self.N - dft + 0.5) / (dft + 0.5))
            for term, dft in df.items()
        }

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        q_terms = tokenize(query)
        scored = []
        for i in range(self.N):
            score = 0.0
            for term in q_terms:
                f = self.tf[i].get(term, 0)
                if not f:
                    continue
                denom = f + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                score += self.idf.get(term, 0.0) * (f * (self.k1 + 1)) / denom
            scored.append((self.doc_ids[i], score))
        scored.sort(key=lambda x: -x[1])
        return [(doc_id, s) for doc_id, s in scored[:top_k] if s > 0]
