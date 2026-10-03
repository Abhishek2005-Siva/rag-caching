"""Hybrid retrieval: dense (meaning) + sparse (keywords), fused with RRF.

DENSE finds documents whose *meaning* is close to the query, even with no shared
words ("sign in" vs "log in"). SPARSE finds documents with the exact keywords,
which is great for names, error codes, and rare terms. Each is strong where the
other is weak, so we run both and merge the two ranked lists.

RECIPROCAL RANK FUSION (RRF) merges by *rank position*, not by raw score. Dense
cosine scores (~0-1) and BM25 scores (unbounded) live on totally different
scales, so adding them directly is meaningless. RRF sidesteps that: a document
gets 1/(k + rank) from each list it appears in, and we sum those. A doc ranked
highly by both retrievers rises to the top.
"""
from .embedder import cosine


class DenseIndex:
    def __init__(self, documents: list[dict], embedder):
        self.embedder = embedder
        self.doc_ids = [d["id"] for d in documents]
        # Embed every document once, up front. In a real system these vectors
        # live in a vector database (FAISS, pgvector, Pinecone, ...).
        self.vectors = [embedder.embed(d["text"]) for d in documents]

    def search(self, query_vec: list[float], top_k: int = 5) -> list[tuple[str, float]]:
        scored = [
            (self.doc_ids[i], cosine(query_vec, self.vectors[i]))
            for i in range(len(self.doc_ids))
        ]
        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]


def rrf_fuse(dense_ranked, sparse_ranked, k: int = 60, top_k: int = 5):
    scores: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(dense_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    for rank, (doc_id, _) in enumerate(sparse_ranked):
        scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: -x[1])
    return fused[:top_k]


class HybridRetriever:
    def __init__(self, documents: list[dict], embedder):
        self.embedder = embedder
        self.dense = DenseIndex(documents, embedder)
        # BM25 is imported lazily so this module has no import-time cost if you
        # only want the dense half.
        from .sparse import BM25
        self.sparse = BM25(documents)

    def retrieve(self, query: str, top_k: int = 3):
        query_vec = self.embedder.embed(query)
        dense_ranked = self.dense.search(query_vec, top_k=5)
        sparse_ranked = self.sparse.search(query, top_k=5)
        fused = rrf_fuse(dense_ranked, sparse_ranked, top_k=top_k)
        return {
            "dense": dense_ranked,
            "sparse": sparse_ranked,
            "hybrid": fused,
        }
