"""Step 0 -- the hybrid RAG foundation, no caching yet.

Run:  python3 demo_00_hybrid.py

Shows how dense (meaning) and sparse (keywords) retrieval disagree, and how RRF
fuses them. Notice the query below uses "sign in" while the relevant doc says
"log in" / "password" -- a case where keyword search alone struggles.
"""
from rag.corpus import DOCUMENTS, BY_ID
from rag.embedder import HashingEmbedder
from rag.hybrid import HybridRetriever


def show(title, ranked):
    print(f"\n  {title}")
    for doc_id, score in ranked:
        snippet = BY_ID[doc_id]["text"][:60]
        print(f"    {score:6.3f}  {doc_id:12} {snippet}...")


def main():
    embedder = HashingEmbedder()
    retriever = HybridRetriever(DOCUMENTS, embedder)

    query = "I can't sign in, forgot my password"
    print(f"Query: {query!r}")

    results = retriever.retrieve(query, top_k=3)
    show("DENSE only (meaning)", results["dense"][:3])
    show("SPARSE only (keywords / BM25)", results["sparse"][:3])
    show("HYBRID (RRF fusion)", results["hybrid"])

    print(f"\nEmbedding model calls this run: {embedder.calls}")
    print("(8 documents embedded at startup + 1 query = 9)")


if __name__ == "__main__":
    main()
