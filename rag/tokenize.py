"""Shared tokenizer used by both the sparse (BM25) and the local embedder.

Keeping one tokenizer everywhere means the lexical signal that BM25 sees is the
same signal the hashing embedder sees. Real systems use a model's own tokenizer
for embeddings, but for learning a shared one keeps things transparent.
"""
import re

_WORD = re.compile(r"[a-z0-9]+")

# A tiny stopword list. Dropping these keeps BM25 focused on content words.
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "to", "of", "in", "on", "for", "with", "as", "at", "by",
    "it", "this", "that", "i", "you", "my", "how", "do", "can", "what", "when",
}


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in STOPWORDS]
