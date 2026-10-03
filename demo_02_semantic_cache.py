"""Step 2 -- semantic cache: hit on MEANING, not just identical text.

Run:  python3 demo_02_semantic_cache.py

We ask a stream of questions where several are paraphrases of an earlier
question (different words, same intent). An exact-match cache (Technique 1/2)
misses all of these. A semantic cache can hit them -- IF the similarity
threshold is tuned well. We also show what happens when the threshold is too
loose: a genuinely different question gets served a wrong cached answer.
"""
import time

from rag.embedder import HashingEmbedder
from rag.cache import CachedEmbedder, SemanticCache

# "value" here stands in for a full RAG answer -- in demo_02 we don't run
# retrieval/generation, we just cache a canned string per *original* question
# so the cache behavior is easy to see in isolation.
CANNED_ANSWERS = {
    "how do I reset my password": "Click 'Reset password' on the sign-in screen; the link expires in 30 minutes.",
    "how much storage do I get": "Free accounts get 2 GB; Pro accounts get 100 GB.",
    "how do I export a note": "Open the note, click the three-dot menu, choose Export (Markdown or PDF).",
}

# Stream: original questions, then paraphrases, then one adversarial pair that
# LOOKS similar in wording but asks something different.
QUERY_STREAM = [
    ("how do I reset my password", "original"),
    ("how can I change my password", "paraphrase of Q1"),
    ("I forgot my password, what do I do", "paraphrase of Q1"),
    ("how much storage do I get", "original"),
    ("what's my storage limit", "paraphrase of Q4"),
    ("how do I export a note", "original"),
    ("how do I export my notes to pdf", "paraphrase of Q6"),
    ("how do I INCREASE my storage", "adversarial: shares words with Q4/Q5 but different intent"),
]


def run_with_cache(embedder, cache: SemanticCache):
    for query, note in QUERY_STREAM:
        t0 = time.perf_counter()
        vec = embedder.embed(query)  # embedding step is itself cached (Technique 1)
        value, sim, matched_text = cache.lookup(vec, query_text=query)

        if value is not None:
            kind = "EXACT " if matched_text == query else "SEMANTIC"
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [{kind} HIT sim={sim:.3f}] {elapsed:5.1f} ms  {query!r}")
            print(f"           -> matched cached question: {matched_text!r}")
            print(f"           -> served answer: {value!r}")
        else:
            # Cache miss: pretend to "compute" the answer (canned lookup here;
            # in the real app this is the full retrieval+generation pipeline).
            answer = CANNED_ANSWERS.get(query, "(no canned answer for this adversarial query)")
            cache.put(vec, query, answer)
            elapsed = (time.perf_counter() - t0) * 1000
            print(f"  [MISS best_sim={sim:.3f}]   {elapsed:5.1f} ms  {query!r}  ({note})")
            print(f"           -> computed and cached: {answer!r}")
        print()


def main():
    print("=" * 78)
    print("Threshold = 0.90 (strict) -- rarely fires on paraphrases with our crude")
    print("hashing embedder; mostly behaves like the exact-match cache.")
    print("=" * 78)
    embedder_strict = CachedEmbedder(HashingEmbedder(simulated_latency=0.03))
    cache_strict = SemanticCache(threshold=0.90)
    run_with_cache(embedder_strict, cache_strict)
    print(cache_strict.stats())

    print()
    print("=" * 78)
    print("Threshold = 0.40 (loose) -- fires on real paraphrases, BUT ALSO on the")
    print("adversarial query, which shares words with Q4 but asks something")
    print("different (increasing storage vs. checking your current limit). Watch")
    print("the last entry: it gets served a WRONG cached answer.")
    print("=" * 78)
    embedder_loose = CachedEmbedder(HashingEmbedder(simulated_latency=0.03))
    cache_loose = SemanticCache(threshold=0.40)
    run_with_cache(embedder_loose, cache_loose)
    print(cache_loose.stats())

    print()
    print("Lesson: the threshold is the whole product decision. Too high and you")
    print("rarely benefit from semantic matching; too low and you silently serve")
    print("wrong answers. Tune it on real query pairs, not by feel.")


if __name__ == "__main__":
    main()
