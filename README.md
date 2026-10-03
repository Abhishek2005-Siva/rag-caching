# RAG Caching

A hands-on demo of how caching makes retrieval-augmented generation (RAG) faster and cheaper. Upload a PDF, ask questions, and watch the timings change as the caches warm up.

The retrieval and caching code is plain Python with no vector database and no ML framework, so every idea is readable in one sitting.

## What it demonstrates

| Technique | File | What it does |
|---|---|---|
| **Hybrid retrieval** | `rag/hybrid.py`, `rag/sparse.py` | Dense (meaning) plus sparse BM25 (keywords) search, merged with Reciprocal Rank Fusion so neither scoring scale dominates |
| **Embedding cache** | `rag/cache.py` (`CachedEmbedder`) | Exact-match cache keyed by a hash of the text. Repeated text never calls the embedding model twice |
| **Semantic cache** | `rag/cache.py` (`SemanticCache`) | Reuses an earlier answer when a new question is close enough in meaning. The similarity threshold is the whole product decision, so it is a slider. **Strict matching** (on by default in the app) adds a guard: a cached answer is reused only if your question adds no topic words the earlier one lacked, so "what are my experience and projects?" is never served the answer to "what are my projects?" |
| **GraphRAG-lite** | `rag/graph_rag.py` | The LLM extracts (subject, relation, object) facts from each chunk. At query time the app walks one hop from the entities in your question to recover facts split across chunks |

## The app

`app.py` is a Streamlit app. In the sidebar you:

1. Pick a **provider**, either NVIDIA (free key) or OpenAI, and paste **one API key**. That single key is used for both embeddings and chat.
2. Upload a PDF and set chunk size, overlap and the semantic-cache threshold.
3. Click **Test key** (optional) to check that embeddings and chat both work, then **Ingest**.
4. Ask questions. Each answer shows its time, whether it came from a cache, and the retrieved chunks.

For NVIDIA the model dropdowns are read live from NVIDIA's model list, because its hosted lineup changes often.
Some catalog models aren't available to every key and answer `404`. If **Test key** fails that way, click **Find working models**: it tries candidates with your key and selects ones that actually respond. The error message includes the provider's own explanation.

**Find working models** tests candidates with your key on a miniature fact-extraction task and picks the fastest one that returns valid JSON. Reasoning models are avoided when a plain one works, because they can take a minute per answer. Transient `503`/`429` errors are retried with backoff, and the knowledge graph is built four chunks at a time and stops early, with an explanation, if the first few all fail.

**Your key stays in your browser session.** It is masked as you type and never stored.

> NVIDIA retires hosted models without notice (`410 Gone`) and not every catalog model is open to every key (`404`). Use **Find working models**, or pick another from the dropdown.

## Run it

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

streamlit run app.py
```

Try it with the included sample, `aethlon_8ka.pdf` (a public SEC 8-K/A filing).

## Offline demos (no API key)

Three scripts build the ideas up one step at a time, using a local hashing embedder, so they run with no network:

```bash
python demo_00_hybrid.py           # dense vs sparse retrieval and how RRF fuses them
python demo_01_embedding_cache.py  # embedding cache: speedup on a stream with repeated queries (about 2x in the sample)
python demo_02_semantic_cache.py   # semantic cache hits, and what a too-loose threshold costs
```

## Deploy on Streamlit Community Cloud

At [share.streamlit.io](https://share.streamlit.io) pick this repository, the default branch and `app.py` as the main file. No secrets are needed because visitors supply their own keys.

## Layout

```
app.py                      Streamlit UI
demo_00_hybrid.py ...       step-by-step offline demos
rag/
  hybrid.py  sparse.py      dense + BM25 retrieval, RRF fusion
  cache.py                  embedding cache and semantic cache
  graph_rag.py              knowledge-graph retrieval
  embedder.py               local hashing embedder and cosine similarity
  llm_client.py             OpenAI-compatible embedding and chat clients, multi-key rotation
  providers.py              provider presets (the app offers NVIDIA and OpenAI)
  pdf_utils.py              PDF text extraction and chunking
```
