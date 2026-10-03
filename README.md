# RAG Caching

A hands-on demo of how caching makes retrieval-augmented generation (RAG) faster and cheaper. Upload a PDF, ask questions, and watch the timings change as the caches warm up.

The retrieval and caching code is plain Python with no vector database and no ML framework, so every idea is readable in one sitting.

## What it demonstrates

| Technique | File | What it does |
|---|---|---|
| **Hybrid retrieval** | `rag/hybrid.py`, `rag/sparse.py` | Dense (meaning) plus sparse BM25 (keywords) search, merged with Reciprocal Rank Fusion so neither scoring scale dominates |
| **Embedding cache** | `rag/cache.py` (`CachedEmbedder`) | Exact-match cache keyed by a hash of the text. Repeated text never calls the embedding model twice |
| **Semantic cache** | `rag/cache.py` (`SemanticCache`) | Reuses an earlier answer when a new question is close enough in meaning. The similarity threshold is the whole product decision, so it is a slider |
| **GraphRAG-lite** | `rag/graph_rag.py` | The LLM extracts (subject, relation, object) facts from each chunk. At query time the app walks one hop from the entities in your question to recover facts split across chunks |

## The app

`app.py` is a Streamlit app. In the sidebar you:

1. Pick an **embedding** provider and key.
2. Pick a **chat** provider and key. It can differ from the embedding one, for example embed with NVIDIA and answer with Groq.
3. Upload a PDF and set chunk size, overlap and the semantic-cache threshold.
4. Ingest, then ask questions. Each answer shows its time, whether it came from a cache, and the retrieved chunks.

Supported providers (anything speaking the OpenAI wire format works): NVIDIA NIM (free key), OpenAI, Together AI, Mistral, Groq (chat only), a local Ollama server, or any custom base URL. You can pool several free-tier keys for one provider and the app rotates to the next on rate limits.

**Your keys stay in your browser session.** The primary key fields are masked. The optional "pool more keys" box is not, because Streamlit cannot mask a multi-line field.

> NVIDIA retires hosted models without notice. If a key test returns `410 Gone`, choose a current model from [build.nvidia.com/models](https://build.nvidia.com/models).

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
  providers.py              provider presets
  pdf_utils.py              PDF text extraction and chunking
```
