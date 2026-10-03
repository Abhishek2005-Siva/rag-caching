"""Streamlit UI: upload a PDF, ask questions, see the embedding-cache speedup.

Works against any OpenAI-compatible API -- NVIDIA NIM (free), OpenAI,
Together AI, Mistral, Groq (chat only), a local Ollama server, or a Custom
base URL. Pick the embedding provider and the chat/generation provider
independently in the sidebar (they don't have to be the same one).

Run:  .venv/bin/streamlit run app.py
"""
import time

import streamlit as st

from rag.cache import CachedEmbedder, SemanticCache
from rag.graph_rag import KnowledgeGraph
from rag.hybrid import HybridRetriever, rrf_fuse
from rag.llm_client import (
    OpenAICompatibleEmbedder, chat_complete, MultiKeyEmbedder, multi_key_chat_complete,
)
from rag.pdf_utils import pdf_to_documents
from rag.providers import PROVIDERS, EMBEDDING_PROVIDERS, CHAT_PROVIDERS

st.set_page_config(page_title="RAG Caching Demo", page_icon="📄", layout="wide")

# ---------------------------------------------------------------------------
# Session state: everything that must survive across Streamlit reruns.
# ---------------------------------------------------------------------------
defaults = {
    "documents": None,        # list[{"id", "text"}] chunked from the PDF
    "cached_embedder": None,  # CachedEmbedder wrapping an OpenAICompatibleEmbedder
    "retriever": None,        # HybridRetriever built on the cached embedder
    "source_name": None,
    "history": [],            # list of past Q&A + timing dicts
    "semantic_cache": None,   # SemanticCache: query embedding -> {"answer", "chunks"}
    "knowledge_graph": None,  # KnowledgeGraph, built lazily via a separate button (costs 1 LLM call/chunk)
    "chat_config": None,      # {"base_url", "api_keys", "model"} snapshot used at ingest time
    "failed_providers": {"embed": set(), "chat": set()},  # provider names excluded (every key failed)
    "valid_keys": {},         # (role, provider_name) -> list[str] of keys that passed their test
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)


def build_index(base_url: str, api_keys: list[str], model: str, needs_input_type: bool, docs: list[dict]):
    """Embed every chunk once (through the cache) and build the retriever."""
    # NOTE: NVIDIA's embedqa-style models distinguish "query" vs "passage"
    # embeddings. We use a single embedder/input_type for both indexing and
    # querying below -- a simplification that keeps one cache warm for both,
    # at the cost of not using the asymmetric mode as intended.
    #
    # MultiKeyEmbedder rotates across api_keys on 401/403/429 -- with several
    # free-tier keys pooled, ingesting a big PDF is less likely to stall on
    # one key's rate limit.
    embedder = CachedEmbedder(MultiKeyEmbedder(
        base_url, api_keys, model, input_type="passage" if needs_input_type else None))
    progress = st.progress(0.0, text="Embedding chunks...")
    for i, d in enumerate(docs):
        embedder.embed(d["text"])
        progress.progress((i + 1) / len(docs), text=f"Embedding chunk {i + 1}/{len(docs)}")
    progress.empty()
    retriever = HybridRetriever(docs, embedder)
    return embedder, retriever


def test_key(role: str, base_url: str, api_key: str, model: str, needs_input_type: bool) -> tuple[bool, str]:
    """One cheap real call to confirm a SINGLE key actually works."""
    try:
        if role == "embed":
            OpenAICompatibleEmbedder(
                base_url, api_key, model, input_type="passage" if needs_input_type else None
            ).embed("connection test")
        else:
            chat_complete(base_url, api_key, system="Reply with OK.", user="OK?",
                           model=model, max_tokens=5)
        return True, "OK"
    except Exception as e:
        return False, str(e)


def _mask(key: str) -> str:
    return f"{key[:6]}...{key[-4:]}" if len(key) > 12 else key


def provider_picker(label_prefix: str, provider_names: list[str], role: str):
    """Render a provider dropdown + base URL / model / API key(s) inputs, plus
    a 'Test' button. Multiple keys (one per line) are supported for the SAME
    provider -- e.g. several free NVIDIA accounts -- and the app rotates
    across them at request time on rate limits (MultiKeyEmbedder /
    multi_key_chat_complete). Testing checks each key individually; a key
    that fails is dropped from what's actually used. If EVERY key for a
    provider fails, the provider itself is removed from the dropdown until
    you reset it.

    Returns (base_url, api_keys: list[str], model, needs_input_type) for
    whichever provider/settings the user currently has selected."""
    failed = st.session_state.failed_providers[role]
    available = [p for p in provider_names if p not in failed]
    if not available:
        st.error(f"All {label_prefix.lower()} providers are excluded (every key failed). "
                 f"Reset below to try again.")
        available = provider_names  # fall back so the UI doesn't dead-end

    provider_name = st.selectbox(f"{label_prefix} provider", available, key=f"{role}_provider")
    cfg = PROVIDERS[provider_name]

    if provider_name == "Custom (OpenAI-compatible)":
        base_url = st.text_input(f"{label_prefix} base URL", placeholder="https://.../v1",
                                  key=f"{role}_base_url")
    else:
        base_url = cfg["base_url"]
        st.caption(f"Base URL: `{base_url}`")

    model = st.text_input(f"{label_prefix} model", value=cfg["default_embed_model"]
                           if label_prefix == "Embedding" else cfg["default_chat_model"],
                           key=f"{role}_model")

    if provider_name == "Ollama (local, no key)":
        api_keys = ["ollama"]  # Ollama ignores the value but the header still needs to be sent
        st.caption("No API key needed for a local Ollama server.")
    else:
        # The first key is masked. Extra keys (to pool free accounts) go in a text area, which
        # Streamlit cannot mask, so it is tucked away and optional.
        primary_key = st.text_input(
            f"{label_prefix} API key", type="password", placeholder=cfg["key_placeholder"],
            key=f"{role}_key_primary",
        ).strip()
        with st.expander("Pool more keys (optional)"):
            extra_raw = st.text_area(
                "Extra keys, one per line", height=70,
                help="Keys from other free accounts for the same provider. The app rotates to "
                     "the next one on rate limits. Note: this box is not masked.",
                key=f"{role}_keys_extra",
            )
        extras = [line.strip() for line in extra_raw.splitlines() if line.strip()]
        api_keys = list(dict.fromkeys(([primary_key] if primary_key else []) + extras))
        if cfg["signup_url"]:
            st.caption(f"Get a free key: {cfg['signup_url']}")
        if provider_name.startswith("NVIDIA"):
            st.caption("NVIDIA retires hosted models without notice. If a test returns 410 (Gone), "
                       "pick a current model at build.nvidia.com/models.")

        # If this exact provider was already tested, use only the keys that
        # passed -- silently pruning ones the user typed in but that don't
        # work, per your earlier request.
        validated = st.session_state.valid_keys.get((role, provider_name))
        if validated is not None:
            still_present = [k for k in validated if k in api_keys]
            if still_present:
                api_keys = still_present

    ready = bool(base_url and model and api_keys)
    col1, col2 = st.columns([2, 1])
    with col1:
        if st.button(f"Test {label_prefix.lower()} key(s)", key=f"{role}_test", disabled=not ready):
            results = []
            with st.spinner(f"Testing {len(api_keys)} key(s) against {provider_name}..."):
                for key in api_keys:
                    ok, message = test_key(role, base_url, key, model, cfg["needs_input_type"])
                    results.append((key, ok, message))
            for key, ok, message in results:
                if ok:
                    st.success(f"✅ {_mask(key)} works")
                else:
                    st.error(f"❌ {_mask(key)} failed: {message}")

            working = [k for k, ok, _ in results if ok]
            if working:
                st.session_state.valid_keys[(role, provider_name)] = working
                if len(working) < len(api_keys):
                    st.warning(f"{len(api_keys) - len(working)} key(s) removed; "
                               f"{len(working)} still in use.")
            else:
                st.session_state.failed_providers[role].add(provider_name)
                st.error(f"{provider_name} failed and was removed from the list "
                         f"(no working keys).")
            st.rerun()
    with col2:
        if failed and st.button("↺ Reset", key=f"{role}_reset",
                                 help="Bring back excluded providers for this role"):
            st.session_state.failed_providers[role] = set()
            st.rerun()

    if failed:
        st.caption(f"Excluded (no working keys): {', '.join(sorted(failed))}")
    if len(api_keys) > 1:
        st.caption(f"{len(api_keys)} keys pooled for {provider_name} -- rotates on rate limits.")

    return base_url, api_keys, model, cfg["needs_input_type"]


# ---------------------------------------------------------------------------
# Sidebar: provider selection + PDF upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Embedding API")
    st.caption("Add more than one key (one per line) to pool multiple free accounts -- "
               "e.g. several NVIDIA NIM keys -- so ingestion rotates to the next key "
               "instead of stalling on one key's rate limit.")
    embed_base_url, embed_api_keys, embed_model, embed_needs_input_type = provider_picker(
        "Embedding", EMBEDDING_PROVIDERS, "embed")

    st.divider()
    st.header("2. Chat / generation API")
    st.caption("Can be a different provider than embeddings -- e.g. embed with NVIDIA, "
               "generate with Groq. Also supports multiple pooled keys.")
    chat_base_url, chat_api_keys, chat_model, _ = provider_picker(
        "Chat", CHAT_PROVIDERS, "chat")

    st.divider()
    st.header("3. Upload a PDF")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, step=100)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, step=50)

    st.header("4. Semantic cache")
    st.caption("Minimum cosine similarity to an earlier question before we reuse its "
               "cached answer instead of recomputing. Lower = more hits, but risks "
               "matching a different question. See demo_02_semantic_cache.py for why.")
    sim_threshold = st.slider("Similarity threshold", 0.50, 0.99, 0.85, step=0.01)

    embed_ready = bool(embed_base_url and embed_model and embed_api_keys)
    ingest_disabled = not (embed_ready and uploaded)
    if st.button("Ingest PDF", disabled=ingest_disabled, type="primary"):
        with st.spinner("Extracting and chunking PDF..."):
            docs = pdf_to_documents(uploaded.getvalue(), uploaded.name,
                                     chunk_size=chunk_size, overlap=overlap)
        if not docs:
            st.error("No extractable text found in that PDF (it may be scanned/image-only).")
        else:
            try:
                embedder, retriever = build_index(
                    embed_base_url, embed_api_keys, embed_model, embed_needs_input_type, docs)
            except Exception as e:
                st.error(f"Embedding API error: {e}")
            else:
                st.session_state.documents = docs
                st.session_state.cached_embedder = embedder
                st.session_state.retriever = retriever
                st.session_state.source_name = uploaded.name
                st.session_state.history = []
                st.session_state.semantic_cache = SemanticCache(threshold=sim_threshold)
                st.session_state.knowledge_graph = None  # must rebuild for the new document
                st.success(f"Ingested {len(docs)} chunks from {uploaded.name}")

    if st.session_state.documents:
        st.divider()
        st.metric("Chunks indexed", len(st.session_state.documents))
        st.caption(st.session_state.cached_embedder.stats())
        st.caption(st.session_state.semantic_cache.stats())

    # Threshold can be tuned live without re-ingesting the PDF.
    if st.session_state.semantic_cache is not None:
        st.session_state.semantic_cache.threshold = sim_threshold

    # Chat config can also change live -- store the latest picks so the main
    # panel (ask flow, knowledge graph) always uses what's currently selected.
    st.session_state.chat_config = {
        "base_url": chat_base_url, "api_keys": chat_api_keys, "model": chat_model,
    }

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("📄 RAG with Caching + GraphRAG")
st.caption("Retrieval-augmented Q&A over your PDF, against any OpenAI-compatible API. "
           "Every question is embedded once through a cache -- ask it again (or a "
           "paraphrase) and watch the answer come back nearly free.")

if not st.session_state.retriever:
    st.info("⬅️ Pick your embedding provider and ingest a PDF to get started.")
    st.stop()

chat_cfg = st.session_state.chat_config
chat_ready = bool(chat_cfg and chat_cfg["base_url"] and chat_cfg["api_keys"] and chat_cfg["model"])

# ---------------------------------------------------------------------------
# All chunks from the ingested PDF -- shown flat, no dropdown/expander, so
# you can see exactly what got indexed and how it was split.
# ---------------------------------------------------------------------------
st.subheader(f"All chunks from *{st.session_state.source_name}* "
             f"({len(st.session_state.documents)} total)")
for i, doc in enumerate(st.session_state.documents, start=1):
    st.markdown(f"**Chunk {i}** — `{doc['id']}`")
    st.text(doc["text"])
st.divider()

# ---------------------------------------------------------------------------
# Knowledge graph -- connects facts that are split across chunks (e.g. a
# reference like "(ii) resigned from the Board" in one chunk whose subject is
# only named in another). Built lazily: it costs one LLM call per chunk, so
# it's an explicit opt-in rather than something ingestion does automatically.
# ---------------------------------------------------------------------------
st.subheader("Knowledge graph (GraphRAG)")
st.caption("Extracts (subject, relation, object) facts from every chunk and links them "
           "by entity, so a question can pull in a fact from a DIFFERENT chunk than the "
           "one vector/BM25 search would retrieve -- useful when a chunk references "
           "something ('(ii)', a pronoun, 'the above') that's only resolved elsewhere.")

if st.button("Build knowledge graph", disabled=not chat_ready):
    kg = KnowledgeGraph()
    progress = st.progress(0.0, text="Extracting facts...")

    def _cb(done, total):
        progress.progress(done / total, text=f"Extracting facts from chunk {done}/{total}")

    with st.spinner("Building knowledge graph (one LLM call per chunk)..."):
        kg.build(st.session_state.documents, chat_cfg["base_url"], chat_cfg["api_keys"],
                  chat_cfg["model"], progress_cb=_cb)
    progress.empty()
    st.session_state.knowledge_graph = kg
    st.success(kg.stats())

if st.session_state.knowledge_graph is not None:
    kg = st.session_state.knowledge_graph
    triples = kg.all_triples()
    st.caption(kg.stats())
    if triples:
        for subject, relation, obj, chunk_id in triples:
            st.markdown(f"**{subject}** → *{relation}* → **{obj}**  \n"
                        f"<span style='color:gray'>source: {chunk_id}</span>",
                        unsafe_allow_html=True)
    else:
        st.caption("No clear facts were extracted from this document.")
st.divider()

st.subheader(f"Ask a question about *{st.session_state.source_name}*")
# A form batches the text input + button into ONE atomic submission. Without
# this, Streamlit can fire the block twice for a single click (e.g. Enter in
# the text box triggers its own rerun right before the button's rerun lands),
# which would silently warm the cache on the "first" pass and make an honest
# cache-miss measurement impossible.
with st.form("ask_form"):
    query = st.text_input("Your question",
                           placeholder="e.g. What is the main conclusion of this document?")
    top_k = st.slider("Chunks to retrieve", 1, 5, 3)
    ask = st.form_submit_button("Ask", type="primary", disabled=not chat_ready)

if ask and query:
    embedder: CachedEmbedder = st.session_state.cached_embedder
    retriever: HybridRetriever = st.session_state.retriever
    cache: SemanticCache = st.session_state.semantic_cache
    by_id = {d["id"]: d for d in st.session_state.documents}

    t0 = time.perf_counter()

    # Embedding this query is itself cache-backed (Technique 1: CachedEmbedder)
    # -- an exact repeat of earlier text skips even this. We need the vector
    # either way, to compare against every previously-cached question.
    query_vec = embedder.embed(query)

    # TECHNIQUE 3: semantic cache lookup. Unlike exact-match, this can hit on
    # a DIFFERENT question text if its embedding is close enough. hit_kind
    # tells us which: "exact" (identical text -> similarity ~1.0), "semantic"
    # (different text, but close enough to reuse), or "miss".
    cached_value, similarity, matched_text = cache.lookup(query_vec, query_text=query)

    if cached_value is not None:
        hit_kind = "exact" if matched_text == query else "semantic"
        answer, fused = cached_value["answer"], cached_value["chunks"]
        graph_facts = cached_value.get("graph_facts", [])
    else:
        hit_kind = "miss"
        with st.spinner("Retrieving relevant chunks and generating an answer..."):
            dense_ranked = retriever.dense.search(query_vec, top_k=top_k)
            sparse_ranked = retriever.sparse.search(query, top_k=top_k)
            fused = rrf_fuse(dense_ranked, sparse_ranked, top_k=top_k)
            context = "\n\n".join(f"[{doc_id}] {by_id[doc_id]['text']}" for doc_id, _ in fused)

            # GraphRAG: pull facts connected to entities the query mentions,
            # even from chunks vector/BM25 search didn't retrieve above.
            graph_facts = []
            if st.session_state.knowledge_graph is not None:
                graph_facts = st.session_state.knowledge_graph.related_facts(query)
                if graph_facts:
                    facts_block = "\n".join(
                        f"- {s} {r} {o} (source: {cid})" for s, r, o, cid in graph_facts
                    )
                    context += f"\n\nKnown facts from the knowledge graph:\n{facts_block}"

            try:
                answer = multi_key_chat_complete(
                    chat_cfg["base_url"], chat_cfg["api_keys"],
                    system=("Answer the user's question using ONLY the provided context. "
                            "The context may include a 'Known facts from the knowledge "
                            "graph' section -- these resolve references (like '(ii)' or "
                            "pronouns) that span multiple chunks; use them to connect "
                            "facts the chunk text alone doesn't spell out. Cite chunk IDs "
                            "like [chunk-3] when relevant. If the answer isn't in the "
                            "context, say so."),
                    user=f"Context:\n{context}\n\nQuestion: {query}",
                    model=chat_cfg["model"],
                )
            except Exception as e:
                answer = f"(Generation failed: {e})"

        cache.put(query_vec, query, {"answer": answer, "chunks": fused, "graph_facts": graph_facts})

    elapsed_ms = (time.perf_counter() - t0) * 1000

    st.session_state.history.insert(0, {
        "query": query,
        "answer": answer,
        "chunks": fused,
        "graph_facts": graph_facts,
        "elapsed_ms": elapsed_ms,
        "hit_kind": hit_kind,        # "exact" | "semantic" | "miss"
        "similarity": similarity,
        "matched_text": matched_text,
    })

# ---------------------------------------------------------------------------
# Render history (most recent first)
# ---------------------------------------------------------------------------
for entry in st.session_state.history:
    st.markdown(f"### 🗨️ {entry['query']}")

    kind = entry["hit_kind"]
    if kind == "exact":
        st.metric("Time to answer", f"{entry['elapsed_ms']:.0f} ms",
                  delta="🎯 EXACT-MATCH cache hit (identical question asked before)",
                  delta_color="off")
    elif kind == "semantic":
        st.metric("Time to answer", f"{entry['elapsed_ms']:.0f} ms",
                  delta=f"🧠 SEMANTIC cache hit (similarity {entry['similarity']:.3f} "
                        f"to a previous, differently-worded question)",
                  delta_color="off")
        st.caption(f"Matched against your earlier question: *\"{entry['matched_text']}\"*")
    else:
        st.metric("Time to answer", f"{entry['elapsed_ms']:.0f} ms",
                  delta=f"⚙️ cache MISS (best similarity {entry['similarity']:.3f}) "
                        f"— computed fresh (embedding + retrieval + generation)",
                  delta_color="off")

    st.markdown(entry["answer"])
    if entry.get("graph_facts"):
        st.caption("🕸️ Graph facts used (from chunks the retriever may not have picked):")
        for s, r, o, cid in entry["graph_facts"]:
            st.caption(f"　• {s} {r} {o} (source: {cid})")
    with st.expander(f"Retrieved chunks ({len(entry['chunks'])})"):
        by_id = {d["id"]: d for d in st.session_state.documents}
        for doc_id, score in entry["chunks"]:
            st.markdown(f"**{doc_id}** (RRF score {score:.4f})")
            st.caption(by_id[doc_id]["text"][:400] + "...")
    st.divider()
