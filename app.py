"""Streamlit UI: upload a PDF, ask questions, see the embedding-cache speedup.

Works with NVIDIA's free hosted models or OpenAI. Pick one provider in the
sidebar and paste ONE API key: it is used for both embeddings and chat.
(rag/providers.py also lists other OpenAI-compatible providers; the UI offers
the two most common.)

Run:  .venv/bin/streamlit run app.py
"""
import json
import re
import sys
import time
import urllib.request
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))  # nvidia_picker.py lives next to this file

from rag.cache import CachedEmbedder, SemanticCache
from rag.graph_rag import KnowledgeGraph
from rag.hybrid import HybridRetriever, rrf_fuse
from rag.llm_client import (
    OpenAICompatibleEmbedder, chat_complete, MultiKeyEmbedder, multi_key_chat_complete,
)
from rag.pdf_utils import pdf_to_documents
from nvidia_picker import find_working_model, key_problem
from rag.providers import PROVIDERS

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


def test_key(role: str, base_url: str, api_key: str, model: str, needs_input_type: bool,
             timeout: float = 30) -> tuple[bool, str]:
    """One cheap real call to confirm a SINGLE key actually works."""
    try:
        if role == "embed":
            OpenAICompatibleEmbedder(
                base_url, api_key, model, input_type="passage" if needs_input_type else None,
                timeout=timeout,
            ).embed("connection test")
        else:
            chat_complete(base_url, api_key, system="Reply with OK.", user="OK?",
                           model=model, max_tokens=5, timeout=timeout)
        return True, "OK"
    except Exception as e:
        return False, str(e)


# NVIDIA's hosted lineup changes often (models get retired without notice), so read the live list.
NVIDIA_MODELS_URL = "https://integrate.api.nvidia.com/v1/models"
_NON_CHAT = re.compile(
    r"embed|rerank|safety|guard|reward|parse|vlm|vision|clip|retriev|riva|neva|vila|kosmos|"
    r"deplot|fuyu|video|cosmos|ising|starcoder", re.I)
_EMBEDDING = re.compile(r"embed", re.I)  # any embedding model; which ones a key can call differs per account
_PREFERRED_CHAT = ["mistralai/mistral-large-2-instruct", "nvidia/llama-3.1-nemotron-70b-instruct",
                   "nvidia/nemotron-nano-3-30b-a3b", "google/diffusiongemma-26b-a4b-it",
                   "google/gemma-4-31b-it", "openai/gpt-oss-20b",
                   "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning", "nvidia/nemotron-3-ultra-550b-a55b"]
_PREFERRED_EMBED = ["nvidia/nemotron-3-embed-1b", "nvidia/nv-embedqa-mistral-7b-v2",
                    "nvidia/llama-3.2-nv-embedqa-1b-v1", "nvidia/llama-nemotron-embed-vl-1b-v2"]
PROVIDER_CHOICES = ["NVIDIA (build.nvidia.com)", "OpenAI"]


@st.cache_data(ttl=1800, show_spinner=False)
def nvidia_models() -> tuple[list[str], list[str]]:
    """(chat models, embedding models) NVIDIA serves right now, preferred ones first."""
    try:
        with urllib.request.urlopen(NVIDIA_MODELS_URL, timeout=8) as resp:
            ids = [m["id"] for m in json.load(resp)["data"]]
        chat = [i for i in ids if not _NON_CHAT.search(i)]
        embed = [i for i in ids if _EMBEDDING.search(i)]

        def order(found, preferred):
            first = [m for m in preferred if m in found]
            return (first + [i for i in found if i not in first]) or list(preferred)

        return order(chat, _PREFERRED_CHAT), order(embed, _PREFERRED_EMBED)
    except Exception:  # noqa: BLE001 - offline or endpoint changed
        return list(_PREFERRED_CHAT), list(_PREFERRED_EMBED)


def find_working_models(key: str, chat_candidates: list[str], embed_candidates: list[str]) -> dict:
    """Find an embedding model and a chat model this key can actually call (shared picker).

    NVIDIA's catalog lists models some keys cannot call (404) and it differs from key to key, so
    each model is tried for real, in parallel, and the fastest good one is used."""
    embed = find_working_model(key, embed_candidates, kind="embed")
    chat = find_working_model(key, chat_candidates, kind="chat")
    return {
        "embed": embed["model"], "chat": chat["model"], "chat_quality": chat["quality"],
        "failures": ([f"embedding {m}: {d}" for m, d in embed["failures"]]
                     + [f"chat {m}: {d}" for m, d in chat["failures"]]),
        "diagnosis": [d for d in (None if embed["model"] else "Embeddings: " + embed["diagnosis"],
                                  None if chat["model"] else "Chat: " + chat["diagnosis"]) if d],
    }


def connection_picker():
    """One provider, one masked API key, used for BOTH embeddings and chat.

    Returns (base_url, api_keys, embed_model, chat_model, needs_input_type). `api_keys` is a list
    of at most one key because the rotation helpers in rag/llm_client.py take a list."""
    # A previous "Find working models" run chose models; apply them before the widgets are built.
    pending = st.session_state.pop("pending_models", None)
    if pending:
        if pending.get("embed"):
            st.session_state["embed_model_sel"] = pending["embed"]
        if pending.get("chat"):
            st.session_state["chat_model_sel"] = pending["chat"]

    provider_name = st.selectbox("Provider", PROVIDER_CHOICES)
    cfg = PROVIDERS[provider_name]
    base_url = cfg["base_url"]

    key = st.text_input(
        "API key", type="password", placeholder=cfg["key_placeholder"],
        help="One key for both embeddings and chat. Used only in this browser session; never stored.",
    ).strip()
    if cfg["signup_url"]:
        st.caption(f"Get a free key: {cfg['signup_url']}")
    if provider_name.startswith("NVIDIA") and key_problem(key):
        st.warning(key_problem(key))
    api_keys = [key] if key else []

    if provider_name.startswith("NVIDIA"):
        chat_models, embed_models = nvidia_models()
        embed_model = st.selectbox("Embedding model", embed_models, key="embed_model_sel")
        chat_model = st.selectbox("Chat model", chat_models, key="chat_model_sel")
    else:
        embed_model = st.text_input("Embedding model", value=cfg["default_embed_model"])
        chat_model = st.text_input("Chat model", value=cfg["default_chat_model"])

    needs_input_type = cfg["needs_input_type"]
    if st.button("Test key", disabled=not (api_keys and embed_model and chat_model)):
        with st.spinner("Testing embeddings and chat..."):
            for role, label, model in (("embed", "Embeddings", embed_model), ("chat", "Chat", chat_model)):
                ok, message = test_key(role, base_url, key, model, needs_input_type)
                if ok:
                    st.success(f"✅ {label} work")
                else:
                    st.error(f"❌ {label} failed: {message}")
    if provider_name.startswith("NVIDIA"):
        if st.button("Find working models", disabled=not api_keys,
                     help="Some models in NVIDIA's catalog aren't available to every key (they "
                          "return 404). This tries candidates with your key and picks ones that work."):
            with st.spinner("Trying models with your key (up to ~40 seconds)..."):
                report = find_working_models(key, chat_models, embed_models)
            st.session_state["probe_report"] = report
            st.session_state["pending_models"] = {"embed": report["embed"], "chat": report["chat"]}
            st.rerun()
        report = st.session_state.get("probe_report")
        if report:
            if report["embed"] and report["chat"]:
                st.success(f"Using {report['embed']} for embeddings and {report['chat']} for chat.")
                if report["chat_quality"] == "untested":
                    st.warning("That chat model responds but returned little text (it may be a slow reasoning "
                               "model), so answers and the knowledge graph may be slow or fail. No better "
                               "model was available to this key right now.")
            else:
                for reason in report["diagnosis"]:
                    st.error(reason)
            if report["failures"]:
                with st.expander(f"Models that didn't work ({len(report['failures'])})"):
                    for line in report["failures"]:
                        st.text(line)
    return base_url, api_keys, embed_model, chat_model, needs_input_type


# ---------------------------------------------------------------------------
# Sidebar: provider selection + PDF upload
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("1. Provider and API key")
    st.caption("One key does everything: it embeds your document and answers your questions. "
               "NVIDIA's models are free; OpenAI needs a paid key.")
    base_url, api_keys, embed_model, chat_model, embed_needs_input_type = connection_picker()

    st.divider()
    st.header("2. Upload a PDF")
    uploaded = st.file_uploader("PDF file", type=["pdf"])
    chunk_size = st.slider("Chunk size (chars)", 300, 2000, 800, step=100)
    overlap = st.slider("Chunk overlap (chars)", 0, 400, 150, step=50)

    st.header("3. Semantic cache")
    st.caption("Minimum cosine similarity to an earlier question before we reuse its "
               "cached answer instead of recomputing. Lower = more hits, but risks "
               "matching a different question. See demo_02_semantic_cache.py for why.")
    sim_threshold = st.slider("Similarity threshold", 0.50, 0.99, 0.85, step=0.01)
    strict_match = st.checkbox(
        "Strict matching (recommended)", value=True,
        help="Embedding similarity can't tell \"projects\" from \"experience and projects\", so a "
             "compound question could be served half an answer. Strict matching only reuses a cached "
             "answer if your question adds no topic words the earlier question lacked.")

    embed_ready = bool(base_url and embed_model and api_keys)
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
                    base_url, api_keys, embed_model, embed_needs_input_type, docs)
            except Exception as e:
                st.error(f"Embedding API error: {e}")
            else:
                st.session_state.documents = docs
                st.session_state.cached_embedder = embedder
                st.session_state.retriever = retriever
                st.session_state.source_name = uploaded.name
                st.session_state.history = []
                st.session_state.semantic_cache = SemanticCache(threshold=sim_threshold, require_word_coverage=strict_match)
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
        st.session_state.semantic_cache.require_word_coverage = strict_match

    # Chat config can also change live -- store the latest picks so the main
    # panel (ask flow, knowledge graph) always uses what's currently selected.
    st.session_state.chat_config = {
        "base_url": base_url, "api_keys": api_keys, "model": chat_model,
    }

# ---------------------------------------------------------------------------
# Main panel
# ---------------------------------------------------------------------------
st.title("📄 RAG with Caching + GraphRAG")
st.caption("Retrieval-augmented Q&A over your PDF, against any OpenAI-compatible API. "
           "Every question is embedded once through a cache -- ask it again (or a "
           "paraphrase) and watch the answer come back nearly free.")

if not st.session_state.retriever:
    st.info("⬅️ Enter your API key, upload a PDF and click Ingest to get started.")
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
    if kg.aborted:
        st.error(kg.aborted)
    else:
        st.session_state.knowledge_graph = kg
        st.success(kg.stats())
    if kg.skipped:
        with st.expander(f"{len(kg.skipped)} chunk(s) were skipped (the model errored on them)"):
            for chunk_id, reason in kg.skipped:
                st.text(f"{chunk_id}: {reason}")

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
    blocked = cache.last_blocked  # set when a near-match was rejected by strict matching

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
        "blocked": blocked,          # a near-match strict matching refused to reuse, or None
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
        if entry.get("blocked"):
            b = entry["blocked"]
            st.caption(f"A similar earlier question (*\"{b['matched_text']}\"*, similarity "
                       f"{b['similarity']:.3f}) was not reused because yours also asks about: "
                       f"**{', '.join(b['extra_words'])}**.")

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
