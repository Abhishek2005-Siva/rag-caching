"""Registry of OpenAI-compatible API providers the app can talk to.

Most hosted LLM APIs -- NVIDIA NIM, OpenAI itself, Together AI, Groq, Mistral,
and a local Ollama server -- all expose (a compatible subset of) the same
wire format: POST {base_url}/embeddings and POST {base_url}/chat/completions,
Bearer-token auth, OpenAI-shaped JSON. rag/llm_client.py talks to any of them
just by varying base_url/api_key/model. This file is only presentation data
for the app's provider dropdowns -- add a provider by adding an entry here.

`needs_input_type` is a real API difference, not cosmetic: NVIDIA's
embedqa-style models are asymmetric (a "query" embedding and a "passage"
embedding for the same text differ) and require an `input_type` field.
Symmetric embedding models (OpenAI, most others) have no such field.
"""

PROVIDERS = {
    "NVIDIA (build.nvidia.com)": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "supports_embeddings": True,
        "supports_chat": True,
        "needs_input_type": True,
        "default_embed_model": "nvidia/nv-embedqa-mistral-7b-v2",
        "default_chat_model": "mistralai/mistral-large-2-instruct",
        "key_placeholder": "nvapi-...",
        "signup_url": "https://build.nvidia.com",
    },
    "OpenAI": {
        "base_url": "https://api.openai.com/v1",
        "supports_embeddings": True,
        "supports_chat": True,
        "needs_input_type": False,
        "default_embed_model": "text-embedding-3-small",
        "default_chat_model": "gpt-4o-mini",
        "key_placeholder": "sk-...",
        "signup_url": "https://platform.openai.com/api-keys",
    },
    "Together AI": {
        "base_url": "https://api.together.xyz/v1",
        "supports_embeddings": True,
        "supports_chat": True,
        "needs_input_type": False,
        "default_embed_model": "BAAI/bge-base-en-v1.5",
        "default_chat_model": "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
        "key_placeholder": "...",
        "signup_url": "https://api.together.ai",
    },
    "Mistral": {
        "base_url": "https://api.mistral.ai/v1",
        "supports_embeddings": True,
        "supports_chat": True,
        "needs_input_type": False,
        "default_embed_model": "mistral-embed",
        "default_chat_model": "mistral-small-latest",
        "key_placeholder": "...",
        "signup_url": "https://console.mistral.ai/api-keys",
    },
    "Groq (chat only, no embeddings)": {
        "base_url": "https://api.groq.com/openai/v1",
        "supports_embeddings": False,
        "supports_chat": True,
        "needs_input_type": False,
        "default_embed_model": "",
        "default_chat_model": "llama-3.1-8b-instant",
        "key_placeholder": "gsk_...",
        "signup_url": "https://console.groq.com/keys",
    },
    "Ollama (local, no key)": {
        "base_url": "http://localhost:11434/v1",
        "supports_embeddings": True,
        "supports_chat": True,
        "needs_input_type": False,
        "default_embed_model": "nomic-embed-text",
        "default_chat_model": "llama3.1",
        "key_placeholder": "(not required)",
        "signup_url": None,
    },
    "Custom (OpenAI-compatible)": {
        "base_url": "",
        "supports_embeddings": True,
        "supports_chat": True,
        "needs_input_type": False,
        "default_embed_model": "",
        "default_chat_model": "",
        "key_placeholder": "...",
        "signup_url": None,
    },
}

EMBEDDING_PROVIDERS = [name for name, cfg in PROVIDERS.items() if cfg["supports_embeddings"]]
CHAT_PROVIDERS = [name for name, cfg in PROVIDERS.items() if cfg["supports_chat"]]
