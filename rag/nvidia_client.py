"""Real embeddings + generation via NVIDIA's free NIM API (build.nvidia.com).

NVIDIA exposes an OpenAI-compatible endpoint. You only need a free API key
(starts with "nvapi-") from https://build.nvidia.com -- no credit card.

This module has the SAME shape as rag/embedder.py's HashingEmbedder
(`.embed(text) -> list[float]`, `.calls` counter) so it's a drop-in swap
everywhere the fake embedder was used, including under CachedEmbedder.
"""
import time

import requests

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

# Free-tier NIM models that work well for this project.
EMBED_MODEL = "nvidia/nv-embedqa-e5-v5"
CHAT_MODEL = "meta/llama-3.1-8b-instruct"


class NvidiaEmbedder:
    """Calls NVIDIA's embeddings endpoint. One real network round-trip per call
    (unless wrapped by CachedEmbedder) -- this is the "slow" part we cache."""

    def __init__(self, api_key: str, model: str = EMBED_MODEL, input_type: str = "query"):
        self.api_key = api_key
        self.model = model
        self.input_type = input_type  # NVIDIA's embedqa models want "query" or "passage"
        self.calls = 0
        self.last_latency_ms = 0.0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        start = time.perf_counter()
        resp = requests.post(
            f"{NVIDIA_BASE_URL}/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            json={
                "input": [text],
                "model": self.model,
                "input_type": self.input_type,
                "encoding_format": "float",
            },
            timeout=30,
        )
        resp.raise_for_status()
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return resp.json()["data"][0]["embedding"]


def chat_complete(api_key: str, system: str, user: str, model: str = CHAT_MODEL,
                   max_tokens: int = 512) -> str:
    """One-shot chat completion against an NVIDIA-hosted LLM (OpenAI-style API)."""
    resp = requests.post(
        f"{NVIDIA_BASE_URL}/chat/completions",
        headers={"Authorization": f"Bearer {api_key}"},
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": 0.2,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]
