"""Generic client for any OpenAI-compatible API -- embeddings and chat.

Replaces the NVIDIA-only rag/nvidia_client.py with something that takes
base_url/api_key/model as parameters, so it works against any provider in
rag/providers.py (or a Custom base_url the user types in). Same shapes as
before: `.embed(text) -> list[float]` with a `.calls` counter (drop-in for
CachedEmbedder / HybridRetriever), and a one-shot `chat_complete(...)`.
"""
import time

import requests


def _check(resp: requests.Response) -> None:
    """raise_for_status(), but keep the server's explanation in the message.

    Providers say *why* in the body, e.g. NVIDIA's 404 "Function ... not found for account"
    means the model exists in the catalog but is not available to this key."""
    try:
        resp.raise_for_status()
    except requests.HTTPError as exc:
        detail = " ".join((resp.text or "").split())[:300]
        raise requests.HTTPError(f"{exc} | {detail}" if detail else str(exc), response=resp) from exc


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str,
                 input_type: str | None = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.input_type = input_type  # only providers like NVIDIA use this
        self.calls = 0
        self.last_latency_ms = 0.0

    def embed(self, text: str) -> list[float]:
        self.calls += 1
        start = time.perf_counter()
        payload = {"input": [text], "model": self.model, "encoding_format": "float"}
        if self.input_type:
            payload["input_type"] = self.input_type
        resp = requests.post(
            f"{self.base_url}/embeddings",
            headers={"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
            json=payload,
            timeout=30,
        )
        _check(resp)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return resp.json()["data"][0]["embedding"]


def chat_complete(base_url: str, api_key: str, system: str, user: str,
                   model: str, max_tokens: int = 512) -> str:
    resp = requests.post(
        f"{base_url.rstrip('/')}/chat/completions",
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
    _check(resp)
    return resp.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Multi-key rotation.
#
# Free-tier keys (NVIDIA NIM's included) typically carry per-key rate limits.
# One request stream against a single key hits 429s under any real load; the
# practical fix is to hold a POOL of keys (e.g. several free NVIDIA accounts)
# and rotate to the next one whenever the current key is rate-limited or
# rejected, instead of failing the whole request. This is generic -- it works
# for any OpenAI-compatible provider, not just NVIDIA.
# ---------------------------------------------------------------------------
_ROTATE_ON_STATUS = {401, 403, 429}  # auth failures + rate limits -> try the next key


class KeyRotator:
    """Round-robins through a list of keys. `.current()` always returns the
    same key until `.rotate()` is called -- callers rotate only on failure,
    so a healthy key stays in use across many calls."""

    def __init__(self, keys: list[str]):
        if not keys:
            raise ValueError("KeyRotator needs at least one API key")
        self.keys = keys
        self.index = 0

    def current(self) -> str:
        return self.keys[self.index]

    def rotate(self):
        self.index = (self.index + 1) % len(self.keys)


def _status_of(exc: requests.HTTPError) -> int | None:
    return exc.response.status_code if exc.response is not None else None


class MultiKeyEmbedder:
    """Same `.embed(text)` / `.calls` shape as OpenAICompatibleEmbedder, but
    backed by a pool of keys. On a 401/403/429 from the current key, rotates
    to the next key and retries the SAME request -- transparent to callers
    (including CachedEmbedder, which just sees `.embed()`)."""

    def __init__(self, base_url: str, keys: list[str], model: str, input_type: str | None = None):
        self.rotator = KeyRotator(keys)
        self.base_url, self.model, self.input_type = base_url, model, input_type
        self.calls = 0
        self.rotations = 0  # how many times a key got rate-limited/rejected

    def embed(self, text: str) -> list[float]:
        last_exc = None
        for _ in range(len(self.rotator.keys)):
            key = self.rotator.current()
            try:
                vec = OpenAICompatibleEmbedder(self.base_url, key, self.model, self.input_type).embed(text)
                self.calls += 1
                return vec
            except requests.HTTPError as e:
                last_exc = e
                if _status_of(e) in _ROTATE_ON_STATUS:
                    self.rotations += 1
                    self.rotator.rotate()
                    continue
                raise  # a real error (bad model name, etc.) -- don't mask it by rotating
        raise last_exc  # every key in the pool failed


def multi_key_chat_complete(base_url: str, keys: list[str], system: str, user: str,
                             model: str, max_tokens: int = 512) -> str:
    """Same as chat_complete(), but rotates through `keys` on 401/403/429."""
    rotator = KeyRotator(keys)
    last_exc = None
    for _ in range(len(keys)):
        try:
            return chat_complete(base_url, rotator.current(), system, user, model, max_tokens)
        except requests.HTTPError as e:
            last_exc = e
            if _status_of(e) in _ROTATE_ON_STATUS:
                rotator.rotate()
                continue
            raise
    raise last_exc
