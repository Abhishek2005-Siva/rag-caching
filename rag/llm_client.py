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


_TRANSIENT_STATUS = {429, 502, 503, 504}  # overloaded / rate-limited: worth a short wait and another try


def _post(url: str, headers: dict, payload: dict, timeout: float, retries: int) -> requests.Response:
    """POST, retrying a couple of times with backoff when the server is momentarily overloaded.

    Free hosted endpoints answer 503 for a moment under load. Timeouts are NOT retried: a request
    that already waited `timeout` seconds will not get faster."""
    delay = 1.0
    for attempt in range(retries + 1):
        resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
        if resp.status_code in _TRANSIENT_STATUS and attempt < retries:
            time.sleep(delay)
            delay *= 3
            continue
        return resp
    return resp  # unreachable, keeps type checkers happy


class OpenAICompatibleEmbedder:
    def __init__(self, base_url: str, api_key: str, model: str,
                 input_type: str | None = None, timeout: float = 30, retries: int = 2):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.retries = retries
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
        resp = _post(
            f"{self.base_url}/embeddings",
            {"Authorization": f"Bearer {self.api_key}", "Accept": "application/json"},
            payload, self.timeout, self.retries,
        )
        _check(resp)
        self.last_latency_ms = (time.perf_counter() - start) * 1000
        return resp.json()["data"][0]["embedding"]


def chat_complete(base_url: str, api_key: str, system: str, user: str,
                   model: str, max_tokens: int = 512, timeout: float = 60,
                   retry_on_length: bool = True, retries: int = 2) -> str:
    """One chat call. Always returns a string (possibly empty), never None.

    Reasoning models can spend the whole token budget thinking and return `content: null` with
    finish_reason "length". In that case retry once with more room, instead of handing the
    caller an empty answer. Set retry_on_length=False to see the raw behaviour (the model probe
    uses this to tell reasoning models, which are slow, from plain ones)."""
    def call(tokens: int) -> tuple[str, str | None]:
        resp = _post(
            f"{base_url.rstrip('/')}/chat/completions",
            {"Authorization": f"Bearer {api_key}"},
            {
                "model": model,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                "max_tokens": tokens,
                "temperature": 0.2,
            },
            timeout, retries,
        )
        _check(resp)
        choice = resp.json()["choices"][0]
        return (choice.get("message", {}).get("content") or ""), choice.get("finish_reason")

    text, finish = call(max_tokens)
    if retry_on_length and not text.strip() and finish == "length":
        text, _ = call(min(max_tokens * 3, 2048))
    return text


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
                             model: str, max_tokens: int = 512, timeout: float = 60) -> str:
    """Same as chat_complete(), but rotates through `keys` on 401/403/429."""
    rotator = KeyRotator(keys)
    last_exc = None
    for _ in range(len(keys)):
        try:
            return chat_complete(base_url, rotator.current(), system, user, model, max_tokens, timeout)
        except requests.HTTPError as e:
            last_exc = e
            if _status_of(e) in _ROTATE_ON_STATUS:
                rotator.rotate()
                continue
            raise
    raise last_exc
