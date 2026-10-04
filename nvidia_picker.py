"""Pick an NVIDIA hosted model that actually works for a visitor's own API key.

Why this exists: NVIDIA's public catalog lists many models a given key cannot call (HTTP 404
"Function ... not found for account"), the set differs from key to key, models get retired (410),
and the free servers are sometimes overloaded (503 / timeouts). So a hardcoded default, or even a
live catalog list, is not enough: the only reliable test is a real request with the visitor's key.

find_working_model() probes candidates in parallel (so a few slow models cannot eat the time
budget), prefers the fastest model that returns a useful answer, and when nothing works it says
why (rejected key / no model access / servers busy) instead of just "no model worked".
Nothing here depends on Streamlit.
"""
from __future__ import annotations

import re
import time
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

import requests

NVIDIA_URL = "https://integrate.api.nvidia.com/v1"
FAST_ENOUGH_SECONDS = 3.0  # stop waiting as soon as a good model answers this quickly

CHAT_PROMPT = "In one short sentence, say what a patent is."
TOOL_PROMPT = "What is the weather in Paris right now?"
TOOL = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def key_problem(api_key: str) -> str | None:
    """A message if the key obviously isn't an NVIDIA key, else None."""
    key = (api_key or "").strip()
    if key and not key.startswith("nvapi-"):
        return "NVIDIA keys start with `nvapi-`. This one doesn't, so it may be a key from another provider."
    return None


def _probe(api_key: str, model: str, kind: str, with_tools: bool, timeout: float, base_url: str):
    """One real request. Returns (state, seconds, detail); state is good / callable / failed.

    good      chat: a real sentence (or, with_tools, a tool call); embed: a vector.
    callable  the call succeeded but returned nothing usable (reasoning models often do this
              with a small token budget).
    failed    HTTP error, timeout, connection problem."""
    headers = {"Authorization": f"Bearer {api_key}"}
    started = time.monotonic()
    try:
        if kind == "embed":
            body = {"model": model, "input": ["hello"], "input_type": "passage", "encoding_format": "float"}
            url = f"{base_url}/embeddings"
        else:
            prompt = TOOL_PROMPT if with_tools else CHAT_PROMPT
            body = {"model": model, "max_tokens": 150, "temperature": 0.2,
                    "messages": [{"role": "user", "content": prompt}]}
            if with_tools:
                body.update(tools=[TOOL], tool_choice="auto")
            url = f"{base_url}/chat/completions"
        resp = None
        for attempt in range(2):  # one retry: free endpoints answer 503 for a moment under load
            resp = requests.post(url, headers=headers, json=body, timeout=timeout)
            if resp.status_code in (429, 502, 503, 504) and attempt == 0:
                time.sleep(1.0)
                continue
            break
        seconds = time.monotonic() - started
        if not resp.ok:
            return "failed", seconds, f"HTTP {resp.status_code}: {' '.join(resp.text.split())[:140]}"
        data = resp.json()
        if kind == "embed":
            vec = (data.get("data") or [{}])[0].get("embedding")
            return ("good" if vec else "callable"), seconds, f"dim={len(vec) if vec else 0}"
        message = (data.get("choices") or [{}])[0].get("message") or {}
        if with_tools and message.get("tool_calls"):
            return "good", seconds, "tool call"
        text = (message.get("content") or "").strip()
        return ("good" if len(text) >= 20 and not with_tools else "callable"), seconds, text[:80] or "(empty reply)"
    except requests.Timeout:
        return "failed", time.monotonic() - started, f"timeout after {timeout:.0f}s"
    except requests.RequestException as exc:
        return "failed", time.monotonic() - started, f"connection problem: {str(exc)[:100]}"


def diagnose(failures: list[tuple[str, str]], tried: int) -> str:
    """Explain, in plain words, why every candidate failed."""
    kinds: Counter = Counter()
    for _, detail in failures:
        if re.search(r"HTTP (401|403)", detail):
            kinds["auth"] += 1
        elif re.search(r"HTTP (404|410)", detail):
            kinds["unavailable"] += 1
        else:
            kinds["busy"] += 1  # 429/5xx/timeouts/connection errors
    total = sum(kinds.values()) or 1
    if kinds["auth"] == total:
        return ("NVIDIA rejected this key (HTTP 401/403). It may be mistyped, revoked or expired. "
                "Create a new one at build.nvidia.com and paste the whole `nvapi-…` string.")
    if kinds["unavailable"] == total:
        return (f"NVIDIA accepted the key, but none of the {tried} models I tried is available to it "
                "(\"not found for account\"). The key probably wasn't created with access to the hosted "
                "models. Open any model page on build.nvidia.com, click **Get API Key**, and use that key.")
    if kinds["busy"] >= total * 0.5:
        return ("NVIDIA's free servers look busy (timeouts or 5xx errors). Wait a minute and try again.")
    parts = ", ".join(f"{n} {label}" for label, n in
                      (("rejected the key", kinds["auth"]), ("not available to this key", kinds["unavailable"]),
                       ("timed out or errored", kinds["busy"])) if n)
    return f"None of the {tried} models worked: {parts}. See the list for details."


def find_working_model(api_key: str, candidates: list[str], *, kind: str = "chat", with_tools: bool = False,
                       workers: int = 8, budget: float = 40, timeout: float = 12,
                       max_candidates: int = 40, base_url: str = NVIDIA_URL) -> dict:
    """Probe `candidates` in parallel and pick the best.

    Returns {"model", "quality" ("good"|"untested"|None), "seconds", "failures": [(model, detail)],
    "diagnosis" (None when a model was found), "tried"}. The best model is the fastest "good" one;
    failing that, the fastest "callable" one, marked untested. `with_tools=True` makes "good" mean
    the model can emit a tool call, which an agent needs."""
    pool = list(dict.fromkeys(candidates))[:max_candidates]
    started = time.monotonic()
    executor = ThreadPoolExecutor(max_workers=workers)
    futures = {executor.submit(_probe, api_key, m, kind, with_tools, timeout, base_url): (i, m)
               for i, m in enumerate(pool)}
    results, remaining, stopped_early = {}, set(futures), False
    while remaining:
        left = budget - (time.monotonic() - started)
        if left <= 0:
            break
        done, remaining = wait(remaining, timeout=left, return_when=FIRST_COMPLETED)
        for future in done:
            results[future] = future.result()
        if any(state == "good" and seconds <= FAST_ENOUGH_SECONDS for state, seconds, _ in results.values()):
            stopped_early = bool(remaining)  # a quick, good model is in hand: no need to wait for stragglers
            break
    executor.shutdown(wait=False, cancel_futures=True)

    good, callable_only, failures = [], [], []
    for future in sorted(results, key=lambda f: futures[f][0]):  # keep candidate order in the failure list
        index, model = futures[future]
        state, seconds, detail = results[future]
        if state == "good":
            good.append((seconds, index, model))
        elif state == "callable":
            callable_only.append((seconds, index, model))
            failures.append((model, f"answered, but not usefully: {detail}"))
        else:
            failures.append((model, detail))
    if not stopped_early:
        for future in remaining:
            failures.append((futures[future][1], f"no answer within {budget:.0f}s"))

    report = {"model": None, "quality": None, "seconds": None, "failures": failures,
              "diagnosis": None, "tried": len(pool)}
    for quality, found in (("good", good), ("untested", callable_only)):
        if found:
            seconds, _, model = min(found)
            report.update(model=model, quality=quality, seconds=seconds)
            break
    if report["model"] is None:
        report["diagnosis"] = diagnose(failures, len(pool))
    return report
