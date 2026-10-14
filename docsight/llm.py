"""LLM client: Groq with retries and an on-disk response cache.

Why the cache: Groq's free tier limits requests per minute and tokens per
day. Eval runs replay the same prompts many times while you tweak other
parts of the pipeline, so caching on (model, messages, params) makes reruns
free and deterministic. Turn it off for live traffic with use_cache=False.

`FakeLLM` lets tests and offline demos run the whole pipeline without a key.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


@dataclass
class LLMResponse:
    text: str
    usage: dict[str, int] = field(default_factory=dict)
    cached: bool = False


class LLM(Protocol):
    def complete(self, messages: list[dict], model: str, temperature: float = 0.0, json_mode: bool = False, max_tokens: int = 1024) -> LLMResponse: ...


class GroqLLM:
    def __init__(self, cache_dir: Path | None = None, use_cache: bool = True, max_retries: int = 5):
        from groq import Groq

        if not os.getenv("GROQ_API_KEY"):
            raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = Groq()
        self.cache_dir = cache_dir
        self.use_cache = use_cache and cache_dir is not None
        self.max_retries = max_retries
        if self.use_cache:
            cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, payload: dict) -> Path:
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def complete(self, messages, model, temperature=0.0, json_mode=False, max_tokens=1024) -> LLMResponse:
        payload = {"messages": messages, "model": model, "temperature": temperature, "json": json_mode, "max_tokens": max_tokens}
        if self.use_cache:
            path = self._key(payload)
            if path.exists():
                data = json.loads(path.read_text())
                return LLMResponse(text=data["text"], usage=data.get("usage", {}), cached=True)

        kwargs = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(**kwargs)
                break
            except Exception as exc:  # groq.RateLimitError, APIConnectionError, 5xx
                status = getattr(exc, "status_code", None)
                retryable = status in (429, 500, 502, 503) or exc.__class__.__name__ in ("APIConnectionError", "APITimeoutError")
                if not retryable or attempt == self.max_retries - 1:
                    raise
                time.sleep(delay)
                delay *= 2  # exponential backoff: 2, 4, 8, 16 s
        text = resp.choices[0].message.content or ""
        usage = {"prompt_tokens": resp.usage.prompt_tokens, "completion_tokens": resp.usage.completion_tokens} if resp.usage else {}
        if self.use_cache:
            self._key(payload).write_text(json.dumps({"text": text, "usage": usage}))
        return LLMResponse(text=text, usage=usage)


class FakeLLM:
    """Scripted LLM for tests. `responder(messages, model)` returns the reply text."""

    def __init__(self, responder: Callable[[list[dict], str], str]):
        self.responder = responder
        self.calls: list[dict] = []

    def complete(self, messages, model, temperature=0.0, json_mode=False, max_tokens=1024) -> LLMResponse:
        self.calls.append({"messages": messages, "model": model})
        text = self.responder(messages, model)
        return LLMResponse(text=text, usage={"prompt_tokens": sum(len(m["content"].split()) for m in messages), "completion_tokens": len(text.split())})
