"""Minimal OpenRouter client. Synchronous; uses httpx.

Determinism: temperature=0. Caps total calls per process to protect
against runaway spend. Logs the routed provider when available.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import httpx

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class CostCapExceeded(RuntimeError):
    pass


@dataclass
class CompletionResult:
    raw_text: str
    model_id: str
    provider: Optional[str]
    tokens_in: int
    tokens_out: int
    latency_s: float
    response_id: Optional[str] = None
    error: Optional[str] = None


@dataclass
class OpenRouterClient:
    api_key: str
    timeout_s: float = 90.0
    max_calls: int = 500
    _calls_made: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        if not self.api_key:
            raise ValueError("OpenRouter API key is empty")

    def complete(
        self,
        model: str,
        system: str,
        user: str,
        response_format: Optional[dict] = None,
        temperature: float = 0.0,
        max_tokens: int = 8192,
        retries: int = 3,
    ) -> CompletionResult:
        if self._calls_made >= self.max_calls:
            raise CostCapExceeded(
                f"call cap of {self.max_calls} reached; aborting to limit spend"
            )
        self._calls_made += 1

        body: dict = {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format is not None:
            body["response_format"] = response_format

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/del-bench/del-bench",
            "X-Title": "DEL-Bench Pilot",
        }

        last_err: Optional[str] = None
        for attempt in range(retries):
            t0 = time.monotonic()
            try:
                with httpx.Client(timeout=self.timeout_s) as client:
                    resp = client.post(OPENROUTER_URL, json=body, headers=headers)
                latency = time.monotonic() - t0
                if resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
                    time.sleep(2**attempt)
                    continue
                if resp.status_code >= 400:
                    # Don't retry deterministic 4xx; if response_format unsupported, retry without it.
                    if (
                        response_format is not None
                        and "response_format" in resp.text.lower()
                    ):
                        body.pop("response_format", None)
                        continue
                    return CompletionResult(
                        raw_text="",
                        model_id=model,
                        provider=None,
                        tokens_in=0,
                        tokens_out=0,
                        latency_s=latency,
                        error=f"HTTP {resp.status_code}: {resp.text[:500]}",
                    )
                try:
                    data = resp.json()
                except ValueError as e:
                    return CompletionResult(
                        raw_text="",
                        model_id=model,
                        provider=None,
                        tokens_in=0,
                        tokens_out=0,
                        latency_s=latency,
                        error=(
                            f"invalid JSON response: {type(e).__name__}: {e}; "
                            f"body={resp.text[:500]!r}"
                        ),
                    )
                if not isinstance(data, dict):
                    return CompletionResult(
                        raw_text="",
                        model_id=model,
                        provider=None,
                        tokens_in=0,
                        tokens_out=0,
                        latency_s=latency,
                        error=f"unexpected JSON response type: {type(data).__name__}",
                    )
                choices = data.get("choices") or []
                if not choices:
                    return CompletionResult(
                        raw_text="",
                        model_id=model,
                        provider=data.get("provider"),
                        tokens_in=0,
                        tokens_out=0,
                        latency_s=latency,
                        response_id=data.get("id"),
                        error="no choices in response",
                    )
                content = choices[0].get("message", {}).get("content", "") or ""
                usage = data.get("usage") or {}
                return CompletionResult(
                    raw_text=content,
                    model_id=data.get("model", model),
                    provider=data.get("provider"),
                    tokens_in=int(usage.get("prompt_tokens", 0) or 0),
                    tokens_out=int(usage.get("completion_tokens", 0) or 0),
                    latency_s=latency,
                    response_id=data.get("id"),
                )
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
                time.sleep(2**attempt)

        return CompletionResult(
            raw_text="",
            model_id=model,
            provider=None,
            tokens_in=0,
            tokens_out=0,
            latency_s=0.0,
            error=last_err or "exhausted retries",
        )


def from_env(env_var: str = "OPENROUTER_API_KEY", **kwargs) -> OpenRouterClient:
    key = os.environ.get(env_var, "").strip()
    if not key:
        # Try .env in cwd as a convenience.
        from pathlib import Path

        env_path = Path.cwd() / ".env"
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, _, v = line.partition("=")
                    if k.strip() == env_var:
                        key = v.strip().strip('"').strip("'")
                        break
    if not key:
        raise RuntimeError(f"{env_var} not set in environment or .env")
    return OpenRouterClient(api_key=key, **kwargs)
