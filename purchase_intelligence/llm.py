"""Shared Groq JSON call with a process-wide free-tier budget guard."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable

# Measured on the free tier: 8,000 tokens/min and 1,000 requests/day; keep headroom for estimate error.
DEFAULT_TPM_LIMIT = 7200
DEFAULT_DAILY_REQUEST_LIMIT = 900
DEFAULT_MODEL = "qwen/qwen3.8-27b"


class LlmBudget:
    """Rolling token and request accounting shared by every Streamlit session in the process."""

    def __init__(self, tpm_limit: int = DEFAULT_TPM_LIMIT, daily_limit: int = DEFAULT_DAILY_REQUEST_LIMIT,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.tpm_limit = tpm_limit
        self.daily_limit = daily_limit
        self._clock = clock
        self._events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    def _trim(self, now: float) -> None:
        while self._events and now - self._events[0][0] > 86_400:
            self._events.popleft()

    def allow(self, estimated_tokens: int) -> tuple[bool, str]:
        with self._lock:
            now = self._clock()
            self._trim(now)
            minute_tokens = sum(tokens for stamp, tokens in self._events if now - stamp <= 60)
            if len(self._events) >= self.daily_limit:
                return False, "daily request budget reached"
            if minute_tokens + estimated_tokens > self.tpm_limit:
                return False, "per-minute token budget reached; retry in about a minute"
            return True, "within budget"

    def record(self, tokens: int) -> None:
        with self._lock:
            now = self._clock()
            self._events.append((now, max(int(tokens), 0)))
            self._trim(now)

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            now = self._clock()
            self._trim(now)
            return {
                "tokens_last_minute": sum(tokens for stamp, tokens in self._events if now - stamp <= 60),
                "requests_last_24h": len(self._events),
                "tpm_limit": self.tpm_limit,
                "daily_limit": self.daily_limit,
            }


BUDGET = LlmBudget()


@dataclass
class LlmReply:
    data: dict[str, Any] | None
    status: str
    tokens: int = 0


def redact(text: str) -> str:
    text = text.replace(os.getenv("GROQ_API_KEY", "\0"), "[REDACTED]")
    text = re.sub(r"gsk_[A-Za-z0-9_-]+", "[REDACTED]", text)
    return re.sub(r"org_[A-Za-z0-9]+", "[ORG]", text)


def estimate_tokens(*texts: str) -> int:
    return int(sum(len(text) for text in texts) / 3.5) + 1


def groq_json(system: str, user: str, max_tokens: int, budget: LlmBudget | None = None) -> LlmReply:
    """Call Groq in JSON mode; never raises, and degrades to status text when unavailable or over budget."""
    from purchase_intelligence.agents.groq_analysis_agent import _get_client

    budget = budget or BUDGET
    client, client_status = _get_client()
    if client is None:
        return LlmReply(None, client_status)
    allowed, reason = budget.allow(estimate_tokens(system, user) + max_tokens)
    if not allowed:
        return LlmReply(None, f"Groq skipped: {reason}")
    try:
        response = client.chat.completions.create(
            model=os.getenv("GROQ_MODEL", DEFAULT_MODEL), temperature=0, max_tokens=max_tokens,
            response_format={"type": "json_object"},
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = getattr(response, "usage", None)
        tokens = int(getattr(usage, "total_tokens", 0) or estimate_tokens(system, user) + max_tokens)
        budget.record(tokens)
        parsed = json.loads(response.choices[0].message.content or "{}")
        if not isinstance(parsed, dict):
            return LlmReply(None, "Groq returned JSON that is not an object", tokens)
        return LlmReply(parsed, "ok", tokens)
    except Exception as exc:
        budget.record(estimate_tokens(system, user))
        code = getattr(exc, "status_code", None)
        suffix = f" (HTTP {code})" if code is not None else ""
        return LlmReply(None, f"Groq request failed: {type(exc).__name__}{suffix}: {redact(str(exc))[:200]}")
