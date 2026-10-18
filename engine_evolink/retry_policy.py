"""Retry policy for N-in-flight provider calls; sequential legacy paths stay separate."""

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import random
import time

_RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}
_FATAL_WORDS = (
    "invalid api key",
    "unauthorized",
    "forbidden",
    "payment required",
    "insufficient balance",
    "insufficient credit",
    "billing",
)
_RETRYABLE_WORDS = (
    "rate limit",
    "ratelimit",
    "resource exhausted",
    "too many requests",
    "throttl",
    "high demand",
    "service unavailable",
    "temporarily unavailable",
    "try again later",
)
_HIGH_DEMAND_WORDS = (
    "high demand",
    "service unavailable",
    "temporarily unavailable",
)


class RetryableResponseError(RuntimeError):
    """A retryable HTTP response returned without raising from the client."""

    def __init__(self, response, *, no_replay: bool = False):
        self.response = response
        self.status_code = getattr(response, "status_code", None)
        self.no_replay = no_replay
        super().__init__(f"HTTP {self.status_code} response")


def _response(exc):
    response = getattr(exc, "response", None)
    if response is None:
        response = getattr(exc, "http_response", None)
    return response


def status_code(value) -> int | None:
    response = _response(value) if isinstance(value, BaseException) else None
    for source in (value, response):
        if source is None:
            continue
        for name in ("status_code", "status", "code"):
            raw = getattr(source, name, None)
            try:
                code = int(raw)
            except (TypeError, ValueError, OverflowError):
                continue
            if 100 <= code <= 599:
                return code
    return None


def classify_error(exc: Exception) -> str:
    """Return fatal, retryable, or other for a provider error."""
    code = status_code(exc)
    text = str(exc).lower()
    if code in {401, 402, 403} or any(word in text for word in _FATAL_WORDS):
        return "fatal"
    if code in _RETRYABLE_STATUS or (code is not None and code >= 500):
        return "retryable"
    if any(word in text for word in _RETRYABLE_WORDS):
        return "retryable"
    return "other"


def is_high_demand(exc: Exception) -> bool:
    code = status_code(exc)
    text = str(exc).lower()
    return code == 503 or any(word in text for word in _HIGH_DEMAND_WORDS)


def retry_after_seconds(exc: Exception) -> float:
    response = _response(exc)
    headers = getattr(response, "headers", None)
    if headers is None:
        headers = getattr(exc, "headers", None)
    if not isinstance(headers, Mapping):
        return 0.0
    value = headers.get("Retry-After") or headers.get("retry-after")
    if value is None:
        return 0.0
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError, OverflowError):
        pass
    try:
        when = parsedate_to_datetime(str(value))
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())
    except (TypeError, ValueError, OverflowError):
        return 0.0


def request_with_retry(engine, operation: Callable, prefix: str):
    """Retry one provider operation with exponential backoff and Retry-After."""
    max_retries = max(0, int(engine._profile.get("retry_on_429", 4)))
    base_delay = max(0.0, float(engine._profile.get("retry_base_delay", 2.0)))
    last_exc = None
    for attempt in range(max_retries + 1):
        try:
            return operation()
        except Exception as exc:
            if classify_error(exc) != "retryable":
                raise
            high_demand = is_high_demand(exc)
            computed = base_delay * (8 if high_demand else 2) ** attempt
            computed += random.uniform(0, 1)
            delay = max(computed, retry_after_seconds(exc))
            label = "high-demand outage" if high_demand else "429 rate limit"
            no_replay = bool(getattr(exc, "no_replay", False))
            if no_replay:
                engine._emit(
                    f"{prefix} ⏳ {label}, backing off without replay in {delay:.1f}s"
                )
                time.sleep(delay)
                raise
            if attempt >= max_retries:
                raise
            last_exc = exc
            engine._emit(
                f"{prefix} ⏳ {label}, retry {attempt + 1}/{max_retries} in {delay:.1f}s"
            )
            time.sleep(delay)
    raise last_exc


class RetryingSession:
    """Retry safe HTTP requests and surface rate-limited responses to policy."""

    def __init__(self, session, engine, prefix: str):
        self._session = session
        self._engine = engine
        self._prefix = prefix

    def _call(self, method: str, *args, **kwargs):
        def operation():
            response = getattr(self._session, method)(*args, **kwargs)
            code = status_code(response)
            if code in _RETRYABLE_STATUS or (code is not None and code >= 500):
                if method == "post" and code not in {425, 429}:
                    raise RetryableResponseError(response, no_replay=True)
                try:
                    response.raise_for_status()
                except Exception:
                    raise
                raise RetryableResponseError(response)
            return response

        return request_with_retry(self._engine, operation, self._prefix)

    def get(self, *args, **kwargs):
        return self._call("get", *args, **kwargs)

    def post(self, *args, **kwargs):
        return self._call("post", *args, **kwargs)
