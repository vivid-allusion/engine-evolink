"""Stub tests for retry limits and provider Retry-After handling."""

import sys
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine_evolink import Engine
from engine_evolink.retry_policy import RetryingSession, request_with_retry


def test_numeric_and_http_date_retry_after_are_honored(tmp_path):
    http_date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30))
    for value, lower, upper in (("0.5", 0.5, None), (http_date, 20, 31)):
        engine = Engine({"retry_on_429": 1, "retry_base_delay": 0}, tmp_path)
        response = SimpleNamespace(
            status_code=429, headers={"Retry-After": value}
        )
        error = RuntimeError("429 Too Many Requests")
        error.response = response
        operation = MagicMock(side_effect=[error, "ok"])
        with patch("engine_evolink.retry_policy.time.sleep") as sleep, patch(
            "engine_evolink.retry_policy.random.uniform", return_value=0
        ):
            assert request_with_retry(engine, operation, "[1/1] input.md") == "ok"
        assert operation.call_count == 2
        assert sleep.call_count == 1
        delay = sleep.call_args.args[0]
        assert delay >= lower
        if upper is not None:
            assert delay <= upper


def test_ambiguous_post_failure_is_not_replayed(tmp_path):
    engine = Engine(
        {"endpoint": "test/model", "retry_on_429": 4, "retry_base_delay": 0},
        tmp_path,
    )

    class Response:
        status_code = 500
        headers = {"Retry-After": "0.5"}

    session = MagicMock()
    session.post.return_value = Response()
    wrapped = RetryingSession(session, engine, "[1/1] input.md")
    with patch("engine_evolink.retry_policy.time.sleep") as sleep, patch(
        "engine_evolink.retry_policy.random.uniform", return_value=0
    ):
        try:
            wrapped.post("https://provider.invalid/generate")
        except Exception as exc:
            assert getattr(exc, "no_replay", False)
        else:
            raise AssertionError("ambiguous POST should fail without replay")
    assert session.post.call_count == 1
    assert sleep.call_count == 1
    assert sleep.call_args.args[0] >= 0.5
