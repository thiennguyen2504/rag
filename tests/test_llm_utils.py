"""Unit tests for LLM utilities: retry logic, usage extraction, and cost computation."""

import pytest
from unittest.mock import MagicMock
from src.llm_utils import (
    compute_cost,
    extract_usage_metadata,
    is_retryable_error,
    parse_retry_delay,
    retry_with_backoff,
)


def test_is_retryable_error():
    """Verify detection of retryable errors (429, 5xx, rate limits, timeouts)."""
    assert is_retryable_error(Exception("HTTP 429 Too Many Requests"))
    assert is_retryable_error(Exception("ResourceExhausted: quota exceeded"))
    assert is_retryable_error(Exception("503 Service Unavailable"))
    assert is_retryable_error(Exception("Connection timed out"))
    assert not is_retryable_error(ValueError("Invalid argument provided"))
    assert not is_retryable_error(KeyError("missing_key"))


def test_parse_retry_delay():
    """Verify parsing retry delays from Google error messages."""
    exc1 = Exception("Please retry in 12.5s due to rate limit")
    assert parse_retry_delay(exc1) == 13.5

    exc2 = Exception("Error payload: {'retryDelay': '30s'}")
    assert parse_retry_delay(exc2) == 31.0

    exc3 = Exception("Regular error without retry delay")
    assert parse_retry_delay(exc3) is None


def test_retry_with_backoff_success_after_failure():
    """Verify decorator retries retryable errors and returns on success."""
    attempts = [0]

    @retry_with_backoff(max_retries=3, initial_delay=0.01, jitter=False)
    def flaky_func():
        attempts[0] += 1
        if attempts[0] < 3:
            raise Exception("429 Resource Exhausted")
        return "SUCCESS"

    result = flaky_func()
    assert result == "SUCCESS"
    assert attempts[0] == 3


def test_retry_with_backoff_fatal_error():
    """Verify decorator immediately raises non-retryable errors without retrying."""
    attempts = [0]

    @retry_with_backoff(max_retries=3, initial_delay=0.01, jitter=False)
    def fatal_func():
        attempts[0] += 1
        raise ValueError("Non-retryable client error")

    with pytest.raises(ValueError, match="Non-retryable"):
        fatal_func()
    assert attempts[0] == 1


def test_extract_usage_metadata():
    """Verify extracting token usage metadata from AIMessage or fallback."""
    msg = MagicMock()
    msg.usage_metadata = {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}

    usage = extract_usage_metadata(msg)
    assert usage == {"input_tokens": 120, "output_tokens": 45, "total_tokens": 165}

    # Fallback when usage_metadata is missing
    msg_empty = MagicMock(spec=[])
    assert extract_usage_metadata(msg_empty) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    assert extract_usage_metadata(None) == {
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }


def test_compute_cost():
    """Verify calculation of cost based on models config."""
    models_cfg = {
        "pricing_usd_per_1m_tokens": {
            "gemini-2.5-flash": {"input": 0.15, "output": 0.60}
        }
    }

    # 10,000 input tokens = 10,000 * 0.15 / 1M = 0.0015
    # 5,000 output tokens = 5,000 * 0.60 / 1M = 0.0030
    # Total = 0.0045 USD
    cost = compute_cost("gemini-2.5-flash", 10_000, 5_000, models_cfg)
    assert cost == 0.0045

    # Missing model returns 0.0
    cost_unknown = compute_cost("unknown-model", 1000, 1000, models_cfg)
    assert cost_unknown == 0.0

