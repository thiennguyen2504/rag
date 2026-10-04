"""Utility functions for LLM calls: retry, usage extraction, and cost computation."""

import functools
import logging
import random
import re
import time
from typing import Any, Callable, Dict, Optional, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


def is_retryable_error(exc: Exception) -> bool:
    """Check whether an exception should be retried (429, 5xx, timeout)."""
    exc_str = str(exc).lower()

    # Status codes check
    for code in RETRYABLE_STATUS_CODES:
        if str(code) in exc_str:
            return True

    # Common keyword checks
    retry_keywords = [
        "rate limit",
        "quota",
        "resource exhausted",
        "resource_exhausted",
        "too many requests",
        "timeout",
        "timed out",
        "connection reset",
        "connection error",
        "temporarily unavailable",
        "service unavailable",
        "internal server error",
        "bad gateway",
        "gateway timeout",
        "503",
        "500",
        "429",
    ]
    if any(kw in exc_str for kw in retry_keywords):
        return True

    # Check for HTTP status code attributes if present
    status_code = getattr(exc, "status_code", None) or getattr(exc, "code", None)
    if status_code and status_code in RETRYABLE_STATUS_CODES:
        return True

    return False


def parse_retry_delay(exc: Exception) -> Optional[float]:
    """Parse Google API suggested retry delay from error message."""
    exc_str = str(exc)

    # Check for "retry in 32.74s" or similar
    match = re.search(r"retry in ([\d\.]+)s", exc_str, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1)) + 1.0
        except ValueError:
            pass

    # Check for "retryDelay': '32s'" or similar
    match_delay = re.search(r"retryDelay['\"]?\s*:\s*['\"]?(\d+)s?", exc_str, re.IGNORECASE)
    if match_delay:
        try:
            return float(match_delay.group(1)) + 1.0
        except ValueError:
            pass

    return None


def retry_with_backoff(
    max_retries: int = 8,
    initial_delay: float = 2.0,
    max_delay: float = 65.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
):
    """Decorator for functions making API calls to Gemini.

    Applies exponential backoff with jitter on retryable HTTP errors (429, 5xx, timeouts).
    Automatically respects server-provided retry delays.
    """

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempt = 0
            delay = initial_delay

            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:
                    attempt += 1
                    if attempt > max_retries or not is_retryable_error(exc):
                        logger.error(
                            "Lệnh gọi %s thất bại sau %d lần thử hoặc lỗi không thể retry: %s",
                            func.__name__,
                            attempt,
                            exc,
                        )
                        raise

                    # Check if server asked for a specific wait time
                    suggested_delay = parse_retry_delay(exc)
                    if suggested_delay is not None:
                        sleep_time = min(suggested_delay, max_delay)
                    else:
                        # Exponential backoff with jitter
                        current_delay = min(delay, max_delay)
                        if jitter:
                            sleep_time = random.uniform(0.5 * current_delay, current_delay)
                        else:
                            sleep_time = current_delay
                        delay *= backoff_factor

                    logger.warning(
                        "Lỗi gọi API ở %s (%s). Thử lại lần %d/%d sau %.2f giây...",
                        func.__name__,
                        str(exc)[:120],
                        attempt,
                        max_retries,
                        sleep_time,
                    )
                    time.sleep(sleep_time)

        return wrapper

    return decorator


def extract_usage_metadata(message: Any) -> Dict[str, int]:
    """Extract token usage from AIMessage.usage_metadata with fallback to 0.

    Args:
        message: LangChain AIMessage or object containing usage_metadata.

    Returns:
        Dict with keys: input_tokens, output_tokens, total_tokens.
    """
    default_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    if message is None:
        return default_usage

    usage = getattr(message, "usage_metadata", None)
    if isinstance(usage, dict):
        in_tok = int(usage.get("input_tokens", 0) or 0)
        out_tok = int(usage.get("output_tokens", 0) or 0)
        tot_tok = int(usage.get("total_tokens", 0) or (in_tok + out_tok))
        return {
            "input_tokens": in_tok,
            "output_tokens": out_tok,
            "total_tokens": tot_tok,
        }

    return default_usage


def compute_cost(
    model: str,
    input_tokens: int,
    output_tokens: int,
    models_cfg: Any,
) -> float:
    """Compute USD cost based on pricing in models_cfg.

    If model pricing is missing, logs warning and returns 0.0.

    Args:
        model: Model name string (e.g. 'gemini-2.5-flash').
        input_tokens: Number of prompt/input tokens.
        output_tokens: Number of completion/output tokens.
        models_cfg: ModelsConfig or dictionary containing pricing_usd_per_1m_tokens.

    Returns:
        Calculated cost in USD (rounded to 6 decimal places).
    """
    if hasattr(models_cfg, "pricing_usd_per_1m_tokens"):
        pricing_table = models_cfg.pricing_usd_per_1m_tokens
    elif isinstance(models_cfg, dict):
        pricing_table = models_cfg.get("pricing_usd_per_1m_tokens", {})
    else:
        pricing_table = {}

    # Pricing might be keyed directly or without prefix
    model_pricing = pricing_table.get(model)
    if not model_pricing:
        clean_model = model.replace("models/", "")
        model_pricing = pricing_table.get(clean_model)

    if not model_pricing:
        logger.warning(
            "Không tìm thấy bảng giá cho model '%s' trong pricing_usd_per_1m_tokens. Chi phí trả về 0.0.",
            model,
        )
        return 0.0

    if hasattr(model_pricing, "input"):
        in_rate = getattr(model_pricing, "input", 0.0)
        out_rate = getattr(model_pricing, "output", 0.0)
    elif isinstance(model_pricing, dict):
        in_rate = model_pricing.get("input", 0.0)
        out_rate = model_pricing.get("output", 0.0)
    else:
        in_rate = 0.0
        out_rate = 0.0

    cost = (input_tokens * in_rate + output_tokens * out_rate) / 1_000_000.0
    return round(cost, 6)
