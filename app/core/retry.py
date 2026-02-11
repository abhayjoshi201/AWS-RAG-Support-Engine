"""Reusable retry decorator with exponential backoff + jitter."""

from __future__ import annotations

from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from app.core.logging import get_logger

_log = get_logger(__name__)


def _log_retry(retry_state) -> None:
    _log.warning(
        "retrying_call",
        attempt=retry_state.attempt_number,
        wait=round(retry_state.next_action.sleep, 2) if retry_state.next_action else 0,
        fn=getattr(retry_state.fn, "__qualname__", str(retry_state.fn)),
        error=str(retry_state.outcome.exception()) if retry_state.outcome and retry_state.outcome.failed else None,
    )


# General-purpose retry for AWS / HTTP calls
bedrock_retry = retry(
    reraise=True,
    stop=stop_after_attempt(4),
    wait=wait_exponential_jitter(initial=1, max=30, jitter=2),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=_log_retry,
)

opensearch_retry = retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential_jitter(initial=0.5, max=20, jitter=1),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=_log_retry,
)

http_retry = retry(
    reraise=True,
    stop=stop_after_attempt(3),
    wait=wait_exponential_jitter(initial=0.5, max=10, jitter=1),
    retry=retry_if_exception_type((Exception,)),
    before_sleep=_log_retry,
)
