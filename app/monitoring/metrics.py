"""Small in-process Prometheus exposition for HTTP and provider usage."""

from __future__ import annotations

import threading
from collections import Counter
from typing import Any

_LOCK = threading.Lock()
_HTTP_REQUESTS: Counter[tuple[str, str, int]] = Counter()
_HTTP_DURATION_COUNT: Counter[str] = Counter()
_HTTP_DURATION_SUM: Counter[str] = Counter()
_HTTP_DURATION_BUCKETS: Counter[tuple[str, float]] = Counter()
_LLM_CALLS: Counter[tuple[str, str]] = Counter()
_LLM_ERRORS: Counter[tuple[str, str]] = Counter()
_LLM_TOKENS: Counter[tuple[str, str, str]] = Counter()
_DURATION_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, float("inf"))


def record_http_request(method: str, route: str, status: int, duration: float) -> None:
    route = route if route.startswith("/") else "unmatched"
    with _LOCK:
        _HTTP_REQUESTS[(method, route, status)] += 1
        _HTTP_DURATION_COUNT[route] += 1
        _HTTP_DURATION_SUM[route] += duration
        for bound in _DURATION_BUCKETS:
            if duration <= bound:
                _HTTP_DURATION_BUCKETS[(route, bound)] += 1


def record_llm_response(feature: str, provider: str, response: Any) -> None:
    """Count an upstream model request and any token usage the SDK returned."""
    usage = getattr(response, "usage_metadata", None)
    if usage is None:
        usage = getattr(response, "usage", None)
        if hasattr(usage, "model_dump"):
            usage = usage.model_dump()
    if not isinstance(usage, dict):
        metadata = getattr(response, "response_metadata", None)
        usage = metadata.get("token_usage") if isinstance(metadata, dict) else None
    usage = usage if isinstance(usage, dict) else {}
    input_tokens = usage.get("input_tokens", usage.get("prompt_tokens", 0))
    output_tokens = usage.get("output_tokens", usage.get("completion_tokens", 0))
    try:
        input_tokens = max(0, int(input_tokens or 0))
        output_tokens = max(0, int(output_tokens or 0))
    except (TypeError, ValueError):
        input_tokens = output_tokens = 0
    with _LOCK:
        _LLM_CALLS[(provider, feature)] += 1
        _LLM_TOKENS[(provider, feature, "input")] += input_tokens
        _LLM_TOKENS[(provider, feature, "output")] += output_tokens


def record_llm_error(feature: str, provider: str) -> None:
    with _LOCK:
        _LLM_ERRORS[(provider, feature)] += 1


def _label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")


def render_metrics() -> str:
    """Return a text/plain Prometheus snapshot; never labels metrics by user/query."""
    lines = [
        "# HELP rag_http_requests_total Total HTTP requests.",
        "# TYPE rag_http_requests_total counter",
    ]
    with _LOCK:
        http_rows = list(_HTTP_REQUESTS.items())
        duration_counts = list(_HTTP_DURATION_COUNT.items())
        duration_sums = list(_HTTP_DURATION_SUM.items())
        duration_buckets = list(_HTTP_DURATION_BUCKETS.items())
        call_rows = list(_LLM_CALLS.items())
        error_rows = list(_LLM_ERRORS.items())
        token_rows = list(_LLM_TOKENS.items())
    for (method, route, status), count in sorted(http_rows):
        lines.append(
            f'rag_http_requests_total{{method="{_label(method)}",route="{_label(route)}",status="{status}"}} {count}'
        )
    lines += [
        "# HELP rag_http_request_duration_seconds Request duration in seconds.",
        "# TYPE rag_http_request_duration_seconds histogram",
    ]
    bucket_lookup = dict(duration_buckets)
    for route, _count in sorted(duration_counts):
        for bound in _DURATION_BUCKETS:
            value = bucket_lookup.get((route, bound), 0)
            bound_text = "+Inf" if bound == float("inf") else str(bound)
            lines.append(
                f'rag_http_request_duration_seconds_bucket{{route="{_label(route)}",le="{bound_text}"}} {value}'
            )
        lines.append(
            f'rag_http_request_duration_seconds_count{{route="{_label(route)}"}} {_count}'
        )
        lines.append(
            f'rag_http_request_duration_seconds_sum{{route="{_label(route)}"}} {dict(duration_sums).get(route, 0.0):.6f}'
        )
    lines += [
        "# HELP rag_llm_calls_total Upstream model calls by provider and feature.",
        "# TYPE rag_llm_calls_total counter",
    ]
    for (provider, feature), count in sorted(call_rows):
        lines.append(
            f'rag_llm_calls_total{{provider="{_label(provider)}",feature="{_label(feature)}"}} {count}'
        )
    lines += [
        "# HELP rag_llm_errors_total Failed upstream model calls by provider and feature.",
        "# TYPE rag_llm_errors_total counter",
    ]
    for (provider, feature), count in sorted(error_rows):
        lines.append(
            f'rag_llm_errors_total{{provider="{_label(provider)}",feature="{_label(feature)}"}} {count}'
        )
    lines += [
        "# HELP rag_llm_tokens_total Model token usage where reported by the provider.",
        "# TYPE rag_llm_tokens_total counter",
    ]
    for (provider, feature, direction), count in sorted(token_rows):
        lines.append(
            f'rag_llm_tokens_total{{provider="{_label(provider)}",feature="{_label(feature)}",direction="{direction}"}} {count}'
        )
    return "\n".join(lines) + "\n"
