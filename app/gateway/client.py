"""Central LLM provider selection for the RAG application.

Direct Groq is the default. Portkey is imported and initialized only when the
explicit ``LLM_PROVIDER=portkey`` path (or the legacy client export) is used.
"""
from functools import lru_cache
from typing import Any

from langchain_groq import ChatGroq

from app.config import settings


GATEWAY_CONFIG = {
    "strategy": {"mode": "fallback"},
    "cache": {"mode": "simple"},
    "retry": {"attempts": 2, "on_status_codes": [429, 503]},
    "targets": [
        {"override_params": {"model": f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}"}},
        {"override_params": {"model": f"@{settings.GROQ_SLUG_2}/{settings.GROQ_MODEL}"}},
    ],
}


@lru_cache(maxsize=1)
def get_portkey_client():
    """Create the legacy Portkey client lazily, only when explicitly needed."""
    from portkey_ai import Portkey

    if not settings.PORTKEY_API_KEY:
        raise RuntimeError("LLM_PROVIDER=portkey requires PORTKEY_API_KEY.")
    return Portkey(
        api_key=settings.PORTKEY_API_KEY,
        config=settings.PORTKEY_CONFIG_ID or GATEWAY_CONFIG,
    )


def get_langchain_llm(feature: str = "rag"):
    """Return the configured LLM with LangChain's ``invoke`` interface.

    The same provider is shared by the planner and responder so there is one
    generation path and no implicit fallback from Groq to Portkey.
    """
    provider = settings.LLM_PROVIDER
    if provider == "groq":
        if not settings.GROQ_API_KEY:
            raise RuntimeError("LLM_PROVIDER=groq requires GROQ_API_KEY.")
        return ChatGroq(
            api_key=settings.GROQ_API_KEY,
            model=settings.GROQ_MODEL,
            temperature=0.1 if feature == "responder" else 0,
            max_tokens=settings.GROQ_MAX_TOKENS,
            reasoning_effort="low",
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
        )

    if provider == "portkey":
        from langchain_openai import ChatOpenAI
        from portkey_ai import createHeaders, PORTKEY_GATEWAY_URL

        if not settings.PORTKEY_API_KEY:
            raise RuntimeError("LLM_PROVIDER=portkey requires PORTKEY_API_KEY.")
        return ChatOpenAI(
            api_key=settings.PORTKEY_API_KEY,
            base_url=PORTKEY_GATEWAY_URL,
            model=f"@{settings.GROQ_SLUG}/{settings.GROQ_MODEL}",
            temperature=0,
            timeout=settings.LLM_TIMEOUT_SECONDS,
            max_retries=settings.LLM_MAX_RETRIES,
            default_headers=createHeaders(
                api_key=settings.PORTKEY_API_KEY,
                config=settings.PORTKEY_CONFIG_ID or GATEWAY_CONFIG,
                metadata={
                    "feature": feature,
                    "_user": "rag-system",
                    "environment": "production",
                },
            ),
        )

    raise ValueError(
        f"Unsupported LLM_PROVIDER {provider!r}; choose 'groq' or 'portkey'."
    )


def classify_provider_error(error: Exception) -> str:
    """Map provider exceptions to safe, actionable categories without logging payloads."""
    current: BaseException | None = error
    seen: set[int] = set()
    statuses: set[int] = set()
    names: list[str] = []
    messages: list[str] = []

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        names.append(type(current).__name__.lower())
        response = getattr(current, "response", None)
        for value in (
            getattr(current, "status_code", None),
            getattr(response, "status_code", None),
        ):
            if isinstance(value, int):
                statuses.add(value)
        # Use text only to classify; callers must never log this value.
        messages.append(str(current).lower())
        current = getattr(current, "__cause__", None) or getattr(current, "__context__", None)

    joined_names = " ".join(names)
    joined_messages = " ".join(messages)
    if 401 in statuses or "authentication" in joined_names or "invalid api key" in joined_messages:
        return "provider_authentication"
    if 403 in statuses or "permission" in joined_names:
        return "provider_permission"
    if 404 in statuses or "model_not_found" in joined_messages:
        return "provider_model_unavailable"
    if 429 in statuses or "rate limit" in joined_messages or "ratelimit" in joined_names:
        return "provider_rate_limit"
    if any("timeout" in name for name in names) or "timed out" in joined_messages:
        return "provider_timeout"
    if any(500 <= status <= 599 for status in statuses):
        return "provider_server_error"
    if 400 in statuses or 422 in statuses:
        return "provider_request_error"
    if "connection" in joined_names:
        return "provider_connection_error"
    return "provider_error"


def extract_cache_status(response: Any) -> str:
    """Extract a Portkey cache status when available; otherwise return MISS."""
    raw_candidates = [
        getattr(response, attr, None)
        for attr in ("_raw_response", "_response", "_http_response")
    ]
    metadata = getattr(response, "response_metadata", None)
    if isinstance(metadata, dict):
        raw_candidates.extend((metadata, metadata.get("headers")))

    for raw in raw_candidates:
        headers = raw if isinstance(raw, dict) else getattr(raw, "headers", None)
        if headers:
            status = headers.get("x-portkey-cache-status", "")
            if status:
                return str(status).upper()
    return "MISS"


def __getattr__(name: str):
    """Keep the former module-level export backwards compatible and lazy."""
    if name == "portkey_client":
        return get_portkey_client()
    raise AttributeError(name)
