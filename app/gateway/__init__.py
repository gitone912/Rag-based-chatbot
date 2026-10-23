from app.gateway.client import get_langchain_llm, get_portkey_client, extract_cache_status


def __getattr__(name: str):
    # Preserve the legacy import while avoiding Portkey initialization unless asked.
    if name == "portkey_client":
        return get_portkey_client()
    raise AttributeError(name)


__all__ = ["get_langchain_llm", "get_portkey_client", "extract_cache_status", "portkey_client"]
