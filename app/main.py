# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import hmac
import math
import re
import uuid
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
logfire_export = os.getenv("LOGFIRE_SEND_TO_LOGFIRE", "false").lower()
if logfire_export not in {"true", "false", "if-token-present"}:
    logfire_export = "if-token-present"
logfire_send = {"true": True, "false": False}.get(logfire_export, "if-token-present")
logfire_options = {
    "token": os.getenv("LOGFIRE_TOKEN"),
    "send_to_logfire": logfire_send,
    # Avoid UnicodeEncodeError on Windows cp1252 consoles; traces still export when enabled.
    "console": False,
}
logfire.configure(**logfire_options)

# Now safe to import app modules - logfire is already active
from fastapi import FastAPI, Response, HTTPException, Header, Request
from fastapi.responses import PlainTextResponse
from app.agents import graph as agent_graph
from app.gateway.client import classify_provider_error
from app.guardrails import initialize_rails, guard
from app.monitoring.metrics import record_http_request, render_metrics
from app.security.auth import authenticate_request, scoped_thread_id
from app.services.rate_limit import rate_limiter
from app.services.memory import open_memory_runtime
from app.config import settings

from pydantic import BaseModel, StringConstraints
from typing import Annotated, Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API", version="1.0.0")


@app.middleware("http")
async def observe_http_request(request: Request, call_next):
    started = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        route = getattr(request.scope.get("route"), "path", "unmatched")
        record_http_request(
            request.method,
            route,
            status,
            time.perf_counter() - started,
        )


@app.get("/metrics", response_class=PlainTextResponse, include_in_schema=False)
def metrics(authorization: Optional[str] = Header(default=None)):
    if settings.METRICS_BEARER_TOKEN:
        provided = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(provided, settings.METRICS_BEARER_TOKEN):
            raise HTTPException(status_code=401, detail="Metrics authorization required.")
    return render_metrics()


def _safe_error_category(error: Exception) -> str:
    """Classify upstream failures without logging provider payloads or secrets."""
    chain = []
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.__cause__ or current.__context__
    message = " ".join(str(item) for item in chain).lower()
    statuses = {getattr(item, "status_code", None) for item in chain}
    if any(term in message for term in ("api_key_invalid", "api key not valid", "invalid api key")):
        return "provider_authentication"
    if any(term in message for term in ("per day", "daily token quota", "daily quota", "tpd limit")):
        return "provider_daily_quota"
    if any(any(term in str(getattr(item, "__module__", "")).lower() for term in ("groq", "openai")) for item in chain):
        return classify_provider_error(error)
    if 401 in statuses:
        return "provider_authentication"
    if 403 in statuses:
        return "provider_permission"
    if 404 in statuses:
        return "provider_model_unavailable"
    if 429 in statuses:
        return "provider_rate_limit"
    if any(term in message for term in ("invalid api key", "unauthorized", "401")):
        return "provider_authentication"
    if "429" in message or "rate limit" in message or "quota" in message:
        return "provider_rate_limit"
    if any(term in message for term in ("qdrant", "collection not found", "vector store", "vector retrieval")):
        return "vector_store"
    if any(term in message for term in ("timeout", "name resolution", "connection error", "connecterror")):
        return "dependency_unavailable"
    if any(isinstance(status, int) and 500 <= status <= 599 for status in statuses):
        return "upstream_server_error"
    return "upstream_failure"


def _retry_after_seconds(error: Exception) -> int | None:
    """Preserve provider retry guidance without returning provider error details."""
    current = error
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        response = getattr(current, "response", None)
        headers = getattr(response, "headers", None)
        if headers:
            value = headers.get("retry-after")
            if value:
                try:
                    return max(1, math.ceil(float(value)))
                except (TypeError, ValueError):
                    pass
        match = re.search(r"try again in (?:(\d+)m)?\s*(\d+(?:\.\d+)?)s", str(current), re.I)
        if match:
            minutes = int(match.group(1) or 0)
            seconds = float(match.group(2))
            return max(1, math.ceil(minutes * 60 + seconds))
        current = current.__cause__ or current.__context__
    return None


def _validate_production_configuration() -> None:
    """Fail closed when production mode is selected with dev-only settings."""
    if settings.APP_ENV not in {"development", "test", "production"}:
        raise RuntimeError("APP_ENV must be development, test, or production.")
    if settings.APP_ENV != "production":
        return

    if settings.AUTH_MODE != "cognito":
        raise RuntimeError("APP_ENV=production requires AUTH_MODE=cognito.")
    if not settings.COGNITO_ISSUER.startswith("https://") or not settings.COGNITO_CLIENT_ID:
        raise RuntimeError("Production requires an HTTPS Cognito issuer and client ID.")
    if not settings.COGNITO_REQUIRED_GROUP:
        raise RuntimeError("Production requires COGNITO_REQUIRED_GROUP for application authorization.")
    if not settings.METRICS_BEARER_TOKEN or len(settings.METRICS_BEARER_TOKEN) < 32:
        raise RuntimeError("Production requires a metrics bearer token of at least 32 characters.")
    if not settings.QDRANT_URL or not settings.QDRANT_API_KEY:
        raise RuntimeError("Production requires a hosted Qdrant URL and API key.")
    qdrant_url = urlparse(settings.QDRANT_URL)
    if qdrant_url.scheme != "https":
        raise RuntimeError("Production Qdrant connections must use HTTPS.")
    if not settings.CHECKPOINT_DATABASE_URL:
        raise RuntimeError("Production requires CHECKPOINT_DATABASE_URL.")
    database_url = urlparse(settings.CHECKPOINT_DATABASE_URL)
    database_options = parse_qs(database_url.query)
    database_tls = database_options.get("sslmode", [""])[0]
    database_ca = database_options.get("sslrootcert", [""])[0]
    if database_tls != "verify-full" or not database_ca or not Path(database_ca).is_file():
        raise RuntimeError(
            "Production Postgres connections must set sslmode=verify-full and a valid sslrootcert path."
        )
    redis_url = urlparse(settings.RATE_LIMIT_REDIS_URL or "")
    if redis_url.scheme != "rediss":
        raise RuntimeError("Production Redis connections must use rediss:// TLS.")
    if settings.LLM_PROVIDER == "groq" and not settings.GROQ_API_KEY:
        raise RuntimeError("Production Groq provider requires GROQ_API_KEY.")
    if settings.LLM_PROVIDER == "gemini" and not settings.GEMINI_API_KEY:
        raise RuntimeError("Production Gemini provider requires GEMINI_API_KEY.")
    if settings.LLM_PROVIDER == "portkey" and (
        not settings.PORTKEY_API_KEY or not settings.PORTKEY_CONFIG_ID
    ):
        raise RuntimeError("Production Portkey provider requires API key and saved config ID.")


@app.on_event("startup")
def startup_event():
    _validate_production_configuration()
    if settings.AUTH_MODE not in {"none", "shared_token", "cognito"}:
        raise RuntimeError("AUTH_MODE must be none, shared_token, or cognito.")
    if settings.AUTH_MODE == "shared_token" and not settings.API_AUTH_TOKEN:
        raise RuntimeError("shared_token mode requires API_AUTH_TOKEN.")
    if settings.AUTH_MODE == "cognito" and not settings.METRICS_BEARER_TOKEN:
        raise RuntimeError("Cognito mode requires METRICS_BEARER_TOKEN to protect /metrics.")
    rate_limiter.initialize()
    memory_runtime = open_memory_runtime()
    if memory_runtime is not None:
        agent_graph.configure_checkpointer(memory_runtime.checkpointer)
    app.state.memory_runtime = memory_runtime
    initialize_rails()


@app.on_event("shutdown")
def shutdown_event():
    memory_runtime = getattr(app.state, "memory_runtime", None)
    if memory_runtime is not None:
        memory_runtime.close()

class QueryRequest(BaseModel):
    q: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    thread_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] = "default_user"


class MemoryDeleteRequest(BaseModel):
    thread_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] | None = None
    
    
@app.get("/")
def home():
    return {"message": "Enterprise LangGraph RAG API is live.", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "checks": {"api": "ok"}}


@app.get("/ready")
def readiness():
    """Readiness probe includes the vector store dependency."""
    try:
        from app.services.retrieval.qdrant_service import client
        from app.config import settings
        names = {item.name for item in client.get_collections().collections}
        if settings.QDRANT_COLLECTION not in names:
            raise HTTPException(status_code=503, detail="The knowledge collection is not initialized; run ingestion first.")
        checks = {"qdrant": "ok", "collection": "ready"}
        rate_limiter.check_ready()
        checks["rate_limiter"] = "ok"
        memory_runtime = getattr(app.state, "memory_runtime", None)
        if memory_runtime is not None:
            with memory_runtime.pool.connection() as connection:
                connection.execute("SELECT 1")
            checks["memory"] = "postgres"
        elif settings.AUTH_MODE == "cognito":
            raise HTTPException(status_code=503, detail="Durable memory is required in Cognito mode.")
        else:
            checks["memory"] = "process_local"
        return {"status": "ready", "checks": checks}
    except HTTPException:
        raise
    except Exception as exc:
        logfire.warning(f"Readiness check failed: {type(exc).__name__}")
        raise HTTPException(status_code=503, detail="Required vector store is unavailable.") from exc


@app.get("/graph")
def get_graph_image():
    """
    Returns the Mermaid image of the agent's workflow.
    """
    try:
        png_bytes = agent_graph.rag_agent.get_graph().draw_mermaid_png()
        return Response(content=png_bytes, media_type="image/png")
    except Exception as e:
        return {"error": f"Could not generate graph image: {e}"}
    
    
@app.post("/query")
def query(request: QueryRequest, authorization: Optional[str] = Header(default=None)):
    """
    Executes the LangGraph RAG flow with memory using a POST request.
    """
    q = request.q
    thread_id = request.thread_id

    identity = authenticate_request(authorization)
    rate_limiter.enforce(identity.subject)

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "citation_ids": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    scoped_id = (
        thread_id
        if identity.provider == "local"
        else scoped_thread_id(identity, thread_id)
    )
    memory_runtime = getattr(app.state, "memory_runtime", None)
    if memory_runtime is not None:
        memory_runtime.retention.touch(identity.subject, scoped_id)
    config = {"configurable": {"thread_id": scoped_id}}
    
    try:
        # Gate 1: NeMo Guardrails — blocks off-topic, jailbreaks, and handles dialog
        rail_fired, rail_response = guard(q)
        if rail_fired:
            logfire.info(f"🛡️ Request blocked by guardrails | thread={thread_id}")
            return {
                "question": q,
                "answer": rail_response,
                "thought_process": ["Intent: Guardrails Fired", "Retrieval: Skipped"],
                "status": "Blocked by guardrails.",
                "sources": [],
                "citations": [],
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = agent_graph.rag_agent.invoke(initial_state, config=config)
        
        documents = final_output.get("documents", [])
        citation_ids = final_output.get("citation_ids", [])
        citations = []
        for citation_id in citation_ids:
            if (
                not isinstance(citation_id, int)
                or isinstance(citation_id, bool)
                or not 1 <= citation_id <= len(documents)
                or not isinstance(documents[citation_id - 1], str)
            ):
                continue
            source_lines = documents[citation_id - 1].splitlines()
            if source_lines:
                first_line = source_lines[0]
                metadata = {"id": citation_id, "source": first_line}
                if " | source_id=" in first_line:
                    source_text = first_line.split("SOURCE:", 1)[-1]
                    parts = [part.strip() for part in source_text.split("|")]
                    metadata["source"] = parts[0]
                    for part in parts[1:]:
                        key, separator, value = part.partition("=")
                        if separator and key in {
                            "source_id", "version", "chunk", "ingested_at"
                        }:
                            metadata[key] = value
                    if "chunk" in metadata:
                        try:
                            metadata["chunk_index"] = int(metadata.pop("chunk"))
                        except ValueError:
                            metadata.pop("chunk", None)
                    if "version" in metadata:
                        metadata["source_version"] = metadata.pop("version")
                    if "ingested_at" in metadata:
                        metadata["ingested_at"] = metadata["ingested_at"]
                citations.append(metadata)
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": documents,
            "citations": citations,
        }
    except Exception as e:
        category = _safe_error_category(e)
        reference = uuid.uuid4().hex[:12]
        deepest_type = type(e).__name__
        cause = e
        while cause.__cause__ is not None:
            cause = cause.__cause__
            deepest_type = type(cause).__name__
        logfire.error(f"Backend request failed | category={category} | type={deepest_type} | ref={reference}")
        headers = {}
        if category == "provider_rate_limit":
            retry_after = _retry_after_seconds(e)
            if retry_after is not None:
                headers["Retry-After"] = str(retry_after)
        public_message = (
            "The configured LLM provider's daily token quota is exhausted."
            if category == "provider_daily_quota"
            else "The RAG service could not complete the request."
        )
        raise HTTPException(
            status_code=503,
            detail={"code": category, "message": public_message, "reference": reference},
            headers=headers,
        ) from e


@app.delete("/memory", status_code=200)
def delete_my_memory(
    request: MemoryDeleteRequest,
    authorization: Optional[str] = Header(default=None),
):
    """Erase all durable conversation checkpoints owned by the authenticated user."""
    identity = authenticate_request(authorization)
    memory_runtime = getattr(app.state, "memory_runtime", None)
    if memory_runtime is None:
        if not request.thread_id:
            raise HTTPException(status_code=400, detail="thread_id is required for local memory deletion.")
        scoped_id = (
            request.thread_id
            if identity.provider == "local"
            else scoped_thread_id(identity, request.thread_id)
        )
        agent_graph.delete_local_thread(scoped_id)
        return {"status": "deleted", "threads_deleted": 1}
    deleted = memory_runtime.retention.delete_identity(identity.subject)
    return {"status": "deleted", "threads_deleted": deleted}
