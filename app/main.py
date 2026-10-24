# ============================================================
# CRITICAL: logfire MUST be configured before ALL other imports
# so that spans from all modules are captured from the start.
# ============================================================
import logfire
import os
import hmac
import logging
import math
import re
import uuid
from dotenv import load_dotenv

load_dotenv()
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
from fastapi import FastAPI, Response, HTTPException, Header
from app.agents.graph import rag_agent
from app.gateway.client import classify_provider_error
from app.guardrails import initialize_rails, guard

from pydantic import BaseModel, StringConstraints
from typing import Annotated, Optional


# Initialize FastAPI
app = FastAPI(title="Enterprise Agentic RAG API", version="1.0.0")


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
    if any(term in message for term in ("qdrant", "collection not found", "vector store")):
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


@app.on_event("startup")
def startup_event():
    # NeMo logs full prompts and user messages at INFO; keep those private.
    logging.getLogger("nemoguardrails").setLevel(logging.WARNING)
    initialize_rails()

class QueryRequest(BaseModel):
    q: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=8000)]
    thread_id: Annotated[str, StringConstraints(min_length=1, max_length=128)] = "default_user"
    
    
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
        return {"status": "ready", "checks": {"qdrant": "ok", "collection": "ready"}}
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
        png_bytes = rag_agent.get_graph().draw_mermaid_png()
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

    # Optional shared bearer token for deployments behind a trusted ingress.
    # Leave unset for local development; use user/session auth for public apps.
    expected_token = os.getenv("API_AUTH_TOKEN")
    if expected_token:
        provided = authorization[7:] if authorization and authorization.lower().startswith("bearer ") else ""
        if not hmac.compare_digest(provided, expected_token):
            raise HTTPException(status_code=401, detail="Invalid or missing bearer token.")

    initial_state = {
        "messages": [{"role": "user", "content": q}],
        "current_query": q,
        "documents": [],
        "plan": ["Start"],
        "status": "Initializing Graph..."
    }
    
    # Configuration for Memory (Thread ID)
    config = {"configurable": {"thread_id": thread_id}}
    
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
                "sources": []
            }

        # Gate 2: LangGraph RAG pipeline
        # Run the graph synchronously to preserve Logfire context variables
        final_output = rag_agent.invoke(initial_state, config=config)
        
        return {
            "question": q,
            "answer": final_output.get("final_answer"),
            "thought_process": final_output.get("plan"),
            "status": final_output.get("status"),
            "sources": final_output.get("documents", [])
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
        raise HTTPException(
            status_code=503,
            detail={"code": category, "type": deepest_type, "message": "The RAG service could not complete the request.", "reference": reference},
            headers=headers,
        ) from e
