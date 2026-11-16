"""
Phase 1 — Live Pipeline.
Calls the running FastAPI /query endpoint for each sample and captures answers,
contexts, routed tools, and end-to-end request latency.
"""


import time
import statistics
import copy
import json
import os
import re
import tempfile
import requests
import logfire

API_BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
API_URL = f"{API_BASE_URL}/query"
_API_TOKEN = os.getenv("EVAL_AUTH_TOKEN") or os.getenv("API_AUTH_TOKEN")
API_HEADERS = {"Authorization": f"Bearer {_API_TOKEN}"} if _API_TOKEN else {}
RESPONSE_TRUNCATE = 2000  # preserve complete concise answers for offline and RAGAS evaluation
DELAY_BETWEEN_CALLS = int(os.getenv("EVAL_DELAY_BETWEEN_CALLS", "20"))
REQUEST_TIMEOUT = 120      # seconds — guardrails + LangGraph + Groq can take >60s
MAX_QUERY_ATTEMPTS = int(os.getenv("EVAL_MAX_QUERY_ATTEMPTS", "3"))
RETRYABLE_ERROR_CODES = {
    "provider_rate_limit",
    "provider_server_error",
    "provider_timeout",
    "provider_connection_error",
    "dependency_unavailable",
}
_INLINE_CITATION = re.compile(r"(?<!\w)\[(\d+)\]")


def extract_inline_citation_ids(answer: str) -> list[int]:
    """Read answer-body citations, excluding the web fallback Sources footer."""
    answer_body = re.split(
        r"(?im)^\s*\*{0,2}sources\*{0,2}\s*:?.*$", answer or "", maxsplit=1
    )[0]
    return [int(value) for value in _INLINE_CITATION.findall(answer_body)]


def _context_citation_ids(contexts: list[str]) -> set[int]:
    ids = set()
    for context in contexts:
        if not isinstance(context, str):
            continue
        first_line = context.splitlines()[0] if context.splitlines() else ""
        match = re.match(r"^\[(\d+)\]\s+SOURCE:|^WEB SOURCE \[(\d+)\]", first_line)
        if match:
            ids.add(int(match.group(1) or match.group(2)))
    return ids


def _citation_matches_context(citation: dict, context: str) -> bool:
    lines = context.splitlines()
    if not lines:
        return False
    first_line = lines[0]
    if citation.get("source") == first_line:
        return True
    source_id = citation.get("source_id")
    source_version = citation.get("source_version")
    return bool(
        source_id
        and source_version
        and f"source_id={source_id}" in first_line
        and f"version={source_version}" in first_line
        and citation.get("source") in first_line
    )




def detect_tool(thought_process: list) -> str:
    """
    Maps the thought_process list from /query response to a tool name.
    Planner sets:  'Intent: Technical' + 'Search Term: ...' → retrieve_documents
                   'Intent: Conversational/Memory'           → direct_answer
    main.py sets:  'Intent: Guardrails Fired'                → guardrails
    """
    joined = " ".join(thought_process).lower()
    if "guardrails fired" in joined:
        return "guardrails"
    if "intent: technical" in joined or "search term:" in joined or "context retrieved" in joined:
        return "retrieve_documents"
    if "conversational" in joined or "memory" in joined:
        return "direct_answer"
    return "unknown"


def _save_checkpoint(dataset: dict, path: str) -> None:
    """Atomically persist query results so a long run can resume after interruption."""
    absolute_path = os.path.abspath(path)
    directory = os.path.dirname(absolute_path)
    os.makedirs(directory, exist_ok=True)
    handle, temporary_path = tempfile.mkstemp(
        dir=directory, prefix="eval_checkpoint_", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(dataset, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, absolute_path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


def run_pipeline(golden_dataset: dict, progress_callback=None, checkpoint_path=None) -> dict:
    """
    Enriches each rag_sample in golden_dataset with live API results.
    Returns a deep copy with API results and per-query end-to-end latency filled.
    progress_callback(i, total, question, stage, response="") is called per step.
    """
    dataset = copy.deepcopy(golden_dataset)
    if checkpoint_path and os.path.exists(checkpoint_path):
        with open(checkpoint_path, encoding="utf-8") as checkpoint_file:
            checkpoint = json.load(checkpoint_file)
        expected = [
            (sample.get("id"), sample.get("question"))
            for sample in dataset.get("rag_samples", [])
        ]
        existing = [
            (sample.get("id"), sample.get("question"))
            for sample in checkpoint.get("rag_samples", [])
        ]
        if expected != existing:
            raise ValueError(
                f"Checkpoint {checkpoint_path} belongs to a different question set."
            )
        expected_profile = (dataset.get("evaluation_manifest") or {}).get("profile_sha256")
        existing_profile = (checkpoint.get("evaluation_manifest") or {}).get("profile_sha256")
        if expected_profile != existing_profile:
            raise ValueError(
                f"Checkpoint {checkpoint_path} belongs to a different model, corpus, "
                "question set, or application version. Choose a new output path."
            )
        dataset = checkpoint
    samples = dataset["rag_samples"]
    n = len(samples)
    run_id = (dataset.get("evaluation_manifest") or {}).get("run_id", "local")

    with logfire.span("🚀 Eval Phase 1 — Live Pipeline", total_samples=n):
        for i, sample in enumerate(samples):
            question = sample["question"]
            if sample.get("actual_response") and sample.get("actual_error") is None:
                if progress_callback:
                    progress_callback(i, n, question, "done", sample["actual_response"])
                continue
            sample_started = time.perf_counter()
            provider_quota_exhausted = False

            if progress_callback:
                progress_callback(i, n, question, "calling")

            with logfire.span(
                f"📤 Live Query {i + 1}/{n}",
                question_characters=len(question),
                domain=sample.get("domain", ""),
            ):
                try:
                    for attempt in range(MAX_QUERY_ATTEMPTS):
                        try:
                            resp = requests.post(
                                API_URL,
                                json={
                                    "q": question,
                                    "thread_id": f"eval_{run_id}_{sample['id']}",
                                },
                                headers=API_HEADERS,
                                timeout=REQUEST_TIMEOUT,
                            )
                        except requests.exceptions.RequestException:
                            if attempt + 1 >= MAX_QUERY_ATTEMPTS:
                                raise
                            delay = 5 * (2 ** attempt)
                            logfire.warn(
                                "Retrying evaluation query after transport error",
                                attempt=attempt + 1,
                                delay_seconds=delay,
                            )
                            time.sleep(delay)
                            continue
                        if resp.ok:
                            break

                        try:
                            detail = resp.json().get("detail", {})
                        except (ValueError, AttributeError):
                            detail = {}
                        code = detail.get("code") if isinstance(detail, dict) else None
                        if code == "provider_daily_quota":
                            sample["actual_error"] = code
                            provider_quota_exhausted = True
                            resp.raise_for_status()
                        if attempt + 1 >= MAX_QUERY_ATTEMPTS or code not in RETRYABLE_ERROR_CODES:
                            sample["actual_error"] = code or f"http_{resp.status_code}"
                            resp.raise_for_status()

                        retry_after = resp.headers.get("Retry-After")
                        try:
                            delay = min(300, max(1, int(float(retry_after))))
                        except (TypeError, ValueError):
                            delay = 30 * (2 ** attempt)
                        if code == "provider_rate_limit":
                            # Groq enforces rolling token-per-minute budgets. The
                            # server's Retry-After can be shorter than the time
                            # needed for the rolling window to clear.
                            delay = max(delay, min(300, 60 * (2 ** attempt)))
                        logfire.warn(
                            "Retrying transient evaluation query",
                            error_code=code,
                            attempt=attempt + 1,
                            delay_seconds=delay,
                        )
                        time.sleep(delay)

                    data = resp.json()

                    raw_answer = data.get("answer") or ""
                    thought_process = data.get("thought_process") or []
                    sources = data.get("sources") or []
                    citations = data.get("citations") or []

                    sample["actual_response"] = raw_answer[:RESPONSE_TRUNCATE]
                    sample["actual_contexts"] = sources[:5]
                    sample["actual_citations"] = citations
                    sample["actual_tools_called"] = [detect_tool(thought_process)]
                    sample["actual_error"] = None if raw_answer.strip() else "empty_answer"

                    logfire.info(
                        "✅ Response captured",
                        tool=sample["actual_tools_called"][0],
                        response_chars=len(raw_answer),
                        context_chunks=len(sources),
                    )

                except requests.exceptions.ConnectionError:
                    logfire.error("❌ Cannot reach FastAPI — is the app running on :8000?")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_citations"] = []
                    sample["actual_tools_called"] = ["unknown"]
                    sample["actual_error"] = "backend_connection_error"

                except Exception as e:
                    logfire.error(f"❌ Query failed: {e}")
                    sample["actual_response"] = ""
                    sample["actual_contexts"] = sample.get("relevant_contexts", [])
                    sample["actual_citations"] = []
                    sample["actual_tools_called"] = ["unknown"]
                    if not sample.get("actual_error"):
                        sample["actual_error"] = type(e).__name__

            sample["actual_latency_ms"] = round(
                (time.perf_counter() - sample_started) * 1000, 1
            )
            if checkpoint_path:
                _save_checkpoint(dataset, checkpoint_path)

            if progress_callback:
                progress_callback(i, n, question, "done", sample["actual_response"])

            if provider_quota_exhausted:
                break

            if i < n - 1:
                time.sleep(DELAY_BETWEEN_CALLS)

    completed_latencies = [
        sample["actual_latency_ms"]
        for sample in samples
        if sample.get("actual_error") is None and sample.get("actual_latency_ms") is not None
    ]
    ordered_latencies = sorted(completed_latencies)
    successful_samples = [
        sample for sample in samples
        if sample.get("actual_error") is None
        and sample.get("actual_latency_ms") is not None
    ]
    inline_citation_count = 0
    valid_inline_citation_count = 0
    citation_metadata_records = []
    valid_metadata_count = 0
    for sample in successful_samples:
        contexts = sample.get("actual_contexts", [])
        context_ids = _context_citation_ids(contexts)
        inline_ids = extract_inline_citation_ids(sample.get("actual_response", ""))
        inline_citation_count += len(inline_ids)
        valid_inline_citation_count += sum(
            citation_id in context_ids for citation_id in inline_ids
        )
        for citation in sample.get("actual_citations", []):
            if not isinstance(citation, dict):
                continue
            citation_metadata_records.append(citation)
            citation_id = citation.get("id")
            if (
                isinstance(citation_id, int)
                and not isinstance(citation_id, bool)
                and citation_id in context_ids
                and any(
                    isinstance(context, str)
                    and _citation_matches_context(citation, context)
                    and citation_id in _context_citation_ids([context])
                    for context in contexts
                )
            ):
                valid_metadata_count += 1
    cited_successes = sum(
        bool(extract_inline_citation_ids(sample.get("actual_response", "")))
        for sample in successful_samples
    )
    p95_index = min(len(ordered_latencies) - 1, int(0.95 * len(ordered_latencies)))
    dataset["pipeline_metrics"] = {
        "status": (
            "partial_due_to_provider_daily_quota"
            if any(sample.get("actual_error") == "provider_daily_quota" for sample in samples)
            and any(sample.get("actual_latency_ms") is None for sample in samples)
            else "completed_with_errors"
            if any(sample.get("actual_error") is not None for sample in samples)
            else "complete"
        ),
        "questions": len(samples),
        "completed": len(completed_latencies),
        "errors": sum(sample.get("actual_error") is not None for sample in samples),
        "pending": sum(sample.get("actual_latency_ms") is None for sample in samples),
        "median_latency_ms": round(statistics.median(completed_latencies), 1)
        if completed_latencies else None,
        "p95_latency_ms": round(ordered_latencies[p95_index], 1)
        if ordered_latencies else None,
        "citation_presence_rate": round(cited_successes / len(successful_samples), 4)
        if successful_samples else None,
        "citation_id_validity_rate": round(
            valid_inline_citation_count / inline_citation_count, 4
        ) if inline_citation_count else None,
        "citation_metadata_source_match_rate": round(
            valid_metadata_count / len(citation_metadata_records), 4
        ) if citation_metadata_records else None,
    }

    if checkpoint_path:
        _save_checkpoint(dataset, checkpoint_path)
    return dataset


def save_results(dataset: dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def load_golden_dataset() -> dict:
    golden_path = os.path.join(os.path.dirname(__file__), "golden_dataset.json")
    with open(golden_path, encoding="utf-8") as f:
        dataset = json.load(f)

    eval_dir = os.path.dirname(golden_path)
    holdout_path = os.path.join(eval_dir, "retrieval_holdout.json")
    # Source-linked holdouts may contain private document excerpts and are
    # intentionally kept out of public repositories. Use them when available;
    # otherwise the public core golden set remains usable on its own.
    if os.path.exists(holdout_path):
        with open(holdout_path, encoding="utf-8") as f:
            holdout = json.load(f).get("samples", [])
    else:
        holdout = []

    for item in holdout:
        dataset["rag_samples"].append({
            "id": item["id"],
            "domain": item["domain"],
            "question": item["question"],
            "reference": item["reference"],
            "relevant_contexts": [item["evidence_context"]],
            "evidence_chunk": item["evidence_chunk"],
            "expected_tools": ["retrieve_documents"],
            "actual_response": "",
            "actual_contexts": [],
            "actual_citations": [],
            "actual_tools_called": [],
            "actual_error": None,
        })

    ids = [sample["id"] for sample in dataset["rag_samples"]]
    if len(ids) != len(set(ids)):
        raise ValueError("Evaluation question IDs must be unique across core and holdout sets.")

    return dataset
