"""Run or resume the citation-aware live API evaluation."""

from __future__ import annotations

import json
import hashlib
import os
import sys
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

import logfire  # noqa: E402

logfire.configure(send_to_logfire=False, console=False)

from app.config import settings  # noqa: E402
from evals.pipeline import load_golden_dataset, run_pipeline, save_results  # noqa: E402

OUTPUT_PATH = Path(
    os.getenv(
        "E2E_OUTPUT_PATH",
        str(ROOT / "evals" / "e2e_consistent_evaluation_results.json"),
    )
)


def _progress(index: int, total: int, _question: str, stage: str, _response: str = "") -> None:
    if stage == "done":
        print(f"Completed {index + 1}/{total}", flush=True)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _evaluation_manifest(dataset: dict) -> dict:
    samples = dataset["rag_samples"]
    question_payload = [
        {
            "id": sample["id"],
            "question": sample["question"],
            "reference": sample["reference"],
            "relevant_contexts": sample.get("relevant_contexts", []),
        }
        for sample in samples
    ]
    question_hash = _sha256(
        json.dumps(question_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    corpus_hash = hashlib.sha256()
    corpus_files = []
    corpus_dir = ROOT / "processed_data" / "true"
    for path in sorted(corpus_dir.glob("*.json")):
        file_hash = _sha256(path.read_bytes())
        corpus_files.append({"file": path.name, "sha256": file_hash})
        corpus_hash.update(path.name.encode("utf-8"))
        corpus_hash.update(file_hash.encode("ascii"))
    if not corpus_files:
        raise RuntimeError(
            f"No prepared source corpus found in {corpus_dir}; refusing an empty-corpus evaluation."
        )
    code_hash = hashlib.sha256()
    for relative in (
        "app/config.py",
        "app/main.py",
        "app/gateway/client.py",
        "app/guardrails/rails.py",
        "app/agents/graph.py",
        "app/agents/nodes/planner.py",
        "app/agents/nodes/retriever.py",
        "app/agents/nodes/responder.py",
        "app/services/retrieval/embedding.py",
        "app/services/retrieval/ranking_service.py",
        "app/services/retrieval/qdrant_service.py",
        "app/services/retrieval/web_search.py",
        "evals/pipeline.py",
    ):
        code_hash.update((ROOT / relative).read_bytes())
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        revision = "unknown"
    profile = {
        "provider": settings.LLM_PROVIDER,
        "model": settings.GEMINI_MODEL if settings.LLM_PROVIDER == "gemini" else settings.GROQ_MODEL,
        "context_limit_chars": settings.RAG_MAX_CONTEXT_CHARS,
        "history_limit_chars": settings.RAG_MAX_HISTORY_CHARS,
        "history_message_limit": settings.RAG_HISTORY_MESSAGES,
        "candidate_limit": settings.RAG_CANDIDATE_LIMIT,
        "min_relevance_score": settings.RAG_MIN_RELEVANCE_SCORE,
        "max_response_tokens": settings.GROQ_MAX_TOKENS,
        "request_timeout_seconds": settings.LLM_TIMEOUT_SECONDS,
        "web_search_enabled": settings.WEB_SEARCH_ENABLED,
        "web_search_results": settings.WEB_SEARCH_RESULTS,
        "question_count": len(samples),
        "question_ids": [sample["id"] for sample in samples],
        "question_set_sha256": question_hash,
        "corpus_sha256": corpus_hash.hexdigest(),
        "corpus_files": corpus_files,
        "application_source_sha256": code_hash.hexdigest(),
        "git_revision": revision,
    }
    profile_hash = _sha256(
        json.dumps(profile, sort_keys=True, ensure_ascii=False).encode("utf-8")
    )
    return {
        "run_id": profile_hash[:16],
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_sha256": profile_hash,
        "profile": profile,
        "status": "running",
        "limitations": [
            "Question/reference items are authored from the prepared corpus; the holdout is source-linked but not independently sampled from production traffic.",
            "This run evaluates one fixed model, prompt, corpus, and retrieval configuration.",
            "Citation ID checks measure provenance format, not semantic entailment.",
        ],
    }


def main() -> None:
    dataset = load_golden_dataset()
    dataset["evaluation_manifest"] = _evaluation_manifest(dataset)
    result = run_pipeline(
        dataset,
        progress_callback=_progress,
        checkpoint_path=str(OUTPUT_PATH),
    )
    result["evaluation_manifest"]["status"] = result["pipeline_metrics"]["status"]
    result["evaluation_manifest"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    save_results(result, str(OUTPUT_PATH))
    print(json.dumps(result.get("pipeline_metrics", {}), sort_keys=True), flush=True)
    print(f"Saved evaluation: {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
