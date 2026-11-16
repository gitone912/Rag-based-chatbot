# Enterprise Agentic RAG

This workspace implements a document-grounded AI assistant: local document ingestion, LangGraph orchestration, Qdrant retrieval, BM25 + FlashRank hybrid reranking, citation-aware answer generation, FastAPI, Streamlit chat, and a retrieval/evaluation dashboard. Application inference uses direct Groq by default; Portkey remains an explicit optional provider.

See [PROJECT_GUIDE.md](PROJECT_GUIDE.md) for the end-to-end architecture, design choices, evaluation method and results, limitations, and recommended next steps.

For a concise portfolio pitch, demo script, interview talking points, and scoped resume bullets, see [PORTFOLIO_GUIDE.md](PORTFOLIO_GUIDE.md).

## Portfolio demo

This project demonstrates a document-grounded answer path: ingest a small, known corpus, ask a question whose evidence is present, and inspect the returned source passages and inline citations. Start Qdrant and the API, ingest `DATA/` if the collection is empty, then start the Streamlit UI. In the chat, ask a question with an answer in the corpus; expand the retrieved context and confirm the cited source labels map to the displayed passages. The evaluation dashboard is a separate view for inspecting repeatability, retrieval metrics, citation diagnostics, and latency.

For a short presentation, show one grounded answer, one question that triggers the web fallback, and the evaluation dashboard. Explain that citation-ID checks validate provenance format, while semantic support needs separate evaluation. Do not use secrets from `.env` in a screen recording or public repository.

## Run locally

1. Create and activate a Python 3.12 virtual environment, then install `requirements.txt`.
2. If you do not already have `.env`, copy `.env.example` to `.env` and add the credentials for the services you use. Keep `.env` private and never commit it. Choose `LLM_PROVIDER=groq` (default), `gemini`, or `portkey`; configure the corresponding key. Gemini generation uses a valid `GEMINI_API_KEY` and `GEMINI_MODEL`. Gemini is available as an opt-in provider; local Sentence Transformers embeddings and the existing Qdrant index remain unchanged.
3. Start Qdrant locally or provide a hosted endpoint. Start the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
4. In another terminal run `streamlit run ui/app.py`. Set `BACKEND_URL` if the API is not at `http://localhost:8000`.
5. Index documents with `python -m app.ingestion.processor DATA`. Add `--wipe` only when you deliberately want to delete and rebuild the whole collection.
6. Run the eval dashboard with `streamlit run evals/app.py`, or print a JSON report with `python -m evals.pipeline`.

For a local container demo, run `docker compose up --build`; it starts Qdrant, the API, Postgres, Redis, and Streamlit. The UI container receives only its backend URL, auth mode, and optional shared token; provider/database secrets stay in the API container. The compose stack is for private development and is not a public production deployment.

## Evaluation

`evals/golden_dataset.json` contains the 15-question core set. An optional 21-question source-linked holdout lives in `evals/retrieval_holdout.json` locally; that file is excluded from this public repository because it contains excerpts from the prepared source documents. Without the local holdout, the evaluator runs the core set. With it, the dashboard combines both sets for a 36-question live API run and records per-query latency. It can run RAGAS Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, and Context Recall using direct Groq; it prefers `JUDGE_GROQ` and falls back to `GROQ_API_KEY`. Evaluation calls consume provider tokens. The added questions are authored from the prepared documents and still need independent human review before their scores become release gates.

Run the local, API-free retrieval ablation with `python -m evals.retrieval_benchmark`. It compares BM25, FlashRank, hybrid tie ordering, and a 15-candidate proxy over the prepared corpus. The report covers 36 questions, with 26 held out from the tie-order selection; the questions are still author-curated from six documents, not an independent production benchmark.

The public aggregate for the post-retrieval-fix 36-case run is [`evals/reports/post_retrieval_fix_summary.md`](evals/reports/post_retrieval_fix_summary.md). The raw local run is `evals/e2e_post_retrieval_fix_36_results.json` (GPT-OSS 20B, 4,000 context characters, 512 response tokens, web search disabled): 36/36 completed, zero errors, median latency 4,347 ms, p95 5,329 ms, citation presence 36/36, citation-ID validity 36/36, and source-metadata match 36/36. A separate Qwen 3.8-27B model-assisted review (raw local output: `evals/independent_review_post_retrieval_fix_36_results.json`) scored correctness 0.9583, faithfulness 0.9819, and citation support 0.9806, with sufficient evidence in 36/36 cases. The API reported 47,225 input and 2,786 output tokens for the 36 responder calls. These results use one author-curated six-document set; they are model-assisted evidence, not human certification or a production-quality guarantee.

The previously missed RBAC question now retrieves `cronjobs.docx` chunk 10 at citation slot 2 and the separate reviewer scored it 1.0 for correctness, faithfulness, and citation support. Raw per-case run and review files are local-only because they contain retrieved excerpts; `.gitignore` excludes them. Have someone who did not author the questions and references review the set before using these scores as release gates.

Token budget defaults are capped at 4,000 context characters and 512 generated tokens; chat history is limited to six messages/2,000 characters. Technical evaluation questions bypass model-based guardrail/planner calls, and invalid citations produce a local safe refusal instead of triggering another model request. Groq browser search is opt-in because its fetched pages can cause large input-token spikes. The earlier quota-limited 512-token run is retained at `evals/e2e_citation_format_36_results.json` as a historical partial artifact; do not resume it under a changed profile.

The post-fix run has its own manifest and profile fingerprint. For future code or corpus changes, use a new output path so this result remains reproducible:

```powershell
$env:E2E_OUTPUT_PATH = "evals/e2e_future_profile_36_results.json"
python evals/run_live_evaluation.py
Remove-Item Env:E2E_OUTPUT_PATH
```

After all 36 cases complete, review that exact artifact:

```powershell
$env:INDEPENDENT_REVIEW_SOURCE = "evals/e2e_future_profile_36_results.json"
$env:INDEPENDENT_REVIEW_OUTPUT = "evals/independent_review_future_profile_36_results.json"
python evals/independent_review.py
Remove-Item Env:INDEPENDENT_REVIEW_SOURCE
Remove-Item Env:INDEPENDENT_REVIEW_OUTPUT
```

Additional engineering bullet: “Implemented citation-aware RAG responses with request-scoped evidence IDs, server-side citation validation, source metadata in the API/UI, and evaluation instrumentation for citation coverage and source-ID integrity.”

The answer path now labels retrieved passages, asks the model to cite them inline, returns cited-source metadata from the API, and rejects missing or invalid source IDs before accepting a document-grounded answer. The live evaluation pipeline records inline citation coverage, citation-ID validity, and API source mapping. These checks validate citation format and provenance; they do not by themselves verify that a cited passage semantically supports its claim. The pipeline can atomically checkpoint completed samples and resume a matching question set after interruption.

Run the live citation-aware pipeline from the repository root with `python evals/run_live_evaluation.py` after starting the API. It writes an atomic checkpoint to `evals/e2e_citation_evaluation_results.json` after each question and resumes completed samples on rerun; set `E2E_OUTPUT_PATH` to select a different checkpoint. The dashboard runner still keeps its data in session state. The guardrail evaluation remains available from the Eval Streamlit dashboard.

To score the saved live run on the 21 new source-linked holdout questions and checkpoint results as they arrive, run `python -m evals.ragas_holdout`. It records Faithfulness, Answer Relevancy, Context Precision, Context Recall, and Answer Correctness in `evals/ragas_holdout_results.json`; rerunning resumes missing question-metric pairs, and transient quota windows trigger a bounded wait and retry. The command uses the configured Groq judge key/model and consumes provider quota.

## Production notes

The API exposes `/health`, `/ready`, `/metrics`, and `/docs`; validates query size; reports dependency readiness; and the container runs as a non-root user. Local mode (`AUTH_MODE=none`) uses process-local memory and a single-process rate limiter. Treat it as private development only.

For public multi-user deployments, set `APP_ENV=production` and configure Cognito OIDC (`AUTH_MODE=cognito`, matching `COGNITO_ISSUER`, `COGNITO_CLIENT_ID`, and required group `COGNITO_REQUIRED_GROUP`), Postgres (`CHECKPOINT_DATABASE_URL` with `sslmode=verify-full` and a mounted RDS `sslrootcert`), Redis (`RATE_LIMIT_REDIS_URL` using `rediss://`), hosted Qdrant over HTTPS with its API key, the selected LLM provider key, and a strong `METRICS_BEARER_TOKEN` of at least 32 characters. Startup fails closed when these settings are incomplete. Supply Streamlit OIDC client secrets through a dedicated secret mount or secret manager, not the application `.env`. Users in the required Cognito group share the configured knowledge corpus; add tenant/document ACLs before mixing restricted corpora. Cognito tokens scope checkpoint thread IDs to the authenticated subject; per-user minute/day limits use Redis; memory expires after `MEMORY_RETENTION_DAYS` and users can delete their memory. Set limits and retention to match the product policy. Deploy over TLS, keep API/DB/Redis/Qdrant private, and use managed encrypted services and a secret manager. The example Compose stack is for local development, not a hardened public deployment.

Ingestion stores a source SHA-256 version, source ID, chunk number, schema version, and ingestion timestamp with each point. Run `python -m app.ingestion.processor DATA` for changed files; it writes new vectors before removing stale versions. `python scripts/backfill_source_metadata.py` previews migration of legacy points; add `--apply` only after reviewing the exact-match summary. The current six-document index was backfilled and verified at 95 versioned points.

Create Qdrant and Postgres backups with `python scripts/backup.py`; `--s3-uri s3://bucket/prefix` uploads with SSE-KMS using the AWS default credential chain, and `--s3-kms-key-id` selects a customer-managed key; production mode requires both S3 options. Schedule it, enforce private bucket access and lifecycle retention in AWS, alert on failures, and test restoration before relying on it. Schedule expired memory cleanup independently of API traffic with `python scripts/cleanup_expired_memory.py` (for example, an EventBridge-triggered ECS task). `/metrics` exports low-cardinality HTTP, latency, provider-error, and reported-token metrics; protect it with `METRICS_BEARER_TOKEN` and scrape it with Prometheus or an equivalent monitoring service. Logfire export remains opt-in through `LOGFIRE_SEND_TO_LOGFIRE=true`.

Groq token limits are enforced at the organization level. The API distinguishes daily exhaustion, Groq SDK retries default to zero, and live evaluation checkpoints completed questions and stops at the hard cap. The runner uses configurable pacing (`EVAL_DELAY_BETWEEN_CALLS`, default 20 seconds) and longer backoff after TPM rejections; these controls do not increase provider quota. `RAG_MAX_CONTEXT_CHARS` bounds evidence sent to the responder to reduce per-request input tokens. Gemini can be selected for generation with `LLM_PROVIDER=gemini`; Groq's browser-search fallback is disabled in that mode to keep requests off the capped provider.

Set `API_AUTH_TOKEN` to enable bearer-token authentication on `/query`; the Streamlit process sends the same server-side token. This shared token is suitable only behind a trusted/private deployment boundary. Use user/session identity and an authorization layer for multi-user public deployments.
