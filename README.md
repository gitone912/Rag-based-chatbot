# Enterprise Agentic RAG

This workspace implements the core of the [8hr-MARATHON reference project](https://github.com/d-hackmt/8hr-MARATHON): local document ingestion, LangGraph orchestration, Qdrant retrieval, FlashRank reranking, FastAPI, Streamlit chat, and a golden-set evaluation dashboard. Application inference uses direct Groq by default; Portkey remains an explicit optional provider.

## Run locally

1. Create and activate a Python 3.12 virtual environment, then install `requirements.txt`.
2. If you do not already have `.env`, copy `.env.example` to `.env` and add the credentials for the services you use. Keep `.env` private and never commit it. Set `GROQ_API_KEY`, and set `LLM_PROVIDER=groq` (the default). To opt back into Portkey, set `LLM_PROVIDER=portkey` and configure `PORTKEY_API_KEY` and optionally `PORTKEY_CONFIG_ID`.
3. Start Qdrant locally or provide a hosted endpoint. Start the API with `uvicorn app.main:app --host 0.0.0.0 --port 8000`.
4. In another terminal run `streamlit run ui/app.py`. Set `BACKEND_URL` if the API is not at `http://localhost:8000`.
5. Index documents with `python -m app.ingestion.processor DATA --wipe`.
6. Run the eval dashboard with `streamlit run evals/app.py`, or print a JSON report with `python -m evals.pipeline`.

For a local container demo, run `docker compose up --build`; it starts Qdrant, the API, and Streamlit. The compose stack uses local Qdrant and is intended for development/smoke demos. It does not replace managed production infrastructure.

## Evaluation

`evals/golden_dataset.json` is the editable golden set. The pipeline calls the live API and reports context coverage and token Jaccard overlap with references. The Streamlit eval app can additionally run RAGAS Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, and Context Recall using direct Groq; it prefers `JUDGE_GROQ` and falls back to `GROQ_API_KEY`. Evaluation calls consume provider tokens. Review and expand the golden set with domain-specific references before treating scores as release gates.

Run the mocked unit tests with `python -m pytest tests`. Run the live golden-set and guardrail evaluation from the Eval Streamlit dashboard after starting the API. It uses the configured `BACKEND_URL` to reach `/query`.

## Production notes

The API exposes `/health`, `/ready`, and `/docs`, validates query lengths, returns a service error instead of a successful HTTP response on internal failures, and the container runs as a non-root user. Provide secrets through the deployment secret manager. For production, use persistent Qdrant, a durable LangGraph checkpointer, user auth/rate limiting at the ingress, and externalized observability credentials. The in-process MemorySaver is suitable for single-process development only. Logfire export is disabled by default; set `LOGFIRE_SEND_TO_LOGFIRE=true` to explicitly enable remote traces.

Set `API_AUTH_TOKEN` to enable bearer-token authentication on `/query`; the Streamlit process sends the same server-side token. This shared token is suitable only behind a trusted/private deployment boundary. Use user/session identity and an authorization layer for multi-user public deployments.
