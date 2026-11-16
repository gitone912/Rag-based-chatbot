# Enterprise Documentation Assistant — Project Guide

## 1. What the project does

This project is a document-grounded question-answering assistant for technical documentation. It indexes documents, retrieves passages for a user's question, drafts a concise answer from those passages, and uses a cited web search only when the indexed material does not contain enough evidence. It includes a chat UI, a FastAPI service, an evaluation dashboard, and a golden set of Kubernetes and Databricks questions.

The main design goal is **grounded answers**: the assistant should show where its answer came from and should not invent missing details. The system does not train a foundation model. It combines existing language models with retrieval, reranking, application logic, and evaluation.

## 2. Why this architecture was chosen

### Retrieval-augmented generation instead of model training

The source of truth is the indexed documentation, and those documents can change independently of the model. Retrieval-augmented generation (RAG) lets the application find relevant passages at query time and give those passages to a language model. That is faster to update than fine-tuning and makes it possible to show source excerpts with answers. It also gives the assistant a route to refuse or search for evidence when local documents are insufficient.

RAG does not guarantee correctness on its own. The search can return the wrong passages, the answer can overstate what a passage says, and the evaluator can mis-score an answer. The retrieval, response, and evaluation stages are therefore measured separately.

### LangGraph for explicit stages

The graph separates planning, retrieval, and response synthesis into nodes. This makes it easier to inspect where a request went wrong: whether it was routed away from retrieval, retrieved weak documents, or produced an unsupported response. The graph is intentionally a small workflow rather than a free-running autonomous agent.

### Qdrant plus semantic and lexical ranking

Qdrant stores document chunk embeddings and supports similarity search. Semantic search helps when a question uses different wording from a document. Exact technical names, commands, API groups, and manifest fields can be harder for vector search, so retrieval now combines:

1. Qdrant vector search to get candidate chunks.
2. FlashRank cross-encoder ranking to judge question-passage relevance.
3. BM25 lexical ranking to reward exact word and identifier matches.
4. Reciprocal Rank Fusion (RRF) to merge the two ranked lists without assuming their scores use the same scale.

The fused top five passages are sent to the answer generator. RRF preserves the first ranking's order when two passages receive the same fusion score, so the retriever puts BM25 first to give exact technical terms priority in ties. Earlier offline runs cover 36 authored questions over 95 chunks from six prepared documents, including 21 source-linked cases. Those historical results are not independent review and are not used as current release-quality claims. In the current miss investigation, the monitoring-status evidence chunk appeared at vector rank 24 and was initially excluded from the final context; adding lifecycle status synonyms moves it to rank 1 in the full local corpus and into the live top-five context. This is one targeted retrieval regression check, not a general quality score.

Earlier BM25 + FlashRank + RRF measurements are preserved as prototype artifacts. They use a small authored corpus and question set, and have not been independently reviewed; do not present them as current production performance.

### Local embeddings and the 768-dimensional collection

The default embedding path uses a local Sentence Transformers model (`all-mpnet-base-v2`) that produces 768-dimensional vectors. Local inference avoids sending every document to a separate embedding API and matches the current Qdrant collection. Gemini embeddings are an opt-in path with a different vector dimension; switching dimensions requires rebuilding the collection. Do not change the embedding model against an existing collection without re-indexing it.

### Groq for inference and evaluation

The application uses the configured Groq model through an OpenAI-compatible client by default. The RAGAS judge defaults to `qwen/qwen3.8-27b`, separately configurable with `RAGAS_JUDGE_MODEL`. `JUDGE_GROQ` is preferred for judge calls so evaluation can use a separate key; it falls back to `GROQ_API_KEY`. LLM judging is useful for semantic questions, but it costs tokens, can be rate-limited, and is not perfectly deterministic. This is why the project adds a cheap local retrieval diagnostic alongside RAGAS rather than replacing RAGAS with another single score.

### FastAPI and Streamlit

FastAPI exposes the query and service-health endpoints independently of the UI. Streamlit provides a quick local chat interface and a three-step evaluation dashboard. This keeps the prototype easy to run and inspect while leaving the core query path available to other clients.

## 3. Main components

| Area | Location | Responsibility |
| --- | --- | --- |
| API and graph request | `app/main.py` | Validates `/query`, applies guardrails, invokes the graph, and returns answers and sources. |
| Graph | `app/agents/graph.py` | Connects planner, retriever, and responder nodes; uses an in-process memory checkpointer. |
| Planner | `app/agents/nodes/planner.py` | Routes greetings/history-only requests and prepares technical search queries. |
| Retriever | `app/agents/nodes/retriever.py` | Queries Qdrant, filters weak matches, and returns five fused passages. |
| Ranking | `app/services/retrieval/ranking_service.py` | Implements FlashRank, BM25, and reciprocal-rank fusion. |
| Responder | `app/agents/nodes/responder.py` | Answers from retrieved context; requests web fallback when context is insufficient. |
| Web search | `app/services/retrieval/web_search.py` | Uses Groq's browser-search tool and returns source URLs for fallback answers. |
| Guardrails | `app/guardrails/` | Blocks known jailbreak/off-topic patterns and uses NeMo Guardrails for additional routing. |
| Ingestion | `app/ingestion/` | Parses files, chunks text, embeds chunks, and writes them to Qdrant. |
| Chat UI | `ui/app.py` | Sends chat messages to FastAPI and displays answers and retrieved source passages. |
| Golden set and live runner | `evals/golden_dataset.json`, `evals/pipeline.py` | Stores expected questions/answers and captures live responses, retrieved passages, and tool calls. |
| Evaluation | `evals/metrics.py`, `evals/guardrails_eval.py`, `evals/app.py` | Computes RAGAS, tool correctness, local evidence coverage, and guardrail metrics. |

## 4. How documents enter the system

The ingestion path is:

1. **Choose files.** The `DATA/` directory contains the document corpus. Subfolders, when present, are recorded as source types.
2. **Parse locally.** PDFs use `pypdf` with a `pdfplumber` fallback for blank pages; HTML is cleaned with BeautifulSoup; text files are read as text; DOCX/PPTX are parsed with Unstructured.
3. **Chunk.** `chunk_text` groups paragraphs into chunks up to about 1,500 characters. A single oversized paragraph is split into fixed-size pieces. The current chunker does not add overlap, so a fact at a chunk boundary can be separated from its explanation.
4. **Embed.** The configured embedding model converts each chunk into a vector.
5. **Index.** Qdrant collection `enterprise_rag` stores the vector plus payload fields such as `text`, `source`, and `source_type`, using cosine distance. Parsed chunk metadata is also saved under `processed_data/`.

The corpus should be curated before indexing. A broad or noisy corpus can return passages from unrelated documents even when vector search is working as designed. Re-ingestion with `--wipe` deletes and rebuilds the Qdrant collection, so use that option only when a full replacement is intended and the target collection is safe to recreate.

## 5. How a user question is answered

1. **Chat UI sends the request.** `ui/app.py` sends the user's text and a session/thread ID to `POST /query`.
2. **API validates and guards.** FastAPI validates input size and supports local, shared-token staging, and Cognito identity modes. In public Cognito mode, signed ID tokens are checked and the user subject scopes the memory thread. The guard checks obvious off-topic and jailbreak patterns, then invokes NeMo Guardrails for ambiguous prompts.
3. **Planner routes.** Greetings or questions answerable from current conversation history can use the conversational route. Technical questions are sent to document retrieval. Some first-turn terms such as Kubernetes, kubectl, Redis, Databricks, and manifest route directly to retrieval so a stochastic planner cannot casually skip the document search.
4. **Qdrant retrieves candidates.** The query is embedded and Qdrant returns up to 15 candidate passages. Candidates below `RAG_MIN_RELEVANCE_SCORE` (default 0.45) are removed.
5. **Hybrid ranker orders candidates.** FlashRank and BM25 each rank the candidates, then RRF combines rankings. The best five passages are formatted with source labels.
6. **Responder writes a concise, cited answer.** Retrieved passages receive stable request-local IDs. The prompt asks for each factual sentence/bullet to cite those IDs, and the server rejects answers with missing or out-of-range IDs by using the web fallback. It also tells the model to treat retrieved text as evidence rather than instructions. This is citation-format validation; it does not prove semantic entailment, so faithfulness still needs separate evaluation.
7. **Web fallback runs only if needed.** When no relevant indexed documents exist or the responder explicitly reports insufficient evidence, the application can call Groq's built-in browser search. It prefers official/primary sources and returns direct URLs. This is not a replacement for the document path.
8. **API returns result and provenance.** The JSON response includes the answer, status, thought-process labels, source passages, and the numbered sources cited by the answer. The chat UI shows the cited sources and expandable retrieved context.

Local mode uses LangGraph `MemorySaver`, which is process-local and appropriate only for a private development demo. Cognito mode requires a Postgres LangGraph checkpointer and Redis-backed per-user rate limits. The API records memory activity, expires it after `MEMORY_RETENTION_DAYS`, and exposes a per-user deletion endpoint. Production must set the corresponding variables and operate Postgres/Redis as private durable managed services.

## 6. How to run it locally

From the repository root in PowerShell:

```powershell
# Activate the existing environment
.\.venv\Scripts\Activate.ps1

# Start the API
uvicorn app.main:app --host 127.0.0.1 --port 8000

# In a second terminal, start the chat UI
streamlit run ui/app.py --server.port 8501

# In another terminal, start the evaluation UI
streamlit run evals/app.py --server.port 8502
```

Set the service values in `.env` and keep that file private. Main settings include `QDRANT_CLUSTER_ENDPOINT`, `QDRANT_API_KEY`, `GROQ_API_KEY`, `LLM_PROVIDER`, `GROQ_MODEL`, `EMBEDDING_PROVIDER`, `RAG_MIN_RELEVANCE_SCORE`, `RAG_MAX_CONTEXT_CHARS`, `BACKEND_URL`, `JUDGE_GROQ`, and `RAGAS_JUDGE_MODEL`. Optional API bearer authentication uses `API_AUTH_TOKEN`. The example file is `.env.example`; it contains names and defaults, not user credentials.

Useful API endpoints are `/health` for process health, `/ready` for Qdrant/collection readiness, `/docs` for the FastAPI schema, and `/graph` for a graph visualization. A successful `/health` does not prove that Qdrant or the LLM provider is ready; check `/ready` and run a real query as well.

To ingest documents, the command is:

```powershell
python -m app.ingestion.processor DATA
```

Only add `--wipe` when deliberately rebuilding the collection. The local Docker stack is also available via `docker compose up --build`.

## 7. How the evaluation works and why these methods are used

### Golden dataset

`evals/golden_dataset.json` contains 15 core RAG questions. `evals/retrieval_holdout.json` adds 21 questions with reference answers and exact evidence chunk IDs. The evaluation dashboard combines both files into a 36-question live run; 26 questions are marked holdout by the offline retrieval benchmark. The new questions were written and checked against their source chunks, but they have not received an independent human review or been collected from real users. There are also 24 guardrail cases, balanced between 12 expected blocks and 12 legitimate passes.

A golden set is used because it makes comparisons repeatable: every version sees the same questions and expected evidence. The expanded set improves topic coverage, but its questions are still authored from six prepared documents. Have a reviewer who did not write the questions validate the references and add representative user queries before treating scores as release gates.

### Step 1 — Review ground truth

The evaluator displays questions, references, relevant evidence snippets, and expected tools. This is a human review step: incorrect or incomplete references create incorrect scores no matter how good the model is.

### Step 2 — Run the live pipeline

The runner sends each of the 36 RAG questions to the configured FastAPI `/query` endpoint and records the answer, retrieved contexts, citation metadata, tool label, errors, and end-to-end request latency. The dashboard displays median and p95 latency, inline citation coverage and ID validity, API source-mapping validity, and can download the full run as JSON. The web Sources footer is excluded from inline coverage. These are format/provenance diagnostics, not semantic support scores. Calls are sequential and spaced out to reduce rate-limit pressure. Run `python evals/run_live_evaluation.py` for an atomic per-question checkpoint that resumes the same question set. The dashboard app still keeps its own in-progress data in session state.

Historical prototype run: 36/36 requests completed, with 16.88 seconds median and 26.15 seconds p95 latency. It predates the current source and retrieval profile and is not a quality claim. The cost-optimized run (`evals/e2e_cost_optimized_36_results.json`) completed 36/36 with zero request errors, 4.62 seconds median and 9.20 seconds p95. It used 4,000 context characters and a 256-token output cap; the API reported 52,956 input plus 4,253 output tokens across the 36 responder calls and one web fallback call. Inline citations appeared in only 9/36 answers; citation-ID and metadata mapping diagnostics were 100%. The separate Qwen review (`evals/independent_review_cost_optimized_36_results.json`) scored answer correctness, faithfulness, and citation support at 0.2222 each, with evidence sufficient in 32/36 cases. This model-assisted review indicates poor answer performance and is not human certification. No positive quality claim is supported.

Older citation and trimmed-context checkpoints are retained as historical diagnostics only. They use different run profiles and must not be merged with the current evaluation. Citation syntax/source mapping are provenance diagnostics, not semantic support scores.

### Step 3 — Score responses

Five RAGAS metrics use the configured Groq judge:

- **Faithfulness:** whether answer claims are supported by retrieved passages.
- **Answer relevancy:** whether the answer addresses the user's question.
- **Context precision:** whether useful passages rank ahead of irrelevant passages.
- **Context recall:** whether retrieved passages contain the evidence required by the reference.
- **Answer correctness:** semantic agreement between the answer and reference.

The suite also computes **tool correctness** as Jaccard overlap between expected and actual tool sets. This is inexpensive but only measures routing labels; it does not measure whether retrieval found the right text.

LLM metrics run one sample at a time with cooldowns because Groq applies token-per-minute and daily quotas. A full run can take about 50 minutes. A failed/quota-limited judge run should be recorded as incomplete; partial or old scores must not be reported as if they represented the whole evaluation.

### Why add local gold-evidence token recall

RAGAS is useful for semantic judgments but it consumes provider tokens and its result depends on exactly which text was passed to the judge. The new local check computes token overlap between each authored `relevant_contexts` snippet and the union of retrieved passages:

- **Evidence recall @1** uses only the first retrieved passage.
- **Evidence recall @5** uses the top five passages.

Common stopwords are removed, overlap is divided by the number of content tokens in each golden evidence snippet, and scores are averaged across snippets and questions. The method is deterministic, fast, local, and does not call Groq. Comparing @1 with @5 helps distinguish a ranking problem (the evidence exists but is too low) from a candidate-retrieval problem (the evidence is absent from the top five).

This is a diagnostic, not a semantic truth score. It can reward long passages that happen to contain many matching words; it does not prove the answer is correct or that a human considers the passage relevant. Keep it alongside RAGAS and manual review rather than replacing them.

### Important caveat about the first full RAGAS run

The first full run produced these means: faithfulness 0.148, answer relevancy 0.755, context precision 0.200, context recall 0.200, answer correctness 0.772, and tool correctness 1.000. Those RAGAS results used the previous evaluation preprocessing: the judge saw only two contexts capped at 300 characters, and saved answers were capped at 300 characters. That dropped evidence from the judge input and can distort grounding/correctness metrics. Treat the first RAGAS numbers as a diagnostic baseline, not a calibrated final score.

The evaluator has since been changed to pass up to five cleaned passages capped at 600 characters each and retain up to 2,000 answer characters. A new full RAGAS run with this input has **not** been completed yet. The separate 15-question live Qdrant comparison measured local gold-evidence coverage of 0.817 at @1 and 0.977 at @5 for the previous ranking, versus 0.817 at @1 and 0.985 at @5 with hybrid BM25 + FlashRank. This result uses a different candidate pool and should not be combined with the offline ablation above. See `evals/hybrid_retrieval_comparison.json` for the live per-question comparison and `evals/full_evaluation_metrics.json` for prior RAGAS scores and their caveat.

The offline ablation can be rerun with `python -m evals.retrieval_benchmark`. It compares BM25, FlashRank, both RRF tie orders, and a confidence-based reranker shortcut on all chunks in `processed_data/true`; it also reports a FlashRank top-15 candidate proxy. The shortcut looked good over the full corpus but lost rank-1 coverage in the smaller candidate proxy, so it was not enabled in production. The report is written to `evals/retrieval_benchmark_results.json`. It does not call Qdrant, Groq, or an embedding API. The 26-question holdout is authored from the same six documents, so it still needs independent review and broader corpus coverage before serving as a resume claim.

### Guardrail policy update and validation

The earlier six-case API run scored 4/6 and missed an SQL-injection exploitation request and a programmer-joke request. The deterministic policy now catches clear entertainment/lifestyle intent, jailbreak patterns, and requests to exploit or steal credentials, while checking for technical-scope terms to avoid blocking legitimate Kubernetes, Databricks, and related questions. These phrase rules are a lightweight first gate, not a complete security classifier; keep expanding examples and review false positives.

The expanded 24-case set has two distinct validation artifacts:

- `evals/guardrail_policy_validation.json` checks only the local deterministic rules, without provider calls. It currently scores 12 TP, 12 TN, 0 FP, and 0 FN (precision/recall/accuracy 1.000) on this authored set. This confirms that the explicit rules match the present cases; it is not a general safety guarantee and does not measure NeMo behavior.
- `evals/guardrail_validation.json` exercises the full API. The updated backend blocked all 12 expected-block inputs. Only 1 of 12 legitimate cases completed; the other 11 returned evaluation errors because Groq hit its daily token quota (plus a timeout). Therefore this API run is **incomplete** and its 1.000 scores over 13 completed cases must not be presented as the full-set score. The runner now excludes errors from metric denominators and displays the error count.

## 8. Recommended next steps

### Completed evaluation and miss-driven retrieval fix

The public aggregate for the post-fix live run is `evals/reports/post_retrieval_fix_summary.md`. The full local result is `evals/e2e_post_retrieval_fix_36_results.json`: 36 completed, zero errors, median latency 4,347 ms, p95 5,329 ms, citation presence 36/36, citation-ID validity 36/36, and source-metadata match 36/36. Its profile used GPT-OSS 20B, 4,000 context characters, 512 response tokens, and web search disabled. The API recorded 47,225 input and 2,786 output tokens across the 36 responder calls.

The miss taxonomy found that question 109’s RBAC evidence was BM25 rank 2 and vector rank 3 but fell to RRF rank 8, outside the top five. The retriever now reserves two top lexical candidates before filling remaining slots from RRF. In the post-fix run, `cronjobs.docx` chunk 10 appeared at citation slot 2, and the answer identified `secrets`/`configmaps` read verbs plus pod `get`/`list`/`watch` permissions. The responder also retains indexed evidence when web fallback is unavailable.

The raw separate Qwen review is local-only at `evals/independent_review_post_retrieval_fix_36_results.json`: correctness 0.9583, faithfulness 0.9819, citation support 0.9806, and sufficient evidence in 36/36 cases. This is model-assisted review, not human certification. Both runs use an authored set derived from six prepared documents; a non-author should review the references before using metrics as release gates. Raw evaluation output and the source-linked holdout are excluded from the public repository because they contain retrieved source excerpts; only the aggregate summary is published. The earlier pre-fix baseline remains at `evals/e2e_gpt_oss_20b_36_results.json` with its review in `evals/independent_review_gpt_oss_20b_36_results.json`.

### Follow-up improvements after the baseline

- Have a non-author review the golden questions and references, then validate future retrieval changes on a frozen independent holdout.
- Add a held-out test set and more golden questions before tuning retrieval weights. Do not tune and report on the same tiny set as if it were independent validation.
- Consider paragraph-aware chunking with modest overlap, source metadata filters, or query expansion if misses show that evidence is split across chunks or absent from Qdrant candidates.
- Evaluate web fallback separately: test when it triggers, whether its URLs are official/relevant, and whether the answer cites only sources actually returned.
- Persist every evaluation run to a timestamped JSON/CSV artifact and let the dashboard reload it; current Streamlit session state is temporary.
- Before public deployment, configure Cognito, private managed Postgres/Redis/Qdrant, TLS ingress, secret injection, protected metrics scraping, scheduled off-site backups, restore drills, retention operations, and alert routing. The local default is intentionally not public-ready.

## 9. Current evaluation artifacts

- `evals/full_evaluation_results.json` — captured answers, retrieved contexts, tool labels, and guardrail outcomes from the previous live run.
- `evals/full_evaluation_metrics.json` — prior RAGAS/tool scores with a note that the RAGAS input was truncated.
- `evals/hybrid_retrieval_comparison.json` — per-question evidence coverage and source ordering before/after hybrid ranking.
- `evals/guardrail_policy_validation.json` — local deterministic 24-case policy check.
- `evals/guardrail_validation.json` — incomplete live API guardrail run; inspect `evaluation_error` before using its metrics.

Historical partial checkpoints remain local-only and should not be resumed under a different evaluation profile. The latest complete post-fix results and their profile identifiers are summarized in `evals/reports/post_retrieval_fix_summary.md`.
