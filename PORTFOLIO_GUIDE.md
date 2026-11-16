# Portfolio Guide for the Enterprise Documentation Assistant

## Short project pitch

I built a document-grounded AI assistant for technical documentation. It ingests PDF, HTML, text, DOCX, and PPTX files; stores chunk embeddings in Qdrant; retrieves candidate passages; combines exact-term BM25 ranking with FlashRank semantic reranking using reciprocal-rank fusion; and generates concise answers with source IDs. FastAPI exposes the workflow, Streamlit provides the chat and evaluation views, and LangGraph keeps planning, retrieval, and response generation as explicit stages.

The central engineering problem is grounding: a fluent answer is not useful if it cannot be traced to the source material. The project therefore returns retrieved context and source mappings, validates citation IDs on the server, and measures retrieval, citation integrity, semantic quality, latency, and request completion as separate properties.

## Three-minute demo script

### 1. Explain the use case

“The assistant answers questions from a prepared technical-document corpus. It does not train a new model. At query time it retrieves evidence and gives that evidence to a language model.”

### 2. Show a grounded answer

Ask a question whose answer you have verified in a document under `DATA/`. Point out the answer’s inline citation, then expand the retrieved context in Streamlit and show that the API returns the cited source label and passage. Keep the example narrow enough that the supporting sentence is easy to find.

### 3. Explain why retrieval is hybrid

“Vector search helps when the question paraphrases the document. BM25 helps with exact technical identifiers and commands. FlashRank jointly scores query and passage. RRF merges ranks without pretending these models’ raw scores are comparable.”

### 4. Show the workflow boundary

Explain the API and graph at a high level: FastAPI validates and guards the request; LangGraph routes through planner, retriever, and responder; Qdrant returns candidates; the responder writes a cited answer or requests fallback. The graph is deliberately small and bounded.

### 5. Show evaluation and limitations

Show the evaluation view or the saved benchmark artifact. State the corpus and question-set size beside any metric. Explain that citation-ID validation confirms the ID maps to a returned source; it does not prove that the source semantically supports every claim. The completed post-retrieval-fix 36-case GPT-OSS 20B run had citations in 36/36 responses and 100% citation-ID and source-metadata mapping validity. A separate Qwen 3.8-27B model-assisted review scored correctness 0.9583, faithfulness 0.9819, and citation support 0.9806, with sufficient evidence in 36/36 cases. These results use an author-curated six-document set and are not human certification; have a non-author review the questions and references before making external quality claims.

## Evaluation status

Older retrieval and live API measurements are historical prototype artifacts. They use six prepared documents and authored questions, have not received independent human review, and are not current performance claims. A targeted miss investigation found the job-status evidence at vector rank 24; after adding state/status lifecycle synonyms it ranks first lexically and appears in the live top-five context. This is a single regression check, not a benchmark result.

The sanitized aggregate is `evals/reports/post_retrieval_fix_summary.md`; raw run and Qwen review outputs are kept local because they contain retrieved document excerpts. This profile used GPT-OSS 20B, a 4,000-character context cap, 512 output tokens, and disabled web search. The missed RBAC evidence now appears at source slot 2, and its reviewed answer scored 1.0 for correctness, faithfulness, and citation support. The complete-set averages improved versus the saved pre-fix baseline, but the authored six-document set is small and requires non-author human review before external claims.

## Resume bullets

- Built a document-grounded RAG assistant with Qdrant, local embeddings, BM25, FlashRank, and reciprocal-rank fusion; added query-expansion diagnostics for state/status lifecycle terms after finding an exact-source miss.
- Implemented a FastAPI and LangGraph answer workflow with citation-labeled evidence, server-side source-ID validation, Streamlit source inspection, and checkpointed evaluation instrumentation.
- Added versioned source IDs, content hashes, ingestion timestamps, schema versions, and stable chunk metadata to indexed source points.

Avoid reporting historical numeric results as current performance. Citation validation proves source-ID mapping, not semantic faithfulness.

## Interview talking points

### Why RAG instead of fine-tuning?

The documents change independently from the model. Re-indexing is a faster and more inspectable way to update the assistant’s source material. Retrieved passages also provide a traceable evidence path. Fine-tuning could help with behavior or format, but it would not replace current-document retrieval.

### Why combine vector search, BM25, and a cross-encoder?

They solve different retrieval problems. Dense vectors handle related wording; BM25 rewards exact names and tokens; a cross-encoder reranks query-passage pairs for relevance. RRF combines rank positions without mixing score scales. The benchmark measures each stage separately and shows that rerank quality depends on the candidate pool.

### What does citation validation guarantee?

It guarantees that the response contains IDs in the allowed range and that the API can map them to returned sources. It does not guarantee that the cited passage entails the claim. Semantic evaluation and human review are still needed.

### What did evaluation teach you?

Retrieval performance, answer faithfulness, citation integrity, request completion, and latency are different dimensions. A successful API call is not necessarily a correct answer. Authored evidence-token overlap is useful as a cheap diagnostic, but it is not a semantic truth score. Quotas can also make partial-run statistics misleading unless the denominator and errors are explicit.

### What would you improve next?

Have a non-author review the golden questions and references, then add a frozen independent holdout before making broader quality claims. Evaluate paragraph-aware chunking and page/section metadata on that holdout. Before multi-user deployment, finish production identity authorization, durable memory, distributed rate limits, retention operations, backups, and monitoring.

## Known limitations to disclose

- The chunker does not add overlap and does not preserve page or section locations.
- Ingestion now fingerprints source content and replaces stale versions after successful upsert; legacy index migration requires an exact-match backfill.
- Citation checks validate IDs and mappings, not semantic support.
- Qdrant failures now surface as dependency errors instead of appearing as empty retrieval.
- Local mode still uses process-local MemorySaver and local rate limits; public mode requires Cognito, Postgres, and Redis configuration.
- Cognito token authentication scopes conversations to a user identity; document-level ACLs are not implemented.
- The evaluation questions are authored from a small prepared corpus and require independent review.
- The post-fix live evaluation and separate-model review are complete; independent human review of the authored set remains outstanding.
- The Groq web fallback is provider-specific and needs separate evaluation.

## Demo preparation checklist

- Verify that `.env` is private and no credentials appear in terminal output, slides, or recordings.
- Confirm Qdrant readiness and that the collection contains the demonstration source.
- Choose one question with a short, directly verifiable evidence passage.
- Have a second question ready that has insufficient indexed evidence, and explain that web fallback behavior is Groq-specific.
- Keep the retrieval benchmark scope visible when quoting numbers.
- State the 36-case corpus and exact profile alongside metrics; distinguish the separate-model review from human certification.
