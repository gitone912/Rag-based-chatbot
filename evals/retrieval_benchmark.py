"""Offline retrieval ablation for the prepared true-document corpus.

Run from the repository root with:
    python -m evals.retrieval_benchmark

The benchmark does not call Qdrant, Groq, or an embedding API. It compares the
local BM25 ranker, FlashRank cross-encoder, and their reciprocal-rank fusion
against the authored evidence passages in the golden set. Results are a local
retrieval diagnostic, not a production-quality estimate.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any

from app.services.retrieval.ranking_service import (
    reciprocal_rank_fusion,
    rerank_documents,
    score_documents_bm25,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "processed_data" / "true"
DEFAULT_GOLDEN = ROOT / "evals" / "golden_dataset.json"
DEFAULT_HOLDOUT = ROOT / "evals" / "retrieval_holdout.json"
DEFAULT_OUTPUT = ROOT / "evals" / "retrieval_benchmark_results.json"
# One question from each of the five domains is reserved from tuning.
HOLDOUT_IDS = {3, 6, 9, 12, 15}
BM25_CONFIDENCE_MARGIN = 0.10
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
    "can", "do", "does", "for", "from", "how", "in", "into", "is", "it",
    "of", "on", "or", "that", "the", "their", "this", "to", "was", "were",
    "what", "when", "which", "with",
}


def evidence_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.casefold())
        if len(token) > 1 and token not in STOP_WORDS
    }


def load_corpus(corpus_dir: Path) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for path in sorted(corpus_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        source = str(data.get("filename") or path.stem)
        for chunk_index, content in enumerate(data.get("chunks") or []):
            if isinstance(content, str) and content.strip():
                documents.append({
                    "id": f"{source}::{chunk_index}",
                    "source": source,
                    "content": content,
                })
    if not documents:
        raise ValueError(f"No processed document chunks found in {corpus_dir}")
    return documents


def _recall(gold: set[str], documents: list[dict[str, str]]) -> float:
    retrieved = set().union(*(evidence_tokens(doc["content"]) for doc in documents))
    return len(gold & retrieved) / len(gold) if gold else 0.0


def _ranking_metrics(
    gold: set[str], documents: list[dict[str, str]], target_chunks: set[str]
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "evidence_recall_at_1": round(_recall(gold, documents[:1]), 4),
        "evidence_recall_at_3": round(_recall(gold, documents[:3]), 4),
        "evidence_recall_at_5": round(_recall(gold, documents[:5]), 4),
        "top5_chunk_ids": [doc["id"] for doc in documents[:5]],
    }
    if target_chunks:
        rank = next(
            (position for position, doc in enumerate(documents, start=1)
             if doc["id"] in target_chunks),
            None,
        )
        metrics.update({
            "exact_evidence_chunk_hit_at_1": rank == 1,
            "exact_evidence_chunk_hit_at_5": rank is not None and rank <= 5,
            "exact_evidence_chunk_reciprocal_rank": round(1 / rank, 4) if rank else 0.0,
        })
    return metrics


def _sample_result(
    sample: dict[str, Any], gold: set[str], ranked: dict[str, list[dict[str, str]]],
    timings: dict[str, float], reranker_invoked: dict[str, bool],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": sample["id"],
        "domain": sample.get("domain", "unknown"),
        "question": sample["question"],
        "split": sample.get(
            "split", "holdout" if sample["id"] in HOLDOUT_IDS else "development"
        ),
        "methods": {},
    }
    for method, documents in ranked.items():
        metrics = {
            **_ranking_metrics(
                gold, documents, set(sample.get("evidence_chunk_ids", []))
            ),
            "latency_ms": round(timings[method], 2),
            "cross_encoder_invoked": reranker_invoked[method],
            "top5_sources": [doc["source"] for doc in documents[:5]],
        }
        result["methods"][method] = metrics
    return result


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for method in (
        "bm25", "flashrank", "hybrid_semantic_first", "hybrid_lexical_first", "adaptive"
    ):
        summary[method] = {}
        for split in ("development", "holdout", "all"):
            selected = [
                row for row in rows
                if split == "all" or row["split"] == split
            ]
            if not selected:
                continue
            values = [row["methods"][method] for row in selected]
            latencies = sorted(value["latency_ms"] for value in values)
            p95_index = min(len(latencies) - 1, int(0.95 * len(latencies)))
            exact_values = [
                value for value in values
                if "exact_evidence_chunk_hit_at_1" in value
            ]
            summary[method][split] = {
                "questions": len(values),
                "mean_evidence_recall_at_1": round(
                    statistics.mean(v["evidence_recall_at_1"] for v in values), 4
                ),
                "mean_evidence_recall_at_3": round(
                    statistics.mean(v["evidence_recall_at_3"] for v in values), 4
                ),
                "mean_evidence_recall_at_5": round(
                    statistics.mean(v["evidence_recall_at_5"] for v in values), 4
                ),
                "median_measured_latency_ms": round(statistics.median(latencies), 2),
                "p95_measured_latency_ms": round(latencies[p95_index], 2),
                "cross_encoder_call_rate": round(
                    statistics.mean(v["cross_encoder_invoked"] for v in values), 4
                ),
                "exact_evidence_chunk_questions": len(exact_values),
                "exact_evidence_chunk_hit_at_1": round(
                    statistics.mean(v["exact_evidence_chunk_hit_at_1"] for v in exact_values), 4
                ) if exact_values else None,
                "exact_evidence_chunk_hit_at_5": round(
                    statistics.mean(v["exact_evidence_chunk_hit_at_5"] for v in exact_values), 4
                ) if exact_values else None,
                "exact_evidence_chunk_mrr": round(
                    statistics.mean(v["exact_evidence_chunk_reciprocal_rank"] for v in exact_values), 4
                ) if exact_values else None,
            }
    return summary


def _summarize_candidate_proxy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = (
        "flashrank_top15",
        "bm25_top15",
        "hybrid_semantic_first_top15",
        "hybrid_lexical_first_top15",
        "adaptive_margin_0_10_top15",
    )
    summary: dict[str, Any] = {}
    for method in methods:
        summary[method] = {}
        for split in ("development", "holdout", "all"):
            selected = [
                row["candidate_pool_proxy"][method]
                for row in rows
                if split == "all" or row["split"] == split
            ]
            exact = [
                value for value in selected
                if "exact_evidence_chunk_hit_at_1" in value
            ]
            summary[method][split] = {
                "questions": len(selected),
                "mean_evidence_recall_at_1": round(
                    statistics.mean(v["evidence_recall_at_1"] for v in selected), 4
                ),
                "mean_evidence_recall_at_5": round(
                    statistics.mean(v["evidence_recall_at_5"] for v in selected), 4
                ),
                "cross_encoder_call_rate": round(
                    statistics.mean(v["cross_encoder_invoked"] for v in selected), 4
                ),
                "exact_evidence_chunk_questions": len(exact),
                "exact_evidence_chunk_hit_at_1": round(
                    statistics.mean(v["exact_evidence_chunk_hit_at_1"] for v in exact), 4
                ) if exact else None,
                "exact_evidence_chunk_hit_at_5": round(
                    statistics.mean(v["exact_evidence_chunk_hit_at_5"] for v in exact), 4
                ) if exact else None,
                "exact_evidence_chunk_mrr": round(
                    statistics.mean(v["exact_evidence_chunk_reciprocal_rank"] for v in exact), 4
                ) if exact else None,
            }
    return summary


def run_benchmark(
    corpus_dir: Path = DEFAULT_CORPUS,
    golden_path: Path = DEFAULT_GOLDEN,
    holdout_path: Path = DEFAULT_HOLDOUT,
) -> dict[str, Any]:
    documents = load_corpus(corpus_dir)
    golden_samples = json.loads(golden_path.read_text(encoding="utf-8")).get(
        "rag_samples", []
    )
    samples = [
        {
            **sample,
            "evidence_texts": sample.get("relevant_contexts", []),
            "split": "holdout" if sample["id"] in HOLDOUT_IDS else "development",
        }
        for sample in golden_samples
    ]
    holdout_data = json.loads(holdout_path.read_text(encoding="utf-8"))
    documents_by_id = {doc["id"]: doc for doc in documents}
    for sample in holdout_data.get("samples", []):
        evidence = documents_by_id.get(sample.get("evidence_chunk", ""))
        if evidence is None:
            raise ValueError(
                f"Unknown evidence_chunk {sample.get('evidence_chunk')!r} "
                f"for holdout question {sample.get('id')}"
            )
        samples.append({
            **sample,
            "evidence_texts": [evidence["content"]],
            "evidence_chunk_ids": [evidence["id"]],
            "split": "holdout",
        })
    if not samples:
        raise ValueError(f"No RAG samples found in {golden_path}")

    contents = [doc["content"] for doc in documents]
    by_content = {doc["content"]: doc for doc in documents}

    # Warm the local cross-encoder before collecting per-question timings.
    rerank_documents("retrieval benchmark warmup", contents[: min(5, len(contents))], top_n=1)

    rows = []
    for sample in samples:
        question = sample["question"]
        start = time.perf_counter()
        bm25_scored = score_documents_bm25(question, contents)
        bm25_contents = [content for content, _ in bm25_scored]
        bm25_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        flashrank_contents = rerank_documents(question, contents, top_n=len(contents))
        flashrank_ms = (time.perf_counter() - start) * 1000

        start = time.perf_counter()
        hybrid_contents = reciprocal_rank_fusion(
            [flashrank_contents, bm25_contents], top_n=len(contents)
        )
        hybrid_lexical_first_contents = reciprocal_rank_fusion(
            [bm25_contents, flashrank_contents], top_n=len(contents)
        )
        top_score = bm25_scored[0][1] if bm25_scored else 0.0
        second_score = bm25_scored[1][1] if len(bm25_scored) > 1 else 0.0
        margin_ratio = (top_score - second_score) / top_score if top_score > 0 else 0.0
        confident_lexical = top_score > 0 and margin_ratio >= BM25_CONFIDENCE_MARGIN
        adaptive_contents = (
            bm25_contents if confident_lexical else hybrid_lexical_first_contents
        )
        hybrid_ms = (time.perf_counter() - start) * 1000

        ranked = {
            "bm25": [by_content[content] for content in bm25_contents],
            "flashrank": [by_content[content] for content in flashrank_contents],
            "hybrid_semantic_first": [by_content[content] for content in hybrid_contents],
            "hybrid_lexical_first": [
                by_content[content] for content in hybrid_lexical_first_contents
            ],
            "adaptive": [by_content[content] for content in adaptive_contents],
        }
        timings = {
            "bm25": bm25_ms,
            "flashrank": flashrank_ms,
            # The hybrid methods include both rankers plus the inexpensive fusion.
            "hybrid_semantic_first": bm25_ms + flashrank_ms + hybrid_ms,
            "hybrid_lexical_first": bm25_ms + flashrank_ms + hybrid_ms,
            "adaptive": (
                bm25_ms if confident_lexical else bm25_ms + flashrank_ms + hybrid_ms
            ),
        }
        reranker_invoked = {
            "bm25": False,
            "flashrank": True,
            "hybrid_semantic_first": True,
            "hybrid_lexical_first": True,
            "adaptive": not confident_lexical,
        }
        gold = set().union(*(
            evidence_tokens(context)
            for context in sample.get("evidence_texts", [])
        ))
        row = _sample_result(sample, gold, ranked, timings, reranker_invoked)
        row["bm25_confidence_margin_ratio"] = round(margin_ratio, 4)

        # Approximate the live 15-candidate stage by taking the local
        # cross-encoder's top 15, then compare lexical, hybrid, and confidence
        # policies inside that fixed pool. This is not a Qdrant simulation.
        proxy_contents = flashrank_contents[:15]
        proxy_scored = score_documents_bm25(question, proxy_contents)
        proxy_lexical = [content for content, _ in proxy_scored]
        proxy_top = proxy_scored[0][1] if proxy_scored else 0.0
        proxy_second = proxy_scored[1][1] if len(proxy_scored) > 1 else 0.0
        proxy_margin = (proxy_top - proxy_second) / proxy_top if proxy_top > 0 else 0.0
        proxy_confident = proxy_top > 0 and proxy_margin >= BM25_CONFIDENCE_MARGIN
        proxy_semantic_first = reciprocal_rank_fusion(
            [proxy_contents, proxy_lexical], top_n=len(proxy_contents)
        )
        proxy_lexical_first = reciprocal_rank_fusion(
            [proxy_lexical, proxy_contents], top_n=len(proxy_contents)
        )
        proxy_methods = {
            "flashrank_top15": (proxy_contents, True),
            "bm25_top15": (proxy_lexical, False),
            "hybrid_semantic_first_top15": (proxy_semantic_first, True),
            "hybrid_lexical_first_top15": (proxy_lexical_first, True),
            "adaptive_margin_0_10_top15": (
                proxy_lexical if proxy_confident else proxy_lexical_first,
                not proxy_confident,
            ),
        }
        target_chunks = set(sample.get("evidence_chunk_ids", []))
        row["candidate_pool_proxy"] = {
            name: {
                **_ranking_metrics(
                    gold,
                    [by_content[content] for content in ranked_contents],
                    target_chunks,
                ),
                "cross_encoder_invoked": invoked,
            }
            for name, (ranked_contents, invoked) in proxy_methods.items()
        }
        rows.append(row)

    return {
        "scope": {
            "corpus_directory": (
                str(corpus_dir.resolve().relative_to(ROOT))
                if corpus_dir.resolve().is_relative_to(ROOT)
                else str(corpus_dir.resolve())
            ),
            "document_files": len({doc["source"] for doc in documents}),
            "chunks": len(documents),
            "questions": len(samples),
            "development_questions": sum(s["split"] == "development" for s in samples),
            "holdout_questions": sum(s["split"] == "holdout" for s in samples),
            "original_holdout_question_ids": sorted(HOLDOUT_IDS),
            "added_holdout_question_ids": [
                s["id"] for s in samples
                if s["split"] == "holdout" and s["id"] not in HOLDOUT_IDS
            ],
            "metric": "token coverage of authored relevant_contexts in top-k chunks",
            "adaptive_policy": (
                f"Use BM25 alone when the top-score margin ratio is at least "
                f"{BM25_CONFIDENCE_MARGIN:.2f}; otherwise run FlashRank and lexical-first RRF."
            ),
            "candidate_pool_proxy": (
                "Top 15 chunks from local FlashRank over all prepared chunks; "
                "used only to assess confidence-policy transfer to a smaller pool."
            ),
            "limitations": [
                "The original holdout contains one question per domain; 21 new questions were added after selecting the tie order.",
                "New holdout questions are author-curated from source chunks, not an external or independently collected benchmark.",
                "Uses all prepared chunks as candidates, not the live Qdrant candidate pool.",
                "Token coverage is a retrieval diagnostic, not answer correctness.",
                "Exact target-chunk metrics apply only to the 21 new source-linked holdout questions.",
                "Measured latency excludes embedding, Qdrant, answer generation, and cold model load.",
            ],
        },
        "summary": _summarize(rows),
        "candidate_pool_proxy_summary": _summarize_candidate_proxy(rows),
        "samples": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    parser.add_argument("--holdout", type=Path, default=DEFAULT_HOLDOUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = run_benchmark(args.corpus, args.golden, args.holdout)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "full_corpus": result["summary"],
        "candidate_pool_proxy": result["candidate_pool_proxy_summary"],
    }, indent=2))
    print(f"Saved full report: {args.output}")


if __name__ == "__main__":
    main()
