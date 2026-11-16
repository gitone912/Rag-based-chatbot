"""Classify retrieval and answer misses in a completed live evaluation artifact."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)
STOP = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "this",
    "to", "what", "when", "where", "which", "with",
}
INLINE_CITATION = re.compile(r"(?<!\w)\[(\d+)\]")


def _tokens(text: str) -> set[str]:
    return {token.casefold() for token in TOKEN_RE.findall(text) if len(token) > 1 and token.casefold() not in STOP}


def _context_ids(contexts: list[str]) -> set[int]:
    ids = set()
    for context in contexts:
        if not isinstance(context, str) or not context.splitlines():
            continue
        match = re.match(r"^\[(\d+)\]\s+SOURCE:|^WEB SOURCE \[(\d+)\]", context.splitlines()[0])
        if match:
            ids.add(int(match.group(1) or match.group(2)))
    return ids


def classify_sample(sample: dict) -> dict:
    question = str(sample.get("question", ""))
    contexts = [item for item in sample.get("actual_contexts", []) if isinstance(item, str)]
    answer = str(sample.get("actual_response", ""))
    expected_contexts = [
        item for item in sample.get("relevant_contexts", []) if isinstance(item, str)
    ]
    expected_tokens = set().union(*(_tokens(item) for item in expected_contexts)) if expected_contexts else set()
    found_tokens = set().union(*(_tokens(item) for item in contexts)) if contexts else set()
    evidence_coverage = len(expected_tokens & found_tokens) / len(expected_tokens) if expected_tokens else None

    expected_chunk = sample.get("evidence_chunk")
    expected_source = str(expected_chunk).split("::", 1)[0] if expected_chunk else None
    source_hit = bool(
        expected_source
        and any(expected_source.casefold() in context.splitlines()[0].casefold() for context in contexts if context.splitlines())
    )
    citation_ids = [int(value) for value in INLINE_CITATION.findall(answer)]
    valid_ids = _context_ids(contexts)
    pending = (
        sample.get("actual_latency_ms") is None
        and not sample.get("actual_error")
        and not answer.strip()
    )

    signals = []
    if pending:
        category = "not_evaluated"
    else:
        if sample.get("actual_error"):
            signals.append("provider_or_service_failure")
        if expected_source and not source_hit:
            signals.append("candidate_or_relevance_filter_miss")
        if evidence_coverage is not None and evidence_coverage < 0.5:
            signals.append("evidence_coverage_miss")
        if not answer.strip() and not sample.get("actual_error"):
            signals.append("empty_answer")
        if answer.strip() and not citation_ids:
            signals.append("citation_omission")
        if any(citation_id not in valid_ids for citation_id in citation_ids):
            signals.append("invalid_citation_id")

        if sample.get("actual_error"):
            category = "provider_or_service_failure"
        elif expected_source and not source_hit:
            category = "candidate_or_relevance_filter_miss"
        elif evidence_coverage is not None and evidence_coverage < 0.5:
            category = "evidence_coverage_miss"
        elif not answer.strip():
            category = "empty_answer"
        elif not citation_ids:
            category = "citation_omission"
        elif any(citation_id not in valid_ids for citation_id in citation_ids):
            category = "invalid_citation_id"
        else:
            category = "no_automated_miss_detected"

    return {
        "id": sample.get("id"),
        "question": question,
        "category": category,
        "signals": signals,
        "evidence_token_coverage": round(evidence_coverage, 4) if evidence_coverage is not None else None,
        "expected_source": expected_source,
        "expected_source_retrieved": source_hit if expected_source else None,
        "actual_sources": [
            context.splitlines()[0][:300] for context in contexts if context.splitlines()
        ],
        "actual_error": sample.get("actual_error"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "evaluation",
        nargs="?",
        type=Path,
        default=ROOT / "evals" / "e2e_consistent_evaluation_results.json",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.loads(args.evaluation.read_text(encoding="utf-8"))
    rows = [classify_sample(sample) for sample in data.get("rag_samples", [])]
    counts = Counter(row["category"] for row in rows)
    signal_counts = Counter(signal for row in rows for signal in row["signals"])
    report = {
        "source_evaluation": str(args.evaluation),
        "evaluation_manifest": data.get("evaluation_manifest", {}),
        "sample_count": len(rows),
        "miss_counts": dict(sorted(counts.items())),
        "signal_counts": dict(sorted(signal_counts.items())),
        "samples": rows,
        "limitations": [
            "Token coverage and source-name matching are retrieval diagnostics, not semantic answer grading.",
            "A sample marked no_automated_miss_detected still requires independent semantic review before quality claims.",
        ],
    }
    output = args.output or args.evaluation.with_name("search_miss_taxonomy.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["miss_counts"], sort_keys=True))
    print(f"Saved miss taxonomy: {output}")


if __name__ == "__main__":
    main()
