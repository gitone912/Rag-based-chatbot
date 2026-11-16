"""Resumable RAGAS scoring for the 21 source-linked live holdout samples.

Run with ``python -m evals.ragas_holdout`` after creating
``evals/e2e_evaluation_results.json``. Scores are checkpointed after each
question so provider limits or an interrupted process do not discard work.
"""

import asyncio
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from openai import RateLimitError

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")

from evals.metrics import (  # noqa: E402
    COOLDOWN_MINI,
    COOLDOWN_STANDARD,
    _build_judge,
    _cooldown,
    compute_gold_evidence_recall,
    _prep_samples,
    get_judge_model,
)
from ragas.metrics.collections import (  # noqa: E402
    AnswerCorrectness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    Faithfulness,
)

SOURCE_PATH = ROOT / "evals" / "e2e_evaluation_results.json"
OUTPUT_PATH = ROOT / "evals" / "ragas_holdout_results.json"
SAMPLE_IDS = set(range(101, 122))
JUDGE_METRICS = (
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_recall",
    "answer_correctness",
)


def _save_checkpoint(data: dict) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        dir=OUTPUT_PATH.parent, prefix="ragas_holdout_", suffix=".tmp"
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())
        Path(temp_name).replace(OUTPUT_PATH)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def _metric_inputs(metric_name: str, sample: dict) -> dict:
    base = {"user_input": sample["question"]}
    if metric_name == "faithfulness":
        return {
            **base,
            "response": sample["actual_response"],
            "retrieved_contexts": sample["actual_contexts"],
        }
    if metric_name == "answer_relevancy":
        return {**base, "response": sample["actual_response"]}
    if metric_name in {"context_precision", "context_recall"}:
        return {
            **base,
            "reference": sample["reference"],
            "retrieved_contexts": sample["actual_contexts"],
        }
    return {
        **base,
        "response": sample["actual_response"],
        "reference": sample["reference"],
    }


def _retry_after_seconds(error: RateLimitError) -> float | None:
    headers = getattr(getattr(error, "response", None), "headers", {}) or {}
    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    match = re.search(r"try again in\s+(?:(\d+)m)?([\d.]+)s", str(error), re.I)
    if not match:
        return None
    return int(match.group(1) or 0) * 60 + float(match.group(2))


def _new_artifact(samples: list[dict]) -> dict:
    return {
        "evaluation": {
            "source_file": str(SOURCE_PATH.relative_to(ROOT)),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "judge_model": get_judge_model(),
            "question_count": len(samples),
            "question_ids": [sample["id"] for sample in samples],
            "split": "new source-linked holdout (ids 101-121)",
            "context_limit": 5,
            "context_char_limit": 600,
            "status": "running",
            "limitations": [
                "Questions are author-curated from six prepared documents and have not been independently reviewed.",
                "RAGAS scores are model-judged and may vary by judge model/version.",
                "Results apply to this saved live run and this corpus; they do not establish production performance.",
            ],
        },
        "metrics": {
            name: {"mean": None, "count": 0, "scores": []}
            for name in JUDGE_METRICS
        },
    }


def _load_artifact(samples: list[dict]) -> dict:
    expected_ids = [sample["id"] for sample in samples]
    if OUTPUT_PATH.exists():
        data = json.loads(OUTPUT_PATH.read_text(encoding="utf-8"))
        evaluation = data.get("evaluation", {})
        if (
            evaluation.get("question_ids") != expected_ids
            or evaluation.get("judge_model") != get_judge_model()
        ):
            raise ValueError(
                f"{OUTPUT_PATH} belongs to a different sample set or judge model; "
                "move it aside before starting a new evaluation."
            )
        for name in JUDGE_METRICS:
            data.setdefault("metrics", {}).setdefault(
                name, {"mean": None, "count": 0, "scores": []}
            )
        return data
    return _new_artifact(samples)


async def _run() -> dict:
    source = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    selected = [
        sample
        for sample in source.get("rag_samples", [])
        if sample.get("id") in SAMPLE_IDS
    ]
    if len(selected) != len(SAMPLE_IDS):
        raise ValueError("Expected all 21 source-linked holdout questions (IDs 101–121).")
    if any(
        not sample.get("actual_response")
        or not sample.get("actual_contexts")
        or sample.get("actual_error")
        for sample in selected
    ):
        raise ValueError("Every holdout sample must have a successful saved live response.")

    samples = _prep_samples({"rag_samples": selected})
    artifact = _load_artifact(samples)
    artifact["evaluation"]["status"] = "running"
    _save_checkpoint(artifact)
    judge_llm, embeddings = _build_judge()
    metric_objects = {
        "faithfulness": Faithfulness(llm=judge_llm),
        "answer_relevancy": AnswerRelevancy(llm=judge_llm, embeddings=embeddings),
        "context_precision": ContextPrecision(llm=judge_llm),
        "context_recall": ContextRecall(llm=judge_llm),
        "answer_correctness": AnswerCorrectness(llm=judge_llm, embeddings=embeddings),
    }

    for metric_index, name in enumerate(JUDGE_METRICS):
        entry = artifact["metrics"][name]
        existing = {row["id"] for row in entry["scores"]}
        metric = metric_objects[name]
        print(f"{name}: {len(existing)}/{len(samples)} already checkpointed", flush=True)
        if len(existing) == len(samples):
            continue
        for index, sample in enumerate(samples):
            if sample["id"] in existing:
                continue
            if index or existing:
                await _cooldown(COOLDOWN_MINI, f"{name} sample {sample['id']}")
            while True:
                try:
                    result = await metric.abatch_score([_metric_inputs(name, sample)])
                    break
                except RateLimitError as error:
                    retry_after = _retry_after_seconds(error)
                    if retry_after is None or retry_after > 1800:
                        artifact["evaluation"]["status"] = "partial"
                        artifact["evaluation"]["last_error_type"] = type(error).__name__
                        artifact["evaluation"]["updated_at_utc"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                        _save_checkpoint(artifact)
                        raise
                    print(
                        f"Provider quota reached; retrying in {int(retry_after)}s.",
                        flush=True,
                    )
                    remaining = retry_after
                    while remaining > 0:
                        interval = min(60, remaining)
                        await asyncio.sleep(interval)
                        remaining -= interval
                        if remaining > 0:
                            print(f"Provider retry in {int(remaining)}s.", flush=True)
                except Exception as error:
                    artifact["evaluation"]["status"] = "partial"
                    artifact["evaluation"]["last_error_type"] = type(error).__name__
                    artifact["evaluation"]["updated_at_utc"] = datetime.now(
                        timezone.utc
                    ).isoformat()
                    _save_checkpoint(artifact)
                    raise
            score = round(float(result[0].value), 4)
            entry["scores"].append({"id": sample["id"], "score": score})
            entry["scores"].sort(key=lambda row: row["id"])
            entry["count"] = len(entry["scores"])
            entry["mean"] = round(
                sum(row["score"] for row in entry["scores"]) / entry["count"], 4
            )
            artifact["evaluation"]["updated_at_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            _save_checkpoint(artifact)
            print(
                f"{name}: checkpointed {entry['count']}/{len(samples)} "
                f"(mean={entry['mean']:.4f})",
                flush=True,
            )
        remaining_metrics = JUDGE_METRICS[metric_index + 1 :]
        has_remaining_work = any(
            len(artifact["metrics"][next_name]["scores"]) < len(samples)
            for next_name in remaining_metrics
        )
        if has_remaining_work:
            await _cooldown(COOLDOWN_STANDARD, name)

    artifact["evaluation"]["status"] = "complete"
    artifact["evaluation"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    tool_scores = []
    for sample in selected:
        called = set(sample.get("actual_tools_called") or [])
        expected = set(sample.get("expected_tools") or [])
        union = called | expected
        tool_scores.append({
            "id": sample["id"],
            "score": round(len(called & expected) / len(union), 4) if union else 0.0,
        })
    artifact["metrics"]["tool_correctness"] = {
        "mean": round(sum(row["score"] for row in tool_scores) / len(tool_scores), 4),
        "count": len(tool_scores),
        "scores": tool_scores,
    }
    evidence1, evidence5 = compute_gold_evidence_recall({"rag_samples": selected})
    for name, frame, column in (
        ("evidence_recall_at_1", evidence1, "evidence_recall_at_1"),
        ("evidence_recall_at_5", evidence5, "evidence_recall_at_5"),
    ):
        rows = [
            {"id": sample["id"], "score": round(float(score), 4)}
            for sample, score in zip(selected, frame[column].tolist())
        ]
        artifact["metrics"][name] = {
            "mean": round(sum(row["score"] for row in rows) / len(rows), 4),
            "count": len(rows),
            "scores": rows,
        }
    _save_checkpoint(artifact)
    return artifact


if __name__ == "__main__":
    result = asyncio.run(_run())
    print(f"Complete: {OUTPUT_PATH}", flush=True)
    print(
        json.dumps(
            {name: value["mean"] for name, value in result["metrics"].items()},
            sort_keys=True,
        ),
        flush=True,
    )
