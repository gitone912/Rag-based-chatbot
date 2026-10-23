"""LLM-judged RAG metrics using the configured direct Groq judge.

Imports are lazy so the chat app remains usable when evaluation dependencies are
not installed. A narrow module alias works around Ragas 0.4 importing the
VertexAI chat class from its pre-split LangChain location.
"""
import asyncio
import importlib
import os
import sys
import types


def _prepare_ragas_imports():
    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
    except ModuleNotFoundError:
        from langchain_google_vertexai import ChatVertexAI

        compat = types.ModuleType("langchain_community.chat_models.vertexai")
        compat.ChatVertexAI = ChatVertexAI
        sys.modules[compat.__name__] = compat


def score_rows(rows: list[dict]) -> list[dict]:
    """Add Faithfulness, AnswerCorrectness, ContextPrecision and ContextRecall."""
    if not rows:
        return []
    _prepare_ragas_imports()
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory
    from ragas.metrics.collections import (
        AnswerCorrectness,
        ContextPrecision,
        ContextRecall,
        Faithfulness,
    )
    from app.config import settings

    api_key = os.getenv("JUDGE_GROQ") or settings.GROQ_API_KEY
    if not api_key:
        raise RuntimeError("Set JUDGE_GROQ or GROQ_API_KEY to run RAGAS scoring.")
    client = AsyncOpenAI(
        api_key=api_key,
        base_url="https://api.groq.com/openai/v1",
        timeout=45.0,
        max_retries=2,
    )
    llm = llm_factory("openai/gpt-oss-120b", provider="openai", client=client)
    metrics = {
        "faithfulness": Faithfulness(llm=llm),
        "answer_correctness": AnswerCorrectness(llm=llm, weights=[1.0, 0.0]),
        "context_precision": ContextPrecision(llm=llm),
        "context_recall": ContextRecall(llm=llm),
    }

    async def score_one(row: dict) -> dict:
        contexts = row.get("retrieved_contexts") or []
        if not row.get("answer") or not contexts:
            return {name: None for name in metrics}
        question = row["question"]
        answer = row["answer"]
        reference = row["ground_truth"]
        results = {
            "faithfulness": await metrics["faithfulness"].ascore(
                user_input=question, response=answer, retrieved_contexts=contexts
            ),
            "answer_correctness": await metrics["answer_correctness"].ascore(
                user_input=question, response=answer, reference=reference
            ),
            "context_precision": await metrics["context_precision"].ascore(
                user_input=question, reference=reference, retrieved_contexts=contexts
            ),
            "context_recall": await metrics["context_recall"].ascore(
                user_input=question, reference=reference, retrieved_contexts=contexts
            ),
        }
        return {name: result.value for name, result in results.items()}

    async def score_all():
        # Sequential requests keep the judge provider within ordinary rate limits.
        results = []
        for row in rows:
            results.append(await score_one(row))
        return results

    scores = asyncio.run(score_all())
    return [{**row, **score} for row, score in zip(rows, scores)]
