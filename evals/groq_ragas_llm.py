"""Strict JSON-schema adapter for Ragas metrics on Groq's OpenAI endpoint."""

import asyncio
from typing import Any

from openai import AsyncOpenAI, BadRequestError
from ragas.llms.base import InstructorBaseRagasLLM
from ragas.llms.base import track
from ragas.llms.base import LLMUsageEvent


def _groq_compatible_schema(model: type) -> dict[str, Any]:
    """Inline Pydantic refs and apply the constraints Groq strict mode requires."""
    raw = model.model_json_schema()
    definitions = raw.get("$defs", {})

    def normalize(node: Any) -> Any:
        if isinstance(node, list):
            return [normalize(item) for item in node]
        if not isinstance(node, dict):
            return node

        ref = node.get("$ref")
        if ref:
            prefix = "#/$defs/"
            if not ref.startswith(prefix) or ref[len(prefix):] not in definitions:
                raise ValueError(f"Unsupported Ragas schema reference: {ref}")
            resolved = normalize(definitions[ref[len(prefix):]])
            siblings = {key: value for key, value in node.items() if key != "$ref"}
            return {**resolved, **normalize(siblings)}

        result = {}
        for key, value in node.items():
            if key == "properties" and isinstance(value, dict):
                result[key] = {name: normalize(schema) for name, schema in value.items()}
            elif key in {"type", "required", "items", "enum", "anyOf", "oneOf", "const"}:
                result[key] = normalize(value)
        if result.get("type") == "object" or "properties" in result:
            properties = result.get("properties", {})
            result["properties"] = properties
            result["required"] = list(properties)
            result["additionalProperties"] = False
        return result

    return normalize(raw)


class GroqRagasLLM(InstructorBaseRagasLLM):
    """Ragas LLM that uses Groq strict JSON Schema and validates with Pydantic."""

    def __init__(self, client: AsyncOpenAI, model: str):
        self.client = client
        self.model = model

    async def agenerate(self, prompt: str, response_model: type):
        request = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            # Keep each structured result below the current Groq on-demand
            # output-token-per-minute ceiling. Ragas responses are short JSON
            # objects; schema validation rejects any truncated response.
            "max_completion_tokens": 600,
        }
        if self.model == "qwen/qwen3.8-27b":
            # Reasoning tokens can consume the JSON completion budget without
            # contributing to the structured Ragas result.
            request["reasoning_effort"] = "none"
        try:
            response = await self.client.chat.completions.create(
                **request,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": response_model.__name__,
                        "strict": True,
                        "schema": _groq_compatible_schema(response_model),
                    },
                },
            )
        except BadRequestError as error:
            # Some Groq model responses fail strict schema-constrained decoding
            # despite a valid request schema. Retry only that provider error in
            # JSON-object mode; Pydantic validation still enforces the Ragas
            # response shape before accepting the score.
            if getattr(error, "code", None) != "json_validate_failed":
                raise
            response = await self.client.chat.completions.create(
                **request,
                response_format={"type": "json_object"},
            )
        content = response.choices[0].message.content
        if not content:
            raise ValueError("Groq returned an empty Ragas evaluation response.")
        result = response_model.model_validate_json(content)
        track(
            LLMUsageEvent(
                provider="groq",
                model=self.model,
                llm_type="groq_strict_json_schema",
                num_requests=1,
                is_async=True,
            )
        )
        return result

    def generate(self, prompt: str, response_model: type):
        return asyncio.run(self.agenerate(prompt, response_model))
