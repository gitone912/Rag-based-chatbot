from types import SimpleNamespace

import pytest

from app.config import settings
from app.gateway import client as gateway_client


def test_direct_groq_client_uses_configured_key_model_and_bounded_retries(monkeypatch):
    captured = {}

    def fake_chatgroq(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(settings, "GROQ_MODEL", "openai/gpt-oss-120b")
    monkeypatch.setattr(gateway_client, "ChatGroq", fake_chatgroq)
    monkeypatch.setattr(
        gateway_client,
        "get_portkey_client",
        lambda: pytest.fail("Groq selection must never initialize Portkey"),
    )

    gateway_client.get_langchain_llm(feature="test")

    assert captured["api_key"] == "unit-test-key"
    assert captured["model"] == "openai/gpt-oss-120b"
    assert captured["max_tokens"] == settings.GROQ_MAX_TOKENS
    assert captured["reasoning_effort"] == "low"
    assert captured["timeout"] == settings.LLM_TIMEOUT_SECONDS
    assert captured["max_retries"] == settings.LLM_MAX_RETRIES
    assert captured["temperature"] == 0


def test_direct_groq_requires_api_key(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)

    with pytest.raises(RuntimeError, match="requires GROQ_API_KEY"):
        gateway_client.get_langchain_llm()


def test_responder_sends_context_and_history_and_returns_expected_shape(monkeypatch):
    from app.agents.nodes import responder

    captured = {}
    result_message = SimpleNamespace(content="Use the retrieved procedure.", response_metadata={})

    class FakeLLM:
        def invoke(self, prompt):
            captured["prompt"] = prompt
            return result_message

    monkeypatch.setattr(responder, "llm", FakeLLM())
    state = {
        "messages": [
            {"role": "user", "content": "Earlier question"},
            {"role": "assistant", "content": "Earlier answer"},
            {"role": "user", "content": "How do I complete this operation?"},
        ],
        "current_query": "complete operation steps",
        "documents": ["CONTENT: The system uses the staged completion operation."],
        "plan": ["Start", "Context Retrieved"],
    }

    actual = responder.generate_node(state)

    assert "The system uses the staged completion operation." in captured["prompt"]
    assert "Earlier answer" in captured["prompt"]
    assert "How do I complete this operation?" in captured["prompt"]
    assert actual["final_answer"] == "Use the retrieved procedure."
    assert actual["messages"] == [{"role": "assistant", "content": "Use the retrieved procedure."}]


def test_responder_keeps_its_existing_temperature(monkeypatch):
    captured = {}
    monkeypatch.setattr(settings, "LLM_PROVIDER", "groq")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "unit-test-key")
    monkeypatch.setattr(gateway_client, "ChatGroq", lambda **kwargs: captured.update(kwargs) or object())

    gateway_client.get_langchain_llm(feature="responder")

    assert captured["temperature"] == 0.1


def test_first_turn_indexed_question_cannot_be_misclassified_as_conversational(monkeypatch):
    from app.agents.nodes import planner

    class UnexpectedLLMCall:
        def invoke(self, prompt):
            pytest.fail("Indexed first-turn questions should route to retrieval directly")

    monkeypatch.setattr(planner, "llm", UnexpectedLLMCall())
    result = planner.planner_node({
        "messages": [{"role": "user", "content": "How do I install Metrics Server for Kubernetes?"}],
        "current_query": "How do I install Metrics Server for Kubernetes?",
        "documents": [],
        "plan": ["Start"],
    })

    assert result["current_query"] == "How do I install Metrics Server for Kubernetes?"
    assert result["plan"][0] == "Intent: Technical"


@pytest.mark.parametrize("content", [None, "", "  ", ["not", "text"]])
def test_responder_rejects_empty_or_non_text_content(monkeypatch, content):
    from app.agents.nodes import responder

    class FakeLLM:
        def invoke(self, prompt):
            return SimpleNamespace(content=content)

    monkeypatch.setattr(responder, "llm", FakeLLM())
    state = {
        "messages": [{"role": "user", "content": "question"}],
        "current_query": "CONVERSATIONAL",
        "documents": [],
        "plan": ["Start"],
    }

    with pytest.raises(RuntimeError, match="empty or non-text"):
        responder.generate_node(state)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (SimpleNamespace(status_code=401), "provider_authentication"),
        (SimpleNamespace(status_code=403), "provider_permission"),
        (SimpleNamespace(status_code=404), "provider_model_unavailable"),
        (SimpleNamespace(status_code=429), "provider_rate_limit"),
        (TimeoutError("provider timeout"), "provider_timeout"),
        (SimpleNamespace(status_code=503), "provider_server_error"),
    ],
)
def test_provider_errors_are_classified_without_exposing_payload(error, expected):
    assert gateway_client.classify_provider_error(error) == expected


def test_invalid_provider_is_reported(monkeypatch):
    monkeypatch.setattr(settings, "LLM_PROVIDER", "unknown")

    with pytest.raises(ValueError, match="Unsupported LLM_PROVIDER"):
        gateway_client.get_langchain_llm()
