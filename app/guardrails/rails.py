import logfire
import re
from langchain_groq import ChatGroq
from nemoguardrails import RailsConfig, LLMRails

from app.config import settings
from app.guardrails.colang_rules import COLANG_CONTENT, YAML_CONTENT, RAIL_INDICATORS


_rails: LLMRails | None = None

_OFF_TOPIC_RE = re.compile(
    r"\b(?:coffee|espresso|cappuccino|latte|tea|recipe|cook(?:ing)?|bake(?:ing)?|"
    r"restaurant|weather|movie|music|sports?|dance|dancing|world history|math homework|"
    r"dinner|breakfast|lunch|food|travel|vacation)\b",
    re.IGNORECASE,
)
_TECHNICAL_SCOPE_RE = re.compile(
    r"\b(?:kubernetes|k8s|kubectl|kubelet|container|docker|helm|pod|cluster|"
    r"redis|databricks|intel|cpu|processor|fpga|sriov|nic|ethernet|network|"
    r"router|switch|subnet|dns|tcp|udp|vlan|bgp|firewall|load balancer|"
    r"linux|windows|server|database|rest api|programming|python|c\+\+)\b",
    re.IGNORECASE,
)
_JAILBREAK_RE = re.compile(
    r"\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above)\s+instructions\b|"
    r"\b(?:developer mode|unrestricted ai|bypass (?:your )?(?:rules|safety|guidelines)|"
    r"override (?:your )?(?:system prompt|safety filters))\b",
    re.IGNORECASE,
)
_OFF_TOPIC_RESPONSE = (
    "I can help with Kubernetes and the technical topics in this assistant’s scope. "
    "Please ask a related question."
)


def _is_obviously_off_topic(message: str) -> bool:
    """Deterministically catch clear off-topic requests before the model classifier."""
    normalized = re.sub(r"\bcoffee\s+lake\b", "", message, flags=re.IGNORECASE)
    if not _OFF_TOPIC_RE.search(normalized):
        return False

    # "Pod" is common in both coffee and Kubernetes contexts. For coffee-related
    # requests require a stronger IT anchor than that ambiguous word alone.
    if re.search(r"\bcoffee\b", normalized, re.IGNORECASE):
        strong_it_anchor = re.search(
            r"\b(?:kubernetes|k8s|kubectl|intel|cpu|processor|fpga|sriov|databricks|redis)\b",
            normalized,
            re.IGNORECASE,
        )
        return not bool(strong_it_anchor)

    return not bool(_TECHNICAL_SCOPE_RE.search(normalized))


def initialize_rails() -> None:
    """
    Build the NeMo LLMRails singleton at app startup.
    Uses the configured Groq model for intent classification at the gate.
    """
    global _rails

    guard_llm = ChatGroq(
        api_key=settings.GROQ_API_KEY,
        model=settings.GROQ_MODEL,
        temperature=0,
        max_tokens=256,
        reasoning_effort="low",
    )

    config = RailsConfig.from_content(
        colang_content=COLANG_CONTENT,
        yaml_content=YAML_CONTENT
    )

    _rails = LLMRails(config, llm=guard_llm)
    logfire.info(f"🛡️ NeMo Guardrails initialised ({settings.GROQ_MODEL}).")


def guard(message: str) -> tuple[bool, str | None]:
    """
    Run a user message through the NeMo rails gate.

    Returns:
        (True,  rail_response) — a rail fired; return this response immediately,
                                 skip the RAG pipeline entirely.
        (False, None)          — message is clean; proceed to LangGraph.
    """
    if _is_obviously_off_topic(message):
        logfire.info("Guardrails blocked an obvious off-topic request", query_characters=len(message))
        return True, _OFF_TOPIC_RESPONSE

    if _JAILBREAK_RE.search(message):
        logfire.info("Guardrails blocked a known jailbreak pattern", query_characters=len(message))
        return True, _OFF_TOPIC_RESPONSE

    if _rails is None:
        logfire.warning("⚠️ Guardrails not initialised — skipping gate.")
        return False, None

    with logfire.span("🛡️ Guardrails Check"):
        try:
            result = _rails.generate(messages=[{"role": "user", "content": message}])
        except Exception as exc:
            # Some Groq models may emit a tool-call-shaped response to NeMo even
            # though no tools are configured. Preserve the deterministic checks
            # above and let legitimate requests continue to document retrieval.
            logfire.warning("NeMo guardrail inference failed; continuing after deterministic checks", error_type=type(exc).__name__)
            return False, None

        # NeMo returns {'role': 'assistant', 'content': '...'} — extract text
        content = result.get("content", "") if isinstance(result, dict) else str(result)

        fired = any(indicator in content for indicator in RAIL_INDICATORS)

        if fired:
            logfire.info("Guardrails fired", query_characters=len(message))
            return True, content

        logfire.info("✅ Guardrails passed.")
        return False, None
