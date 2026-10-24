import logfire
import re
from app.agents.state import AgentState
from app.gateway import get_langchain_llm, extract_cache_status
from app.services.retrieval.web_search import search_web_answer


llm = get_langchain_llm(feature="responder")

_WEB_FALLBACK_MARKER = "INSUFFICIENT_DOCUMENT_CONTEXT"


def _web_fallback(query: str, state: AgentState) -> dict:
    """Use Groq browser search when indexed documents are insufficient."""
    indexed_context = "\n\n".join(state.get("documents") or [])
    answer, web_results = search_web_answer(query, indexed_context=indexed_context)
    if not answer or not web_results:
        refusal = (
            "The indexed documents don’t contain enough information to answer this, "
            "and web search didn’t return usable sources."
        )
        return {
            "final_answer": refusal,
            "status": "Document context insufficient; web search unavailable or empty.",
            "plan": state["plan"] + ["Web fallback: no usable results"],
            "documents": [],
            "messages": [{"role": "assistant", "content": refusal}],
        }

    formatted_sources = []
    for index, item in enumerate(web_results, start=1):
        source = (
            f"WEB SOURCE [{index}]: {item['title']}\n"
            f"URL: [{item['url']}]({item['url']})"
        )
        formatted_sources.append(source)
    source_list = "\n".join(
        f"- [{index}] [{item['title']}]({item['url']})"
        for index, item in enumerate(web_results, start=1)
    )
    # Replace Groq's opaque line markers with compact bibliography numbers and
    # keep one clickable Sources section rather than repeating it twice.
    answer_body = re.sub(
        r"[\u3010\u3016]?\s*(\d+)\S{0,3}L\d+-L\d+\S{0,3}",
        r"[\1]",
        answer,
    )
    answer_body = re.split(r"(?im)^\s*\*{0,2}sources\*{0,2}\s*:?.*$", answer_body)[0].rstrip()
    content = f"{answer_body}\n\n**Sources**\n{source_list}"

    return {
        "final_answer": content,
        "status": "Answered using web search because indexed documents were insufficient.",
        "plan": state["plan"] + ["Indexed documents insufficient", "Web search fallback used"],
        "documents": formatted_sources,
        "messages": [{"role": "assistant", "content": content}],
    }


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the same provider-selected LangChain client as the planner. Retrieved
    chunks and conversation history remain part of the generation prompt.
    """
    query = state["current_query"]

    if query != "CONVERSATIONAL" and not state.get("documents"):
        logfire.info("No relevant indexed docs; using web fallback.")
        return _web_fallback(query, state)

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = f"""
        You are a friendly and helpful Enterprise AI Assistant.
        Answer the user's latest message using the CONVERSATION HISTORY below.
        Be concise: answer in 1–3 sentences, with no more than 80 words unless
        the user explicitly asks for a detailed explanation. Avoid preambles.

        CONVERSATION HISTORY:
        {history_str}

        LATEST MESSAGE:
        "{user_msg}"
        """
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = f"""
        You are a concise documentation assistant focused on Kubernetes.
        Answer only with facts supported by the TECHNICAL CONTEXT. Do not use
        general knowledge or fill gaps. If the context does not contain enough
        information to answer, return exactly { _WEB_FALLBACK_MARKER } and nothing
        else so a web search can be tried. Be direct and concise: use at most 4
        bullets and 120 words unless the user explicitly asks for more detail.
        Do not repeat the question or add a long preamble.

        TECHNICAL CONTEXT:
        {full_context}

        CONVERSATION HISTORY:
        {history_str}

        USER QUESTION:
        "{user_msg}"
        """

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = llm.invoke(prompt)
            content = getattr(response, "content", None)
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("The LLM returned an empty or non-text response.")
            if query != "CONVERSATIONAL" and content.strip() == _WEB_FALLBACK_MARKER:
                logfire.info("Responder found document context insufficient; using web fallback.")
                return _web_fallback(user_msg, state)
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("Response synthesized via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM generation failed | type={type(e).__name__}")
            raise
