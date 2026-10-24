from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire
import re

# Provider-selected LLM: direct Groq by default, with optional Portkey support.
llm = get_langchain_llm(feature="planner")

_DOCUMENT_QUERY_TERMS = (
    "kubernetes", "kubectl", "redis", "databricks", "hpa", "vpa",
    "minikube", "metrics server", "parallelism", "completions",
    "restart policy", "sriov", "intel", "networking", "manifest",
    "pod", "job", "cli", "rest api",
)


def _is_document_query(message: str) -> bool:
    lowered = message.casefold()
    return any(term in lowered for term in _DOCUMENT_QUERY_TERMS)

def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""

    # First-turn questions that clearly target indexed technical content should
    # go directly to retrieval; don't let a stochastic planner label them as
    # conversational and skip the knowledge base.
    if not history.strip() and _is_document_query(user_message):
        logfire.info("Planner routed first-turn knowledge query to retrieval")
        return {
            "current_query": user_message,
            "status": "Technical research needed. Searching the knowledge base.",
            "plan": ["Intent: Technical", f"Search Term: {user_message}"],
        }
    
    prompt = f"""
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is a technical question about Kubernetes, Intel, networking, Databricks, or another topic indexed in the knowledge base that requires documentation, output a refined search query.
    
    Output ONLY 'CONVERSATIONAL' or the search query.
    """
    
    with logfire.span("🧠 Planner Decision"):
        response = llm.invoke(prompt)
        content = getattr(response, "content", None)
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("The planner LLM returned an empty or non-text response.")
        decision = re.sub(r"[\x00-\x1f\x7f]", " ", content).strip()
        logfire.info(
            "Planner completed",
            intent="conversational" if decision == "CONVERSATIONAL" else "technical",
            output_characters=len(decision),
        )
    
    if decision == "CONVERSATIONAL":
        # Substantive prompts must be grounded in indexed documentation.
        # Guardrails handle greetings and other fixed dialog replies before
        # the graph is entered; do not let a planner label become a general QA path.
        return {
            "current_query": user_message,
            "status": "Checking the indexed documentation for a relevant answer.",
            "plan": ["Intent: Document search", f"Search Term: {user_message}"],
        }
    
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": ["Intent: Technical", f"Search Term: {decision}"]
    }
