from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.agents.state import AgentState
from app.agents.nodes.planner import planner_node
from app.agents.nodes.retriever import retrieve_node
from app.agents.nodes.responder import generate_node


def route_planner(state: AgentState):
    """
    Routes the workflow based on the planner's decision.
    """
    if state["current_query"] == "CONVERSATIONAL":
        return "responder"
    return "retriever"

def build_rag_agent(checkpointer=None):
    """Compile the graph with an injected durable or local checkpointer."""
    workflow = StateGraph(AgentState)
    workflow.add_node("planner", planner_node)
    workflow.add_node("retriever", retrieve_node)
    workflow.add_node("responder", generate_node)
    workflow.set_entry_point("planner")
    workflow.add_conditional_edges(
        "planner",
        route_planner,
        {"retriever": "retriever", "responder": "responder"},
    )
    workflow.add_edge("retriever", "responder")
    workflow.add_edge("responder", END)
    return workflow.compile(
        checkpointer=checkpointer if checkpointer is not None else MemorySaver()
    )


local_memory_saver = MemorySaver()
rag_agent = build_rag_agent(local_memory_saver)


def configure_checkpointer(checkpointer) -> None:
    """Recompile the agent with the process's durable checkpoint store."""
    global rag_agent
    rag_agent = build_rag_agent(checkpointer)


def delete_local_thread(thread_id: str) -> None:
    local_memory_saver.delete_thread(thread_id)

