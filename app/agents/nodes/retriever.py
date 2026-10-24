import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import rerank_documents
from app.config import settings

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    
    
    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info("Searching Qdrant", query_characters=len(query))
        raw_results = search_enterprise_knowledge(query, limit=15)
        relevant_results = [
            doc for doc in raw_results
            if float(doc.get("score") or 0.0) >= settings.RAG_MIN_RELEVANCE_SCORE
        ]
        logfire.info(
            "Filtered Qdrant candidates",
            candidates=len(raw_results),
            relevant=len(relevant_results),
            min_score=settings.RAG_MIN_RELEVANCE_SCORE,
        )

        if not relevant_results:
            return {
                "documents": [],
                "status": "No relevant indexed documentation found.",
                "plan": state["plan"] + ["No relevant document match; checking the web fallback"],
            }
        
        doc_contents = [doc['content'] for doc in relevant_results if doc.get("content")]
        
        with logfire.span("⚖️ Semantic Reranking"):
            reranked_contents = rerank_documents(query, doc_contents, top_n=5)
            logfire.info("Reranking complete. Kept top 5 most relevant chunks.")
            
        source_by_content = {
            doc.get("content", ""): doc.get("source", "Indexed documentation")
            for doc in relevant_results
        }
        formatted_docs = [
            f"SOURCE: {source_by_content.get(doc, 'Indexed documentation')}\nCONTENT: {doc}"
            for doc in reranked_contents
        ]
    
    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
