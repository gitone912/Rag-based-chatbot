import logfire
from app.agents.state import AgentState
from app.services.retrieval.qdrant_service import search_enterprise_knowledge
from app.services.retrieval.ranking_service import (
    expand_query_terms,
    score_documents_bm25,
    rank_documents_bm25,
    reciprocal_rank_fusion,
    rerank_documents,
)
from app.config import settings

def retrieve_node(state: AgentState):
    """
    Performs vector search and semantic reranking for technical queries.
    """
    query = state["current_query"]
    
    
    # Standard Retrieval Logic
    with logfire.span("🔍 Knowledge Retrieval"):
        logfire.info("Searching Qdrant", query_characters=len(query))
        raw_results = search_enterprise_knowledge(
            query, limit=max(15, settings.RAG_CANDIDATE_LIMIT)
        )
        raw_contents = [doc["content"] for doc in raw_results if doc.get("content")]
        lexical_scores = score_documents_bm25(expand_query_terms(query), raw_contents)
        lexical_by_content = dict(lexical_scores)
        relevant_results = [
            doc for doc in raw_results
            if (
                float(doc.get("score") or 0.0) >= settings.RAG_MIN_RELEVANCE_SCORE
                or lexical_by_content.get(doc.get("content", ""), 0.0) > 0
            )
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
        lexical_ranked = rank_documents_bm25(expand_query_terms(query), doc_contents)
        ranked_scores = score_documents_bm25(expand_query_terms(query), doc_contents)
        top_score = ranked_scores[0][1] if ranked_scores else 0.0
        second_score = ranked_scores[1][1] if len(ranked_scores) > 1 else 0.0
        margin_ratio = (top_score - second_score) / top_score if top_score > 0 else 0.0
        confident_lexical = (
            top_score > 0 and margin_ratio >= settings.RAG_BM25_CONFIDENCE_MARGIN
        )
        
        with logfire.span("⚖️ Hybrid Reranking"):
            lexical_anchor_count = 0
            if confident_lexical:
                reranked_contents = lexical_ranked[:5]
                semantic_ranked = []
                logfire.info("Adaptive retrieval selected BM25 without cross-encoder")
            else:
                semantic_ranked = rerank_documents(
                    query, doc_contents, top_n=len(doc_contents)
                )
                fused_contents = reciprocal_rank_fusion(
                    [lexical_ranked, semantic_ranked], top_n=5
                )
                # RRF can bury an exact lexical match when semantic rank is
                # weak. Reserve two slots for BM25's strongest candidates,
                # then fill the remaining slots from the fused ranking.
                lexical_anchor_count = min(2, len(lexical_ranked))
                reranked_contents = list(dict.fromkeys(
                    lexical_ranked[:lexical_anchor_count] + fused_contents
                ))[:5]
            logfire.info(
                "Hybrid reranking complete",
                semantic_candidates=len(semantic_ranked),
                lexical_candidates=len(lexical_ranked),
                kept=len(reranked_contents),
                lexical_anchors=lexical_anchor_count,
                lexical_margin=round(margin_ratio, 4),
                cross_encoder_invoked=not confident_lexical,
            )
            
        source_by_content = {doc.get("content", ""): doc for doc in relevant_results}
        source_version_by_content = {doc.get("content", ""): doc for doc in relevant_results}
        formatted_docs = [
            (
                f"[{index}] SOURCE: {source_by_content.get(doc, {}).get('source', 'Indexed documentation')}"
                f" | source_id={source_version_by_content.get(doc, {}).get('source_id') or 'legacy'}"
                f" | version={source_version_by_content.get(doc, {}).get('source_version') or 'unknown'}"
                f" | chunk={source_version_by_content.get(doc, {}).get('chunk_index') or 0}"
                f" | ingested_at={source_version_by_content.get(doc, {}).get('ingested_at') or 'unknown'}\n"
                f"CONTENT: {doc}"
            )
            for index, doc in enumerate(reranked_contents, start=1)
        ]
    
    return {
        "documents": formatted_docs,
        "status": f"Found technical context.",
        "plan": state["plan"] + ["Context Retrieved"]
    }
