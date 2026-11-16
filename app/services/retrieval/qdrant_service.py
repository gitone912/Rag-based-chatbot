import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embedding import embed_query


# Initialize Qdrant Client
client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    # Avoid an eager network/version probe during API module import.
    check_compatibility=False,
)

def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """
    try:
        query_vector = embed_query(query)

        # Using query_points - the modern standard for Qdrant
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True # JSON
        )

        results = []
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),
                "source": res.payload.get("source", "Unknown"),
                "source_id": res.payload.get("source_id", ""),
                "source_version": res.payload.get("source_version", ""),
                "content_sha256": res.payload.get("content_sha256", ""),
                "ingested_at": res.payload.get("ingested_at", ""),
                "chunk_index": res.payload.get("chunk_index"),
                "ingestion_schema_version": res.payload.get("ingestion_schema_version"),
                "score": res.score
            })
        
        return results
    except Exception as e:
        logfire.error(
            "Qdrant search failed",
            error_type=type(e).__name__,
        )
        raise RuntimeError("Qdrant vector retrieval failed.") from e
