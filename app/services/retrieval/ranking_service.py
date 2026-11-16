import time
import math
import re
from collections import Counter
import logfire
from flashrank import Ranker, RerankRequest

# Lazy initialization - Ranker is loaded on first use to ensure logfire.configure() has run
_ranker = None


def _get_ranker() -> Ranker:
    """
    Initializes the FlashRank engine lazily. 
    FlashRank uses a local ONNX model (ms-marco-MiniLM-L-6-v2) for ultra-fast reranking.
    """
    global _ranker
    if _ranker is None:
        logfire.info("🧠 Initializing FlashRank Model (TinyBERT) locally...")
        try:
            # We use a specific cache directory to avoid permission issues in production
            _ranker = Ranker(cache_dir="/tmp/flashrank")
        except Exception:
            _ranker = Ranker()
    return _ranker



def rerank_documents(query: str, documents: list[str], top_n: int = 5) -> list[str]:
    """
    Refines retrieval results by re-scoring documents against the query semantically.
    
    Why FlashRank? 
    Standard vector search (Cosine Similarity) is fast but mathematically "fuzzy."
    FlashRank uses a Cross-Encoder approach which is much more precise but usually slow.
    FlashRank solves this by using highly optimized, quantized ONNX models locally.
    """
    if not documents:
        return []

    start_time = time.time()
    logfire.info(f"📡 [Reranker] Sending {len(documents)} docs to FlashRank Cross-Encoder...")

    try:
        ranker = _get_ranker()
        
        # FlashRank expects a list of dictionaries with 'id' and 'text'
        passages = [
            {"id": i, "text": doc}
            for i, doc in enumerate(documents)
        ]

        request = RerankRequest(query=query, passages=passages)
        results = ranker.rerank(request)
        
        # Results are returned sorted by highest semantic score first
        reranked_docs = []
        for res in results[:top_n]:
            reranked_docs.append(res['text'])

        duration = time.time() - start_time
        top_score = results[0]['score'] if results else 'N/A'
        logfire.info(f"✅ [Reranker] Done in {duration:.2f}s. Top semantic score: {top_score}")
        
        return reranked_docs

    except Exception as e:
        logfire.error(f"❌ [Reranker] Semantic Reranking Failed: {e}")
        # Fallback to the original Qdrant order to ensure the user still gets an answer
        return documents[:top_n]


_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
    "for", "from", "how", "in", "is", "it", "of", "on", "or", "the", "this",
    "to", "what", "when", "where", "which", "with",
}

_QUERY_SYNONYMS = {
    "status": ("state", "phase"),
    "statuses": ("states", "phases", "active", "completed", "failed"),
    # State/status terminology is not interchangeable lexically, but users
    # often phrase the exact same lookup either way. Include lifecycle values
    # because terse status questions otherwise rank introductory chunks above
    # the operational reference passage.
    "state": ("status", "phase", "active", "completed", "failed"),
    "states": ("statuses", "phases", "active", "completed", "failed"),
    "phase": ("state", "status", "active", "completed", "failed"),
    "phases": ("states", "statuses", "active", "completed", "failed"),
    "active": ("state", "status", "running"),
    "completed": ("state", "status", "succeeded", "successful"),
    "failed": ("state", "status", "failure", "error"),
    "report": ("show", "list"),
    "reports": ("shows", "lists"),
    "start": ("launch", "run", "create"),
    "starts": ("launches", "runs", "creates"),
    "install": ("setup", "deploy"),
    "installed": ("setup", "deployed"),
}


def _tokens(text: str) -> list[str]:
    return [
        token for token in re.findall(r"[a-z0-9]+", text.casefold())
        if token not in _STOP_WORDS and len(token) > 1
    ]


def expand_query_terms(query: str) -> str:
    """Add conservative lexical variants for common operational vocabulary."""
    terms = _tokens(query)
    expanded = list(terms)
    for term in terms:
        expanded.extend(_QUERY_SYNONYMS.get(term, ()))
    return " ".join(dict.fromkeys(expanded))


def score_documents_bm25(query: str, documents: list[str]) -> list[tuple[str, float]]:
    """Return lexical candidates and BM25 scores in descending score order."""
    if not documents:
        return []

    query_terms = set(_tokens(query))
    if not query_terms:
        return [(document, 0.0) for document in documents]

    tokenized = [_tokens(document) for document in documents]
    average_length = sum(map(len, tokenized)) / max(len(tokenized), 1)
    document_frequency = Counter(term for terms in tokenized for term in set(terms))
    k1, b = 1.5, 0.75
    scores = []
    for index, terms in enumerate(tokenized):
        frequencies = Counter(terms)
        length_norm = 1 - b + b * len(terms) / max(average_length, 1)
        score = 0.0
        for term in query_terms:
            frequency = frequencies[term]
            if not frequency:
                continue
            df = document_frequency[term]
            inverse_frequency = math.log1p(
                (len(documents) - df + 0.5) / (df + 0.5)
            )
            score += inverse_frequency * frequency * (k1 + 1) / (
                frequency + k1 * length_norm
            )
        scores.append((score, index))

    scores.sort(key=lambda item: (-item[0], item[1]))
    return [(documents[index], score) for score, index in scores]


def rank_documents_bm25(query: str, documents: list[str]) -> list[str]:
    """Rank retrieved candidates lexically to recover exact names and commands."""
    return [document for document, _ in score_documents_bm25(query, documents)]


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, top_n: int = 5, rank_constant: int = 10
) -> list[str]:
    """Fuse rankings; equal scores follow the first list's stable order."""
    fused_scores: dict[str, float] = {}
    first_seen: dict[str, int] = {}
    seen_counter = 0
    for ranked in ranked_lists:
        for rank, document in enumerate(ranked, start=1):
            if document not in first_seen:
                first_seen[document] = seen_counter
                seen_counter += 1
            fused_scores[document] = fused_scores.get(document, 0.0) + 1.0 / (
                rank_constant + rank
            )

    fused = sorted(
        fused_scores,
        key=lambda document: (-fused_scores[document], first_seen[document]),
    )
    return fused[:top_n]
