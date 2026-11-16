import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    # Keep embeddings aligned with the existing 768-dimensional Qdrant collection.
    # Select Gemini only when the collection has been deliberately reindexed to 3072 dimensions.
    EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "sentence-transformers").lower()

    # --- GEMINI EMBEDDINGS ---
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
    GEMINI_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "0"))

    # --- VECTOR DB (QDRANT) ---
    QDRANT_URL = os.getenv("QDRANT_CLUSTER_ENDPOINT")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    QDRANT_COLLECTION = "enterprise_rag"

    # --- REASONING ENGINE (GROQ) ---
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    GROQ_FALLBACK_API_KEY = os.getenv("GROQ_FALLBACK_API_KEY")
    GROQ_MAX_RETRIES = int(os.getenv("GROQ_MAX_RETRIES", "0"))
    GROQ_MAX_TOKENS = int(os.getenv("GROQ_MAX_TOKENS", "512"))
    RAG_MIN_RELEVANCE_SCORE = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.45"))
    RAG_MAX_CONTEXT_CHARS = int(os.getenv("RAG_MAX_CONTEXT_CHARS", "4000"))
    RAG_MAX_HISTORY_CHARS = int(os.getenv("RAG_MAX_HISTORY_CHARS", "2000"))
    RAG_HISTORY_MESSAGES = int(os.getenv("RAG_HISTORY_MESSAGES", "6"))
    RAG_CANDIDATE_LIMIT = int(os.getenv("RAG_CANDIDATE_LIMIT", "50"))
    RAG_BM25_CONFIDENCE_MARGIN = float(os.getenv("RAG_BM25_CONFIDENCE_MARGIN", "0.10"))
    # Groq browser search can inject large, unpredictable page payloads into a
    # request. Keep it opt-in so routine RAG misses do not trigger a token spike.
    WEB_SEARCH_ENABLED = os.getenv("WEB_SEARCH_ENABLED", "false").strip().lower() == "true"
    WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "3"))

    # Select the generation provider without changing the embedding/index setup.
    LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq").strip().lower()
    LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "45"))
    LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "2"))

    # Identity, durable memory, rate limits, and retention. Local development
    # defaults to anonymous, process-local behavior; production must opt in.
    AUTH_MODE = os.getenv("AUTH_MODE", "none").strip().lower()
    API_AUTH_TOKEN = os.getenv("API_AUTH_TOKEN")
    COGNITO_ISSUER = os.getenv("COGNITO_ISSUER", "").rstrip("/")
    COGNITO_CLIENT_ID = os.getenv("COGNITO_CLIENT_ID", "")
    CHECKPOINT_DATABASE_URL = os.getenv("CHECKPOINT_DATABASE_URL")
    MEMORY_RETENTION_DAYS = int(os.getenv("MEMORY_RETENTION_DAYS", "30"))
    RATE_LIMIT_REDIS_URL = os.getenv("RATE_LIMIT_REDIS_URL")
    RATE_LIMIT_PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "20"))
    RATE_LIMIT_PER_DAY = int(os.getenv("RATE_LIMIT_PER_DAY", "1000"))
    METRICS_BEARER_TOKEN = os.getenv("METRICS_BEARER_TOKEN")

    # --- LLM GATEWAY (PORTKEY) ---
    PORTKEY_API_KEY = os.getenv("PORTKEY_API_KEY")
    PORTKEY_CONFIG_ID = os.getenv("PORTKEY_CONFIG_ID")
    GROQ_SLUG =  "rag"     # primary Portkey virtual key
    GROQ_SLUG_2 = "brag"  # fallback Portkey virtual key

    
    # --- OBSERVABILITY ---
    LANGSMITH_TRACING = os.getenv("LANGSMITH_TRACING", "false")
    LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
    LANGSMITH_ENDPOINT = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

# Apply LangChain environment variables for automatic tracing
os.environ["LANGCHAIN_TRACING_V2"] = str(
    os.getenv("LANGSMITH_TRACING", "false").lower() == "true"
    and bool(os.getenv("LANGSMITH_API_KEY"))
).lower()
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGSMITH_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGSMITH_PROJECT", "rag_scale_test")
os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com")

settings = Settings()
