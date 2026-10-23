import os
import sys
import uuid
import json
import logfire

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings
from app.services.retrieval.embedding import embed_texts, get_embedding_dim
from app.ingestion.loaders.pdf import parse_pdf
from app.ingestion.loaders.html import parse_html
from app.ingestion.loaders.text import parse_text
from app.ingestion.chunking.splitter import chunk_text

logfire_export = os.getenv("LOGFIRE_SEND_TO_LOGFIRE", "false").lower()
if logfire_export not in {"true", "false", "if-token-present"}:
    logfire_export = "if-token-present"
logfire_send = {"true": True, "false": False}.get(logfire_export, "if-token-present")
logfire_options = {
    "service_name": "enterprise-ingestion-service",
    "send_to_logfire": logfire_send,
    "console": False,
}
logfire.configure(**logfire_options)

# Local folder where parsed + chunked JSON metadata is saved (replaces GCS processed bucket)
PROCESSED_DATA_DIR = "processed_data"

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
    check_compatibility=False,
    timeout=30,
)
_indexed_sources: set[str] | None = None


def save_processed_locally(data: dict, source_type: str, filename: str) -> str:
    """Save parsed chunk metadata as JSON in processed_data/<source_type>/."""
    folder = os.path.join(PROCESSED_DATA_DIR, source_type)
    os.makedirs(folder, exist_ok=True)
    dest = os.path.join(folder, f"{filename}.json")
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return dest


def source_is_indexed(filename: str) -> bool:
    """Return whether this collection already has chunks for the source filename."""
    global _indexed_sources
    if _indexed_sources is None:
        _indexed_sources = set()
        offset = None
        while True:
            points, offset = qdrant_client.scroll(
                collection_name=settings.QDRANT_COLLECTION,
                limit=256,
                offset=offset,
                with_payload=["source"],
                with_vectors=False,
            )
            _indexed_sources.update(
                str(point.payload["source"])
                for point in points
                if point.payload and point.payload.get("source")
            )
            if offset is None:
                break
    return filename in _indexed_sources


def process_file(file_path: str, filename: str, source_type: str):
    """Parse → chunk → save locally → embed → index, skipping existing sources."""
    with logfire.span("Processing File", file=filename, source=source_type):
        try:
            if source_is_indexed(filename):
                logfire.info(f"Skipping already indexed source: {filename}")
                return {"indexed": 0, "skipped": 1, "failed": 0}

            # 1. Extract text based on file extension
            ext = filename.lower().rsplit(".", 1)[-1]
            if ext == "pdf":
                full_text = parse_pdf(file_path)
            elif ext in ("html", "htm"):
                full_text = parse_html(file_path)
            elif ext == "txt":
                full_text = parse_text(file_path)
            elif ext in ("docx", "pptx"):
                from app.ingestion.loaders.office import parse_office
                full_text = parse_office(file_path)
            else:
                logfire.warning(f"Skipping unsupported file type: {filename}")
                return

            if not full_text or not full_text.strip():
                logfire.warning(f"No text extracted from {filename} — skipping.")
                return

            # 2. Chunk text
            chunks = chunk_text(full_text)
            if not chunks:
                return

            # 3. Save processed metadata locally
            processed_data = {
                "filename": filename,
                "source_type": source_type,
                "chunks": chunks,
            }
            local_path = save_processed_locally(processed_data, source_type, filename)
            logfire.info(f"Saved processed data → {local_path}")

            # 4. Embed and index in Qdrant
            with logfire.span("Vectorizing & Indexing"):
                embeddings = embed_texts(chunks)
                points = [
                    models.PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "text": chunk,
                            "source": filename,
                            "source_type": source_type,
                        },
                    )
                    for chunk, vector in zip(chunks, embeddings)
                ]

                qdrant_client.upsert(
                    collection_name=settings.QDRANT_COLLECTION,
                    points=points,
                )
                if _indexed_sources is not None:
                    _indexed_sources.add(filename)
                logfire.info(f"Indexed {len(points)} points to Qdrant from {filename}.")
            return {"indexed": len(points), "skipped": 0, "failed": 0}

        except Exception as e:
            logfire.error(f"Failed to process {filename}: {e}")
            return {"indexed": 0, "skipped": 0, "failed": 1}


def process_directory(dir_path: str, source_type: str):
    """Process every file in a directory and return indexed/skipped/failed totals."""
    with logfire.span("Scanning Directory", path=dir_path, source=source_type):
        files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
        logfire.info(f"Found {len(files)} files in {dir_path}.")
        totals = {"indexed": 0, "skipped": 0, "failed": 0}
        for filename in files:
            result = process_file(os.path.join(dir_path, filename), filename, source_type)
            for key in totals:
                totals[key] += result[key]
        return totals


def run_universal_ingestion(base_dir: str, explicit_source_type: str = None, wipe: bool = False):
    """
    Scan base_dir, map sub-folders to source types, and ingest all documents.
    Pass --wipe to drop and recreate the Qdrant collection before ingestion.
    """
    totals = {"indexed": 0, "skipped": 0, "failed": 0}
    with logfire.span("Universal Ingestion Started", base_directory=base_dir):

        # Wipe collection if requested
        if wipe:
            with logfire.span("Wiping Collection"):
                if qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
                    qdrant_client.delete_collection(settings.QDRANT_COLLECTION)
                    logfire.info(f"Collection '{settings.QDRANT_COLLECTION}' deleted.")

        # Recreate collection — dimension resolved at runtime after embedding model probe
        if not qdrant_client.collection_exists(settings.QDRANT_COLLECTION):
            dim = get_embedding_dim()
            qdrant_client.create_collection(
                collection_name=settings.QDRANT_COLLECTION,
                vectors_config=models.VectorParams(
                    size=dim,
                    distance=models.Distance.COSINE,
                ),
            )
            logfire.info(
                f"Created collection '{settings.QDRANT_COLLECTION}' "
                f"({dim}-dim, Cosine)."
            )

        # Route to sub-folders or treat the whole dir as one source
        subdirs = [
            d for d in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, d))
        ]

        if not subdirs:
            if explicit_source_type:
                source_type = explicit_source_type
            else:
                base_name = os.path.basename(os.path.normpath(base_dir)).lower()
                source_type = (
                    "true" if "true" in base_name
                    else "noisy" if "noisy" in base_name
                    else "general"
                )
            logfire.info(f"No sub-folders found — processing '{base_dir}' as '{source_type}'.")
            totals = process_directory(base_dir, source_type)
        else:
            for subdir in subdirs:
                source_type = (
                    "true" if "true" in subdir.lower()
                    else "noisy" if "noisy" in subdir.lower()
                    else subdir
                )
                result = process_directory(os.path.join(base_dir, subdir), source_type)
                for key in totals:
                    totals[key] += result[key]
    return totals


if __name__ == "__main__":
    # Usage:
    #   python -m app.ingestion.processor DATA --wipe
    #   python -m app.ingestion.processor DATA/true_data true
    wipe_requested = "--wipe" in sys.argv
    clean_args = [a for a in sys.argv if a != "--wipe"]

    target_dir = clean_args[1] if len(clean_args) > 1 else "DATA"
    explicit_type = clean_args[2] if len(clean_args) > 2 else None

    if not os.path.exists(target_dir):
        print(f"Error: path '{target_dir}' does not exist.")
        sys.exit(1)

    summary = run_universal_ingestion(target_dir, explicit_source_type=explicit_type, wipe=wipe_requested)
    print(
        "Ingestion complete: "
        f"{summary['indexed']} points indexed, "
        f"{summary['skipped']} files already indexed, "
        f"{summary['failed']} files failed."
    )
    if summary["failed"]:
        sys.exit(1)
    logfire.info("Ingestion job completed.")
