"""Backfill source version metadata onto matching existing Qdrant vectors.

The command defaults to a dry run. It refuses to mutate the collection unless
every current source chunk already has an exact text match in the index.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from qdrant_client.http import models  # noqa: E402

from app.config import settings  # noqa: E402
from app.ingestion.chunking.splitter import chunk_text  # noqa: E402
from app.services.retrieval.qdrant_service import client  # noqa: E402


def _parse(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        from app.ingestion.loaders.pdf import parse_pdf

        return parse_pdf(str(path))
    if ext in {".html", ".htm"}:
        from app.ingestion.loaders.html import parse_html

        return parse_html(str(path))
    if ext == ".txt":
        from app.ingestion.loaders.text import parse_text

        return parse_text(str(path))
    if ext in {".docx", ".pptx"}:
        from app.ingestion.loaders.office import parse_office

        return parse_office(str(path))
    raise ValueError(f"Unsupported source type: {path.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=ROOT / "DATA" / "true_data")
    parser.add_argument("--apply", action="store_true", help="Write metadata and prune duplicate/stale point IDs.")
    args = parser.parse_args()
    source_dir = args.source_dir.resolve()
    timestamp = datetime.now(timezone.utc).isoformat()

    points = []
    offset = None
    while True:
        batch, offset = client.scroll(
            collection_name=settings.QDRANT_COLLECTION,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        points.extend(batch)
        if offset is None:
            break

    updates: list[tuple[object, dict]] = []
    removals: list[object] = []
    missing: list[str] = []
    summaries = []
    for path in sorted(item for item in source_dir.iterdir() if item.is_file()):
        raw = path.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        source_version = f"sha256:{digest}"
        source_id = f"true/{path.name}"
        chunks = chunk_text(_parse(path))
        candidates = [
            point for point in points
            if (point.payload or {}).get("source") == path.name
            and (point.payload or {}).get("source_type") == "true"
        ]
        selected = []
        for index, text in enumerate(chunks):
            matches = [point for point in candidates if (point.payload or {}).get("text") == text]
            if not matches:
                missing.append(f"{path.name} chunk {index}")
                continue
            preferred = next(
                (point for point in matches if (point.payload or {}).get("source_version") == source_version),
                matches[0],
            )
            selected.append(preferred)
            updates.append((preferred.id, {
                "source": path.name,
                "source_id": source_id,
                "source_version": source_version,
                "content_sha256": digest,
                "ingested_at": timestamp,
                "chunk_index": index,
                "ingestion_schema_version": 2,
                "source_type": "true",
            }))
        keep_ids = {point.id for point in selected}
        removals.extend(point.id for point in candidates if point.id not in keep_ids)
        summaries.append({"source": path.name, "chunks": len(chunks), "matched": len(selected), "duplicate_or_stale": sum(point.id not in keep_ids for point in candidates)})

    print(json.dumps({
        "collection": settings.QDRANT_COLLECTION,
        "mode": "apply" if args.apply else "dry_run",
        "source_count": len(summaries),
        "point_count": len(points),
        "metadata_updates": len(updates),
        "duplicate_or_stale_deletions": len(removals),
        "missing_chunks": missing,
        "sources": summaries,
    }, indent=2))
    if missing:
        raise SystemExit("Refusing metadata backfill: some source chunks have no exact indexed text match.")
    if not args.apply:
        return

    for point_id, payload in updates:
        client.set_payload(
            collection_name=settings.QDRANT_COLLECTION,
            payload=payload,
            points=[point_id],
        )
    for start in range(0, len(removals), 100):
        client.delete(
            collection_name=settings.QDRANT_COLLECTION,
            points_selector=models.PointIdsList(points=removals[start : start + 100]),
            wait=True,
        )
    print("Backfill complete.")


if __name__ == "__main__":
    main()
