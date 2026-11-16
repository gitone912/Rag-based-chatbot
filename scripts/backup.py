"""Create a local/off-site backup of Qdrant and the Postgres memory database."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse, unquote

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.config import settings  # noqa: E402


def _qdrant_backup(destination: Path) -> dict:
    if not settings.QDRANT_URL:
        raise RuntimeError("QDRANT_CLUSTER_ENDPOINT is required for a Qdrant backup.")
    base = settings.QDRANT_URL.rstrip("/")
    parsed_endpoint = urlparse(base)
    if parsed_endpoint.scheme != "https" and parsed_endpoint.hostname not in {
        "localhost", "127.0.0.1", "qdrant"
    }:
        raise RuntimeError("Remote Qdrant backups require an HTTPS endpoint.")
    collection = settings.QDRANT_COLLECTION
    headers = {"api-key": settings.QDRANT_API_KEY} if settings.QDRANT_API_KEY else {}
    response = requests.post(
        f"{base}/collections/{collection}/snapshots",
        headers=headers,
        timeout=120,
    )
    response.raise_for_status()
    snapshot_name = response.json()["result"]["name"]
    snapshot_response = requests.get(
        f"{base}/collections/{collection}/snapshots/{snapshot_name}",
        headers=headers,
        timeout=300,
        stream=True,
    )
    snapshot_response.raise_for_status()
    snapshot_path = destination / f"qdrant-{collection}-{snapshot_name}"
    digest = hashlib.sha256()
    with snapshot_path.open("wb") as output:
        for block in snapshot_response.iter_content(chunk_size=1024 * 1024):
            if block:
                output.write(block)
                digest.update(block)
    return {
        "kind": "qdrant_snapshot",
        "collection": collection,
        "snapshot_name": snapshot_name,
        "path": snapshot_path.name,
        "size_bytes": snapshot_path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _postgres_backup(destination: Path) -> dict | None:
    database_url = settings.CHECKPOINT_DATABASE_URL
    if not database_url:
        return None
    parsed = urlparse(database_url)
    output_path = destination / "conversation-memory.dump"
    args = ["pg_dump", "--format=custom", "--no-owner", "--no-privileges"]
    if parsed.hostname:
        args += ["--host", parsed.hostname]
    if parsed.port:
        args += ["--port", str(parsed.port)]
    if parsed.username:
        args += ["--username", unquote(parsed.username)]
    if parsed.path.strip("/"):
        args += ["--dbname", unquote(parsed.path.strip("/"))]
    query = dict(item.split("=", 1) for item in parsed.query.split("&") if "=" in item)
    args += ["--file", str(output_path)]
    env = os.environ.copy()
    if query.get("sslmode"):
        env["PGSSLMODE"] = query["sslmode"]
    if parsed.password:
        env["PGPASSWORD"] = unquote(parsed.password)
    subprocess.run(args, env=env, check=True, capture_output=True, text=True)
    digest = hashlib.sha256(output_path.read_bytes()).hexdigest()
    return {
        "kind": "postgres_dump",
        "path": output_path.name,
        "size_bytes": output_path.stat().st_size,
        "sha256": digest,
    }


def _upload_s3(destination: Path, s3_uri: str, kms_key_id: str | None = None) -> None:
    import boto3

    bucket, _, prefix = s3_uri[5:].partition("/")
    prefix = prefix.rstrip("/")
    client = boto3.client("s3")
    for file_path in destination.iterdir():
        if file_path.is_file():
            key = f"{prefix}/{destination.name}/{file_path.name}".lstrip("/")
            extra_args = {"ServerSideEncryption": "aws:kms"}
            if kms_key_id:
                extra_args["SSEKMSKeyId"] = kms_key_id
            client.upload_file(str(file_path), bucket, key, ExtraArgs=extra_args)
            print(f"Uploaded s3://{bucket}/{key}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "backups",
        help="Directory where backup artifacts and their manifest are written.",
    )
    parser.add_argument(
        "--s3-uri",
        help="Optional s3://bucket/prefix destination; uses the AWS default credential chain.",
    )
    parser.add_argument(
        "--s3-kms-key-id",
        help="Optional customer-managed KMS key ID or ARN; S3 uploads always request SSE-KMS.",
    )
    args = parser.parse_args()

    if args.s3_uri:
        parsed_s3 = urlparse(args.s3_uri)
        if parsed_s3.scheme != "s3" or not parsed_s3.netloc:
            raise ValueError("--s3-uri must be an s3://bucket/prefix URI.")
    if args.s3_kms_key_id and not args.s3_uri:
        raise ValueError("--s3-kms-key-id requires --s3-uri.")
    if settings.APP_ENV == "production" and not args.s3_uri:
        raise RuntimeError("Production backups must be uploaded to an encrypted S3 destination.")
    if settings.APP_ENV == "production" and not args.s3_kms_key_id:
        raise RuntimeError("Production backups require an explicit customer-managed KMS key ID.")

    timestamp = datetime.now(timezone.utc)
    destination = args.output.resolve() / timestamp.strftime("%Y%m%dT%H%M%SZ")
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    if os.name != "nt":
        os.chmod(destination, 0o700)
    artifacts = [_qdrant_backup(destination)]
    postgres_artifact = _postgres_backup(destination)
    if postgres_artifact:
        artifacts.append(postgres_artifact)
    manifest = {
        "created_at_utc": timestamp.isoformat(),
        "collection": settings.QDRANT_COLLECTION,
        "artifacts": artifacts,
        "offsite_destination": args.s3_uri,
    }
    manifest_path = destination / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if args.s3_uri:
        _upload_s3(destination, args.s3_uri, args.s3_kms_key_id)
    print(f"Backup complete: {destination}", flush=True)


if __name__ == "__main__":
    main()
