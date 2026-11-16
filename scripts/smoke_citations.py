"""Run a small citation-format smoke against the local query API."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("LOGFIRE_IGNORE_NO_CONFIG", "1")
load_dotenv(ROOT / ".env")

from evals.pipeline import load_golden_dataset  # noqa: E402


def main() -> int:
    requested_ids = [int(value) for value in sys.argv[1:]] or [1, 4, 103, 115]
    samples = {
        sample["id"]: sample for sample in load_golden_dataset()["rag_samples"]
    }
    missing = [sample_id for sample_id in requested_ids if sample_id not in samples]
    if missing:
        print(f"Unknown golden IDs: {missing}")
        return 2

    base_url = os.getenv("BACKEND_URL", "http://127.0.0.1:8000").rstrip("/")
    token = os.getenv("EVAL_AUTH_TOKEN") or os.getenv("API_AUTH_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    citation_pattern = re.compile(r"(?<!\w)\[(\d+)\]")

    for sample_id in requested_ids:
        sample = samples[sample_id]
        try:
            response = requests.post(
                f"{base_url}/query",
                json={
                    "q": sample["question"],
                    "thread_id": f"citation-smoke-{sample_id}",
                },
                headers=headers,
                timeout=90,
            )
        except requests.RequestException as exc:
            print(f"id={sample_id} transport_error={type(exc).__name__}")
            return 1

        if not response.ok:
            try:
                detail = response.json().get("detail", {})
                code = detail.get("code") if isinstance(detail, dict) else None
            except (ValueError, AttributeError):
                code = None
            print(f"id={sample_id} http={response.status_code} error_code={code}")
            return 1

        data = response.json()
        answer = data.get("answer") or ""
        citation_ids = [int(value) for value in citation_pattern.findall(answer)]
        sources = data.get("sources") or []
        print(
            f"id={sample_id} citations={len(citation_ids)} "
            f"ids_in_range={bool(citation_ids) and all(1 <= value <= len(sources) for value in citation_ids)} "
            f"citation_refusal={answer.startswith('I found relevant documentation')} "
            f"source_count={len(sources)}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
