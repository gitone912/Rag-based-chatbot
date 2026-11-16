"""Delete expired LangGraph checkpoints; schedule this command independently of API traffic."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from app.config import settings  # noqa: E402
from app.services.memory import open_memory_runtime  # noqa: E402


def main() -> int:
    if not settings.CHECKPOINT_DATABASE_URL:
        print("CHECKPOINT_DATABASE_URL is required for retention cleanup.", file=sys.stderr)
        return 2
    runtime = open_memory_runtime()
    if runtime is None:
        print("Durable memory runtime could not be initialized.", file=sys.stderr)
        return 2
    try:
        deleted = runtime.retention.cleanup_if_due()
        print(f"Expired memory threads deleted: {deleted}")
        return 0
    finally:
        runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
