"""Postgres-backed LangGraph checkpoints and per-identity retention controls."""

from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from app.config import settings
from app.security.auth import identity_scope


@dataclass
class MemoryRuntime:
    pool: Any
    checkpointer: Any
    retention: "MemoryRetention"

    def close(self) -> None:
        self.pool.close()


class MemoryRetention:
    """Track active checkpoint threads and delete them after their TTL."""

    def __init__(self, pool: Any, retention_days: int) -> None:
        self.pool = pool
        self.retention_days = max(1, retention_days)
        self._cleanup_lock = threading.Lock()
        self._last_cleanup = 0.0
        with self.pool.connection() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_memory_threads (
                    thread_id TEXT PRIMARY KEY,
                    identity_hash TEXT NOT NULL,
                    last_activity TIMESTAMPTZ NOT NULL DEFAULT now(),
                    expires_at TIMESTAMPTZ NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rag_memory_threads_expiry_idx "
                "ON rag_memory_threads (expires_at)"
            )

    def touch(self, subject: str, thread_id: str) -> None:
        identity_hash = identity_scope(subject)
        with self.pool.connection() as connection:
            connection.execute(
                """
                INSERT INTO rag_memory_threads (thread_id, identity_hash, last_activity, expires_at)
                VALUES (%s, %s, now(), now() + (%s * interval '1 day'))
                ON CONFLICT (thread_id) DO UPDATE SET
                    identity_hash = EXCLUDED.identity_hash,
                    last_activity = now(),
                    expires_at = EXCLUDED.expires_at
                """,
                (thread_id, identity_hash, self.retention_days),
            )
        self.cleanup_if_due()

    def delete_identity(self, subject: str) -> int:
        identity_hash = identity_scope(subject)
        with self.pool.connection() as connection:
            with connection.transaction():
                rows = connection.execute(
                    "SELECT thread_id FROM rag_memory_threads WHERE identity_hash = %s",
                    (identity_hash,),
                ).fetchall()
                thread_ids = [row["thread_id"] for row in rows]
                self._delete_checkpoints(connection, thread_ids)
                connection.execute(
                    "DELETE FROM rag_memory_threads WHERE identity_hash = %s",
                    (identity_hash,),
                )
            return len(thread_ids)

    def cleanup_if_due(self) -> int:
        now = time.monotonic()
        if now - self._last_cleanup < 300 or not self._cleanup_lock.acquire(blocking=False):
            return 0
        try:
            with self.pool.connection() as connection:
                with connection.transaction():
                    rows = connection.execute(
                        "SELECT thread_id FROM rag_memory_threads WHERE expires_at <= now()"
                    ).fetchall()
                    thread_ids = [row["thread_id"] for row in rows]
                    self._delete_checkpoints(connection, thread_ids)
                    connection.execute(
                        "DELETE FROM rag_memory_threads WHERE expires_at <= now()"
                    )
            self._last_cleanup = now
            return len(thread_ids)
        finally:
            self._cleanup_lock.release()

    @staticmethod
    def _delete_checkpoints(connection: Any, thread_ids: list[str]) -> None:
        if not thread_ids:
            return
        # These are the tables created and managed by PostgresSaver.setup().
        connection.execute(
            "DELETE FROM checkpoint_writes WHERE thread_id = ANY(%s)", (thread_ids,)
        )
        connection.execute(
            "DELETE FROM checkpoint_blobs WHERE thread_id = ANY(%s)", (thread_ids,)
        )
        connection.execute(
            "DELETE FROM checkpoints WHERE thread_id = ANY(%s)", (thread_ids,)
        )


def open_memory_runtime() -> MemoryRuntime | None:
    """Return durable Postgres memory when configured, else local MemorySaver."""
    database_url = settings.CHECKPOINT_DATABASE_URL
    if not database_url:
        if settings.AUTH_MODE == "cognito":
            raise RuntimeError("Cognito mode requires CHECKPOINT_DATABASE_URL.")
        return None

    os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")
    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    pool = ConnectionPool(
        conninfo=database_url,
        min_size=1,
        max_size=10,
        kwargs={"autocommit": True, "row_factory": dict_row},
        open=True,
    )
    try:
        pool.wait(timeout=10)
        checkpointer = PostgresSaver(pool)
        checkpointer.setup()
        retention = MemoryRetention(pool, settings.MEMORY_RETENTION_DAYS)
        return MemoryRuntime(pool, checkpointer, retention)
    except Exception:
        pool.close()
        raise
