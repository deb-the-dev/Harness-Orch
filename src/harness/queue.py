"""Run queue with leases on PostgreSQL (decided: a lease and queue in PostgreSQL first, a workflow engine later).

Semantics: at-least-once delivery. A worker holds a run for a lease and must heartbeat; if it stops, the lease expires,
the reaper returns the run to the queue and another worker resumes from the event log. The worker design makes duplicate
execution safe (responses are reused, non-idempotent tool calls with an unknown outcome need a human).
The queue holds no state the log lacks: a queue row can always be rebuilt from a run's events.
"""
from __future__ import annotations

from dataclasses import dataclass

from psycopg_pool import ConnectionPool


@dataclass(frozen=True)
class Claim:
    org_id: str
    run_id: str
    attempts: int
    worker_id: str


class PostgresQueue:
    def __init__(self, dsn: str | None = None, pool: ConnectionPool | None = None, max_size: int = 10):
        self.pool = pool or ConnectionPool(dsn, min_size=1, max_size=max_size, open=True)

    def close(self) -> None:
        self.pool.close()

    def enqueue(self, org_id: str, run_id: str, delay_s: float = 0.0) -> None:
        """Make a run available. Idempotent. If a worker currently holds the run, flag it so the worker's release
        redelivers instead of deleting (otherwise work arriving in that window would be lost)."""
        with self.pool.connection() as c:
            c.execute(
                "INSERT INTO run_queue (run_id, org_id, available_at) VALUES (%s, %s, clock_timestamp() + make_interval(secs => %s)) "
                "ON CONFLICT (run_id) DO UPDATE SET wake = (run_queue.lease_owner IS NOT NULL) "
                "WHERE run_queue.org_id = EXCLUDED.org_id", (run_id, org_id, delay_s))

    def claim(self, worker_id: str, lease_s: float = 30.0) -> Claim | None:
        with self.pool.connection() as c:
            row = c.execute(
                "UPDATE run_queue SET lease_owner = %(w)s, lease_expires_at = clock_timestamp() + make_interval(secs => %(l)s), "
                "attempts = attempts + 1, wake = FALSE "
                "WHERE run_id = (SELECT run_id FROM run_queue WHERE lease_owner IS NULL AND available_at <= clock_timestamp() "
                "                ORDER BY available_at FOR UPDATE SKIP LOCKED LIMIT 1) "
                "RETURNING org_id, run_id, attempts", {"w": worker_id, "l": lease_s}).fetchone()
        return Claim(row[0], row[1], row[2], worker_id) if row else None

    def heartbeat(self, run_id: str, worker_id: str, lease_s: float = 30.0) -> bool:
        """False means the lease is no longer ours (expired and possibly taken by another worker): stop working."""
        with self.pool.connection() as c:
            n = c.execute("UPDATE run_queue SET lease_expires_at = clock_timestamp() + make_interval(secs => %s) "
                          "WHERE run_id = %s AND lease_owner = %s AND lease_expires_at > clock_timestamp()",
                          (lease_s, run_id, worker_id)).rowcount
        return n == 1

    def release(self, run_id: str, worker_id: str, retry_after_s: float | None = None) -> str:
        """Give the run back. Returns 'deleted', 'redelivered' (work arrived meanwhile, or a retry was asked for) or 'lost'."""
        with self.pool.connection() as c:
            row = c.execute("SELECT wake FROM run_queue WHERE run_id = %s AND lease_owner = %s FOR UPDATE", (run_id, worker_id)).fetchone()
            if row is None:
                return "lost"                                          # the lease expired and the run was re-queued or taken over
            if row[0] or retry_after_s is not None:
                c.execute("UPDATE run_queue SET lease_owner = NULL, lease_expires_at = NULL, wake = FALSE, "
                          "available_at = clock_timestamp() + make_interval(secs => %s) WHERE run_id = %s", (retry_after_s or 0.0, run_id))
                return "redelivered"
            c.execute("DELETE FROM run_queue WHERE run_id = %s", (run_id,))
            return "deleted"

    def remove(self, run_id: str) -> None:
        """Drop a run from the queue (cancelled or finished elsewhere)."""
        with self.pool.connection() as c:
            c.execute("DELETE FROM run_queue WHERE run_id = %s", (run_id,))

    def expire_leases(self) -> list[tuple[str, str]]:
        """Return runs whose worker stopped heartbeating to the queue. Returns (org_id, run_id) pairs so the caller can
        record run.worker_lost in each run's log."""
        with self.pool.connection() as c:
            rows = c.execute("UPDATE run_queue SET lease_owner = NULL, lease_expires_at = NULL, wake = FALSE, available_at = clock_timestamp() "
                             "WHERE lease_owner IS NOT NULL AND lease_expires_at < clock_timestamp() RETURNING org_id, run_id").fetchall()
        return [(r[0], r[1]) for r in rows]

    def depth(self) -> dict[str, int]:
        with self.pool.connection() as c:
            r = c.execute("SELECT count(*) FILTER (WHERE lease_owner IS NULL), count(*) FILTER (WHERE lease_owner IS NOT NULL) FROM run_queue").fetchone()
        return {"ready_or_waiting": r[0], "leased": r[1]}

    def row(self, run_id: str) -> dict | None:
        with self.pool.connection() as c:
            r = c.execute("SELECT lease_owner, attempts, wake FROM run_queue WHERE run_id = %s", (run_id,)).fetchone()
        return {"lease_owner": r[0], "attempts": r[1], "wake": r[2]} if r else None
