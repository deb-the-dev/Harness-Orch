"""PostgreSQL implementation of the event log protocol (same guarantees as the SQLite one)."""
from __future__ import annotations

import json
import time
from typing import Iterable

import psycopg
from psycopg_pool import ConnectionPool

from .events import SCHEMA_VERSION, Event, canonical, payload_hash
from .ids import uuid7

_COLS = "id, org_id, run_id, seq, type, step_id, parent_event_id, payload, payload_hash, schema_version, EXTRACT(EPOCH FROM created_at)::float8"


def _event(r) -> Event:
    return Event(str(r[0]), r[1], r[2], r[3], r[4], r[5], str(r[6]) if r[6] else None, r[7], r[8], r[9], r[10])


class PostgresEventLog:
    """Connect as the application role, not a superuser: row-level security and the missing UPDATE/DELETE grants only
    protect you from a role that is subject to them. The code also filters by org_id, so tenancy does not rest on one layer."""

    def __init__(self, dsn: str | None = None, pool: ConnectionPool | None = None, max_size: int = 10):
        self.pool = pool or ConnectionPool(dsn, min_size=1, max_size=max_size, open=True)

    def close(self) -> None:
        self.pool.close()

    def _tx(self, org_id: str):
        class Ctx:
            def __enter__(s):
                s.cm = self.pool.connection()
                s.conn = s.cm.__enter__()
                s.tx = s.conn.transaction()
                s.tx.__enter__()
                s.conn.execute("SELECT set_config('app.org_id', %s, true)", (org_id,))
                return s.conn

            def __exit__(s, *exc):
                try:
                    return s.tx.__exit__(*exc)
                finally:
                    s.cm.__exit__(*exc)
        return Ctx()

    def append(self, org_id, run_id, type, payload=None, step_id=None, parent_event_id=None) -> Event:
        payload = json.loads(canonical(payload or {}))            # normalise exactly as it will be stored, then hash that
        h = payload_hash(payload)
        with self._tx(org_id) as conn:
            row = conn.execute(
                "INSERT INTO run_seq (run_id, org_id, last_seq) VALUES (%s, %s, 1) "
                "ON CONFLICT (run_id) DO UPDATE SET last_seq = run_seq.last_seq + 1 WHERE run_seq.org_id = EXCLUDED.org_id "
                "RETURNING last_seq", (run_id, org_id)).fetchone()
            if row is None:
                raise PermissionError("run belongs to another org")   # no cross-tenant writes
            eid, now = uuid7(), time.time()
            conn.execute("INSERT INTO events (id, org_id, run_id, seq, type, step_id, parent_event_id, payload, payload_hash, schema_version, created_at) "
                         "VALUES (%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,to_timestamp(%s))",
                         (eid, org_id, run_id, row[0], type, step_id, parent_event_id, json.dumps(payload), h, SCHEMA_VERSION, now))
        return Event(eid, org_id, run_id, row[0], type, step_id, parent_event_id, payload, h, SCHEMA_VERSION, now)

    def read(self, org_id, run_id, after_seq=0) -> list[Event]:
        with self._tx(org_id) as conn:
            rows = conn.execute(f"SELECT {_COLS} FROM events WHERE org_id = %s AND run_id = %s AND seq > %s ORDER BY seq",
                                (org_id, run_id, after_seq)).fetchall()
        return [_event(r) for r in rows]

    def verify(self, org_id: str, run_id: str) -> list[int]:
        evs = self.read(org_id, run_id)
        bad = [e.seq for e in evs if payload_hash(e.payload) != e.payload_hash]
        have = {e.seq for e in evs}
        return bad + [s for s in range(1, len(evs) + 1) if s not in have]

    def run_ids(self, org_id: str, type: str | None = None) -> Iterable[str]:
        q, args = "SELECT DISTINCT run_id FROM events WHERE org_id = %s", [org_id]
        if type:
            q, args = q + " AND type = %s", args + [type]
        with self._tx(org_id) as conn:
            return [r[0] for r in conn.execute(q, args).fetchall()]
