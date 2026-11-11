"""Append-only event log: the spine every other record hangs off (ERD: event).

Invariants enforced here, not in callers:
  * events are never updated or deleted (database triggers);
  * (run_id, seq) is unique and gapless per run;
  * every row carries org_id and every read filters on it;
  * payload_hash is verified on demand, so a request can be rebuilt from the log.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Any, Iterable, Protocol

from .ids import uuid7

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Event:
    id: str
    org_id: str
    run_id: str
    seq: int
    type: str
    step_id: str | None
    parent_event_id: str | None
    payload: dict[str, Any]
    payload_hash: str
    schema_version: int
    created_at: float


class EventLog(Protocol):
    def append(self, org_id: str, run_id: str, type: str, payload: dict[str, Any] | None = None,
               step_id: str | None = None, parent_event_id: str | None = None) -> Event: ...

    def read(self, org_id: str, run_id: str, after_seq: int = 0) -> list[Event]: ...


def canonical(payload: dict[str, Any]) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)


def payload_hash(payload: dict[str, Any]) -> str:
    return hashlib.sha256(canonical(payload).encode()).hexdigest()


_DDL = """
CREATE TABLE IF NOT EXISTS events (
  id TEXT PRIMARY KEY,
  org_id TEXT NOT NULL,
  run_id TEXT NOT NULL,
  seq INTEGER NOT NULL,
  type TEXT NOT NULL,
  step_id TEXT,
  parent_event_id TEXT,
  payload TEXT NOT NULL,
  payload_hash TEXT NOT NULL,
  schema_version INTEGER NOT NULL,
  created_at REAL NOT NULL,
  UNIQUE (run_id, seq)
);
CREATE INDEX IF NOT EXISTS events_org_type ON events (org_id, type, created_at);
CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
BEGIN SELECT RAISE(ABORT, 'events are append-only'); END;
"""


class SQLiteEventLog:
    """Development and test implementation. The PostgreSQL implementation follows the same protocol."""

    def __init__(self, path: str = ":memory:"):
        self._db = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._db.executescript(_DDL)
        self._lock = threading.Lock()

    def append(self, org_id, run_id, type, payload=None, step_id=None, parent_event_id=None) -> Event:
        payload = payload or {}
        with self._lock:
            self._db.execute("BEGIN IMMEDIATE")
            try:
                row = self._db.execute("SELECT org_id, COALESCE(MAX(seq), 0) FROM events WHERE run_id = ?", (run_id,)).fetchone()
                if row[0] is not None and row[0] != org_id:
                    raise PermissionError("run belongs to another org")  # no cross-tenant writes
                ev = Event(uuid7(), org_id, run_id, row[1] + 1, type, step_id, parent_event_id, payload,
                           payload_hash(payload), SCHEMA_VERSION, time.time())
                self._db.execute("INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                                 (ev.id, ev.org_id, ev.run_id, ev.seq, ev.type, ev.step_id, ev.parent_event_id,
                                  canonical(ev.payload), ev.payload_hash, ev.schema_version, ev.created_at))
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return ev

    def read(self, org_id, run_id, after_seq=0) -> list[Event]:
        with self._lock:
            rows = self._db.execute(
                "SELECT id, org_id, run_id, seq, type, step_id, parent_event_id, payload, payload_hash, schema_version, created_at "
                "FROM events WHERE org_id = ? AND run_id = ? AND seq > ? ORDER BY seq", (org_id, run_id, after_seq)).fetchall()
        return [Event(r[0], r[1], r[2], r[3], r[4], r[5], r[6], json.loads(r[7]), r[8], r[9], r[10]) for r in rows]

    def verify(self, org_id: str, run_id: str) -> list[int]:
        """Sequence numbers whose stored payload no longer matches its hash, and any gaps."""
        evs = self.read(org_id, run_id)
        bad = [e.seq for e in evs if payload_hash(e.payload) != e.payload_hash]
        bad += [s for s in range(1, len(evs) + 1) if s not in {e.seq for e in evs}]
        return bad

    def run_ids(self, org_id: str, type: str | None = None) -> Iterable[str]:
        q, args = "SELECT DISTINCT run_id FROM events WHERE org_id = ?", [org_id]
        if type:
            q, args = q + " AND type = ?", args + [type]
        with self._lock:
            return [r[0] for r in self._db.execute(q, args).fetchall()]
