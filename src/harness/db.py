"""Migrations and partition management for the PostgreSQL store."""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import psycopg

MIGRATIONS = Path(__file__).parent / "migrations"
APP_TABLES_RW = ("events", "run_seq", "run_queue")


def migrate(admin_dsn: str, app_role: str | None = None) -> list[str]:
    """Apply pending migrations as an administrative role, then grant the application role least privilege.

    The application role gets SELECT and INSERT on events (no UPDATE or DELETE), and read-write on the counter and queue.
    """
    applied: list[str] = []
    with psycopg.connect(admin_dsn, autocommit=True) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS schema_migrations (version TEXT PRIMARY KEY, applied_at TIMESTAMPTZ NOT NULL DEFAULT now())")
        done = {r[0] for r in conn.execute("SELECT version FROM schema_migrations")}
        for f in sorted(MIGRATIONS.glob("*.sql")):
            if f.name in done:
                continue
            with conn.transaction():
                conn.execute(f.read_text())
                conn.execute("INSERT INTO schema_migrations (version) VALUES (%s)", (f.name,))
            applied.append(f.name)
        ensure_partitions(conn)
        if app_role:
            conn.execute(f'GRANT USAGE ON SCHEMA public TO "{app_role}"')
            conn.execute(f'GRANT SELECT, INSERT ON events TO "{app_role}"')
            conn.execute(f'GRANT SELECT, INSERT, UPDATE ON run_seq TO "{app_role}"')
            conn.execute(f'GRANT SELECT, INSERT, UPDATE, DELETE ON run_queue TO "{app_role}"')
    return applied


def ensure_partitions(conn: psycopg.Connection, months_back: int = 0, months_ahead: int = 3) -> list[str]:
    """Create monthly event partitions. Run at start-up and on a schedule, before rows for a month arrive:
    PostgreSQL refuses to create a partition whose range already has rows in the default partition."""
    first = dt.date.today().replace(day=1)
    made = []
    for k in range(-months_back, months_ahead + 1):
        y, m = divmod(first.year * 12 + first.month - 1 + k, 12)
        start, end = dt.date(y, m + 1, 1), dt.date(y + (m + 1) // 12, (m + 1) % 12 + 1, 1)
        name = f"events_y{start.year}m{start.month:02d}"
        conn.execute(f"CREATE TABLE IF NOT EXISTS {name} PARTITION OF events FOR VALUES FROM ('{start}') TO ('{end}')")
        made.append(name)
    return made
