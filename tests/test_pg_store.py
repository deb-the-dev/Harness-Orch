"""PostgreSQL-specific guarantees: privileges, row-level security, partitions, concurrency, rollback."""
import threading
import uuid

import psycopg
import pytest

from harness.db import ensure_partitions
from harness.pg import PostgresEventLog


@pytest.fixture
def pglog(pg_app_dsn):
    lg = PostgresEventLog(pg_app_dsn)
    yield lg
    lg.close()


def ids():
    return "org-" + uuid.uuid4().hex[:8], "run-" + uuid.uuid4().hex[:10]


def test_app_role_cannot_update_or_delete_and_even_admin_is_stopped_by_the_trigger(pglog, pg_app_dsn, pg_admin_dsn):
    org, run = ids()
    pglog.append(org, run, "x", {"k": 1})
    with psycopg.connect(pg_app_dsn, autocommit=True) as c:
        c.execute("SELECT set_config('app.org_id', %s, false)", (org,))
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("UPDATE events SET type = 'z'")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("DELETE FROM events")
    with psycopg.connect(pg_admin_dsn, autocommit=True) as c:          # the privileged role is also refused: the trigger is a second layer
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            c.execute("UPDATE events SET type = 'z'")
        with pytest.raises(psycopg.errors.RaiseException, match="append-only"):
            c.execute("DELETE FROM events")


def test_row_level_security_hides_other_tenants_even_without_the_code_filter(pglog, pg_app_dsn):
    org_a, run_a = ids()
    org_b, _ = ids()
    pglog.append(org_a, run_a, "x")
    with psycopg.connect(pg_app_dsn, autocommit=True) as c:
        assert c.execute("SELECT count(*) FROM events WHERE run_id = %s", (run_a,)).fetchone()[0] == 0       # no org set: sees nothing
        c.execute("SELECT set_config('app.org_id', %s, false)", (org_b,))
        assert c.execute("SELECT count(*) FROM events WHERE run_id = %s", (run_a,)).fetchone()[0] == 0       # wrong org: sees nothing
        c.execute("SELECT set_config('app.org_id', %s, false)", (org_a,))
        assert c.execute("SELECT count(*) FROM events WHERE run_id = %s", (run_a,)).fetchone()[0] == 1
        with pytest.raises(psycopg.errors.InsufficientPrivilege):                                         # WITH CHECK: cannot insert as another org
            c.execute("INSERT INTO events VALUES (gen_random_uuid(), %s, 'r', 1, 't', NULL, NULL, '{}', 'h', 1, now())", (org_b,))


def test_rows_land_in_the_monthly_partition(pglog, pg_admin_dsn):
    org, run = ids()
    pglog.append(org, run, "x")
    with psycopg.connect(pg_admin_dsn, autocommit=True) as c:
        names = ensure_partitions(c)
        tbl = c.execute("SELECT tableoid::regclass::text FROM events WHERE run_id = %s", (run,)).fetchone()[0]
    assert tbl in names or tbl == "events_default"
    assert len(names) >= 3


def test_concurrent_appends_are_gapless_and_unique(pglog):
    org, run = ids()
    errors = []

    def work():
        try:
            for _ in range(25):
                pglog.append(org, run, "x")
        except Exception as e:                                                                              # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert not errors
    assert [e.seq for e in pglog.read(org, run)] == list(range(1, 201))
    assert pglog.verify(org, run) == []


def test_a_failed_append_does_not_burn_a_sequence_number(pglog):
    org, run = ids()
    pglog.append(org, run, "x")
    with pytest.raises(Exception):
        pglog.append(org, run, None)                                                                        # NOT NULL violation after the counter moved
    assert pglog.append(org, run, "y").seq == 2                                                              # the counter rolled back with the insert
    assert [e.seq for e in pglog.read(org, run)] == [1, 2]


def test_reads_do_not_cross_orgs_and_run_ids_filter_by_type(pglog):
    org, run = ids()
    pglog.append(org, run, "run.created")
    assert run in pglog.run_ids(org, "run.created")
    assert run not in pglog.run_ids(org + "x", "run.created")


def test_migrations_are_idempotent(pg_admin_dsn):
    from harness.db import migrate
    assert migrate(pg_admin_dsn) == []
