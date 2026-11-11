import os
import pytest
from pathlib import Path

from harness.policy import OpaProcess, build_policy_data, policy_version

ROOT = Path(__file__).resolve().parents[1]
OPA = ROOT / ".tools/opa/opa"
REGO = ROOT / "policies/hooks.rego"

from harness.packs import InvoiceLoggingPack
PACKS = {"ap_logging": InvoiceLoggingPack().policy_entry(),
         "big_result": {"tools": ["dump_rows", "read_artifact"], "models": ["m-small", "m-large"], "restricted_regions": ["me"]},
         "graded": {"tools": [], "models": ["m-small", "m-large"], "restricted_regions": ["me"]},
         "compacting": {"tools": [], "models": ["m-small", "m-large"], "restricted_regions": ["me"]}}


@pytest.fixture(scope="session")
def opa():
    if not OPA.exists():
        pytest.skip("OPA binary not found; see spikes/01-hook-latency/README for how it was fetched")
    p = OpaProcess([str(REGO)], build_policy_data(PACKS), opa_bin=str(OPA)).start()
    yield p
    p.stop()


@pytest.fixture(scope="session")
def pver():
    return policy_version(REGO)


# ---- PostgreSQL fixtures: a throwaway database per test session, application role without superuser rights ----------------
import os
import uuid
import psycopg

ADMIN_DSN = os.environ.get("HARNESS_PG_ADMIN_DSN", "postgresql://postgres:pgpass@127.0.0.1:55432/postgres")
APP_ROLE, APP_PASSWORD = "harness_app", "apppass"


def _swap_db(dsn: str, db: str) -> str:
    return dsn.rsplit("/", 1)[0] + "/" + db


@pytest.fixture(scope="session")
def pg_admin_dsn():
    try:
        psycopg.connect(ADMIN_DSN, connect_timeout=2).close()
    except Exception:
        pytest.skip("PostgreSQL not reachable; set HARNESS_PG_ADMIN_DSN (see README)")
    db = "harness_test_" + uuid.uuid4().hex[:8]
    with psycopg.connect(ADMIN_DSN, autocommit=True) as c:
        c.execute(f'CREATE DATABASE "{db}"')
        if not c.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (APP_ROLE,)).fetchone():
            c.execute(f"CREATE ROLE {APP_ROLE} LOGIN PASSWORD '{APP_PASSWORD}' NOSUPERUSER NOBYPASSRLS")
    dsn = _swap_db(ADMIN_DSN, db)
    from harness.db import migrate
    migrate(dsn, app_role=APP_ROLE)
    yield dsn
    with psycopg.connect(ADMIN_DSN, autocommit=True) as c:
        c.execute(f'DROP DATABASE IF EXISTS "{db}" WITH (FORCE)')


@pytest.fixture(scope="session")
def pg_app_dsn(pg_admin_dsn):
    rest = pg_admin_dsn.split("@", 1)[1]
    return f"postgresql://{APP_ROLE}:{APP_PASSWORD}@{rest}"


@pytest.fixture(params=["sqlite", "postgres"])
def log(request):
    """The same contract tests run against both event log implementations."""
    from harness.events import SQLiteEventLog
    if request.param == "sqlite":
        yield SQLiteEventLog()
        return
    from harness.pg import PostgresEventLog
    dsn = request.getfixturevalue("pg_app_dsn")
    lg = PostgresEventLog(dsn)
    yield lg
    lg.close()
