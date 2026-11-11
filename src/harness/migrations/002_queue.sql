-- Run queue with leases. A system table: workers claim across organisations, so there is no per-org row security here.
-- Access is limited to the orchestrator and worker roles.
CREATE TABLE run_queue (
  run_id           TEXT PRIMARY KEY,
  org_id           TEXT        NOT NULL,
  enqueued_at      TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  available_at     TIMESTAMPTZ NOT NULL DEFAULT clock_timestamp(),
  lease_owner      TEXT,
  lease_expires_at TIMESTAMPTZ,
  attempts         INTEGER     NOT NULL DEFAULT 0,
  wake             BOOLEAN     NOT NULL DEFAULT FALSE   -- set when work arrives while a worker still holds the lease
);
CREATE INDEX run_queue_ready  ON run_queue (available_at) WHERE lease_owner IS NULL;
CREATE INDEX run_queue_leases ON run_queue (lease_expires_at) WHERE lease_owner IS NOT NULL;
