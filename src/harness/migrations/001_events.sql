-- Event log (ERD: event). Append-only, tenant-scoped, monthly partitions for retention.
--
-- ERD correction: PostgreSQL cannot enforce UNIQUE (run_id, seq) across time partitions unless the partition column is in
-- the key. The per-run counter in run_seq is the authority for seq: one row lock per run, incremented in the same
-- transaction as the insert, so numbers are unique and gapless and a rolled-back append releases its number.

CREATE TABLE run_seq (
  run_id   TEXT PRIMARY KEY,
  org_id   TEXT NOT NULL,
  last_seq INTEGER NOT NULL
);

CREATE TABLE events (
  id              UUID        NOT NULL,
  org_id          TEXT        NOT NULL,
  run_id          TEXT        NOT NULL,
  seq             INTEGER     NOT NULL,
  type            TEXT        NOT NULL,
  step_id         TEXT,
  parent_event_id UUID,
  payload         JSONB       NOT NULL,
  payload_hash    TEXT        NOT NULL,
  schema_version  INTEGER     NOT NULL,
  created_at      TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (id, created_at),
  UNIQUE (run_id, seq, created_at)
) PARTITION BY RANGE (created_at);

CREATE TABLE events_default PARTITION OF events DEFAULT;
CREATE INDEX events_run_seq ON events (run_id, seq);
CREATE INDEX events_org_type ON events (org_id, type, created_at);

CREATE FUNCTION events_immutable() RETURNS trigger LANGUAGE plpgsql AS
$$ BEGIN RAISE EXCEPTION 'events are append-only'; END $$;
CREATE TRIGGER events_no_change BEFORE UPDATE OR DELETE ON events
  FOR EACH ROW EXECUTE FUNCTION events_immutable();

-- Row-level security keyed on the session's org. The application sets app.org_id per transaction.
ALTER TABLE events ENABLE ROW LEVEL SECURITY;
ALTER TABLE events FORCE ROW LEVEL SECURITY;
CREATE POLICY events_tenant ON events
  USING (org_id = current_setting('app.org_id', true))
  WITH CHECK (org_id = current_setting('app.org_id', true));
