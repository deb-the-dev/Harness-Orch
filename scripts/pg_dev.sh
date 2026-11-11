#!/usr/bin/env bash
# Start a throwaway PostgreSQL 17 in Docker for development and tests (port 55432).
# Tests create their own database and role. Without Docker, see README: "No Docker?".
set -euo pipefail
if ! command -v docker >/dev/null 2>&1; then
  cat <<'MSG'
Docker is not installed (or not on your PATH).
Either install Docker Desktop and open it once, or use PostgreSQL from Homebrew:
  brew install postgresql@17 && brew services start postgresql@17
  export HARNESS_PG_ADMIN_DSN="postgresql://$(whoami)@127.0.0.1:5432/postgres"
  .venv/bin/python -m pytest -q
MSG
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  echo "Docker is installed but not running. Open Docker Desktop, wait until it says running, then rerun."
  exit 1
fi
docker rm -f harness-pg >/dev/null 2>&1 || true
docker run -d --name harness-pg -e POSTGRES_PASSWORD=pgpass -p 55432:5432 postgres:17 >/dev/null
until docker exec harness-pg pg_isready -U postgres >/dev/null 2>&1; do sleep 1; done
echo "ready: postgresql://postgres:pgpass@127.0.0.1:55432/postgres  (override with HARNESS_PG_ADMIN_DSN)"
