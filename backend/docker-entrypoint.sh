#!/usr/bin/env bash
# Container entrypoint: wait for Postgres, apply migrations, then exec the CMD.
#
# Migrations run here rather than in a separate step so a fresh `docker compose
# up` yields a working schema. Only the API container migrates (RUN_MIGRATIONS=1);
# workers wait for the schema instead, so concurrent `alembic upgrade` runs
# cannot race each other.
set -euo pipefail

DB_HOST="${POSTGRES_HOST:-postgres}"
DB_PORT="${POSTGRES_PORT:-5432}"

wait_for_db() {
  local attempts=${DB_WAIT_ATTEMPTS:-60}
  for ((i = 1; i <= attempts; i++)); do
    if python -c "
import socket, sys
s = socket.socket()
s.settimeout(2)
try:
    s.connect(('${DB_HOST}', ${DB_PORT}))
except OSError:
    sys.exit(1)
finally:
    s.close()
" 2>/dev/null; then
      echo "[entrypoint] database reachable at ${DB_HOST}:${DB_PORT}"
      return 0
    fi
    echo "[entrypoint] waiting for database (${i}/${attempts})..."
    sleep 2
  done
  echo "[entrypoint] ERROR: database never became reachable" >&2
  return 1
}

wait_for_db

if [[ "${RUN_MIGRATIONS:-0}" == "1" ]]; then
  echo "[entrypoint] applying database migrations"
  alembic upgrade head
  echo "[entrypoint] migrations applied"
else
  # Workers need the schema to exist but must not migrate concurrently.
  schema_ready=0
  attempts=${SCHEMA_WAIT_ATTEMPTS:-60}
  for ((i = 1; i <= attempts; i++)); do
    # Report the real error on the last attempt so a genuine failure is
    # visible instead of looking like an endless wait.
    if [[ $i -eq $attempts ]]; then redirect=/dev/stderr; else redirect=/dev/null; fi
    if python -c "
import asyncio, sys
from sqlalchemy import text
from app.db.session import async_engine

async def main():
    async with async_engine.connect() as conn:
        await conn.execute(text('SELECT 1 FROM users LIMIT 1'))

asyncio.run(main())
" 2>"$redirect"; then
      echo "[entrypoint] schema present"
      schema_ready=1
      break
    fi
    echo "[entrypoint] waiting for schema (${i}/${attempts})..."
    sleep 2
  done
  if [[ "$schema_ready" != "1" ]]; then
    echo "[entrypoint] ERROR: schema never appeared; is the API container migrating?" >&2
    exit 1
  fi
fi

exec "$@"
