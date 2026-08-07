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
  for ((i = 1; i <= ${SCHEMA_WAIT_ATTEMPTS:-60}; i++)); do
    if python -c "
import asyncio, sys
from sqlalchemy import text
from app.db.session import engine

async def main():
    async with engine.connect() as conn:
        await conn.execute(text('SELECT 1 FROM users LIMIT 1'))

try:
    asyncio.run(main())
except Exception:
    sys.exit(1)
" 2>/dev/null; then
      echo "[entrypoint] schema present"
      break
    fi
    echo "[entrypoint] waiting for schema (${i})..."
    sleep 2
  done
fi

exec "$@"
