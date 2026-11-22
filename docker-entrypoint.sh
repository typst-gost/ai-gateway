#!/usr/bin/env sh
set -eu

DATA_DIR="${DATA_DIR:-/app/data}"
DB_PATH="${DOCS_DB_PATH:-${DATA_DIR}/docs_vector_db}"

mkdir -p "$DATA_DIR" "$(dirname "$DB_PATH")"

if [ ! -d "$DB_PATH/docs.lance" ] && [ "${SYNC_DOCS_ON_START:-true}" = "true" ]; then
  echo "[entrypoint] Initial docs sync and indexing..."
  ai-gateway-reindex
fi

exec "$@"