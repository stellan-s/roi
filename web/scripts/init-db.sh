#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Resolve SQLite location the same way Prisma does for `file:` URLs:
# relative paths are anchored to the `prisma/` directory.
if [[ -f "$ROOT_DIR/.env" ]]; then
  # shellcheck disable=SC1090
  source "$ROOT_DIR/.env"
fi

if [[ "${1:-}" != "" ]]; then
  DB_PATH="$1"
else
  DB_URL="${DATABASE_URL:-file:./dev.db}"
  if [[ "$DB_URL" == file:* ]]; then
    RAW_PATH="${DB_URL#file:}"
    if [[ "$RAW_PATH" == /* ]]; then
      DB_PATH="$RAW_PATH"
    else
      DB_PATH="$ROOT_DIR/prisma/$RAW_PATH"
    fi
  else
    echo "Unsupported DATABASE_URL for init-db: $DB_URL" >&2
    exit 1
  fi
fi

mkdir -p "$(dirname "$DB_PATH")"

sqlite3 "$DB_PATH" <<'SQL'
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS "Run" (
  "id" TEXT NOT NULL PRIMARY KEY,
  "mode" TEXT NOT NULL CHECK ("mode" IN ('STATIC', 'ADAPTIVE')),
  "status" TEXT NOT NULL CHECK ("status" IN ('RUNNING', 'COMPLETED', 'FAILED')),
  "startedAt" DATETIME,
  "endedAt" DATETIME,
  "error" TEXT,
  "summaryJson" TEXT,
  "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  "updatedAt" DATETIME NOT NULL
);

CREATE TABLE IF NOT EXISTS "Recommendation" (
  "id" TEXT NOT NULL PRIMARY KEY,
  "runId" TEXT NOT NULL,
  "date" DATETIME,
  "ticker" TEXT NOT NULL,
  "decision" TEXT NOT NULL,
  "close" REAL,
  "expectedReturn" REAL,
  "probPositive" REAL,
  "decisionConfidence" REAL,
  "uncertainty" REAL,
  "portfolioWeight" REAL,
  "auditJson" TEXT,
  "createdAt" DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY ("runId") REFERENCES "Run" ("id") ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX IF NOT EXISTS "Run_createdAt_idx" ON "Run"("createdAt");
CREATE INDEX IF NOT EXISTS "Recommendation_runId_idx" ON "Recommendation"("runId");
CREATE INDEX IF NOT EXISTS "Recommendation_ticker_idx" ON "Recommendation"("ticker");
SQL

# Add newly introduced columns for existing databases.
if ! sqlite3 "$DB_PATH" "PRAGMA table_info('Recommendation');" | grep -q '|auditJson|'; then
  sqlite3 "$DB_PATH" 'ALTER TABLE "Recommendation" ADD COLUMN "auditJson" TEXT;'
fi

echo "Initialized SQLite schema at $DB_PATH"
