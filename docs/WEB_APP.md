# ROI Web App (Next.js Layer)

This app layer does three things:

1. Starts a quant run (`static` or `adaptive`).
2. Stores run metadata and recommendations in SQLite.
3. Shows recent runs and run outputs in a UI.

## Components

- `web/app/api/runs`: create/list runs.
- `web/app/api/runs/[id]`: fetch run details + recommendations.
- `web/prisma/schema.prisma`: SQLite schema (`Run`, `Recommendation`).
- `quant/bridge/run_once.py`: Python bridge that executes the quant engine and emits JSON.

## Run Flow

1. UI calls `POST /api/runs`.
2. API creates `Run(status=RUNNING)`.
3. API executes `quant/bridge/run_once.py`.
4. API stores recommendations (including per-row `auditJson`) and sets run status to `COMPLETED` or `FAILED`.

## Decision Audit

- In Run Details, each recommendation row is clickable.
- Selecting a row opens a rationale panel with:
  - summary explanation,
  - thresholds and pass/fail checks,
  - metrics/signals/risk snapshots,
  - raw captured fields from the engine output.

## Local Notes

- The quant engine still fetches market/news/macro data directly.
- Fundamentals data can come from the local Equity API based on the existing Python config.
- For UI testing without a full run, set `ROI_QUANT_DRY_RUN=1` in `web/.env`.
- Initialize SQLite schema with `cd web && npm run db:init`.
- If Prisma migration tooling works in your local setup, `npx prisma migrate dev --name init` is also supported.
