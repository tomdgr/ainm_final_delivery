#!/bin/bash
# Auto-submit pipeline: pulls latest observations from git, triggers GCE calibration + submission.
# Designed to run via cron at :08 and :38 (after GitHub Actions fetches at :00 and :30).
#
# Usage:
#   ./infra/auto-submit.sh              # normal operation (from cron)
#   ./infra/auto-submit.sh --dry-run    # check for new data without triggering GCE
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
STATE_DIR="$PROJECT_DIR/state"
LOG_PREFIX="$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
DRY_RUN=false

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=true
fi

# --- Prevent concurrent runs (macOS-compatible) ---
LOCK_FILE="$STATE_DIR/auto-submit.lock"
mkdir -p "$STATE_DIR"
if ! mkdir "$LOCK_FILE" 2>/dev/null; then
  echo "[$LOG_PREFIX] Pipeline already running, skipping."
  exit 0
fi
trap 'rmdir "$LOCK_FILE" 2>/dev/null' EXIT

echo "=========================================="
echo "[$LOG_PREFIX] Auto-submit pipeline"
echo "=========================================="

# --- Pull latest from git ---
cd "$PROJECT_DIR"
git pull --rebase origin main 2>&1 || {
  echo "[$LOG_PREFIX] ERROR: git pull failed"
  exit 1
}

# --- Check if HEAD has changed since last run ---
CURRENT_COMMIT=$(git rev-parse HEAD)
LAST_COMMIT_FILE="$STATE_DIR/last_processed_commit"

if [ -f "$LAST_COMMIT_FILE" ]; then
  LAST_COMMIT=$(cat "$LAST_COMMIT_FILE")
  if [ "$CURRENT_COMMIT" = "$LAST_COMMIT" ]; then
    echo "[$LOG_PREFIX] No new commits (HEAD=$CURRENT_COMMIT). Nothing to do."
    exit 0
  fi
  echo "[$LOG_PREFIX] New commits detected: $LAST_COMMIT -> $CURRENT_COMMIT"
else
  echo "[$LOG_PREFIX] First run (no previous commit recorded)."
fi

# --- Find latest round with observations ---
LATEST_OBS_DIR=$(find "$PROJECT_DIR/data/rounds" -path "*/observations/seed_0.json" -type f 2>/dev/null \
  | sort -V | tail -1)

if [ -z "$LATEST_OBS_DIR" ]; then
  echo "[$LOG_PREFIX] No observation data found. Saving commit and exiting."
  echo "$CURRENT_COMMIT" > "$LAST_COMMIT_FILE"
  exit 0
fi

# Extract round number from path like data/rounds/round_5/observations/seed_0.json
ROUND_DIR=$(echo "$LATEST_OBS_DIR" | sed 's|/observations/.*||')
ROUND_NUM=$(basename "$ROUND_DIR" | sed 's/round_//')
echo "[$LOG_PREFIX] Latest round with observations: round_$ROUND_NUM"

# --- Check if we already submitted for this round ---
SUBMITTED_FILE="$STATE_DIR/submitted_round_${ROUND_NUM}"
if [ -f "$SUBMITTED_FILE" ]; then
  echo "[$LOG_PREFIX] Already submitted for round $ROUND_NUM (at $(cat "$SUBMITTED_FILE")). Saving commit and exiting."
  echo "$CURRENT_COMMIT" > "$LAST_COMMIT_FILE"
  exit 0
fi

# --- Check if observations actually changed (not just analysis or other data) ---
if [ -f "$LAST_COMMIT_FILE" ]; then
  OBS_CHANGED=$(git diff --name-only "$LAST_COMMIT" "$CURRENT_COMMIT" -- "data/rounds/round_${ROUND_NUM}/observations/" 2>/dev/null | head -1)
  if [ -z "$OBS_CHANGED" ]; then
    echo "[$LOG_PREFIX] New commits but no observation changes for round $ROUND_NUM. Saving commit and exiting."
    echo "$CURRENT_COMMIT" > "$LAST_COMMIT_FILE"
    exit 0
  fi
  echo "[$LOG_PREFIX] Observation data changed for round $ROUND_NUM."
fi

# --- Verify budget is fully spent (all observations fetched) ---
BUDGET_FILE="$ROUND_DIR/budget.json"
if [ -f "$BUDGET_FILE" ]; then
  QUERIES_USED=$(python3 -c "import json; b=json.load(open('$BUDGET_FILE')); print(b.get('queries_used', 0))")
  QUERIES_MAX=$(python3 -c "import json; b=json.load(open('$BUDGET_FILE')); print(b.get('queries_max', 50))")
  if [ "$QUERIES_USED" -lt "$QUERIES_MAX" ]; then
    echo "[$LOG_PREFIX] Budget not fully spent ($QUERIES_USED/$QUERIES_MAX). Waiting for GHA to finish fetching."
    exit 0
  fi
  echo "[$LOG_PREFIX] Budget fully spent ($QUERIES_USED/$QUERIES_MAX). All observations are in."
else
  echo "[$LOG_PREFIX] No budget.json found for round $ROUND_NUM. Waiting for GHA to finish fetching."
  exit 0
fi

# --- Trigger GCE calibration + submission ---
if [ "$DRY_RUN" = "true" ]; then
  echo "[$LOG_PREFIX] DRY RUN: Would trigger 'bash infra/gce-run.sh --submit' for round $ROUND_NUM"
  echo "$CURRENT_COMMIT" > "$LAST_COMMIT_FILE"
  exit 0
fi

echo "[$LOG_PREFIX] Triggering GCE calibration + submission for round $ROUND_NUM..."
if bash "$SCRIPT_DIR/gce-run.sh" --submit; then
  echo "[$LOG_PREFIX] GCE submission completed successfully for round $ROUND_NUM."
  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$SUBMITTED_FILE"
  echo "$CURRENT_COMMIT" > "$LAST_COMMIT_FILE"

  # Commit submission results back to git
  cd "$PROJECT_DIR"
  git add data/rounds/ data/calibrated_params.json 2>/dev/null || true
  if ! git diff --cached --quiet; then
    git commit -m "Auto-submit: round $ROUND_NUM calibration + submission results"
    git push origin main || echo "[$LOG_PREFIX] WARNING: git push failed (non-fatal)"
  fi
else
  echo "[$LOG_PREFIX] ERROR: GCE submission failed for round $ROUND_NUM. Will retry on next run."
  # Don't save commit or mark as submitted — will retry next cron cycle
  exit 1
fi

echo "[$LOG_PREFIX] Pipeline complete."
