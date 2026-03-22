#!/bin/bash
# Install or remove the auto-submit cron job.
#
# Usage:
#   bash infra/install-cron.sh           # install cron (runs at :08 and :38)
#   bash infra/install-cron.sh --remove  # remove cron
#   bash infra/install-cron.sh --status  # show current cron entries
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
AUTO_SUBMIT="$SCRIPT_DIR/auto-submit.sh"
LOG_FILE="$PROJECT_DIR/logs/auto-submit.log"
CRON_MARKER="# ainm-auto-submit"

case "${1:-install}" in
  --remove)
    crontab -l 2>/dev/null | grep -v "$CRON_MARKER" | crontab -
    echo "Cron job removed."
    ;;
  --status)
    echo "Current cron entries:"
    crontab -l 2>/dev/null | grep "ainm" || echo "  (none)"
    echo ""
    echo "Recent log:"
    tail -20 "$LOG_FILE" 2>/dev/null || echo "  (no log file yet)"
    ;;
  install|"")
    mkdir -p "$PROJECT_DIR/state" "$PROJECT_DIR/logs"

    # Remove old entry if exists, then add new one
    EXISTING=$(crontab -l 2>/dev/null | grep -v "$CRON_MARKER" || true)
    CRON_LINE="8,38 * * * * $AUTO_SUBMIT >> $LOG_FILE 2>&1 $CRON_MARKER"

    echo "$EXISTING
$CRON_LINE" | crontab -

    echo "Cron job installed: runs at :08 and :38 past each hour."
    echo ""
    echo "Monitor:  tail -f $LOG_FILE"
    echo "Status:   bash infra/install-cron.sh --status"
    echo "Remove:   bash infra/install-cron.sh --remove"
    echo "Dry run:  bash infra/auto-submit.sh --dry-run"
    ;;
  *)
    echo "Usage: bash infra/install-cron.sh [--remove|--status]"
    exit 1
    ;;
esac
