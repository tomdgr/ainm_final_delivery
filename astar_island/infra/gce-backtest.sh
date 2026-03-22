#!/bin/bash
# Startup script for GCE VM — runs per-round inference backtest
set -euo pipefail

export HOME="${HOME:-/root}"
WORKDIR=/opt/ainm
RESULTS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/results-bucket" -H "Metadata-Flavor: Google")

# Install system deps
apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv curl > /dev/null 2>&1

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Download code from GCS
mkdir -p "$WORKDIR"
gsutil -m cp -r "gs://${RESULTS_BUCKET}/code/*" "$WORKDIR/"
cd "$WORKDIR"

# Install Python deps
uv sync

# Run per-round inference backtest
echo "=== Per-round inference backtest ==="
uv run python backtest_per_round.py 2>&1 | tee /tmp/backtest_results.txt

# Upload results
gsutil cp /tmp/backtest_results.txt "gs://${RESULTS_BUCKET}/backtest_per_round_results.txt"

# Signal success
echo "SUCCESS" > /tmp/status.txt
gsutil cp /tmp/status.txt "gs://${RESULTS_BUCKET}/status.txt"

# Self-destruct
shutdown -h now
