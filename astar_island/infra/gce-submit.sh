#!/bin/bash
# Startup script for GCE GPU VM — train models + generate round 16 predictions
set -euo pipefail

export HOME="${HOME:-/root}"
WORKDIR=/opt/ainm
RESULTS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/results-bucket" -H "Metadata-Flavor: Google")

# Install system deps + CUDA
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

# Run round 16 prediction pipeline (NO submission — dry run only)
echo "=== Round 16 prediction pipeline ==="
uv run python submit_round16.py 2>&1 | tee /tmp/round16_results.txt

# Upload results
gsutil cp /tmp/round16_results.txt "gs://${RESULTS_BUCKET}/round16_results.txt"
gsutil -m cp -r data/rounds/round_16/predictions/ "gs://${RESULTS_BUCKET}/round16_predictions/" 2>/dev/null || true

# Signal success
echo "SUCCESS" > /tmp/status.txt
gsutil cp /tmp/status.txt "gs://${RESULTS_BUCKET}/status_submit.txt"

# Self-destruct
shutdown -h now
