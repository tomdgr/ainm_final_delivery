#!/bin/bash
# Startup script for GCE VM — downloads code from GCS, runs calibration +
# prediction + submit, uploads results, then shuts down.
set -euo pipefail

export HOME="${HOME:-/root}"
WORKDIR=/opt/ainm
RESULTS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/results-bucket" -H "Metadata-Flavor: Google")
AINM_ACCESS_TOKEN=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/ainm-token" -H "Metadata-Flavor: Google")
DO_SUBMIT=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/do-submit" -H "Metadata-Flavor: Google" || echo "false")

export AINM_ACCESS_TOKEN

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

# Fetch data + analysis
uv run python main.py --fetch
uv run python main.py --analysis

# Run calibration + predict (+ submit if requested)
if [ "$DO_SUBMIT" = "true" ]; then
    uv run python main.py --calibrate --submit
else
    uv run python main.py --calibrate
fi

# Upload results to GCS
gsutil -m cp -r data/rounds/ "gs://${RESULTS_BUCKET}/data/rounds/"
gsutil cp data/calibrated_params.json "gs://${RESULTS_BUCKET}/data/calibrated_params.json" 2>/dev/null || true

# Signal success
echo "SUCCESS" > /tmp/status.txt
gsutil cp /tmp/status.txt "gs://${RESULTS_BUCKET}/status.txt"

# Self-destruct
shutdown -h now
