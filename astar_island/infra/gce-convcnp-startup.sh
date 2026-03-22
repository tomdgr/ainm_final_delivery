#!/bin/bash
# Startup script for GCE VM — runs ConvCNP experiments with many configs.
# Downloads code from GCS, runs full LOO-CV for each config, uploads results.
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

# Create results directory
mkdir -p data/convcnp_results

echo "=== Starting ConvCNP experiments ==="
echo "Machine: $(nproc) cores, $(free -h | awk '/Mem:/{print $2}') RAM"
date

# Run all configs — eval_convcnp.py handles everything
# Use --save-models so we keep the best one
uv run python eval_convcnp.py --config all --save-models 2>&1 | tee data/convcnp_results/full_log.txt

echo "=== All configs done ==="
date

# Upload results to GCS
gsutil -m cp -r data/convcnp_results/ "gs://${RESULTS_BUCKET}/data/convcnp_results/"
gsutil -m cp -r data/models/ "gs://${RESULTS_BUCKET}/data/models/" 2>/dev/null || true

# Signal success
echo "SUCCESS" > /tmp/status.txt
gsutil cp /tmp/status.txt "gs://${RESULTS_BUCKET}/status.txt"

# Self-destruct
shutdown -h now
