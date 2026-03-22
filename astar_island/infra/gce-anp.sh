#!/bin/bash
# Train Attentive Neural Process on GPU, predict round 16
set -euo pipefail

export HOME="${HOME:-/root}"
WORKDIR=/opt/ainm
RESULTS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/results-bucket" -H "Metadata-Flavor: Google")

apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv curl > /dev/null 2>&1

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$WORKDIR"
gsutil -m cp -r "gs://${RESULTS_BUCKET}/code/*" "$WORKDIR/"
cd "$WORKDIR"

uv sync

echo "=== ANP Training + Prediction ==="
uv run python run_anp.py 2>&1 | tee /tmp/anp_results.txt

# Upload results
gsutil cp /tmp/anp_results.txt "gs://${RESULTS_BUCKET}/anp_results.txt"
gsutil -m cp data/rounds/round_16/predictions/seed_*_anp.npy "gs://${RESULTS_BUCKET}/anp_predictions/" 2>/dev/null || true
gsutil cp data/anp_model.pt "gs://${RESULTS_BUCKET}/anp_model.pt" 2>/dev/null || true

echo "SUCCESS" > /tmp/status_anp.txt
gsutil cp /tmp/status_anp.txt "gs://${RESULTS_BUCKET}/status_anp.txt"

shutdown -h now
