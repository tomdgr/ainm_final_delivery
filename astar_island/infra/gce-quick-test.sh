#!/bin/bash
# Quick ensemble validation — 30 epoch models, leave-one-out on all rounds
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

uv add torch tqdm 2>&1 | tail -3
uv sync

echo "=== Quick Ensemble Validation ==="
uv run python quick_ensemble_test.py 2>&1 | tee /tmp/quick_ensemble_results.txt

gsutil cp /tmp/quick_ensemble_results.txt "gs://${RESULTS_BUCKET}/quick_ensemble_results.txt"

echo "SUCCESS" > /tmp/status_quick.txt
gsutil cp /tmp/status_quick.txt "gs://${RESULTS_BUCKET}/status_quick.txt"

shutdown -h now
