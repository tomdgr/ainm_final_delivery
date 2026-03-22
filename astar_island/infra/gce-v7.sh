#!/bin/bash
set -euo pipefail
export HOME="${HOME:-/root}"
WORKDIR=/opt/ainm
RESULTS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/results-bucket" -H "Metadata-Flavor: Google")
TARGET_ROUND=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/target-round" -H "Metadata-Flavor: Google" || echo "22")

apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv curl > /dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$WORKDIR"
gsutil -m cp -r "gs://${RESULTS_BUCKET}/code/*" "$WORKDIR/"
cd "$WORKDIR"

uv add torch tqdm xgboost lightgbm 2>&1 | tail -3
uv sync

echo "=== V7: Deep CNP Ensemble + Gated Blend + Eval ==="
uv run python experiment_v7.py --round "${TARGET_ROUND}" --synth-episodes 500 --n-cnp-models 3 2>&1 | tee /tmp/v7_results.txt

gsutil -m cp -r data/model_predictions_v7/ "gs://${RESULTS_BUCKET}/model_predictions_v7/" 2>/dev/null || true
gsutil cp /tmp/v7_results.txt "gs://${RESULTS_BUCKET}/v7_r${TARGET_ROUND}_results.txt"
echo "SUCCESS" > /tmp/status_v7.txt
gsutil cp /tmp/status_v7.txt "gs://${RESULTS_BUCKET}/status_v7.txt"
shutdown -h now
