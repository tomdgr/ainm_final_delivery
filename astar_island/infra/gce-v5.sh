#!/bin/bash
set -euo pipefail
export HOME="${HOME:-/root}"
WORKDIR=/opt/ainm
RESULTS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/results-bucket" -H "Metadata-Flavor: Google")
TARGET_ROUND=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/target-round" -H "Metadata-Flavor: Google" || echo "17")

apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv curl > /dev/null 2>&1
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

mkdir -p "$WORKDIR"
gsutil -m cp -r "gs://${RESULTS_BUCKET}/code/*" "$WORKDIR/"
cd "$WORKDIR"

uv add torch tqdm xgboost lightgbm 2>&1 | tail -5
uv sync

echo "=== V5 Pipeline: Parallel synth + XGB + CNP ==="
uv run python round_pipeline_v5.py --round "${TARGET_ROUND}" 2>&1 | tee /tmp/v5_results.txt

gsutil -m cp -r data/model_predictions_v5/ "gs://${RESULTS_BUCKET}/model_predictions_v5/"
gsutil cp /tmp/v5_results.txt "gs://${RESULTS_BUCKET}/v5_results.txt"
echo "SUCCESS" > /tmp/status_v5.txt
gsutil cp /tmp/status_v5.txt "gs://${RESULTS_BUCKET}/status_v5.txt"
shutdown -h now
