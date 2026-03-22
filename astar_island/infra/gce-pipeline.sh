#!/bin/bash
# Full round pipeline on GCE — parallel model training, predict ALL rounds
set -euo pipefail

export HOME="${HOME:-/root}"
WORKDIR=/opt/ainm
RESULTS_BUCKET=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/results-bucket" -H "Metadata-Flavor: Google")
TARGET_ROUND=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/attributes/target-round" -H "Metadata-Flavor: Google" || echo "17")

# Install system deps
apt-get update -qq && apt-get install -y -qq python3.10 python3.10-venv curl > /dev/null 2>&1

# Try to install NVIDIA drivers for GPU support
if ! command -v nvidia-smi &> /dev/null; then
    echo "Attempting NVIDIA driver install..."
    apt-get install -y -qq ubuntu-drivers-common > /dev/null 2>&1
    ubuntu-drivers install > /dev/null 2>&1 || echo "Auto driver install failed"
    # Fallback: manual driver install
    apt-get install -y -qq nvidia-driver-550 > /dev/null 2>&1 || echo "Manual driver install failed, using CPU"
fi

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

# Download code
mkdir -p "$WORKDIR"
gsutil -m cp -r "gs://${RESULTS_BUCKET}/code/*" "$WORKDIR/"
cd "$WORKDIR"

# Install deps
uv add torch tqdm 2>&1 | tail -3
uv sync

# Check GPU
echo "=== GPU Status ==="
nvidia-smi 2>/dev/null || echo "No NVIDIA GPU detected"
uv run python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, GPUs: {torch.cuda.device_count()}')" 2>&1

# Run pipeline (all models in parallel)
echo "=== Round ${TARGET_ROUND} Pipeline ==="
uv run python round_pipeline.py \
    --round "${TARGET_ROUND}" \
    --cnp-epochs 100 \
    --anp-epochs 80 \
    --val-cnp-epochs 30 \
    --val-anp-epochs 20 \
    --output-dir data/model_predictions \
    2>&1 | tee /tmp/pipeline_results.txt

# Upload ALL predictions
echo "Uploading predictions..."
gsutil -m cp -r data/model_predictions/ "gs://${RESULTS_BUCKET}/model_predictions/"
gsutil cp /tmp/pipeline_results.txt "gs://${RESULTS_BUCKET}/pipeline_round${TARGET_ROUND}_results.txt"

echo "SUCCESS" > /tmp/status_pipeline.txt
gsutil cp /tmp/status_pipeline.txt "gs://${RESULTS_BUCKET}/status_pipeline.txt"

# Self-destruct
shutdown -h now
