#!/bin/bash
# One-command round submission with safety net.
# Usage: bash submit_round.sh 22
#
# Step 1: Submit RF-only immediately (safe baseline ~85-87)
# Step 2: Train ensemble on GCE, validate with OOF
# Step 3: If ensemble beats RF → resubmit ensemble
set -euo pipefail

ROUND="${1:?Usage: submit_round.sh <round_number>}"
cd "$(dirname "$0")"

echo "=== Round ${ROUND} Submission Pipeline ==="
echo ""

# Step 1: Fetch data
echo "[1/4] Fetching round data..."
uv run python main.py --fetch 2>&1 | tail -5

# Step 2: Submit RF immediately (safety net)
echo ""
echo "[2/4] Submitting RF baseline (safety net)..."
uv run python -c "
import json, numpy as np
from pathlib import Path
from nm_ai_ml.astar.spatial_predictor_rf import predict_round
from nm_ai_ml.astar.client import AstarClient

rd = Path('data/rounds/round_${ROUND}')
with open(rd / 'round_detail.json') as f:
    detail = json.load(f)

preds = predict_round(str(rd))
client = AstarClient()
for si, pred in enumerate(preds):
    pred = np.maximum(pred, 0.005)
    pred /= pred.sum(axis=2, keepdims=True)
    resp = client.submit(detail['id'], si, pred.tolist())
    print(f'  Seed {si}: {resp[\"status\"]}')
client.close()
print('RF baseline submitted!')
"

# Step 3: Upload to GCE and run full pipeline
echo ""
echo "[3/4] Launching full pipeline on GCE..."
# Upload latest code
gsutil -m -q rsync -r -x '\.git/|\.venv/|__pycache__/|\.npy$|\.pyc$|\.pkl$' . "gs://ainm-results-ainm26osl-708/code/" 2>/dev/null

# Push to existing VM or create new one
VM_NAME="tom-ainm-r${ROUND}"
ZONE="europe-west1-c"
PROJECT="ainm26osl-708"

if gcloud compute instances describe "$VM_NAME" --zone="$ZONE" --project="$PROJECT" &>/dev/null; then
    echo "VM $VM_NAME exists, pushing code..."
    gcloud compute scp experiment_v6.py "root@${VM_NAME}:/opt/ainm/" --zone="$ZONE" --project="$PROJECT"
else
    echo "No VM found. Use existing tom-ainm-r17 or create new one."
fi

echo ""
echo "[4/4] Monitor with:"
echo "  gcloud compute ssh root@tom-ainm-r17 --zone=europe-west1-c --project=ainm26osl-708 --command='tail -f /tmp/v6_oof.txt'"
echo ""
echo "When ensemble results are ready, resubmit with:"
echo "  uv run python submit_ensemble.py --round ${ROUND} --models rf cnp anp --weights 0.45 0.35 0.20"
