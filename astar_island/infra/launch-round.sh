#!/bin/bash
# Launch full round pipeline on GCE.
# Usage: bash infra/launch-round.sh 17
#        bash infra/launch-round.sh 17 --skip-validation
set -euo pipefail

TARGET_ROUND="${1:?Usage: launch-round.sh <round_number>}"
EXTRA_ARGS="${2:-}"

GCP_PROJECT="ainm26osl-708"
GCS_BUCKET="ainm-results-${GCP_PROJECT}"
VM_NAME="tom-ainm-r${TARGET_ROUND}"
# g2-standard-48: 48 vCPUs + 4x L4 GPUs
MACHINE_TYPE="g2-standard-48"
ZONE="europe-west1-c"

echo "=== Round ${TARGET_ROUND} Pipeline ==="
echo "  VM:      ${VM_NAME} (${MACHINE_TYPE})"
echo "  Zone:    ${ZONE}"
echo "  Bucket:  gs://${GCS_BUCKET}"
echo ""

# Upload latest code
echo "Uploading code..."
cd "$(dirname "$0")/.."
gsutil -m -q rsync -r -x '\.git/|\.venv/|__pycache__/|\.npy$|\.pyc$|\.pkl$' . "gs://${GCS_BUCKET}/code/"

# Ensure torch + tqdm in deps
grep -q torch pyproject.toml || { echo "ERROR: torch not in pyproject.toml"; exit 1; }

# Upload fresh deps
gsutil cp pyproject.toml "gs://${GCS_BUCKET}/code/pyproject.toml"
gsutil cp uv.lock "gs://${GCS_BUCKET}/code/uv.lock"

# Clear old status
gsutil rm "gs://${GCS_BUCKET}/status_pipeline.txt" 2>/dev/null || true

# Create VM
echo "Creating VM..."
gcloud compute instances create "${VM_NAME}" \
  --project="${GCP_PROJECT}" \
  --zone="${ZONE}" \
  --machine-type="${MACHINE_TYPE}" \
  --accelerator="type=nvidia-l4,count=4" \
  --scopes=storage-rw \
  --metadata-from-file=startup-script="infra/gce-pipeline.sh" \
  --metadata=results-bucket="${GCS_BUCKET}",target-round="${TARGET_ROUND}" \
  --boot-disk-size=50GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --maintenance-policy=TERMINATE

echo ""
echo "VM created: ${VM_NAME}"
echo "Console: https://console.cloud.google.com/compute/instancesDetail/zones/${ZONE}/instances/${VM_NAME}?project=${GCP_PROJECT}"
echo ""
echo "Monitor:"
echo "  gcloud compute ssh root@${VM_NAME} --zone=${ZONE} --project=${GCP_PROJECT} --command='tail -f /tmp/pipeline_results.txt'"
echo ""
echo "Results will be at:"
echo "  gs://${GCS_BUCKET}/pipeline_round${TARGET_ROUND}_results.txt"
echo "  gs://${GCS_BUCKET}/round${TARGET_ROUND}_predictions/"
