#!/bin/bash
# Spin up a GCE VM to run ConvCNP experiments (LOO-CV across all configs).
# Usage:
#   ./infra/gce-convcnp-run.sh
#   ./infra/gce-convcnp-run.sh --machine c3-highcpu-22
set -euo pipefail

# --- Ensure gcloud is in PATH ---
export PATH="$HOME/google-cloud-sdk/bin:/opt/homebrew/Caskroom/google-cloud-sdk/latest/google-cloud-sdk/bin:$PATH"

# --- Config ---
GCP_PROJECT="ainm26osl-708"
GCE_ZONE="europe-west1-b"
VM_NAME="ainm-convcnp-$(date +%s)"
MACHINE_TYPE="c3-highcpu-44"
GCS_BUCKET="ainm-results-${GCP_PROJECT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --machine) MACHINE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

echo "=== GCE ConvCNP Experiment Run ==="
echo "  VM:       $VM_NAME ($MACHINE_TYPE)"
echo "  Bucket:   gs://$GCS_BUCKET"
echo ""

# --- Ensure GCS bucket exists ---
gsutil ls "gs://$GCS_BUCKET" &>/dev/null || gsutil mb -p "$GCP_PROJECT" -l europe-west1 "gs://$GCS_BUCKET"
gsutil rm "gs://$GCS_BUCKET/status.txt" 2>/dev/null || true

# --- Upload code to GCS ---
echo "Uploading code to GCS..."
cd "$PROJECT_DIR"
# Upload code — allow individual file failures (e.g. binary files that confuse gsutil)
gsutil -m -q rsync -r -x '\.git/|\.venv/|__pycache__/|\.npy$|\.pyc$|data/models/|data/convcnp_results/|\.egg-info/' . "gs://$GCS_BUCKET/code/" || true
# Force-upload critical files that may have been cached
gsutil -q cp eval_convcnp.py "gs://$GCS_BUCKET/code/eval_convcnp.py"
gsutil -q cp pyproject.toml "gs://$GCS_BUCKET/code/pyproject.toml"
gsutil -q cp uv.lock "gs://$GCS_BUCKET/code/uv.lock"
gsutil -q cp nm_ai_ml/astar/convcnp.py "gs://$GCS_BUCKET/code/nm_ai_ml/astar/convcnp.py"

# --- Create VM ---
echo "Creating VM..."
gcloud compute instances create "$VM_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$GCE_ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --scopes=storage-rw \
  --metadata-from-file=startup-script="$SCRIPT_DIR/gce-convcnp-startup.sh" \
  --metadata=results-bucket="$GCS_BUCKET" \
  --boot-disk-size=20GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

echo "VM created. Waiting for completion..."
echo "(This will take a while — 17 configs × 14 folds × ~5 min each ≈ hours)"

# --- Poll for completion ---
cleanup() {
  echo ""
  echo "Cleaning up VM..."
  gcloud compute instances delete "$VM_NAME" --zone="$GCE_ZONE" --project="$GCP_PROJECT" --quiet 2>/dev/null || true
}
trap cleanup EXIT

MAX_WAIT=720  # 720 * 30s = 6 hours
FINISHED=false
for i in $(seq 1 $MAX_WAIT); do
  if gsutil -q stat "gs://$GCS_BUCKET/status.txt" 2>/dev/null; then
    STATUS=$(gsutil cat "gs://$GCS_BUCKET/status.txt")
    echo ""
    echo "VM finished with status: $STATUS"
    if [ "$STATUS" = "SUCCESS" ]; then
      FINISHED=true
      break
    else
      echo "ERROR: VM reported failure"
      gcloud compute instances get-serial-port-output "$VM_NAME" \
        --zone="$GCE_ZONE" --project="$GCP_PROJECT" 2>/dev/null | tail -100
      exit 1
    fi
  fi

  # Check for intermediate results every 5 min
  if (( i % 10 == 0 )); then
    echo ""
    echo "--- Checking intermediate results ($(( i * 30 / 60 )) min elapsed) ---"
    INTERMEDIATE=$(gsutil ls "gs://$GCS_BUCKET/data/convcnp_results/*.json" 2>/dev/null || true)
    if [ -n "$INTERMEDIATE" ]; then
      echo "$INTERMEDIATE" | while read f; do
        echo "  $(basename "$f")"
      done
    else
      echo "  (no results yet)"
    fi
    echo ""
  fi

  # Check if VM still exists
  if ! gcloud compute instances describe "$VM_NAME" --zone="$GCE_ZONE" --project="$GCP_PROJECT" &>/dev/null; then
    echo ""
    echo "ERROR: VM disappeared (preempted or crashed)"
    echo "Downloading any partial results..."
    gsutil -m cp -r "gs://$GCS_BUCKET/data/convcnp_results/" "$PROJECT_DIR/data/convcnp_results/" 2>/dev/null || true
    exit 1
  fi

  printf "\r  Waiting... %d/%d (%.0f min)" "$i" "$MAX_WAIT" "$(echo "$i * 0.5" | bc)"
  sleep 30
done

if [ "$FINISHED" != "true" ]; then
  echo ""
  echo "ERROR: Timed out"
  echo "Downloading partial results..."
  gsutil -m cp -r "gs://$GCS_BUCKET/data/convcnp_results/" "$PROJECT_DIR/data/convcnp_results/" 2>/dev/null || true
  exit 1
fi

# --- Download results ---
echo ""
echo "Downloading results from GCS..."
mkdir -p "$PROJECT_DIR/data/convcnp_results" "$PROJECT_DIR/data/models"
gsutil -m cp -r "gs://$GCS_BUCKET/data/convcnp_results/" "$PROJECT_DIR/data/convcnp_results/"
gsutil -m cp -r "gs://$GCS_BUCKET/data/models/" "$PROJECT_DIR/data/models/" 2>/dev/null || true

echo ""
echo "=== Done! ==="
echo "Results in $PROJECT_DIR/data/convcnp_results/"
cat "$PROJECT_DIR/data/convcnp_results/summary.json" 2>/dev/null || true
