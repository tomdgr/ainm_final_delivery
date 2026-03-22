#!/bin/bash
# Spin up a GCE VM, upload code, run calibration + prediction, download results, tear down.
# Usage:
#   ./infra/gce-run.sh                    # calibrate only
#   ./infra/gce-run.sh --submit           # calibrate + submit
#   ./infra/gce-run.sh --machine c3-highcpu-44  # use a bigger machine
set -euo pipefail

# --- Config ---
GCP_PROJECT="ainm26osl-708"
GCE_ZONE="europe-west1-b"
VM_NAME="ainm-calibrate-$(date +%s)"
MACHINE_TYPE="c3-highcpu-44"
GCS_BUCKET="ainm-results-${GCP_PROJECT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DO_SUBMIT="false"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
  case $1 in
    --submit) DO_SUBMIT="true"; shift ;;
    --machine) MACHINE_TYPE="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# --- Load AINM token from .env ---
AINM_TOKEN=$(grep AINM_ACCESS_TOKEN "$PROJECT_DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")
if [ -z "$AINM_TOKEN" ]; then
  echo "ERROR: AINM_ACCESS_TOKEN not found in .env"
  exit 1
fi

echo "=== GCE Calibration Run ==="
echo "  VM:       $VM_NAME ($MACHINE_TYPE)"
echo "  Submit:   $DO_SUBMIT"
echo "  Bucket:   gs://$GCS_BUCKET"
echo ""

# --- Ensure GCS bucket exists ---
gsutil ls "gs://$GCS_BUCKET" &>/dev/null || gsutil mb -p "$GCP_PROJECT" -l europe-west1 "gs://$GCS_BUCKET"
gsutil rm "gs://$GCS_BUCKET/status.txt" 2>/dev/null || true

# --- Upload code to GCS ---
echo "Uploading code to GCS..."
cd "$PROJECT_DIR"

# Save code provenance so GCE manifest knows which code ran
python3 -c "
import hashlib, json, subprocess
def hash_file(p):
    try:
        return hashlib.sha256(open(p,'rb').read()).hexdigest()[:16]
    except: return 'missing'
commit = subprocess.check_output(['git','rev-parse','--short','HEAD'], text=True).strip()
dirty = bool(subprocess.check_output(['git','status','--porcelain'], text=True).strip())
json.dump({
    'git_commit': commit,
    'git_dirty': 'dirty' if dirty else 'clean',
    'source_hashes': {
        'runner.py': hash_file('nm_ai_ml/astar/runner.py'),
        'simulator.py': hash_file('nm_ai_ml/astar/simulator.py'),
        'improved_strategy.py': hash_file('nm_ai_ml/astar/improved_strategy.py'),
        'calibrate.py': hash_file('nm_ai_ml/astar/calibrate.py'),
    }
}, open('code_manifest.json','w'), indent=2)
"
echo "  Code manifest: $(cat code_manifest.json)"

gsutil -m -q rsync -r -x '\.git/|\.venv/|__pycache__/|\.npy$|\.pyc$' . "gs://$GCS_BUCKET/code/"

# --- Create VM ---
echo "Creating VM..."
gcloud compute instances create "$VM_NAME" \
  --project="$GCP_PROJECT" \
  --zone="$GCE_ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --scopes=storage-rw \
  --metadata-from-file=startup-script="$SCRIPT_DIR/gce-startup.sh" \
  --metadata=results-bucket="$GCS_BUCKET",ainm-token="$AINM_TOKEN",do-submit="$DO_SUBMIT" \
  --boot-disk-size=20GB \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud

echo "VM created. Waiting for completion..."

# --- Poll for completion ---
cleanup() {
  echo ""
  echo "Cleaning up VM..."
  gcloud compute instances delete "$VM_NAME" --zone="$GCE_ZONE" --project="$GCP_PROJECT" --quiet 2>/dev/null || true
}
trap cleanup EXIT

MAX_WAIT=160  # 160 * 30s = 80 minutes
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
        --zone="$GCE_ZONE" --project="$GCP_PROJECT" 2>/dev/null | tail -50
      exit 1
    fi
  fi

  # Check if VM still exists
  if ! gcloud compute instances describe "$VM_NAME" --zone="$GCE_ZONE" --project="$GCP_PROJECT" &>/dev/null; then
    echo ""
    echo "ERROR: VM disappeared (preempted or crashed)"
    exit 1
  fi

  printf "\r  Waiting... %d/%d (%.0f min)" "$i" "$MAX_WAIT" "$(echo "$i * 0.5" | bc)"
  sleep 30
done

if [ "$FINISHED" != "true" ]; then
  echo ""
  echo "ERROR: Timed out waiting for VM to complete"
  exit 1
fi

# --- Download results ---
echo ""
echo "Downloading results from GCS..."
gsutil -m cp -r "gs://$GCS_BUCKET/data/rounds/" "$PROJECT_DIR/data/rounds/"
gsutil cp "gs://$GCS_BUCKET/data/calibrated_params.json" "$PROJECT_DIR/data/calibrated_params.json" 2>/dev/null || true

echo ""
echo "=== Done! ==="
echo "Results saved to $PROJECT_DIR/data/"
echo "To commit: git add data/ && git commit -m 'GCE calibration results'"
