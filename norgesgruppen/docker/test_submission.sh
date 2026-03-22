#!/bin/bash
# Test a submission ZIP locally using a Docker container matching the sandbox.
#
# Usage:
#   ./docker/test_submission.sh submissions/yolov8x_onnx_clean.zip
#
# Requirements: Docker installed and running.

set -e

ZIP_PATH="$1"
if [ -z "$ZIP_PATH" ]; then
    echo "Usage: $0 <submission.zip>"
    exit 1
fi

if [ ! -f "$ZIP_PATH" ]; then
    echo "Error: $ZIP_PATH not found"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="nmai-sandbox-test"

echo "=== Building sandbox Docker image ==="
docker build -t "$IMAGE_NAME" -f "$SCRIPT_DIR/Dockerfile.test" "$SCRIPT_DIR"

echo ""
echo "=== Preparing test data ==="
TMPDIR=$(mktemp -d)
trap "rm -rf $TMPDIR" EXIT

# Unzip submission
mkdir -p "$TMPDIR/submission"
unzip -q "$ZIP_PATH" -d "$TMPDIR/submission"

# Copy a few test images
mkdir -p "$TMPDIR/images"
ls "$PROJECT_DIR/data/raw/coco_dataset/train/images/"img_*.jpg | head -5 | while read f; do
    cp "$f" "$TMPDIR/images/"
done

echo "Submission contents:"
ls -la "$TMPDIR/submission/"
echo ""
echo "Test images:"
ls "$TMPDIR/images/"

echo ""
echo "=== Running submission in Docker container ==="
echo "Command: python run.py --input /data/images --output /output/predictions.json"
echo ""

docker run --rm \
    -v "$TMPDIR/submission:/submission" \
    -v "$TMPDIR/images:/data/images:ro" \
    -v "$TMPDIR:/output" \
    -w /submission \
    --memory=8g \
    --cpus=4 \
    --network=none \
    "$IMAGE_NAME" \
    python run.py --input /data/images --output /output/predictions.json

echo ""
echo "=== Results ==="
if [ -f "$TMPDIR/predictions.json" ]; then
    PRED_COUNT=$(python3 -c "import json; print(len(json.load(open('$TMPDIR/predictions.json'))))")
    echo "SUCCESS: $PRED_COUNT predictions"
    echo "Sample:"
    python3 -c "import json; d=json.load(open('$TMPDIR/predictions.json')); print(json.dumps(d[:3], indent=2))" 2>/dev/null
else
    echo "FAILED: No predictions.json generated"
    exit 1
fi
