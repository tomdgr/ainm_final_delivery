#!/bin/bash
# Download all completed job outputs, skipping already-downloaded ones
SUB="0a2942e9-987d-4858-a1e9-a46350d0c669"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DL_DIR="$SCRIPT_DIR/../downloads"
mkdir -p "$DL_DIR"

# Get all completed jobs
az ml job list --subscription "$SUB" -w nmai-experis -g rg-nmai-workspace --max-results 100 \
  --query "[?experiment_name=='norgesgruppen_detection' && status=='Completed'].{name:name, display_name:display_name}" \
  -o json 2>/dev/null | python3 -c "
import json, sys
for j in json.load(sys.stdin):
    print(f\"{j['name']} {j['display_name']}\")
" | while read job_name display_name; do
  dest="$DL_DIR/$display_name"
  if [ -d "$dest/named-outputs/model_output" ] && [ -f "$dest/named-outputs/model_output/results.csv" ]; then
    echo "SKIP $display_name (already downloaded)"
    continue
  fi
  echo "DOWNLOADING $display_name ..."
  az ml job download --name "$job_name" --subscription "$SUB" -w nmai-experis -g rg-nmai-workspace \
    --output-name model_output --download-path "$dest" 2>/dev/null
  if [ $? -eq 0 ]; then
    echo "  OK"
  else
    echo "  FAILED"
  fi
done

echo ""
echo "=== Results Summary ==="
python3 -c "
import csv, os, glob

dl_dir = '$DL_DIR'
results = []
for d in sorted(os.listdir(dl_dir)):
    csv_path = os.path.join(dl_dir, d, 'named-outputs/model_output/results.csv')
    if os.path.exists(csv_path):
        with open(csv_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            if rows:
                map50_col = [c for c in rows[0].keys() if 'mAP50(B)' in c and 'mAP50-95' not in c][0]
                map5095_col = [c for c in rows[0].keys() if 'mAP50-95' in c][0]
                best_row = max(rows, key=lambda r: float(r[map50_col].strip()))
                map50 = float(best_row[map50_col].strip())
                map5095 = float(best_row[map5095_col].strip())
                epoch = list(best_row.values())[0].strip()
                results.append((d, map50, map5095, epoch))

results.sort(key=lambda x: x[1], reverse=True)
print(f\"{'Model':<35} {'mAP50':>8} {'mAP50-95':>10} {'BestEp':>8}\")
print('-' * 65)
for name, map50, map5095, ep in results:
    print(f'{name:<35} {map50:>8.4f} {map5095:>10.4f} {ep:>8}')
"
