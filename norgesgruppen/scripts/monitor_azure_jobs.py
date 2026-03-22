"""Monitor Azure ML jobs and download completed ones.

Usage:
  uv run python scripts/monitor_azure_jobs.py --interval 900  # check every 15 min
  uv run python scripts/monitor_azure_jobs.py --download-all   # download all completed
"""
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path

SUB = "0a2942e9-987d-4858-a1e9-a46350d0c669"
WS = "nmai-experis"
RG = "rg-nmai-workspace"
DOWNLOAD_DIR = Path("/tmp/azure_models")


def get_jobs(max_results=100):
    result = subprocess.run(
        ["az", "ml", "job", "list",
         "--subscription", SUB, "-w", WS, "-g", RG,
         "--max-results", str(max_results),
         "-o", "json"],
        capture_output=True, text=True, timeout=120
    )
    if result.returncode != 0:
        print(f"  Error listing jobs: {result.stderr[-200:]}")
        return []
    return json.loads(result.stdout)


def download_job(job_name, display_name):
    dl_dir = DOWNLOAD_DIR / display_name
    if dl_dir.exists() and any(dl_dir.rglob("*.pt")):
        return True  # already downloaded

    dl_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["az", "ml", "job", "download",
         "--name", job_name,
         "--subscription", SUB, "-w", WS, "-g", RG,
         "--output-name", "model_output",
         "--download-path", str(dl_dir)],
        capture_output=True, text=True, timeout=300
    )
    # Check if we got a .pt file
    pt_files = list(dl_dir.rglob("*.pt"))
    return len(pt_files) > 0


def check_and_report():
    now = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'='*60}")
    print(f"Azure Job Monitor — {now}")
    print(f"{'='*60}")

    jobs = get_jobs()
    if not jobs:
        print("  No jobs found or error")
        return

    completed = [j for j in jobs if j.get("status") == "Completed"]
    running = [j for j in jobs if j.get("status") == "Running"]
    queued = [j for j in jobs if j.get("status") == "Queued"]
    failed = [j for j in jobs if j.get("status") == "Failed"]

    print(f"  Completed: {len(completed)} | Running: {len(running)} | Queued: {len(queued)} | Failed: {len(failed)}")

    if running:
        print(f"\n  Running:")
        for j in running[:10]:
            print(f"    {j['display_name']}")
        if len(running) > 10:
            print(f"    ... and {len(running)-10} more")

    # Check for newly completed
    downloaded = 0
    failed_dl = 0
    for j in completed:
        display = j.get("display_name", j["name"])
        dl_dir = DOWNLOAD_DIR / display
        if dl_dir.exists() and any(dl_dir.rglob("*.pt")):
            continue  # already downloaded

        print(f"  Downloading {display}...", end=" ", flush=True)
        if download_job(j["name"], display):
            downloaded += 1
            print("OK")
        else:
            failed_dl += 1
            print("EMPTY (no .pt output)")

    if downloaded:
        print(f"\n  Downloaded {downloaded} new models")

    # Summary of available models
    all_pts = list(DOWNLOAD_DIR.rglob("*.pt"))
    if all_pts:
        print(f"\n  Total models available: {len(all_pts)}")
        # Group by category
        categories = {}
        for pt in all_pts:
            name = pt.stem
            if "noflip" in name or "nf_" in name:
                cat = "noflip"
            elif "800" in name:
                cat = "800px"
            elif "_l_" in name:
                cat = "yolov8l"
            elif "_m_" in name:
                cat = "yolov8m"
            else:
                cat = "other"
            categories.setdefault(cat, []).append(name)

        for cat, models in sorted(categories.items()):
            print(f"    {cat}: {len(models)} models")

    return len(running) == 0 and len(queued) == 0


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=900, help="Check interval in seconds")
    parser.add_argument("--download-all", action="store_true", help="Download all completed and exit")
    args = parser.parse_args()

    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if args.download_all:
        check_and_report()
        return

    print(f"Monitoring Azure jobs every {args.interval}s. Press Ctrl+C to stop.")
    while True:
        all_done = check_and_report()
        if all_done:
            print("\n  ALL JOBS COMPLETE! Downloading everything...")
            check_and_report()
            print("\n  Done. All models downloaded.")
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
