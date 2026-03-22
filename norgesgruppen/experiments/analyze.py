#!/usr/bin/env python3
"""Analyze experiment results from Azure ML."""

import argparse
import csv
import json
import re
import subprocess
from collections import defaultdict
from pathlib import Path


DIM_LABELS = {
    "a": "AUGMENTATION",
}

DIM_DEFAULTS = {
    "a": "medium",
}


def get_completed_jobs(experiment_name: str) -> list[dict]:
    cmd = [
        "az", "ml", "job", "list",
        "--resource-group", "rg-nmai-workspace",
        "--workspace-name", "nmai-experis",
        "--query",
        f"[?experiment_name=='{experiment_name}' && status=='Completed']."
        "{name:name, display_name:display_name}",
        "-o", "json",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  Error listing jobs: {result.stderr[:200]}")
        return []
    return json.loads(result.stdout)


def download_job_logs(job_name: str, download_dir: Path) -> Path | None:
    job_dir = download_dir / job_name
    log_path = job_dir / "artifacts" / "user_logs" / "std_log.txt"
    if log_path.exists():
        return job_dir

    cmd = [
        "az", "ml", "job", "download",
        "--name", job_name,
        "--resource-group", "rg-nmai-workspace",
        "--workspace-name", "nmai-experis",
        "--download-path", str(job_dir),
        "--all",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FAILED ({result.stderr[:100]})")
        return None
    return job_dir


def parse_metrics_from_log(log_path: Path) -> dict | None:
    if not log_path.exists():
        return None

    content = log_path.read_text()
    config_match = re.search(r"CONFIG_JSON: ({.*})", content)
    results_match = re.search(r"RESULTS_JSON: ({.*})", content)

    if not config_match:
        return None

    config = json.loads(config_match.group(1))
    metrics = json.loads(results_match.group(1)) if results_match else {}

    parts = [f"{k}_{config.get(k, DIM_DEFAULTS.get(k, 'none'))}" for k in DIM_LABELS]
    parts.append(f"m_{config.get('model', 'unknown')}")
    config_key = "-".join(parts)

    active = []
    for dim_key, default_val in DIM_DEFAULTS.items():
        val = config.get(dim_key, default_val)
        if val != default_val:
            active.append((DIM_LABELS[dim_key].lower(), val))

    if len(active) == 0:
        technique_type = "baseline"
        technique = "baseline"
    elif len(active) == 1:
        technique_type = active[0][0]
        technique = active[0][1]
    else:
        technique_type = "combination"
        technique = "+".join(t[1] for t in active)

    return {
        "config": config,
        "config_key": config_key,
        "model": config.get("model", "unknown"),
        "type": technique_type,
        "technique": technique,
        **metrics,
    }


def print_leaderboard(results: list[dict], output_csv: str | None = None):
    if not results:
        print("No results found!")
        return

    best_by_config = {}
    for r in results:
        key = r["config_key"]
        if key not in best_by_config or r.get("f1", 0) > best_by_config[key].get("f1", 0):
            best_by_config[key] = r

    results = sorted(best_by_config.values(), key=lambda x: x.get("f1", 0), reverse=True)
    print(f"After deduplication: {len(results)} unique configs\n")

    lines = []

    def log(line=""):
        print(line)
        lines.append(line)

    log("=" * 120)
    log("EXPERIMENT RESULTS — Ranked by F1 (higher is better)")
    log("=" * 120)
    log()
    log(f"{'Rank':<5} {'Type':<15} {'Technique':<30} {'Model':<22} "
        f"{'F1':>8} {'Acc':>8} {'Prec':>8} {'Rec':>8}")
    log("-" * 120)

    for i, r in enumerate(results, 1):
        f1_s = f"{r.get('f1', 0):.4f}" if "f1" in r else "—"
        acc_s = f"{r.get('accuracy', 0):.4f}" if "accuracy" in r else "—"
        prec_s = f"{r.get('precision', 0):.4f}" if "precision" in r else "—"
        rec_s = f"{r.get('recall', 0):.4f}" if "recall" in r else "—"
        model = r.get("model", "unknown")

        log(f"{i:<5} {r['type']:<15} {r['technique']:<30} {model:<22} "
            f"{f1_s:>8} {acc_s:>8} {prec_s:>8} {rec_s:>8}")

    log()
    log(f"Total experiments analyzed: {len(results)}")

    if output_csv:
        fieldnames = [
            "rank", "type", "technique", "model",
            *DIM_LABELS.keys(),
            "f1", "accuracy", "precision", "recall",
        ]
        with open(output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for rank, r in enumerate(results, 1):
                row = {"rank": rank, **r, **r.get("config", {})}
                writer.writerow(row)
        print(f"\nCSV saved to: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="experiment_suite")
    parser.add_argument("-o", "--output", help="Output CSV")
    parser.add_argument("--download-dir", default="experiments/logs")
    args = parser.parse_args()

    experiments = [e.strip() for e in args.experiment.split(",")]
    jobs = []
    for exp in experiments:
        print(f"Fetching completed jobs from: {exp}")
        exp_jobs = get_completed_jobs(exp)
        print(f"  Found {len(exp_jobs)} completed jobs")
        jobs.extend(exp_jobs)
    print(f"Total: {len(jobs)} completed jobs\n")

    if not jobs:
        return

    download_dir = Path(args.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for job in jobs:
        job_name = job["name"]
        job_dir = download_job_logs(job_name, download_dir)
        if job_dir is None:
            continue

        log_path = job_dir / "artifacts" / "user_logs" / "std_log.txt"
        metrics = parse_metrics_from_log(log_path)
        if metrics:
            metrics["job_name"] = job_name
            results.append(metrics)

    output_path = args.output or "experiments/results/leaderboard.csv"
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print_leaderboard(results, output_path)


if __name__ == "__main__":
    main()
