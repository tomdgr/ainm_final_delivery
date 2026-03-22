"""Astar Island — main entry point.

Usage:
    uv run python main.py --fetch                          # Fetch round data + observations (run first!)
    uv run python main.py                                  # Build predictions (no submit)
    uv run python main.py --calibrate                      # Calibrate sim params, then predict
    uv run python main.py --calibrate --submit             # Calibrate, predict, and submit
    uv run python main.py --calibrate --submit --no-fetch  # Same but skip API fetch (use saved data)
    uv run python main.py --submit                         # Predict with saved params and submit
    uv run python main.py --analysis                       # Fetch ground truth for completed rounds
"""
import sys

from nm_ai_ml.astar.runner import fetch, fetch_analysis, fetch_targeted, run

if __name__ == "__main__":
    if "--analysis" in sys.argv:
        fetch_analysis()
    elif "--targeted" in sys.argv:
        # Round 8 strategy: focused queries on settlement clusters for seed 0
        ROUND_8_TARGETS = [
            (0, 19, 6, 13),   # VP1: settlement cluster at (19,6)
            (0, 11, 22, 13),  # VP2: settlement cluster at (11,22)
            (0, 0, 1, 13),    # VP3: settlement cluster at (0,1)
        ]
        print("Targeted fetch plan:")
        for seed, vx, vy, n in ROUND_8_TARGETS:
            print(f"  Seed {seed}, viewport ({vx},{vy}) 15x15, {n} queries")
        print(f"  Total: {sum(n for _, _, _, n in ROUND_8_TARGETS)} queries")
        confirm = input("Proceed? [y/N] ")
        if confirm.strip().lower() == "y":
            fetch_targeted(ROUND_8_TARGETS)
        else:
            print("Aborted.")
    elif "--fetch" in sys.argv:
        fetch()
    else:
        do_submit = "--submit" in sys.argv
        do_calibrate = "--calibrate" in sys.argv
        skip_fetch = "--no-fetch" in sys.argv
        run(submit=do_submit, do_calibrate=do_calibrate, skip_fetch=skip_fetch)
