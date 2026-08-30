#!/usr/bin/env python3
"""Audit bacpipe embedding progress: which (location, date, model) still need resume.

Ground truth is the actual sea-work output folder (summary.json + .ckpt chunk
state), not PBS job state -- so this still works after a job has already died
from walltime and left no trace in qstat.

Status per (location, date, model):
  DONE       - {date}_summary.json exists and covers every method present on disk
  RESUMABLE  - .ckpt/{date}_state.json exists but no complete summary yet
               (job died mid-stream; last_n is the exact WAV index to resume from)
  ACTIVE     - same as RESUMABLE, but a live run_pilot.py process for that date
               is currently found running on a compute node (not actually stuck)
  NOT_STARTED - raw WAVs exist for that date but no summary/.ckpt at all

Usage (run on CX3, e.g. login node is enough -- no GPU/venv needed):
  python3 audit_resume_status.py                  # all locations, only rows needing attention
  python3 audit_resume_status.py --location 2A400
  python3 audit_resume_status.py --all             # also print DONE rows
  python3 audit_resume_status.py --no-live-check   # skip qstat/ssh cross-check
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

DATA_DIR = Path(
    os.environ.get(
        "ANALYSIS_OUTPUT", f"/rds/general/user/{os.environ.get('USER', 'ri322')}/ephemeral/sea-work"
    )
)
METHODS = ("bf_LabIR", "bf_SPIR", "sa", "mono")
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def count_wavs(date_dir: Path) -> int:
    total = 0
    for method in METHODS:
        d = date_dir / method
        if d.is_dir():
            total += sum(1 for p in d.rglob("*.wav") if not p.name.startswith("._"))
    return total


def model_status(model_dir: Path, date: str, methods_present: set) -> tuple:
    summary_path = model_dir / f"{date}_summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
            done_methods = set(summary.get("methods", {}).keys())
            if methods_present <= done_methods:
                n = sum(summary["methods"][m]["n_embeddings"] for m in done_methods)
                return "DONE", n
        except (json.JSONDecodeError, OSError, KeyError):
            pass

    state_path = model_dir / ".ckpt" / f"{date}_state.json"
    if state_path.exists():
        try:
            state = json.loads(state_path.read_text())
            return "RESUMABLE", state.get("last_n", 0)
        except (json.JSONDecodeError, OSError):
            return "RESUMABLE", 0

    return "NOT_STARTED", 0


def get_active_dates() -> set:
    """Best-effort: dates currently being processed by a live run_pilot.py on any node."""
    active = set()
    try:
        out = subprocess.run(
            ["bash", "-lc", "qstat -nu $USER -1"],
            capture_output=True, text=True, timeout=15,
        ).stdout
    except Exception:
        return active

    for line in out.splitlines():
        parts = line.split()
        if len(parts) < 12 or not parts[0][:1].isdigit():
            continue
        state, node_field = parts[9], parts[11]
        if state != "R":
            continue
        node = node_field.split("/")[0]
        jobid = parts[0].split(".")[0]
        try:
            ps_out = subprocess.run(
                ["bash", "-lc",
                 f"export PBS_JOBID={jobid}; "
                 f"ssh -o BatchMode=yes -o ConnectTimeout=5 {node} "
                 f"'ps -u $USER -o cmd --no-headers'"],
                capture_output=True, text=True, timeout=15,
            ).stdout
        except Exception:
            continue
        for m in re.finditer(r"run_pilot\.py.*?--date\s+(\S+)", ps_out):
            active.add(m.group(1))
    return active


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--location")
    p.add_argument("--all", action="store_true", help="also print DONE rows")
    p.add_argument("--no-live-check", action="store_true")
    args = p.parse_args()

    active_dates = set() if args.no_live_check else get_active_dates()
    if active_dates:
        print(f"Live run_pilot.py dates right now: {sorted(active_dates)}\n")

    locations = (
        [args.location]
        if args.location
        else sorted(
            d.name for d in DATA_DIR.iterdir()
            if d.is_dir() and (d / "embeddings" / "bacpipe").is_dir()
        )
    )

    for loc in locations:
        loc_dir = DATA_DIR / loc
        bacpipe_dir = loc_dir / "embeddings" / "bacpipe"
        models = sorted(d.name for d in bacpipe_dir.iterdir() if d.is_dir())
        dates = sorted(
            d.name for d in loc_dir.iterdir() if d.is_dir() and DATE_RE.match(d.name)
        )

        print(f"=== {loc} ===")
        for date in dates:
            methods_present = {m for m in METHODS if (loc_dir / date / m).is_dir()}
            if not methods_present:
                continue
            n_wavs = count_wavs(loc_dir / date)
            rows = []
            for model in models:
                status, last_n = model_status(bacpipe_dir / model, date, methods_present)
                if status == "RESUMABLE" and date in active_dates:
                    status = "ACTIVE"
                if status == "DONE" and not args.all:
                    continue
                rows.append((model, status, last_n))
            if not rows:
                continue
            print(f"  {date}  ({n_wavs} WAVs on disk)")
            for model, status, last_n in rows:
                print(f"    {model:<22} {status:<12} last_n={last_n}/{n_wavs}")
        print()


if __name__ == "__main__":
    main()
