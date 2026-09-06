#!/usr/bin/env python3
"""
sea_top.py - Real-Time Centralized Telemetry & Fleet Monitor for SEA on CX3 HPC.

Displays a comprehensive, colorized dashboard of all distributed GPU workers,
active inference jobs, GPU health metrics (temp, VRAM), and atomic task queue states.
Works seamlessly on both macOS and CX3 cluster login nodes.
"""

import os
import sys
import time
import argparse
from datetime import datetime, timezone
from pathlib import Path

# Add src to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

try:
    from telemetry import get_all_telemetry, ensure_telemetry_dirs
    from queue_manager import get_queue_stats, ensure_queue_dirs
except ImportError:
    print("❌ Failed to import telemetry/queue_manager modules.", file=sys.stderr)
    sys.exit(1)

# ANSI Colors
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
MAGENTA = "\033[35m"
BLUE = "\033[34m"
WHITE = "\033[37m"
BG_BLUE = "\033[44m"


def format_progress_bar(current: int, total: int, width: int = 15) -> str:
    if total <= 0:
        return f"[{' ' * width}]   0.0%"
    pct = min(max(current / total, 0.0), 1.0)
    filled = int(round(width * pct))
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}] {pct * 100:5.1f}%"


def render_dashboard(clear_screen: bool = False):
    if clear_screen:
        sys.stdout.write("\033[H\033[J")

    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    stats = get_queue_stats()
    workers = get_all_telemetry()

    # Header
    print(f"{BOLD}{BG_BLUE}{WHITE}  🛰️  SEA CLUSTER TELEMETRY & TASK QUEUE DASHBOARD  {RESET}  {DIM}{now_str}{RESET}")
    print(f"{CYAN}Tasks in Queue:{RESET} {YELLOW}{stats['pending_count']} Pending{RESET} | {GREEN}{stats['active_count']} Active{RESET} | {BLUE}{stats['done_count']} Completed{RESET} (Total: {stats['total_count']})")
    print("-" * 115)

    # Worker Fleet Table
    print(f"{BOLD}{'WORKER ID':<22} | {'NODE':<10} | {'GPU':<4} | {'TEMP':<6} | {'VRAM (GB)':<12} | {'LOCATION/DATE':<20} | {'PROGRESS':<25} | {'SPEED':<11} | {'STATUS'}{RESET}")
    print("-" * 115)

    if not workers:
        print(f"{DIM}  (No registered workers found in telemetry hub. Start workers with run_queue_worker.py){RESET}")
    else:
        for w in workers:
            wid = w.get("worker_id", "-")
            if len(wid) > 22:
                wid = wid[:19] + "..."
            node = w.get("node", "-")
            gpu_id = str(w.get("gpu_id", 0))

            gpu_info = w.get("gpu", {})
            temp = gpu_info.get("gpu_temp_c")
            temp_str = f"{temp}°C" if temp is not None else "-"
            if temp is not None and temp > 80:
                temp_str = f"{RED}{temp_str}{RESET}"
            elif temp is not None:
                temp_str = f"{GREEN}{temp_str}{RESET}"

            vram_used = gpu_info.get("vram_used_mb")
            vram_total = gpu_info.get("vram_total_mb")
            if vram_used is not None and vram_total is not None:
                vram_str = f"{vram_used/1024:.1f} / {vram_total/1024:.1f}"
            elif vram_used is not None:
                vram_str = f"{vram_used/1024:.1f} GB"
            else:
                vram_str = "-"

            loc = w.get("location", "-")
            dt = w.get("date", "-")
            loc_dt = f"{loc} {dt}" if loc != "-" and dt != "-" else "-"

            cur = w.get("current_rec", 0)
            tot = w.get("total_recs", 0)
            prog_bar = format_progress_bar(cur, tot, width=12)

            speed = w.get("win_per_sec", 0.0)
            speed_str = f"{speed:.0f} win/s" if speed > 0 else "-"

            liveness = w.get("liveness", "ALIVE")
            st = w.get("status", "ACTIVE")
            age = w.get("age_sec", 0)

            if liveness == "STALE":
                status_colored = f"{RED}STALE ({age:.0f}s){RESET}"
            elif "STANDBY" in st or "IDLE" in st:
                status_colored = f"{YELLOW}{st}{RESET}"
            else:
                status_colored = f"{GREEN}{st}{RESET}"

            print(f"{wid:<22} | {node:<10} | {gpu_id:<4} | {temp_str:<15} | {vram_str:<12} | {loc_dt:<20} | {prog_bar:<25} | {speed_str:<11} | {status_colored}")

    print("-" * 115)


def main():
    parser = argparse.ArgumentParser(description="SEA Fleet Telemetry Top Monitor")
    parser.add_argument("--watch", "-w", action="store_true", help="Continuously refresh dashboard")
    parser.add_argument("--interval", "-n", type=int, default=5, help="Refresh interval in seconds (default: 5)")
    args = parser.parse_args()

    ensure_telemetry_dirs()
    ensure_queue_dirs()

    if args.watch:
        try:
            while True:
                render_dashboard(clear_screen=True)
                time.sleep(args.interval)
        except KeyboardInterrupt:
            print(f"\n{DIM}Dashboard exited.{RESET}")
    else:
        render_dashboard(clear_screen=False)


if __name__ == "__main__":
    main()
