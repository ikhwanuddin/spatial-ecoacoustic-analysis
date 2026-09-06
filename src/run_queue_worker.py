#!/usr/bin/env python3
"""
run_queue_worker.py - Distributed Atomic Queue Worker for SEA on CX3 HPC.

Runs on any compute node (GPU 0, 1, 2, or 3). Atomically claims tasks from
queue/pending/, executes the end-to-end SEA pipeline for that date, emits
heartbeats to telemetry/workers/, and marks tasks done in queue/done/.

DOES NOT auto-release cluster nodes: when the queue is empty, enters a polite
STANDBY polling loop while keeping the node/GPU allocation alive and ready.
"""

import os
import sys
import time
import signal
import socket
import argparse
import traceback
from datetime import datetime, timezone

# Add src to path
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from queue_manager import claim_next_task, complete_task, release_task, ensure_queue_dirs
from telemetry import send_heartbeat, mark_idle, ensure_telemetry_dirs
from run_date_pipeline import process_date

active_task_ref = None


def sig_handler(signum, frame):
    global active_task_ref
    print(f"\n⚠️ Worker received signal {signum}. Releasing current task...")
    if active_task_ref:
        release_task(active_task_ref, reason=f"Signal {signum} received")
        print("Task released back to pending/.")
    sys.exit(0)


signal.signal(signal.SIGINT, sig_handler)
signal.signal(signal.SIGTERM, sig_handler)


def main():
    global active_task_ref
    parser = argparse.ArgumentParser(description="SEA Distributed Atomic Queue Worker")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU Device ID on this node (default: 0)")
    parser.add_argument("--worker-id", type=str, default=None, help="Unique worker identifier")
    parser.add_argument("--processes", type=int, default=4, help="CPU helper processes for audio STFT (default: 4)")
    parser.add_argument("--poll-interval", type=int, default=30, help="Seconds to sleep when queue is empty (default: 30)")
    parser.add_argument("--max-tasks", type=int, default=0, help="Max tasks to process before exiting (0 = infinite loop)")
    args = parser.parse_args()

    ensure_queue_dirs()
    ensure_telemetry_dirs()

    hostname = socket.gethostname()
    worker_id = args.worker_id or f"{hostname}_gpu{args.gpu_id}"

    # Set CUDA_VISIBLE_DEVICES if specific GPU requested
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    print("=" * 70)
    print(f"🛰️  SEA DISTRIBUTED WORKER INITIALIZED")
    print(f"   Worker ID    : {worker_id}")
    print(f"   Host Node    : {hostname}")
    print(f"   GPU Index    : {args.gpu_id}")
    print(f"   PID          : {os.getpid()}")
    print(f"   Poll Interval: {args.poll_interval}s")
    print("=" * 70)

    tasks_completed = 0

    while True:
        task = claim_next_task(worker_id)
        active_task_ref = task

        if task:
            loc = task.get("location")
            dt = task.get("date")
            print(f"\n🚀 [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] CLAIMED TASK: Location {loc} | Date {dt}")

            send_heartbeat(
                worker_id=worker_id,
                node=hostname,
                gpu_id=args.gpu_id,
                location=loc,
                date_str=dt,
                current_rec=0,
                total_recs=0,
                win_per_sec=0.0,
                status=f"STARTING {loc} {dt}"
            )

            try:
                t0 = time.time()
                res = process_date(
                    location=loc,
                    date_str=dt,
                    max_files=0,
                    processes=args.processes,
                    worker_id=worker_id,
                    gpu_id=args.gpu_id
                )
                elapsed = time.time() - t0

                if res == 0 or res is None:
                    complete_task(task, results_summary={"walltime_sec": round(elapsed, 2)})
                    print(f"✅ [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] COMPLETED TASK: {loc} {dt} in {elapsed:.1f}s")
                else:
                    print(f"⚠️ process_date returned code {res}. Releasing task...")
                    release_task(task, reason=f"Returned exit code {res}")
            except Exception as e:
                print(f"❌ Error processing task {loc} {dt}: {e}")
                traceback.print_exc()
                release_task(task, reason=str(e))

            active_task_ref = None
            tasks_completed += 1

            if args.max_tasks > 0 and tasks_completed >= args.max_tasks:
                print(f"🏁 Reached max tasks limit ({args.max_tasks}). Exiting.")
                mark_idle(worker_id, node=hostname, gpu_id=args.gpu_id, message="FINISHED")
                break
        else:
            # Standby mode: keeps the PBS node alive without releasing resource!
            mark_idle(worker_id, node=hostname, gpu_id=args.gpu_id, message="STANDBY (IDLE)")
            sys.stdout.write(f"\r💤 [{datetime.now(timezone.utc).strftime('%H:%M:%S')}] Queue empty. Standing by on {hostname} (polling every {args.poll_interval}s)...")
            sys.stdout.flush()
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
