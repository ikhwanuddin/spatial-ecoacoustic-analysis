"""
Atomic Task Queue Manager for Spatial Ecoacoustic Analysis (SEA).

Provides thread-safe and multi-node safe task claiming using POSIX atomic renames
(os.replace) on shared filesystems (RDS/NFS). Eliminates race conditions and
overlapping executions across distributed GPU workers.
"""

import os
import sys
import json
import socket
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

try:
    from config import QUEUE_DIR
    # Test if parent directory of QUEUE_DIR exists/writable, else fallback
    if not os.path.exists(os.path.dirname(QUEUE_DIR)):
        QUEUE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queue")
except ImportError:
    QUEUE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "queue")

PENDING_DIR = os.path.join(QUEUE_DIR, "pending")
ACTIVE_DIR = os.path.join(QUEUE_DIR, "active")
DONE_DIR = os.path.join(QUEUE_DIR, "done")


def ensure_queue_dirs():
    """Initializes queue directory structure."""
    os.makedirs(PENDING_DIR, exist_ok=True)
    os.makedirs(ACTIVE_DIR, exist_ok=True)
    os.makedirs(DONE_DIR, exist_ok=True)


def get_task_filename(location: str, date_str: str) -> str:
    return f"{location}_{date_str}.json"


def enqueue_date(location: str, date_str: str, priority: int = 10, metadata: Optional[Dict] = None) -> bool:
    """Enqueues a single date task into pending/ if not already present."""
    ensure_queue_dirs()
    fname = get_task_filename(location, date_str)
    pending_path = os.path.join(PENDING_DIR, fname)
    done_path = os.path.join(DONE_DIR, fname)

    # Do not re-enqueue if already in pending or done
    if os.path.exists(pending_path) or os.path.exists(done_path):
        return False

    task_payload = {
        "location": location,
        "date": date_str,
        "priority": priority,
        "enqueued_at": datetime.now(timezone.utc).isoformat(),
        "metadata": metadata or {},
    }

    # Write atomically via temp file
    tmp_path = pending_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(task_payload, f, indent=2)
    os.replace(tmp_path, pending_path)
    return True


def claim_next_task(worker_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """
    Atomically claims the next pending task using POSIX os.replace().
    Safe across multiple nodes and concurrent workers.
    """
    ensure_queue_dirs()
    if not worker_id:
        worker_id = f"{socket.gethostname()}_{os.getpid()}"

    pending_files = sorted([
        f for f in os.listdir(PENDING_DIR)
        if f.endswith(".json") and not f.startswith(".")
    ])

    for fname in pending_files:
        pending_path = os.path.join(PENDING_DIR, fname)
        active_fname = f"{worker_id}_{fname}"
        active_path = os.path.join(ACTIVE_DIR, active_fname)

        try:
            # Atomic rename from pending/ to active/
            os.replace(pending_path, active_path)
        except (FileNotFoundError, OSError):
            # Another worker claimed it concurrently; try next candidate
            continue

        # Successfully claimed: annotate with worker metadata
        try:
            with open(active_path, "r") as f:
                task = json.load(f)
        except Exception:
            task = {}

        task["worker_id"] = worker_id
        task["hostname"] = socket.gethostname()
        task["pid"] = os.getpid()
        task["claimed_at"] = datetime.now(timezone.utc).isoformat()
        task["_active_path"] = active_path
        task["_fname"] = fname

        with open(active_path, "w") as f:
            json.dump(task, f, indent=2)

        return task

    return None


def complete_task(task: Dict[str, Any], results_summary: Optional[Dict] = None) -> bool:
    """Marks an active task as complete and moves to done/."""
    active_path = task.get("_active_path")
    fname = task.get("_fname")
    if not active_path or not os.path.exists(active_path):
        return False

    done_path = os.path.join(DONE_DIR, fname)
    task["completed_at"] = datetime.now(timezone.utc).isoformat()
    if results_summary:
        task["results_summary"] = results_summary

    tmp_path = done_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(task, f, indent=2)

    os.replace(tmp_path, done_path)
    if os.path.exists(active_path):
        try:
            os.remove(active_path)
        except OSError:
            pass
    return True


def release_task(task: Dict[str, Any], reason: str = "aborted") -> bool:
    """Releases an active task back into pending/ upon failure or interruption."""
    active_path = task.get("_active_path")
    fname = task.get("_fname")
    if not active_path or not os.path.exists(active_path):
        return False

    pending_path = os.path.join(PENDING_DIR, fname)
    task["released_at"] = datetime.now(timezone.utc).isoformat()
    task["release_reason"] = reason

    tmp_path = pending_path + f".tmp.{os.getpid()}"
    with open(tmp_path, "w") as f:
        json.dump(task, f, indent=2)

    os.replace(tmp_path, pending_path)
    if os.path.exists(active_path):
        try:
            os.remove(active_path)
        except OSError:
            pass
    return True


def get_queue_stats() -> Dict[str, Any]:
    """Returns the current state and counts of all queue buckets."""
    ensure_queue_dirs()
    pending = [f for f in os.listdir(PENDING_DIR) if f.endswith(".json")]
    active = [f for f in os.listdir(ACTIVE_DIR) if f.endswith(".json")]
    done = [f for f in os.listdir(DONE_DIR) if f.endswith(".json")]

    return {
        "pending_count": len(pending),
        "active_count": len(active),
        "done_count": len(done),
        "total_count": len(pending) + len(active) + len(done),
        "pending": sorted(pending),
        "active": sorted(active),
        "done": sorted(done),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SEA Atomic Task Queue Manager")
    parser.add_argument("--status", action="store_true", help="Print queue status")
    parser.add_argument("--enqueue-loc", type=str, help="Location code (e.g. 2A400)")
    parser.add_argument("--enqueue-date", type=str, help="Date YYYY-MM-DD")
    args = parser.parse_args()

    if args.enqueue_loc and args.enqueue_date:
        res = enqueue_date(args.enqueue_loc, args.enqueue_date)
        print(f"Enqueued {args.enqueue_loc} {args.enqueue_date}: {'OK' if res else 'ALREADY_EXISTS'}")

    stats = get_queue_stats()
    print(f"Queue Status: {stats['pending_count']} Pending | {stats['active_count']} Active | {stats['done_count']} Done")
