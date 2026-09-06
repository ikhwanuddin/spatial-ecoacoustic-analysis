"""
Centralized Telemetry Reporter for Spatial Ecoacoustic Analysis (SEA).

Lightweight worker heartbeat reporting. Each worker writes its current processing
state to telemetry/workers/<worker_id>.json via atomic replace.
Provides centralized visibility across all distributed cluster nodes.
"""

import os
import sys
import json
import socket
import subprocess
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List

try:
    from config import TELEMETRY_DIR
    if not os.path.exists(os.path.dirname(TELEMETRY_DIR)):
        TELEMETRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "telemetry")
except ImportError:
    TELEMETRY_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "telemetry")

WORKERS_DIR = os.path.join(TELEMETRY_DIR, "workers")


def ensure_telemetry_dirs():
    os.makedirs(WORKERS_DIR, exist_ok=True)


def get_gpu_metrics(gpu_id: int = 0) -> Dict[str, Any]:
    """Queries nvidia-smi for temperature, power, and VRAM if present."""
    metrics = {"gpu_temp_c": None, "vram_used_mb": None, "vram_total_mb": None, "gpu_util_pct": None}
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            f"--id={gpu_id}",
            "--query-gpu=temperature.gpu,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL, timeout=2).decode().strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 4:
            metrics["gpu_temp_c"] = int(parts[0])
            metrics["vram_used_mb"] = int(parts[1])
            metrics["vram_total_mb"] = int(parts[2])
            metrics["gpu_util_pct"] = int(parts[3])
    except Exception:
        pass
    return metrics


def send_heartbeat(
    worker_id: str,
    node: Optional[str] = None,
    gpu_id: int = 0,
    location: str = "-",
    date_str: str = "-",
    current_rec: int = 0,
    total_recs: int = 0,
    win_per_sec: float = 0.0,
    status: str = "ACTIVE",
    extra: Optional[Dict] = None
) -> None:
    """Writes an atomic heartbeat update for this worker."""
    ensure_telemetry_dirs()
    if not node:
        node = socket.gethostname()

    gpu_metrics = get_gpu_metrics(gpu_id)
    pct = round((current_rec / total_recs * 100), 1) if total_recs > 0 else 0.0

    payload = {
        "worker_id": worker_id,
        "node": node,
        "pid": os.getpid(),
        "gpu_id": gpu_id,
        "location": location,
        "date": date_str,
        "current_rec": current_rec,
        "total_recs": total_recs,
        "pct": pct,
        "win_per_sec": round(win_per_sec, 1),
        "status": status,
        "last_heartbeat": datetime.now(timezone.utc).isoformat(),
        "last_heartbeat_epoch": datetime.now(timezone.utc).timestamp(),
        "gpu": gpu_metrics,
        "extra": extra or {},
    }

    target_path = os.path.join(WORKERS_DIR, f"{worker_id}.json")
    tmp_path = target_path + f".tmp.{os.getpid()}"

    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, target_path)


def mark_idle(worker_id: str, node: Optional[str] = None, gpu_id: int = 0, message: str = "STANDBY") -> None:
    """Marks the worker as idle/standby."""
    send_heartbeat(
        worker_id=worker_id,
        node=node,
        gpu_id=gpu_id,
        location="-",
        date_str="-",
        current_rec=0,
        total_recs=0,
        win_per_sec=0.0,
        status=message
    )


def get_all_telemetry(stale_threshold_sec: float = 180.0) -> List[Dict[str, Any]]:
    """Reads all worker telemetry files and computes liveness."""
    ensure_telemetry_dirs()
    workers = []
    now = datetime.now(timezone.utc).timestamp()

    for fname in sorted(os.listdir(WORKERS_DIR)):
        if not fname.endswith(".json") or fname.startswith("."):
            continue
        fpath = os.path.join(WORKERS_DIR, fname)
        try:
            with open(fpath, "r") as f:
                data = json.load(f)
            epoch = data.get("last_heartbeat_epoch", 0)
            age_sec = now - epoch
            if age_sec > stale_threshold_sec:
                data["liveness"] = "STALE"
            else:
                data["liveness"] = "ALIVE"
            data["age_sec"] = round(age_sec, 1)
            workers.append(data)
        except Exception:
            continue

    return workers


if __name__ == "__main__":
    ensure_telemetry_dirs()
    print(f"Telemetry directory: {WORKERS_DIR}")
    active = get_all_telemetry()
    print(f"Registered workers: {len(active)}")
