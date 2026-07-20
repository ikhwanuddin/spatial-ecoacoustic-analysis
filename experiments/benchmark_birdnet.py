#!/usr/bin/env python3
"""
BirdNET Inference Benchmark — experiments for Mac Mini M2.

Compares:
  1. FP32 baseline (current default, XNNPACK delegate)
  2. FP16 model (already shipped with birdnetlib, 14MB vs 49MB)
  3. Core ML delegate (convert tflite → mlmodel → Apple Neural Engine)

Usage:
    cd spatial-ecoacoustic-analysis
    ./venv/bin/python experiments/benchmark_birdnet.py

Target data: Q0 2026-04-16 beamforming output (19 WAV files, 120s each).
"""

import os
import sys
import time
import json
import argparse
import threading
from typing import List, Dict, Optional, Tuple

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


# ============================================================
# Experiment 1 & 2: FP32 vs FP16 model swap
# ============================================================

def benchmark_fp32_vs_fp16(
    wav_dir: str,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    workers: int = 4,
    num_runs: int = 1,
) -> Dict:
    """
    Monkey-patch birdnetlib to use FP16 model and compare with FP32 baseline.

    Strategy:
      - Import birdnetlib.analyzer
      - Patch MODEL_PATH -> FP16 model
      - Each Analyzer() created after patching will use the FP16 model
    """
    from birdnetlib.analyzer import Analyzer, MODEL_PATH
    from birdnetlib.main import Recording
    from datetime import datetime

    # --- Find WAV files ---
    wav_files = sorted([
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.lower().endswith(".wav") and not f.startswith("._")
    ])

    if not wav_files:
        print("  ❌ No WAV files found.")
        return {}

    total = len(wav_files)
    print(f"  Files: {total}")

    # Thread-local Analyzer
    _tl = threading.local()

    def get_analyzer(model_path: str):
        """Lazy-init per-thread Analyzer with custom model path."""
        if not hasattr(_tl, "analyzer"):
            _tl.analyzer = Analyzer()
        return _tl.analyzer

    def analyze_one(filepath: str, analyzer: Analyzer) -> Tuple[str, int, float, str]:
        fname = os.path.basename(filepath)
        try:
            rec = Recording(
                analyzer, filepath,
                date=datetime(2026, 4, 16),
                min_conf=0.4,
                overlap=0.0,
                lat=lat,
                lon=lon,
            )
            t0 = time.time()
            rec.analyze()
            elapsed = time.time() - t0
            return fname, len(rec.detections), elapsed, ""
        except Exception as e:
            return fname, 0, 0.0, str(e)

    # --- Determine model paths ---
    fp32_model = MODEL_PATH
    fp16_model = os.path.join(os.path.dirname(MODEL_PATH), "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite")

    results = {}

    for variant, model_path in [("FP32", fp32_model), ("FP16", fp16_model)]:
        if variant == "FP16" and not os.path.exists(fp16_model):
            print(f"  ⚠ FP16 model not found at {fp16_model} — skipping")
            continue

        print(f"\n  ── {variant} ({os.path.basename(model_path)}) ──")

        # Monkey-patch MODEL_PATH
        import birdnetlib.analyzer as ba
        ba.MODEL_PATH = model_path

        # Clear any cached analyzers in thread-local
        if hasattr(_tl, "analyzer"):
            del _tl.analyzer

        for run in range(num_runs):
            if num_runs > 1:
                print(f"\n    Run {run + 1}/{num_runs}:")

            t0 = time.time()
            completed = 0
            errors = 0
            times = []

            # Single-threaded for accurate per-file measurement
            for wav in wav_files:
                analyzer = get_analyzer(model_path)
                fname, dets, elapsed, error = analyze_one(wav, analyzer)
                completed += 1
                if error:
                    errors += 1
                    if errors <= 3:
                        print(f"    [{completed}/{total}] ❌ {fname}: {error}")
                else:
                    times.append(elapsed)

                if completed % 5 == 0 or completed == total:
                    pct = completed * 100 // total
                    avg = sum(times) / len(times) if times else 0
                    print(f"    [{completed}/{total}] {pct}%  avg: {avg:.2f}s")

            total_elapsed = time.time() - t0
            key = f"{variant}_{run + 1}" if num_runs > 1 else variant
            results[key] = {
                "variant": variant,
                "model": os.path.basename(model_path),
                "model_size_mb": round(os.path.getsize(model_path) / (1024 * 1024), 1),
                "total_files": total,
                "total_time_sec": round(total_elapsed, 1),
                "per_file_sec": round(total_elapsed / total, 2),
                "files_per_sec": round(total / total_elapsed, 3),
                "errors": errors,
                "individual_times": [round(t, 3) for t in times] if times else [],
            }

            # Clear for next run
            if hasattr(_tl, "analyzer"):
                del _tl.analyzer

        # Restore FP32 path
        ba.MODEL_PATH = fp32_model

    return results


# ============================================================
# Experiment 3: Parallel scaling (FP16)
# ============================================================

def benchmark_parallel_scaling(
    wav_dir: str,
    model_variant: str = "FP16",
    worker_counts: List[int] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Dict:
    """Test BirdNET throughput scaling with different worker counts."""
    from birdnetlib.analyzer import Analyzer, MODEL_PATH
    from birdnetlib.main import Recording
    from datetime import datetime
    from concurrent.futures import ThreadPoolExecutor, as_completed

    if worker_counts is None:
        worker_counts = [1, 2, 4, 8]

    wav_files = sorted([
        os.path.join(wav_dir, f)
        for f in os.listdir(wav_dir)
        if f.lower().endswith(".wav") and not f.startswith("._")
    ])

    if not wav_files:
        return {}

    total = len(wav_files)

    # Set model path
    import birdnetlib.analyzer as ba
    fp32_model = MODEL_PATH
    fp16_model = os.path.join(os.path.dirname(MODEL_PATH), "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite")

    if model_variant == "FP16":
        ba.MODEL_PATH = fp16_model
        model_path = fp16_model
    else:
        model_path = fp32_model

    _tl = threading.local()

    def get_analyzer():
        if not hasattr(_tl, "analyzer"):
            _tl.analyzer = Analyzer()
        return _tl.analyzer

    def analyze_one(filepath):
        fname = os.path.basename(filepath)
        try:
            a = get_analyzer()
            rec = Recording(a, filepath, date=datetime(2026, 4, 16),
                          min_conf=0.4, overlap=0.0, lat=lat, lon=lon)
            rec.analyze()
            return fname, len(rec.detections), ""
        except Exception as e:
            return fname, 0, str(e)

    results = {}
    print(f"\n  Model: {model_variant} ({os.path.basename(model_path)})")
    print(f"  Files: {total}")

    for workers in worker_counts:
        print(f"\n  ── Workers: {workers} ──")
        t0 = time.time()
        completed = 0
        errors = 0
        files_with_dets = 0
        total_dets = 0

        # Reset thread-local
        if hasattr(_tl, "analyzer"):
            del _tl.analyzer

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_one, w): w for w in wav_files}
            for future in as_completed(futures):
                fname, dets, error = future.result()
                completed += 1
                if error:
                    errors += 1
                else:
                    if dets > 0:
                        files_with_dets += 1
                        total_dets += dets

                if completed % 5 == 0 or completed == total:
                    print(f"    [{completed}/{total}] {completed * 100 // total}%")

        elapsed = time.time() - t0
        results[f"workers_{workers}"] = {
            "workers": workers,
            "total_time_sec": round(elapsed, 1),
            "per_file_sec": round(elapsed / total, 2),
            "files_per_sec": round(total / elapsed, 3),
            "files_with_detections": files_with_dets,
            "total_detections": total_dets,
            "errors": errors,
        }
        print(f"    ⏱  {elapsed:.1f}s  |  {elapsed / total:.2f}s/file  |  {total / elapsed:.2f} files/sec")

    # Restore
    ba.MODEL_PATH = fp32_model
    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="BirdNET benchmark experiments")
    parser.add_argument("--wav-dir", default="/Volumes/WD2TB/sea-data/Q0/2026-04-16/beamforming_LabIR",
                       help="Directory with beamforming WAV files")
    parser.add_argument("--lat", type=float, default=-5.6585004)
    parser.add_argument("--lon", type=float, default=104.4046997)
    parser.add_argument("--runs", type=int, default=1, help="Number of runs for FP32/FP16 comparison")
    parser.add_argument("--fp32-fp16", action="store_true", default=True, help="Run FP32 vs FP16 comparison")
    parser.add_argument("--scaling", action="store_true", help="Run parallel scaling test")
    parser.add_argument("--coreml", action="store_true", help="Run Core ML experiment")

    args = parser.parse_args()

    print("=" * 60)
    print("🔬 BirdNET Inference Benchmark")
    print("=" * 60)
    print(f"  WAV dir: {args.wav_dir}")
    print(f"  GPS:     {args.lat}, {args.lon}")
    print("=" * 60)

    all_results = {}

    # ── FP32 vs FP16 ─────────────────────────────────────────
    if args.fp32_fp16:
        print("\n📊 Experiment 1: FP32 vs FP16 (single-threaded, cold start)")
        r = benchmark_fp32_vs_fp16(
            wav_dir=args.wav_dir,
            lat=args.lat,
            lon=args.lon,
            workers=1,
            num_runs=args.runs,
        )
        all_results.update(r)

    # ── Parallel scaling (FP16) ──────────────────────────────
    if args.scaling:
        print("\n📊 Experiment 2: Parallel Scaling (FP16)")
        r = benchmark_parallel_scaling(
            wav_dir=args.wav_dir,
            model_variant="FP16",
            worker_counts=[1, 2, 4, 8],
            lat=args.lat,
            lon=args.lon,
        )
        all_results.update(r)

    # ── Core ML ──────────────────────────────────────────────
    if args.coreml:
        print("\n📊 Experiment 3: Core ML delegate")
        # This will be implemented in coreml_birdnet.py
        print("  (see experiments/coreml_birdnet.py)")

    # ── Summary ──────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    for key, val in all_results.items():
        variant = val.get("variant", "")
        workers = val.get("workers", "")
        t = val.get("total_time_sec", "?")
        p = val.get("per_file_sec", "?")
        if variant:
            print(f"  {variant:>6s}: {t:>6.1f}s total, {p:>5.2f}s/file  ({val.get('files_per_sec', '?')} f/s)")
        elif workers:
            print(f"  {workers} workers: {t:>6.1f}s total, {p:>5.2f}s/file  ({val.get('files_per_sec', '?')} f/s)")

    # Save results
    out_path = os.path.join(os.path.dirname(__file__), "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
