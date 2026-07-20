#!/usr/bin/env python3
"""
Profiling script for spatial-ecoacoustic-analysis pipeline.
Measures per-stage timing on a single FLAC file.

Usage:
    python profile_pipeline.py

Set TARGET_RPIID, TARGET_DATE below.
"""

import os
import sys
import time
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (
    MONITORING_DATA,
    ANALYSIS_OUTPUT,
    IR_TYPES,
    PROTOTYPE_IR_SUBSETS,
    LOCATION_MAP,
    RPIID_TO_LOCATION,
    SITE_COORDS,
    IRType,
)
from ircache import IRCache
from beamforming import Beamformer
from signal_averaging import SignalAverager

# ============================================================
# CONFIG — edit these or pass via CLI
# ============================================================

TARGET_RPIID = "RPiID-000000005acf5969"   # Q0
TARGET_DATE  = "2026-04-16"
TARGET_IR    = "LabIR"                     # single IR type for profiling
MAX_FILES    = 1                           # profile 1 file

# ============================================================
# PROFILING HELPERS
# ============================================================

def get_flac_files(rpiid, date_str, max_files=1):
    date_dir = os.path.join(MONITORING_DATA, rpiid, date_str)
    if not os.path.isdir(date_dir):
        print(f"ERROR: {date_dir} not found")
        return []
    flacs = sorted([
        os.path.join(date_dir, f)
        for f in os.listdir(date_dir)
        if f.lower().endswith(".flac")
    ])
    if max_files and len(flacs) > max_files:
        flacs = flacs[:max_files]
    return flacs


def build_output_path(output_base, location_name, date_str, processing_type):
    return os.path.join(output_base, location_name, date_str, processing_type)


def profile_beamforming(flac_path, ir_type, output_dir):
    """Profile beamforming for ONE IR type. Returns timing dict."""
    print(f"\n{'─'*50}")
    print(f"🎯 BEAMFORMING [{ir_type.name}]")
    print(f"   Combos: {len(ir_type.param_values)} params × {len(ir_type.degree_values)} degrees"
          + (f" × {len(ir_type.rep_values)} reps" if ir_type.rep_values else ""))

    t0 = time.time()
    bf = Beamformer(flac_path=flac_path, output_dir=output_dir, ir_type_or_name=ir_type)
    t_init = time.time() - t0

    print(f"\n   ⏱  Init (load+filter+STFT): {t_init:.1f}s")

    # Run all combinations
    t_run_start = time.time()
    bf.run()
    t_run = time.time() - t_run_start

    # Count outputs
    n_outputs = 0
    total_size = 0
    for f in os.listdir(output_dir):
        if f.endswith(".wav") and not f.startswith("._"):
            n_outputs += 1
            total_size += os.path.getsize(os.path.join(output_dir, f))

    total_elapsed = time.time() - t0

    result = {
        "ir_type": ir_type.name,
        "init_sec": round(t_init, 1),
        "run_sec": round(t_run, 1),
        "total_sec": round(total_elapsed, 1),
        "n_outputs": n_outputs,
        "output_size_mb": round(total_size / (1024 * 1024), 2),
        "per_direction_sec": round(t_run / n_outputs, 3) if n_outputs else 0,
    }

    print(f"\n   📊 Results:")
    print(f"      Init:         {result['init_sec']:>8.1f}s")
    print(f"      Run (all dir):{result['run_sec']:>8.1f}s")
    print(f"      Total:         {result['total_sec']:>8.1f}s")
    print(f"      Outputs:       {result['n_outputs']:>8d} files")
    print(f"      Size:          {result['output_size_mb']:>8.2f} MB")
    print(f"      Per direction: {result['per_direction_sec']:>8.3f}s")

    # Additional: breakdown of a single direction vs total
    # (we already measured all; the per_direction gives average)
    # Check I/O write speed
    io_speed = result['output_size_mb'] / t_run if t_run > 0 else 0
    print(f"      I/O write rate:{io_speed:>8.2f} MB/s")

    return result


def profile_signal_averaging(flac_path, output_dir):
    """Profile signal averaging."""
    print(f"\n{'─'*50}")
    print(f"🎯 SIGNAL AVERAGING")

    t0 = time.time()
    sa = SignalAverager(flac_path=flac_path, output_dir=output_dir)
    t_init = time.time() - t0

    print(f"\n   ⏱  Init (load+filter): {t_init:.1f}s")

    t_run = time.time()
    sa.run()
    t_run = time.time() - t_run

    total_elapsed = time.time() - t0

    result = {
        "init_sec": round(t_init, 1),
        "run_sec": round(t_run, 1),
        "total_sec": round(total_elapsed, 1),
    }

    print(f"\n   📊 Results:")
    print(f"      Init:  {result['init_sec']:>8.1f}s")
    print(f"      Run:   {result['run_sec']:>8.1f}s")
    print(f"      Total: {result['total_sec']:>8.1f}s")

    return result


def profile_birdnet(output_dir, lat=None, lon=None):
    """Profile BirdNET on a directory of WAVs. Import inline to avoid loading TF."""
    import json
    from concurrent.futures import ThreadPoolExecutor, as_completed
    import threading
    from datetime import datetime

    from birdnetlib.analyzer import Analyzer
    from birdnetlib.main import Recording

    print(f"\n{'─'*50}")
    print(f"🎯 BIRDNET ANALYSIS")

    wav_files = sorted([
        os.path.join(output_dir, f)
        for f in os.listdir(output_dir)
        if f.lower().endswith(".wav") and not f.startswith("._")
    ])
    total = len(wav_files)
    print(f"   Files to analyze: {total}")

    # Thread-local analyzer
    _tl = threading.local()

    def get_analyzer():
        if not hasattr(_tl, "analyzer"):
            _tl.analyzer = Analyzer()
        return _tl.analyzer

    def analyze_one(wav_path):
        fname = os.path.basename(wav_path)
        try:
            rec = Recording(
                get_analyzer(), wav_path,
                date=datetime.now(), min_conf=0.4, overlap=0.0,
            )
            if lat is not None and lon is not None:
                rec.lat = lat
                rec.lon = lon
            rec.analyze()
            return fname, rec.detections, ""
        except Exception as e:
            return fname, [], str(e)

    # Test with different worker counts
    for workers in [1, 2, 4, 8]:
        print(f"\n   ── Workers: {workers} ──")
        t0 = time.time()

        results = {}
        errors = 0
        completed = 0

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(analyze_one, w): w for w in wav_files}
            for future in as_completed(futures):
                fname, detections, error = future.result()
                completed += 1
                if error:
                    errors += 1
                else:
                    results[fname] = detections
                if completed % 10 == 0 or completed == total:
                    print(f"      [{completed}/{total}] {completed*100//total}%")

        elapsed = time.time() - t0
        files_with_dets = sum(1 for v in results.values() if v)
        total_dets = sum(len(v) for v in results.values())

        print(f"      ⏱  {elapsed:.1f}s  |  "
              f"{files_with_dets} files w/ dets, {total_dets} total dets, {errors} errors")
        print(f"      → {elapsed/total:.2f}s per file, {total/elapsed:.2f} files/sec")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Pipeline profiler")
    parser.add_argument("--rpiid", default=TARGET_RPIID)
    parser.add_argument("--date", default=TARGET_DATE)
    parser.add_argument("--ir", default=TARGET_IR, choices=["LabIR", "SPIR1", "SPIR2"])
    parser.add_argument("--max-files", type=int, default=MAX_FILES)
    parser.add_argument("--full", action="store_true", help="Use full IR sets")
    parser.add_argument("--output-base", default=ANALYSIS_OUTPUT,
                        help=f"Override output base path (default: {ANALYSIS_OUTPUT})")
    parser.add_argument("--birdnet", action="store_true", help="Also profile BirdNET (auto-continues)")
    parser.add_argument("--skip-bf", action="store_true", help="Skip beamforming")
    parser.add_argument("--skip-sa", action="store_true", help="Skip signal averaging")
    parser.add_argument("--no-cleanup", action="store_true", help="Don't clean up previous output")

    args = parser.parse_args()

    if args.rpiid in LOCATION_MAP:
        location_name = LOCATION_MAP[args.rpiid]
    elif args.rpiid in RPIID_TO_LOCATION:
        location_name = RPIID_TO_LOCATION[args.rpiid]
    else:
        location_name = args.rpiid

    flacs = get_flac_files(args.rpiid, args.date, max_files=args.max_files)
    if not flacs:
        sys.exit(1)

    flac_path = flacs[0]
    base_name = os.path.splitext(os.path.basename(flac_path))[0]

    print("=" * 60)
    print("🔬 PIPELINE PROFILER")
    print("=" * 60)
    print(f"   File:     {base_name}")
    print(f"   Location: {location_name}")
    print(f"   Date:     {args.date}")
    print(f"   IR:       {args.ir} ({'full' if args.full else 'prototype'})")
    print("=" * 60)

    ir_config = IR_TYPES if args.full else PROTOTYPE_IR_SUBSETS
    ir_type = ir_config[args.ir]

    # ── Clean up previous output (unless --no-cleanup) ────────
    if not args.no_cleanup and not args.skip_bf:
        bf_dir = build_output_path(args.output_base, location_name, args.date, f"beamforming_{args.ir}")
        if os.path.isdir(bf_dir):
            import shutil
            print(f"🧹 Cleaning previous beamforming output: {bf_dir}")
            shutil.rmtree(bf_dir)
    if not args.no_cleanup and not args.skip_sa:
        sa_dir = build_output_path(args.output_base, location_name, args.date, "signal_averaging")
        if os.path.isdir(sa_dir):
            import shutil
            print(f"🧹 Cleaning previous SA output: {sa_dir}")
            shutil.rmtree(sa_dir)

    # ── Beamforming profile ──────────────────────────────────
    if not args.skip_bf:
        bf_dir = build_output_path(args.output_base, location_name, args.date, f"beamforming_{args.ir}")
        bf_result = profile_beamforming(flac_path, ir_type, bf_dir)

    # ── Signal Averaging profile ─────────────────────────────
    if not args.skip_sa:
        sa_dir = build_output_path(args.output_base, location_name, args.date, "signal_averaging")
        sa_result = profile_signal_averaging(flac_path, sa_dir)

    # ── BirdNET profile ──────────────────────────────────────
    if args.birdnet:
        lat, lon = None, None
        if location_name in SITE_COORDS:
            lat = SITE_COORDS[location_name]["lat"]
            lon = SITE_COORDS[location_name]["lon"]

        print(f"\n⚠  BirdNET will contend with any running BirdNET process.")
        print(f"   Continuing in 5s... Ctrl+C to abort.")
        try:
            import time as _t
            _t.sleep(5)
        except KeyboardInterrupt:
            print("\n   Aborted.")
            sys.exit(0)

        bf_dir = build_output_path(args.output_base, location_name, args.date, f"beamforming_{args.ir}")
        profile_birdnet(bf_dir, lat=lat, lon=lon)

    print(f"\n{'='*60}")
    print(f"✅ Profiling complete")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
