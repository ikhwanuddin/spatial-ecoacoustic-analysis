#!/usr/bin/env python3
"""
Spatial Ecoacoustic Analysis Pipeline — Main Entry Point.

Orchestrates:
  1. Beamforming (LabIR / SPIR1 / SPIR2) on FLAC recordings
  2. Signal Averaging (6-ch → 1-ch direct sum)
  3. BirdNET analysis on outputs
  4. Confidence comparison → processed.json
  5. Cleanup: keep only best-variant beamforming files

Selector-based: manually choose RPiID, date, and file(s) to process.

Usage (quick prototype):
    python run_pipeline.py

The SELECTOR section at the top of __main__ controls which files
are processed — edit it for your prototype runs.

When the pipeline is mature, a CLI/batch mode can be added.
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import List, Optional

from config import (
    MONITORING_DATA,
    ANALYSIS_OUTPUT,
    IR_BASE_PATH,
    LOCATION_MAP,
    RPIID_TO_LOCATION,
    IR_TYPES,
    PROTOTYPE_IR_SUBSETS,
)
from beamforming import Beamformer
from signal_averaging import SignalAverager
from birdnet_processor import process_directory_pipeline


# ============================================================
# HELPERS
# ============================================================

def get_flac_files(rpiid: str, date_str: str, max_files: int = 1) -> List[str]:
    """
    Return absolute paths to .flac files for a given RPiID and date.

    Args:
        rpiid:      Full RPiID folder name, e.g. "RPiID-0000000091668b26"
        date_str:   Date folder, e.g. "2026-04-16"
        max_files:  Max number of files to return (default 1 for prototype)

    Returns:
        List of absolute paths to .flac files (sorted)
    """
    date_dir = os.path.join(MONITORING_DATA, rpiid, date_str)
    if not os.path.isdir(date_dir):
        print(f"❌ Directory not found: {date_dir}")
        return []

    flacs = sorted([
        os.path.join(date_dir, f)
        for f in os.listdir(date_dir)
        if f.lower().endswith(".flac")
    ])

    if max_files and len(flacs) > max_files:
        flacs = flacs[:max_files]

    print(f"📁 {len(flacs)} FLAC file(s) selected from {date_dir}")
    for f in flacs:
        print(f"    → {os.path.basename(f)}")
    return flacs


def build_output_path(
    location_name: str,
    date_str: str,
    processing_type: str,
) -> str:
    """
    Build structured output path:
      {ANALYSIS_OUTPUT}/{location_name}/{date_str}/{processing_type}/

    processing_type examples: "beamforming_LabIR", "signal_averaging"
    """
    return os.path.join(ANALYSIS_OUTPUT, location_name, date_str, processing_type)


def get_location_name_from_rpiid(rpiid: str) -> str:
    """Map full RPiID folder name → short location name."""
    if rpiid in RPIID_TO_LOCATION:
        return RPIID_TO_LOCATION[rpiid]
    return rpiid  # fallback


def parse_flac_date(flac_path: str, folder_date_str: str) -> datetime:
    """
    Try to extract a date from the FLAC context. Falls back to folder date.

    The folder path is like .../RPiID-xxx/2026-04-16/
    We use the folder date as the authoritative date.
    """
    try:
        return datetime.strptime(folder_date_str, "%Y-%m-%d")
    except ValueError:
        return datetime.now()


# ============================================================
# SINGLE-FILE PIPELINE
# ============================================================

def process_one_flac(
    flac_path: str,
    location_name: str,
    date_str: str,
    ir_types: List[str],
    run_sa: bool = True,
    run_birdnet: bool = True,
    cleanup: bool = False,
    dry_run: bool = False,
    use_prototype_subsets: bool = False,
):
    """
    Run the full pipeline for ONE FLAC file.

    Order:
      1. Beamforming for each IR type → WAVs
      2. Signal averaging → WAV
      3. BirdNET on beamforming dirs + SA dir
      4. Confidence comparison → processed.json
      5. Cleanup (optional, disabled by default for prototyping)
    """
    base_name = os.path.splitext(os.path.basename(flac_path))[0]
    print(f"\n{'='*60}")
    print(f"🎙  Processing: {base_name}")
    print(f"📍 Location:  {location_name}")
    print(f"📅 Date:      {date_str}")
    print(f"📡 IR Types:  {', '.join(ir_types)}")
    print(f"{'='*60}")

    overall_start = time.time()
    bf_dirs = []
    sa_dir = ""

    # Choose IR config
    ir_configs = PROTOTYPE_IR_SUBSETS if use_prototype_subsets else IR_TYPES

    # ── Step 1: Beamforming ──────────────────────────────────
    for ir_name in ir_types:
        if ir_name not in ir_configs:
            print(f"⚠  Unknown IR type: {ir_name} — skipping")
            continue

        ir_type = ir_configs[ir_name]
        bf_dir = build_output_path(location_name, date_str, f"beamforming_{ir_name}")
        bf_dirs.append(bf_dir)

        print(f"\n── Beamforming [{ir_name}] ──")
        bf = Beamformer(
            flac_path=flac_path,
            output_dir=bf_dir,
            ir_type=ir_type,
            ir_base_path=IR_BASE_PATH,
        )
        bf.run()

    # ── Step 2: Signal Averaging ─────────────────────────────
    if run_sa:
        sa_dir = build_output_path(location_name, date_str, "signal_averaging")
        print(f"\n── Signal Averaging ──")
        sa = SignalAverager(flac_path=flac_path, output_dir=sa_dir)
        sa.run()

    # ── Step 3-5: BirdNET → processed.json → cleanup ────────
    if run_birdnet:
        recording_date = parse_flac_date(flac_path, date_str)

        # Process each beamforming directory
        for bf_dir in bf_dirs:
            # Extract IR type name from path for identifier pattern
            ir_label = os.path.basename(bf_dir).replace("beamforming_", "")
            pattern = f"_{ir_label}("  # e.g. "_LabIR(" in filenames

            process_directory_pipeline(
                directory=bf_dir,
                location="waycanguk",
                date=recording_date,
                identifier_pattern=pattern,
                cleanup=cleanup,
                dry_run=dry_run,
            )

        # Process signal averaging directory
        if run_sa and sa_dir:
            process_directory_pipeline(
                directory=sa_dir,
                location="waycanguk",
                date=recording_date,
                identifier_pattern="",   # SA uses all detections
                cleanup=False,           # only 1 file, no cleanup needed
                dry_run=dry_run,
            )

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✅ Done — {base_name} processed in {elapsed:.1f}s")
    print(f"{'='*60}")

    return {
        "flac": flac_path,
        "beamforming_dirs": bf_dirs,
        "sa_dir": sa_dir,
        "elapsed": elapsed,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Spatial Ecoacoustic Analysis Pipeline"
    )
    parser.add_argument(
        "--rpiid",
        type=str,
        default=None,
        help="Full RPiID name (default: first available with data)",
    )
    parser.add_argument(
        "--location",
        type=str,
        default=None,
        help="Short location code, e.g. '2A400' (overrides --rpiid)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Date string YYYY-MM-DD (default: earliest available)",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=1,
        help="Max FLAC files to process (default 1 for prototype)",
    )
    parser.add_argument(
        "--ir-types",
        type=str,
        default="LabIR",
        help="Comma-separated IR types: LabIR,SPIR1,SPIR2",
    )
    parser.add_argument(
        "--no-sa",
        action="store_true",
        help="Skip signal averaging",
    )
    parser.add_argument(
        "--no-birdnet",
        action="store_true",
        help="Skip BirdNET analysis",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Delete low-confidence beamforming files after processing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use full IR parameter sets (not prototype subsets)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available RPiIDs and dates, then exit",
    )

    args = parser.parse_args()

    # ── Resolve RPiID ────────────────────────────────────────
    if args.location:
        loc_code = args.location
        if loc_code not in LOCATION_MAP:
            print(f"❌ Unknown location code: {loc_code}")
            print(f"   Known: {list(LOCATION_MAP.keys())}")
            sys.exit(1)
        rpiid = LOCATION_MAP[loc_code]
        location_name = loc_code
    elif args.rpiid:
        rpiid = args.rpiid
        location_name = get_location_name_from_rpiid(rpiid)
    else:
        # Auto-detect: pick first RPiID with data
        available = [
            d for d in os.listdir(MONITORING_DATA)
            if d.startswith("RPiID-") and os.path.isdir(os.path.join(MONITORING_DATA, d))
        ]
        if not available:
            print(f"❌ No RPiID folders found in {MONITORING_DATA}")
            sys.exit(1)
        rpiid = available[0]
        location_name = get_location_name_from_rpiid(rpiid)
        print(f"🔍 Auto-selected RPiID: {rpiid} ({location_name})")

    # ── List mode ────────────────────────────────────────────
    if args.list:
        print(f"\n📂 Monitoring data at: {MONITORING_DATA}\n")
        for rp in sorted(os.listdir(MONITORING_DATA)):
            rp_path = os.path.join(MONITORING_DATA, rp)
            if not os.path.isdir(rp_path) or not rp.startswith("RPiID-"):
                continue
            loc = get_location_name_from_rpiid(rp)
            dates = sorted([
                d for d in os.listdir(rp_path)
                if os.path.isdir(os.path.join(rp_path, d))
                and d != "logs"
            ])
            print(f"  {loc:>6s}  {rp}  ({len(dates)} dates: {dates[0]} → {dates[-1]})")

        print(f"\n📡 Available IR types: {list(IR_TYPES.keys())}")
        print(f"   (prototype subsets use reduced param sets)")
        sys.exit(0)

    # ── Resolve date ─────────────────────────────────────────
    rpiid_dir = os.path.join(MONITORING_DATA, rpiid)
    if not os.path.isdir(rpiid_dir):
        print(f"❌ RPiID directory not found: {rpiid_dir}")
        sys.exit(1)

    available_dates = sorted([
        d for d in os.listdir(rpiid_dir)
        if os.path.isdir(os.path.join(rpiid_dir, d))
    ])

    if args.date:
        date_str = args.date
    else:
        date_str = available_dates[0]
        print(f"📅 Auto-selected date: {date_str}")

    if date_str not in available_dates:
        print(f"❌ Date {date_str} not found. Available: {available_dates}")
        sys.exit(1)

    # ── Parse IR types ───────────────────────────────────────
    ir_types = [t.strip() for t in args.ir_types.split(",")]
    for t in ir_types:
        if t not in IR_TYPES:
            print(f"❌ Unknown IR type: {t}")
            print(f"   Known: {list(IR_TYPES.keys())}")
            sys.exit(1)

    use_prototype = not args.full
    if use_prototype:
        print(f"🧪 Prototype mode: reduced IR parameter sets")

    # ── Get FLAC files ───────────────────────────────────────
    flac_paths = get_flac_files(rpiid, date_str, max_files=args.max_files)
    if not flac_paths:
        print("❌ No FLAC files found.")
        sys.exit(1)

    # ── Validate output volume ───────────────────────────────
    if not os.path.exists(ANALYSIS_OUTPUT):
        print(f"❌ Output volume not mounted: {ANALYSIS_OUTPUT}")
        print("   Please connect your external HDD.")
        sys.exit(1)

    # ── Run pipeline ─────────────────────────────────────────
    results = []
    for flac_path in flac_paths:
        result = process_one_flac(
            flac_path=flac_path,
            location_name=location_name,
            date_str=date_str,
            ir_types=ir_types,
            run_sa=not args.no_sa,
            run_birdnet=not args.no_birdnet,
            cleanup=args.cleanup,
            dry_run=args.dry_run,
            use_prototype_subsets=use_prototype,
        )
        results.append(result)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"🏁 Pipeline Complete")
    print(f"{'='*60}")
    for r in results:
        bname = os.path.splitext(os.path.basename(r["flac"]))[0]
        print(f"  {bname}: {r['elapsed']:.1f}s")
        for d in r["beamforming_dirs"]:
            print(f"    BF: {d}")
        if r["sa_dir"]:
            print(f"    SA: {r['sa_dir']}")


if __name__ == "__main__":
    main()
