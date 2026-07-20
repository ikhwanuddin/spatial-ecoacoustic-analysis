#!/usr/bin/env python3
"""
Spatial Ecoacoustic Analysis Pipeline — Main Entry Point.

Orchestrates:
  1. Beamforming (LabIR / SPIR1 / SPIR2) on FLAC recordings
  2. Signal Averaging (6-ch → 1-ch direct sum)
  3. BirdNET analysis on outputs (with site-specific GPS)
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
    IR_TYPES,
    PROTOTYPE_IR_SUBSETS,
    LOCATION_MAP,
    RPIID_TO_LOCATION,
    SITE_COORDS,
)
from ircache import IRCache
from beamforming import Beamformer
from signal_averaging import SignalAverager
from birdnet_processor import process_directory_pipeline


# ============================================================
# HELPERS
# ============================================================

def get_flac_files(rpiid: str, date_str: str, max_files: int = 1) -> List[str]:
    """Return absolute paths to .flac files for a given RPiID and date."""
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


def build_output_path(location_name: str, date_str: str, processing_type: str) -> str:
    """Build structured output path under sea-data/."""
    return os.path.join(ANALYSIS_OUTPUT, location_name, date_str, processing_type)


def get_location_name_from_rpiid(rpiid: str) -> str:
    """Map full RPiID folder name → short location name."""
    if rpiid in RPIID_TO_LOCATION:
        return RPIID_TO_LOCATION[rpiid]
    return rpiid


def parse_flac_date(flac_path: str, folder_date_str: str) -> datetime:
    """Extract date from folder context; fallback to now."""
    try:
        return datetime.strptime(folder_date_str, "%Y-%m-%d")
    except ValueError:
        return datetime.now()


def get_site_coords(location_name: str):
    """Return (lat, lon) for a given location name, or (None, None)."""
    if location_name in SITE_COORDS:
        c = SITE_COORDS[location_name]
        return c["lat"], c["lon"]
    return None, None


def _beamforming_complete(output_dir: str, base_name: str, ir_type):
    if not os.path.isdir(output_dir):
        return False
    zenith = ir_type.zenith_speakers or set()
    for param in ir_type.param_values:
        degrees = [0] if param in zenith else ir_type.degree_values
        for deg in degrees:
            reps = ir_type.rep_values or [None]
            for rep in reps:
                fmt_kwargs = {}
                if ir_type.param_label == "speaker":
                    fmt_kwargs["speaker"] = param
                else:
                    fmt_kwargs["distance"] = param
                fmt_kwargs["degrees"] = deg
                if rep is not None:
                    fmt_kwargs["rep"] = rep
                suffix = ir_type.output_suffix_pattern.format(**fmt_kwargs)
                fname = base_name + "_" + suffix + ".wav"
                if not os.path.isfile(os.path.join(output_dir, fname)):
                    return False
    return True


def _sa_complete(output_dir: str, base_name: str):
    return os.path.isfile(os.path.join(output_dir, base_name + "_sa.wav"))


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
    force_bf: bool = False,
):
    """Run the full pipeline for ONE FLAC file."""
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
        if not force_bf and _beamforming_complete(bf_dir, base_name, ir_type):
            print(f"  ✓ {ir_name} outputs already exist — skipping beamforming")
        else:
            bf = Beamformer(
                flac_path=flac_path,
                output_dir=bf_dir,
                ir_type_or_name=ir_type,
            )
            bf.run()

    # ── Step 2: Signal Averaging ─────────────────────────────
    if run_sa:
        sa_dir = build_output_path(location_name, date_str, "signal_averaging")
        print(f"\n── Signal Averaging ──")
        if not force_bf and _sa_complete(sa_dir, base_name):
            print(f"  ✓ SA output already exists — skipping")
        else:
            sa = SignalAverager(flac_path=flac_path, output_dir=sa_dir)
            sa.run()

    # ── Step 3-5: BirdNET → processed.json → cleanup ────────
    if run_birdnet:
        recording_date = parse_flac_date(flac_path, date_str)
        lat, lon = get_site_coords(location_name)
        if lat is not None:
            print(f"\n  🌍 Site coordinates: {lat}, {lon}")

        # Process each beamforming directory
        for bf_dir in bf_dirs:
            ir_label = os.path.basename(bf_dir).replace("beamforming_", "")
            pattern = f"_{ir_label}("

            process_directory_pipeline(
                directory=bf_dir,
                date=recording_date,
                identifier_pattern=pattern,
                cleanup=cleanup,
                dry_run=dry_run,
                lat=lat,
                lon=lon,
            )

        # Process signal averaging directory
        if run_sa and sa_dir:
            process_directory_pipeline(
                directory=sa_dir,
                date=recording_date,
                identifier_pattern="",
                cleanup=False,
                dry_run=dry_run,
                lat=lat,
                lon=lon,
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
        "--rpiid", type=str, default=None,
        help="Full RPiID name (default: first available with data)",
    )
    parser.add_argument(
        "--location", type=str, default=None,
        help="Short location code, e.g. '2A400' (overrides --rpiid)",
    )
    parser.add_argument(
        "--date", type=str, default=None,
        help="Date string YYYY-MM-DD (default: earliest available)",
    )
    parser.add_argument(
        "--max-files", type=int, default=1,
        help="Max FLAC files to process (default 1 for prototype)",
    )
    parser.add_argument(
        "--ir-types", type=str, default="LabIR",
        help="Comma-separated IR types: LabIR,SPIR1,SPIR2",
    )
    parser.add_argument(
        "--no-sa", action="store_true",
        help="Skip signal averaging",
    )
    parser.add_argument(
        "--no-birdnet", action="store_true",
        help="Skip BirdNET analysis",
    )
    parser.add_argument(
        "--cleanup", action="store_true",
        help="Delete low-confidence beamforming files after processing",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be deleted without actually deleting",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use full IR parameter sets (not prototype subsets)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available RPiIDs and dates, then exit",
    )
    parser.add_argument(
        "--precompute", action="store_true",
        help="Precompute IR steering-vector caches before running",
    )
    parser.add_argument(
        "--force-bf", action="store_true",
        help="Force re-run beamforming even if outputs already exist",
    )

    args = parser.parse_args()

    # ── Precompute IR caches (if requested) ──────────────────
    if args.precompute:
        from ircache import build_all_caches
        print("🔧 Precomputing IR steering-vector caches ...")
        build_all_caches()
        print("Ready.\n")
        if not args.location and not args.rpiid and not args.list:
            sys.exit(0)

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
                if os.path.isdir(os.path.join(rp_path, d)) and d != "logs"
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
        if os.path.isdir(os.path.join(rpiid_dir, d)) and d != "logs"
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
            force_bf=args.force_bf,
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
