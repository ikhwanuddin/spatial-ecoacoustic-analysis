#!/usr/bin/env python3
"""
Migrate sea-data from HDD to SSD and reorganize into hour-based subfolders.

Usage:
    python migrate_to_ssd.py --location 2A400
    python migrate_to_ssd.py --location 2A400 --dry-run

What it does:
    1. Move WAV files from /Volumes/HD Data/sea-data/{location}/{date}/{method}/*.wav
       to /Volumes/WD2TB/sea-data/{location}/{date}/{method}/hour_XX/*.wav
    2. Rebuild results.json and processed.json per hour based on WAV filename prefix.
    3. Skip macOS resource fork files (._*).
    4. Skip location if already migrated (hour_XX subdirs exist on SSD).
"""

import os
import sys
import json
import shutil
import argparse
from typing import Dict, List, Optional

HDD_BASE = "/Volumes/HD Data/sea-data"
SSD_BASE = "/Volumes/WD2TB/sea-data"

PROCESSING_DIRS = [
    "beamforming_LabIR",
    "beamforming_SPIR1",
    "beamforming_SPIR2",
    "signal_averaging",
]


def extract_hour(filename: str) -> Optional[str]:
    """Extract hour prefix from filename like '08-17-52_dur=...wav' -> '08'."""
    # WAV files start with HH-MM-SS
    if len(filename) >= 2 and filename[:2].isdigit():
        return filename[:2]
    return None


def migrate_wav_files(src_dir: str, dst_dir: str, dry_run: bool = False) -> int:
    """Move WAV files from src_dir to hour subfolders in dst_dir. Returns count moved."""
    if not os.path.isdir(src_dir):
        print(f"  ⚠  Source not found: {src_dir}")
        return 0

    moved = 0
    wav_files = sorted([
        f for f in os.listdir(src_dir)
        if f.lower().endswith(".wav") and not f.startswith("._")
    ])

    if not wav_files:
        print(f"  No WAV files in {src_dir}")
        return 0

    # Extract hours from filenames
    hours = set()
    hour_files: Dict[str, List[str]] = {}
    for f in wav_files:
        hour = extract_hour(f)
        if hour:
            hours.add(hour)
            hour_files.setdefault(hour, []).append(f)

    print(f"  Moving {len(wav_files)} WAV files → {len(hours)} hour(s)")
    for hour in sorted(hours):
        hour_dst = os.path.join(dst_dir, f"hour_{hour}")
        files = hour_files[hour]
        if dry_run:
            print(f"    [DRY RUN] hour_{hour}/: {len(files)} files")
        else:
            os.makedirs(hour_dst, exist_ok=True)
            for f in files:
                src = os.path.join(src_dir, f)
                dst = os.path.join(hour_dst, f)
                shutil.move(src, dst)
            print(f"    hour_{hour}/: {len(files)} files moved")
        moved += len(files)

    return moved


def split_json_by_hour(src_dir: str, dst_dir: str, json_name: str, dry_run: bool = False):
    """
    For a directory with hour subfolders, split a combined JSON (flat WAV filenames)
    into per-hour JSON files.

    For results.json: keys are WAV filenames -> filter by hour prefix.
    For processed.json: primary_channel_list contains WAV filenames -> filter by hour prefix.
    """
    json_path = os.path.join(src_dir, json_name)
    if not os.path.isfile(json_path):
        print(f"  ⚠  {json_name} not found in {src_dir}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Determine which hours exist in the destination
    if not os.path.isdir(dst_dir):
        return

    hour_dirs = sorted([
        d for d in os.listdir(dst_dir)
        if d.startswith("hour_") and os.path.isdir(os.path.join(dst_dir, d))
    ])

    for hour_dir in hour_dirs:
        hour_str = hour_dir.replace("hour_", "")
        hour_dst = os.path.join(dst_dir, hour_dir)
        hour_json_path = os.path.join(hour_dst, json_name)

        if json_name == "results.json":
            # Filter entries whose key starts with the hour prefix
            filtered = {}
            for key, dets in data.items():
                base = os.path.basename(key)
                if base.startswith(hour_str):
                    filtered[key] = dets

            if filtered:
                if not dry_run:
                    with open(hour_json_path, "w", encoding="utf-8") as f:
                        json.dump(filtered, f, indent=4, ensure_ascii=False)
                print(f"    {hour_dir}/{json_name}: {len(filtered)} entries")
            else:
                # Empty results.json for hours with no detections
                if not dry_run:
                    with open(hour_json_path, "w", encoding="utf-8") as f:
                        json.dump({}, f, indent=4)
                print(f"    {hour_dir}/{json_name}: empty")

        elif json_name == "processed.json":
            # Filter species entries by primary_channel_list filename hour prefix
            filtered = {}
            for species, sp_data in data.items():
                channels = sp_data.get("primary_channel_list", [])
                confs = sp_data.get("conf_list", [])
                starts = sp_data.get("start_time_list", [])

                new_channels = []
                new_confs = []
                new_starts = []
                for i, ch in enumerate(channels):
                    base = os.path.basename(ch)
                    if base.startswith(hour_str):
                        new_channels.append(ch)
                        new_confs.append(confs[i] if i < len(confs) else 0)
                        new_starts.append(starts[i] if i < len(starts) else 0)

                if new_channels:
                    filtered[species] = {
                        "primary_channel_list": new_channels,
                        "conf_list": new_confs,
                        "start_time_list": new_starts,
                        "count": len(new_channels),
                        "conf_avg": round(sum(new_confs) / len(new_confs), 3),
                        "conf_max": round(max(new_confs), 3),
                        "conf_min": round(min(new_confs), 3),
                    }

            if not dry_run:
                with open(hour_json_path, "w", encoding="utf-8") as f:
                    json.dump(filtered, f, indent=4, ensure_ascii=False)
            print(f"    {hour_dir}/{json_name}: {len(filtered)} species")


def migrate_location(location: str, dry_run: bool = False):
    """Migrate all data for one location from HDD to SSD."""
    hdd_loc = os.path.join(HDD_BASE, location)
    ssd_loc = os.path.join(SSD_BASE, location)

    if not os.path.isdir(hdd_loc):
        print(f"❌ Location not found in HDD: {hdd_loc}")
        return

    os.makedirs(ssd_loc, exist_ok=True)

    # Iterate dates
    for date_entry in sorted(os.listdir(hdd_loc)):
        date_path = os.path.join(hdd_loc, date_entry)
        if not os.path.isdir(date_path):
            continue

        print(f"\n📅 {location}/{date_entry}")

        for method in PROCESSING_DIRS:
            src_method = os.path.join(date_path, method)
            dst_method = os.path.join(ssd_loc, date_entry, method)

            if not os.path.isdir(src_method):
                continue

            # Check if already migrated
            already = os.path.isdir(dst_method) and any(
                d.startswith("hour_") for d in os.listdir(dst_method)
                if os.path.isdir(os.path.join(dst_method, d))
            )
            if already:
                print(f"  ✓ {method}: already has hour subfolders — skipping")
                continue

            print(f"  📦 {method}")

            # Step 1: Move WAV files into hour subfolders
            os.makedirs(dst_method, exist_ok=True)
            moved = migrate_wav_files(src_method, dst_method, dry_run)
            if moved == 0:
                continue

            # Step 2: Rebuild per-hour JSON files
            split_json_by_hour(src_method, dst_method, "results.json", dry_run)
            split_json_by_hour(src_method, dst_method, "processed.json", dry_run)

            # Also check for daily_report.html at method level (old structure) or date level
            for extra_file in ["daily_report.html"]:
                src_extra = os.path.join(src_method, extra_file)
                if os.path.isfile(src_extra):
                    # Daily report belongs at the location level, not method level
                    # Just note it for now
                    print(f"    ℹ  Found {extra_file} — can be regenerated on SSD")


def main():
    parser = argparse.ArgumentParser(
        description="Migrate sea-data from HDD to SSD with hour-based reorganization"
    )
    parser.add_argument("--location", type=str, required=True,
                        help="Location code (e.g., 2A400)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without actually doing it")
    args = parser.parse_args()

    if not os.path.isdir(HDD_BASE):
        print(f"❌ HDD not mounted: {HDD_BASE}")
        sys.exit(1)
    if not os.path.isdir(SSD_BASE):
        print(f"❌ SSD not mounted: {SSD_BASE}")
        sys.exit(1)

    mode = "DRY RUN" if args.dry_run else "LIVE"
    print(f"🔧 Migration {mode}: {args.location}")
    print(f"   From: {HDD_BASE}/{args.location}")
    print(f"   To:   {SSD_BASE}/{args.location}")
    print()

    migrate_location(args.location, dry_run=args.dry_run)

    print(f"\n✅ Migration {mode.lower()} complete")


if __name__ == "__main__":
    main()
