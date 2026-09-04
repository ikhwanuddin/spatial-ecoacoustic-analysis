"""
Integrated Pipeline Runner for Spatial Ecoacoustic Analysis (SEA).
Executes Modules 1 -> 2 -> 3 -> 4 sequentially across all recordings in a date.

Usage:
  python src/run_date_pipeline.py --location 2A400 --date 2026-04-22 --processes 4
"""

import os
import sys
import glob
import time
import argparse
from datetime import datetime

# Local module imports
from config import (
    MONITORING_DATA,
    SCRATCH_DIR,
    OUTPUT_DIR,
    LOCATION_MAP,
    DEFAULT_THRESHOLDS,
)
from src.render_signals import render_single_flac
from src.birdnet_infer import run_birdnet_batch
from src.extract_detections import process_results_file
from src.pair_and_recap import pair_methods, evaluate_threshold_counts, format_markdown_table
import json


def process_date(location: str, date_str: str, max_files: int = 0, processes: int = 4):
    rpi_id = LOCATION_MAP.get(location, location)
    flac_dir = os.path.join(MONITORING_DATA, rpi_id, date_str)
    flac_files = sorted(glob.glob(os.path.join(flac_dir, "*.flac")))

    if not flac_files:
        print(f"❌ No FLAC files found in {flac_dir}")
        return 1

    if max_files > 0:
        flac_files = flac_files[:max_files]

    scratch_date_dir = os.path.join(SCRATCH_DIR, location, date_str)
    output_date_dir = os.path.join(OUTPUT_DIR, location, date_str)
    os.makedirs(scratch_date_dir, exist_ok=True)
    os.makedirs(output_date_dir, exist_ok=True)

    date_obj = datetime.strptime(date_str, "%Y-%m-%d")

    print("=" * 70)
    print(f"🚀 SEA PIPELINE RUNNER: {location} | Date: {date_str}")
    print(f"📁 Total recordings to process: {len(flac_files)}")
    print(f"💾 Scratch directory: {scratch_date_dir}")
    print(f"🏆 Permanent output:  {output_date_dir}")
    print("=" * 70)

    t_global = time.time()
    daily_collated = {
        "mono_channel": {},
        "sa_channel": {},
        "beamformed_LabIR": {},
        "beamformed_SPIR": {},
        "beamformed_all": {},
    }

    for idx, flac in enumerate(flac_files, 1):
        rec_name = os.path.splitext(os.path.basename(flac))[0]
        rec_scratch = os.path.join(scratch_date_dir, rec_name)
        rec_output = os.path.join(output_date_dir, rec_name)
        os.makedirs(rec_scratch, exist_ok=True)
        os.makedirs(rec_output, exist_ok=True)

        print(f"\n[{idx}/{len(flac_files)}] >>> Recording: {rec_name}")
        t_rec = time.time()

        # Step 1: Render Signals
        results_json = os.path.join(rec_scratch, "results.json")
        processed_json = os.path.join(rec_scratch, "processed.json")

        # Check if already rendered
        wav_count = len([f for f in os.listdir(rec_scratch) if f.endswith(".wav")])
        if wav_count < 52:
            print("  1️⃣  Rendering 52 audio streams (Mono, SA, LabIR, SPIR)...")
            render_single_flac(flac, rec_scratch, render_beams=True, workers=processes)
        else:
            print("  1️⃣  [Skipped] 52 WAV streams already exist.")

        # Step 2: BirdNET Batch Inference
        if not os.path.exists(results_json):
            print("  2️⃣  Running BirdNET batch inference...")
            run_birdnet_batch(rec_scratch, date_obj=date_obj, processes=processes)
        else:
            print("  2️⃣  [Skipped] results.json already exists.")

        # Step 3: Automated Source Selection
        print("  3️⃣  Extracting winning directions (prim_key = species_time)...")
        processed = process_results_file(results_json, conf_thresh=0.0)
        with open(processed_json, "w") as f:
            json.dump(processed, f, indent=4, ensure_ascii=False)

        # Step 4: Paired Comparison
        print("  4️⃣  Pairing detections and evaluating thresholds...")
        paired_labir = pair_methods(processed.get("mono_channel", {}), processed.get("beamformed_LabIR", {}))
        paired_spir = pair_methods(processed.get("mono_channel", {}), processed.get("beamformed_SPIR", {}))

        paired_file = os.path.join(rec_output, "paired_detections.json")
        with open(paired_file, "w") as f:
            json.dump({"mono_vs_LabIR": paired_labir, "mono_vs_SPIR": paired_spir}, f, indent=4, ensure_ascii=False)

        summary = evaluate_threshold_counts(processed, DEFAULT_THRESHOLDS)
        with open(os.path.join(rec_output, "threshold_summary.json"), "w") as f:
            json.dump(summary, f, indent=4)

        # Collate into daily total
        for m in daily_collated:
            for sp, sinfo in processed.get(m, {}).items():
                if sp not in daily_collated[m]:
                    daily_collated[m][sp] = {"conf_list": [], "start_time_list": []}
                daily_collated[m][sp]["conf_list"].extend(sinfo.get("conf_list", []))

        print(f"  ✓ Recording completed in {time.time() - t_rec:.2f}s")

    # Step 5: Generate Daily Summary
    print("\n" + "=" * 70)
    print(f"📊 GENERATING DAILY SUMMARY: {location} | {date_str}")
    daily_summary = evaluate_threshold_counts(daily_collated, DEFAULT_THRESHOLDS)
    daily_summary_path = os.path.join(output_date_dir, "daily_summary.json")
    with open(daily_summary_path, "w") as f:
        json.dump(daily_summary, f, indent=4)

    md_table = format_markdown_table(daily_summary, DEFAULT_THRESHOLDS)
    daily_md_path = os.path.join(output_date_dir, "daily_summary.md")
    with open(daily_md_path, "w") as f:
        f.write(f"# Daily Detection Summary: {location} ({date_str})\n\n")
        f.write(f"Total recordings: {len(flac_files)}\n\n")
        f.write(md_table + "\n")

    print(md_table)
    print(f"\n✅ Daily summary saved to: {daily_md_path}")
    print(f"🏁 Total execution time: {time.time() - t_global:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Run complete SEA pipeline for a given date.")
    parser.add_argument("--location", default="2A400", help="Location code (default: 2A400)")
    parser.add_argument("--date", default="2026-04-22", help="Date string YYYY-MM-DD")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to process (0 = all)")
    parser.add_argument("--processes", type=int, default=8, help="Number of CPU worker processes")
    args = parser.parse_args()

    process_date(args.location, args.date, max_files=args.max_files, processes=args.processes)


if __name__ == "__main__":
    main()
