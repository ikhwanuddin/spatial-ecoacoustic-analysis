"""
Integrated Pipeline Runner for Spatial Ecoacoustic Analysis (SEA).
Executes Modules 1 -> 2 -> 3 -> 4 -> 5 sequentially across all recordings in a date.
Includes automatic detection of corrupted FLAC files, repair attempts via ffmpeg/flac,
graceful skipping of unrecoverable files, and structured skip reporting.

Usage:
  python src/run_date_pipeline.py --location 2A400 --date 2026-04-22 --processes 8
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
from render_signals import render_single_flac
from birdnet_infer import run_birdnet_batch
from extract_detections import process_results_file
from pair_and_recap import pair_methods, evaluate_threshold_counts, format_markdown_table
from generate_audit_manifest import generate_date_audit_manifest
from extract_audit_clips import extract_clips_for_date
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
    corrupted_skipped = []
    processed_count = 0

    for idx, flac in enumerate(flac_files, 1):
        rec_name = os.path.splitext(os.path.basename(flac))[0]
        rec_scratch = os.path.join(scratch_date_dir, rec_name)
        rec_output = os.path.join(output_date_dir, rec_name)
        os.makedirs(rec_scratch, exist_ok=True)
        os.makedirs(rec_output, exist_ok=True)

        print(f"\n[{idx}/{len(flac_files)}] >>> Recording: {rec_name}")
        t_rec = time.time()

        # Step 1: Render Signals (with corruption verification & auto-repair)
        results_json = os.path.join(rec_scratch, "results.json")
        processed_json = os.path.join(rec_output, "processed.json")

        wav_count = len([f for f in os.listdir(rec_scratch) if f.endswith(".wav")])
        if wav_count < 52:
            print("  1️⃣  Rendering 52 audio streams (Mono, SA, LabIR, SPIR)...")
            ok, err_info = render_single_flac(flac, rec_scratch, render_beams=True, workers=processes)
            if not ok:
                print(f"  ❌ GAGAL OLAH: Berkas FLAC rusak dan tidak dapat dipulihkan.")
                print(f"     Berkas   : {err_info['file_name']}")
                print(f"     Ukuran   : {err_info['file_size_human']} ({err_info['file_size_bytes']} bytes)")
                print(f"     Penyebab : {err_info['initial_error']}")
                print(f"     Keputusan: SKIP rekaman ini dan lanjut ke rekaman berikutnya.\n")
                corrupted_skipped.append(err_info)
                continue
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
        rec_md = format_markdown_table(summary, DEFAULT_THRESHOLDS)
        with open(os.path.join(rec_output, "threshold_summary.md"), "w") as f:
            f.write(rec_md + "\n")

        # Collate into daily total
        for m in daily_collated:
            for sp, sinfo in processed.get(m, {}).items():
                if sp not in daily_collated[m]:
                    daily_collated[m][sp] = {"conf_list": [], "start_time_list": []}
                daily_collated[m][sp]["conf_list"].extend(sinfo.get("conf_list", []))

        processed_count += 1
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
        f.write(f"Total recordings in source : {len(flac_files)}\n")
        f.write(f"Successfully processed    : {processed_count}\n")
        f.write(f"Corrupted & skipped       : {len(corrupted_skipped)}\n\n")
        f.write(md_table + "\n")

        if corrupted_skipped:
            f.write("\n## ⚠️ Corrupted & Skipped Recordings\n\n")
            f.write(f"Total: **{len(corrupted_skipped)}** berkas FLAC rusak tidak dapat dipulihkan.\n\n")
            f.write("| # | File Name | File Size | Initial Error | Decision |\n")
            f.write("|---|---|---|---|:---:|\n")
            for c_idx, c in enumerate(corrupted_skipped, 1):
                f.write(f"| {c_idx} | `{c['file_name']}` | {c['file_size_human']} | {c['initial_error']} | **{c['decision']}** |\n")

    print(md_table)
    print(f"\n✅ Daily summary saved to: {daily_md_path}")

    # Step 6: Generate Ground-Truth Detection Audit Manifest
    print("\n" + "=" * 70)
    print(f"📋 GENERATING DETECTION AUDIT MANIFEST: {location} | {date_str}")
    generate_date_audit_manifest(location, date_str, min_conf=0.25, out_dir=output_date_dir)

    # Step 7: Save Standalone Corrupted Files Report (if any)
    if corrupted_skipped:
        corr_json_path = os.path.join(output_date_dir, "corrupted_files.json")
        with open(corr_json_path, "w", encoding="utf-8") as f:
            json.dump(corrupted_skipped, f, indent=4, ensure_ascii=False)

        corr_md_path = os.path.join(output_date_dir, "corrupted_files.md")
        with open(corr_md_path, "w", encoding="utf-8") as f:
            f.write(f"# Laporan Berkas FLAC Rusak (Skipped): {location} ({date_str})\n\n")
            f.write(f"Total berkas dilewati: **{len(corrupted_skipped)}** dari {len(flac_files)} berkas.\n\n")
            f.write("| # | File Name | File Size | Initial Error | Repair Attempts | Decision |\n")
            f.write("|---|---|---|---|---|:---:|\n")
            for c_idx, c in enumerate(corrupted_skipped, 1):
                rep_str = "; ".join(c.get("repair_attempts", [])) or "Gagal rekonstruksi stream"
                f.write(f"| {c_idx} | `{c['file_name']}` | {c['file_size_human']} | {c['initial_error']} | {rep_str} | **{c['decision']}** |\n")

        print("\n" + "!" * 70)
        print(f"⚠️  PERINGATAN: Terdapat {len(corrupted_skipped)} berkas FLAC rusak yang di-skip.")
        print(f"   Laporan lengkap disimpan di: {corr_md_path}")
        print("!" * 70)

    # Step 8: Extract 5-Second Audit Snippets & Reclaim Ephemeral Scratch
    print("\n" + "=" * 70)
    print(f"✂️  EXTRACTING 5-SECOND AUDIT SNIPPETS & RECLAIMING SCRATCH: {location} | {date_str}")
    try:
        extract_clips_for_date(location, date_str, clean_scratch=True)
    except Exception as e:
        print(f"⚠️  Gagal ekstraksi klip audit / pembersihan scratch: {e}")

    print(f"🏁 Total execution time: {time.time() - t_global:.2f}s")


def main():
    parser = argparse.ArgumentParser(description="Run complete SEA pipeline for a given date or list of dates.")
    parser.add_argument("--location", default="2A400", help="Location code (default: 2A400)")
    parser.add_argument("--date", default=None, help="Single date string YYYY-MM-DD")
    parser.add_argument("--dates", nargs="+", default=None, help="One or more date strings YYYY-MM-DD")
    parser.add_argument("--max-files", type=int, default=0, help="Max files to process per date (0 = all)")
    parser.add_argument("--processes", type=int, default=8, help="Number of CPU worker processes")
    parser.add_argument("--smoke-test", action="store_true", help="Vibe check: fast end-to-end smoke test on 1 recording")
    args = parser.parse_args()

    max_files = 1 if args.smoke_test else args.max_files
    if args.smoke_test:
        print("⚡ VIBE SMOKE TEST ACTIVE: Fast 1-recording end-to-end verification (<20s)!")

    date_list = []
    if args.dates:
        date_list = args.dates
    elif args.date:
        date_list = [args.date]
    else:
        date_list = ["2026-04-22"]

    print("Queued dates:", date_list)
    for d_idx, d in enumerate(date_list, 1):
        print("=" * 70)
        print(f"[{d_idx}/{len(date_list)}] PROCESSING DATE: {d} (Location: {args.location})")
        print("=" * 70)
        try:
            process_date(args.location, d, max_files=max_files, processes=args.processes)
        except Exception as e:
            print(f"ERROR processing date {d}: {e}")
            import traceback
            traceback.print_exc()
            print(f"Skipping date {d} and continuing to next in queue...")


if __name__ == "__main__":
    main()
