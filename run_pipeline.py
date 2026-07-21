#!/usr/bin/env python3
"""
Spatial Ecoacoustic Analysis Pipeline — Main Entry Point.

Orchestrates:
  1. Beamforming (LabIR / SPIR1 / SPIR2) on FLAC recordings
  2. Signal Averaging (6-ch -> 1-ch direct sum)
  3. BirdNET analysis on outputs (with site-specific GPS) — PARALLEL
  4. Confidence comparison -> processed.json
  5. Cleanup: keep only best-variant beamforming files

Resume support: pipeline_state.json on the output volume tracks which
steps are complete per FLAC file. Interrupted runs resume automatically.

Usage:
    python run_pipeline.py
    python run_pipeline.py --location 2A400 --date 2026-04-21 --max-files 10 --ir-types LabIR,SPIR1,SPIR2
"""

import os
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional

from config import (
    MONITORING_DATA, ANALYSIS_OUTPUT, IR_TYPES, PROTOTYPE_IR_SUBSETS,
    LOCATION_MAP, RPIID_TO_LOCATION, SITE_COORDS,
)
from beamforming import Beamformer
from signal_averaging import SignalAverager
from birdnet_processor import process_directory_pipeline
from pipeline_state import PipelineState, STEP_BF_PREFIX, STEP_SA, STEP_BIRNET_PREFIX, STEP_BIRNET_SA, STEP_MONO, STEP_BIRNET_MONO

# Max parallel BirdNET directories (all beamforming + SA = up to 4 dirs at once)
BIRDNET_PARALLEL_DIRS = int(os.environ.get("BIRDNET_PARALLEL_DIRS", "4"))


# ============================================================
# HELPERS
# ============================================================

def get_flac_files(rpiid: str, date_str: str, max_files: int = 1) -> List[str]:
    date_dir = os.path.join(MONITORING_DATA, rpiid, date_str)
    if not os.path.isdir(date_dir):
        print(f"❌ Directory not found: {date_dir}")
        return []
    flacs = sorted([
        os.path.join(date_dir, f)
        for f in os.listdir(date_dir)
        if f.lower().endswith(".flac") and not f.startswith("._")
    ])
    if max_files and len(flacs) > max_files:
        flacs = flacs[:max_files]
    print(f"📁 {len(flacs)} FLAC file(s) selected from {date_dir}")
    for f in flacs:
        print(f"    → {os.path.basename(f)}")
    return flacs


def build_output_path(location_name: str, date_str: str, processing_type: str, hour_subdir: str = "") -> str:
    path = os.path.join(ANALYSIS_OUTPUT, location_name, date_str, processing_type)
    if hour_subdir:
        path = os.path.join(path, hour_subdir)
    return path


def get_location_name_from_rpiid(rpiid: str) -> str:
    if rpiid in RPIID_TO_LOCATION:
        return RPIID_TO_LOCATION[rpiid]
    return rpiid


def parse_flac_date(flac_path: str, folder_date_str: str) -> datetime:
    try:
        return datetime.strptime(folder_date_str, "%Y-%m-%d")
    except ValueError:
        return datetime.now()


def get_site_coords(location_name: str):
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


def _extract_hour(flac_path: str) -> str:
    base = os.path.basename(flac_path)
    hour = base[:2]
    return f"hour_{hour}"


# ============================================================
# SINGLE-FILE PIPELINE
# ============================================================

def process_one_flac(
    flac_path: str, location_name: str, date_str: str,
    ir_types: List[str], run_sa: bool = True, run_birdnet: bool = True,
    cleanup: bool = False, dry_run: bool = False,
    use_prototype_subsets: bool = False, force_bf: bool = False,
    state: Optional["PipelineState"] = None,
):
    base_name = os.path.splitext(os.path.basename(flac_path))[0]
    hour_subdir = _extract_hour(flac_path)
    key = None
    if state:
        key = state.make_key(location_name, date_str, hour_subdir, base_name)
        state.auto_detect_from_disk(
            location_name, date_str, hour_subdir, base_name,
            ir_types, run_sa=run_sa, run_birdnet=run_birdnet,
        )

    print(f"\n{'='*60}")
    print(f"🎙  Processing: {base_name}")
    print(f"📍 Location:  {location_name}")
    print(f"📅 Date:      {date_str}")
    print(f"🕐 Hour:      {hour_subdir}")
    print(f"📡 IR Types:  {', '.join(ir_types)}")
    if state and key:
        entry = state.get_entry(key)
        if entry:
            statuses = {k: v for k, v in entry.items() if k != "last_updated"}
            print(f"📋 Resume:   {statuses}")
    print(f"{'='*60}")

    overall_start = time.time()
    bf_dirs = []
    sa_dir = ""

    ir_configs = PROTOTYPE_IR_SUBSETS if use_prototype_subsets else IR_TYPES

    # ── Step 1: Beamforming ──────────────────────────────────
    for ir_name in ir_types:
        if ir_name not in ir_configs:
            print(f"⚠  Unknown IR type: {ir_name} — skipping")
            continue

        ir_type = ir_configs[ir_name]
        bf_dir = build_output_path(location_name, date_str, f"beamforming_{ir_name}", hour_subdir)
        bf_dirs.append((bf_dir, ir_name))

        print(f"\n── Beamforming [{ir_name}] → {bf_dir} ──")
        bf_step = f"{STEP_BF_PREFIX}{ir_name}"
        if not force_bf and state and key and state.is_complete(key, bf_step):
            print(f"  ✓ {ir_name} already complete (state) — skipping")
            continue
        if not force_bf and _beamforming_complete(bf_dir, base_name, ir_type):
            print(f"  ✓ {ir_name} outputs already exist — skipping")
            if state and key:
                state.mark_complete(key, bf_step)
            continue

        bf = Beamformer(flac_path=flac_path, output_dir=bf_dir, ir_type_or_name=ir_type)
        bf.run()
        if state and key:
            state.mark_complete(key, bf_step)

    # ── Step 2: Signal Averaging ─────────────────────────────
    if run_sa:
        sa_dir = build_output_path(location_name, date_str, "signal_averaging", hour_subdir)
        print(f"\n── Signal Averaging → {sa_dir} ──")
        if not force_bf and state and key and state.is_complete(key, STEP_SA):
            print(f"  ✓ SA already complete (state) — skipping")
        elif not force_bf and _sa_complete(sa_dir, base_name):
            print(f"  ✓ SA output already exists — skipping")
            if state and key:
                state.mark_complete(key, STEP_SA)
        else:
            sa = SignalAverager(flac_path=flac_path, output_dir=sa_dir)
            sa.run()
            if state and key:
                state.mark_complete(key, STEP_SA)

    # ── Step 2.5: Monochannel Baseline (first channel) ────────
    import librosa as _librosa
    import soundfile as _sf
    from config import FS_TARGET

    mono_dir = build_output_path(location_name, date_str, "mono_baseline", hour_subdir)
    mono_out = os.path.join(mono_dir, base_name + "_mono.wav")

    print(f"\n── Monochannel Baseline → {mono_dir} ──")
    if state and key and state.is_complete(key, STEP_MONO):
        print(f"  ✓ Mono baseline already complete (state) — skipping")
    elif os.path.isfile(mono_out):
        print(f"  ✓ Mono baseline already exists — skipping")
        if state and key:
            state.mark_complete(key, STEP_MONO)
    else:
        try:
            raw, _ = _librosa.load(flac_path, sr=FS_TARGET, mono=False)
            ch0 = raw[0, :] if raw.ndim > 1 else raw
            amax = max(abs(ch0))
            if amax > 1.0:
                ch0 = ch0 / amax
            os.makedirs(mono_dir, exist_ok=True)
            _sf.write(
                mono_out,
                (ch0 * 32767).clip(-32768, 32767).astype("int16"),
                FS_TARGET, subtype="PCM_16",
            )
            print(f"  ✓ Mono baseline: {mono_out}")
            if state and key:
                state.mark_complete(key, STEP_MONO)
        except Exception as e:
            print(f"  ❌ Mono baseline failed: {e}")

    # ── Step 3: BirdNET — PARALLEL across all dirs ───────────
    if run_birdnet:
        recording_date = parse_flac_date(flac_path, date_str)
        lat, lon = get_site_coords(location_name)
        if lat is not None:
            print(f"\n  🌍 Site coordinates: {lat}, {lon}")

        # Collect all BirdNET tasks
        birdnet_tasks = []

        for bf_dir, ir_name in bf_dirs:
            ir_label = os.path.basename(os.path.dirname(bf_dir)).replace("beamforming_", "")
            bn_step = f"{STEP_BIRNET_PREFIX}{ir_name}"

            if state and key and state.is_complete(key, bn_step):
                print(f"  ✓ BirdNET [{ir_label}] already complete — skipping")
                continue
            if os.path.isfile(os.path.join(bf_dir, "results.json")) and \
               os.path.isfile(os.path.join(bf_dir, "processed.json")):
                print(f"  ✓ BirdNET [{ir_label}] results exist — skipping")
                if state and key:
                    state.mark_complete(key, bn_step)
                continue

            pattern = f"_{ir_label}("
            birdnet_tasks.append((bf_dir, ir_label, bn_step, pattern))

        if run_sa and sa_dir:
            if state and key and state.is_complete(key, STEP_BIRNET_SA):
                print(f"  ✓ BirdNET [SA] already complete — skipping")
            elif os.path.isfile(os.path.join(sa_dir, "results.json")) and \
                 os.path.isfile(os.path.join(sa_dir, "processed.json")):
                print(f"  ✓ BirdNET [SA] results exist — skipping")
                if state and key:
                    state.mark_complete(key, STEP_BIRNET_SA)
            else:
                birdnet_tasks.append((sa_dir, "SA", STEP_BIRNET_SA, ""))

        # Monochannel baseline BirdNET
        if os.path.isfile(mono_out):
            if state and key and state.is_complete(key, STEP_BIRNET_MONO):
                print(f"  ✓ BirdNET [Mono] already complete — skipping")
            elif os.path.isfile(os.path.join(mono_dir, "results.json")) and \
                 os.path.isfile(os.path.join(mono_dir, "processed.json")):
                print(f"  ✓ BirdNET [Mono] results exist — skipping")
                if state and key:
                    state.mark_complete(key, STEP_BIRNET_MONO)
            else:
                birdnet_tasks.append((mono_dir, "Mono", STEP_BIRNET_MONO, ""))

        if birdnet_tasks:
            print(f"\n  🚀 BirdNET parallel: {len(birdnet_tasks)} directories, "
                  f"{min(BIRDNET_PARALLEL_DIRS, len(birdnet_tasks))} concurrent")

            def _run_birdnet_dir(directory, label, step_name, pattern):
                """Wrapper for parallel BirdNET execution."""
                try:
                    process_directory_pipeline(
                        directory=directory, date=recording_date,
                        identifier_pattern=pattern,
                        cleanup=(cleanup and label not in ("SA", "Mono")),
                        dry_run=dry_run, lat=lat, lon=lon,
                    )
                    if state and key:
                        state.mark_complete(key, step_name)
                    return (label, True, "")
                except Exception as e:
                    return (label, False, str(e))

            t0 = time.time()
            n_workers = min(BIRDNET_PARALLEL_DIRS, len(birdnet_tasks))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {
                    pool.submit(_run_birdnet_dir, d, l, s, p): l
                    for d, l, s, p in birdnet_tasks
                }
                for future in as_completed(futures):
                    label, ok, err = future.result()
                    if ok:
                        print(f"    ✅ BirdNET [{label}] done")
                    else:
                        print(f"    ❌ BirdNET [{label}] failed: {err}")

            print(f"    ⏱  BirdNET parallel: {time.time() - t0:.1f}s")

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"✅ Done — {base_name} processed in {elapsed:.1f}s")
    print(f"{'='*60}")

    return {
        "flac": flac_path,
        "beamforming_dirs": [d for d, _ in bf_dirs],
        "sa_dir": sa_dir,
        "mono_dir": mono_dir,
        "elapsed": elapsed,
    }


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Spatial Ecoacoustic Analysis Pipeline")
    parser.add_argument("--rpiid", type=str, default=None)
    parser.add_argument("--location", type=str, default=None)
    parser.add_argument("--date", type=str, default=None)
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--ir-types", type=str, default="LabIR")
    parser.add_argument("--no-sa", action="store_true")
    parser.add_argument("--no-birdnet", action="store_true")
    parser.add_argument("--cleanup", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--precompute", action="store_true")
    parser.add_argument("--force-bf", action="store_true")
    parser.add_argument("--state-status", action="store_true")
    parser.add_argument("--reset-state", type=str, default=None)

    args = parser.parse_args()

    if args.reset_state is not None:
        state = PipelineState()
        key = None if args.reset_state == "all" else args.reset_state
        print(f"Resetting state: {args.reset_state}")
        state.reset_key(key)
        print(state.detailed_summary())
        sys.exit(0)

    if args.state_status:
        state = PipelineState()
        print(state.summary())
        print()
        print(state.detailed_summary())
        sys.exit(0)

    if args.precompute:
        from ircache import build_all_caches
        print("🔧 Precomputing IR steering-vector caches ...")
        build_all_caches()
        print("Ready.\n")
        if not args.location and not args.rpiid and not args.list:
            sys.exit(0)

    # Resolve RPiID
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
        sys.exit(0)

    # Resolve date
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

    # Parse IR types
    ir_types = [t.strip() for t in args.ir_types.split(",")]
    for t in ir_types:
        if t not in IR_TYPES:
            print(f"❌ Unknown IR type: {t}")
            print(f"   Known: {list(IR_TYPES.keys())}")
            sys.exit(1)

    use_prototype = not args.full
    if use_prototype:
        print(f"🧪 Prototype mode: reduced IR parameter sets")

    # Get FLAC files
    flac_paths = get_flac_files(rpiid, date_str, max_files=args.max_files)
    if not flac_paths:
        print("❌ No FLAC files found.")
        sys.exit(1)

    # Validate output volume
    if not os.path.exists(ANALYSIS_OUTPUT):
        print(f"❌ Output volume not mounted: {ANALYSIS_OUTPUT}")
        sys.exit(1)

    # Pipeline state
    state = PipelineState()
    state.clean_stale_keys(stale_days=7)

    # Run pipeline
    results = []
    for flac_path in flac_paths:
        result = process_one_flac(
            flac_path=flac_path, location_name=location_name, date_str=date_str,
            ir_types=ir_types, run_sa=not args.no_sa, run_birdnet=not args.no_birdnet,
            cleanup=args.cleanup, dry_run=args.dry_run,
            use_prototype_subsets=use_prototype, force_bf=args.force_bf,
            state=state,
        )
        results.append(result)

    # Summary
    total_elapsed = sum(r["elapsed"] for r in results)
    n_files = len(results)
    total_wavs = 0
    for r in results:
        for d in r["beamforming_dirs"]:
            try:
                total_wavs += len([f for f in os.listdir(d) if f.endswith(".wav")])
            except:
                pass
        for sub in ["sa_dir", "mono_dir"]:
            d = r.get(sub)
            if d:
                try:
                    total_wavs += len([f for f in os.listdir(d) if f.endswith(".wav")])
                except:
                    pass

    print(f"\n{'='*60}")
    print(f"🏁 Pipeline Complete")
    print(f"{'='*60}")
    print(f"  📂 Files processed : {n_files}")
    print(f"  🔊 WAV generated   : {total_wavs}")
    print(f"  ⏱  Total elapsed   : {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")
    if n_files > 1:
        print(f"  ⏱  Avg per file    : {total_elapsed/n_files:.1f}s")
    print(f"  📍 Location        : {location_name}")
    print(f"  📅 Date            : {date_str}")
    print(f"  📡 IR Types        : {', '.join(ir_types)}")
    print(f"  💾 Output          : {ANALYSIS_OUTPUT}")
    print(f"{'-'*60}")
    for r in results:
        bname = os.path.splitext(os.path.basename(r["flac"]))[0]
        print(f"  {bname}: {r['elapsed']:.1f}s")
    print(f"{'='*60}")

    # ── Write JSON run report ───────────────────────────────
    import json as _json
    from datetime import datetime as _dt
    report_path = os.path.join(ANALYSIS_OUTPUT, location_name, date_str,
                               f"_run_report_{_dt.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = {
        "run_at": _dt.now().isoformat(),
        "hostname": os.uname().nodename,
        "location": location_name,
        "date": date_str,
        "ir_types": ir_types,
        "files_processed": n_files,
        "wavs_generated": total_wavs,
        "total_elapsed_s": round(total_elapsed, 1),
        "avg_per_file_s": round(total_elapsed / n_files, 1) if n_files else 0,
        "files": [
            {
                "name": os.path.splitext(os.path.basename(r["flac"]))[0],
                "elapsed_s": round(r["elapsed"], 1),
            }
            for r in results
        ],
    }
    with open(report_path, "w") as f:
        _json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  📋 Report: {report_path}")


if __name__ == "__main__":
    main()
