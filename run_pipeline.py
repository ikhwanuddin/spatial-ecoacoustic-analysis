#!/usr/bin/env python3
"""
Spatial Ecoacoustic Analysis Pipeline — Main Entry Point.

Orchestrates:
  1. Beamforming (LabIR / SPIR1 / SPIR2) on FLAC recordings → 6s chunk WAVs
  2. Signal Averaging (6-ch → 1-ch direct sum) → full WAV
  3. Monochannel baseline → full WAV
  4. Pre-filter (RMS energy threshold): pool chunks across IR types within groups,
     keep only high-RMS survivors.  SPIR1 + SPIR2 are merged into bf_SPIR.
  5. BirdNET analysis on each per-minute directory (chunks for BF, full for SA/Mono)
  6. Confidence source-selection → processed.json per-minute
  7. Cleanup: delete losing chunk WAVs, keep only winners

Directory structure (per-date):
  sea-data/{location}/{date}/
    bf_LabIR/h_23/m_02/    ← 6s chunk WAVs, results.json, processed.json
    bf_SPIR/h_23/m_02/     ← merged SPIR1 + SPIR2 chunks (post-pre-filter)
    bf_SPIR1/h_23/m_02/    ← raw beamforming (deleted after pre-filter merge)
    bf_SPIR2/h_23/m_02/    ← raw beamforming (deleted after pre-filter merge)
    sa/h_23/m_02/          ← full WAV, results.json, processed.json
    mono/h_23/m_02/        ← full WAV, results.json, processed.json

Usage:
    python run_pipeline.py --location 2A400 --date 2026-04-20 --max-files 3
"""

import os
import re
import sys
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import List, Optional, Tuple

from config import (
    MONITORING_DATA, ANALYSIS_OUTPUT, IR_TYPES, PROTOTYPE_IR_SUBSETS,
    LOCATION_MAP, RPIID_TO_LOCATION, SITE_COORDS,
    resolve_birdnet_filter, PREFILTER_GROUPS, PREFILTER_RMS_THRESHOLD,
)
from beamforming import Beamformer
from signal_averaging import SignalAverager
from birdnet_processor import (
    process_directory_pipeline, slice_wav_to_chunks, CHUNK_SECONDS,
)
from prefilter import prefilter_directory, prefilter_merged
from pipeline_state import (
    PipelineState, STEP_BF_PREFIX, STEP_SA, STEP_BIRNET_PREFIX,
    STEP_BIRNET_SA, STEP_MONO, STEP_BIRNET_MONO, STEP_PREFILTER_PREFIX,
)

BIRDNET_PARALLEL_DIRS = int(os.environ.get("BIRDNET_PARALLEL_DIRS", "4"))

# ── Helpers ────────────────────────────────────────────────

_HM_RE = re.compile(r"^(\d{2})-(\d{2})-\d{2}_dur=")


def _extract_hour_minute(flac_path: str) -> Tuple[str, str]:
    """Return (hour, minute) from filename like '23-02-27_dur=240secs.flac'."""
    base = os.path.basename(flac_path)
    m = _HM_RE.match(base)
    if m:
        return m.group(1), m.group(2)
    return "00", "00"


def get_flac_files(rpiid: str, date_str: str, max_files: int = 1) -> List[str]:
    date_dir = os.path.join(MONITORING_DATA, rpiid, date_str)
    if not os.path.isdir(date_dir):
        print(f"\u274c Directory not found: {date_dir}")
        return []
    flacs = sorted([
        os.path.join(date_dir, f)
        for f in os.listdir(date_dir)
        if f.lower().endswith(".flac") and not f.startswith("._")
    ])
    if max_files and len(flacs) > max_files:
        flacs = flacs[:max_files]
    print(f"\U0001f4c1 {len(flacs)} FLAC file(s) selected from {date_dir}")
    for f in flacs:
        print(f"    \u2192 {os.path.basename(f)}")
    return flacs


def build_output_path(location_name: str, date_str: str,
                      processing_type: str, hour: str = "",
                      minute: str = "") -> str:
    """Build output path with h_HH/m_MM structure.
    Example: .../bf_LabIR/h_23/m_02/
    """
    path = os.path.join(ANALYSIS_OUTPUT, location_name, date_str, processing_type)
    if hour:
        path = os.path.join(path, f"h_{hour}")
    if minute:
        path = os.path.join(path, f"m_{minute}")
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


def _minute_complete(output_dir: str, base_name: str) -> bool:
    """Check if chunk WAVs exist for this FLAC in the minute directory."""
    if not os.path.isdir(output_dir):
        return False
    try:
        files = [f for f in os.listdir(output_dir)
                 if f.startswith("s_") and base_name in f and f.endswith(".wav")]
        return len(files) > 0
    except OSError:
        return False


def _sa_output_exists(output_dir: str, base_name: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, base_name + "_sa.wav"))


def _mono_output_exists(output_dir: str, base_name: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, base_name + "_mono.wav"))


# ── Single-FLAC pipeline (beamforming + slice + SA + mono) ──

def process_one_flac(
    flac_path: str, location_name: str, date_str: str,
    ir_types: List[str], run_sa: bool = True,
    use_prototype_subsets: bool = False, force_bf: bool = False,
    state: Optional["PipelineState"] = None,
) -> dict:
    base_name = os.path.splitext(os.path.basename(flac_path))[0]
    hour_str, minute_str = _extract_hour_minute(flac_path)

    print(f"\n{'='*60}")
    print(f"\U0001f399  Processing: {base_name}")
    print(f"\U0001f4cd Location: {location_name}")
    print(f"\U0001f4c5 Date:     {date_str}")
    print(f"\U0001f550 Hour:     {hour_str}  Minute: {minute_str}")
    print(f"{'='*60}")

    overall_start = time.time()
    ir_configs = PROTOTYPE_IR_SUBSETS if use_prototype_subsets else IR_TYPES

    bf_dirs: List[Tuple[str, str]] = []

    # ── Step 1: Beamforming + slice to chunks ───────────────
    for ir_name in ir_types:
        if ir_name not in ir_configs:
            print(f"\u26a0  Unknown IR type: {ir_name} — skipping")
            continue
        ir_type = ir_configs[ir_name]
        # Old-style dir for beamforming temp output (to avoid breaking Beamformer internals)
        # We still need to tell Beamformer where to put full WAVs.
        # Strategy: Beamformer writes full WAVs to the minute dir, we slice in-place, delete originals.
        bf_dir = build_output_path(location_name, date_str, f"bf_{ir_name}",
                                   hour_str, minute_str)
        bf_dirs.append((bf_dir, ir_name))

        bf_step = f"{STEP_BF_PREFIX}{ir_name}"
        if not force_bf and _minute_complete(bf_dir, base_name):
            print(f"  \u2713 bf_{ir_name} chunks already exist — skipping")
            continue

        # Force re-run: delete old chunk WAVs + full WAVs so we start clean
        if force_bf and os.path.isdir(bf_dir):
            removed = 0
            for fname in list(os.listdir(bf_dir)):
                if fname.endswith(".wav") and not fname.startswith("._"):
                    try:
                        os.remove(os.path.join(bf_dir, fname))
                        removed += 1
                    except OSError:
                        pass
            if removed > 0:
                print(f"  \U0001f5d1  Cleaned {removed} old WAV(s) from bf_{ir_name}")

        print(f"\n── Beamforming [{ir_name}] \u2192 {bf_dir} ──")
        beamformer = Beamformer(flac_path=flac_path, output_dir=bf_dir,
                                ir_type_or_name=ir_type)
        beamformer.run()

        # Slice all full WAVs for this source into 6s chunks, delete originals
        _slice_and_clean(bf_dir, base_name)

        if not _minute_complete(bf_dir, base_name):
            print(f"  \u26a0 Beamforming+slice [{ir_name}] incomplete")
        elif state:
            state.mark_complete(state.make_key(location_name, date_str,
                                               f"h_{hour_str}", base_name), bf_step)

    # ── Step 2: Signal Averaging ────────────────────────────
    sa_dir = ""
    if run_sa:
        sa_dir = build_output_path(location_name, date_str, "sa",
                                   hour_str, minute_str)
        print(f"\n── Signal Averaging → {sa_dir} ──")
        if _sa_output_exists(sa_dir, base_name):
            print("  ✓ SA already exists — skipping")
        else:
            sa = SignalAverager(flac_path=flac_path, output_dir=sa_dir)
            sa.run()
            if state:
                key = state.make_key(location_name, date_str,
                                     f"h_{hour_str}", base_name)
                state.mark_complete(key, STEP_SA)

    # ── Step 3: Mono baseline ───────────────────────────────
    import librosa as _librosa
    import soundfile as _sf
    from config import FS_TARGET

    mono_dir = build_output_path(location_name, date_str, "mono",
                                 hour_str, minute_str)
    mono_file = os.path.join(mono_dir, base_name + "_mono.wav")
    print(f"\n── Mono Baseline \u2192 {mono_dir} ──")
    if _mono_output_exists(mono_dir, base_name):
        print(f"  \u2713 Mono baseline already exists — skipping")
    else:
        try:
            os.makedirs(mono_dir, exist_ok=True)
            raw, _ = _librosa.load(flac_path, sr=FS_TARGET, mono=False)
            ch0 = raw[0, :] if raw.ndim > 1 else raw
            amax = max(abs(ch0))
            if amax > 1.0:
                ch0 = ch0 / amax
            _sf.write(mono_file,
                      (ch0 * 32767).clip(-32768, 32767).astype("int16"),
                      FS_TARGET, subtype="PCM_16")
            print(f"  \u2713 Mono baseline: {mono_file}")
            if state:
                key = state.make_key(location_name, date_str,
                                     f"h_{hour_str}", base_name)
                state.mark_complete(key, STEP_MONO)
        except Exception as e:
            print(f"  \u274c Mono baseline failed: {e}")

    elapsed = time.time() - overall_start
    print(f"\n{'='*60}")
    print(f"\u2705 Done — {base_name} in {elapsed:.1f}s")
    print(f"{'='*60}")

    return {
        "flac": flac_path,
        "base_name": base_name,
        "hour": hour_str,
        "minute": minute_str,
        "beamforming_dirs": [d for d, _ in bf_dirs],
        "sa_dir": sa_dir,
        "mono_dir": mono_dir,
        "elapsed": elapsed,
    }


def _slice_and_clean(output_dir: str, base_name: str):
    """Slice all full WAVs matching base_name into 6s chunks, delete originals."""
    all_wavs = sorted([
        f for f in os.listdir(output_dir)
        if f.endswith(".wav") and not f.startswith("._")
        and not f.startswith("s_")  # already a chunk — skip
    ])
    wavs = [f for f in all_wavs if f.startswith(base_name)]
    if not wavs:
        return
    print(f"  \u2702  Slicing {len(wavs)} full WAV(s) into {CHUNK_SECONDS}s chunks …")
    total_chunks = 0
    for fname in wavs:
        full = os.path.join(output_dir, fname)
        n = slice_wav_to_chunks(full, output_dir, chunk_seconds=CHUNK_SECONDS)
        total_chunks += n
        os.remove(full)
    print(f"  \u2713 Sliced \u2192 {total_chunks} chunks, full WAVs deleted")


# ── Per-date pipeline ──────────────────────────────────────

def _collect_minute_dirs(all_results: List[dict], run_sa: bool) -> List[Tuple[str, str, str]]:
    """Collect all (directory, label, step_name) tuples for BirdNET pass.

    Uses PREFILTER_GROUPS from config — BirdNET runs on the post-pre-filter
    target directories, not the raw beamforming per-IR-type directories.
    """
    tasks: List[Tuple[str, str, str]] = []

    # BF directories: one per PREFILTER_GROUP
    for group_name, group_cfg in PREFILTER_GROUPS.items():
        prefix = group_cfg["target_dir_prefix"]
        seen = set()
        for r in all_results:
            key = (r["hour"], r["minute"])
            if key in seen:
                continue
            seen.add(key)
            path = build_output_path(r.get("location", ""), r.get("date", ""),
                                     prefix, r["hour"], r["minute"])
            if os.path.isdir(path):
                step = f"{STEP_BIRNET_PREFIX}{group_name}"
                tasks.append((path, prefix, step))

    if run_sa:
        seen = set()
        for r in all_results:
            key = (r["hour"], r["minute"])
            if key in seen:
                continue
            seen.add(key)
            path = build_output_path(r.get("location", ""), r.get("date", ""),
                                     "sa", r["hour"], r["minute"])
            if os.path.isdir(path):
                tasks.append((path, "sa", STEP_BIRNET_SA))

    seen = set()
    for r in all_results:
        key = (r["hour"], r["minute"])
        if key in seen:
            continue
        seen.add(key)
        path = build_output_path(r.get("location", ""), r.get("date", ""),
                                 "mono", r["hour"], r["minute"])
        if os.path.isdir(path):
            tasks.append((path, "mono", STEP_BIRNET_MONO))

    return tasks


def process_date(
    flac_paths: List[str],
    location_name: str, date_str: str,
    ir_types: List[str], run_sa: bool = True, run_birdnet: bool = True,
    cleanup: bool = False, dry_run: bool = False,
    use_prototype_subsets: bool = False, force_bf: bool = False,
    force_birdnet: bool = False,
    state: Optional["PipelineState"] = None,
) -> dict:
    """Process all FLACs for one date end-to-end.

    Fase 1:   Beamforming + slice + SA + mono for every FLAC
    Fase 1.5: RMS-energy pre-filtering (pool chunks across IR groups,
              keep only those with RMS ≥ threshold × max RMS)
    Fase 2:   BirdNET per-minute per-method
    Fase 3:   processed.json + cleanup per-minute per-method
    """
    n_flacs = len(flac_paths)
    print(f"\n{'#'*60}")
    print(f"# \U0001f4c5 Date: {date_str}  ({n_flacs} FLACs)")
    print(f"{'#'*60}")

    t_start = time.time()

    # ── Fase 1 ──────────────────────────────────────────────
    print(f"\n── Fase 1: Beamforming + SA + Mono ({n_flacs} FLACs) ──\n")
    all_results = []
    for i, flac_path in enumerate(flac_paths, 1):
        print(f"\n[{i}/{n_flacs}]")
        result = process_one_flac(
            flac_path=flac_path, location_name=location_name, date_str=date_str,
            ir_types=ir_types, run_sa=run_sa,
            use_prototype_subsets=use_prototype_subsets, force_bf=force_bf,
            state=state,
        )
        # Attach location/date for reuse in BirdNET phase
        result["location"] = location_name
        result["date"] = date_str
        all_results.append(result)

    # ── Fase 1.5: Pre-filtering by RMS energy ──────────────
    # Pool chunks across IR types within each prefilter group,
    # keep only those with RMS >= threshold of max RMS in the group × minute.
    # This drastically reduces the number of files sent to BirdNET.
    print(f"\n── Pre-filter (RMS energy threshold: {int(PREFILTER_RMS_THRESHOLD * 100)}% of max) ──")
    pf_start = time.time()
    pf_kept_total = 0
    pf_del_total = 0

    # Gather unique (hour, minute, location, date) combos
    minute_keys: List[Tuple[str, str, str, str]] = []
    seen_mk = set()
    for r in all_results:
        mk = (r["hour"], r["minute"], r.get("location", ""), r.get("date", ""))
        if mk not in seen_mk:
            seen_mk.add(mk)
            minute_keys.append(mk)

    for group_name, group_cfg in PREFILTER_GROUPS.items():
        prefix = group_cfg["target_dir_prefix"]
        source_ir_names = group_cfg["sources"]

        for hour_str, minute_str, loc, dt in minute_keys:
            source_dirs = [
                build_output_path(loc, dt, f"bf_{irn}", hour_str, minute_str)
                for irn in source_ir_names
            ]
            target_dir = build_output_path(loc, dt, prefix, hour_str, minute_str)

            # Force re-run: delete results.json, processed.json.
            # Also clean old chunk WAVs from target dir UNLESS target is also
            # a source dir (in-place case → chunks were already cleaned by force_bf).
            results_json = os.path.join(target_dir, "results.json")
            processed_json = os.path.join(target_dir, "processed.json")
            if force_birdnet:
                for f in [results_json, processed_json]:
                    if os.path.isfile(f):
                        os.remove(f)
                        print(f"  \U0001f5d1  Deleted {os.path.basename(f)} [{group_name}]")
                # Only clean chunks if target != any source dir (merge case)
                target_abs = os.path.abspath(target_dir)
                is_in_place = any(os.path.abspath(sd) == target_abs for sd in source_dirs if os.path.isdir(sd))
                if not is_in_place and os.path.isdir(target_dir):
                    removed = 0
                    for fname in list(os.listdir(target_dir)):
                        if fname.lower().endswith(".wav") and not fname.startswith("._"):
                            try:
                                os.remove(os.path.join(target_dir, fname))
                                removed += 1
                            except OSError:
                                pass
                    if removed > 0:
                        print(f"  \U0001f5d1  Cleaned {removed} old chunk WAV(s) from {prefix}")
            elif os.path.isfile(results_json) and os.path.isfile(processed_json):
                print(f"  \u2713 Pre-filter [{group_name}] h_{hour_str}/m_{minute_str} already fully processed — skipping")
                continue

            # If only one source and same as target → in-place
            if len(source_dirs) == 1 and source_dirs[0] == target_dir:
                if os.path.isdir(target_dir):
                    k, d = prefilter_directory(target_dir, threshold_ratio=PREFILTER_RMS_THRESHOLD, dry_run=dry_run)
                    pf_kept_total += k
                    pf_del_total += d
            else:
                k, d = prefilter_merged(
                    source_dirs, target_dir,
                    threshold_ratio=PREFILTER_RMS_THRESHOLD, dry_run=dry_run,
                )
                pf_kept_total += k
                pf_del_total += d

            # Mark prefilter step complete in state
            if state and not dry_run:
                for r in all_results:
                    if r["hour"] == hour_str and r["minute"] == minute_str:
                        state.mark_complete(
                            state.make_key(loc, dt, f"h_{hour_str}", r["base_name"]),
                            f"{STEP_PREFILTER_PREFIX}{group_name}",
                        )
                        break

    print(f"  \u23f1  Pre-filter: {time.time() - pf_start:.1f}s "
          f"({pf_kept_total} kept, {pf_del_total} deleted across all groups)")

    # ── Fase 2 + 3: BirdNET + processed.json ──────────────
    if run_birdnet:
        recording_date = parse_flac_date(flac_paths[0], date_str)
        # Way Canguk → custom species list (no lat/lon).
        # Other sites → geo lat/lon only. birdnetlib forbids combining both.
        species_list_path, lat, lon, filter_mode = resolve_birdnet_filter(location_name)
        print(f"\n  \U0001f426 BirdNET filter [{location_name}]: {filter_mode}")
        if species_list_path:
            print(f"     custom list: {species_list_path}")
        if lat is not None and lon is not None:
            print(f"     geo: lat={lat}, lon={lon}")

        tasks = _collect_minute_dirs(all_results, run_sa)

        # Filter: skip if results.json + processed.json already exist (unless forced)
        pending = []
        for directory, label, step_name in tasks:
            results_json = os.path.join(directory, "results.json")
            processed_json = os.path.join(directory, "processed.json")
            if force_birdnet:
                for f in [results_json, processed_json]:
                    if os.path.isfile(f):
                        os.remove(f)
                        print(f"  \U0001f5d1  Deleted {os.path.basename(f)} [{label}]")
            elif os.path.isfile(results_json) and os.path.isfile(processed_json):
                print(f"  \u2713 BirdNET [{label}] already done — skipping {directory}")
                continue
            pending.append((directory, label, step_name))

        if pending:
            print(f"\n  \U0001f680 BirdNET: {len(pending)} directories, "
                  f"{min(BIRDNET_PARALLEL_DIRS, len(pending))} concurrent")

            t0 = time.time()
            n_workers = min(BIRDNET_PARALLEL_DIRS, len(pending))
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {}
                for directory, label, step_name in pending:
                    fut = pool.submit(
                        process_directory_pipeline,
                        directory=directory, date=recording_date,
                        cleanup=(cleanup and not label.startswith("sa")
                                 and label != "mono"),
                        dry_run=dry_run,
                        lat=lat, lon=lon,
                        species_list_path=species_list_path,
                    )
                    futures[fut] = (label, step_name)

                for future in as_completed(futures):
                    label, step_name = futures[future]
                    try:
                        results_path, processed_path, deleted = future.result()
                        print(f"    \u2705 BirdNET [{label}] done"
                              + (f" ({deleted} chunks deleted)" if deleted else ""))
                    except Exception as e:
                        print(f"    \u274c BirdNET [{label}] failed: {e}")

            print(f"    \u23f1  BirdNET: {time.time() - t0:.1f}s")

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"\u2705 Date {date_str} done in {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"{'='*60}")

    return {
        "date": date_str,
        "n_flacs": n_flacs,
        "elapsed": elapsed,
        "files": [
            {"name": r["base_name"], "elapsed_s": round(r["elapsed"], 1)}
            for r in all_results
        ],
    }


# ── Main ───────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Spatial Ecoacoustic Analysis Pipeline"
    )
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
    parser.add_argument("--force-bf", action="store_true")
    parser.add_argument("--force-birdnet", action="store_true")
    parser.add_argument("--reset-state", action="store_true")
    args = parser.parse_args()

    # ── Resolve location ────────────────────────────────────
    if args.location:
        location_name = args.location
    elif args.rpiid:
        location_name = get_location_name_from_rpiid(args.rpiid)
    else:
        # Auto-detect from MONITORING_DATA
        rpiid = os.path.basename(MONITORING_DATA.rstrip("/"))
        location_name = get_location_name_from_rpiid(rpiid)

    # ── Resolve RPiID ───────────────────────────────────────
    rpiid = args.rpiid or LOCATION_MAP.get(location_name, "")
    if not rpiid:
        # Reverse lookup
        for rid, loc in LOCATION_MAP.items():
            if loc == location_name:
                rpiid = rid
                break
    if not rpiid:
        print(f"\u274c Could not resolve RPiID for location '{location_name}'")
        sys.exit(1)

    print(f"\U0001f3af Location: {location_name}   RPiID: {rpiid}")
    print()

    # ── Pipeline state ──────────────────────────────────────
    state = PipelineState()
    if args.reset_state:
        state.reset_key(None)
    print(state.summary())

    # ── IR config ───────────────────────────────────────────
    use_prototype = True  # Always use production subsets
    ir_types = [t.strip() for t in args.ir_types.split(",") if t.strip()]
    valid = set(IR_TYPES)
    ir_types = [t for t in ir_types if t in valid]
    if not ir_types:
        print(f"\u274c No valid IR types (choose from {sorted(valid)})")
        sys.exit(1)

    # ── Collect dates ───────────────────────────────────────
    if args.full:
        # Process all dates in MONITORING_DATA for this RPiID
        rpiid_dir = os.path.join(MONITORING_DATA, rpiid)
        dates = sorted([
            d for d in os.listdir(rpiid_dir)
            if os.path.isdir(os.path.join(rpiid_dir, d))
            and not d.startswith(".")
        ])
    elif args.date:
        dates = [args.date]
    else:
        # Default: today
        dates = [datetime.now().strftime("%Y-%m-%d")]

    print(f"\U0001f4c5 Dates to process: {dates}")

    # ── Process each date ───────────────────────────────────
    grand_start = time.time()

    for date_str in dates:
        flac_paths = get_flac_files(rpiid, date_str, max_files=args.max_files)
        if not flac_paths:
            print(f"\u26a0  No FLAC files for {date_str} — skipping")
            continue

        result = process_date(
            flac_paths=flac_paths,
            location_name=location_name, date_str=date_str,
            ir_types=ir_types, run_sa=not args.no_sa,
            run_birdnet=not args.no_birdnet,
            cleanup=args.cleanup, dry_run=args.dry_run,
            use_prototype_subsets=use_prototype, force_bf=args.force_bf,
            force_birdnet=args.force_birdnet,
            state=state,
        )
        print(f"\n  \u2705 {result['n_flacs']} FLAC(s) for {date_str}"
              f" in {result['elapsed']:.0f}s")

    print(f"\n{'='*60}")
    print(f"\U0001f389 All done  —  total {time.time() - grand_start:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
