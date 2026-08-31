#!/usr/bin/env python3
"""Generate official noise references across all audio representations (LabIR, SPIR, SA, Mono).

Scientific Policy:
- Source of truth for clean temporal interval is bf_LabIR (derived from detect_noise_references_temporal.py).
- The exact same time window (t_start, t_end) is sliced synchronously across:
  1. bf_LabIR: All 19 speaker beams (S01 to S12, azimuths 000-300).
  2. bf_SPIR: All distance and azimuth beam files (SPIR1 and SPIR2).
  3. sa: 4-channel spatial audio.
  4. mono: 1-channel raw audio.
- Preserves output structure compatible with process_noise_reference.py:
    noise_references/
        <condition>/
            LabIR/
                <stem>_LabIR(Sxx_yyy)_noise.wav
            SPIR/
                <stem>_SPIR(...)_noise.wav
            sa/
                <stem>_sa_noise.wav
            mono/
                <stem>_mono_noise.wav
            manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def slice_wav(input_wav: Path, output_wav: Path, start_sec: float, end_sec: float) -> bool:
    """Slice audio between start_sec and end_sec and write to output_wav."""
    try:
        with sf.SoundFile(str(input_wav)) as audio:
            sr = int(audio.samplerate)
            channels = int(audio.channels)
            start_sample = round(start_sec * sr)
            n_samples = round((end_sec - start_sec) * sr)
            audio.seek(start_sample)
            data = audio.read(n_samples, dtype="float32", always_2d=True)
            
        output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_wav), data if channels > 1 else data[:, 0], sr)
        return True
    except Exception as e:
        print(f"    ⚠️ Error slicing {input_wav.name}: {e}", file=sys.stderr)
        return False


def generate_noise_for_recording(
    location: str,
    date: str,
    condition: str,
    detection_json_path: Path,
    sea_work_root: Path,
    output_base_dir: Path,
    top_n_windows: int = 1,
    chosen_window_idx: Optional[int] = None,
) -> dict:
    """Generate official noise references for a single verified recording."""
    if not detection_json_path.exists():
        raise FileNotFoundError(f"Detection JSON not found: {detection_json_path}")

    with open(detection_json_path) as f:
        meta = json.load(f)

    input_labir_path = Path(meta["input"])
    input_stem = input_labir_path.name.replace(".wav", "")
    
    # Identify hour and minute folder (e.g. h_06/m_27)
    hour = input_labir_path.parent.parent.name
    minute = input_labir_path.parent.name

    # Select candidate window
    windows = meta.get("windows", [])
    candidates = [w for w in windows if w.get("candidate")]
    if not candidates:
        raise ValueError(f"No candidate windows in {detection_json_path.name}")

    if chosen_window_idx is not None:
        selected_windows = [w for w in candidates if w["index"] == chosen_window_idx]
        if not selected_windows:
            raise ValueError(f"Window index {chosen_window_idx} not found among candidates")
    else:
        # Sort by highest background score
        sorted_candidates = sorted(candidates, key=lambda w: w.get("background_score", 0), reverse=True)
        selected_windows = sorted_candidates[:top_n_windows]

    best_w = selected_windows[0]
    t_start = best_w["start_sec"]
    t_end = best_w["end_sec"]
    duration = round(t_end - t_start, 3)

    print(f"\n================================================================================")
    print(f"🎯 Promoting Official Noise Reference: {location} | {date} | {condition.upper()}")
    print(f"   Source Recording: {hour}/{minute} | {input_stem}")
    print(f"   Selected Interval: {t_start:.3f}s - {t_end:.3f}s (duration: {duration}s, score: {best_w.get('background_score', 0):.4f})")
    print(f"================================================================================")

    condition_out = output_base_dir / location / date / "noise_references" / condition
    labir_out = condition_out / "LabIR"
    spir_out = condition_out / "SPIR"
    sa_out = condition_out / "sa"
    mono_out = condition_out / "mono"

    manifest_entries = []

    # 1. Slice all bf_LabIR beams
    labir_dir = sea_work_root / location / date / "bf_LabIR" / hour / minute
    if labir_dir.exists():
        labir_files = sorted(labir_dir.glob("*.wav"))
        print(f"  [1/4] Processing LabIR ({len(labir_files)} beam files)...")
        for f in labir_files:
            # Suffix with _noise.wav as expected by process_noise_reference.py
            clean_name = f.stem
            match = re.search(r"LabIR\((S\d{2}_\d{3})\)", clean_name)
            tag = match.group(1) if match else "S05_000"
            dest = labir_out / f"{input_stem}_{tag}_noise.wav"
            if slice_wav(f, dest, t_start, t_end):
                manifest_entries.append({"stream": "LabIR", "source": str(f), "noise_file": str(dest)})

    # 2. Slice all bf_SPIR beams
    spir_dir = sea_work_root / location / date / "bf_SPIR" / hour / minute
    if spir_dir.exists():
        spir_files = sorted(spir_dir.glob("*.wav"))
        print(f"  [2/4] Processing SPIR ({len(spir_files)} beam files)...")
        for f in spir_files:
            dest = spir_out / f"{f.stem}_noise.wav"
            if slice_wav(f, dest, t_start, t_end):
                manifest_entries.append({"stream": "SPIR", "source": str(f), "noise_file": str(dest)})

    # 3. Slice SA (4-channel)
    sa_dir = sea_work_root / location / date / "sa" / hour / minute
    if sa_dir.exists():
        sa_files = sorted(sa_dir.glob("*.wav"))
        print(f"  [3/4] Processing Spatial Audio ({len(sa_files)} files)...")
        for f in sa_files:
            dest = sa_out / f"{f.stem}_noise.wav"
            if slice_wav(f, dest, t_start, t_end):
                manifest_entries.append({"stream": "sa", "source": str(f), "noise_file": str(dest)})

    # 4. Slice Mono (1-channel)
    mono_dir = sea_work_root / location / date / "mono" / hour / minute
    if mono_dir.exists():
        mono_files = sorted(mono_dir.glob("*.wav"))
        print(f"  [4/4] Processing Mono ({len(mono_files)} files)...")
        for f in mono_files:
            dest = mono_out / f"{f.stem}_noise.wav"
            if slice_wav(f, dest, t_start, t_end):
                manifest_entries.append({"stream": "mono", "source": str(f), "noise_file": str(dest)})

    # Write condition manifest
    manifest_data = {
        "location": location,
        "date": date,
        "condition": condition,
        "source_hour": hour,
        "source_minute": minute,
        "source_labir": str(input_labir_path),
        "start_sec": t_start,
        "end_sec": t_end,
        "duration_sec": duration,
        "background_score": best_w.get("background_score"),
        "foreground_time_fraction": best_w.get("foreground_time_fraction"),
        "max_excess_db": best_w.get("max_excess_db"),
        "total_noise_files_generated": len(manifest_entries),
        "files": manifest_entries,
    }

    manifest_path = condition_out / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"  ✅ Generated {len(manifest_entries)} noise reference files in: {condition_out}")
    print(f"  📄 Manifest: {manifest_path}")
    return manifest_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--location", type=str, default="2A400")
    parser.add_argument("--date", type=str, default="2026-05-15")
    parser.add_argument("--condition", type=str, choices=["dawn", "day", "dusk", "night", "all"], default="all")
    parser.add_argument("--sea-root", type=Path, default=Path("/rds/general/user/ri322/ephemeral/sea-work"))
    parser.add_argument("--review-root", type=Path, default=Path("/rds/general/user/ri322/ephemeral/sea-work/noise_auto_review"))
    parser.add_argument("--output-base", type=Path, default=Path("/rds/general/user/ri322/ephemeral/sea-work"))
    args = parser.parse_args()

    # Pre-selected, manually accepted recordings for 2026-05-15:
    selected_recordings = {
        "dawn": "06-27-09_dur_240secs_LabIR_S05_000_",
        "day": "15-10-06_dur_240secs_LabIR_S05_000_",
        "dusk": "17-58-51_dur_240secs_LabIR_S05_000_",
        "night": "22-23-20_dur_240secs_LabIR_S05_000_",
    }

    conditions_to_run = ["dawn", "day", "dusk", "night"] if args.condition == "all" else [args.condition]

    summary = {}
    for cond in conditions_to_run:
        stem = selected_recordings.get(cond)
        if not stem:
            continue
        json_path = args.review_root / args.location / args.date / cond / stem / f"{stem}_temporal_noise_detection.json"
        if not json_path.exists():
            print(f"⚠️ Detection JSON for {cond} not found: {json_path}", file=sys.stderr)
            continue

        res = generate_noise_for_recording(
            location=args.location,
            date=args.date,
            condition=cond,
            detection_json_path=json_path,
            sea_work_root=args.sea_root,
            output_base_dir=args.output_base,
            top_n_windows=1,
        )
        summary[cond] = res

    print("\n" + "="*80)
    print(f"🎉 All Official Noise References Successfully Created!")
    print(f"Output Directory: {args.output_base / args.location / args.date / 'noise_references'}")
    print("="*80)


if __name__ == "__main__":
    main()
