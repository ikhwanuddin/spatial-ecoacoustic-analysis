#!/usr/bin/env python3
"""
Signal-processing pipeline for MAARU monitoring recordings.

Converts raw FLAC recordings into the signal methods used by the downstream
experiments. This script deliberately stops after signal processing; it does
not import or run BirdNET, bacpipe, embeddings, or species classification.

Per FLAC:
  1. Beamforming: LabIR -> bf_LabIR; SPIR1/SPIR2 -> bf_SPIR
  2. Signal averaging: 6-channel -> 1-channel -> sa
  3. Mono baseline: channel 0 -> mono

Usage:
  # All configured LabIR directions, all FLACs, overwrite every signal output
  python pipeline_signal_processing.py --location 2A400 \
      --date 2026-04-26 --ir-types LabIR,SPIR1,SPIR2 \
      --max-files 0 --force-signal

  # Selected LabIR directions, all FLACs, overwrite every signal output
  # S01/S05/S09: 0,60,120,180,240,300; S12 is automatically direction 0.
  python pipeline_signal_processing.py --location 2A400 \
      --date 2026-04-26 --ir-types LabIR,SPIR1,SPIR2 \
      --labir-speakers S01,S05,S09,S12 \
      --labir-degrees 0,60,120,180,240,300 \
      --max-files 0 --force-signal
"""

import argparse
import os
import re
import time
from dataclasses import replace
from typing import List, Optional, Tuple

import soundfile as sf

from config import (
    ANALYSIS_OUTPUT,
    FS_TARGET,
    IR_TYPES,
    LOCATION_MAP,
    MONITORING_DATA,
    RPIID_TO_LOCATION,
)
from beamforming import Beamformer
from signal_averaging import SignalAverager
from audio_loader import load_audio_robust


_HM_RE = re.compile(r"^(\d{2})-(\d{2})-\d{2}_dur=")


def _parse_int_csv(
    value: Optional[str], option_name: str, prefix: Optional[str] = None
) -> Optional[List[int]]:
    """Parse a comma-separated integer option, optionally accepting a prefix."""
    if value is None:
        return None

    values: List[int] = []
    for raw in value.split(","):
        token = raw.strip()
        if prefix and token.upper().startswith(prefix.upper()):
            token = token[len(prefix):]
        if not token:
            raise ValueError(f"{option_name} contains an empty value")
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ValueError(
                f"{option_name} must contain integers, got {raw!r}"
            ) from exc
        if parsed not in values:
            values.append(parsed)

    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values


def _extract_hour_minute(flac_path: str) -> Tuple[str, str]:
    base = os.path.basename(flac_path)
    match = _HM_RE.match(base)
    if match:
        return match.group(1), match.group(2)
    return "00", "00"


def get_flac_files(rpiid: str, date_str: str, max_files: int = 0) -> List[str]:
    date_dir = os.path.join(MONITORING_DATA, rpiid, date_str)
    if not os.path.isdir(date_dir):
        print(f"❌ Directory not found: {date_dir}")
        return []

    flacs = sorted(
        os.path.join(date_dir, filename)
        for filename in os.listdir(date_dir)
        if filename.lower().endswith(".flac") and not filename.startswith("._")
    )
    if max_files and len(flacs) > max_files:
        flacs = flacs[:max_files]

    print(f"📁 {len(flacs)} FLAC file(s) selected from {date_dir}")
    for flac in flacs:
        print(f"    → {os.path.basename(flac)}")
    return flacs


def build_output_path(
    location_name: str,
    date_str: str,
    processing_type: str,
    hour: str = "",
    minute: str = "",
) -> str:
    path = os.path.join(ANALYSIS_OUTPUT, location_name, date_str, processing_type)
    if hour:
        path = os.path.join(path, f"h_{hour}")
    if minute:
        path = os.path.join(path, f"m_{minute}")
    return path


def _minute_complete(output_dir: str, base_name: str) -> bool:
    """Check whether any beamforming output exists for this FLAC."""
    if not os.path.isdir(output_dir):
        return False
    try:
        return any(
            filename.endswith(".wav")
            and not filename.startswith("._")
            and base_name in filename
            for filename in os.listdir(output_dir)
        )
    except OSError:
        return False


def _sa_output_exists(output_dir: str, base_name: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, base_name + "_sa.wav"))


def _mono_output_exists(output_dir: str, base_name: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, base_name + "_mono.wav"))


def _spir_type_complete(
    bf_spir_dir: str, base_name: str, spir_type: str
) -> bool:
    """Check whether SPIR1 or SPIR2 output exists in bf_SPIR."""
    if not os.path.isdir(bf_spir_dir):
        return False
    pattern = spir_type + "("
    try:
        return any(
            filename.endswith(".wav")
            and not filename.startswith("._")
            and base_name in filename
            and pattern in filename
            for filename in os.listdir(bf_spir_dir)
        )
    except OSError:
        return False


def process_one_flac(
    flac_path: str,
    location_name: str,
    date_str: str,
    ir_types: List[str],
    force_bf: bool = False,
    force_signal: bool = False,
    labir_speakers: Optional[List[int]] = None,
    labir_degrees: Optional[List[int]] = None,
) -> dict:
    """Run beamforming, signal averaging, and mono for one FLAC."""
    base_name = os.path.splitext(os.path.basename(flac_path))[0]
    hour_str, minute_str = _extract_hour_minute(flac_path)

    print(f"\n{'=' * 60}")
    print(f"🎙  Processing: {base_name}")
    print(f"📍 Location: {location_name}")
    print(f"📅 Date:     {date_str}")
    print(f"🕐 Hour:     {hour_str}  Minute: {minute_str}")
    print(f"{'=' * 60}")

    overall_start = time.time()
    bf_dirs: List[Tuple[str, str]] = []
    cleaned_bf_spir = False

    # 1. Resilient audio load with auto-recovery for corrupted/truncated FLACs
    try:
        raw_audio, was_repaired = load_audio_robust(
            flac_path=flac_path,
            target_sr=FS_TARGET,
            expected_channels=6,
        )
    except Exception as exc:
        print(f"  ❌ Unrecoverable audio error on {base_name}: {exc}")
        print(f"  ⏩ Skipping {base_name} to ensure pipeline continues.")
        return None

    for ir_name in ir_types:
        if ir_name not in IR_TYPES:
            print(f"⚠️  Unknown IR type: {ir_name} — skipping")
            continue

        ir_type = IR_TYPES[ir_name]
        if ir_name == "LabIR" and (
            labir_speakers is not None or labir_degrees is not None
        ):
            ir_type = replace(
                ir_type,
                param_values=(
                    labir_speakers
                    if labir_speakers is not None
                    else ir_type.param_values
                ),
                degree_values=(
                    labir_degrees
                    if labir_degrees is not None
                    else ir_type.degree_values
                ),
            )
            print(
                "  LabIR subset: "
                f"speakers={ir_type.param_values}, "
                f"degrees={ir_type.degree_values} "
                "(zenith speakers use degree 0)"
            )

        if ir_name in ("SPIR1", "SPIR2"):
            bf_dir = build_output_path(
                location_name, date_str, "bf_SPIR", hour_str, minute_str
            )
            if not force_bf and _spir_type_complete(bf_dir, base_name, ir_name):
                print(f"  ✓ bf_SPIR ({ir_name}) already exists — skipping")
                if (bf_dir, "SPIR") not in bf_dirs:
                    bf_dirs.append((bf_dir, "SPIR"))
                continue

            # Clean bf_SPIR once before SPIR1. Do not clean again before SPIR2.
            if force_bf and not cleaned_bf_spir:
                removed = 0
                if os.path.isdir(bf_dir):
                    for filename in list(os.listdir(bf_dir)):
                        if filename.endswith(".wav") and not filename.startswith("._"):
                            try:
                                os.remove(os.path.join(bf_dir, filename))
                                removed += 1
                            except OSError:
                                pass
                if removed:
                    print(f"  🗑  Cleaned {removed} old WAV(s) from bf_SPIR")
                cleaned_bf_spir = True
            label = "SPIR"
        else:
            bf_dir = build_output_path(
                location_name, date_str, f"bf_{ir_name}", hour_str, minute_str
            )
            label = ir_name
            if not force_bf and _minute_complete(bf_dir, base_name):
                print(f"  ✓ bf_{ir_name} already exists — skipping")
                bf_dirs.append((bf_dir, ir_name))
                continue
            if force_bf and os.path.isdir(bf_dir):
                removed = 0
                for filename in list(os.listdir(bf_dir)):
                    if filename.endswith(".wav") and not filename.startswith("._"):
                        try:
                            os.remove(os.path.join(bf_dir, filename))
                            removed += 1
                        except OSError:
                            pass
                if removed:
                    print(f"  🗑  Cleaned {removed} old WAV(s) from bf_{ir_name}")

        if (bf_dir, label) not in bf_dirs:
            bf_dirs.append((bf_dir, label))

        print(f"\n── Beamforming [{ir_name}] → {bf_dir} ──")
        try:
            beamformer = Beamformer(
                flac_path=flac_path,
                output_dir=bf_dir,
                ir_type_or_name=ir_type,
                raw_audio=raw_audio,
            )
            beamformer.run()
        except Exception as exc:
            print(f"  ❌ Beamforming [{ir_name}] failed on {base_name}: {exc} (skipping this IR)")

    sa_dir = build_output_path(location_name, date_str, "sa", hour_str, minute_str)
    print(f"\n── Signal Averaging → {sa_dir} ──")
    if not force_signal and _sa_output_exists(sa_dir, base_name):
        print("  ✓ SA already exists — skipping")
    else:
        if force_signal and _sa_output_exists(sa_dir, base_name):
            print("  🗑  Overwriting existing SA output")
        try:
            SignalAverager(flac_path=flac_path, output_dir=sa_dir, raw_audio=raw_audio).run()
        except Exception as exc:
            print(f"  ❌ Signal averaging failed on {base_name}: {exc}")

    mono_dir = build_output_path(location_name, date_str, "mono", hour_str, minute_str)
    mono_file = os.path.join(mono_dir, base_name + "_mono.wav")
    print(f"\n── Mono Baseline → {mono_dir} ──")
    if not force_signal and _mono_output_exists(mono_dir, base_name):
        print("  ✓ Mono baseline already exists — skipping")
    else:
        if force_signal and _mono_output_exists(mono_dir, base_name):
            print("  🗑  Overwriting existing mono output")
        try:
            os.makedirs(mono_dir, exist_ok=True)
            ch0 = raw_audio[0, :].copy() if raw_audio.ndim > 1 else raw_audio.copy()
            amplitude = max(abs(ch0))
            if amplitude > 1.0:
                ch0 = ch0 / amplitude
            sf.write(
                mono_file,
                (ch0 * 32767).clip(-32768, 32767).astype("int16"),
                FS_TARGET,
                subtype="PCM_16",
            )
            print(f"  ✓ Mono baseline: {mono_file}")
        except Exception as exc:
            print(f"  ❌ Mono baseline failed: {exc}")

    elapsed = time.time() - overall_start
    print(f"\n{'=' * 60}")
    print(f"✅ Done — {base_name} in {elapsed:.1f}s")
    print(f"{'=' * 60}")

    return {
        "flac": flac_path,
        "base_name": base_name,
        "hour": hour_str,
        "minute": minute_str,
        "beamforming_dirs": [directory for directory, _ in bf_dirs],
        "sa_dir": sa_dir,
        "mono_dir": mono_dir,
        "elapsed": elapsed,
        "was_repaired": was_repaired,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Signal processing: FLAC → beamforming + SA + mono"
    )
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--date", type=str, required=True, help="Date(s), comma-separated")
    parser.add_argument(
        "--ir-types",
        type=str,
        default="LabIR,SPIR1,SPIR2",
        help="IR types for beamforming (default: LabIR,SPIR1,SPIR2)",
    )
    parser.add_argument(
        "--labir-speakers",
        type=str,
        default=None,
        help="LabIR speakers, e.g. S01,S05,S09,S12 (default: all configured)",
    )
    parser.add_argument(
        "--labir-degrees",
        type=str,
        default=None,
        help="LabIR azimuths, e.g. 0,60,120,180,240,300 (default: all configured)",
    )
    parser.add_argument("--max-files", type=int, default=0, help="Max FLACs per date (0=all)")
    parser.add_argument(
        "--force-bf",
        action="store_true",
        help="Regenerate beamforming outputs even if they exist",
    )
    parser.add_argument(
        "--force-signal",
        action="store_true",
        help="Overwrite and regenerate BF, SA, and mono outputs",
    )
    args = parser.parse_args()

    try:
        labir_speakers = _parse_int_csv(
            args.labir_speakers, "--labir-speakers", prefix="S"
        )
        labir_degrees = _parse_int_csv(args.labir_degrees, "--labir-degrees")
    except ValueError as exc:
        parser.error(str(exc))

    if labir_speakers and any(speaker < 1 or speaker > 12 for speaker in labir_speakers):
        parser.error("--labir-speakers values must be between S01 and S12")
    if labir_degrees and any(degree < 0 or degree >= 360 for degree in labir_degrees):
        parser.error("--labir-degrees values must be between 0 and 359")

    rpiid = LOCATION_MAP.get(args.location, args.location)
    location_name = RPIID_TO_LOCATION.get(rpiid, rpiid)
    dates = [date.strip() for date in args.date.split(",")]
    ir_types = [ir_type.strip() for ir_type in args.ir_types.split(",")]

    print(f"🎯 Location: {location_name}  RPiID: {rpiid}")
    print(f"📅 Dates: {dates}")
    print(f"🎙  IR types: {ir_types}")
    if labir_speakers is not None:
        print(f"LabIR speakers: {labir_speakers}")
    if labir_degrees is not None:
        print(f"LabIR degrees:  {labir_degrees} (S12 uses degree 0)")
    print(f"📦 Signal output: {ANALYSIS_OUTPUT}")
    print()

    grand_start = time.time()
    for date_str in dates:
        print(f"\n{'#' * 60}")
        print(f"# 📅 Date: {date_str}")
        print(f"{'#' * 60}")

        flac_paths = get_flac_files(rpiid, date_str, max_files=args.max_files)
        if not flac_paths:
            print(f"⚠️  No FLAC files for {date_str} — skipping\n")
            continue

        print(f"\n── Signal Processing ({len(flac_paths)} FLACs) ──")
        start = time.time()
        failed_flacs = []
        for index, flac_path in enumerate(flac_paths, 1):
            print(f"\n[{index}/{len(flac_paths)}]")
            try:
                process_one_flac(
                    flac_path=flac_path,
                    location_name=location_name,
                    date_str=date_str,
                    ir_types=ir_types,
                    force_bf=args.force_bf or args.force_signal,
                    force_signal=args.force_signal,
                    labir_speakers=labir_speakers,
                    labir_degrees=labir_degrees,
                )
            except Exception as exc:
                print(f"❌ Error processing {os.path.basename(flac_path)}: {exc} (skipping)")
                failed_flacs.append((os.path.basename(flac_path), str(exc)))

        print(f"  ⏱  Signal processing: {time.time() - start:.1f}s")
        if failed_flacs:
            print(f"\n⚠️  {len(failed_flacs)} FLAC file(s) failed / were corrupted and skipped:")
            for fname, err in failed_flacs:
                print(f"    → {fname}: {err}")

    print(f"\n{'=' * 60}")
    print(f"🎉 All done — total {time.time() - grand_start:.0f}s")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
