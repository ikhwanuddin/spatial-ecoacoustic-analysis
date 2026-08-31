#!/usr/bin/env python3
"""Build official noise references for one date, one file per stream per condition.

Policy
------
1. Detection runs only on the reference beam ``LabIR(S05_000)`` of every recording
   in the date. Recordings are binned into dawn / day / dusk / night.
2. Every accepted 2 s window is kept, not just the best one. The detector hops 1 s
   with a 2 s window, so accepted windows overlap and some audio repeats in the
   concatenation. For a background-noise profile that repetition is harmless, so it
   is kept by default; ``--no-overlap`` drops it. The manifest always reports both
   the concatenated length and the unique wall-clock length.
3. The intervals accepted on the reference beam are the *only* source of truth.
   The identical (recording, start, end) triples are sliced from every other
   stream: all LabIR beams, all SPIR beams, the 4-channel SA, and mono.
4. All slices of one beam are concatenated into a single file holding background
   noise only, so each stream ends up with one long reference per condition.
   A short edge fade is applied to each slice so the joins do not click.

Every stream therefore covers exactly the same instants of time, and a method is
only ever paired with a reference built from that same method.

Output:
  <sea-root>/<location>/<date>/noise_references/<condition>/
      LabIR/<condition>_<beam>_noise.wav      one per LabIR beam
      SPIR/<condition>_<beam>_noise.wav       one per SPIR beam
      sa/<condition>_sa_noise.wav
      mono/<condition>_mono_noise.wav
      manifest.json
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

REFERENCE_BEAM = "LabIR(S05_000)"
STREAMS = ("LabIR", "SPIR", "sa", "mono")

# local time bins
CONDITION_BINS = (("dawn", 5, 7), ("day", 7, 17), ("dusk", 17, 19))


def condition_of(hour: int) -> str:
    for name, lo, hi in CONDITION_BINS:
        if lo <= hour < hi:
            return name
    return "night"


def hour_of(wav: Path) -> Optional[int]:
    m = re.match(r"(\d{2})-(\d{2})-(\d{2})_", wav.name)
    return int(m.group(1)) if m else None


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def beam_tag(name: str) -> Optional[str]:
    """Beam identifier used to pair the same direction across recordings."""
    m = re.search(r"(LabIR\(S\d{2}_\d{3}\)|SPIR[12]\([^)]*\))", name)
    if not m:
        return None
    tag = m.group(1)
    inner = re.search(r"LabIR\((S\d{2}_\d{3})\)", tag)
    return inner.group(1) if inner else tag


# ── detection ────────────────────────────────────────────────────────────────

def run_detector(detector: Path, python: str, wav: Path, out_dir: Path,
                 window_sec: float, hop_sec: float) -> Path:
    json_path = out_dir / f"{safe(wav.stem)}_temporal_noise_detection.json"
    if json_path.is_file():
        print(f"    reuse  {json_path.name}")
        return json_path
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [python, str(detector), "--input", str(wav), "--output-dir", str(out_dir),
           "--window-sec", str(window_sec), "--hop-sec", str(hop_sec)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"detector failed on {wav.name}:\n{proc.stderr[-800:]}")
    print(f"    {proc.stdout.strip().splitlines()[0] if proc.stdout.strip() else 'done'}")
    return json_path


def accepted_intervals(json_path: Path, drop_overlap: bool = False) -> List[dict]:
    """Accepted windows in time order; optionally de-overlapped by best score."""
    result = json.loads(json_path.read_text())
    accepted = [w for w in result.get("windows", []) if w.get("candidate")]
    if not drop_overlap:
        return sorted(accepted, key=lambda w: w["start_sec"])
    accepted.sort(key=lambda w: w.get("background_score", 0.0), reverse=True)
    kept: List[dict] = []
    for w in accepted:
        if any(w["start_sec"] < k["end_sec"] and k["start_sec"] < w["end_sec"] for k in kept):
            continue
        kept.append(w)
    kept.sort(key=lambda w: w["start_sec"])
    return kept


def unique_seconds(intervals: List[dict]) -> float:
    """Wall-clock seconds covered, counting overlapping windows only once."""
    merged: List[List[float]] = []
    for w in sorted(intervals, key=lambda w: w["start_sec"]):
        if merged and w["start_sec"] <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], w["end_sec"])
        else:
            merged.append([w["start_sec"], w["end_sec"]])
    return sum(end - start for start, end in merged)


# ── slicing and concatenation ────────────────────────────────────────────────

def read_slice(wav: Path, start_sec: float, end_sec: float, fade_ms: float
               ) -> Tuple[np.ndarray, int]:
    with sf.SoundFile(str(wav)) as handle:
        sr = int(handle.samplerate)
        handle.seek(round(start_sec * sr))
        data = handle.read(round((end_sec - start_sec) * sr), dtype="float32", always_2d=True)
    n_fade = min(int(sr * fade_ms / 1000.0), len(data) // 2)
    if n_fade > 0:
        ramp = np.linspace(0.0, 1.0, n_fade, dtype=np.float32)[:, None]
        data[:n_fade] *= ramp
        data[-n_fade:] *= ramp[::-1]
    return data, sr


def write_concat(pieces: List[Tuple[Path, float, float]], dest: Path, fade_ms: float
                 ) -> Optional[float]:
    """Concatenate every (wav, start, end) piece into one file. Returns seconds."""
    chunks, rate = [], None
    for wav, start, end in pieces:
        data, sr = read_slice(wav, start, end, fade_ms)
        if rate is None:
            rate = sr
        elif sr != rate:
            raise ValueError(f"sample-rate mismatch in {wav.name}: {sr} vs {rate}")
        if chunks and data.shape[1] != chunks[0].shape[1]:
            raise ValueError(f"channel mismatch in {wav.name}")
        chunks.append(data)
    if not chunks or rate is None:
        return None
    joined = np.concatenate(chunks, axis=0)
    dest.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(dest), joined if joined.shape[1] > 1 else joined[:, 0], rate)
    return len(joined) / rate


def stream_dir(sea_root: Path, location: str, date: str, stream: str,
               hour: str, minute: str) -> Path:
    folder = {"LabIR": "bf_LabIR", "SPIR": "bf_SPIR", "sa": "sa", "mono": "mono"}[stream]
    return sea_root / location / date / folder / hour / minute


def build_condition(condition: str, records: List[dict], sea_root: Path,
                    location: str, date: str, out_root: Path, fade_ms: float) -> dict:
    """Write one concatenated reference per beam for this condition."""
    out_dir = out_root / condition
    total_sec = sum(w["end_sec"] - w["start_sec"] for r in records for w in r["intervals"])
    uniq_sec = sum(unique_seconds(r["intervals"]) for r in records)
    print(f"\n── {condition}: {len(records)} rekaman, "
          f"{sum(len(r['intervals']) for r in records)} interval, "
          f"{total_sec:.1f} s tersambung ({uniq_sec:.1f} s unik)")

    manifest_files: List[dict] = []
    for stream in STREAMS:
        # beam tag -> ordered list of (wav, start, end)
        pieces: Dict[str, List[Tuple[Path, float, float]]] = defaultdict(list)
        for rec in records:
            src_dir = stream_dir(sea_root, location, date, stream, rec["hour"], rec["minute"])
            if not src_dir.is_dir():
                continue
            for wav in sorted(src_dir.glob("*.wav")):
                tag = beam_tag(wav.name) if stream in ("LabIR", "SPIR") else stream
                if tag is None:
                    continue
                for w in rec["intervals"]:
                    pieces[tag].append((wav, w["start_sec"], w["end_sec"]))

        if not pieces:
            print(f"  {stream:6s}: tidak ada sumber, dilewati")
            continue
        for tag, items in sorted(pieces.items()):
            dest = out_dir / stream / f"{condition}_{safe(tag)}_noise.wav"
            seconds = write_concat(items, dest, fade_ms)
            if seconds:
                manifest_files.append({"stream": stream, "beam": tag,
                                       "file": str(dest), "duration_sec": round(seconds, 3),
                                       "n_segments": len(items)})
        made = [f for f in manifest_files if f["stream"] == stream]
        print(f"  {stream:6s}: {len(made)} file × {made[0]['duration_sec']:.1f} s")

    manifest = {
        "location": location, "date": date, "condition": condition,
        "reference_beam": REFERENCE_BEAM,
        "policy": "intervals detected on the reference beam, sliced identically into every stream",
        "n_recordings": len(records),
        "n_intervals": sum(len(r["intervals"]) for r in records),
        "total_noise_sec": round(total_sec, 3),
        "unique_noise_sec": round(uniq_sec, 3),
        "fade_ms": fade_ms,
        "recordings": [
            {"stem": r["stem"], "hour": r["hour"], "minute": r["minute"],
             "source": str(r["reference_wav"]),
             "intervals": [{"start_sec": w["start_sec"], "end_sec": w["end_sec"],
                            "background_score": w.get("background_score")}
                           for w in r["intervals"]]}
            for r in records
        ],
        "files": manifest_files,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--location", default="2A400")
    parser.add_argument("--date", required=True)
    parser.add_argument("--sea-root", type=Path,
                        default=Path("/rds/general/user/ri322/ephemeral/sea-work"))
    parser.add_argument("--review-root", type=Path,
                        default=Path("/rds/general/user/ri322/ephemeral/sea-work/noise_auto_review"))
    parser.add_argument("--detector", type=Path,
                        default=Path(__file__).resolve().parent / "detect_noise_references_temporal.py")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--target-sec", type=float, default=60.0,
                        help="minimum noise seconds wanted per condition (default 60)")
    parser.add_argument("--max-sec", type=float, default=0.0,
                        help="cap per condition in seconds; 0 keeps every accepted interval")
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--hop-sec", type=float, default=1.0)
    parser.add_argument("--fade-ms", type=float, default=5.0)
    parser.add_argument("--no-overlap", action="store_true",
                        help="keep only non-overlapping windows; by default overlap is kept "
                             "because repetition is harmless in a noise profile")
    args = parser.parse_args()

    date_root = args.sea_root / args.location / args.date
    references = sorted((date_root / "bf_LabIR").rglob(f"*{REFERENCE_BEAM}.wav"))
    if not references:
        sys.exit(f"No {REFERENCE_BEAM} recordings under {date_root / 'bf_LabIR'}")

    print(f"{args.location} {args.date}: {len(references)} rekaman referensi {REFERENCE_BEAM}")

    by_condition: Dict[str, List[dict]] = defaultdict(list)
    for wav in references:
        hour = hour_of(wav)
        if hour is None:
            print(f"  ⚠️  lewati, nama tidak berpola waktu: {wav.name}")
            continue
        condition = condition_of(hour)
        print(f"  {wav.name}  ->  {condition}")
        review_dir = args.review_root / args.location / args.date / condition / safe(wav.stem)
        json_path = run_detector(args.detector, args.python, wav, review_dir,
                                 args.window_sec, args.hop_sec)
        intervals = accepted_intervals(json_path, drop_overlap=args.no_overlap)
        print(f"    {len(intervals)} interval bersih "
              f"({sum(w['end_sec'] - w['start_sec'] for w in intervals):.1f} s tersambung, "
              f"{unique_seconds(intervals):.1f} s unik)")
        if intervals:
            by_condition[condition].append({
                "stem": wav.stem, "reference_wav": wav,
                "hour": wav.parent.parent.name, "minute": wav.parent.name,
                "intervals": intervals,
            })

    if not by_condition:
        sys.exit("Tidak ada interval yang lolos di seluruh rekaman.")

    out_root = date_root / "noise_references"
    summary = {}
    for condition, records in sorted(by_condition.items()):
        if args.max_sec > 0:
            budget = args.max_sec
            for rec in records:
                rec["intervals"].sort(key=lambda w: w.get("background_score", 0), reverse=True)
                kept = []
                for w in rec["intervals"]:
                    if budget <= 0:
                        break
                    kept.append(w)
                    budget -= w["end_sec"] - w["start_sec"]
                rec["intervals"] = sorted(kept, key=lambda w: w["start_sec"])
            records = [r for r in records if r["intervals"]]
        summary[condition] = build_condition(condition, records, args.sea_root,
                                             args.location, args.date, out_root, args.fade_ms)

    print("\n" + "=" * 70)
    for condition, man in sorted(summary.items()):
        total = man["total_noise_sec"]
        mark = "OK " if total >= args.target_sec else "KURANG"
        print(f"{mark} {condition:6s} {total:8.1f} s tersambung / {man['unique_noise_sec']:7.1f} s unik"
              f"  dari {man['n_recordings']} rekaman, {len(man['files'])} file")
    missing = [c for c in ("dawn", "day", "dusk", "night") if c not in summary]
    if missing:
        print(f"not_available: {', '.join(missing)} (tidak ada rekaman pada kondisi itu)")
    print(f"Output: {out_root}")


if __name__ == "__main__":
    main()
