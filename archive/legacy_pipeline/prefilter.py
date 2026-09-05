"""
RMS-energy pre-filtering for beamforming chunk outputs.

After beamforming + 6s chunk slicing, this module computes the RMS energy
of every chunk WAV within an IR-type group and keeps only those with
RMS >= threshold_ratio * max RMS in that group.

Two modes:
  - prefilter_directory:      filter chunks in-place within one directory
  - prefilter_merged:         pool chunks from multiple source directories,
                              compute global threshold, move survivors to a
                              unified target directory, delete source dirs.

Motivation: multidirectional beamforming produces many outputs per minute,
but only a subset of directions contain meaningful signal.  Pre-filtering
by RMS energy reduces the number of files sent to BirdNET (the bottleneck)
by 50-90% while retaining the strongest directional signals.
"""

import json
import os
import shutil
import numpy as np
import soundfile as sf
from typing import List, Tuple

from config import PREFILTER_RMS_THRESHOLD


def compute_rms(wav_path: str) -> float:
    """Compute root-mean-square energy of a mono WAV file."""
    audio, _ = sf.read(wav_path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio[:, 0]
    return float(np.sqrt(np.mean(audio ** 2)))


def prefilter_directory(
    directory: str,
    threshold_ratio: float = PREFILTER_RMS_THRESHOLD,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Filter chunk WAVs within *directory*, keeping only those
    whose RMS >= threshold_ratio * max RMS in that directory.

    Returns:
        (kept, deleted) — number of files kept and deleted.
    """
    if not os.path.isdir(directory):
        return 0, 0

    wav_files = sorted([
        f for f in os.listdir(directory)
        if f.lower().endswith(".wav") and f.startswith("s_") and not f.startswith("._")
    ])
    if len(wav_files) <= 1:
        return len(wav_files), 0

    # Compute RMS for every chunk
    rms_map: List[Tuple[str, float]] = []
    for fname in wav_files:
        full = os.path.join(directory, fname)
        try:
            rms = compute_rms(full)
        except Exception:
            rms = 0.0
        rms_map.append((fname, rms))

    max_rms = max(r for _, r in rms_map)
    if max_rms <= 0:
        # All silent — keep everything (degenerate case)
        return len(wav_files), 0

    threshold = threshold_ratio * max_rms

    kept = 0
    deleted = 0
    file_records: list = []
    for fname, rms in rms_map:
        keep = rms >= threshold
        file_records.append({"filename": fname, "rms": round(rms, 6), "kept": keep})
        if keep:
            kept += 1
        else:
            full = os.path.join(directory, fname)
            try:
                if not dry_run:
                    os.remove(full)
                deleted += 1
            except OSError:
                pass

    if not dry_run:
        _print_prefilter_summary(directory, len(wav_files), kept, deleted, threshold_ratio)
        _write_prefilter_log(directory, threshold_ratio, max_rms, threshold,
                            len(wav_files), kept, deleted, file_records)
    else:
        print(f"  [DRY RUN] {directory}: would keep {kept}/{len(wav_files)}, "
              f"delete {deleted}")

    return kept, deleted


def prefilter_merged(
    source_dirs: List[str],
    target_dir: str,
    threshold_ratio: float = PREFILTER_RMS_THRESHOLD,
    dry_run: bool = False,
) -> Tuple[int, int]:
    """Pool chunk WAVs from multiple *source_dirs*, compute a single
    global max RMS, keep only chunks with RMS >= threshold_ratio * global_max,
    move survivors to *target_dir*, then delete source directories.

    When *source_dirs* is a single directory and equals *target_dir*,
    this degenerates to in-place filtering (delete-only, no move).

    Returns:
        (kept, deleted) — total across all source dirs.
    """
    # Collect all chunks with their source directory
    entries: List[Tuple[str, str, float]] = []  # (fname, source_dir, rms)
    for src_dir in source_dirs:
        if not os.path.isdir(src_dir):
            continue
        wav_files = sorted([
            f for f in os.listdir(src_dir)
            if f.lower().endswith(".wav") and f.startswith("s_") and not f.startswith("._")
        ])
        for fname in wav_files:
            full = os.path.join(src_dir, fname)
            try:
                rms = compute_rms(full)
            except Exception:
                rms = 0.0
            entries.append((fname, src_dir, rms))

    if not entries:
        return 0, 0

    max_rms = max(r for _, _, r in entries)
    if max_rms <= 0:
        # All silent — keep everything
        threshold = -1.0
    else:
        threshold = threshold_ratio * max_rms

    # Detect in-place mode (single source == target)
    in_place = (len(source_dirs) == 1 and os.path.abspath(source_dirs[0]) == os.path.abspath(target_dir))

    if not in_place:
        os.makedirs(target_dir, exist_ok=True)

    total = len(entries)
    kept = 0
    deleted = 0
    file_records: list = []
    for fname, src_dir, rms in entries:
        src_path = os.path.join(src_dir, fname)
        dst_path = os.path.join(target_dir, fname)
        keep = rms >= threshold
        file_records.append({"filename": fname, "rms": round(rms, 6), "kept": keep})
        if keep:
            if not dry_run:
                if not in_place:
                    shutil.move(src_path, dst_path)
            kept += 1
        else:
            if not dry_run:
                try:
                    os.remove(src_path)
                except OSError:
                    pass
            deleted += 1

    # Remove now-empty source directories (and their parent h_XX/, m_YY/ chain)
    if not dry_run:
        for src_dir in source_dirs:
            if in_place:
                continue  # don't delete target dir
            _remove_empty_dir_chain(src_dir)

    if not dry_run:
        _print_prefilter_summary(target_dir, total, kept, deleted, threshold_ratio)
        _write_prefilter_log(target_dir, threshold_ratio, max_rms, threshold,
                            total, kept, deleted, file_records)
    else:
        print(f"  [DRY RUN] merge → {target_dir}: would keep {kept}/{total}, "
              f"delete {deleted}")

    return kept, deleted


# ── helpers ───────────────────────────────────────────────────

def _print_prefilter_summary(directory: str, total: int, kept: int, deleted: int, threshold_ratio: float = 0.5):
    pct = (kept / total * 100) if total > 0 else 0
    # Use a short label
    label = os.path.basename(directory) or directory
    pct_thresh = int(threshold_ratio * 100)
    print(f"  \u2713 Pre-filter [{label}]: {kept}/{total} kept ({pct:.0f}%), "
          f"{deleted} deleted  (threshold \u2265 {pct_thresh}% max RMS)")


def _remove_empty_dir_chain(directory: str):
    """Remove *directory* if empty, then walk up removing empty parents
    up to (but not including) the date-level directory (pattern: YYYY-MM-DD)."""
    import re
    _date_re = re.compile(r"^\d{4}-\d{2}-\d{2}$")

    d = os.path.abspath(directory)
    while d and os.path.isdir(d):
        try:
            entries = os.listdir(d)
            # Also ignore ._ dotfiles
            real = [e for e in entries if not e.startswith("._")]
            if real:
                break
            os.rmdir(d)
        except OSError:
            break
        parent = os.path.dirname(d)
        # Stop at date-level or analysis output root
        if _date_re.match(os.path.basename(d)):
            break
        d = parent


def _write_prefilter_log(
    directory: str,
    threshold_ratio: float,
    max_rms: float,
    threshold: float,
    total: int,
    kept: int,
    deleted: int,
    file_records: list,
) -> None:
    """Write a human-readable JSON log of the pre-filter run.

    Saved as ``prefilter_log.json`` in *directory* so the user can
    later inspect RMS distributions, which directions survived, etc.
    """
    import datetime
    log = {
        "generated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "threshold_ratio": threshold_ratio,
        "max_rms": round(max_rms, 6),
        "threshold": round(threshold, 6),
        "total": total,
        "kept": kept,
        "deleted": deleted,
        "files": file_records,
    }
    path = os.path.join(directory, "prefilter_log.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
