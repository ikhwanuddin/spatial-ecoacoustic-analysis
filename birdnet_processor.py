"""
BirdNET analysis and confidence-level comparison.

Runs BirdNET on beamforming or signal-averaging output directories,
produces results.json, then processed.json with the best-direction
selection and confidence metrics.
"""

import os
import json

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from birdnetlib.analyzer import Analyzer
from birdnetlib.main import Recording

from config import SITE_COORDS, BIRDNET_MIN_CONF, BIRDNET_OVERLAP


# ============================================================
# BIRDNET ANALYSIS
# ============================================================

def run_birdnet_on_dir(
    directory: str,
    date: Optional[datetime] = None,
    min_conf: float = BIRDNET_MIN_CONF,
    overlap: float = BIRDNET_OVERLAP,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> str:
    """
    Run BirdNET sequentially on all WAV files in a directory.

    Uses individual Recording objects rather than the batch
    DirectoryMultiProcessingAnalyzer to avoid race conditions
    with external HDD files on macOS.

    Args:
        directory: Path containing WAV files
        date:      Recording date
        min_conf:  Minimum confidence threshold
        overlap:   Overlap between segments
        lat, lon:  GPS coordinates for species location filter

    Returns:
        Path to the written results.json
    """
    results_path = os.path.join(directory, "results.json")

    if date is None:
        date = datetime.now()

    analyzer = Analyzer()
    results_dict: Dict = {}

    wav_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav")
    ])

    total = len(wav_files)
    if total == 0:
        print("    ⚠ No WAV files found in directory")
        with open(results_path, "w") as f:
            json.dump({}, f, indent=4)
        return results_path

    print(f"    Processing {total} WAV files sequentially ...")

    for i, wav_path in enumerate(wav_files, 1):
        fname = os.path.basename(wav_path)

        rec_kwargs: dict = {
            "date": date,
            "min_conf": min_conf,
            "overlap": overlap,
        }
        if lat is not None and lon is not None:
            rec_kwargs["lat"] = lat
            rec_kwargs["lon"] = lon

        try:
            rec = Recording(analyzer, wav_path, **rec_kwargs)
            rec.analyze()
            results_dict[fname] = rec.detections
            n_dets = len(rec.detections)
            if n_dets > 0:
                print(f"    [{i:3d}/{total}] {fname}: {n_dets} detections")
            else:
                # Only show every 10th empty result to keep output clean
                if i % 10 == 0 or i == total:
                    print(f"    [{i:3d}/{total}] {fname}: 0 (progress check)")
        except Exception as e:
            print(f"    [{i:3d}/{total}] ⚠ {fname}: {e}")
            results_dict[fname] = []

    # Write results
    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    total_dets = sum(len(v) for v in results_dict.values())
    files_with_dets = sum(1 for v in results_dict.values() if v)
    print(f"    ✓ results.json written ({files_with_dets}/{total} files with detections, {total_dets} total)")

    return results_path


# ============================================================
# CONFIDENCE LEVEL COMPARISON
# ============================================================

def read_results(file_path: str) -> Dict:
    """Read a BirdNET results.json file."""
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  ❌ Error reading {file_path}: {e}")
        return {}


def extract_unique_bf_detections(
    results_dict: Dict,
    conf_thresh: float,
    identifier_pattern: str = "",
) -> List[Dict]:
    """
    For each species × start_time, keep ONLY the detection with
    highest confidence across all beamforming channels.

    Args:
        results_dict:       Loaded results.json
        conf_thresh:        Minimum confidence to consider
        identifier_pattern: Only consider channels matching this pattern
                            (e.g. "_LabIR" for LabIR beamformed files)

    Returns:
        List of unique detections, each with 'primary_channel' set
        to the WAV filename that achieved the highest confidence.
    """
    best_per_key: Dict[str, Dict] = {}

    for channel, detections in results_dict.items():
        if identifier_pattern and identifier_pattern.lower() not in channel.lower():
            continue

        for det in detections:
            conf = det.get("confidence", 0)
            if conf < conf_thresh:
                continue

            species = det.get("common_name", "Unknown")
            start_t = round(det.get("start_time", 0), 1)
            key = f"{species}_{start_t}"

            if key not in best_per_key or conf > best_per_key[key].get("confidence", 0):
                det_copy = det.copy()
                det_copy["start_time"] = start_t
                det_copy["primary_channel"] = channel
                best_per_key[key] = det_copy

    return list(best_per_key.values())


def build_processed(
    results_dict: Dict,
    identifier_pattern: str = "",
    conf_thresh: float = BIRDNET_MIN_CONF,
) -> Dict:
    """
    Build processed.json structure from a BirdNET results.json.

    Returns a dict with per-species confidence lists and metrics.
    """
    detections = extract_unique_bf_detections(results_dict, conf_thresh, identifier_pattern)

    species_data: Dict[str, Dict] = {}
    for det in detections:
        sp = det.get("common_name", "Unknown")
        conf = det.get("confidence", 0)
        start_t = det.get("start_time", 0)
        channel = det.get("primary_channel", "")

        if sp not in species_data:
            species_data[sp] = {
                "conf_list": [],
                "start_time_list": [],
                "primary_channel_list": [],
            }

        species_data[sp]["conf_list"].append(conf)
        species_data[sp]["start_time_list"].append(start_t)
        species_data[sp]["primary_channel_list"].append(channel)

    # Compute metrics
    for sp, sd in species_data.items():
        cl = sd["conf_list"]
        n = len(cl)
        sd.update({
            "count": n,
            "conf_avg": round(float(np.mean(cl)), 3) if n else 0.0,
            "conf_median": round(float(np.median(cl)), 3) if n else 0.0,
            "conf_stdev": round(float(np.std(cl)), 3) if n > 1 else 0.0,
            "conf_max": round(float(np.max(cl)), 3) if n else 0.0,
            "conf_min": round(float(np.min(cl)), 3) if n else 0.0,
        })

    return species_data


def write_processed(processed: Dict, directory: str) -> str:
    """Write processed.json to a directory."""
    path = os.path.join(directory, "processed.json")
    with open(path, "w") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)
    print(f"    ✓ processed.json written ({len(processed)} species)")
    return path


def get_files_to_keep(processed: Dict) -> set:
    """
    From processed.json, return the set of WAV filenames that are
    the 'primary_channel' for any detection. All other beamforming
    WAVs in the same directory can be deleted to save space.
    """
    keep: set = set()
    for sp_data in processed.values():
        for ch in sp_data.get("primary_channel_list", []):
            keep.add(ch)
    return keep


def cleanup_beamforming_files(
    directory: str,
    keep_files: set,
    dry_run: bool = False,
) -> int:
    """
    Delete WAV files in `directory` that are NOT in `keep_files`.

    Returns count of deleted files.
    """
    deleted = 0
    for fname in os.listdir(directory):
        full = os.path.join(directory, fname)
        if not os.path.isfile(full):
            continue
        if not fname.lower().endswith(".wav"):
            continue
        if fname in keep_files:
            continue
        if dry_run:
            print(f"    [DRY RUN] Would delete: {fname}")
        else:
            os.remove(full)
            print(f"    🗑  Deleted: {fname}")
        deleted += 1
    return deleted


# ============================================================
# HIGH-LEVEL: Process one directory end-to-end
# ============================================================

def process_directory_pipeline(
    directory: str,
    date: Optional[datetime] = None,
    identifier_pattern: str = "",
    cleanup: bool = True,
    dry_run: bool = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
) -> Tuple[str, str, int]:
    """
    Full pipeline for one output directory:
      1. Run BirdNET → results.json
      2. Build confidence comparison → processed.json
      3. Optionally delete low-confidence beamforming files

    Returns:
        (results_path, processed_path, files_deleted)
    """
    print(f"\n  🐦 BirdNET: {directory}")

    # Step 1: BirdNET (sequential, no multiprocessing)
    results_path = run_birdnet_on_dir(directory, date=date, lat=lat, lon=lon)

    # Step 2: Confidence comparison
    results = read_results(results_path)
    if not results:
        print("    ⚠ No BirdNET results — skipping processed.json & cleanup")
        return results_path, "", 0

    processed = build_processed(results, identifier_pattern)
    processed_path = write_processed(processed, directory)

    # Step 3: Cleanup
    deleted = 0
    if cleanup:
        keep = get_files_to_keep(processed)
        print(f"    Keeping {len(keep)} best-variant files")
        deleted = cleanup_beamforming_files(directory, keep, dry_run=dry_run)
        print(f"    Cleanup: {deleted} files removed")

    return results_path, processed_path, deleted
