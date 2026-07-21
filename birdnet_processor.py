"""
BirdNET analysis and confidence-level comparison.

Runs BirdNET on beamforming or signal-averaging output directories,
produces results.json, then processed.json with the best-direction
selection and confidence metrics.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from config import BIRDNET_MIN_CONF, BIRDNET_OVERLAP, BIRDNET_FP16_MODEL

# FP16 model monkey-patch (27% faster inference, 4x faster cold start)
if BIRDNET_FP16_MODEL:
    import birdnetlib.analyzer as _ba
    _ba.MODEL_PATH = os.path.join(
        os.path.dirname(_ba.MODEL_PATH),
        "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite",
    )

from birdnetlib.analyzer import Analyzer
from birdnetlib.main import Recording

BIRDNET_WORKERS = int(os.environ.get("BIRDNET_WORKERS", "4"))

# Thread-local Analyzer (lazy init per thread)
_thread_local = threading.local()


def _get_analyzer() -> Analyzer:
    if not hasattr(_thread_local, "analyzer"):
        _thread_local.analyzer = Analyzer()
    return _thread_local.analyzer


def _analyze_one(wav_path: str, rec_kwargs: dict) -> Tuple[str, List[Dict], str]:
    fname = os.path.basename(wav_path)
    try:
        analyzer = _get_analyzer()
        rec = Recording(analyzer, wav_path, **rec_kwargs)
        rec.analyze()
        return fname, rec.detections, ""
    except Exception as e:
        return fname, [], str(e)


def run_birdnet_on_dir(
    directory: str,
    date: Optional[datetime] = None,
    min_conf: float = BIRDNET_MIN_CONF,
    overlap: float = BIRDNET_OVERLAP,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    workers: int = BIRDNET_WORKERS,
    base_name: Optional[str] = None,
) -> str:
    results_fname = f"results_{base_name}.json" if base_name else "results.json"
    results_path = os.path.join(directory, results_fname)
    if date is None:
        date = datetime.now()

    wav_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav") and not f.startswith("._")
    ])
    if base_name:
        wav_files = [w for w in wav_files if os.path.basename(w).startswith(base_name)]

    total = len(wav_files)
    if total == 0:
        print("    No WAV files found in directory")
        with open(results_path, "w") as f:
            json.dump({}, f, indent=4)
        return results_path

    rec_kwargs: dict = {"date": date, "min_conf": min_conf, "overlap": overlap}
    if lat is not None and lon is not None:
        rec_kwargs["lat"] = lat
        rec_kwargs["lon"] = lon

    print(f"    Processing {total} WAV files ({workers} workers) ...")

    results_dict: Dict[str, List] = {}
    errors = 0
    completed = 0
    files_with_dets = 0
    total_dets = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_analyze_one, wav, rec_kwargs): wav for wav in wav_files}
        for future in as_completed(futures):
            fname, detections, error = future.result()
            completed += 1
            if error:
                errors += 1
                if errors <= 5:
                    print(f"    [{completed:3d}/{total}] {fname}: {error}")
            else:
                results_dict[fname] = detections
                n = len(detections)
                if n > 0:
                    files_with_dets += 1
                    total_dets += n
            if completed % 20 == 0 or completed == total:
                pct = completed * 100 // total
                print(f"    [{completed:3d}/{total}] {pct}%  "
                      f"({files_with_dets} files with {total_dets} dets, {errors} errors)")

    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"    results.json written "
          f"({files_with_dets}/{total} files with detections, "
          f"{total_dets} total, {errors} errors)")
    return results_path


# ── Confidence comparison ────────────────────────────────────

def read_results(file_path: str) -> Dict:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"  Error reading {file_path}: {e}")
        return {}


def extract_unique_bf_detections(
    results_dict: Dict, conf_thresh: float, identifier_pattern: str = "",
) -> List[Dict]:
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
    results_dict: Dict, identifier_pattern: str = "",
    conf_thresh: float = BIRDNET_MIN_CONF,
) -> Dict:
    detections = extract_unique_bf_detections(results_dict, conf_thresh, identifier_pattern)
    species_data: Dict[str, Dict] = {}
    for det in detections:
        sp = det.get("common_name", "Unknown")
        conf = det.get("confidence", 0)
        start_t = det.get("start_time", 0)
        channel = det.get("primary_channel", "")
        if sp not in species_data:
            species_data[sp] = {"conf_list": [], "start_time_list": [], "primary_channel_list": []}
        species_data[sp]["conf_list"].append(conf)
        species_data[sp]["start_time_list"].append(start_t)
        species_data[sp]["primary_channel_list"].append(channel)
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
    path = os.path.join(directory, "processed.json")
    with open(path, "w") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)
    print(f"    processed.json written ({len(processed)} species)")
    return path


def get_files_to_keep(processed: Dict) -> set:
    keep: set = set()
    for sp_data in processed.values():
        for ch in sp_data.get("primary_channel_list", []):
            keep.add(ch)
    return keep


def cleanup_beamforming_files(directory: str, keep_files: set, dry_run: bool = False, base_name: Optional[str] = None) -> int:
    deleted = 0
    for fname in os.listdir(directory):
        full = os.path.join(directory, fname)
        if not os.path.isfile(full):
            continue
        if not fname.lower().endswith(".wav"):
            continue
        if fname in keep_files:
            continue
        # Only delete WAVs belonging to this FLAC (prevent cross-FLAC deletion)
        if base_name and not fname.startswith(base_name):
            continue
        if fname.startswith("._"):
            os.remove(full)
            deleted += 1
            continue
        if dry_run:
            print(f"    [DRY RUN] Would delete: {fname}")
        else:
            os.remove(full)
            print(f"    Deleted: {fname}")
        deleted += 1
    return deleted


def process_directory_pipeline(
    directory: str,
    date: Optional[datetime] = None,
    identifier_pattern: str = "",
    cleanup: bool = True,
    dry_run: bool = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    base_name: Optional[str] = None,
) -> Tuple[str, str, int]:
    import time
    print(f"\n  BirdNET: {directory}")
    t0 = time.time()

    results_path = run_birdnet_on_dir(directory, date=date, lat=lat, lon=lon, base_name=base_name)
    t1 = time.time()
    print(f"    BirdNET: {t1 - t0:.1f}s")

    results = read_results(results_path)
    if not results:
        print("    No BirdNET results - skipping processed.json & cleanup")
        return results_path, "", 0

    processed = build_processed(results, identifier_pattern)
    processed_fname = f"processed_{base_name}.json" if base_name else "processed.json"
    processed_path = os.path.join(directory, processed_fname)
    with open(processed_path, "w") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)
    print(f"    {processed_fname} written ({len(processed)} species)")

    deleted = 0
    if cleanup:
        keep = get_files_to_keep(processed)
        print(f"    Keeping {len(keep)} best-variant files")
        deleted = cleanup_beamforming_files(directory, keep, dry_run=dry_run, base_name=base_name)
        print(f"    Cleanup: {deleted} files removed")
        for fname in os.listdir(directory):
            if fname.startswith("._"):
                full = os.path.join(directory, fname)
                if os.path.isfile(full):
                    os.remove(full)

    print(f"    Total: {time.time() - t0:.1f}s")
    return results_path, processed_path, deleted
