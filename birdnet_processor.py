"""
ARCHIVED (species-ID path) — used by run_pipeline.py only.

Active signal path uses pipeline_signal_processing.py; downstream embeddings are separate,
not species labels. Kept for historical analysis and optional FP audits.

BirdNET analysis and confidence-level comparison.

Runs BirdNET on beamforming / signal-processing output directories,
produces results.json, then processed.json with the best-direction
selection and confidence metrics.
"""

import os
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

import numpy as np
import soundfile as sf
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set

from config import (
    BIRDNET_MIN_CONF,
    BIRDNET_OVERLAP,
    BIRDNET_FP16_MODEL,
    resolve_birdnet_filter,
)

# Suppress TFLite C++ logging (XNNPACK INFO, etc.) — must be set BEFORE
# any TFLite import, otherwise the env var has no effect.
# - TF_CPP_MIN_LOG_LEVEL  → TensorFlow proper
# - TFLITE_MIN_LOG_LEVEL  → tflite_runtime (pip package used by birdnetlib)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TFLITE_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TFLITE_MIN_LOG_LEVEL", "3")

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
CHUNK_SECONDS = int(os.environ.get("CHUNK_SECONDS", "6"))



# ── Regex helpers ───────────────────────────────────────────

# Extract source-recording base (e.g. "23-08-27_dur=240secs")
# from chunked or full WAV filenames:
#   s_000_23-02-27_dur=240secs_LabIR(S01_000).wav
#   s_006_23-02-27_dur=240secs_LabIR(S12_000).wav
#   23-02-27_dur=240secs_sa.wav
#   23-02-27_dur=240secs_mono.wav
_SOURCE_RE = re.compile(
    r"^(?:s_\d{3}_)?(?P<src>.+?)_(?:LabIR|SPIR1|SPIR2)\([^)]*\)\.wav$"
    r"|^(?:s_\d{3}_)?(?P<src_sa>.+?)_sa\.wav$"
    r"|^(?:s_\d{3}_)?(?P<src_mono>.+?)_mono\.wav$"
)

_CHUNK_OFFSET_RE = re.compile(r"^s_(\d{3})_")


def parse_source_base(wav_name: str) -> str:
    """Return the source-recording identifier from a WAV filename."""
    m = _SOURCE_RE.match(wav_name)
    if not m:
        return wav_name[:-4] if wav_name.lower().endswith(".wav") else wav_name
    return m.group("src") or m.group("src_sa") or m.group("src_mono")


def _parse_chunk_offset(wav_name: str) -> int:
    """Extract chunk-second offset from s_NNN_ prefix, or 0 for full WAVs."""
    m = _CHUNK_OFFSET_RE.match(wav_name)
    return int(m.group(1)) if m else 0


# ── WAV slicing ─────────────────────────────────────────────

def slice_wav_to_chunks(
    wav_path: str, output_dir: str,
    chunk_seconds: int = CHUNK_SECONDS, fs: int = 16000,
) -> int:
    """Slice a mono WAV into fixed-duration chunks, write to output_dir.

    Each chunk is named   s_NNN_<original_name>.wav   where NNN is the
    zero-padded start second of the chunk in the source recording.

    Chunks shorter than half the expected duration are discarded
    (they are tail residue).  Returns the number of chunks written.
    """
    audio, _sr = sf.read(wav_path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]  # take first channel if somehow multi-channel
    chunk_len = chunk_seconds * fs
    base = os.path.basename(wav_path)
    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for i in range(0, len(audio), chunk_len):
        segment = audio[i:i + chunk_len]
        if len(segment) < chunk_len // 2:
            break  # skip short tail
        sec = i // fs
        out_name = f"s_{sec:03d}_{base}"
        sf.write(os.path.join(output_dir, out_name), segment, fs, subtype="PCM_16")
        written += 1
    return written


# ── BirdNET analysis ───────────────────────────────────────

# Thread-local cache so each worker thread gets its own TFLite
# interpreter (TFLite is NOT thread-safe, sharing one across
# threads causes "reference to internal data" errors).
_thread_local = threading.local()

# ── fd-level output suppression (fd 1, fd 2 → /dev/null) ──
#
# tflite_runtime (used by birdnetlib) ignores environment variables
# for certain log messages; its XNNPACK INFO and "Labels loaded" go
# to raw stdout/stderr via C++ std::cout/std::cerr.  Python-level
# redirect_stdout/stderr cannot catch those — we must suppress at
# the process-wide fd level.  The strategy: suppress once before any
# TFLite call and restore once after all analysis is done.  Because
# multiple threads may enter _analyze_one concurrently, suppress is
# always paired with a lock — but the suppression itself persists
# across threads (which is fine: we want ALL output gone).
_fd_lock = threading.Lock()
_fd_suppressed = False
_fd_saved_out: Optional[int] = None
_fd_saved_err: Optional[int] = None


def _fd_maybe_suppress() -> None:
    """Suppress C-level stdout/stderr if not already suppressed."""
    import os as _os, sys as _sys
    global _fd_suppressed, _fd_saved_out, _fd_saved_err
    with _fd_lock:
        if not _fd_suppressed:
            _sys.stdout.flush()
            _sys.stderr.flush()
            null_fd = _os.open(_os.devnull, _os.O_WRONLY)
            _fd_saved_out = _os.dup(1)
            _fd_saved_err = _os.dup(2)
            _os.dup2(null_fd, 1)
            _os.dup2(null_fd, 2)
            _os.close(null_fd)
            _fd_suppressed = True


def _fd_restore_once() -> None:
    """Restore C-level stdout/stderr if they were suppressed."""
    import os as _os, sys as _sys
    global _fd_suppressed, _fd_saved_out, _fd_saved_err
    with _fd_lock:
        if _fd_suppressed and _fd_saved_out is not None:
            _sys.stdout.flush()
            _sys.stderr.flush()
            _os.dup2(_fd_saved_out, 1)
            _os.dup2(_fd_saved_err, 2)
            _os.close(_fd_saved_out)
            _os.close(_fd_saved_err)
            _fd_suppressed = False
            _fd_saved_out = None
            _fd_saved_err = None


def _get_analyzer(species_list_path: Optional[str] = None) -> Analyzer:
    """Return a thread-local Analyzer.

    birdnetlib + TFLite dump species list / delegate info to both
    stdout and stderr on construction. We suppress both.
    """
    import io
    from contextlib import redirect_stdout, redirect_stderr

    cache = getattr(_thread_local, "cache", None)
    if cache is None:
        cache = {}
        _thread_local.cache = cache

    key = species_list_path or ""
    if key in cache:
        return cache[key]

    if species_list_path:
        if not os.path.isfile(species_list_path):
            raise FileNotFoundError(
                f"BirdNET species list not found: {species_list_path}"
            )
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cache[key] = Analyzer(custom_species_list_path=species_list_path)
    else:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cache[key] = Analyzer()

    return cache[key]


def _analyze_one(
    wav_path: str,
    rec_kwargs: dict,
    species_list_path: Optional[str] = None,
) -> Tuple[str, List[Dict], str]:
    fname = os.path.basename(wav_path)
    # fd suppression is already active from run_birdnet_on_dir.
    # _fd_maybe_suppress() is a no-op here (already suppressed),
    # but it's safe to call in case this function is ever called
    # outside the normal pipeline context.
    _fd_maybe_suppress()
    try:
        import io as _io
        from contextlib import redirect_stdout as _rso, redirect_stderr as _rse
        analyzer = _get_analyzer(species_list_path)
        rec = Recording(analyzer, wav_path, **rec_kwargs)
        # Python-level redirect catches birdnetlib species-list dump;
        # fd-level suppression catches tflite C++ output.
        with _rso(_io.StringIO()), _rse(_io.StringIO()):
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
    species_list_path: Optional[str] = None,
    workers: int = BIRDNET_WORKERS,
) -> str:
    """Run BirdNET on all WAVs in *directory*.

    Filter modes (mutually exclusive in birdnetlib):
      - species_list_path set → custom allow-list; lat/lon ignored
      - lat & lon set → eBird-range geo filter
      - neither → full 6K labels above min_conf
    """
    results_path = os.path.join(directory, "results.json")
    if date is None:
        date = datetime.now()

    # Custom list XOR geo (birdnetlib hard rule)
    use_list = bool(species_list_path)
    if use_list and (lat is not None or lon is not None):
        print("    Note: custom species list active — ignoring lat/lon geo filter")
        lat, lon = None, None

    wav_files = sorted([
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".wav") and not f.startswith("._")
    ])

    total = len(wav_files)
    if total == 0:
        print("    No WAV files found in directory")
        with open(results_path, "w") as f:
            json.dump({}, f, indent=4)
        return results_path

    rec_kwargs: dict = {"date": date, "min_conf": min_conf, "overlap": overlap}
    if not use_list and lat is not None and lon is not None:
        rec_kwargs["lat"] = lat
        rec_kwargs["lon"] = lon

    t_start = time.time()
    start_iso = datetime.now().isoformat(timespec="seconds")
    print(f"    BirdNET: {total} files ({workers} workers)  start={start_iso}", flush=True)

    results_dict: Dict[str, List] = {}
    errors = 0
    completed = 0
    files_with_dets = 0
    total_dets = 0

    # Suppress C-level output (TFLite XNNPACK INFO, Labels loaded, etc.)
    # for the entire BirdNET run.  Restore when done, even on crash.
    _fd_maybe_suppress()
    try:
        # Warm one analyzer so load errors surface before the pool.
        _get_analyzer(species_list_path if use_list else None)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {
                pool.submit(
                    _analyze_one,
                    wav,
                    rec_kwargs,
                    species_list_path if use_list else None,
                ): wav
                for wav in wav_files
            }
            for future in as_completed(futures):
                fname, detections, error = future.result()
                completed += 1
                if error:
                    errors += 1
                    if errors <= 3:
                        print(f"    [{completed:3d}/{total}] err: {error[:80]}")
                else:
                    results_dict[fname] = detections
                    n = len(detections)
                    if n > 0:
                        files_with_dets += 1
                        total_dets += n
                elapsed = time.time() - t_start
                pct = completed * 100 // total
                print(f"    [{completed:3d}/{total}] {pct}%  "
                      f"({files_with_dets}f/{total_dets}d)  {elapsed:.0f}s", flush=True)
    except BaseException as e:
        # Crash-proof: write partial results + error before re-raising.
        # Without this, the directory is left without results.json and
        # the pipeline has no record it ever attempted this directory.
        elapsed = time.time() - t_start
        end_iso = datetime.now().isoformat(timespec="seconds")
        timing = {
            "directory": directory,
            "start": start_iso,
            "end": end_iso,
            "elapsed_s": round(elapsed, 1),
            "n_files_total": total,
            "n_files_with_detections": files_with_dets,
            "n_total_detections": total_dets,
            "n_errors": errors,
            "workers": workers,
            "fatal_error": str(e),
            "status": "crashed",
        }
        try:
            timing_path = os.path.join(directory, "timing.json")
            with open(timing_path, "w") as f:
                json.dump(timing, f, indent=4, ensure_ascii=False)
        except Exception:
            pass
        try:
            results_dict["__fatal_error__"] = str(e)
            with open(results_path, "w") as f:
                json.dump(results_dict, f, indent=4)
        except Exception:
            pass
        print(f"    ❌ CRASHED after {completed}/{total} files: {e}", flush=True)
        raise
    finally:
        _fd_restore_once()

    elapsed = time.time() - t_start
    end_iso = datetime.now().isoformat(timespec="seconds")

    timing = {
        "directory": directory,
        "start": start_iso,
        "end": end_iso,
        "elapsed_s": round(elapsed, 1),
        "n_files_total": total,
        "n_files_with_detections": files_with_dets,
        "n_total_detections": total_dets,
        "n_errors": errors,
        "workers": workers,
    }
    timing_path = os.path.join(directory, "timing.json")
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=4, ensure_ascii=False)

    with open(results_path, "w") as f:
        json.dump(results_dict, f, indent=4)

    print(f"    ✅ done: {total}f/{files_with_dets}dets/{errors}err  {elapsed:.1f}s  "
          f"→ results.json + timing.json", flush=True)
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
    results_dict: Dict, conf_thresh: float,
) -> List[Dict]:
    """Pick the best-confidence detection per (source_recording, species, absolute_second).

    Chunk-level offset adjustment: when a WAV is a 6-second slice prefixed
    ``s_006_``, BirdNET reports start_time relative to the chunk (0.0 / 3.0).
    We add the chunk offset so the absolute second in the original recording
    is used for dedup — preventing unrelated chunks from competing.
    """
    best_per_key: Dict[str, Dict] = {}
    for channel, detections in results_dict.items():
        source_base = parse_source_base(channel)
        chunk_offset = _parse_chunk_offset(channel)
        for det in detections:
            conf = det.get("confidence", 0)
            if conf < conf_thresh:
                continue
            species = det.get("common_name", "Unknown")
            rel_start = round(det.get("start_time", 0), 1)
            absolute_st = chunk_offset + rel_start
            # Dedup scope: one source recording, one species, one absolute second
            key = f"{source_base}|{species}_{absolute_st}"
            cur = best_per_key.get(key)
            if cur is None or conf > cur.get("confidence", 0):
                det_copy = det.copy()
                det_copy["start_time"] = absolute_st
                det_copy["primary_channel"] = channel
                det_copy["source_base"] = source_base
                best_per_key[key] = det_copy
    return list(best_per_key.values())


def build_processed(
    results_dict: Dict,
    conf_thresh: float = BIRDNET_MIN_CONF,
) -> Dict:
    """Build nested processed.json: {source_base: {species: {stats}}}.

    Outer key is the source-recording identifier (e.g. "23-08-27_dur=240secs")
    so each minute stays isolated.  Inner key is the species.
    """
    detections = extract_unique_bf_detections(results_dict, conf_thresh)
    by_source: Dict[str, Dict[str, Dict]] = {}
    for det in detections:
        src = det.get("source_base", "")
        sp = det.get("common_name", "Unknown")
        sd = by_source.setdefault(src, {}).setdefault(
            sp, {"conf_list": [], "start_time_list": [], "primary_channel_list": []},
        )
        sd["conf_list"].append(det.get("confidence", 0))
        sd["start_time_list"].append(det.get("start_time", 0))
        sd["primary_channel_list"].append(det.get("primary_channel", ""))

    for _src, sp_map in by_source.items():
        for _sp, sd in sp_map.items():
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
    return by_source


def get_files_to_keep(processed: Dict) -> Set[str]:
    """Collect winning WAV filenames from the nested processed.json."""
    keep: Set[str] = set()
    for src_data in processed.values():
        for sp_data in src_data.values():
            for ch in sp_data.get("primary_channel_list", []):
                keep.add(ch)
    return keep


def cleanup_losing_chunks(directory: str, keep_files: Set[str],
                          dry_run: bool = False) -> int:
    """Delete chunk WAV files that are NOT in the keep set (losers)."""
    deleted = 0
    for fname in os.listdir(directory):
        if not fname.lower().endswith(".wav"):
            continue
        if fname in keep_files:
            continue
        full = os.path.join(directory, fname)
        if not os.path.isfile(full):
            continue
        if fname.startswith("._"):
            os.remove(full)
            deleted += 1
            continue
        if dry_run:
            deleted += 1
        else:
            os.remove(full)
            deleted += 1
    if dry_run:
        print(f"    [DRY RUN] Would delete {deleted} losing chunk(s)")
    return deleted


def process_directory_pipeline(
    directory: str,
    date: Optional[datetime] = None,
    cleanup: bool = True,
    dry_run: bool = False,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    species_list_path: Optional[str] = None,
    location_name: Optional[str] = None,
) -> Tuple[str, str, int]:
    """Run BirdNET on a per-minute directory, then build processed.json,
    then optionally delete losing chunk WAVs.

    If *location_name* is given and *species_list_path* / lat / lon are all
    omitted, filter mode is resolved via ``config.resolve_birdnet_filter``.
    """
    # Show just h_XX/m_YY
    parts = directory.rstrip("/").split("/")
    short_label = "/".join(parts[-2:]) if len(parts) >= 2 else os.path.basename(directory)
    print(f"\n  BirdNET [{short_label}]", flush=True)
    t0 = time.time()

    if (
        location_name is not None
        and species_list_path is None
        and lat is None
        and lon is None
    ):
        species_list_path, lat, lon, mode = resolve_birdnet_filter(location_name)

    results_path = run_birdnet_on_dir(
        directory,
        date=date,
        lat=lat,
        lon=lon,
        species_list_path=species_list_path,
    )

    results = read_results(results_path)
    if not results:
        elapsed = time.time() - t0
        end_iso = datetime.now().isoformat(timespec="seconds")
        print(f"    No detections — skipping processed.json  {elapsed:.1f}s")
        timing_path = os.path.join(directory, "timing.json")
        timing = {
            "directory": directory,
            "phase": "pipeline",
            "elapsed_s": round(elapsed, 1),
            "end": end_iso,
            "n_sources": 0,
            "n_species_entries": 0,
            "n_chunks_deleted": 0,
        }
        if os.path.isfile(timing_path):
            try:
                with open(timing_path, "r") as f:
                    existing = json.load(f)
                timing.update(existing)
            except Exception:
                pass
        with open(timing_path, "w") as f:
            json.dump(timing, f, indent=4, ensure_ascii=False)
        return results_path, timing_path, 0

    processed = build_processed(results)
    processed_path = os.path.join(directory, "processed.json")
    with open(processed_path, "w") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)
    n_sources = len(processed)
    n_sp_entries = sum(len(v) for v in processed.values())

    deleted = 0
    if cleanup:
        keep = get_files_to_keep(processed)
        deleted = cleanup_losing_chunks(directory, keep, dry_run=dry_run)
        for fname in os.listdir(directory):
            if fname.startswith("._"):
                full = os.path.join(directory, fname)
                if os.path.isfile(full):
                    os.remove(full)

    elapsed = time.time() - t0
    end_iso = datetime.now().isoformat(timespec="seconds")
    print(f"    Done: {n_sources}s/{n_sp_entries}sp  "
          f"{deleted} cleaned  {elapsed:.1f}s")

    # Save/update timing.json with full pipeline stats (BirdNET + processing)
    timing_path = os.path.join(directory, "timing.json")
    timing = {
        "directory": directory,
        "phase": "pipeline",
        "elapsed_s": round(elapsed, 1),
        "end": end_iso,
        "n_sources": n_sources,
        "n_species_entries": n_sp_entries,
        "n_chunks_deleted": deleted,
    }
    # Merge with BirdNET-level timing if it already exists
    if os.path.isfile(timing_path):
        try:
            with open(timing_path, "r") as f:
                existing = json.load(f)
            timing.update(existing)
        except Exception:
            pass
    with open(timing_path, "w") as f:
        json.dump(timing, f, indent=4, ensure_ascii=False)

    return results_path, processed_path, deleted
