#!/usr/bin/env python3
"""
Unified pipeline: Beamforming → SA → Mono → Dense BirdNET Embeddings.

Replaces the old species-ID pipeline.  No results.json, no processed.json,
no BirdNET label classification, no prefilter RMS filtering, no 6s chunking.

Phases per date:
  1.  Beamforming (LabIR → bf_LabIR, SPIR1/SPIR2 → bf_SPIR directly)
  2.  Signal Averaging (6-ch → 1-ch) + Mono baseline
  3.  Extract 1024-dim embeddings from all full WAVs via dense sliding window
  4.  Save embeddings/*.npy + *_meta.json

Output per date per method:
  embeddings/{date}_{method}_embeddings.npy
  embeddings/{date}_{method}_meta.json

Usage:
  # PoC (12-sec recordings)
  python pipeline_embeddings.py --location 2A400 \\
      --date 2026-03-19,2026-03-20 --ir-types LabIR,SPIR1,SPIR2 --max-files 0

  # Real data (240-sec recordings)
  python pipeline_embeddings.py --location 2A400 \\
      --date 2026-04-22 --ir-types LabIR,SPIR1,SPIR2 --max-files 0
"""

import os
import os
import re
import sys
import time
import argparse
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import soundfile as sf

from config import (
    MONITORING_DATA, ANALYSIS_OUTPUT, IR_TYPES, PROTOTYPE_IR_SUBSETS,
    LOCATION_MAP, RPIID_TO_LOCATION, FS_TARGET, LABIR_SPEAKER_ELEVATION,
)
from beamforming import Beamformer
from signal_averaging import SignalAverager
from embedding_schema import (
    BACKEND_BIRDNET,
    BIRDNET_EMBEDDING_DIM,
    BIRDNET_MODEL_ID,
    BIRDNET_SLIDE_SEC,
    BIRDNET_WINDOW_SEC,
    make_window_meta,
    resolve_birdnet_out_dir,
)


# ── Embedding constants ─────────────────────────────────
EMBEDDING_DIM = BIRDNET_EMBEDDING_DIM
MODEL_SAMPLE_RATE = 48000
MODEL_INPUT_SAMPLES = 144000   # 3 s @ 48 kHz
WINDOW_SEC = BIRDNET_WINDOW_SEC
SLIDE_SEC = BIRDNET_SLIDE_SEC

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TFLITE_MIN_LOG_LEVEL", "3")

from birdnetlib.analyzer import Analyzer

# ── Helpers ──────────────────────────────────────────────

_HM_RE = re.compile(r"^(\d{2})-(\d{2})-\d{2}_dur=")

# Regex patterns for parsing direction metadata from WAV filenames
_LABIR_RE = re.compile(r"LabIR\(S(\d{2})_(\d{3})\)")  # S01_060 → speaker=1, azimuth=60
_SPIR1_RE = re.compile(r"SPIR1\((\d{2})m_(\d{3})\)")   # 02m_060 → distance=2m, azimuth=60
_SPIR2_RE = re.compile(r"SPIR2\((\d{2})m_(\d{3})_r(\d)\)")  # 08m_180_r2 → distance=8m, azimuth=180, rep=2


def _parse_direction_metadata(wav_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse azimuth and elevation from WAV filename.
    
    Returns:
        (azimuth, elevation) in degrees, or (None, None) if not parseable.
        
    Examples:
        "s_000_23-26-29_dur=240secs_LabIR(S01_060).wav" → (60, 0)
        "s_000_23-26-29_dur=240secs_LabIR(S12_000).wav" → (0, 90)
        "s_000_23-26-29_dur=240secs_SPIR1(02m_180).wav" → (180, 0)
        "s_000_23-26-29_dur=240secs_sa.wav" → (None, None)
    """
    # Try LabIR pattern
    m = _LABIR_RE.search(wav_name)
    if m:
        speaker = int(m.group(1))
        azimuth = int(m.group(2))
        elevation = LABIR_SPEAKER_ELEVATION.get(speaker)
        return (azimuth, elevation)
    
    # Try SPIR1 pattern (elevation always 0 for horizontal plane)
    m = _SPIR1_RE.search(wav_name)
    if m:
        azimuth = int(m.group(2))
        return (azimuth, 0)
    
    # Try SPIR2 pattern (elevation always 0 for horizontal plane)
    m = _SPIR2_RE.search(wav_name)
    if m:
        azimuth = int(m.group(2))
        return (azimuth, 0)
    
    # No direction info (SA, Mono, or unrecognized pattern)
    return (None, None)



def _extract_hour_minute(flac_path: str) -> Tuple[str, str]:
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
    path = os.path.join(ANALYSIS_OUTPUT, location_name, date_str, processing_type)
    if hour:
        path = os.path.join(path, f"h_{hour}")
    if minute:
        path = os.path.join(path, f"m_{minute}")
    return path


def _minute_complete(output_dir: str, base_name: str) -> bool:
    """Check whether beamforming output (full WAVs) already exists."""
    if not os.path.isdir(output_dir):
        return False
    try:
        files = [f for f in os.listdir(output_dir)
                 if f.endswith(".wav") and not f.startswith("._")
                 and base_name in f]
        return len(files) > 0
    except OSError:
        return False


def _sa_output_exists(output_dir: str, base_name: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, base_name + "_sa.wav"))


def _mono_output_exists(output_dir: str, base_name: str) -> bool:
    return os.path.isfile(os.path.join(output_dir, base_name + "_mono.wav"))


# ── Phase 1: Signal Processing ──────────────────────────

def _spir_type_complete(bf_spir_dir: str, base_name: str, spir_type: str) -> bool:
    """Check if SPIR1 or SPIR2 beamforming output already exists in bf_SPIR."""
    if not os.path.isdir(bf_spir_dir):
        return False
    pattern = spir_type + "("  # e.g. "SPIR1(" or "SPIR2("
    try:
        files = [f for f in os.listdir(bf_spir_dir)
                 if f.endswith(".wav") and not f.startswith("._")
                 and base_name in f and pattern in f]
        return len(files) > 0
    except OSError:
        return False


def process_one_flac(
    flac_path: str, location_name: str, date_str: str,
    ir_types: List[str],
    use_prototype_subsets: bool = False,
    force_bf: bool = False,
) -> dict:
    """Beamforming + SA + Mono for a single FLAC."""
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
    _cleaned_bf_spir = False  # track whether bf_SPIR has been force-cleaned

    # -- Beamforming (full WAVs, no slice) -----------
    for ir_name in ir_types:
        if ir_name not in ir_configs:
            print(f"\u26a0  Unknown IR type: {ir_name} — skipping")
            continue
        ir_type = ir_configs[ir_name]

        # SPIR1 / SPIR2 -> both write directly to bf_SPIR (no merge needed)
        if ir_name in ("SPIR1", "SPIR2"):
            bf_dir = build_output_path(location_name, date_str, "bf_SPIR",
                                       hour_str, minute_str)
            # Check if THIS specific SPIR type already has output in bf_SPIR
            if not force_bf and _spir_type_complete(bf_dir, base_name, ir_name):
                print(f"  \u2713 bf_SPIR ({ir_name}) already exists — skipping")
                if (bf_dir, "SPIR") not in bf_dirs:
                    bf_dirs.append((bf_dir, "SPIR"))
                continue
            # force_bf: clean bf_SPIR only once (on first SPIR type processed)
            if force_bf and not _cleaned_bf_spir and os.path.isdir(bf_dir):
                removed = 0
                for fname in list(os.listdir(bf_dir)):
                    if fname.endswith(".wav") and not fname.startswith("._"):
                        try:
                            os.remove(os.path.join(bf_dir, fname))
                            removed += 1
                        except OSError:
                            pass
                if removed > 0:
                    print(f"  \U0001f5d1  Cleaned {removed} old WAV(s) from bf_SPIR")
                _cleaned_bf_spir = True
            label = "SPIR"
        else:
            bf_dir = build_output_path(location_name, date_str, f"bf_{ir_name}",
                                       hour_str, minute_str)
            label = ir_name
            if not force_bf and _minute_complete(bf_dir, base_name):
                print(f"  \u2713 bf_{ir_name} already exists — skipping")
                bf_dirs.append((bf_dir, ir_name))
                continue
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

        if (bf_dir, label) not in bf_dirs:
            bf_dirs.append((bf_dir, label))

        print(f"\n── Beamforming [{ir_name}] \u2192 {bf_dir} ──")
        beamformer = Beamformer(flac_path=flac_path, output_dir=bf_dir,
                                ir_type_or_name=ir_type)
        beamformer.run()

    # ── Signal Averaging ─────────────────────────────
    sa_dir = build_output_path(location_name, date_str, "sa",
                               hour_str, minute_str)
    print(f"\n── Signal Averaging \u2192 {sa_dir} ──")
    if _sa_output_exists(sa_dir, base_name):
        print("  \u2713 SA already exists — skipping")
    else:
        sa = SignalAverager(flac_path=flac_path, output_dir=sa_dir)
        sa.run()

    # ── Mono baseline ────────────────────────────────
    import librosa as _librosa
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
            _sf = sf
            _sf.write(mono_file,
                      (ch0 * 32767).clip(-32768, 32767).astype("int16"),
                      FS_TARGET, subtype="PCM_16")
            print(f"  \u2713 Mono baseline: {mono_file}")
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


# -- Phase 3: Embedding extraction -----------------------

# Thread-local analyzer — each worker thread gets its own TFLite interpreter
_thread_local = __import__("threading").local()


def _get_analyzer() -> Analyzer:
    cache = getattr(_thread_local, "cache", None)
    if cache is None:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        cache = {}
        _thread_local.cache = cache
    key = "default"
    if key not in cache:
        import io
        from contextlib import redirect_stdout, redirect_stderr
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cache[key] = Analyzer()
    return cache[key]


def _audio_to_model_input(audio_16k: np.ndarray) -> np.ndarray:
    """Resample 16 kHz → 48 kHz, pad/trim to MODEL_INPUT_SAMPLES."""
    from scipy.interpolate import interp1d
    n_in = len(audio_16k)
    duration = n_in / 16000.0
    n_target = int(duration * MODEL_SAMPLE_RATE)
    t_old = np.linspace(0, duration, n_in, endpoint=False)
    t_new = np.linspace(0, duration, n_target, endpoint=False)
    t_new = np.clip(t_new, 0, duration - 1e-12)
    interp = interp1d(t_old, audio_16k, kind="linear", copy=False,
                      assume_sorted=True, fill_value=0.0, bounds_error=False)
    audio_48k = interp(t_new).astype(np.float32)
    if len(audio_48k) < MODEL_INPUT_SAMPLES:
        audio_48k = np.pad(audio_48k, (0, MODEL_INPUT_SAMPLES - len(audio_48k)))
    elif len(audio_48k) > MODEL_INPUT_SAMPLES:
        audio_48k = audio_48k[:MODEL_INPUT_SAMPLES]
    return audio_48k


def _extract_one_embedding(analyzer: Analyzer, audio_48k: np.ndarray) -> np.ndarray:
    batch = audio_48k.reshape(1, -1).astype(np.float32)
    return analyzer._return_embeddings(batch)[0].astype(np.float32)


def _scan_wav(wav_path: str, wav_name: str, method: str) -> Tuple[np.ndarray, List[dict]]:
    """Dense sliding-window embedding extraction for one WAV."""
    analyzer = _get_analyzer()
    emb_list: List[np.ndarray] = []
    meta_list: List[dict] = []

    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
    except Exception:
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    # Resample to 16 kHz if needed (pipeline writes 16 kHz)
    if sr != 16000:
        from scipy.interpolate import interp1d
        duration = len(audio) / sr
        n_new = int(duration * 16000)
        t_old = np.linspace(0, duration, len(audio), endpoint=False)
        t_new = np.linspace(0, duration, n_new, endpoint=False)
        audio = interp1d(t_old, audio, kind="linear", copy=False,
                         assume_sorted=True, fill_value=0.0)(t_new).astype(np.float32)

    total_sec = len(audio) / 16000.0
    win_samp = int(WINDOW_SEC * 16000)
    step_samp = int(SLIDE_SEC * 16000)

    for start_sample in range(0, len(audio), step_samp):
        end_sample = start_sample + win_samp
        segment = audio[start_sample:end_sample].astype(np.float32)
        start_sec = start_sample / 16000.0

        if len(segment) < 16000:  # < 1 second
            break

        model_input = _audio_to_model_input(segment)
        try:
            emb = _extract_one_embedding(analyzer, model_input)
        except Exception:
            continue

        emb_list.append(emb)
        azimuth, elevation = _parse_direction_metadata(wav_name)
        meta_list.append(
            make_window_meta(
                wav=wav_name,
                method=method,
                start_sec=start_sec,
                end_sec=min(start_sec + WINDOW_SEC, total_sec),
                model=BIRDNET_MODEL_ID,
                backend=BACKEND_BIRDNET,
                window_sec=WINDOW_SEC,
                slide_sec=SLIDE_SEC,
                embedding_dim=EMBEDDING_DIM,
                azimuth=azimuth,
                elevation=elevation,
            )
        )

    if emb_list:
        return np.stack(emb_list, axis=0).astype(np.float32), meta_list
    return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []


def _find_minute_dirs(base_dir: str, location: str, date_str: str,
                      methods: List[str]) -> List[Tuple[str, str, str]]:
    """Return (minute_dir, method, date_str) for all minute dirs."""
    date_dir = os.path.join(base_dir, location, date_str)
    found = []
    for method in methods:
        method_dir = os.path.join(date_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for hour_entry in sorted(os.listdir(method_dir)):
            if not hour_entry.startswith("h_"):
                continue
            hour_dir = os.path.join(method_dir, hour_entry)
            if not os.path.isdir(hour_dir):
                continue
            for minute_entry in sorted(os.listdir(hour_dir)):
                if not minute_entry.startswith("m_"):
                    continue
                minute_dir = os.path.join(hour_dir, minute_entry)
                if not os.path.isdir(minute_dir):
                    continue
                found.append((minute_dir, method, date_str))
    return found


def extract_embeddings_for_date(
    location: str, date_str: str,
    methods: List[str],
    data_dir: str,
    out_dir: str,
    workers: int = 4,
) -> Dict[str, Any]:
    """Extract dense embeddings for all minute dirs of one date.

    Uses ThreadPoolExecutor so each worker thread gets its own
    thread-local TFLite interpreter.
    """
    os.makedirs(out_dir, exist_ok=True)
    minute_dirs = _find_minute_dirs(data_dir, location, date_str, methods)
    print(f"  Embedding dirs: {len(minute_dirs)} minute dir(s)")

    # Warm one analyzer on main thread so TFLite loads before pool
    _get_analyzer()

    # Collect (wav_path, wav_name, method) for all WAV files
    tasks: List[Tuple[str, str, str]] = []
    for minute_dir, method, _ in minute_dirs:
        for fname in os.listdir(minute_dir):
            if fname.lower().endswith(".wav") and not fname.startswith("._"):
                tasks.append((os.path.join(minute_dir, fname), fname, method))

    if not tasks:
        return {"date": date_str, "methods": {}, "n_wavs": 0}

    total_wavs = len(tasks)
    print(f"  Extracting from {total_wavs} WAVs ({workers} workers)…")

    t0 = time.time()
    by_method_emb: Dict[str, List[np.ndarray]] = {}
    by_method_meta: Dict[str, List[dict]] = {}
    done = 0

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(_scan_wav, wav_path, wav_name, method): (wav_name, method)
            for wav_path, wav_name, method in tasks
        }
        for future in as_completed(futures):
            wav_name, method = futures[future]
            done += 1
            try:
                emb, meta = future.result()
            except Exception as e:
                print(f"  ⚠️  {wav_name}: {e}", file=sys.stderr)
                continue
            if len(emb) > 0:
                by_method_emb.setdefault(method, []).append(emb)
                by_method_meta.setdefault(method, []).extend(meta)
            if done % 50 == 0 or done == total_wavs:
                pct = done * 100 // total_wavs
                total_emb = sum(
                    sum(len(e) for e in lst)
                    for lst in by_method_emb.values()
                )
                elapsed = time.time() - t0
                print(f"    [{done:3d}/{total_wavs}] {pct}%  "
                      f"{total_emb} embeddings  {elapsed:.0f}s", flush=True)

    # Save per-method .npy + .json
    summary: Dict[str, Any] = {"date": date_str, "methods": {}, "n_wavs": total_wavs}
    for method in sorted(by_method_emb.keys()):
        all_emb = np.concatenate(by_method_emb[method], axis=0).astype(np.float32)
        meta = by_method_meta.get(method, [])

        emb_path = os.path.join(out_dir, f"{date_str}_{method}_embeddings.npy")
        np.save(emb_path, all_emb)

        meta_path = os.path.join(out_dir, f"{date_str}_{method}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        summary["methods"][method] = {
            "n_embeddings": int(len(all_emb)),
            "embeddings_file": os.path.basename(emb_path),
            "metadata_file": os.path.basename(meta_path),
        }

    elapsed = time.time() - t0
    total_embs = sum(
        m["n_embeddings"] for m in summary["methods"].values()
    )
    print(f"  \u2705 Embeddings: {total_embs} total in {elapsed:.1f}s")

    return summary


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Unified pipeline: Beamforming → Embeddings (no species ID)")
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--date", type=str, required=True,
                        help="Date(s), comma-separated")
    parser.add_argument("--ir-types", type=str, default="LabIR,SPIR1,SPIR2",
                        help="IR types for beamforming (default: LabIR,SPIR1,SPIR2)")
    parser.add_argument("--max-files", type=int, default=0,
                        help="Max FLACs per date (0=all)")
    parser.add_argument("--force-bf", action="store_true",
                        help="Re-run beamforming even if chunks exist")
    parser.add_argument("--embeddings-out", type=str, default=None,
                        help="Output dir for .npy files "
                             "(default: data_dir/location/embeddings/birdnet)")
    parser.add_argument("--embed-workers", type=int, default=4,
                        help="Worker threads for embedding extraction")
    parser.add_argument("--data-dir", type=str, default=ANALYSIS_OUTPUT)
    parser.add_argument("--methods", type=str,
                        default="bf_LabIR,bf_SPIR,sa,mono",
                        help="Methods for embedding extraction")
    parser.add_argument("--no-sa", action="store_true",
                        help="Skip Signal Averaging")
    parser.add_argument("--prototype", action="store_true",
                        help="Use prototype IR subsets")
    args = parser.parse_args()

    # Resolve location → RPiID
    rpiid = LOCATION_MAP.get(args.location, args.location)
    location_name = RPIID_TO_LOCATION.get(rpiid, rpiid)
    dates = [d.strip() for d in args.date.split(",")]
    ir_types = [i.strip() for i in args.ir_types.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    emb_out = resolve_birdnet_out_dir(
        args.data_dir, location_name, args.embeddings_out
    )

    print(f"\u2699  Location: {location_name}  RPiID: {rpiid}")
    print(f"\U0001f4c5 Dates: {dates}")
    print(f"\U0001f399  IR types: {ir_types}")
    print(f"\U0001f4e6 Embedding output: {emb_out}")
    print()

    grand_start = time.time()

    for date_str in dates:
        print(f"\n{'#'*60}")
        print(f"# \U0001f4c5 Date: {date_str}")
        print(f"{'#'*60}")

        # ═══ Phase 1: Signal Processing ═══
        flac_paths = get_flac_files(rpiid, date_str, max_files=args.max_files)
        if not flac_paths:
            print(f"\u26a0  No FLAC files for {date_str} — skipping\n")
            continue

        print(f"\n── Phase 1: Signal Processing ({len(flac_paths)} FLACs) ──")
        t1 = time.time()
        for i, flac_path in enumerate(flac_paths, 1):
            print(f"\n[{i}/{len(flac_paths)}]")
            process_one_flac(
                flac_path=flac_path,
                location_name=location_name,
                date_str=date_str,
                ir_types=ir_types,
                use_prototype_subsets=args.prototype,
                force_bf=args.force_bf,
            )
        print(f"  \u23f1  Phase 1: {time.time() - t1:.1f}s")

        # ═══ Phase 2: Embedding extraction ═══
        print(f"\n── Phase 2: Dense Embedding Extraction ──")
        t3 = time.time()
        summary = extract_embeddings_for_date(
            location=location_name,
            date_str=date_str,
            methods=methods,
            data_dir=args.data_dir,
            out_dir=emb_out,
            workers=args.embed_workers,
        )

        # Write per-date summary for reference
        sum_path = os.path.join(emb_out, f"{date_str}_summary.json")
        with open(sum_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        total_embs = sum(
            m["n_embeddings"] for m in summary["methods"].values()
        )
        print(f"\n  \u2705 {date_str} done: {total_embs} embeddings "
              f"in {time.time() - t1:.1f}s")

    print(f"\n{'='*60}")
    print(f"\U0001f389 All done — total {time.time() - grand_start:.0f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
