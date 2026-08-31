#!/usr/bin/env python3
"""
Phase 1: Dense BirdNET embedding extraction — independent of species detections.

Scans every WAV file in the output directories, splits into 3-second
sliding windows, and extracts the 1024-dim embedding from BirdNET's
GLOBAL_AVG_POOL layer for EVERY window — not just detection points.

This gives us a dense acoustic map of each recording, enabling
method comparison based on embedding quality rather than unreliable
species labels.

Usage:
    python extract_embeddings.py --location 2A400 --date 2026-03-19
    python extract_embeddings.py --location 2A400 --date 2026-03-19,2026-03-20
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

# ── Paths ──────────────────────────────────────────────
try:
    from config import ANALYSIS_OUTPUT
except ImportError:
    ANALYSIS_OUTPUT = "/Volumes/WD2TB/sea-data"

from embedding_schema import (
    BACKEND_BIRDNET,
    BIRDNET_EMBEDDING_DIM,
    BIRDNET_MODEL_ID,
    BIRDNET_SLIDE_SEC,
    BIRDNET_WINDOW_SEC,
    make_window_meta,
    resolve_birdnet_out_dir,
)

EMBEDDING_DIM = BIRDNET_EMBEDDING_DIM
MODEL_SAMPLE_RATE = 48000       # BirdNET native rate
MODEL_INPUT_SAMPLES = 144000    # 3 seconds at 48kHz
WINDOW_SEC = BIRDNET_WINDOW_SEC
SLIDE_SEC = BIRDNET_SLIDE_SEC

# ── Suppress TFLite noise ──────────────────────────────
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TFLITE_MIN_LOG_LEVEL", "3")

from birdnetlib.analyzer import Analyzer

# Global analyzer — lazy init
_ANALYZER: Optional[Analyzer] = None


def get_analyzer(species_list_path: Optional[str] = None) -> Analyzer:
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    import io
    from contextlib import redirect_stdout, redirect_stderr
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        if species_list_path:
            _ANALYZER = Analyzer(custom_species_list_path=species_list_path)
        else:
            _ANALYZER = Analyzer()
    return _ANALYZER


# ── Audio helpers ──────────────────────────────────────

def audio_to_model_input(audio_16k: np.ndarray) -> np.ndarray:
    """Resample float32 audio at 16kHz → 48kHz, pad/trim to MODEL_INPUT_SAMPLES."""
    from scipy.interpolate import interp1d
    n_in = len(audio_16k)
    duration = n_in / 16000.0
    n_target = int(duration * MODEL_SAMPLE_RATE)
    t_old = np.linspace(0, duration, n_in, endpoint=False)
    t_new = np.linspace(0, duration, n_target, endpoint=False)
    # Clip to avoid floating-point interpolation boundary errors
    t_new = np.clip(t_new, 0, duration - 1e-12)
    interp = interp1d(t_old, audio_16k, kind="linear", copy=False,
                      assume_sorted=True, fill_value=0.0, bounds_error=False)
    audio_48k = interp(t_new).astype(np.float32)

    # Pad or trim to exactly MODEL_INPUT_SAMPLES
    if len(audio_48k) < MODEL_INPUT_SAMPLES:
        audio_48k = np.pad(audio_48k, (0, MODEL_INPUT_SAMPLES - len(audio_48k)))
    elif len(audio_48k) > MODEL_INPUT_SAMPLES:
        audio_48k = audio_48k[:MODEL_INPUT_SAMPLES]

    return audio_48k


def extract_embedding(analyzer: Analyzer, audio_48k: np.ndarray) -> np.ndarray:
    """Run the BirdNET model on *audio_48k* (shape: MODEL_INPUT_SAMPLES, float32).

    Returns 1024-dim float32 embedding from GLOBAL_AVG_POOL layer.
    """
    batch = audio_48k.reshape(1, -1).astype(np.float32)
    features = analyzer._return_embeddings(batch)  # (1, 1024)
    return features[0].astype(np.float32)


# ── Dense scanning ─────────────────────────────────────

def scan_wav(analyzer: Analyzer, wav_path: str, wav_name: str,
             method: str) -> Tuple[np.ndarray, List[Dict]]:
    """Sliding-window scan of a single WAV file.

    Reads the WAV (16kHz mono), slides a 3-second window with
    1.5-second step, extracts an embedding per window.

    Returns:
        embeddings: (n_windows, 1024) float32
        metadata: list of dicts with window info
    """
    emb_list: List[np.ndarray] = []
    meta_list: List[Dict] = []

    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]  # take first channel
    except Exception as e:
        print(f"    ⚠️  Cannot read {wav_name}: {e}", file=sys.stderr)
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    # Resample if not 16kHz (pipeline outputs 16kHz, but be safe)
    if sr != 16000:
        from scipy.interpolate import interp1d
        duration = len(audio) / sr
        n_new = int(duration * 16000)
        t_old = np.linspace(0, duration, len(audio), endpoint=False)
        t_new = np.linspace(0, duration, n_new, endpoint=False)
        interp = interp1d(t_old, audio, kind="linear", copy=False,
                          assume_sorted=True, fill_value=0.0)
        audio = interp(t_new).astype(np.float32)

    total_sec = len(audio) / 16000.0
    window_samples = int(WINDOW_SEC * 16000)     # 48000 samples at 16kHz
    slide_samples = int(SLIDE_SEC * 16000)        # 24000 samples at 16kHz

    for start_sample in range(0, len(audio), slide_samples):
        end_sample = start_sample + window_samples
        segment = audio[start_sample:end_sample].astype(np.float32)
        start_sec = start_sample / 16000.0

        # Skip windows shorter than 1 second
        if len(segment) < 16000:
            break

        # Resample segment to 48kHz & pad/trim to MODEL_INPUT_SAMPLES
        model_input = audio_to_model_input(segment)

        # Extract embedding
        try:
            emb = extract_embedding(analyzer, model_input)
        except Exception as e:
            print(f"    ⚠️  Embedding failed for {wav_name} @ {start_sec:.1f}s: {e}",
                  file=sys.stderr)
            continue

        emb_list.append(emb)
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
            )
        )

    if emb_list:
        embeddings = np.stack(emb_list, axis=0).astype(np.float32)
    else:
        embeddings = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

    return embeddings, meta_list


# ── Directory traversal ────────────────────────────────

def find_minute_dirs(
    base_dir: str,
    location: str,
    date_str: str,
    methods: List[str],
) -> List[Tuple[str, str, str]]:
    """Find all m_XX directories for a given date and methods."""
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


def process_date(
    analyzer: Analyzer,
    date_str: str,
    minute_dirs: List[Tuple[str, str, str]],
    out_dir: str,
) -> Dict[str, Any]:
    """Extract dense embeddings for all minute dirs of one date."""
    os.makedirs(out_dir, exist_ok=True)

    # Collect all embeddings per method
    by_method_emb: Dict[str, List[np.ndarray]] = {}
    by_method_meta: Dict[str, List[Dict]] = {}
    total_wavs = 0
    total_emb = 0

    for minute_dir, method, _ in minute_dirs:
        # Find all chunk WAVs (s_NNN_) or full WAVs (_sa.wav, _mono.wav)
        wav_files = sorted(
            f for f in os.listdir(minute_dir)
            if f.lower().endswith(".wav") and not f.startswith("._")
        )
        if not wav_files:
            continue

        print(f"  {method} {os.path.basename(os.path.dirname(minute_dir))}/"
              f"{os.path.basename(minute_dir)}: {len(wav_files)} WAVs",
              end="", flush=True)
        t0 = time.time()

        dir_emb: List[np.ndarray] = []
        dir_meta: List[Dict] = []
        for wav_name in wav_files:
            wav_path = os.path.join(minute_dir, wav_name)
            emb, meta = scan_wav(analyzer, wav_path, wav_name, method)
            if len(emb) > 0:
                dir_emb.append(emb)
                dir_meta.extend(meta)

        n_emb = sum(len(e) for e in dir_emb) if dir_emb else 0
        total_wavs += len(wav_files)
        total_emb += n_emb

        elapsed = time.time() - t0
        print(f" → {n_emb} embeddings  ({elapsed:.1f}s)", flush=True)

        if dir_emb:
            by_method_emb.setdefault(method, []).extend(dir_emb)
            by_method_meta.setdefault(method, []).extend(dir_meta)

    # Save per-method files
    summary: Dict[str, Any] = {"date": date_str, "methods": {}, "n_wavs": total_wavs}
    for method in sorted(by_method_emb.keys()):
        all_emb = np.concatenate(by_method_emb[method], axis=0).astype(np.float32)
        meta = by_method_meta.get(method, [])

        emb_path = os.path.join(out_dir, f"{date_str}_{method}.npy")
        np.save(emb_path, all_emb)

        meta_path = os.path.join(out_dir, f"{date_str}_{method}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

        summary["methods"][method] = {
            "n_embeddings": int(len(all_emb)),
            "embeddings_file": os.path.basename(emb_path),
            "metadata_file": os.path.basename(meta_path),
        }

    return summary


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract dense BirdNET embeddings from pipeline output")
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--date", type=str, required=True,
                        help="Date(s), comma-separated")
    parser.add_argument("--data-dir", type=str, default=ANALYSIS_OUTPUT)
    parser.add_argument("--methods", type=str,
                        default="bf_LabIR,bf_SPIR,sa,mono")
    parser.add_argument("--species-list", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    dates = [d.strip() for d in args.date.split(",")]
    methods = [m.strip() for m in args.methods.split(",")]
    out_dir = resolve_birdnet_out_dir(args.data_dir, args.location, args.output)

    print(f"Location: {args.location}  Dates: {dates}")
    print(f"Methods: {methods}")
    print(f"Window: {WINDOW_SEC}s  Slide: {SLIDE_SEC}s")
    print(f"Output: {out_dir}")

    # Load model once
    species_list = args.species_list
    if species_list and not os.path.isfile(species_list):
        print(f"⚠️  Species list not found: {species_list}", file=sys.stderr)
        species_list = None
    print("Loading BirdNET model (FP32)...")
    t_load = time.time()
    analyzer = get_analyzer(species_list)
    print(f"  ✓ Model loaded in {time.time() - t_load:.1f}s")

    all_summaries = []
    grand_start = time.time()

    for date_str in dates:
        print(f"\n{'='*60}")
        print(f"📅 Date: {date_str}")
        print(f"{'='*60}")
        t0 = time.time()

        minute_dirs = find_minute_dirs(args.data_dir, args.location,
                                       date_str, methods)
        print(f"Found {len(minute_dirs)} minute dir(s)")

        summary = process_date(analyzer, date_str, minute_dirs, out_dir)
        all_summaries.append(summary)

        print(f"  ⏱  {date_str} done in {time.time() - t0:.1f}s")

    # Write global summary
    summary_path = os.path.join(out_dir, "embeddings_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2, ensure_ascii=False)

    total_embs = sum(
        sum(m["n_embeddings"] for m in s["methods"].values())
        for s in all_summaries
    )
    total_wavs = sum(s.get("n_wavs", 0) for s in all_summaries)
    print(f"\n{'='*60}")
    print(f"✅ Done — {total_embs} embeddings from {total_wavs} WAV files")
    print(f"   Total time: {time.time() - grand_start:.1f}s")
    print(f"   Summary: {summary_path}")


if __name__ == "__main__":
    main()
