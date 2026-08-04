#!/usr/bin/env python3
"""
Phase 4: Noise Reference Embedding Extraction.

Reads pre-beamformed noise reference WAVs (provided by the researcher),
extracts BirdNET embeddings, and saves them for use as baseline noise
vectors in cluster analysis.

No beamforming needed — noise WAVs are already beamformed outputs
with _noise suffix (e.g. S12_000_noise.wav, SPIR1_02m_000_noise.wav).

Usage:
  python process_noise_reference.py --location 2A400   \
      --noise-dir /Volumes/WD2TB/sea-data/2A400/noise_references

Output:
  embeddings/noise_LabIR_embeddings.npy   + noise_LabIR_meta.json
  embeddings/noise_SPIR_embeddings.npy    + noise_SPIR_meta.json
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

try:
    from config import ANALYSIS_OUTPUT
except ImportError:
    ANALYSIS_OUTPUT = "/Volumes/WD2TB/sea-data"

EMBEDDING_DIM = 1024
MODEL_SAMPLE_RATE = 48000       # BirdNET native rate
MODEL_INPUT_SAMPLES = 144000    # 3 seconds at 48 kHz
WINDOW_SEC = 3.0
SLIDE_SEC = 1.5

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TFLITE_MIN_LOG_LEVEL", "3")

from birdnetlib.analyzer import Analyzer

# ── Globals ──────────────────────────────────────────────

_ANALYZER: Optional[Analyzer] = None


def _get_analyzer() -> Analyzer:
    global _ANALYZER
    if _ANALYZER is not None:
        return _ANALYZER
    import io
    from contextlib import redirect_stdout, redirect_stderr
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        _ANALYZER = Analyzer()
    return _ANALYZER


# ── Audio helpers ──────────────────────────────────────

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


# ── Scanning ───────────────────────────────────────────

def _scan_wav(wav_path: str, wav_name: str, group: str,
              azimuth: Optional[int] = None,
              elevation: Optional[int] = None) -> Tuple[np.ndarray, List[dict]]:
    """Dense sliding-window embedding extraction for one noise WAV."""
    analyzer = _get_analyzer()
    emb_list: List[np.ndarray] = []
    meta_list: List[dict] = []

    try:
        audio, sr = sf.read(wav_path, dtype="float32")
        if audio.ndim > 1:
            audio = audio[:, 0]
    except Exception as e:
        print(f"    ⚠️  Cannot read {wav_name}: {e}", file=sys.stderr)
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    # Resample to 16 kHz if needed
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
        meta_list.append({
            "wav": wav_name,
            "group": group,
            "type": "noise_reference",
            "start_sec": round(start_sec, 1),
            "end_sec": round(min(start_sec + WINDOW_SEC, total_sec), 1),
            "azimuth": azimuth,
            "elevation": elevation,
        })

    if emb_list:
        return np.stack(emb_list, axis=0).astype(np.float32), meta_list
    return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []


# ── Find noise WAVs ────────────────────────────────────

def _find_noise_wavs(noise_dir: str) -> Dict[str, List[Tuple[str, str, Optional[int], Optional[int]]]]:
    """Scan noise_references/ for _noise.wav files organised by group.

    Expected structure:
        noise_references/
            LabIR/
                S01_060_noise.wav
                S12_000_noise.wav
            SPIR/
                SPIR1_02m_000_noise.wav
                SPIR2_08m_180_r2_noise.wav
            sa/
                *_noise.wav
            mono/
                *_noise.wav

    Returns: dict  group_name → [(wav_path, wav_name, azimuth, elevation), ...]
    """
    if not os.path.isdir(noise_dir):
        return {}

    import re

    # Parse LabIR speaker/azimuth: LabIR(S{speaker}_{azimuth}) or just S{speaker}_{azimuth}
    _LABIR_PARSE = re.compile(r"S(\d{2})_(\d{3})")
    _LABIR_ELEVATION = {1: -45, 5: 0, 9: 45, 12: 90}
    # Parse SPIR1: SPIR1({dist}m_{azimuth})
    _SPIR1_PARSE = re.compile(r"SPIR1\((\d{2})m_(\d{3})\)")
    # Parse SPIR2: SPIR2({dist}m_{azimuth}_r{rep})
    _SPIR2_PARSE = re.compile(r"SPIR2\((\d{2})m_(\d{3})_r(\d)\)")

    groups: Dict[str, List[Tuple[str, str, Optional[int], Optional[int]]]] = {}

    for root, dirs, files in os.walk(noise_dir):
        for fname in files:
            if not fname.endswith("_noise.wav") or fname.startswith("._"):
                continue

            wav_path = os.path.join(root, fname)
            parent = os.path.basename(root)

            # Simple pass-through groups (sa, mono) — no direction metadata
            if parent in ("sa", "mono"):
                groups.setdefault(parent, []).append((wav_path, fname, None, None))
                continue

            # Beamforming groups — parse direction metadata
            if parent in ("LabIR", "SPIR"):
                group = parent
            else:
                # Deduce from filename pattern
                if "LabIR" in fname or "S" in fname.replace("_noise.wav", "").split("_")[-1]:
                    group = "LabIR"
                elif "SPIR1" in fname or "SPIR2" in fname:
                    group = "SPIR"
                else:
                    group = "unknown"

            # Parse azimuth/elevation if possible
            azimuth = None
            elevation = None
            m = _LABIR_PARSE.search(fname)
            if m:
                speaker = int(m.group(1))
                azimuth = int(m.group(2))
                elevation = _LABIR_ELEVATION.get(speaker)
            m = _SPIR1_PARSE.search(fname)
            if m:
                azimuth = int(m.group(2))
                elevation = 0
            m = _SPIR2_PARSE.search(fname)
            if m:
                azimuth = int(m.group(2))
                elevation = 0

            groups.setdefault(group, []).append((wav_path, fname, azimuth, elevation))

    return groups


# ── Main ────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract BirdNET embeddings from noise reference WAVs")
    parser.add_argument("--location", type=str, required=True,
                        help="Location ID (e.g. 2A400)")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Directory of noise _noise.wav files "
                             "(default: ANALYSIS_OUTPUT/location/noise_references)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for .npy and .json files "
                             "(default: ANALYSIS_OUTPUT/location/embeddings)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker threads for parallel extraction")
    args = parser.parse_args()

    noise_dir = args.noise_dir or os.path.join(
        ANALYSIS_OUTPUT, args.location, "noise_references"
    )
    output_dir = args.output_dir or os.path.join(
        ANALYSIS_OUTPUT, args.location, "embeddings"
    )

    if not os.path.isdir(noise_dir):
        print(f"❌ Noise directory not found: {noise_dir}")
        print("   Expected structure:\n"
              f"     {noise_dir}/\n"
              "       LabIR/S01_060_noise.wav\n"
              "       SPIR/SPIR1_02m_000_noise.wav")
        sys.exit(1)

    print(f"Location:  {args.location}")
    print(f"Noise dir: {noise_dir}")
    print(f"Output:    {output_dir}")

    # Find noise WAVs
    groups = _find_noise_wavs(noise_dir)
    if not groups:
        print("❌ No _noise.wav files found!")
        sys.exit(1)

    total_files = sum(len(v) for v in groups.values())
    print(f"\nFound {total_files} noise WAV(s) in {len(groups)} group(s):")
    for group_name, files in sorted(groups.items()):
        print(f"  {group_name}: {len(files)} file(s)")
        for _, fname, az, el in files:
            az_str = f"{az}°" if az is not None else "N/A"
            el_str = f"{el}°" if el is not None else "N/A"
            print(f"      {fname:<50s} azimuth={az_str:<6s} elevation={el_str}")

    # Warm model
    print("\nLoading BirdNET model...")
    t_load = time.time()
    _get_analyzer()
    print(f"  ✓ Model loaded in {time.time() - t_load:.1f}s")

    os.makedirs(output_dir, exist_ok=True)

    # Extract embeddings per group
    grand_start = time.time()

    from concurrent.futures import ThreadPoolExecutor, as_completed

    for group_name, files in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"📂 Group: {group_name}  ({len(files)} WAVs)")
        print(f"{'='*60}")

        all_emb: List[np.ndarray] = []
        all_meta: List[dict] = []
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_scan_wav, wav_path, fname, group_name, azimuth, elevation):
                    fname
                for wav_path, fname, azimuth, elevation in files
            }
            for future in as_completed(futures):
                fname = futures[future]
                try:
                    emb, meta = future.result()
                except Exception as e:
                    print(f"  ⚠️  {fname}: {e}", file=sys.stderr)
                    continue
                if len(emb) > 0:
                    all_emb.append(emb)
                    all_meta.extend(meta)

        if all_emb:
            all_emb_arr = np.concatenate(all_emb, axis=0).astype(np.float32)
        else:
            all_emb_arr = np.zeros((0, EMBEDDING_DIM), dtype=np.float32)

        emb_path = os.path.join(output_dir, f"noise_{group_name}_embeddings.npy")
        meta_path = os.path.join(output_dir, f"noise_{group_name}_meta.json")

        np.save(emb_path, all_emb_arr)
        with open(meta_path, "w") as f:
            json.dump(all_meta, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - t0
        print(f"  ✓ {len(all_emb_arr)} embeddings → {os.path.basename(emb_path)}")
        print(f"  ⏱  {group_name} done in {elapsed:.1f}s")

    grand_elapsed = time.time() - grand_start
    print(f"\n{'='*60}")
    print(f"✅ All noise reference embeddings saved to: {output_dir}")
    print(f"   Total time: {grand_elapsed:.1f}s")


if __name__ == "__main__":
    main()
