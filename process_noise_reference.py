#!/usr/bin/env python3
"""
Phase 4: Noise Reference Embedding Extraction.

Reads pre-beamformed noise reference WAVs (provided by the researcher),
extracts BirdNET embeddings, and saves them for use as baseline noise
vectors in cluster analysis and noise-distance scoring.

Supports both flat layout and condition-partitioned layouts:
  Flat layout:
    noise_references/
      LabIR/*.wav
      SPIR/*.wav
      sa/*.wav
      mono/*.wav

  Multi-condition layout:
    noise_references/
      dawn/LabIR/  dawn/SPIR/  dawn/sa/  dawn/mono/
      day/...      dusk/...    night/...

Output:
  <output-dir>/noise_LabIR_embeddings.npy  + noise_LabIR_meta.json
  <output-dir>/noise_SPIR_embeddings.npy   + noise_SPIR_meta.json
  <output-dir>/noise_sa_embeddings.npy     + noise_sa_meta.json
  <output-dir>/noise_mono_embeddings.npy   + noise_mono_meta.json
"""

import argparse
import io
import json
import os
import re
import shutil
import sys
import threading
import time
from contextlib import redirect_stderr, redirect_stdout
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

import tensorflow as tf
from birdnetlib.analyzer import Analyzer


# ── Thread-local BirdNET Extractor ──────────────────────────────────────────

class BirdNetEmbeddingExtractor:
    """Thread-safe BirdNET feature extractor using TFLite interpreter."""

    def __init__(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            analyzer = Analyzer()
            self.model_path = analyzer.model_path

        # Preserve intermediate tensors to extract GLOBAL_AVG_POOL layer
        self.interp = tf.lite.Interpreter(
            model_path=self.model_path,
            experimental_preserve_all_tensors=True
        )
        self.interp.allocate_tensors()
        self.input_idx = self.interp.get_input_details()[0]["index"]

        # Dynamically locate the 1024-dim embedding layer
        self.emb_idx = None
        for t in self.interp.get_tensor_details():
            name = t["name"]
            shape = list(t["shape"])
            if "GLOBAL_AVG_POOL" in name or (len(shape) == 2 and shape[1] == 1024 and "Mean" in name):
                self.emb_idx = t["index"]
                break
        if self.emb_idx is None:
            self.emb_idx = 545  # Standard BirdNET global avg pool fallback

    def extract(self, audio_48k: np.ndarray) -> np.ndarray:
        """Run model on (144000,) float32 audio and return 1024-dim float32 vector."""
        batch = audio_48k.reshape(1, -1).astype(np.float32)
        self.interp.set_tensor(self.input_idx, batch)
        self.interp.invoke()
        emb = self.interp.get_tensor(self.emb_idx)
        return emb[0].astype(np.float32)


_thread_local = threading.local()


def _get_extractor() -> BirdNetEmbeddingExtractor:
    """Return a per-thread BirdNET extractor instance."""
    if not hasattr(_thread_local, "extractor"):
        _thread_local.extractor = BirdNetEmbeddingExtractor()
    return _thread_local.extractor


# ── Audio helpers ───────────────────────────────────────────────────────────

def _audio_to_model_input(audio_16k: np.ndarray) -> np.ndarray:
    """Resample 16 kHz -> 48 kHz, pad/trim to MODEL_INPUT_SAMPLES."""
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


# ── Scanning ────────────────────────────────────────────────────────────────

def _scan_wav(wav_path: str, wav_name: str, group: str,
              azimuth: Optional[int] = None,
              elevation: Optional[int] = None,
              condition: Optional[str] = None) -> Tuple[np.ndarray, List[dict]]:
    """Dense sliding-window embedding extraction for one noise WAV."""
    extractor = _get_extractor()

    try:
        audio, sr = sf.read(wav_path, dtype="float32")
    except Exception as e:
        print(f"  [ERROR] Cannot read {wav_name}: {e}", file=sys.stderr)
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    if audio.ndim > 1:
        audio = audio[:, 0]  # First channel if multichannel

    if sr != 16000:
        duration = len(audio) / float(sr)
        n_16k = int(duration * 16000)
        from scipy.interpolate import interp1d
        t_old = np.linspace(0, duration, len(audio), endpoint=False)
        t_new = np.linspace(0, duration, n_16k, endpoint=False)
        interp = interp1d(t_old, audio, kind="linear", copy=False,
                          assume_sorted=True, fill_value=0.0, bounds_error=False)
        audio = interp(t_new).astype(np.float32)

    total_sec = len(audio) / 16000.0
    win_samp = int(WINDOW_SEC * 16000)
    step_samp = int(SLIDE_SEC * 16000)

    emb_list: List[np.ndarray] = []
    meta_list: List[dict] = []

    # If audio is shorter than 1 window, process it as a single padded window
    if len(audio) < win_samp:
        model_input = _audio_to_model_input(audio)
        try:
            emb = extractor.extract(model_input)
            emb_list.append(emb)
            meta_list.append({
                "wav": wav_name,
                "group": group,
                "type": "noise_reference",
                "condition": condition,
                "start_sec": 0.0,
                "end_sec": round(total_sec, 2),
                "azimuth": azimuth,
                "elevation": elevation,
            })
        except Exception as e:
            print(f"  [ERROR] Embedding failed for {wav_name}: {e}", file=sys.stderr)

        if emb_list:
            return np.stack(emb_list, axis=0).astype(np.float32), meta_list
        return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []

    for start_sample in range(0, len(audio), step_samp):
        end_sample = start_sample + win_samp
        segment = audio[start_sample:end_sample]
        start_sec = start_sample / 16000.0

        if len(segment) < 16000:  # Skip slivers shorter than 1.0s
            break

        model_input = _audio_to_model_input(segment)
        try:
            emb = extractor.extract(model_input)
            emb_list.append(emb)
            meta_list.append({
                "wav": wav_name,
                "group": group,
                "type": "noise_reference",
                "condition": condition,
                "start_sec": round(start_sec, 2),
                "end_sec": round(min(start_sec + WINDOW_SEC, total_sec), 2),
                "azimuth": azimuth,
                "elevation": elevation,
            })
        except Exception:
            continue

    if emb_list:
        return np.stack(emb_list, axis=0).astype(np.float32), meta_list
    return np.zeros((0, EMBEDDING_DIM), dtype=np.float32), []


# ── Find noise WAVs ─────────────────────────────────────────────────────────

def _find_noise_wavs(noise_dir: str) -> Dict[str, List[Tuple[str, str, Optional[int], Optional[int], Optional[str]]]]:
    """Scan noise_references/ for _noise.wav files organised by group and condition."""
    if not os.path.isdir(noise_dir):
        return {}

    _LABIR_PARSE = re.compile(r"S(\d{2})_(\d{3})")
    _LABIR_ELEVATION = {1: -45, 5: 0, 9: 45, 12: 90}
    _SPIR1_PARSE = re.compile(r"SPIR1\((\d{2})m_(\d{3})\)")
    _SPIR2_PARSE = re.compile(r"SPIR2\((\d{2})m_(\d{3})_r(\d)\)")

    groups: Dict[str, List[Tuple[str, str, Optional[int], Optional[int], Optional[str]]]] = {}

    for root, dirs, files in os.walk(noise_dir):
        for fname in sorted(files):
            if not fname.endswith(".wav") or fname.startswith("._"):
                continue

            wav_path = os.path.join(root, fname)
            rel_parts = os.path.relpath(wav_path, noise_dir).split(os.sep)

            condition = None
            for p in rel_parts:
                if p in ("dawn", "day", "dusk", "night"):
                    condition = p
                    break

            group = "unknown"
            for p in rel_parts:
                if p in ("LabIR", "SPIR", "sa", "mono"):
                    group = p
                    break

            if group == "unknown":
                parent = os.path.basename(root)
                if parent in ("LabIR", "SPIR", "sa", "mono"):
                    group = parent
                elif "LabIR" in fname or "S" in fname.replace("_noise.wav", "").split("_")[-1]:
                    group = "LabIR"
                elif "SPIR" in fname:
                    group = "SPIR"
                elif "sa" in fname:
                    group = "sa"
                elif "mono" in fname:
                    group = "mono"

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

            groups.setdefault(group, []).append((wav_path, fname, azimuth, elevation, condition))

    return groups


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract BirdNET embeddings from noise reference WAVs")
    parser.add_argument("--location", type=str, required=True,
                        help="Location ID (e.g. 2A400)")
    parser.add_argument("--noise-dir", type=str, default=None,
                        help="Directory of noise _noise.wav files")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for .npy and .json files")
    parser.add_argument("--workers", type=int, default=4,
                        help="Worker threads for parallel extraction")
    parser.add_argument("--copy-to-bacpipe", action="store_true", default=True,
                        help="Copy generated noise embeddings into bacpipe model directories")
    args = parser.parse_args()

    noise_dir = args.noise_dir or os.path.join(
        ANALYSIS_OUTPUT, args.location, "noise_references"
    )
    output_dir = args.output_dir or os.path.join(
        ANALYSIS_OUTPUT, args.location, "embeddings", "noise_references"
    )

    if not os.path.isdir(noise_dir):
        print(f"❌ Noise directory not found: {noise_dir}")
        sys.exit(1)

    print(f"Location:  {args.location}")
    print(f"Noise dir: {noise_dir}")
    print(f"Output:    {output_dir}")

    groups = _find_noise_wavs(noise_dir)
    if not groups:
        print("❌ No WAV files found in noise directory!")
        sys.exit(1)

    total_files = sum(len(v) for v in groups.values())
    print(f"\nFound {total_files} noise WAV(s) in {len(groups)} group(s):")
    for group_name, files in sorted(groups.items()):
        print(f"  {group_name}: {len(files)} file(s)")

    print("\nWarming BirdNET model...")
    t_load = time.time()
    _get_extractor()
    print(f"  ✓ Model ready in {time.time() - t_load:.1f}s")

    os.makedirs(output_dir, exist_ok=True)

    grand_start = time.time()
    from concurrent.futures import ThreadPoolExecutor, as_completed

    saved_npy_paths = []

    for group_name, files in sorted(groups.items()):
        print(f"\n{'='*60}")
        print(f"📂 Group: {group_name} ({len(files)} WAVs)")
        print(f"{'='*60}")

        all_emb: List[np.ndarray] = []
        all_meta: List[dict] = []
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {
                pool.submit(_scan_wav, wav_path, fname, group_name, azimuth, elevation, condition):
                    fname
                for wav_path, fname, azimuth, elevation, condition in files
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
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(all_meta, f, indent=2, ensure_ascii=False)

        saved_npy_paths.append((group_name, emb_path, meta_path))
        elapsed = time.time() - t0
        print(f"  ✓ {len(all_emb_arr)} embeddings (shape: {all_emb_arr.shape}) -> {os.path.basename(emb_path)}")
        print(f"  ⏱  {group_name} completed in {elapsed:.1f}s")

    # Optionally mirror embeddings into bacpipe model directories
    if args.copy_to_bacpipe:
        bacpipe_dir = os.path.join(ANALYSIS_OUTPUT, args.location, "embeddings", "bacpipe")
        if os.path.isdir(bacpipe_dir):
            for model_dir in os.listdir(bacpipe_dir):
                target_model_path = os.path.join(bacpipe_dir, model_dir)
                if os.path.isdir(target_model_path):
                    for group_name, emb_path, meta_path in saved_npy_paths:
                        dest_emb = os.path.join(target_model_path, os.path.basename(emb_path))
                        dest_meta = os.path.join(target_model_path, os.path.basename(meta_path))
                        try:
                            shutil.copy2(emb_path, dest_emb)
                            shutil.copy2(meta_path, dest_meta)
                        except Exception:
                            pass

    grand_elapsed = time.time() - grand_start
    print(f"\n{'='*60}")
    print(f"✅ All noise reference embeddings successfully saved to: {output_dir}")
    print(f"   Total elapsed: {grand_elapsed:.1f}s")


if __name__ == "__main__":
    main()
