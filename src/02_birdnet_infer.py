"""
Module 2: BirdNET Batch Inference (GPU-accelerated with CPU fallback).
Runs Cornell BirdNET V2.4 inference across all rendered WAV files (Mono, SA, LabIR, SPIR)
in a scratch recording folder, outputting a complete results.json.
"""

import os
import sys
import glob
import time
import json
import ctypes
import argparse
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional

# Set environment before any library loads
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Dynamic library preloading for CUDA 12 and Conda libstdc++
_conda_lib = os.path.expanduser("~/miniforge3/envs/sea/lib")
for _so in ["libstdc++.so.6"]:
    _p = os.path.join(_conda_lib, _so)
    if os.path.exists(_p):
        try: ctypes.CDLL(_p, mode=ctypes.RTLD_GLOBAL)
        except Exception: pass

_nv_pattern = os.path.expanduser("~/miniforge3/envs/sea/lib/python3.11/site-packages/nvidia/*/lib")
for _d in glob.glob(_nv_pattern):
    for _so in sorted(glob.glob(f"{_d}/*.so*")):
        try: ctypes.CDLL(_so, mode=ctypes.RTLD_GLOBAL)
        except Exception: pass

# Checkpoint paths
CKPT_DIR = os.environ.get(
    "BIRDNET_CKPT_DIR",
    "/rds/general/user/ri322/ephemeral/sea-work/bacpipe-checkpoints/birdnet"
)
LABEL_PATH = os.path.join(CKPT_DIR, "BirdNET_GLOBAL_6K_V2.4_Labels.txt")
MODEL_PATH = os.path.join(CKPT_DIR, "birdnetv2.4.keras")
PREP_PATH = os.path.join(CKPT_DIR, "BirdNET_Preprocessor")

# Check GPU availability
HAS_GPU = os.path.exists("/dev/nvidia0") and os.path.exists(MODEL_PATH) and os.path.exists(PREP_PATH)

if HAS_GPU:
    import tensorflow as tf
    import soundfile as sf
    import scipy.signal


def is_gpu_available() -> bool:
    return HAS_GPU


_GPU_MODEL = None
_GPU_PREP_FN = None
_LABELS = None


def get_birdnet_gpu_model():
    global _GPU_MODEL, _GPU_PREP_FN, _LABELS
    if _LABELS is None:
        if not os.path.exists(LABEL_PATH):
            lbl_fallback = os.path.expanduser(
                "~/miniforge3/envs/sea/lib/python3.11/site-packages/birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
            )
            with open(lbl_fallback, "r", encoding="utf-8") as f:
                _LABELS = [line.strip().split("_", 1)[-1] if "_" in line else line.strip() for line in f]
        else:
            with open(LABEL_PATH, "r", encoding="utf-8") as f:
                _LABELS = [line.strip().split("_", 1)[-1] if "_" in line else line.strip() for line in f]

    if _GPU_MODEL is None:
        t0_model = time.time()
        prep = tf.saved_model.load(PREP_PATH)
        _GPU_MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        _GPU_PREP_FN = prep.signatures["serving_default"]
        print(f"   ✓ BirdNET GPU model & preprocessor loaded in {time.time() - t0_model:.2f}s")
        # Warmup
        dummy = np.zeros((1, 144000), dtype=np.float32)
        _ = _GPU_MODEL(_GPU_PREP_FN(tf.convert_to_tensor(dummy))["concatenate"], training=False)

    return _GPU_MODEL, _GPU_PREP_FN, _LABELS


def run_birdnet_gpu(folder_path: str, batch_size: int = 256, min_conf: float = 0.05, out_json: Optional[str] = None) -> str:
    wav_files = sorted(glob.glob(os.path.join(folder_path, "*.wav")))
    if not wav_files:
        print(f"⚠️  No WAV files found in: {folder_path}")
        return ""

    model, prep_fn, labels = get_birdnet_gpu_model()
    print(f"🚀 Running GPU BirdNET on {len(wav_files)} WAV files (batch_size={batch_size}, min_conf={min_conf})...")

    # Read and resample WAVs
    t0_load = time.time()
    all_clips = []
    clip_index = []
    for w in wav_files:
        fname = os.path.basename(w)
        sig, sr = sf.read(w, dtype="float32")
        if sig.ndim > 1:
            sig = sig[:, 0]
        if sr == 16000:
            sig = scipy.signal.resample_poly(sig, 3, 1).astype(np.float32)
        elif sr != 48000:
            import librosa
            sig = librosa.resample(sig, orig_sr=sr, target_sr=48000)

        win_samples = 144000
        n_win = len(sig) // win_samples
        for k in range(n_win):
            all_clips.append(sig[k * win_samples : (k + 1) * win_samples])
            clip_index.append((fname, k, float(k * 3.0), float((k + 1) * 3.0)))

    all_clips_arr = np.array(all_clips, dtype=np.float32)
    n_windows = len(all_clips_arr)
    t_load = time.time() - t0_load

    # Batched inference
    t0_infer = time.time()
    conf_list = []
    for i in range(0, n_windows, batch_size):
        b = tf.convert_to_tensor(all_clips_arr[i : i + batch_size], dtype=tf.float32)
        mel = prep_fn(b)["concatenate"]
        logits = model(mel, training=False)
        probs = tf.sigmoid(logits).numpy()
        conf_list.append(probs)

    all_probs = np.vstack(conf_list)
    t_infer = time.time() - t0_infer

    # Format detections
    t0_extract = time.time()
    results_dict = {os.path.basename(w): [] for w in wav_files}
    for idx, (fname, k, s_time, e_time) in enumerate(clip_index):
        row = all_probs[idx]
        above = np.where(row >= min_conf)[0]
        for c_idx in above:
            results_dict[fname].append({
                "common_name": labels[c_idx],
                "confidence": round(float(row[c_idx]), 4),
                "start_time": s_time,
                "end_time": e_time,
            })
    t_extract = time.time() - t0_extract

    if out_json is None:
        out_json = os.path.join(folder_path, "results.json")

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=4, ensure_ascii=False)

    total_t = t_load + t_infer + t_extract
    print(f"✅ GPU BirdNET complete in {total_t:.2f}s (Load: {t_load:.2f}s | Infer: {t_infer:.2f}s [{n_windows/t_infer:.1f} win/s]) -> {out_json}")
    return out_json


def run_birdnet_cpu(folder_path: str, date_obj: Optional[datetime] = None, processes: int = 8, min_conf: float = 0.05, out_json: Optional[str] = None) -> str:
    ffmpeg_bin = os.path.expanduser("~/miniforge3/envs/sea/bin")
    if os.path.exists(ffmpeg_bin) and ffmpeg_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = ffmpeg_bin + ":" + os.environ.get("PATH", "")

    from birdnetlib.analyzer import Analyzer
    from birdnetlib.batch import DirectoryMultiProcessingAnalyzer

    if date_obj is None:
        date_obj = datetime(2026, 4, 22)

    if out_json is None:
        out_json = os.path.join(folder_path, "results.json")

    results_dict = {}

    def on_complete(recordings):
        for rec in recordings:
            if rec.error:
                print(f"⚠️  Error in {os.path.basename(rec.path)}: {rec.error_message}")
            else:
                file_name = os.path.basename(rec.path)
                results_dict[file_name] = [
                    {
                        "common_name": d.get("common_name"),
                        "confidence": round(float(d.get("confidence", 0.0)), 4),
                        "start_time": float(d.get("start_time", 0.0)),
                        "end_time": float(d.get("end_time", 0.0)),
                    }
                    for d in rec.detections
                    if float(d.get("confidence", 0.0)) >= min_conf
                ]

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4, ensure_ascii=False)
        print(f"✅ Saved BirdNET CPU results ({len(results_dict)} channels) to: {out_json}")

    analyzer = Analyzer()
    batch = DirectoryMultiProcessingAnalyzer(
        folder_path,
        analyzers=[analyzer],
        date=date_obj,
        min_conf=min_conf,
        overlap=0.0,
        processes=processes
    )
    batch.on_analyze_directory_complete = on_complete
    batch.process()
    return out_json


def run_birdnet_batch(
    folder_path: str,
    date_obj: Optional[datetime] = None,
    processes: int = 8,
    device: str = "auto",
    batch_size: int = 256,
    min_conf: float = 0.05,
    out_json: Optional[str] = None
) -> str:
    target_device = device.lower()
    if target_device == "auto":
        target_device = "gpu" if is_gpu_available() else "cpu"

    if target_device == "gpu":
        return run_birdnet_gpu(folder_path, batch_size=batch_size, min_conf=min_conf, out_json=out_json)
    else:
        return run_birdnet_cpu(folder_path, date_obj=date_obj, processes=processes, min_conf=min_conf, out_json=out_json)


def main():
    parser = argparse.ArgumentParser(description="Run BirdNET batch inference on rendered WAV folder.")
    parser.add_argument("folder", help="Path to recording folder containing rendered WAV files")
    parser.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto", help="Compute device (default: auto)")
    parser.add_argument("--batch-size", type=int, default=256, help="GPU batch size (default: 256)")
    parser.add_argument("--min-conf", type=float, default=0.05, help="Minimum confidence threshold (default: 0.05)")
    parser.add_argument("--processes", type=int, default=8, help="Number of CPU worker processes (for CPU mode)")
    parser.add_argument("--out", default=None, help="Output JSON path (default: <folder>/results.json)")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"❌ Directory not found: {args.folder}")
        return 1

    t0 = time.time()
    out_path = run_birdnet_batch(
        args.folder,
        processes=args.processes,
        device=args.device,
        batch_size=args.batch_size,
        min_conf=args.min_conf,
        out_json=args.out
    )
    print(f"🏁 Finished in {time.time() - t0:.2f}s -> {out_path}")


if __name__ == "__main__":
    main()
