#!/usr/bin/env python3
"""
A/B Benchmark: BirdNET CPU (Multi-threaded TFLite) vs GPU (Batched Keras)
Evaluated on 63 dense grid WAV files of 2A400 2026-04-22 (00-02-33).
Uses subprocess execution to guarantee zero environment variable cross-contamination.
"""

import os
import sys
import glob
import time
import json
import argparse
import subprocess
import numpy as np


def run_cpu_worker(wav_dir, out_file, workers=8, min_conf=0.20):
    import ctypes
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from birdnetlib.analyzer import Analyzer
    from birdnetlib.main import Recording
    import threading

    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    print(f"\n[Scenario A: CPU TFLite] Processing {len(wav_files)} WAVs with {workers} workers...")

    _local = threading.local()

    def get_thread_analyzer():
        if not hasattr(_local, "analyzer"):
            _local.analyzer = Analyzer()
        return _local.analyzer

    def process_one(wav_path):
        fname = os.path.basename(wav_path)
        t0 = time.time()
        analyzer = get_thread_analyzer()
        rec = Recording(analyzer, wav_path, min_conf=min_conf)
        rec.analyze()
        detections = []
        for d in rec.detections:
            detections.append({
                "common_name": d.get("common_name"),
                "confidence": float(d.get("confidence", 0.0)),
                "start_time": float(d.get("start_time", 0.0)),
                "end_time": float(d.get("end_time", 0.0)),
            })
        return fname, detections, time.time() - t0

    t_start = time.time()
    latencies = []
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_one, w): w for w in wav_files}
        for fut in as_completed(futures):
            fname, dets, lat = fut.result()
            results[fname] = dets
            latencies.append(lat)

    t_total = time.time() - t_start
    print(f"[Scenario A: CPU TFLite] Complete in {t_total:.2f}s (mean {np.mean(latencies):.2f}s/file)")

    out_data = {
        "walltime_s": t_total,
        "mean_latency_s": float(np.mean(latencies)),
        "detections": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f)


def run_gpu_worker(wav_dir, out_file, batch_size=256, min_conf=0.20):
    import ctypes
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    conda_lib = os.path.expanduser("~/miniforge3/envs/sea/lib")
    for so in ["libstdc++.so.6"]:
        try: ctypes.CDLL(os.path.join(conda_lib, so), mode=ctypes.RTLD_GLOBAL)
        except: pass
    for d in glob.glob(os.path.expanduser("~/miniforge3/envs/sea/lib/python3.11/site-packages/nvidia/*/lib")):
        for so in sorted(glob.glob(f"{d}/*.so*")):
            try: ctypes.CDLL(so, mode=ctypes.RTLD_GLOBAL)
            except: pass

    import tensorflow as tf
    import soundfile as sf
    import librosa

    print(f"\n[Scenario B: GPU Keras RTX 6000] Initializing TensorFlow on GPU...")
    print("  Visible GPUs:", tf.config.list_physical_devices("GPU"))

    CKPT_DIR = "/rds/general/user/ri322/ephemeral/sea-work/bacpipe-checkpoints/birdnet"
    LABEL_PATH = os.path.expanduser(
        "~/miniforge3/envs/sea/lib/python3.11/site-packages/birdnetlib/models/analyzer/BirdNET_GLOBAL_6K_V2.4_Labels.txt"
    )

    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip().split("_", 1)[-1] if "_" in line else line.strip() for line in f]

    t_load = time.time()
    prep = tf.saved_model.load(os.path.join(CKPT_DIR, "BirdNET_Preprocessor"))
    model = tf.keras.models.load_model(os.path.join(CKPT_DIR, "birdnetv2.4.keras"), compile=False)
    prep_fn = prep.signatures["serving_default"]
    print(f"[Scenario B: GPU Keras RTX 6000] Model loaded in {time.time() - t_load:.2f}s")

    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    t_start = time.time()

    all_clips = []
    clip_index = []

    for wav_path in wav_files:
        fname = os.path.basename(wav_path)
        sig, sr = sf.read(wav_path, dtype="float32")
        if sig.ndim > 1: sig = sig[:, 0]
        if sr != 48000: sig = librosa.resample(sig, orig_sr=sr, target_sr=48000)
        win_samples = 144000
        n_win = len(sig) // win_samples
        for k in range(n_win):
            all_clips.append(sig[k * win_samples : (k + 1) * win_samples])
            clip_index.append((fname, k, float(k * 3.0), float((k + 1) * 3.0)))

    all_clips = np.array(all_clips, dtype=np.float32)
    n_total_windows = len(all_clips)
    print(f"  Total windows to infer: {n_total_windows} across {len(wav_files)} files")

    # Warmup pass
    _ = model(prep_fn(tf.convert_to_tensor(all_clips[:8], dtype=tf.float32))["concatenate"], training=False)

    t_infer_start = time.time()
    all_confidences = []
    for i in range(0, n_total_windows, batch_size):
        batch = all_clips[i : i + batch_size]
        batch_tensor = tf.convert_to_tensor(batch, dtype=tf.float32)
        mel = prep_fn(batch_tensor)["concatenate"]
        logits = model(mel, training=False)
        probs = tf.sigmoid(logits).numpy()
        all_confidences.append(probs)

    all_confidences = np.vstack(all_confidences)
    t_infer_only = time.time() - t_infer_start
    t_total = time.time() - t_start

    results = {os.path.basename(w): [] for w in wav_files}
    for idx, (fname, k, s_time, e_time) in enumerate(clip_index):
        row = all_confidences[idx]
        above_thresh = np.where(row >= min_conf)[0]
        for c_idx in above_thresh:
            results[fname].append({
                "common_name": labels[c_idx],
                "confidence": float(row[c_idx]),
                "start_time": s_time,
                "end_time": e_time,
            })

    print(f"[Scenario B: GPU Keras RTX 6000] Inference only: {t_infer_only:.2f}s ({n_total_windows / t_infer_only:.1f} win/s)")
    print(f"[Scenario B: GPU Keras RTX 6000] Total E2E: {t_total:.2f}s ({len(wav_files) / t_total:.2f} files/s)")

    out_data = {
        "walltime_s": t_total,
        "inference_only_s": t_infer_only,
        "throughput_win_per_s": float(n_total_windows / t_infer_only),
        "detections": results,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(out_data, f)


def compare_ab(cpu_file, gpu_file, wav_dir, out_report):
    with open(cpu_file, "r") as f: cpu_res = json.load(f)
    with open(gpu_file, "r") as f: gpu_res = json.load(f)

    wav_files = sorted(glob.glob(os.path.join(wav_dir, "*.wav")))
    thresholds = [0.30, 0.40, 0.50]

    t_cpu = cpu_res["walltime_s"]
    t_gpu = gpu_res["walltime_s"]
    speedup_e2e = t_cpu / t_gpu
    speedup_infer = t_cpu / gpu_res["inference_only_s"]

    print("\n" + "=" * 70)
    print("      A/B SPEED & PARITY COMPARISON (CPU TFLite vs GPU Keras)")
    print("=" * 70)
    print(f"Total Walltime (63 files) : CPU = {t_cpu:.2f}s | GPU = {t_gpu:.2f}s (E2E Speedup: {speedup_e2e:.2f}x)")
    print(f"GPU Pure Inference (5040w): {gpu_res['inference_only_s']:.2f}s ({gpu_res['throughput_win_per_s']:.1f} win/s, {speedup_infer:.2f}x vs CPU)")
    print(f"Per 4-min WAV Inference   : CPU = {cpu_res['mean_latency_s']:.2f}s | GPU = {gpu_res['inference_only_s']/len(wav_files):.4f}s")

    parity_summary = {}
    print("\nDetection Counts Comparison across Thresholds:")
    print("Threshold    | CPU TFLite   | GPU Keras    | Diff    ")
    print("-" * 50)
    for tau in thresholds:
        cpu_count = sum(1 for f in wav_files for d in cpu_res["detections"].get(os.path.basename(f), []) if d["confidence"] >= tau)
        gpu_count = sum(1 for f in wav_files for d in gpu_res["detections"].get(os.path.basename(f), []) if d["confidence"] >= tau)
        parity_summary[f"tau_{tau:.2f}"] = {
            "cpu_detections": cpu_count,
            "gpu_detections": gpu_count,
            "diff": gpu_count - cpu_count,
        }
        print(f"tau = {tau:.2f}     | {cpu_count:<12} | {gpu_count:<12} | {gpu_count - cpu_count:+d}")

    mono_fname = "mono.wav"
    if mono_fname in cpu_res["detections"] and mono_fname in gpu_res["detections"]:
        print(f"\nSample Comparison on {mono_fname} (>= 0.30):")
        cpu_m = [f"{d['common_name']}: {d['confidence']:.3f} ({d['start_time']}s)" for d in cpu_res["detections"][mono_fname] if d["confidence"] >= 0.30]
        gpu_m = [f"{d['common_name']}: {d['confidence']:.3f} ({d['start_time']}s)" for d in gpu_res["detections"][mono_fname] if d["confidence"] >= 0.30]
        print("  CPU Detections:", cpu_m)
        print("  GPU Detections:", gpu_m)

    final_report = {
        "cpu_walltime_s": t_cpu,
        "cpu_mean_latency_s": cpu_res["mean_latency_s"],
        "gpu_walltime_s": t_gpu,
        "gpu_inference_only_s": gpu_res["inference_only_s"],
        "gpu_throughput_win_s": gpu_res["throughput_win_per_s"],
        "speedup_e2e": speedup_e2e,
        "speedup_pure_inference": speedup_infer,
        "parity_thresholds": parity_summary,
    }

    os.makedirs(os.path.dirname(out_report), exist_ok=True)
    with open(out_report, "w", encoding="utf-8") as f:
        json.dump(final_report, f, indent=2)
    print(f"\n[Report Saved] -> {out_report}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-mode", choices=["all", "cpu", "gpu"], default="all")
    parser.add_argument("--wav-dir", default="/rds/general/user/ri322/ephemeral/sea-scratch/temp_benchmark_grid/dense_61")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-conf", type=float, default=0.20)
    parser.add_argument("--out-report", default="/rds/general/user/ri322/ephemeral/sea-scratch/temp_benchmark_grid/cpu_vs_gpu_report.json")
    args = parser.parse_args()

    cpu_tmp = "/rds/general/user/ri322/ephemeral/sea-scratch/temp_benchmark_grid/cpu_raw.json"
    gpu_tmp = "/rds/general/user/ri322/ephemeral/sea-scratch/temp_benchmark_grid/gpu_raw.json"

    if args.worker_mode == "cpu":
        run_cpu_worker(args.wav_dir, cpu_tmp, workers=args.workers, min_conf=args.min_conf)
        return
    elif args.worker_mode == "gpu":
        run_gpu_worker(args.wav_dir, gpu_tmp, batch_size=args.batch_size, min_conf=args.min_conf)
        return

    # Master runner: launch isolated subprocesses
    py_bin = sys.executable
    script_path = os.path.abspath(__file__)

    print("=== Launching Scenario A (CPU TFLite, 8 workers) in Subprocess ===")
    subprocess.run([py_bin, script_path, "--worker-mode", "cpu", "--wav-dir", args.wav_dir, "--workers", str(args.workers), "--min-conf", str(args.min_conf)], check=True)

    print("\n=== Launching Scenario B (GPU Keras, RTX 6000) in Subprocess ===")
    subprocess.run([py_bin, script_path, "--worker-mode", "gpu", "--wav-dir", args.wav_dir, "--batch-size", str(args.batch_size), "--min-conf", str(args.min_conf)], check=True)

    # Compare
    compare_ab(cpu_tmp, gpu_tmp, args.wav_dir, args.out_report)


if __name__ == "__main__":
    main()
