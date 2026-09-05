"""
A/B Benchmark Experiment: Coarse LabIR Grid (19 Beams) vs Dense LabIR Grid (61 Beams).
Evaluates compute latency, spatial detection gain, species richness, and confidence boosts
on real 4-minute 6-channel rainforest soundscapes (2A400 / 2026-04-22).
"""

import os
import sys
import time
import json
import shutil
import multiprocessing
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa

PROJECT_ROOT = "/rds/general/user/ri322/home/spatial-ecoacoustic-analysis"
if os.path.join(PROJECT_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import config
from config import (
    FS_TARGET, HIGH_PASS_CUTOFF, FRAME_LEN, HOP_LEN,
    IR_WINDOW_LEN, IR_PRE_PEAK_SAMPLES, IR_BASE_PATH
)
import importlib
render_mod = importlib.import_module("01_render_signals")
infer_mod = importlib.import_module("02_birdnet_infer")
extract_mod = importlib.import_module("03_extract_detections")
pair_mod = importlib.import_module("04_pair_and_recap")

# -------------------------------------------------------------
# GRID DEFINITIONS
# -------------------------------------------------------------
# 1. Coarse Grid: (3 elevations x 6 azimuths) + 1 zenith = 19 beams
COARSE_SPEAKERS = [1, 5, 9]
COARSE_AZIMUTHS = [0, 60, 120, 180, 240, 300]

# 2. Dense Grid: (5 elevations x 12 azimuths) + 1 zenith = 61 beams
DENSE_SPEAKERS = [1, 3, 5, 7, 9]
DENSE_AZIMUTHS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]

def build_grid_catalog(speakers, azimuths):
    beams = []
    labir_folder = os.path.join(IR_BASE_PATH, "Lab_IR")
    for spk in speakers:
        for deg in azimuths:
            fname = f"LabIR(S{spk:02d}_{deg:03d}).wav"
            ir_file = f"Lab_IR_S{spk:02d}_{deg:03d}.wav"
            beams.append((fname, os.path.join(labir_folder, ir_file)))
    # Add zenith S12_000
    fname = "LabIR(S12_000).wav"
    ir_file = "Lab_IR_S12_000.wav"
    beams.append((fname, os.path.join(labir_folder, ir_file)))
    return beams

COARSE_CATALOG = build_grid_catalog(COARSE_SPEAKERS, COARSE_AZIMUTHS)
DENSE_CATALOG = build_grid_catalog(DENSE_SPEAKERS, DENSE_AZIMUTHS)

def load_weights_tensor(catalog):
    weights = [render_mod.get_onset_steering_weights(ir_p) for _, ir_p in catalog]
    return np.stack(weights, axis=0)

COARSE_W_TENSOR = load_weights_tensor(COARSE_CATALOG)
DENSE_W_TENSOR = load_weights_tensor(DENSE_CATALOG)

def _worker_istft_save(task):
    spec, out_path, hop_len, fs = task
    z = librosa.istft(spec, hop_length=hop_len, window='hamming')
    z = z / (np.max(np.abs(z)) + 1e-12)
    sf.write(out_path, z.astype(np.float32), fs)

def render_grid(flac_path, output_dir, catalog, W_tensor, workers=8):
    os.makedirs(output_dir, exist_ok=True)
    audio_raw, sr = sf.read(flac_path)
    if sr != FS_TARGET:
        audio_raw = librosa.resample(audio_raw.T, orig_sr=sr, target_sr=FS_TARGET).T
    audio_filt = render_mod.butter_highpass_filter(audio_raw, cutoff=HIGH_PASS_CUTOFF, fs=FS_TARGET)
    
    # Mono & SA
    mono = audio_filt[:, 0] / (np.max(np.abs(audio_filt[:, 0])) + 1e-12)
    sa = np.mean(audio_filt, axis=1) / (np.max(np.abs(np.mean(audio_filt, axis=1))) + 1e-12)
    sf.write(os.path.join(output_dir, "mono.wav"), mono.astype(np.float32), FS_TARGET)
    sf.write(os.path.join(output_dir, "sa.wav"), sa.astype(np.float32), FS_TARGET)
    
    # 6-channel STFT
    X = np.stack([
        librosa.stft(audio_filt[:, ch], n_fft=FRAME_LEN, hop_length=HOP_LEN, window='hamming')
        for ch in range(6)
    ], axis=0)
    
    # Vectorized matrix contraction
    Z_all = np.einsum('kcf, cft -> kft', W_tensor, X)
    
    # Parallel ISTFT
    tasks = [
        (Z_all[i], os.path.join(output_dir, catalog[i][0]), HOP_LEN, FS_TARGET)
        for i in range(len(catalog))
    ]
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(_worker_istft_save, tasks)

def main():
    flac_test = "/rds/general/user/ri322/ephemeral/monitoring_data/RPiID-0000000091668b26/2026-04-22/00-02-33_dur=240secs.flac"
    base_scratch = "/rds/general/user/ri322/ephemeral/sea-scratch/temp_benchmark_grid"
    dir_coarse = os.path.join(base_scratch, "coarse_19")
    dir_dense = os.path.join(base_scratch, "dense_61")
    
    for d in [dir_coarse, dir_dense]:
        if os.path.exists(d):
            shutil.rmtree(d)
        os.makedirs(d, exist_ok=True)
        
    print("=" * 75)
    print("🔬 SPATIAL ECOACOUSTIC ANALYSIS: 19-BEAM vs 61-BEAM LABIR BENCHMARK")
    print(f"Target file: {flac_test}")
    print(f"Method 1 (Coarse Grid): 19 LabIR Beams (3 elevations x 6 azimuths + zenith) + Mono + SA = 21 WAVs")
    print(f"Method 2 (Dense Grid):  61 LabIR Beams (5 elevations x 12 azimuths + zenith) + Mono + SA = 63 WAVs")
    print("=" * 75)
    
    workers = int(os.environ.get("BENCH_WORKERS", multiprocessing.cpu_count()))
    print(f"Workers: {workers} CPU processes")
    
    # --- METHOD 1: COARSE 19 BEAMS ---
    print("\n--- [1/2] RUNNING COARSE GRID (19 LabIR Beams) ---")
    t0_r1 = time.time()
    render_grid(flac_test, dir_coarse, COARSE_CATALOG, COARSE_W_TENSOR, workers=workers)
    t_r1 = time.time() - t0_r1
    print(f"⏱️  Coarse Render Time: {t_r1:.2f}s ({len(os.listdir(dir_coarse))} WAVs)")
    
    t0_i1 = time.time()
    infer_mod.run_birdnet_batch(dir_coarse, processes=workers)
    t_i1 = time.time() - t0_i1
    print(f"⏱️  Coarse BirdNET Time: {t_i1:.2f}s")
    
    proc_coarse = extract_mod.process_results_file(os.path.join(dir_coarse, "results.json"))
    with open(os.path.join(dir_coarse, "processed.json"), "w") as f:
        json.dump(proc_coarse, f, indent=4)
    sum_coarse = pair_mod.evaluate_threshold_counts(proc_coarse, pair_mod.DEFAULT_THRESHOLDS)
    
    # --- METHOD 2: DENSE 61 BEAMS ---
    print("\n--- [2/2] RUNNING DENSE GRID (61 LabIR Beams) ---")
    t0_r2 = time.time()
    render_grid(flac_test, dir_dense, DENSE_CATALOG, DENSE_W_TENSOR, workers=workers)
    t_r2 = time.time() - t0_r2
    print(f"⏱️  Dense Render Time: {t_r2:.2f}s ({len(os.listdir(dir_dense))} WAVs)")
    
    t0_i2 = time.time()
    infer_mod.run_birdnet_batch(dir_dense, processes=workers)
    t_i2 = time.time() - t0_i2
    print(f"⏱️  Dense BirdNET Time: {t_i2:.2f}s")
    
    proc_dense = extract_mod.process_results_file(os.path.join(dir_dense, "results.json"))
    with open(os.path.join(dir_dense, "processed.json"), "w") as f:
        json.dump(proc_dense, f, indent=4)
    sum_dense = pair_mod.evaluate_threshold_counts(proc_dense, pair_mod.DEFAULT_THRESHOLDS)
    
    # --- COMPARISON & REPORT ---
    print("\n" + "=" * 75)
    print("📊 DETECTION & COMPUTATIONAL COMPARISON: 19 BEAMS vs 61 BEAMS")
    print("=" * 75)
    
    thresholds = pair_mod.DEFAULT_THRESHOLDS
    print(f"| Threshold | Mono Det | SA Det | 19-Beam LabIR | 61-Beam LabIR | Gain (61 vs 19) | 19 Spp | 61 Spp |")
    print(f"|-----------|----------|--------|---------------|---------------|-----------------|--------|--------|")
    
    comparison_rows = []
    for t in thresholds:
        t_str = f"{t:.2f}"
        mono_n = sum_coarse["mono_channel"]["total_detections"][t_str]
        sa_n = sum_coarse["sa_channel"]["total_detections"][t_str]
        c19_n = sum_coarse["beamformed_LabIR"]["total_detections"][t_str]
        d61_n = sum_dense["beamformed_LabIR"]["total_detections"][t_str]
        
        c19_sp = sum_coarse["beamformed_LabIR"]["species_count"][t_str]
        d61_sp = sum_dense["beamformed_LabIR"]["species_count"][t_str]
        
        diff = d61_n - c19_n
        pct_diff = f"{diff:+d} ({(diff/c19_n*100):+.1f}%)" if c19_n > 0 else (f"{diff:+d}" if diff > 0 else "0")
        print(f"|   {t_str}    | {mono_n:8d} | {sa_n:6d} | {c19_n:13d} | {d61_n:13d} | {pct_diff:15s} | {c19_sp:6d} | {d61_sp:6d} |")
        
        comparison_rows.append({
            "threshold": t,
            "mono_detections": mono_n,
            "sa_detections": sa_n,
            "coarse_19_detections": c19_n,
            "dense_61_detections": d61_n,
            "coarse_19_species": c19_sp,
            "dense_61_species": d61_sp
        })
        
    print("=" * 75)
    print(f"\n⏱️  Latency Breakdown:")
    print(f"  - 19 Beams (21 WAVs): Render = {t_r1:.2f}s | BirdNET = {t_i1:.2f}s | Total = {t_r1+t_i1:.2f}s")
    print(f"  - 61 Beams (63 WAVs): Render = {t_r2:.2f}s | BirdNET = {t_i2:.2f}s | Total = {t_r2+t_i2:.2f}s")
    print(f"  - Overhead factor for 3.2x more beams: {(t_r2+t_i2)/(t_r1+t_i1):.2f}x total walltime")
    
    # Species-level confidence analysis
    print("\n🔍 Species Confidence & Direction Analysis (LabIR 61 vs 19):")
    sp_coarse = proc_coarse["beamformed_LabIR"]
    sp_dense = proc_dense["beamformed_LabIR"]
    all_spp = sorted(set(list(sp_coarse.keys()) + list(sp_dense.keys())))
    
    for sp in all_spp:
        c_max = sp_coarse.get(sp, {}).get("conf_max", 0.0)
        d_max = sp_dense.get(sp, {}).get("conf_max", 0.0)
        c_cnt = sp_coarse.get(sp, {}).get("count", 0)
        d_cnt = sp_dense.get(sp, {}).get("count", 0)
        
        delta = d_max - c_max
        star = "🌟 (NEW / HIGHER CONF)" if delta > 0.02 else ""
        print(f"  • {sp:30s} | 19-beam: max={c_max:.3f} (N={c_cnt}) | 61-beam: max={d_max:.3f} (N={d_cnt}) | diff={delta:+.3f} {star}")
        
    # Save full JSON report
    report = {
        "file": os.path.basename(flac_test),
        "coarse_19": {
            "num_beams": 19,
            "render_sec": round(t_r1, 2),
            "infer_sec": round(t_i1, 2),
            "total_sec": round(t_r1 + t_i1, 2)
        },
        "dense_61": {
            "num_beams": 61,
            "render_sec": round(t_r2, 2),
            "infer_sec": round(t_i2, 2),
            "total_sec": round(t_r2 + t_i2, 2)
        },
        "comparison_table": comparison_rows
    }
    
    out_json = os.path.join(base_scratch, "grid_benchmark_report.json")
    with open(out_json, "w") as f:
        json.dump(report, f, indent=4)
    print(f"\n💾 Full report saved to: {out_json}")

if __name__ == "__main__":
    main()
