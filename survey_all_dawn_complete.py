import json
import shutil
from pathlib import Path
import soundfile as sf
import numpy as np
from scipy.signal import spectrogram
from scipy.ndimage import label
from concurrent.futures import ProcessPoolExecutor

EPS = 1e-12

def process_file_dawn(input_path_str, window_sec=2.0, hop_sec=1.0):
    input_path = Path(input_path_str)
    try:
        with sf.SoundFile(str(input_path)) as audio:
            x = audio.read(dtype="float32", always_2d=True)[:, 0]
            sr = int(audio.samplerate)
    except Exception as e:
        return {"path": input_path_str, "error": str(e)}

    band_hz = 200
    min_hz = 200
    max_hz = 4000
    frame_sec = 0.032
    nperseg = min(max(128, round(sr * frame_sec)), len(x))
    noverlap = min(round(nperseg * 0.75), nperseg - 1)
    f, t, power = spectrogram(x, fs=sr, window="hann", nperseg=nperseg, noverlap=noverlap, mode="psd", scaling="density")
    frame_hop_sec = float(t[1] - t[0])

    edges = np.arange(min_hz, max_hz + band_hz, band_hz)
    levels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (f >= lo) & (f < hi)
        levels.append(10 * np.log10(np.mean(power[mask], axis=0) + EPS))
    full_band_db = np.asarray(levels)

    rec_floor = np.nanpercentile(full_band_db, 50, axis=1, keepdims=True)
    excess_rec = full_band_db - rec_floor

    onset_db = 7.0
    active = excess_rec >= onset_db
    labels, n_labels = label(active, structure=np.ones((3, 3), dtype=int))

    events = []
    for cid in range(1, n_labels + 1):
        r, c = np.where(labels == cid)
        dur = (int(c.max()) - int(c.min()) + 1) * frame_hop_sec
        if dur < 0.030 or dur > 0.800:
            continue
        peak_db = float(np.max(excess_rec[r, c]))
        bands = int(len(np.unique(r)))
        cells = int(len(r))
        low_f = int(min_hz + r.min() * band_hz)
        high_f = int(min_hz + (r.max() + 1) * band_hz)
        
        is_bird = (
            (bands >= 2 and cells >= 6 and peak_db >= 7.5)
            or (bands == 1 and cells >= 5 and peak_db >= 8.5)
            or (bands == 1 and dur >= 0.080 and cells >= 8 and peak_db >= 7.5)
            or (peak_db >= 14.0 and cells >= 5)
        )
        
        events.append({
            "start_sec": float(c.min() * frame_hop_sec),
            "end_sec": float((c.max() + 1) * frame_hop_sec),
            "duration_sec": float(dur),
            "low_hz": low_f,
            "high_hz": high_f,
            "bands": bands,
            "cells": cells,
            "peak_db": round(peak_db, 2),
            "is_bird": is_bird,
        })

    bird_events = [e for e in events if e["is_bird"]]
    active_mask = excess_rec >= onset_db

    n_window = round(sr * window_sec)
    n_hop = round(sr * hop_sec)
    starts = list(range(0, len(x) - n_window + 1, n_hop))
    windows = []

    for idx, start_sample in enumerate(starts):
        w_start_sec = start_sample / sr
        w_end_sec = (start_sample + n_window) / sr
        
        w_bird_events = [
            e for e in bird_events
            if not (e["end_sec"] < w_start_sec or e["start_sec"] > w_end_sec)
        ]
        
        w_start_frame = round(w_start_sec / frame_hop_sec)
        w_end_frame = min(full_band_db.shape[1], round(w_end_sec / frame_hop_sec))
        w_active = active_mask[:, w_start_frame:w_end_frame]
        w_excess = excess_rec[:, w_start_frame:w_end_frame]
        
        fg_time_frac = float(np.mean(np.any(w_active, axis=0))) if w_active.shape[1] > 0 else 0.0
        w_max_excess = float(np.nanmax(w_excess)) if w_excess.size > 0 else 0.0
        
        candidate = (len(w_bird_events) == 0) and (fg_time_frac <= 0.05) and (w_max_excess < 12.0)
        score = 1.0 - min(1.0, fg_time_frac + 0.02 * max(0.0, w_max_excess))
        
        windows.append({
            "index": idx,
            "start_sec": round(w_start_sec, 3),
            "end_sec": round(w_end_sec, 3),
            "candidate": candidate,
            "score": round(score, 4),
            "fg_time_frac": round(fg_time_frac, 4),
            "max_excess_db": round(w_max_excess, 2),
            "n_bird_events": len(w_bird_events),
        })

    candidates = [w for w in windows if w["candidate"]]
    return {
        "path": input_path_str,
        "hour": input_path.parent.parent.name,
        "minute": input_path.parent.name,
        "filename": input_path.name,
        "n_windows": len(windows),
        "n_bird": len(bird_events),
        "n_candidates": len(candidates),
        "candidates": candidates,
        "error": None
    }

def main():
    root = Path("/rds/general/user/ri322/ephemeral/sea-work/2A400/2026-05-15/bf_LabIR")
    files_to_test = []
    for h in ["h_05", "h_06"]:
        h_path = root / h
        if h_path.exists():
            for m_path in sorted(h_path.glob("m_*")):
                wavs = sorted(m_path.glob("*(S05_000).wav"))
                if wavs:
                    files_to_test.append(str(wavs[0]))

    print(f"Scanning all {len(files_to_test)} Dawn recordings on 2026-05-15 (h_05 and h_06)...", flush=True)

    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_file_dawn, files_to_test))

    print("\n" + "="*110, flush=True)
    print(f"{'Hour':<6} | {'Minute':<7} | {'Filename':<35} | {'Biophony Ev':<11} | {'Candidates':<10} | Top Candidate Interval", flush=True)
    print("-" * 110, flush=True)

    for r in sorted(results, key=lambda x: (x.get("hour", ""), x.get("minute", ""), x.get("filename", ""))):
        if r.get("error"):
            print(f"{r['hour']:<6} | {r['minute']:<7} | {r['filename']:<35} | ERROR: {r['error']}", flush=True)
            continue
        cands = r["candidates"]
        top_cand = f"{cands[0]['start_sec']:.1f}-{cands[0]['end_sec']:.1f}s (score={cands[0]['score']:.3f})" if cands else "0 candidates (clean 0)"
        print(f"{r['hour']:<6} | {r['minute']:<7} | {r['filename']:<35} | {r['n_bird']:11d} | {r['n_candidates']:10d} | {top_cand}", flush=True)

    valid_files = [r for r in results if r.get("n_candidates", 0) > 0]
    print("\n" + "="*110, flush=True)
    print(f"Total Dawn recordings scanned: {len(results)}", flush=True)
    print(f"Dawn recordings with clean noise windows (>0 candidates): {len(valid_files)}", flush=True)
    for vf in sorted(valid_files, key=lambda x: x["n_candidates"], reverse=True):
        print(f"  -> {vf['hour']}/{vf['minute']} | {vf['filename']}: {vf['n_candidates']} clean 2.0s candidates", flush=True)

if __name__ == "__main__":
    main()
