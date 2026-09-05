#!/usr/bin/env python3
"""Auditable first-pass detector for 3-second LabIR background candidates.
The detector reads the supplied LabIR WAV, scans 500-Hz bands, measures
within-window stability/transients, and measures recurrence across one
recording. A conservative low/mid-band event veto flags structured activity
below 4 kHz for manual review. It writes review artefacts; it never writes
directly to bacpipe's official references.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import spectrogram, stft

EPS = 1e-10


def mad(x, axis=None):
    m = np.nanmedian(x, axis=axis, keepdims=True)
    return np.nanmedian(np.abs(x - m), axis=axis)


def load_ch0(path):
    with sf.SoundFile(str(path)) as f:
        sr, channels = int(f.samplerate), int(f.channels)
        data = f.read(dtype="float32", always_2d=True)
    if not data.size:
        raise ValueError("empty audio")
    return data[:, 0], sr, channels


def stft_features(x, sr, band_hz):
    nperseg = min(max(256, round(sr * 0.064)), len(x))
    noverlap = min(round(nperseg * 0.75), nperseg - 1)
    nfft = 2 ** int(np.ceil(np.log2(nperseg)))
    f, _, z = stft(x, fs=sr, window="hann", nperseg=nperseg,
                   noverlap=noverlap, nfft=nfft, detrend=False,
                   boundary=None, padded=False)
    power = np.abs(z).astype(np.float64) ** 2 + EPS
    nyq = sr / 2
    edges = np.arange(0, nyq + band_hz, band_hz)
    if edges[-1] < nyq:
        edges = np.append(edges, nyq)
    med, iqr, flux = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (f >= lo) & ((f < hi) if hi < nyq else (f <= hi))
        p = power[mask]
        if not p.size:
            med.append(np.nan); iqr.append(np.nan); flux.append(np.nan)
            continue
        db = 10 * np.log10(np.mean(p, axis=0) + EPS)
        med.append(np.median(db))
        iqr.append(np.percentile(db, 75) - np.percentile(db, 25))
        q = p / (p.sum(axis=0, keepdims=True) + EPS)
        flux.append(np.median(np.sqrt(np.mean(np.diff(q, axis=1) ** 2, axis=0))))
    frame_db = 10 * np.log10(power.sum(axis=0) + EPS)
    threshold = np.median(frame_db) + max(6, 4 * mad(frame_db))
    return {
        "med": np.asarray(med), "iqr": np.asarray(iqr), "flux": np.asarray(flux),
        "rms_db": float(10 * np.log10(np.mean(x.astype(float) ** 2) + EPS)),
        "spectral_flux": float(np.nanmedian(flux)),
        "transient_ratio": float(np.mean(frame_db > threshold)),
        "energy_iqr_db": float(np.percentile(frame_db, 75) - np.percentile(frame_db, 25)),
        "edges": edges,
    }


def tonal_prominence(med, edges, min_hz, max_hz):
    """Return each eligible band's dB prominence over adjacent bands."""
    out = np.full(med.shape, np.nan, dtype=float)
    for j, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        if lo < min_hz or hi > max_hz:
            continue
        neighbours = []
        if j:
            neighbours.append(med[j - 1])
        if j + 1 < len(med):
            neighbours.append(med[j + 1])
        if neighbours and np.all(np.isfinite(neighbours + [med[j]])):
            out[j] = med[j] - np.median(neighbours)
    return out


def longest_true_run(mask):
    best = current = 0
    for value in mask:
        current = current + 1 if value else 0
        best = max(best, current)
    return best


def analyse(x, sr, window_sec=3, hop_sec=1.5, band_hz=500,
            persistence=0.75, threshold=0.58, bird_veto_hz=4000,
            low_mid_iqr_db=6, low_mid_unstable_fraction=0.25,
            low_mid_flux_percentile=90, tonal_prominence_db=6,
            low_mid_band_hz=200, low_mid_max_hz=4000,
            low_mid_excess_db=6, low_mid_mad_multiplier=2,
            low_mid_cluster_excess_db=2, low_mid_cluster_bins=3):
    nwin, nhop = round(sr * window_sec), round(sr * hop_sec)
    starts = list(range(0, len(x) - nwin + 1, nhop))
    fs = [stft_features(x[s:s + nwin], sr, band_hz) for s in starts]
    matrix = np.stack([v["med"] for v in fs])
    fine_fs = [stft_features(x[s:s + nwin], sr, low_mid_band_hz) for s in starts]
    fine_matrix = np.stack([v["med"] for v in fine_fs])
    fine_baseline = np.nanmedian(fine_matrix, axis=0)
    fine_mad = 1.4826 * np.nanmedian(
        np.abs(fine_matrix - fine_baseline[None, :]), axis=0
    )
    fine_threshold = np.maximum(low_mid_excess_db, low_mid_mad_multiplier * fine_mad)
    fine_count = max(1, int(np.ceil(min(low_mid_max_hz, sr / 2) / low_mid_band_hz)))
    global_fine_tonal = tonal_prominence(
        fine_baseline, fine_fs[0]["edges"], 500, low_mid_max_hz
    )
    typical = np.nanmedian(matrix, axis=0)
    recurrent = np.nanmean(matrix >= typical[None, :] - 3, axis=0) >= persistence
    fluxes = np.asarray([v["spectral_flux"] for v in fs])
    flo, fhi = np.nanpercentile(fluxes, [10, 90])
    low_mid_count = max(1, int(np.ceil(min(bird_veto_hz, sr / 2) / band_hz)))
    low_mid_fluxes = np.asarray([
        np.nanmedian(v["flux"][:low_mid_count]) for v in fs
    ])
    lmlo, lmhi = np.nanpercentile(low_mid_fluxes, [10, 90])
    low_mid_flux_cutoff = float(np.nanpercentile(low_mid_fluxes, low_mid_flux_percentile))
    tonal_by_window = [
        tonal_prominence(v["med"], v["edges"], 500, bird_veto_hz) for v in fs
    ]
    fine_tonal_by_window = [
        tonal_prominence(v["med"], v["edges"], 500, low_mid_max_hz) for v in fine_fs
    ]
    globally_tonal = np.isfinite(global_fine_tonal) & (global_fine_tonal >= tonal_prominence_db)
    windows = []
    for i, (s, v) in enumerate(zip(starts, fs)):
        stable = v["iqr"] <= 6
        present = recurrent & (v["med"] >= typical - 3)
        stable_fraction = float(np.nanmean(stable))
        recurrent_fraction = float(np.nanmean(present))
        flux_norm = float(np.clip((v["spectral_flux"] - flo) / (fhi - flo + EPS), 0, 1))
        low_mid_flux = float(np.nanmedian(v["flux"][:low_mid_count]))
        low_mid_flux_norm = float(np.clip(
            (low_mid_flux - lmlo) / (lmhi - lmlo + EPS), 0, 1
        ))
        low_mid_unstable = float(np.nanmean(v["iqr"][:low_mid_count] > low_mid_iqr_db))
        low_mid_event = (
            low_mid_unstable >= low_mid_unstable_fraction
            or low_mid_flux >= low_mid_flux_cutoff
        )
        tonal = tonal_by_window[i]
        tonal_peak = float(np.nanmax(tonal)) if np.any(np.isfinite(tonal)) else float("nan")
        tonal_peak_idx = int(np.nanargmax(tonal)) if np.any(np.isfinite(tonal)) else -1
        tonal_event = bool(np.isfinite(tonal_peak) and tonal_peak >= tonal_prominence_db)
        fine_excess = fine_matrix[i] - fine_baseline
        fine_energy_mask = np.zeros(fine_matrix.shape[1], dtype=bool)
        fine_energy_mask[:fine_count] = (
            fine_excess[:fine_count] > fine_threshold[:fine_count]
        )
        fine_energy_event = bool(np.any(fine_energy_mask))
        cluster_mask = fine_excess[:fine_count] >= low_mid_cluster_excess_db
        cluster_bins = int(np.sum(cluster_mask))
        cluster_run = longest_true_run(cluster_mask)
        fine_cluster_event = bool(cluster_run >= low_mid_cluster_bins)
        fine_tonal = fine_tonal_by_window[i]
        fine_tonal_peak = float(np.nanmax(fine_tonal)) if np.any(np.isfinite(fine_tonal)) else float("nan")
        fine_tonal_idx = int(np.nanargmax(fine_tonal)) if np.any(np.isfinite(fine_tonal)) else -1
        fine_tonal_event = bool(
            np.isfinite(fine_tonal_peak)
            and fine_tonal_peak >= tonal_prominence_db
            and fine_tonal_idx >= 0
            and fine_excess[fine_tonal_idx] >= low_mid_cluster_excess_db
        )
        fine_iqr = fine_fs[i]["iqr"][:fine_count]
        fine_dynamic_fraction = float(np.nanmean(fine_iqr > 3))
        fine_dynamic_peak = float(np.nanmax(fine_iqr)) if np.any(np.isfinite(fine_iqr)) else float("nan")
        fine_dynamic_band_idx = int(np.nanargmax(fine_iqr)) if np.any(np.isfinite(fine_iqr)) else -1
        fine_global_event = bool(np.any(
            globally_tonal[:fine_count]
            & (fine_excess[:fine_count] >= low_mid_cluster_excess_db)
        ))
        fine_event = fine_energy_event or fine_cluster_event or fine_global_event or fine_tonal_event
        fine_peak_idx = int(np.nanargmax(np.where(fine_energy_mask, fine_excess, -np.inf))) if fine_energy_event else -1
        fine_peak_excess = float(fine_excess[fine_peak_idx]) if fine_peak_idx >= 0 else float("nan")
        stability = float(np.clip(1 - np.nanmedian(v["iqr"]) / 12, 0, 1))
        quiet = float(np.clip(1 - v["transient_ratio"] / 0.35, 0, 1))
        score = 0.35 * stability + 0.25 * (1 - flux_norm) + 0.20 * quiet + 0.20 * min(1, recurrent_fraction / 0.5)
        event = (
            v["transient_ratio"] > 0.35 or flux_norm > 0.90
            or low_mid_event or tonal_event or fine_event
        )
        bands = [f"{j * band_hz:g}-{min((j + 1) * band_hz, sr / 2):g} Hz" for j in np.flatnonzero(present)]
        reasons = (["stable_spectral_texture"] if stable_fraction >= .55 else [])
        reasons += (["low_transient_activity"] if quiet >= .65 else [])
        reasons += (["recurrent_frequency_bands"] if recurrent_fraction >= .20 else [])
        reasons += (["event_like_flux_or_burst"] if event else [])
        reasons += (["low_mid_band_event_veto"] if low_mid_event else [])
        reasons += (["low_mid_tonal_peak_veto"] if tonal_event else [])
        reasons += (["low_mid_fine_tonal_veto"] if fine_tonal_event else [])
        reasons += (["low_mid_energy_excess_veto"] if fine_energy_event else [])
        reasons += (["low_mid_cluster_energy_veto"] if fine_cluster_event else [])
        reasons += (["low_mid_global_tonal_band_veto"] if fine_global_event else [])
        windows.append({
            "index": i, "start_sec": round(s / sr, 3), "end_sec": round((s + nwin) / sr, 3),
            "score": round(float(score), 5), "candidate": bool(score >= threshold and not event),
            "rms_db": round(v["rms_db"], 3), "spectral_flux": v["spectral_flux"],
            "transient_ratio": round(v["transient_ratio"], 5), "energy_iqr_db": round(v["energy_iqr_db"], 3),
            "low_mid_flux": round(low_mid_flux, 7), "low_mid_flux_norm": round(low_mid_flux_norm, 5),
            "low_mid_unstable_fraction": round(low_mid_unstable, 5),
            "low_mid_event_veto": bool(low_mid_event),
            "low_mid_tonal_prominence_db": round(tonal_peak, 3) if np.isfinite(tonal_peak) else None,
            "low_mid_tonal_band": (
                f"{tonal_peak_idx * band_hz:g}-{min((tonal_peak_idx + 1) * band_hz, sr / 2):g} Hz"
                if tonal_peak_idx >= 0 else None
            ),
            "low_mid_tonal_peak_veto": tonal_event,
            "low_mid_energy_excess_db": round(fine_peak_excess, 3) if np.isfinite(fine_peak_excess) else None,
            "low_mid_energy_excess_band": (
                f"{fine_peak_idx * low_mid_band_hz:g}-{min((fine_peak_idx + 1) * low_mid_band_hz, sr / 2):g} Hz"
                if fine_peak_idx >= 0 else None
            ),
            "low_mid_energy_excess_veto": fine_energy_event,
            "low_mid_cluster_bins": cluster_bins,
            "low_mid_cluster_run": cluster_run,
            "low_mid_cluster_veto": fine_cluster_event,
            "low_mid_global_tonal_veto": fine_global_event,
            "low_mid_fine_tonal_veto": fine_tonal_event,
            "low_mid_dynamic_fraction": round(fine_dynamic_fraction, 5),
            "low_mid_dynamic_peak_db": round(fine_dynamic_peak, 3) if np.isfinite(fine_dynamic_peak) else None,
            "low_mid_dynamic_band": (
                f"{fine_dynamic_band_idx * low_mid_band_hz:g}-{min((fine_dynamic_band_idx + 1) * low_mid_band_hz, sr / 2):g} Hz"
                if fine_dynamic_band_idx >= 0 else None
            ),
            "low_mid_fine_tonal_prominence_db": round(fine_tonal_peak, 3) if np.isfinite(fine_tonal_peak) else None,
            "low_mid_fine_tonal_band": (
                f"{fine_tonal_idx * low_mid_band_hz:g}-{min((fine_tonal_idx + 1) * low_mid_band_hz, sr / 2):g} Hz"
                if fine_tonal_idx >= 0 else None
            ),
            "stable_band_fraction": round(stable_fraction, 5), "recurrent_band_fraction": round(recurrent_fraction, 5),
            "persistent_bands": bands, "reasons": reasons,
        })
    bands = [{"low_hz": j * band_hz, "high_hz": min((j + 1) * band_hz, sr / 2),
              "median_db": round(float(typical[j]), 3), "occupancy_fraction": round(float(np.mean(matrix[:, j] >= typical[j] - 3)), 5),
              "persistent": bool(recurrent[j])} for j in range(matrix.shape[1])]
    return {"config": {"window_sec": window_sec, "hop_sec": hop_sec, "band_hz": band_hz,
                        "min_persistence": persistence, "candidate_threshold": threshold,
                        "source_method": "LabIR", "bird_veto_hz": bird_veto_hz,
                        "low_mid_iqr_db": low_mid_iqr_db,
                        "low_mid_unstable_fraction": low_mid_unstable_fraction,
                        "low_mid_flux_percentile": low_mid_flux_percentile,
                        "tonal_prominence_db": tonal_prominence_db,
                        "low_mid_band_hz": low_mid_band_hz,
                        "low_mid_max_hz": low_mid_max_hz,
                        "low_mid_excess_db": low_mid_excess_db,
                        "low_mid_mad_multiplier": low_mid_mad_multiplier,
                        "low_mid_cluster_excess_db": low_mid_cluster_excess_db,
                        "low_mid_cluster_bins": low_mid_cluster_bins},
            "sample_rate": sr, "duration_sec": round(len(x) / sr, 3), "n_windows": len(windows),
            "nyquist_hz": sr / 2, "scope": "single_recording", "selected_channel": 0,
            "warning": "Candidate means stable/recurrent LabIR background-like texture; it does not prove insect identity or absence of birds. Low/mid-band flux, instability, and tonal-peak vetoes are conservative screening diagnostics. Manual review is required.",
            "bands": bands, "windows": windows,
            "ranked_candidate_indices": [v["index"] for v in sorted(windows, key=lambda z: z["score"], reverse=True) if v["candidate"]]}


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", s)


def plots(out, x, sr, result):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nperseg = min(max(256, round(sr * .064)), len(x)); noverlap = min(round(nperseg * .75), nperseg - 1)
    f, t, z = spectrogram(x, fs=sr, nperseg=nperseg, noverlap=noverlap, mode="magnitude")
    db = 20 * np.log10(z + EPS); w = result["windows"]
    fig, ax = plt.subplots(2, 1, figsize=(16, 10), sharex=True, gridspec_kw={"height_ratios": [1, 4]})
    ax[0].plot([v["start_sec"] for v in w], [v["score"] for v in w], lw=.8, color="#334155")
    ax[0].axhline(result["config"]["candidate_threshold"], ls="--", color="#dc2626")
    for v in w:
        if v["candidate"]: ax[0].axvspan(v["start_sec"], v["end_sec"], color="#16a34a", alpha=.15)
    ax[0].set(ylabel="candidate score", ylim=(0, 1)); ax[0].grid(alpha=.2)
    im = ax[1].pcolormesh(t, f, db, shading="auto", cmap="magma", vmin=np.nanpercentile(db, 5), vmax=np.nanpercentile(db, 99))
    for v in w: ax[1].axvspan(v["start_sec"], v["end_sec"], color="#22c55e" if v["candidate"] else "#ef4444", alpha=.14 if v["candidate"] else .025)
    ax[1].set(xlabel="time (s)", ylabel="frequency (Hz)"); fig.colorbar(im, ax=ax[1], label="magnitude (dB)"); fig.tight_layout(); fig.savefig(out / "diagnostic_overview.png", dpi=140); plt.close(fig)
    selected = sorted(w, key=lambda v: v["score"], reverse=True)[:12]
    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    for a, v in zip(axes.flat, selected):
        s, e = round(v["start_sec"] * sr), round(v["end_sec"] * sr); ff, tt, zz = spectrogram(x[s:e], fs=sr, nperseg=nperseg, noverlap=noverlap, mode="magnitude")
        local = 20 * np.log10(zz + EPS); a.pcolormesh(tt, ff, local, shading="auto", cmap="magma", vmin=np.nanpercentile(local, 5), vmax=np.nanpercentile(local, 99)); a.set_ylim(0, sr / 2); a.set_title(f"#{v['index']} {v['score']:.2f} {'CANDIDATE' if v['candidate'] else 'REVIEW'}", fontsize=9, color="#16a34a" if v["candidate"] else "#dc2626")
    for a in axes.flat[len(selected):]: a.axis("off")
    fig.suptitle("Highest-scoring 3-second windows — manual review required"); fig.tight_layout(); fig.savefig(out / "candidate_contact_sheet.png", dpi=140); plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--input", type=Path, required=True); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--export-candidates", action="store_true"); p.add_argument("--window-sec", type=float, default=3); p.add_argument("--hop-sec", type=float, default=1.5); p.add_argument("--band-hz", type=float, default=500); p.add_argument("--min-persistence", type=float, default=.75); p.add_argument("--candidate-threshold", type=float, default=.58); a = p.parse_args()
    x, sr, channels = load_ch0(a.input); out = a.output_dir; out.mkdir(parents=True, exist_ok=True); result = analyse(x, sr, a.window_sec, a.hop_sec, a.band_hz, a.min_persistence, a.candidate_threshold); result.update({"input": str(a.input), "input_channels": channels})
    if a.export_candidates:
        cd = out / "candidate_wav"; cd.mkdir(exist_ok=True); n = round(sr * a.window_sec)
        for v in sorted((z for z in result["windows"] if z["candidate"]), key=lambda z: z["score"], reverse=True)[:24]:
            s = round(v["start_sec"] * sr); sf.write(cd / f"{safe(a.input.stem)}_ch0_candidate_{v['start_sec']:09.3f}_{v['score']:.3f}.wav", x[s:s+n], sr, subtype="PCM_16")
    (out / f"{safe(a.input.stem)}_noise_detection.json").write_text(json.dumps(result, indent=2)); fields = list(result["windows"][0]);
    with (out / f"{safe(a.input.stem)}_window_scores.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for row in result["windows"]: w.writerow({k: ";".join(row[k]) if isinstance(row[k], list) else row[k] for k in fields})
    plots(out, x, sr, result); candidates = [v for v in result["windows"] if v["candidate"]]
    print(f"sample_rate={sr} channels={channels} duration_sec={result['duration_sec']} windows={result['n_windows']} candidates={len(candidates)}")
    print(f"output={out}"); [print(f"candidate index={v['index']} time={v['start_sec']:.3f}-{v['end_sec']:.3f}s score={v['score']:.3f} bands={','.join(v['persistent_bands'][:6])}") for v in sorted(candidates, key=lambda z: z["score"], reverse=True)[:10]]


if __name__ == "__main__": main()
