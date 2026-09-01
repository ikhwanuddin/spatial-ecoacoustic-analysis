#!/usr/bin/env python3
"""Temporal foreground-exclusion detector for LabIR noise-reference review.

This detector is scientifically conservative:
1. Computes stationary background baseline floors per 200-Hz band across 200–4000 Hz.
2. Identifies connected time-frequency biophonic components across 200–4000 Hz.
3. Excludes foreground biophonic vocalisations (avian songs, nocturnal calls/chirps, staccato syllables, tonal whistles).
4. Accepts 2.0-second candidate windows only when background dominates >= 95% of the
   window duration across 200–4000 Hz and zero foreground biophonic events are present.
5. Emits review diagnostics under source-derived location/date/condition folders.
6. Does NOT create official noise references without manual review.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import label
from scipy.signal import spectrogram

EPS = 1e-12


def safe(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def load_labir(path: Path) -> tuple[np.ndarray, int, int]:
    with sf.SoundFile(str(path)) as audio:
        sample_rate = int(audio.samplerate)
        channels = int(audio.channels)
        data = audio.read(dtype="float32", always_2d=True)
    if data.size == 0:
        raise ValueError("empty audio")
    return data[:, 0], sample_rate, channels


def extract_recording_events(
    full_band_db: np.ndarray,
    rec_floor: np.ndarray,
    frame_hop_sec: float,
    band_hz: int,
    min_hz: int,
    *,
    onset_db: float = 7.0,
    min_event_sec: float = 0.030,
    max_event_sec: float = 0.800,
) -> tuple[list[dict], np.ndarray]:
    """Detect all connected foreground biophonic events across the entire recording."""
    excess_rec = full_band_db - rec_floor
    active = excess_rec >= onset_db
    labels, n_labels = label(active, structure=np.ones((3, 3), dtype=int))

    events: list[dict] = []
    for cid in range(1, n_labels + 1):
        r, c = np.where(labels == cid)
        if len(r) == 0:
            continue
        dur = (int(c.max()) - int(c.min()) + 1) * frame_hop_sec
        if dur < min_event_sec or dur > max_event_sec:
            # Skip sub-transient blips (<30ms) or long stationary background texture (>800ms)
            continue

        peak_db = float(np.max(excess_rec[r, c]))
        bands = int(len(np.unique(r)))
        cells = int(len(r))
        low_f = int(min_hz + r.min() * band_hz)
        high_f = int(min_hz + (r.max() + 1) * band_hz)

        # Biophonic vocalisation / chirp / syllable classification across 200-4000 Hz:
        # 1. Multi-band syllable/chirp (>= 2 bands, >= 6 cells, >= 7.5 dB peak)
        # 2. Narrowband staccato syllable / chirp (1 band, >= 5 cells, >= 8.5 dB peak)
        # 3. Tonal whistle / sustained note (1 band, dur >= 0.080s, >= 8 cells, >= 7.5 dB peak)
        # 4. High-energy transient burst (>= 14.0 dB peak, >= 5 cells)
        is_bird = (
            (bands >= 2 and cells >= 6 and peak_db >= 7.5)
            or (bands == 1 and cells >= 5 and peak_db >= 8.5)
            or (bands == 1 and dur >= 0.080 and cells >= 8 and peak_db >= 7.5)
            or (peak_db >= 14.0 and cells >= 5)
        )

        events.append({
            "start_sec": round(float(c.min() * frame_hop_sec), 4),
            "end_sec": round(float((c.max() + 1) * frame_hop_sec), 4),
            "duration_sec": round(float(dur), 4),
            "low_hz": low_f,
            "high_hz": high_f,
            "band_count": bands,
            "frame_count": int(c.max() - c.min() + 1),
            "peak_excess_db": round(peak_db, 3),
            "tf_cells": cells,
            "is_bird": is_bird,
        })

    return events, excess_rec


def analyse(
    x: np.ndarray,
    sample_rate: int,
    *,
    window_sec: float = 2.0,
    hop_sec: float = 1.0,
    band_hz: int = 200,
    min_hz: int = 200,
    max_hz: int = 4000,
    frame_sec: float = 0.032,
    floor_percentile: float = 50,
    onset_db: float = 7.0,
    min_event_sec: float = 0.030,
    max_event_sec: float = 0.800,
    max_foreground_time: float = 0.05,
    max_allowed_excess_db: float = 12.0,
) -> dict:
    nperseg = min(max(128, round(sample_rate * frame_sec)), len(x))
    noverlap = min(round(nperseg * 0.75), nperseg - 1)
    f, t, power = spectrogram(
        x, fs=sample_rate, window="hann", nperseg=nperseg, noverlap=noverlap,
        mode="psd", scaling="density",
    )
    frame_hop_sec = float(t[1] - t[0]) if len(t) > 1 else frame_sec / 4
    edges = np.arange(min_hz, min(max_hz, sample_rate / 2) + band_hz, band_hz)
    if len(edges) < 2:
        raise ValueError("invalid low/mid frequency range")

    levels = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (f >= lo) & ((f < hi) if hi < sample_rate / 2 else (f <= hi))
        if not np.any(mask):
            levels.append(np.full(len(t), np.nan))
            continue
        levels.append(10 * np.log10(np.mean(power[mask], axis=0) + EPS))
    full_band_db = np.asarray(levels)

    # 1. Recording-level stationary baseline floor
    rec_floor = np.nanpercentile(full_band_db, floor_percentile, axis=1, keepdims=True)

    # 2. Extract all events across full recording in 200-4000 Hz
    events, excess_rec = extract_recording_events(
        full_band_db, rec_floor, frame_hop_sec, band_hz, min_hz,
        onset_db=onset_db, min_event_sec=min_event_sec, max_event_sec=max_event_sec,
    )
    bird_events = [e for e in events if e["is_bird"]]
    active_mask = excess_rec >= onset_db

    # 3. Evaluate candidate windows
    n_window = round(sample_rate * window_sec)
    n_hop = round(sample_rate * hop_sec)
    starts = list(range(0, len(x) - n_window + 1, n_hop))
    windows = []

    for index, start_sample in enumerate(starts):
        w_start_sec = start_sample / sample_rate
        w_end_sec = (start_sample + n_window) / sample_rate

        w_start_frame = round(w_start_sec / frame_hop_sec)
        w_end_frame = min(full_band_db.shape[1], round(w_end_sec / frame_hop_sec))

        w_active = active_mask[:, w_start_frame:w_end_frame]
        w_excess = excess_rec[:, w_start_frame:w_end_frame]

        fg_time_frac = float(np.mean(np.any(w_active, axis=0))) if w_active.shape[1] > 0 else 0.0
        fg_tf_frac = float(np.mean(w_active)) if w_active.size > 0 else 0.0
        w_max_excess = float(np.nanmax(w_excess)) if w_excess.size > 0 else 0.0

        # Find events overlapping with this window
        window_events = [
            e for e in events
            if not (e["end_sec"] < w_start_sec or e["start_sec"] > w_end_sec)
        ]
        window_bird_events = [e for e in window_events if e["is_bird"]]

        # Strict conservative candidate rule:
        # - Zero bird/biophonic foreground events in 200-4000 Hz
        # - Background dominates >= 95% of duration (fg_time_frac <= 0.05)
        # - No excessive transient burst (max_excess < 12.0 dB)
        candidate = (
            (len(window_bird_events) == 0)
            and (fg_time_frac <= max_foreground_time)
            and (w_max_excess < max_allowed_excess_db)
        )

        score = 1.0 - min(1.0, fg_time_frac + 0.02 * max(0.0, w_max_excess) + 0.5 * fg_tf_frac)

        windows.append({
            "index": index,
            "start_sec": round(w_start_sec, 3),
            "end_sec": round(w_end_sec, 3),
            "candidate": candidate,
            "background_score": round(score, 5),
            "foreground_time_fraction": round(fg_time_frac, 5),
            "foreground_tf_fraction": round(fg_tf_frac, 5),
            "max_excess_db": round(w_max_excess, 3),
            "n_events": len(window_events),
            "n_bird_events": len(window_bird_events),
            "bird_events": window_bird_events,
            "events": window_events,
        })

    return {
        "config": {
            "source_method": "LabIR",
            "window_sec": window_sec,
            "hop_sec": hop_sec,
            "low_mid_range_hz": [min_hz, max_hz],
            "low_mid_band_hz": band_hz,
            "frame_sec": frame_sec,
            "background_floor_percentile": floor_percentile,
            "foreground_onset_db": onset_db,
            "foreground_duration_sec": [min_event_sec, max_event_sec],
            "max_foreground_time_fraction": max_foreground_time,
            "max_allowed_excess_db": max_allowed_excess_db,
            "candidate_rule": f"zero biophonic events in 200-4000 Hz, fg_time <= {max_foreground_time}, max_excess < {max_allowed_excess_db}dB",
        },
        "sample_rate": sample_rate,
        "duration_sec": round(len(x) / sample_rate, 3),
        "n_windows": len(windows),
        "total_recording_events": len(events),
        "total_bird_events": len(bird_events),
        "windows": windows,
        "ranked_candidate_indices": [
            item["index"] for item in sorted(
                (item for item in windows if item["candidate"]),
                key=lambda item: item["background_score"], reverse=True,
            )
        ],
    }


def make_plots(out: Path, x: np.ndarray, sample_rate: int, result: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nperseg = min(max(128, round(sample_rate * 0.032)), len(x))
    noverlap = min(round(nperseg * 0.75), nperseg - 1)
    f, t, p = spectrogram(
        x, fs=sample_rate, window="hann", nperseg=nperseg,
        noverlap=noverlap, mode="psd", scaling="density",
    )
    db = 10 * np.log10(p + EPS)
    windows = result["windows"]

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 4]})
    axes[0].plot(
        [item["start_sec"] for item in windows],
        [item["foreground_time_fraction"] for item in windows],
        color="#b91c1c", lw=0.9, label="Foreground Time Fraction (200-4000 Hz)",
    )
    axes[0].axhline(result["config"]["max_foreground_time_fraction"], color="#f59e0b", ls="--", lw=0.8, label="Veto Threshold")
    axes[0].set(ylabel="foreground time fraction", ylim=(0, 1))
    axes[0].legend(loc="upper right")
    axes[0].grid(alpha=0.2)

    for item in windows:
        color = "#16a34a" if item["candidate"] else "#dc2626"
        alpha = 0.15 if item["candidate"] else 0.04
        axes[0].axvspan(item["start_sec"], item["end_sec"], color=color, alpha=alpha)

    image = axes[1].pcolormesh(
        t, f, db, shading="auto", cmap="magma",
        vmin=np.nanpercentile(db, 5), vmax=np.nanpercentile(db, 99),
    )
    axes[1].set(xlabel="time (s)", ylabel="frequency (Hz)", ylim=(0, 4000))
    for item in windows:
        if item["candidate"]:
            axes[1].axvspan(item["start_sec"], item["end_sec"], color="#16a34a", alpha=0.12)
    fig.colorbar(image, ax=axes[1], label="PSD (dB)")
    fig.suptitle(f"LabIR temporal foreground exclusion ({result['config']['window_sec']}s window) — {(result.get('source_scope', {}).get('condition') or 'review').capitalize()} Review")
    fig.tight_layout()
    fig.savefig(out / "temporal_diagnostic_overview.png", dpi=160)
    plt.close(fig)

    selected = sorted(
        (item for item in windows if item["candidate"]),
        key=lambda item: item["background_score"], reverse=True,
    )[:12]

    fig, axes = plt.subplots(3, 4, figsize=(16, 10), squeeze=False)
    for axis, item in zip(axes.flat, selected):
        start = round(item["start_sec"] * sample_rate)
        end = round(item["end_sec"] * sample_rate)
        ff, tt, pp = spectrogram(
            x[start:end], fs=sample_rate, window="hann",
            nperseg=nperseg, noverlap=noverlap,
            mode="psd", scaling="density",
        )
        local = 10 * np.log10(pp + EPS)
        axis.pcolormesh(
            tt, ff, local, shading="auto", cmap="magma",
            vmin=np.nanpercentile(local, 5), vmax=np.nanpercentile(local, 99),
        )
        axis.set(ylim=(0, 4000), title=f"{item['start_sec']:.1f}-{item['end_sec']:.1f} s (score: {item['background_score']:.3f})")
    for axis in axes.flat[len(selected):]:
        axis.axis("off")
    title_text = f"Accepted {result['config']['window_sec']}s windows — temporal foreground absent" if selected else "No candidate windows passed conservative veto"
    fig.suptitle(title_text)
    fig.tight_layout()
    fig.savefig(out / "temporal_candidate_contact_sheet.png", dpi=160)
    plt.close(fig)


def infer_source_scope(input_path: Path) -> tuple[Path, str, str, str]:
    """Infer review root, location, source date, and time condition from input."""
    parts = input_path.absolute().parts
    try:
        sea_index = parts.index("sea-work")
    except ValueError as exc:
        raise ValueError("input must be under an RDS sea-work tree") from exc
    if sea_index + 2 >= len(parts):
        raise ValueError("input path does not contain location/date under sea-work")
    location = parts[sea_index + 1]
    date = parts[sea_index + 2]
    if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", date):
        raise ValueError(f"source date is not YYYY-MM-DD: {date}")
    match = re.search(r"(?<!\d)(\d{2})-(\d{2})-(\d{2})(?!\d)", input_path.name)
    if not match:
        raise ValueError("input filename has no HH-MM-SS source time")
    hour = int(match.group(1))
    if 5 <= hour < 7:
        condition = "dawn"
    elif 7 <= hour < 17:
        condition = "day"
    elif 17 <= hour < 19:
        condition = "dusk"
    else:
        condition = "night"
    source_root = Path("/").joinpath(*parts[1:sea_index + 1])
    return source_root / "noise_auto_review", location, date, condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--output-root", type=Path,
                        help="optional review root; otherwise inferred from sea-work")
    parser.add_argument("--window-sec", type=float, default=2.0,
                        help="candidate window duration in seconds (default: 2.0)")
    parser.add_argument("--hop-sec", type=float, default=1.0,
                        help="candidate window hop in seconds (default: 1.0)")
    parser.add_argument("--export-candidates", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        review_root, location, source_date, condition = infer_source_scope(args.input)
        args.output_dir = (args.output_root or review_root) / location / source_date / condition / safe(args.input.stem)
    else:
        location = source_date = condition = None

    x, sample_rate, channels = load_labir(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = analyse(
        x, sample_rate, window_sec=args.window_sec, hop_sec=args.hop_sec,
    )
    result["input"] = str(args.input)
    result["input_channels"] = channels
    result["source_scope"] = {"location": location, "date": source_date, "condition": condition}

    stem = safe(args.input.stem)
    (args.output_dir / f"{stem}_temporal_noise_detection.json").write_text(
        json.dumps(result, indent=2)
    )

    if not result["windows"]:
        # A recording shorter than one analysis window produces nothing to score.
        print(f"no analysis window fits in {args.input.name} "
              f"(duration_sec={result['duration_sec']}, window_sec={args.window_sec})")
        raise SystemExit(0)

    fields = [key for key in result["windows"][0] if key not in ("events", "bird_events")]
    with (args.output_dir / f"{stem}_temporal_window_scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in result["windows"]:
            writer.writerow({key: row[key] for key in fields})

    candidate_dir = args.output_dir / "candidate_wav"
    if args.export_candidates:
        if candidate_dir.exists():
            shutil.rmtree(candidate_dir)
        candidate_dir.mkdir(exist_ok=True)
        n_window = round(sample_rate * result["config"]["window_sec"])
        accepted = sorted((item for item in result["windows"] if item["candidate"]),
                          key=lambda item: item["background_score"], reverse=True)[:24]
        for item in accepted:
            start = round(item["start_sec"] * sample_rate)
            destination = candidate_dir / (
                f"{stem}_temporal_candidate_{item['start_sec']:09.3f}_{item['background_score']:.3f}.wav"
            )
            sf.write(destination, x[start:start + n_window], sample_rate)

    make_plots(args.output_dir, x, sample_rate, result)
    candidates = [item for item in result["windows"] if item["candidate"]]
    print(f"sample_rate={sample_rate} channels={channels} duration_sec={result['duration_sec']} "
          f"window_sec={args.window_sec} windows={result['n_windows']} candidates={len(candidates)} biophony_events_total={result['total_bird_events']}")
    print(f"output={args.output_dir}")


if __name__ == "__main__":
    main()
