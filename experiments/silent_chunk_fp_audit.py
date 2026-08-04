#!/usr/bin/env python3
"""
Silent-window energy + (optional) BirdNET confidence audit.

Quantifies how often high-confidence species predictions appear on
low-energy ("silent") windows — the phenomenon discussed with Vincent.

Does **not** re-enable species-ID as the beamforming metric; this is an
audit of classifier false-positive rate on near-silence.

Default path: energy-only (no BirdNET install required).
Optional: --with-birdnet uses main-repo birdnetlib Analyzer if available.

Usage:
  # Energy-only on mono WAVs (works in any env with numpy/soundfile)
  python experiments/silent_chunk_fp_audit.py \\
    --location 2A400 --date 2026-03-19 --methods mono --max-wavs 4

  # With BirdNET confidences (main venv recommended)
  python experiments/silent_chunk_fp_audit.py \\
    --location 2A400 --date 2026-03-19 --methods mono \\
    --with-birdnet --conf-threshold 0.7 --max-wavs 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import ANALYSIS_OUTPUT, LOCATION_MAP, RPIID_TO_LOCATION  # noqa: E402
from embedding_schema import audits_dir  # noqa: E402

WINDOW_SEC = 3.0
SLIDE_SEC = 1.5
TARGET_SR = 16000


def _resolve_location(location: str) -> str:
    rpiid = LOCATION_MAP.get(location, location)
    return RPIID_TO_LOCATION.get(rpiid, rpiid)


def find_wavs(
    data_dir: str, location: str, date_str: str, methods: List[str], max_wavs: int
) -> List[Tuple[str, str, str]]:
    found: List[Tuple[str, str, str]] = []
    date_dir = os.path.join(data_dir, location, date_str)
    for method in methods:
        method_dir = os.path.join(date_dir, method)
        if not os.path.isdir(method_dir):
            continue
        for root, _d, files in os.walk(method_dir):
            for fname in sorted(files):
                if fname.lower().endswith(".wav") and not fname.startswith("._"):
                    found.append((os.path.join(root, fname), fname, method))
    if max_wavs and len(found) > max_wavs:
        found = found[:max_wavs]
    return found


def load_mono_16k(path: str) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sr != TARGET_SR:
        # linear resample
        duration = len(audio) / float(sr)
        n_new = max(1, int(duration * TARGET_SR))
        t_old = np.linspace(0, duration, len(audio), endpoint=False)
        t_new = np.linspace(0, duration, n_new, endpoint=False)
        audio = np.interp(t_new, t_old, audio).astype(np.float32)
    return audio


def window_rms(audio: np.ndarray, window_sec: float, slide_sec: float) -> List[Dict[str, Any]]:
    win = int(window_sec * TARGET_SR)
    step = int(slide_sec * TARGET_SR)
    rows = []
    for start in range(0, len(audio), step):
        seg = audio[start : start + win]
        if len(seg) < TARGET_SR:  # < 1 s
            break
        # pad short tail windows for consistent length
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
        rows.append(
            {
                "start_sec": round(start / TARGET_SR, 3),
                "end_sec": round((start + win) / TARGET_SR, 3),
                "rms": rms,
                "segment": seg,
            }
        )
    return rows


def classify_silent(
    windows: List[Dict[str, Any]],
    mode: str,
    ratio: float,
    abs_rms: float,
    percentile: float = 20.0,
) -> None:
    """Mutate windows with is_silent flag.

    modes:
      absolute   — rms < abs_rms
      median/max — rms < ratio * ref
      percentile — rms <= P{percentile} of this file (default bottom 20%)
      hybrid     — percentile OR absolute (whichever marks silent)
    """
    if not windows:
        return
    rms_vals = np.array([w["rms"] for w in windows], dtype=np.float64)
    thr: float
    if mode == "absolute":
        thr = abs_rms
        mask = rms_vals < thr
    elif mode == "percentile":
        thr = float(np.percentile(rms_vals, percentile))
        mask = rms_vals <= thr
    elif mode == "hybrid":
        thr_p = float(np.percentile(rms_vals, percentile))
        thr = thr_p
        mask = (rms_vals <= thr_p) | (rms_vals < abs_rms)
    elif mode == "median":
        ref = float(np.median(rms_vals))
        thr = ratio * ref if ref > 0 else abs_rms
        mask = rms_vals < thr
    else:  # max
        ref = float(np.max(rms_vals))
        thr = ratio * ref if ref > 0 else abs_rms
        mask = rms_vals < thr
    for w, sil in zip(windows, mask):
        w["is_silent"] = bool(sil)
        w["rms_threshold"] = thr


def _to_48k(segment_16k: np.ndarray) -> np.ndarray:
    duration = len(segment_16k) / TARGET_SR
    n48 = int(round(duration * 48000))
    n48 = max(n48, 1)
    t_old = np.linspace(0, duration, len(segment_16k), endpoint=False)
    t_new = np.linspace(0, duration, n48, endpoint=False)
    return np.interp(t_new, t_old, segment_16k).astype(np.float32)


def birdnet_max_conf(analyzer, segment_16k: np.ndarray) -> Tuple[float, str]:
    """Return (max_confidence, top_label) for a 16 kHz segment via birdnetlib.

    Uses Analyzer.predict on a 3 s @ 48 kHz window (fast, no tempfile).
    """
    audio_48 = _to_48k(segment_16k)
    n_need = 144000  # 3 s @ 48 kHz
    if len(audio_48) < n_need:
        audio_48 = np.pad(audio_48, (0, n_need - len(audio_48)))
    else:
        audio_48 = audio_48[:n_need]

    pred = analyzer.predict(audio_48)
    scores = np.asarray(pred, dtype=np.float32).reshape(-1)
    if scores.size == 0:
        return 0.0, ""
    idx = int(np.argmax(scores))
    conf = float(scores[idx])
    labels = getattr(analyzer, "labels", None) or []
    label = str(labels[idx]) if idx < len(labels) else ""
    return conf, label


def select_windows_for_birdnet(
    windows: List[Dict[str, Any]],
    max_silent: int,
    max_active: int,
    rng: np.random.Generator,
) -> List[int]:
    """Indices to run BirdNET on: all/limited silent + sample of non-silent."""
    sil_idx = [i for i, w in enumerate(windows) if w["is_silent"]]
    act_idx = [i for i, w in enumerate(windows) if not w["is_silent"]]
    if max_silent and len(sil_idx) > max_silent:
        sil_idx = list(rng.choice(sil_idx, size=max_silent, replace=False))
    if max_active and len(act_idx) > max_active:
        act_idx = list(rng.choice(act_idx, size=max_active, replace=False))
    return sorted(set(sil_idx + act_idx))


def run_audit(args: argparse.Namespace) -> Dict[str, Any]:
    location = _resolve_location(args.location)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    wavs = find_wavs(args.data_dir, location, args.date, methods, args.max_wavs)

    analyzer = None
    species_list_used = None
    if args.with_birdnet:
        try:
            os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
            from birdnetlib.analyzer import Analyzer

            # Resolve species list: explicit flag → Way Canguk default for known sites
            sl = args.species_list
            if not sl and not args.no_species_list:
                try:
                    from config import resolve_birdnet_filter

                    sl, _lat, _lon, mode = resolve_birdnet_filter(location)
                    print(f"  BirdNET filter: {mode}")
                except Exception:
                    sl = None
            if sl and os.path.isfile(sl):
                analyzer = Analyzer(custom_species_list_path=sl)
                species_list_used = sl
                print(f"  Species list: {sl}")
            else:
                analyzer = Analyzer()
                print("  Species list: (full global labels)")
        except Exception as e:
            print(f"BirdNET unavailable ({e}); continuing energy-only.", file=sys.stderr)
            analyzer = None

    per_file: List[Dict[str, Any]] = []
    all_windows: List[Dict[str, Any]] = []

    rng = np.random.default_rng(args.seed)

    for wav_path, wav_name, method in wavs:
        print(f"  {method}/{wav_name}")
        audio = load_mono_16k(wav_path)
        windows = window_rms(audio, args.window_sec, args.slide_sec)
        classify_silent(
            windows,
            args.silent_mode,
            args.silent_ratio,
            args.silent_abs_rms,
            percentile=args.silent_percentile,
        )

        birdnet_idx: set = set()
        if analyzer is not None:
            birdnet_idx = set(
                select_windows_for_birdnet(
                    windows,
                    max_silent=args.max_silent_birdnet,
                    max_active=args.max_active_birdnet,
                    rng=rng,
                )
            )
            print(
                f"    windows={len(windows)} silent="
                f"{sum(1 for w in windows if w['is_silent'])} "
                f"birdnet_on={len(birdnet_idx)}"
            )

        file_rows = []
        for i, w in enumerate(windows):
            row = {
                "wav": wav_name,
                "method": method,
                "start_sec": w["start_sec"],
                "end_sec": w["end_sec"],
                "rms": w["rms"],
                "is_silent": w["is_silent"],
                "rms_threshold": w["rms_threshold"],
            }
            if analyzer is not None and i in birdnet_idx:
                try:
                    conf, label = birdnet_max_conf(analyzer, w["segment"])
                except Exception as e:
                    conf, label = 0.0, f"error:{e}"
                row["max_conf"] = conf
                row["top_label"] = label
                row["high_conf"] = conf >= args.conf_threshold
                row["birdnet_scored"] = True
            elif analyzer is not None:
                row["birdnet_scored"] = False
            # drop heavy segment
            del w["segment"]
            file_rows.append(row)
            all_windows.append(row)

        n_sil = sum(1 for r in file_rows if r["is_silent"])
        n_bn = sum(1 for r in file_rows if r.get("birdnet_scored"))
        per_file.append(
            {
                "wav": wav_name,
                "method": method,
                "n_windows": len(file_rows),
                "n_silent": n_sil,
                "n_birdnet_scored": n_bn,
            }
        )

    silent = [r for r in all_windows if r["is_silent"]]
    non_silent = [r for r in all_windows if not r["is_silent"]]

    summary: Dict[str, Any] = {
        "location": location,
        "date": args.date,
        "methods": methods,
        "n_wavs": len(wavs),
        "n_windows": len(all_windows),
        "n_silent": len(silent),
        "n_non_silent": len(non_silent),
        "silent_fraction": (len(silent) / len(all_windows)) if all_windows else 0.0,
        "silent_mode": args.silent_mode,
        "silent_ratio": args.silent_ratio,
        "silent_abs_rms": args.silent_abs_rms,
        "window_sec": args.window_sec,
        "slide_sec": args.slide_sec,
        "with_birdnet": analyzer is not None,
        "conf_threshold": args.conf_threshold,
        "species_list": species_list_used,
        "per_file": per_file,
    }

    if analyzer is not None and all_windows:
        def _rate(rows: List[dict]) -> Dict[str, Any]:
            scored = [r for r in rows if r.get("birdnet_scored") and "max_conf" in r]
            if not scored:
                return {
                    "n_scored": 0,
                    "n_high_conf": 0,
                    "pct_high_conf": 0.0,
                    "mean_max_conf": 0.0,
                    "median_max_conf": 0.0,
                }
            confs = np.array([float(r.get("max_conf", 0.0)) for r in scored])
            thr = args.conf_threshold
            multi = {
                f"pct_ge_{t}": float(100.0 * np.mean(confs >= t))
                for t in (0.1, 0.2, 0.3, 0.4, 0.5, 0.7)
            }
            return {
                "n_scored": len(scored),
                "n_high_conf": int(np.sum(confs >= thr)),
                "pct_high_conf": float(100.0 * np.mean(confs >= thr)),
                "mean_max_conf": float(np.mean(confs)),
                "median_max_conf": float(np.median(confs)),
                "p90_max_conf": float(np.percentile(confs, 90)),
                "max_max_conf": float(np.max(confs)),
                "thresholds": multi,
            }

        summary["silent_conf"] = _rate(silent)
        summary["non_silent_conf"] = _rate(non_silent)
        # classic FP proxy: high conf on silent (among scored silent)
        summary["fp_proxy_pct_silent_high_conf"] = summary["silent_conf"]["pct_high_conf"]

    # write outputs
    out_dir = args.output_dir or audits_dir(args.data_dir, location)
    os.makedirs(out_dir, exist_ok=True)
    tag = "birdnet" if analyzer is not None else "energy"
    base = f"{args.date}_silent_fp_audit_{tag}"
    json_path = os.path.join(out_dir, f"{base}.json")
    md_path = os.path.join(out_dir, f"{base}.md")
    csv_path = os.path.join(out_dir, f"{base}_windows.csv")

    payload = {"summary": summary, "windows": all_windows}
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)

    # CSV
    if all_windows:
        keys = list(all_windows[0].keys())
        with open(csv_path, "w") as f:
            f.write(",".join(keys) + "\n")
            for r in all_windows:
                f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

    lines = [
        f"# Silent-window audit — {location} {args.date}",
        "",
        f"- WAVs: {summary['n_wavs']}",
        f"- Windows: {summary['n_windows']} (silent={summary['n_silent']}, "
        f"{100*summary['silent_fraction']:.1f}%)",
        f"- Silent rule: mode={args.silent_mode}, ratio={args.silent_ratio}, "
        f"abs_rms={args.silent_abs_rms}",
        f"- BirdNET: {summary['with_birdnet']}",
    ]
    if summary.get("silent_conf"):
        sc = summary["silent_conf"]
        nsc = summary["non_silent_conf"]
        lines += [
            "",
            "## Classifier confidence (BirdNET-scored subset)",
            f"- Silent windows with max_conf ≥ {args.conf_threshold}: "
            f"{sc['n_high_conf']}/{sc['n_scored']} ({sc['pct_high_conf']:.1f}%)",
            f"- Non-silent high-conf: {nsc['n_high_conf']}/{nsc['n_scored']} "
            f"({nsc['pct_high_conf']:.1f}%)",
            f"- Mean max_conf silent / non-silent: "
            f"{sc['mean_max_conf']:.3f} / {nsc['mean_max_conf']:.3f}",
            f"- P90 max_conf silent / non-silent: "
            f"{sc.get('p90_max_conf', 0):.3f} / {nsc.get('p90_max_conf', 0):.3f}",
            "",
            f"**FP proxy (silent high-conf %): {summary['fp_proxy_pct_silent_high_conf']:.1f}%**",
        ]
    lines += ["", f"JSON: `{json_path}`", f"CSV: `{csv_path}`"]
    with open(md_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n".join(lines))
    summary["paths"] = {"json": json_path, "md": md_path, "csv": csv_path}
    return summary


def main() -> None:
    p = argparse.ArgumentParser(description="Silent-window FP audit")
    p.add_argument("--location", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--data-dir", default=ANALYSIS_OUTPUT)
    p.add_argument("--methods", default="mono,sa")
    p.add_argument("--max-wavs", type=int, default=0)
    p.add_argument("--window-sec", type=float, default=WINDOW_SEC)
    p.add_argument("--slide-sec", type=float, default=SLIDE_SEC)
    p.add_argument(
        "--silent-mode",
        choices=["median", "max", "absolute", "percentile", "hybrid"],
        default="percentile",
        help="How to set silent RMS threshold (default: bottom percentile per file)",
    )
    p.add_argument(
        "--silent-ratio",
        type=float,
        default=0.25,
        help="Silent if rms < ratio * ref (median or max)",
    )
    p.add_argument(
        "--silent-percentile",
        type=float,
        default=20.0,
        help="Bottom P%% of RMS per file marked silent (percentile/hybrid)",
    )
    p.add_argument(
        "--silent-abs-rms",
        type=float,
        default=1e-4,
        help="Absolute RMS floor / threshold when mode=absolute or hybrid",
    )
    p.add_argument("--with-birdnet", action="store_true")
    p.add_argument(
        "--species-list",
        default=None,
        help="BirdNET custom species list path (default: resolve_birdnet_filter for location)",
    )
    p.add_argument(
        "--no-species-list",
        action="store_true",
        help="Force full global BirdNET labels (ignore site list)",
    )
    p.add_argument("--conf-threshold", type=float, default=0.7)
    p.add_argument(
        "--max-silent-birdnet",
        type=int,
        default=40,
        help="Max silent windows per file to score with BirdNET (0=all)",
    )
    p.add_argument(
        "--max-active-birdnet",
        type=int,
        default=40,
        help="Max non-silent windows per file to score with BirdNET (0=all)",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()
    run_audit(args)


if __name__ == "__main__":
    main()
