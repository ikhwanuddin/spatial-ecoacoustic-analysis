"""
Module 5: Detection Audit Manifest Generator for Ground-Truth Binary Calibration.
Extracts candidate detection windows across recordings into structured JSON and Markdown
tables with exact timestamps, beam channels, and audio links for rapid human listening audit
(Avian Presence/Absence Binary Gate).

Usage:
  python src/05_generate_audit_manifest.py --location 2A400 --date 2026-04-22 --min-conf 0.25
"""

import os
import sys
import glob
import json
import argparse
from typing import Dict, List, Any, Optional

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SCRATCH_DIR,
    OUTPUT_DIR,
    MONITORING_DATA,
    LOCATION_MAP,
)


def generate_manifest_for_recording(
    rec_scratch_dir: str,
    rec_name: str,
    location: str,
    date_str: str,
    min_conf: float = 0.25
) -> List[Dict[str, Any]]:
    results_json = os.path.join(rec_scratch_dir, "results.json")
    if not os.path.isfile(results_json):
        return []

    with open(results_json, "r", encoding="utf-8") as f:
        results = json.load(f)

    rpi_id = LOCATION_MAP.get(location, location)
    flac_path_cx3 = os.path.join(MONITORING_DATA, rpi_id, date_str, f"{rec_name}.flac")
    flac_path_mac = f"/Volumes/ri322/ephemeral/monitoring_data/{rpi_id}/{date_str}/{rec_name}.flac"

    windows = {}
    for ch, dets in results.items():
        for d in dets:
            st = round(float(d.get("start_time", 0.0)), 1)
            et = round(float(d.get("end_time", st + 3.0)), 1)
            conf = round(float(d.get("confidence", 0.0)), 4)
            lbl = d.get("common_name", "Unknown")

            if st not in windows:
                windows[st] = []
            windows[st].append({
                "channel": ch,
                "confidence": conf,
                "label": lbl,
                "start_time": st,
                "end_time": et
            })

    candidate_windows = []
    for st in sorted(windows.keys()):
        dets = windows[st]
        best_det = max(dets, key=lambda x: x["confidence"])

        if best_det["confidence"] >= min_conf:
            mono_matches = [
                d for d in dets
                if d["channel"] == "mono.wav" and d["label"] == best_det["label"]
            ]
            mono_conf = mono_matches[0]["confidence"] if mono_matches else None
            delta = round(best_det["confidence"] - (mono_conf or 0.0), 4)

            st_sec = int(st)
            et_sec = int(best_det["end_time"])
            time_str = f"{st_sec // 60:02d}:{st_sec % 60:02d} - {et_sec // 60:02d}:{et_sec % 60:02d}"

            best_ch = best_det["channel"]
            wav_path_cx3 = os.path.join(rec_scratch_dir, best_ch)
            wav_path_mac = f"/Volumes/ri322/ephemeral/sea-scratch/{location}/{date_str}/{rec_name}/{best_ch}"
            mono_path_mac = f"/Volumes/ri322/ephemeral/sea-scratch/{location}/{date_str}/{rec_name}/mono.wav"

            candidate_windows.append({
                "recording_id": rec_name,
                "location": location,
                "date": date_str,
                "window_seconds": [st, best_det["end_time"]],
                "time_str": time_str,
                "best_channel": best_ch,
                "best_conf": best_det["confidence"],
                "mono_conf": mono_conf,
                "conf_delta": delta,
                "tentative_label": best_det["label"],
                "audio_paths": {
                    "cx3_wav": wav_path_cx3,
                    "mac_wav": wav_path_mac,
                    "mac_mono_wav": mono_path_mac,
                    "cx3_flac": flac_path_cx3,
                    "mac_flac": flac_path_mac,
                },
                "ground_truth": {
                    "bird_present": None,
                    "audited_by": None,
                    "notes": ""
                }
            })

    return candidate_windows


def format_manifest_markdown(manifest_items: List[Dict[str, Any]], location: str, date_str: str) -> str:
    lines = [
        f"# Detection Audit Manifest: {location} ({date_str})",
        "",
        "**Tujuan:** Kalibrasi empiris gerbang biner keberadaan (*Avian Presence/Absence Ground Truth*).",
        "Dengarkan rekaman pada rentang detik yang tertera, lalu beri tanda pada kolom `Present?`:",
        "* Isi `1` jika terdengar suara kicau/panggilan burung asli.",
        "* Isi `0` jika derau murni (jangkrik, serangga, angin, hujan, ranting patah).",
        "",
        f"**Total Candidate Windows:** {len(manifest_items)} windows",
        "",
        "| # | Window | Rec ID | Best Channel | Mono | BF Conf | Gain (Delta) | Tentative Label | Present? [1/0] | Listen (Mac Clickable) |",
        "|---|---|---|---|---|---|---|---|:---:|---|"
    ]

    for idx, item in enumerate(manifest_items, 1):
        rec_short = item["recording_id"].replace("_dur=240secs", "")
        t_str = item["time_str"]
        ch_short = item["best_channel"].replace(".wav", "")
        m_conf = f"{item['mono_conf']:.2f}" if item['mono_conf'] is not None else "---"
        bf_conf = f"**{item['best_conf']:.2f}**"
        delta = f"+{item['conf_delta']:.2f}" if item['conf_delta'] > 0 else f"{item['conf_delta']:.2f}"
        lbl = item["tentative_label"]
        mac_wav = item["audio_paths"]["mac_wav"]
        mac_mono = item["audio_paths"]["mac_mono_wav"]

        listen_links = f"[Play BF](file://{mac_wav}) | [Mono](file://{mac_mono})"
        lines.append(
            f"| {idx} | `{t_str}` | `{rec_short}` | `{ch_short}` | {m_conf} | {bf_conf} | {delta} | {lbl} | ` ` | {listen_links} |"
        )

    return "\n".join(lines)


def evaluate_annotated_manifest(manifest_path: str):
    with open(manifest_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    audited = [d for d in data if d.get("ground_truth", {}).get("bird_present") is not None]
    if not audited:
        print(f"⚠️  No audited entries found in {manifest_path}. Annotate bird_present with 1 or 0 first.")
        return

    print("=" * 65)
    print(f"📊 GROUND-TRUTH CALIBRATION RESULTS: {len(audited)} Audited Windows")
    print("=" * 65)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    header = f"{chr(84)}hreshold (tau)   | Total Audited  | TP (Bird)  | FP (Noise) | Precision"
    print(header)
    print("-" * 65)

    for tau in thresholds:
        passing = [d for d in audited if d["best_conf"] >= tau]
        tp = sum(1 for d in passing if d["ground_truth"]["bird_present"] == 1)
        fp = sum(1 for d in passing if d["ground_truth"]["bird_present"] == 0)
        prec = (tp / len(passing) * 100) if passing else 0.0
        print(f"tau >= {tau:.2f}          | {len(passing):<14} | {tp:<10} | {fp:<10} | {prec:6.1f}%")

    print("=" * 65)


def generate_date_audit_manifest(
    location: str,
    date_str: str,
    min_conf: float = 0.25,
    out_dir: Optional[str] = None
) -> str:
    scratch_date_dir = os.path.join(SCRATCH_DIR, location, date_str)
    if not os.path.isdir(scratch_date_dir):
        print(f"❌ Scratch directory not found: {scratch_date_dir}")
        return ""

    if out_dir is None:
        out_dir = os.path.join(OUTPUT_DIR, location, date_str)
    os.makedirs(out_dir, exist_ok=True)

    rec_folders = sorted([
        f for f in os.listdir(scratch_date_dir)
        if os.path.isdir(os.path.join(scratch_date_dir, f))
    ])

    all_candidates = []
    for rec in rec_folders:
        rec_scratch = os.path.join(scratch_date_dir, rec)
        items = generate_manifest_for_recording(rec_scratch, rec, location, date_str, min_conf=min_conf)
        all_candidates.extend(items)

    json_path = os.path.join(out_dir, "detection_audit_manifest.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_candidates, f, indent=4, ensure_ascii=False)

    md_content = format_manifest_markdown(all_candidates, location, date_str)
    md_path = os.path.join(out_dir, "detection_audit_manifest.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    print(f"✅ Audit manifest generated: {len(all_candidates)} candidate windows (min_conf >= {min_conf})")
    print(f"   📁 JSON: {json_path}")
    print(f"   📄 MD:   {md_path}")
    return md_path


def main():
    parser = argparse.ArgumentParser(description="Generate Ground-Truth Binary Audit Manifest for BirdNET detections.")
    parser.add_argument("--location", default="2A400", help="Location code (default: 2A400)")
    parser.add_argument("--date", default="2026-04-22", help="Date string YYYY-MM-DD")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Minimum confidence threshold (default: 0.25)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: output/<loc>/<date>)")
    parser.add_argument("--evaluate", default=None, help="Path to annotated JSON manifest to compute Precision and tau*")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_annotated_manifest(args.evaluate)
        return

    generate_date_audit_manifest(args.location, args.date, min_conf=args.min_conf, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
