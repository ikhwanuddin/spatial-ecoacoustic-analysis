"""
Module 5: Detection Audit Manifest Generator for Ground-Truth Binary Calibration.
Extracts candidate detection windows across recordings into structured JSON and Markdown
tables with exact timestamps, beam channels, and audio links for rapid human listening audit
(Avian Presence/Absence Binary Gate).
"""

import os
import re
import sys
import glob
import json
import argparse
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SCRATCH_DIR, OUTPUT_DIR, MONITORING_DATA, LOCATION_MAP


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
        m_conf = f"{item['mono_conf']:.2f}" if item["mono_conf"] is not None else "---"
        bf_conf = f"**{item['best_conf']:.2f}**"
        delta = f"+{item['conf_delta']:.2f}" if item["conf_delta"] > 0 else f"{item['conf_delta']:.2f}"
        lbl = item["tentative_label"]
        mac_wav = item["audio_paths"]["mac_wav"]
        mac_mono = item["audio_paths"]["mac_mono_wav"]

        present_val = item.get("ground_truth", {}).get("bird_present")
        p_str = str(present_val) if present_val is not None else " "

        listen_links = f"[Play BF](file://{mac_wav}) | [Mono](file://{mac_mono})"
        lines.append(
            f"| {idx} | `{t_str}` | `{rec_short}` | `{ch_short}` | {m_conf} | {bf_conf} | {delta} | {lbl} | `{p_str}` | {listen_links} |"
        )

    return "\n".join(lines)


def parse_markdown_manifest(md_path: str) -> List[Dict[str, Any]]:
    with open(md_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    items = []
    for line in lines:
        line_s = line.strip()
        if not line_s.startswith("|") or line_s.startswith("| #") or line_s.startswith("|---"):
            continue
        parts = [p.strip() for p in line_s.split("|")[1:-1]]
        if len(parts) < 9:
            continue

        idx_str, time_str, rec_id, ch_str, mono_str, bf_str, delta_str, label, present_str = parts[:9]
        
        present_clean = present_str.replace("`", "").strip()
        bird_present = None
        if present_clean in ["1", "true", "True", "y", "Y", "v", "x"]:
            bird_present = 1
        elif present_clean in ["0", "false", "False", "n", "N"]:
            bird_present = 0

        try:
            bf_clean = re.sub(r"[\*\_]", "", bf_str)
            best_conf = float(bf_clean)
        except Exception:
            best_conf = 0.0

        try:
            mono_clean = re.sub(r"[\*\_]", "", mono_str)
            mono_conf = float(mono_clean) if mono_clean != "---" else None
        except Exception:
            mono_conf = None

        items.append({
            "idx": int(idx_str) if idx_str.isdigit() else len(items) + 1,
            "time_str": time_str.replace("`", "").strip(),
            "recording_id": rec_id.replace("`", "").strip(),
            "best_channel": ch_str.replace("`", "").strip(),
            "mono_conf": mono_conf,
            "best_conf": best_conf,
            "tentative_label": label,
            "ground_truth": {
                "bird_present": bird_present
            }
        })
    return items


def evaluate_annotated_manifest(manifest_path: str):
    if manifest_path.endswith(".md"):
        items = parse_markdown_manifest(manifest_path)
        json_path = manifest_path.replace(".md", ".json")
        if os.path.isfile(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    orig_json = json.load(f)
                for idx, parsed in enumerate(items):
                    if idx < len(orig_json):
                        orig_json[idx]["ground_truth"]["bird_present"] = parsed["ground_truth"]["bird_present"]
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(orig_json, f, indent=4, ensure_ascii=False)
            except Exception:
                pass
    else:
        with open(manifest_path, "r", encoding="utf-8") as f:
            items = json.load(f)

    audited = [d for d in items if d.get("ground_truth", {}).get("bird_present") is not None]
    if not audited:
        print(f"⚠️  No audited entries found in {manifest_path}.")
        print("   Silakan buka detection_audit_manifest.md dan isi kolom 'Present?' dengan angka 1 atau 0.")
        return

    n_total = len(items)
    n_aud = len(audited)
    n_tp_total = sum(1 for d in audited if d["ground_truth"]["bird_present"] == 1)
    n_fp_total = sum(1 for d in audited if d["ground_truth"]["bird_present"] == 0)

    print("=" * 70)
    print("📊 GROUND-TRUTH CALIBRATION AUDIT REPORT")
    print(f"   Total Windows: {n_total} | Audited: {n_aud} ({n_aud/n_total*100:.1f}%)")
    print(f"   True Birds (1): {n_tp_total} | Noise/Insects (0): {n_fp_total}")
    print("=" * 70)

    thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.70]
    header = f"{'Threshold (tau)':<16} | {'Passing':<9} | {'TP (Bird)':<9} | {'FP (Noise)':<10} | {'Precision':<10}"
    print(header)
    print("-" * 70)

    rec_tau = None
    rep_lines = [
        "# Ground-Truth Binary Calibration Report",
        "",
        f"**Audited Samples:** {n_aud} / {n_total} windows ({n_aud/n_total*100:.1f}%)  ",
        f"**True Avian Vocalizations (TP):** {n_tp_total}  ",
        f"**Noise / Insect Hallucinations (FP):** {n_fp_total}  ",
        "",
        "## Precision vs Threshold Sweep",
        "",
        "| Threshold ($\\tau$) | Passing Windows | True Positives (Birds) | False Positives (Noise) | Empirical Precision |",
        "|---|---|---|---|---|"
    ]

    for tau in thresholds:
        passing = [d for d in audited if d["best_conf"] >= tau]
        tp = sum(1 for d in passing if d["ground_truth"]["bird_present"] == 1)
        fp = sum(1 for d in passing if d["ground_truth"]["bird_present"] == 0)
        prec = (tp / len(passing) * 100) if passing else 0.0

        if rec_tau is None and prec >= 85.0 and len(passing) > 0:
            rec_tau = tau

        row_str = f"tau >= {tau:.2f}          | {len(passing):<9} | {tp:<9} | {fp:<10} | {prec:6.1f}%"
        print(row_str)
        rep_lines.append(f"| **{tau:.2f}** | {len(passing)} | {tp} | {fp} | **{prec:.1f}%** |")

    print("=" * 70)
    if rec_tau:
        print(f"🎯 RECOMMENDED CALIBRATED THRESHOLD: tau* = {rec_tau:.2f} (Precision >= 85%)")
        rep_lines.append(f"\n### 🎯 Recommended Calibrated Threshold\n**$\\tau^* = {rec_tau:.2f}$** yields $\\ge 85\\%$ empirical precision in this soundscape.")
    else:
        print("💡 TIP: Annotate more high-confidence windows or check dawn chorus recordings to find the knee.")

    out_dir = os.path.dirname(manifest_path)
    rep_path = os.path.join(out_dir, "calibration_report.md")
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write("\n".join(rep_lines) + "\n")
    print(f"✅ Calibration report saved to: {rep_path}")


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
    parser = argparse.ArgumentParser(description="Generate & Evaluate Ground-Truth Binary Audit Manifest for BirdNET detections.")
    parser.add_argument("--location", default="2A400", help="Location code (default: 2A400)")
    parser.add_argument("--date", default="2026-04-22", help="Date string YYYY-MM-DD")
    parser.add_argument("--min-conf", type=float, default=0.25, help="Minimum confidence threshold (default: 0.25)")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: output/<loc>/<date>)")
    parser.add_argument("--evaluate", default=None, help="Path to annotated Markdown or JSON manifest to compute Precision and tau*")
    args = parser.parse_args()

    if args.evaluate:
        evaluate_annotated_manifest(args.evaluate)
        return

    generate_date_audit_manifest(args.location, args.date, min_conf=args.min_conf, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
