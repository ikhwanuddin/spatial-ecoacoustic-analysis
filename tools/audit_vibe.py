#!/usr/bin/env python3
"""
audit_vibe.py - Fast-Feedback Audio Auditing CLI for Spatial Ecoacoustics (SEA).

Follows Karpathy's Vibe Coding sensory feedback principle:
Listen to exact 5-second candidate windows over macOS audio (afplay)
and label avian vocalization ground-truth [1/0] with single keystrokes.
Works seamlessly on macOS (native speaker playback via SMB) and on CX3 Linux.
"""

import os
import sys
import json
import csv
import glob
import random
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

DEFAULT_SMB_BASE = "/Volumes/ri322/home/spatial-ecoacoustic-analysis/output"
DEFAULT_CX3_BASE = "/rds/general/user/ri322/home/spatial-ecoacoustic-analysis/output"
DEFAULT_LOCAL_BASE = "./output"


def get_base_dir(custom_path: Optional[str] = None) -> str:
    if custom_path and os.path.isdir(custom_path):
        return custom_path
    if os.path.isdir(DEFAULT_SMB_BASE):
        return DEFAULT_SMB_BASE
    if os.path.isdir(DEFAULT_CX3_BASE):
        return DEFAULT_CX3_BASE
    if os.path.isdir(DEFAULT_LOCAL_BASE):
        return DEFAULT_LOCAL_BASE
    return "."


def play_audio(audio_path: Optional[str]) -> bool:
    """Plays audio via macOS afplay or available Linux players."""
    if not audio_path or not os.path.exists(audio_path):
        print(f"⚠️  File audio tidak ditemukan: {audio_path}")
        return False

    if sys.platform == "darwin":
        try:
            subprocess.run(["afplay", audio_path], check=False)
            return True
        except Exception as e:
            print(f"⚠️  Gagal memutar audio via afplay: {e}")
            return False
    else:
        # Linux (e.g. CX3 or remote server)
        for player in ["paplay", "aplay", "ffplay", "mpv"]:
            if shutil.which(player):
                try:
                    if player == "ffplay":
                        subprocess.run([player, "-nodisp", "-autoexit", audio_path], stderr=subprocess.DEVNULL, check=False)
                    else:
                        subprocess.run([player, audio_path], stderr=subprocess.DEVNULL, check=False)
                    return True
                except Exception:
                    pass
        print(f"🔈 [Audio File]: {audio_path}")
        print("💡 TIPS: Jalankan audit_vibe.py di Terminal Mac mini agar audio otomatis berputar di speaker/headphone!")
        return False


def find_available_dates(base_dir: str) -> List[Dict[str, str]]:
    """Finds all dates that have detection_audit_manifest.json."""
    results = []
    locations = ["2A400", "2B400", "2D400", "S0", "Q0", "O0"]
    for loc in locations:
        loc_dir = os.path.join(base_dir, loc)
        if not os.path.isdir(loc_dir):
            continue
        for d in sorted(os.listdir(loc_dir)):
            manifest_path = os.path.join(loc_dir, d, "detection_audit_manifest.json")
            if os.path.isfile(manifest_path):
                results.append({"location": loc, "date": d, "manifest": manifest_path})
    return results


def resolve_audio_paths(item: Dict[str, Any], date_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Resolves local/remote audio paths for both beam and mono clips."""
    audio_paths = item.get("audio_paths", {})
    beam_clip = None
    mono_clip = None

    if sys.platform == "darwin":
        beam_clip = audio_paths.get("mac_wav")
        mono_clip = audio_paths.get("mac_mono_wav")
    else:
        beam_clip = audio_paths.get("cx3_wav")
        mono_clip = audio_paths.get("cx3_mono_wav")

    # Fallback to local date directory clips
    if not beam_clip or not os.path.exists(beam_clip):
        clips_dir = os.path.join(date_dir, "audit_clips")
        rec_id = item.get("recording_id", "")
        start_s = 0
        if "window_seconds" in item and isinstance(item["window_seconds"], list):
            start_s = int(item["window_seconds"][0])
        best_ch = item.get("best_channel", "").replace("(", "_").replace(")", "").replace(".wav", "")
        cand_beam = os.path.join(clips_dir, f"{rec_id}_{start_s:04d}s_{best_ch}.wav")
        if os.path.exists(cand_beam):
            beam_clip = cand_beam

        cand_mono = os.path.join(clips_dir, f"{rec_id}_{start_s:04d}s_mono.wav")
        if os.path.exists(cand_mono):
            mono_clip = cand_mono

    return beam_clip, mono_clip


def load_existing_annotations(gt_path: str) -> Dict[str, Dict[str, Any]]:
    """Loads existing ground-truth labels if present."""
    if not os.path.exists(gt_path):
        return {}
    try:
        with open(gt_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return {item["candidate_key"]: item for item in data}
    except Exception:
        return {}


def save_annotations(gt_json_path: str, gt_csv_path: str, annotations: List[Dict[str, Any]]):
    """Saves ground-truth annotations to both JSON and CSV."""
    with open(gt_json_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    if annotations:
        keys = list(annotations[0].keys())
        with open(gt_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(annotations)


def update_markdown_manifest(md_path: str, annotations_dict: Dict[str, int]):
    """Patches Present? [1/0] column in markdown manifest."""
    if not os.path.exists(md_path):
        return
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        updated_lines = []
        for line in lines:
            if line.startswith("|") and not line.startswith("| #") and not line.startswith("|---"):
                parts = [p.strip() for p in line.split("|")]
                if len(parts) >= 11:
                    window = parts[2]
                    rec_id = parts[3]
                    key = f"{rec_id}_{window}"
                    if key in annotations_dict:
                        label = str(annotations_dict[key])
                        parts[9] = f" `{label}` "
                        line = "| " + " | ".join(parts[1:-1]) + " |\n"
            updated_lines.append(line)

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
    except Exception as e:
        print(f"⚠️  Gagal memperbarui Markdown manifest: {e}")


def run_audit(manifest_path: str, sample_size: int = 20, sort_by_gain: bool = True, min_conf: float = 0.30):
    date_dir = os.path.dirname(manifest_path)
    gt_json_path = os.path.join(date_dir, "audit_ground_truth.json")
    gt_csv_path = os.path.join(date_dir, "audit_ground_truth.csv")
    md_path = os.path.join(date_dir, "detection_audit_manifest.md")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest_data = json.load(f)

    # Handle both list format and dict format
    if isinstance(manifest_data, list):
        candidates = manifest_data
        loc_name = candidates[0].get("location", "Unknown") if candidates else "Unknown"
        date_name = candidates[0].get("date", "Unknown") if candidates else "Unknown"
    elif isinstance(manifest_data, dict):
        candidates = manifest_data.get("candidates", [])
        loc_name = manifest_data.get("location", "Unknown")
        date_name = manifest_data.get("date", "Unknown")
    else:
        candidates = []
        loc_name = "Unknown"
        date_name = "Unknown"

    if not candidates:
        print("❌ Tidak ada kandidat deteksi dalam manifest ini.")
        return

    # Filter by minimum confidence
    filtered = []
    for c in candidates:
        b_conf = c.get("best_conf") if c.get("best_conf") is not None else c.get("best_beam_conf", 0.0)
        if b_conf is not None and b_conf >= min_conf:
            filtered.append(c)

    # Sort by confidence delta (gain) or shuffle
    if sort_by_gain:
        def get_delta(x):
            d = x.get("conf_delta")
            if d is not None:
                return float(d)
            g = x.get("gain")
            if g is not None:
                return float(g)
            b = x.get("best_conf", 0.0) or x.get("best_beam_conf", 0.0) or 0.0
            m = x.get("mono_conf", 0.0) or 0.0
            return float(b) - float(m)
        filtered.sort(key=get_delta, reverse=True)
    else:
        random.shuffle(filtered)

    existing_annotations = load_existing_annotations(gt_json_path)

    unannotated = []
    for c in filtered:
        rec_id = c.get("recording_id", "")
        win_str = c.get("time_str") or str(c.get("window", ""))
        key = f"{rec_id}_{win_str}"
        if key not in existing_annotations:
            unannotated.append(c)

    targets = unannotated[:sample_size] if unannotated else filtered[:sample_size]
    total_to_audit = len(targets)

    if total_to_audit == 0:
        print("✅ Seluruh kandidat pada tanggal ini sudah diaudit sebelumnya!")
        return

    print("=" * 70)
    print(f"🎧 SEA VIBE AUDIT: {loc_name} | {date_name}")
    print(f"📁 Manifest: {manifest_path}")
    print(f"🎯 Sampel audit: {total_to_audit} windows (dari total {len(candidates)} kandidat, {len(filtered)} di atas conf {min_conf})")
    print("=" * 70)
    print("Petunjuk Navigasi:")
    print("  [1] / [y]  : Present (Kicau/panggilan burung asli terkonfirmasi)")
    print("  [0] / [n]  : Absent (Derau/serangga/hujan/angin/false positive)")
    print("  [b]        : Replay audio Best Beam (SPIR / SA / LabIR)")
    print("  [m]        : Play audio Mono asli untuk perbandingan")
    print("  [s]        : Skip jendela ini")
    print("  [q]        : Simpan & Keluar")
    print("=" * 70)

    results_dict = dict(existing_annotations)
    labels_patch_dict = {k: v.get("ground_truth") for k, v in existing_annotations.items()}

    tp_count = sum(1 for v in existing_annotations.values() if v.get("ground_truth") == 1)
    fp_count = sum(1 for v in existing_annotations.values() if v.get("ground_truth") == 0)

    try:
        for idx, item in enumerate(targets, 1):
            rec_id = item.get("recording_id", "")
            window = item.get("time_str") or str(item.get("window", ""))
            cand_key = f"{rec_id}_{window}"
            species = item.get("tentative_label") or item.get("tentative_species", "Unknown")
            best_ch = item.get("best_channel", "BestBeam")
            mono_c = item.get("mono_conf") if item.get("mono_conf") is not None else 0.0
            best_c = item.get("best_conf") if item.get("best_conf") is not None else item.get("best_beam_conf", 0.0)
            gain = item.get("conf_delta") if item.get("conf_delta") is not None else item.get("gain", 0.0)

            beam_clip, mono_clip = resolve_audio_paths(item, date_dir)

            mono_c_val = float(mono_c) if mono_c is not None else 0.0
            best_c_val = float(best_c) if best_c is not None else 0.0
            gain_val = float(gain) if gain is not None else (best_c_val - mono_c_val)

            print(f"\n──────────────────────────────────────────────────────────────────────")
            print(f"📍 [{idx}/{total_to_audit}] Window: {window} | Rec: {rec_id}")
            print(f"🐦 Species : {species}")
            print(f"📡 Channel : {best_ch}")
            print(f"📊 Conf    : Mono {mono_c_val:.2f}  ──▶  Beam {best_c_val:.2f}  (Delta: +{gain_val:.2f})")
            print(f"🔊 Playing Beam Audio...")

            play_audio(beam_clip)

            while True:
                prompt_text = "   Keputusan [1=Burung, 0=Derau, b=Replay Beam, m=Play Mono, s=Skip, q=Quit]: "
                choice = input(prompt_text).strip().lower()

                if choice in ["1", "y"]:
                    item_record = dict(item)
                    item_record["candidate_key"] = cand_key
                    item_record["ground_truth"] = 1
                    item_record["audit_source"] = "manual_vibe"
                    results_dict[cand_key] = item_record
                    labels_patch_dict[cand_key] = 1
                    tp_count += 1
                    print("   ✅ DITANDAI: 1 (True Avian Presence)")
                    break
                elif choice in ["0", "n"]:
                    item_record = dict(item)
                    item_record["candidate_key"] = cand_key
                    item_record["ground_truth"] = 0
                    item_record["audit_source"] = "manual_vibe"
                    results_dict[cand_key] = item_record
                    labels_patch_dict[cand_key] = 0
                    fp_count += 1
                    print("   ❌ DITANDAI: 0 (False Positive / Noise)")
                    break
                elif choice == "b":
                    print("   🔊 Replaying Beam...")
                    play_audio(beam_clip)
                elif choice == "m":
                    print("   🔊 Playing Mono Channel...")
                    play_audio(mono_clip)
                elif choice == "s":
                    print("   ⏩ Skipped.")
                    break
                elif choice == "q":
                    print("\n💾 Menyimpan anotasi sebelum keluar...")
                    save_annotations(gt_json_path, gt_csv_path, list(results_dict.values()))
                    update_markdown_manifest(md_path, labels_patch_dict)
                    print(f"✅ Selesai! {len(results_dict)} total jendela tersimpan.")
                    return
                else:
                    print("   ⚠️  Pilihan tidak dikenali. Ketik 1, 0, b, m, s, atau q.")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interupsi terdeteksi. Menyimpan progres...")
    finally:
        save_annotations(gt_json_path, gt_csv_path, list(results_dict.values()))
        update_markdown_manifest(md_path, labels_patch_dict)
        print("\n" + "=" * 70)
        print(f"📊 REKAP AUDIT TERBARU: {loc_name} | {date_name}")
        print(f"   • Total Jendela Terverifikasi: {len(results_dict)}")
        print(f"   • True Positive  (1 - Burung): {tp_count}")
        print(f"   • False Positive (0 - Derau) : {fp_count}")
        precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0
        print(f"   • Empirical Precision : {precision:.1f}%")
        print(f"   • Berkas Tersimpan    : {gt_json_path}")
        print(f"   • Manifest Terupdate  : {md_path}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="SEA Fast-Feedback Audio Auditing CLI (Vibe Coding)")
    parser.add_argument("--base-dir", type=str, default=None, help="Root folder output SEA (default: /Volumes/ri322/home/...)")
    parser.add_argument("--location", type=str, default=None, help="Deployment unit (e.g. 2D400, 2A400, S0, Q0, O0)")
    parser.add_argument("--date", type=str, default=None, help="Tanggal spesifik (e.g. 2026-07-16)")
    parser.add_argument("--sample", type=int, default=20, help="Jumlah sampel per sesi (default: 20)")
    parser.add_argument("--min-conf", type=float, default=0.30, help="Confidence threshold minimum (default: 0.30)")
    parser.add_argument("--random", action="store_true", help="Acak urutan kandidat (default: urutkan gain tertinggi)")

    args = parser.parse_args()
    base_dir = get_base_dir(args.base_dir)

    if args.location and args.date:
        manifest = os.path.join(base_dir, args.location, args.date, "detection_audit_manifest.json")
        if not os.path.exists(manifest):
            print(f"❌ Manifest tidak ditemukan di: {manifest}")
            sys.exit(1)
        run_audit(manifest, sample_size=args.sample, sort_by_gain=not args.random, min_conf=args.min_conf)
    else:
        available = find_available_dates(base_dir)
        if not available:
            print(f"❌ Tidak ditemukan tanggal yang selesai di {base_dir}")
            sys.exit(1)

        print("\n📅 PILIH TANGGAL UNTUK DIAUDIT:")
        for idx, item in enumerate(available[-15:], 1):
            print(f"  [{idx:2d}] {item['location']} | {item['date']}")
        print("  [0 ] Keluar")

        try:
            choice = int(input("\nMasukkan nomor pilihan (default: 15 / tanggal terbaru): ") or "15")
            if choice == 0 or choice > len(available[-15:]):
                print("Keluar.")
                return
            target = available[-15:][choice - 1]
            run_audit(target["manifest"], sample_size=args.sample, sort_by_gain=not args.random, min_conf=args.min_conf)
        except (ValueError, IndexError):
            print("Pilihan tidak valid.")


if __name__ == "__main__":
    main()
