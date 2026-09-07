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
import time
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor

LOCAL_CACHE_DIR = "/tmp/sea_audit_cache"
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


current_player_process: Optional[subprocess.Popen] = None


def stop_audio(app_name: Optional[str] = "ocenaudio"):
    """Stops any currently playing background audio process or visual app playback."""
    global current_player_process
    if current_player_process and current_player_process.poll() is None:
        try:
            current_player_process.terminate()
            current_player_process.wait(timeout=0.2)
        except Exception:
            pass
        current_player_process = None

    if app_name and app_name.lower() != "none":
        try:
            stop_app_audio(app_name)
        except Exception:
            pass


def play_audio(audio_path: Optional[str], blocking: bool = False) -> bool:
    """Plays audio via macOS afplay or available Linux players."""
    global current_player_process
    stop_audio()
    if not audio_path or not os.path.exists(audio_path):
        print(f"⚠️  Audio file not found: {audio_path}")
        return False

    if sys.platform == "darwin":
        try:
            if blocking:
                subprocess.run(["afplay", audio_path], check=False)
            else:
                current_player_process = subprocess.Popen(
                    ["afplay", audio_path],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            return True
        except Exception as e:
            print(f"⚠️  Failed to play audio via afplay: {e}")
            return False
    else:
        # Linux (e.g. CX3 or remote server)
        for player in ["paplay", "aplay", "ffplay", "mpv"]:
            if shutil.which(player):
                try:
                    if player == "ffplay":
                        cmd = [player, "-nodisp", "-autoexit", audio_path]
                    else:
                        cmd = [player, audio_path]
                    if blocking:
                        subprocess.run(cmd, stderr=subprocess.DEVNULL, check=False)
                    else:
                        current_player_process = subprocess.Popen(
                            cmd,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL
                        )
                    return True
                except Exception:
                    pass
        print(f"🔈 [Audio File]: {audio_path}")
        print("💡 TIP: Run audit_vibe.py in Mac mini terminal for automatic audio playback via speakers/headphones!")
        return False


def open_in_app(audio_path: Optional[str], app_name: str = "ocenaudio", background: bool = True) -> bool:
    """Opens audio file in the specified visual application (e.g., ocenaudio on macOS)."""
    if not audio_path or not os.path.exists(audio_path):
        return False
    if sys.platform == "darwin":
        try:
            cmd = ["open"]
            if background:
                cmd.append("-g")
            cmd.extend(["-a", app_name, audio_path])
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                p.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            return True
        except Exception as e:
            print(f"⚠️  Failed to open in {app_name}: {e}")
            return False
    else:
        if "DISPLAY" in os.environ and shutil.which("xdg-open"):
            try:
                subprocess.Popen(["xdg-open", audio_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                return True
            except Exception:
                pass
        return False


def open_in_app_multi(audio_paths: List[str], app_name: str = "ocenaudio", background: bool = True) -> bool:
    """Opens multiple audio files in the specified visual application."""
    valid_paths = [p for p in audio_paths if p and os.path.exists(p)]
    if not valid_paths:
        return False
    if sys.platform == "darwin":
        try:
            cmd = ["open"]
            if background:
                cmd.append("-g")
            cmd.extend(["-a", app_name] + valid_paths)
            p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                p.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                pass
            return True
        except Exception as e:
            print(f"⚠️  Failed to open in {app_name}: {e}")
            return False
    return False


def close_app_files(app_name: str = "ocenaudio") -> bool:
    """Safely closes open files in the visual application via targeted AX menu click without any keystrokes."""
    if sys.platform != "darwin" or not app_name or app_name.lower() == "none":
        return False

    script = f'''
    tell application "System Events"
        if not (exists process "{app_name}") then return
        set origProc to first application process whose frontmost is true
        set origName to name of origProc
    end tell

    -- Activate ocenaudio briefly so Qt menu bar is responsive
    tell application "{app_name}" to activate
    delay 0.15

    -- Targeted menu click only. Never use keystroke to keep terminal tabs safe.
    tell application "System Events" to tell process "{app_name}"
        try
            tell menu bar 1
                tell menu bar item "File"
                    tell menu "File"
                        click menu item "Close All"
                    end tell
                end tell
            end tell
        end try
    end tell

    delay 0.05

    -- Restore focus back to original terminal (e.g. Ghostty)
    if origName is not "" and origName is not "{app_name}" then
        tell application origName to activate
    end if
    '''
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0
        )
        return True
    except Exception:
        return False


def play_app_audio(app_name: str = "ocenaudio") -> bool:
    """Triggers playback in the visual application (e.g., ocenaudio) while keeping terminal focus."""
    if sys.platform != "darwin" or not app_name or app_name.lower() == "none":
        return False

    script = f'''
    tell application "System Events"
        if not (exists process "{app_name}") then return
        set origProc to first application process whose frontmost is true
        set origName to name of origProc
    end tell

    tell application "{app_name}" to activate
    delay 0.12

    tell application "System Events" to tell process "{app_name}"
        try
            tell menu bar 1 to tell menu bar item "Controls" to tell menu "Controls"
                set firstItem to name of menu item 1
                if firstItem is "Pause" then
                    click menu item "Stop"
                    delay 0.05
                end if
                click menu item "Play"
            end tell
        end try
    end tell

    delay 0.05
    if origName is not "" and origName is not "{app_name}" then
        tell application origName to activate
    end if
    '''
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=2.0
        )
        return True
    except Exception:
        return False


def stop_app_audio(app_name: str = "ocenaudio") -> bool:
    """Stops playback in the visual application from background without activating the app."""
    if sys.platform != "darwin" or not app_name or app_name.lower() == "none":
        return False

    script = f'''
    tell application "System Events"
        if not (exists process "{app_name}") then return
        tell process "{app_name}"
            try
                tell menu bar 1 to tell menu bar item "Controls" to tell menu "Controls"
                    if name of menu item 1 is "Pause" then
                        click menu item "Stop"
                    end if
                end tell
            end try
        end tell
    end tell
    '''
    try:
        subprocess.run(
            ["osascript", "-e", script],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=1.0
        )
        return True
    except Exception:
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


def get_prioritized_dates(base_dir: str, location: str) -> List[Dict[str, Any]]:
    """
    Ranks dates for a given location by audit priority:
    1. Dates with 0 audited windows (sorted chronologically).
    2. Dates with fewest audited windows that still have remaining unaudited candidates.
    """
    results = []
    use_ssh = False
    if sys.platform == "darwin":
        try:
            res = subprocess.run(["ssh", "-o", "ConnectTimeout=2", "-q", "cx3", "true"], check=False)
            if res.returncode == 0:
                use_ssh = True
        except Exception:
            use_ssh = False

    if use_ssh:
        try:
            cx3_cmd = "python3 -c '\''import os, json; loc = \"" + location + "\"; base = \"/rds/general/user/ri322/home/spatial-ecoacoustic-analysis/output/\" + loc; res = []; [res.append({\"location\": loc, \"date\": d, \"audited_count\": len(json.load(open(os.path.join(base, d, \"audit_ground_truth.json\")))) if os.path.exists(os.path.join(base, d, \"audit_ground_truth.json\")) else 0}) for d in sorted(os.listdir(base)) if os.path.exists(os.path.join(base, d, \"detection_audit_manifest.json\"))]; print(json.dumps(res))'\''"
            out = subprocess.check_output(["ssh", "-q", "cx3", cx3_cmd], timeout=5.0).decode().strip()
            raw_dates = json.loads(out)
            for item in raw_dates:
                mf_path = os.path.join(base_dir, location, item["date"], "detection_audit_manifest.json")
                item["manifest"] = mf_path
                results.append(item)
        except Exception:
            results = []

    if not results:
        loc_dir = os.path.join(base_dir, location)
        if os.path.isdir(loc_dir):
            for d in sorted(os.listdir(loc_dir)):
                mf = os.path.join(loc_dir, d, "detection_audit_manifest.json")
                if not os.path.exists(mf):
                    continue
                gt = os.path.join(loc_dir, d, "audit_ground_truth.json")
                aud = 0
                if os.path.exists(gt):
                    try:
                        with open(gt, "r", encoding="utf-8") as f:
                            aud = len(json.load(f))
                    except Exception:
                        aud = 0
                results.append({"location": location, "date": d, "audited_count": aud, "manifest": mf})

    results.sort(key=lambda x: (x.get("audited_count", 0), x.get("date", "")))
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


def clean_local_cache():
    """Removes temporary local audio cache immediately."""
    if os.path.exists(LOCAL_CACHE_DIR):
        try:
            shutil.rmtree(LOCAL_CACHE_DIR, ignore_errors=True)
            print(f"🧹 Cleaned up local cache ({LOCAL_CACHE_DIR})")
        except Exception:
            pass


def precache_clips(targets: List[Dict[str, Any]], date_dir: str) -> Dict[str, Tuple[Optional[str], Optional[str]]]:
    """Copies all sampled audio clips to local /tmp SSD concurrently for zero-latency inspection."""
    if os.path.exists(LOCAL_CACHE_DIR):
        shutil.rmtree(LOCAL_CACHE_DIR, ignore_errors=True)
    os.makedirs(LOCAL_CACHE_DIR, exist_ok=True)

    cache_map = {}
    copy_tasks = []

    for item in targets:
        rec_id = item.get("recording_id", "")
        win_str = item.get("time_str") or str(item.get("window", ""))
        key = f"{rec_id}_{win_str}"

        beam_src, mono_src = resolve_audio_paths(item, date_dir)
        beam_cached = None
        mono_cached = None

        if beam_src and os.path.exists(beam_src):
            dst = os.path.join(LOCAL_CACHE_DIR, os.path.basename(beam_src))
            copy_tasks.append((beam_src, dst))
            beam_cached = dst
        else:
            beam_cached = beam_src

        if mono_src and os.path.exists(mono_src):
            dst = os.path.join(LOCAL_CACHE_DIR, os.path.basename(mono_src))
            copy_tasks.append((mono_src, dst))
            mono_cached = dst
        else:
            mono_cached = mono_src

        cache_map[key] = (beam_cached, mono_cached)

    if copy_tasks:
        unique_tasks = list({src: dst for src, dst in copy_tasks}.items())
        print(f"⚡ Pre-caching audio for {len(targets)} windows ({len(unique_tasks)} clips: Mono & Beam pairs) to local /tmp SSD...")
        t0 = time.time()

        use_rsync_ssh = False
        if sys.platform == "darwin":
            try:
                res = subprocess.run(["ssh", "-o", "ConnectTimeout=2", "-q", "cx3", "true"], check=False)
                if res.returncode == 0:
                    use_rsync_ssh = True
            except Exception:
                use_rsync_ssh = False

        if use_rsync_ssh:
            cx3_prefix = "/rds/general/user/ri322/home/spatial-ecoacoustic-analysis/"
            file_list = []
            for src, _ in unique_tasks:
                if "/output/" in src:
                    rel = "output/" + src.split("/output/", 1)[1]
                    file_list.append(rel)

            if file_list:
                list_file = os.path.join(LOCAL_CACHE_DIR, "_rsync_files.txt")
                try:
                    with open(list_file, "w") as f:
                        f.write("\n".join(file_list) + "\n")
                    cmd = [
                        "rsync", "-a", "--no-relative",
                        f"--files-from={list_file}",
                        f"cx3:{cx3_prefix}",
                        LOCAL_CACHE_DIR + "/"
                    ]
                    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15.0)
                    if os.path.exists(list_file):
                        os.remove(list_file)
                    elapsed = time.time() - t0
                    print(f"✅ Ready! {len(unique_tasks)} clips for {len(targets)} windows transferred via SSH in {elapsed:.1f}s (zero SMB latency).")
                    return cache_map
                except Exception:
                    pass

        # Fallback: multi-threaded local/SMB copy
        try:
            def _copy(pair):
                src, dst = pair
                if os.path.exists(src) and not os.path.exists(dst):
                    shutil.copyfile(src, dst)

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(_copy, unique_tasks))
            elapsed = time.time() - t0
            print(f"✅ Ready! {len(unique_tasks)} clips for {len(targets)} windows cached in {elapsed:.1f}s.")
        except Exception as e:
            print(f"⚠️  Pre-caching note: {e}")

    return cache_map


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
    if not annotations and not os.path.exists(gt_json_path):
        return

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
                    win_clean = parts[2].replace("`", "").strip()
                    rec_clean = parts[3].replace("`", "").strip()
                    match_label = None
                    for k, lbl in annotations_dict.items():
                        if rec_clean in k and win_clean in k:
                            match_label = lbl
                            break
                    if match_label is not None:
                        parts[9] = f" `{match_label}` "
                        line = "| " + " | ".join(parts[1:-1]) + " |\n"
            updated_lines.append(line)

        with open(md_path, "w", encoding="utf-8") as f:
            f.writelines(updated_lines)
    except Exception as e:
        print(f"⚠️  Failed to update Markdown manifest: {e}")


def reset_audit(manifest_path: str):
    """Resets all ground-truth audit annotations for the given manifest."""
    date_dir = os.path.dirname(manifest_path)
    gt_json_path = os.path.join(date_dir, "audit_ground_truth.json")
    gt_csv_path = os.path.join(date_dir, "audit_ground_truth.csv")
    md_path = os.path.join(date_dir, "detection_audit_manifest.md")

    removed = []
    if os.path.exists(gt_json_path):
        try:
            os.remove(gt_json_path)
            removed.append("audit_ground_truth.json (deleted)")
        except Exception as e:
            print(f"⚠️  Failed to delete {gt_json_path}: {e}")

    if os.path.exists(gt_csv_path):
        try:
            os.remove(gt_csv_path)
            removed.append("audit_ground_truth.csv (deleted)")
        except Exception as e:
            print(f"⚠️  Failed to delete {gt_csv_path}: {e}")

    if os.path.exists(md_path):
        try:
            with open(md_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            updated_lines = []
            for line in lines:
                if line.startswith("|") and not line.startswith("| #") and not line.startswith("|---"):
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 11:
                        parts[9] = " ` ` "
                        line = "| " + " | ".join(parts[1:-1]) + " |\n"
                updated_lines.append(line)
            with open(md_path, "w", encoding="utf-8") as f:
                f.writelines(updated_lines)
            removed.append("detection_audit_manifest.md ('Present?' column reset)")
        except Exception as e:
            print(f"⚠️  Failed to reset Markdown manifest: {e}")

    if removed:
        print(f"🔄 Successfully reset audit progress in: {date_dir}")
        for r in removed:
            print(f"   • {r}")
    else:
        print(f"ℹ️  No previous audit data found in: {date_dir}")



def run_audit(manifest_path: str, sample_size: int = 20, sort_by_gain: bool = True, min_conf: float = 0.30, app_name: str = "ocenaudio", play_sound: bool = True):
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
        print("❌ No detection candidates found in this manifest.")
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
        print("✅ All candidates for this date have already been audited!")
        return "completed"

    # Pre-cache sampled audio clips to local /tmp SSD concurrently
    cached_paths_map = precache_clips(targets, date_dir)

    use_app = bool(app_name and app_name.lower() != "none")

    print("=" * 70)
    print(f"🎧 SEA VIBE AUDIT: {loc_name} | {date_name}")
    print(f"📁 Manifest: {manifest_path}")
    print(f"🎯 Audit sample: {total_to_audit} windows (out of {len(candidates)} candidates, {len(filtered)} above conf {min_conf})")
    if use_app:
        print(f"🖥️  Visual App: {app_name} (waveform & spectrogram)")
    print("=" * 70)
    print("Navigation:")
    print("  [1] / [y]  : Present (Confirmed true avian vocalization)")
    print("  [0] / [n]  : Absent (Noise / insects / rain / wind / false positive)")
    print("  [b]        : Replay & inspect Best Beam audio (SPIR / SA / LabIR)")
    print("  [m]        : Play & inspect original Mono audio for comparison")
    if use_app:
        print(f"  [o]        : Bring {app_name} window to front")
    print("  [s]        : Skip this window")
    print("  [q]        : Save & Quit")
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

            if cand_key in cached_paths_map:
                beam_clip, mono_clip = cached_paths_map[cand_key]
            else:
                beam_clip, mono_clip = resolve_audio_paths(item, date_dir)

            mono_c_val = float(mono_c) if mono_c is not None else 0.0
            best_c_val = float(best_c) if best_c is not None else 0.0
            gain_val = float(gain) if gain is not None else (best_c_val - mono_c_val)

            print(f"\n──────────────────────────────────────────────────────────────────────")
            print(f"📍 [{idx}/{total_to_audit}] Window: {window} | Rec: {rec_id}")
            print(f"🐦 Species : {species}")
            print(f"📡 Channel : {best_ch}")
            print(f"📊 Conf    : Mono {mono_c_val:.2f}  ──▶  Beam {best_c_val:.2f}  (Delta: +{gain_val:.2f})")

            # Audiovisual inspection in ocenaudio
            # Determine which clip has higher confidence
            if best_c_val >= mono_c_val:
                primary_clip = beam_clip
                secondary_clip = mono_clip
                active_label = f"Beam ({best_c_val:.2f})"
            else:
                primary_clip = mono_clip
                secondary_clip = beam_clip
                active_label = f"Mono ({mono_c_val:.2f})"

            if use_app:
                # Close previous candidate clips in ocenaudio (only after first candidate)
                if idx > 1:
                    close_app_files(app_name=app_name)

                # Open secondary first, then primary second so ocenaudio highlights primary
                if secondary_clip and os.path.exists(secondary_clip):
                    open_in_app(secondary_clip, app_name=app_name, background=True)
                    time.sleep(0.05)
                if primary_clip and os.path.exists(primary_clip):
                    open_in_app(primary_clip, app_name=app_name, background=True)
                    time.sleep(0.05)

                print(f"🖥️  Opened in {app_name} [Highlighted: {active_label}]")
            else:
                primary_clip = beam_clip or mono_clip

            if play_sound:
                if use_app and app_name.lower() == "ocenaudio":
                    print(f"🔊 Auto-playing in {app_name} [{active_label}]...")
                    play_app_audio(app_name=app_name)
                elif primary_clip:
                    print(f"🔊 Playing audio ({active_label})...")
                    play_audio(primary_clip, blocking=False)
            else:
                print(f"🔇 Visual-only mode (--no-play). Press [Space] in {app_name} to play.")

            while True:
                prompt_opts = "1=Bird, 0=Noise, b=Beam, m=Mono"
                if use_app:
                    prompt_opts += ", o=Focus App"
                prompt_opts += ", s=Skip, q=Quit"
                prompt_text = f"   Decision [{prompt_opts}]: "
                choice = input(prompt_text).strip().lower()

                if choice in ["1", "y"]:
                    stop_audio(app_name=app_name)
                    item_record = dict(item)
                    item_record["candidate_key"] = cand_key
                    item_record["ground_truth"] = 1
                    item_record["audit_source"] = "manual_vibe"
                    results_dict[cand_key] = item_record
                    labels_patch_dict[cand_key] = 1
                    tp_count += 1
                    print("   ✅ LABELED: 1 (True Avian Presence)")
                    break
                elif choice in ["0", "n"]:
                    stop_audio(app_name=app_name)
                    item_record = dict(item)
                    item_record["candidate_key"] = cand_key
                    item_record["ground_truth"] = 0
                    item_record["audit_source"] = "manual_vibe"
                    results_dict[cand_key] = item_record
                    labels_patch_dict[cand_key] = 0
                    fp_count += 1
                    print("   ❌ LABELED: 0 (False Positive / Noise)")
                    break
                elif choice == "b":
                    print("   🔊 Switching to Beam & Playing...")
                    if use_app and beam_clip:
                        open_in_app(beam_clip, app_name=app_name, background=True)
                        if play_sound:
                            play_app_audio(app_name=app_name)
                    elif play_sound and beam_clip:
                        play_audio(beam_clip, blocking=False)
                elif choice == "m":
                    print("   🔊 Switching to Mono & Playing...")
                    if use_app and mono_clip:
                        open_in_app(mono_clip, app_name=app_name, background=True)
                        if play_sound:
                            play_app_audio(app_name=app_name)
                    elif play_sound and mono_clip:
                        play_audio(mono_clip, blocking=False)
                elif choice == "o" and use_app:
                    print(f"   🖥️  Focusing {app_name} to front...")
                    active_clip = primary_clip or beam_clip or mono_clip
                    if active_clip:
                        open_in_app(active_clip, app_name=app_name, background=False)
                    print("   💡 Tip: Click back to the terminal window to enter label [1/0].")
                elif choice == "s":
                    stop_audio(app_name=app_name)
                    print("   ⏩ Skipped.")
                    break
                elif choice == "q":
                    stop_audio(app_name=app_name)
                    print("\n💾 Saving annotations and exiting...")
                    return "quit"
                else:
                    valid_keys = "1, 0, b, m, o, s, or q" if use_app else "1, 0, b, m, s, or q"
                    print(f"   ⚠️  Unrecognized option. Enter {valid_keys}.")

    except (KeyboardInterrupt, EOFError):
        stop_audio(app_name=app_name)
        print("\n\n⚠️  Session interrupted. Saving progress...")
        return "interrupted"
    finally:
        stop_audio(app_name=app_name)
        if use_app:
            close_app_files(app_name=app_name)
        clean_local_cache()
        save_annotations(gt_json_path, gt_csv_path, list(results_dict.values()))
        update_markdown_manifest(md_path, labels_patch_dict)
        print("\n" + "=" * 70)
        print(f"📊 LATEST AUDIT SUMMARY: {loc_name} | {date_name}")
        print(f"   • Total Windows Verified : {len(results_dict)}")
        print(f"   • True Positive  (1 - Bird) : {tp_count}")
        print(f"   • False Positive (0 - Noise): {fp_count}")
        precision = (tp_count / (tp_count + fp_count) * 100) if (tp_count + fp_count) > 0 else 0
        print(f"   • Empirical Precision    : {precision:.1f}%")
        print(f"   • Saved Annotation File  : {gt_json_path}")
        print(f"   • Updated Manifest       : {md_path}")
        print("=" * 70)

    return "completed"


def run_auto_audit_loop(base_dir: str, location: str, args: argparse.Namespace, app_name: str, play_sound: bool):
    """Continuously audits prioritized dates for a location (0 audited dates first, then least audited)."""
    while True:
        prioritized = get_prioritized_dates(base_dir, location)
        if not prioritized:
            print(f"❌ No valid audit manifests found for location: {location}")
            sys.exit(1)

        # Pick top prioritized date
        target = prioritized[0]
        date_str = target["date"]

        print("\n" + "=" * 70)
        print(f"🎯 AUTO-PRIORITIZED DATE QUEUE: {location}")
        for idx, item in enumerate(prioritized[:5], 1):
            marker = " ◀ TARGET" if idx == 1 else ""
            status_text = "0 audited" if item["audited_count"] == 0 else f"{item['audited_count']} audited"
            print(f"   [{idx}] {item['date']} ({status_text}){marker}")
        print("=" * 70)

        if args.reset:
            reset_audit(target["manifest"])
            return

        status = run_audit(
            target["manifest"],
            sample_size=args.sample,
            sort_by_gain=not args.random,
            min_conf=args.min_conf,
            app_name=app_name,
            play_sound=play_sound
        )

        if status != "completed":
            break

        # Re-fetch priority queue to check next target
        next_queue = get_prioritized_dates(base_dir, location)
        if not next_queue:
            print(f"\n🎉 All dates for {location} have been audited!")
            break

        next_target = next_queue[0]
        next_date_str = next_target["date"]
        next_status = "0 audited" if next_target["audited_count"] == 0 else f"{next_target['audited_count']} audited"
        prompt = f"\n👉 Continue to next prioritized date [{next_date_str} - {next_status}]? [Enter=Yes / q=Exit]: "
        try:
            ans = input(prompt).strip().lower()
            if ans in ["q", "n", "exit"]:
                print("Exiting audit session.")
                break
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="SEA Fast-Feedback Audio Auditing CLI (Vibe Coding)")
    parser.add_argument("--base-dir", type=str, default=None, help="Root SEA output directory (default: /Volumes/ri322/home/...)")
    parser.add_argument("--location", type=str, default=None, help="Deployment unit (e.g. 2D400, 2A400, S0, Q0, O0)")
    parser.add_argument("--date", type=str, default=None, help="Specific date (e.g. 2026-07-16)")
    parser.add_argument("--sample", type=int, default=20, help="Number of samples per session (default: 20)")
    parser.add_argument("--min-conf", type=float, default=0.30, help="Minimum confidence threshold (default: 0.30)")
    parser.add_argument("--random", action="store_true", help="Randomize candidate order (default: sort by highest gain)")
    parser.add_argument("--app", type=str, default="ocenaudio", help="Audio visual application (default: ocenaudio, 'none' for terminal only)")
    parser.add_argument("--no-play", action="store_true", help="Disable background audio playback (visual inspection in app only)")
    parser.add_argument("--reset", action="store_true", help="Reset/delete all audit annotations for the selected date/location")

    args = parser.parse_args()
    base_dir = get_base_dir(args.base_dir)
    play_sound = not args.no_play
    app_name = args.app

    if args.location and args.date:
        manifest = os.path.join(base_dir, args.location, args.date, "detection_audit_manifest.json")
        if not os.path.exists(manifest):
            print(f"❌ Manifest not found at: {manifest}")
            sys.exit(1)
        if args.reset:
            reset_audit(manifest)
            return
        run_audit(manifest, sample_size=args.sample, sort_by_gain=not args.random, min_conf=args.min_conf, app_name=app_name, play_sound=play_sound)
    elif args.location:
        run_auto_audit_loop(base_dir, args.location.upper(), args, app_name, play_sound)
    else:
        # Prompt for location, then auto-prioritize
        locations = ["2A400", "2B400", "2D400", "S0", "Q0", "O0"]
        print("\n📍 SELECT LOCATION TO AUDIT:")
        for idx, loc in enumerate(locations, 1):
            print(f"  [{idx}] {loc}")
        print("  [0] Exit")

        try:
            choice = input("\nEnter selection number (default: 1 / 2A400): ").strip()
            if not choice:
                chosen_loc = "2A400"
            elif choice == "0":
                return
            else:
                idx = int(choice) - 1
                if 0 <= idx < len(locations):
                    chosen_loc = locations[idx]
                else:
                    print("Invalid choice.")
                    return
        except Exception:
            return

        run_auto_audit_loop(base_dir, chosen_loc, args, app_name, play_sound)


if __name__ == "__main__":
    main()
