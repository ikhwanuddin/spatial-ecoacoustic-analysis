"""
High-Performance Audio Snippet Extractor for Ground-Truth Binary Calibration (Option 3).
Batches audio reads by recording in memory (KISS & ultra-fast), slices 5-second candidate
windows into permanent $HOME, links them in detection_audit_manifest.md, and cleans up
intermediate scratch WAV directories.
"""

import os
import sys
import json
import shutil
import argparse
import soundfile as sf
from typing import List, Dict, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import SCRATCH_DIR, OUTPUT_DIR, HOME_DIR
from generate_audit_manifest import format_manifest_markdown


def clip_names(rec_id: str, item: Dict[str, Any]):
    """File names of the beam clip and the mono clip of one audit window."""
    st = item["window_seconds"][0]
    ch_tag = os.path.splitext(item["best_channel"])[0].replace("(", "_").replace(")", "").replace(" ", "_")
    time_tag = f"{int(st):04d}s"
    return f"{rec_id}_{time_tag}_{ch_tag}.wav", f"{rec_id}_{time_tag}_mono.wav"


def cut_recording_clips(rec_scratch: str, rec_id: str, rec_items: List[Dict[str, Any]], clips_dir: str,
                        buffer_sec: float = 1.0) -> int:
    """Cut the beam + mono clip of every audit window of one recording from its scratch WAVs.
    Clips that already exist are kept. Returns the number of windows handled."""
    os.makedirs(clips_dir, exist_ok=True)

    # Cache unique needed channels in RAM
    needed_channels = {"mono.wav"} | {item["best_channel"] for item in rec_items}
    audio_cache = {}
    for ch in needed_channels:
        src_p = os.path.join(rec_scratch, ch)
        if os.path.isfile(src_p):
            audio_cache[ch] = sf.read(src_p, dtype='float32')

    for item in rec_items:
        st, et = item["window_seconds"]
        clip_st = max(0.0, float(st) - buffer_sec)
        clip_et = float(et) + buffer_sec
        bf_name, mono_name = clip_names(rec_id, item)
        for ch, name in ((item["best_channel"], bf_name), ("mono.wav", mono_name)):
            dst = os.path.join(clips_dir, name)
            if ch in audio_cache and not os.path.isfile(dst):
                data, sr = audio_cache[ch]
                sf.write(dst, data[int(clip_st * sr):min(len(data), int(clip_et * sr))], sr, subtype='PCM_16')
    return len(rec_items)


def extract_clips_for_date(
    location: str,
    date_str: str,
    clean_scratch: bool = False,
    buffer_sec: float = 1.0
) -> Dict[str, Any]:
    out_date_dir = os.path.join(OUTPUT_DIR, location, date_str)
    json_path = os.path.join(out_date_dir, "detection_audit_manifest.json")
    md_path = os.path.join(out_date_dir, "detection_audit_manifest.md")
    scratch_date_dir = os.path.join(SCRATCH_DIR, location, date_str)

    space_freed_mb = 0.0

    if not os.path.isfile(json_path):
        if clean_scratch and os.path.isdir(scratch_date_dir):
            try:
                total_b = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(scratch_date_dir) for f in files)
                space_freed_mb = total_b / (1024 * 1024)
                shutil.rmtree(scratch_date_dir)
            except Exception:
                pass
        return {"status": "no_manifest", "clips_extracted": 0, "space_freed_mb": space_freed_mb}

    with open(json_path, "r", encoding="utf-8") as f:
        manifest_items = json.load(f)

    clips_dir = os.path.join(out_date_dir, "audit_clips")
    os.makedirs(clips_dir, exist_ok=True)

    # Group candidate windows by recording_id
    by_rec: Dict[str, List[Dict[str, Any]]] = {}
    for item in manifest_items:
        by_rec.setdefault(item["recording_id"], []).append(item)

    extracted_count = 0

    for rec_id, rec_items in by_rec.items():
        rec_scratch = os.path.join(scratch_date_dir, rec_id)
        if os.path.isdir(rec_scratch):
            extracted_count += cut_recording_clips(rec_scratch, rec_id, rec_items, clips_dir, buffer_sec)

        for item in rec_items:
            bf_name, mono_name = clip_names(rec_id, item)
            mac_dir = f"/Volumes/ri322/home/spatial-ecoacoustic-analysis/output/{location}/{date_str}/audit_clips"
            item.setdefault("audio_paths", {})
            item["audio_paths"]["cx3_wav"] = os.path.join(clips_dir, bf_name)
            item["audio_paths"]["cx3_mono_wav"] = os.path.join(clips_dir, mono_name)
            item["audio_paths"]["mac_wav"] = f"{mac_dir}/{bf_name}"
            item["audio_paths"]["mac_mono_wav"] = f"{mac_dir}/{mono_name}"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(manifest_items, f, indent=4, ensure_ascii=False)

    md_content = format_manifest_markdown(manifest_items, location, date_str)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content + "\n")

    if clean_scratch and os.path.isdir(scratch_date_dir):
        try:
            total_b = sum(os.path.getsize(os.path.join(r, f)) for r, d, files in os.walk(scratch_date_dir) for f in files)
            space_freed_mb = total_b / (1024 * 1024)
            shutil.rmtree(scratch_date_dir)
            print(f"  🗑️  Cleaned {scratch_date_dir} -> Reclaimed {space_freed_mb:.1f} MB ({space_freed_mb/1024:.2f} GB)")
        except Exception as e:
            print(f"  ⚠️ Could not clean scratch {scratch_date_dir}: {e}")

    return {
        "status": "success",
        "clips_extracted": extracted_count,
        "space_freed_mb": space_freed_mb
    }


def _worker_wrapper(args_tuple):
    loc, d, clean = args_tuple
    try:
        return (loc, d, extract_clips_for_date(loc, d, clean_scratch=clean))
    except Exception as e:
        return (loc, d, {"status": "error", "error": str(e)})


def main():
    parser = argparse.ArgumentParser(description="Extract 5-second audio snippets for Ground-Truth Audit & clean scratch.")
    parser.add_argument("--location", default=None, help="Location code (e.g. S0, 2D400)")
    parser.add_argument("--date", default=None, help="Date string (YYYY-MM-DD)")
    parser.add_argument("--all-completed", action="store_true", help="Process all completed dates across all locations")
    parser.add_argument("--clean-scratch", action="store_true", help="Delete scratch directory after extracting clips")
    parser.add_argument("--workers", type=int, default=4, help="Parallel worker processes (default: 4)")
    args = parser.parse_args()

    if args.location and args.date:
        res = extract_clips_for_date(args.location, args.date, clean_scratch=args.clean_scratch)
        print(f"Done: {args.location} {args.date} -> {res}")
    elif args.all_completed:
        from multiprocessing import Pool
        locations = ["2A400", "2B400", "2D400", "Q0", "S0"]
        tasks = []
        for loc in locations:
            loc_scratch = os.path.join(SCRATCH_DIR, loc)
            if not os.path.isdir(loc_scratch):
                continue
            for d in sorted(os.listdir(loc_scratch)):
                sum_p = os.path.join(OUTPUT_DIR, loc, d, "daily_summary.md")
                if os.path.isfile(sum_p):
                    tasks.append((loc, d, args.clean_scratch))

        print(f"🚀 Starting parallel snippet extraction on {len(tasks)} completed dates with {args.workers} workers...")
        total_freed_mb = 0.0
        total_clips = 0
        with Pool(processes=args.workers) as pool:
            for loc, d, res in pool.imap_unordered(_worker_wrapper, tasks):
                freed = res.get("space_freed_mb", 0.0)
                clips = res.get("clips_extracted", 0)
                total_freed_mb += freed
                total_clips += clips
                print(f"  ✓ {loc} {d}: {clips} clips | Freed: {freed:.1f} MB ({freed/1024:.2f} GB)")

        print("=" * 70)
        print(f"🎉 ALL COMPLETED DATES PROCESSED!")
        print(f"   Total candidate windows clipped: {total_clips}")
        print(f"   Total Ephemeral space reclaimed: {total_freed_mb/1024:.2f} GB ({total_freed_mb/(1024*1024):.2f} TB)")
        print("=" * 70)


if __name__ == "__main__":
    main()
