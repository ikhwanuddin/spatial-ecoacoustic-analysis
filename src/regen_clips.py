"""
Recreate audit clips that ephemeral deleted (30 days after creation). Clips are deterministic:
FLAC + RTF + audit_clips.json (in HOME output_rtf) give the same clip again.

  python src/regen_clips.py --check                        # count missing clips, no GPU
  python src/regen_clips.py --part 0/4                     # regenerate this quarter of the location-dates
  python src/regen_clips.py --loc S0 --date 2026-08-30     # one location-date
  python src/regen_clips.py --loc S0 --date 2026-08-30 --clips-dir /tmp/x   # write elsewhere (test)

Only missing clips are written. Run one process per GPU with --part i/n.
"""

import os
import sys
import glob
import json
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_RTF_DIR, MONITORING_DATA, LOCATION_MAP, clip_dir

ap = argparse.ArgumentParser()
ap.add_argument("--check", action="store_true")
ap.add_argument("--part", default="0/1")
ap.add_argument("--loc")
ap.add_argument("--date")
ap.add_argument("--clips-dir", help="write here (flat) instead of config.clip_dir (test)")
args = ap.parse_args()

part, n_parts = map(int, args.part.split("/"))
locdates = sorted(p.split("/")[-2:] for p in glob.glob(os.path.join(OUTPUT_RTF_DIR, "*", "20*"))
                  if not os.path.basename(os.path.dirname(p)).startswith("_"))
locdates = [(l, d) for l, d in locdates if (not args.loc or l == args.loc) and (not args.date or d == args.date)]
locdates = locdates[part::n_parts]

if not args.check:
    import sea_gpu

total_missing = total_written = n_recs = 0
t0 = time.time()
for loc, date in locdates:
    for rec_json in sorted(glob.glob(os.path.join(OUTPUT_RTF_DIR, loc, date, "*", "audit_clips.json"))):
        rec_dir = os.path.dirname(rec_json)
        clips_dir = args.clips_dir or clip_dir(loc, date, os.path.basename(rec_dir))
        have = set(os.listdir(clips_dir)) if os.path.isdir(clips_dir) else set()
        rows = json.load(open(rec_json))
        missing = {c for r in rows for c in (r["clip"], r["mono_clip"]) if c not in have}
        if not missing:
            continue
        total_missing += len(missing)
        if args.check:
            continue
        rec = os.path.basename(rec_dir)
        flac = os.path.join(MONITORING_DATA, LOCATION_MAP[loc], date, rec + ".flac")
        r = sea_gpu.regen_clips(flac, loc, date, rec_dir, args.clips_dir)
        if "error" in r:
            print("ERROR", loc, date, rec, r["error"], flush=True)
            continue
        total_written += r["written"]
        n_recs += 1
        if n_recs % 20 == 0:
            print(f"{time.time() - t0:.0f}s recs {n_recs} written {total_written} (loc-date {loc} {date})", flush=True)

print(f"part {part}/{n_parts}: location-dates {len(locdates)}, missing clips {total_missing}, "
      f"written {total_written}, recordings regenerated {n_recs}, {time.time() - t0:.0f}s")
