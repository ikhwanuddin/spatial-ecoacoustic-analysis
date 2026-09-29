"""
One-off: move flat audit clips  <loc>/<date>/audit_clips/<clip>  ->  <loc>/<date>/audit_clips/<rec>/<clip>
(rename only, no data copied). Clip names are <rec>_<start>s_<channel>, rec = HH-MM-SS_dur=Nsecs.

  python src/move_clips_per_rec.py --dry-run
  python src/move_clips_per_rec.py --workers 8
"""

import os
import re
import sys
import glob
import argparse
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import AUDIT_CLIPS_DIR

NAME = re.compile(r"^(\d\d-\d\d-\d\d_dur=\d+secs)_\d+\.\ds_.+\.wav$")


def move_dir(d, dry_run):
    moved = bad = 0
    with os.scandir(d) as it:
        files = [e.name for e in it if e.is_file()]
    for name in files:
        m = NAME.match(name)
        if not m:
            bad += 1
            continue
        if not dry_run:
            os.makedirs(os.path.join(d, m.group(1)), exist_ok=True)
            os.rename(os.path.join(d, name), os.path.join(d, m.group(1), name))
        moved += 1
    return d, moved, bad


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    dirs = sorted(glob.glob(os.path.join(AUDIT_CLIPS_DIR, "*", "20*", "audit_clips")))
    tot_moved = tot_bad = 0
    with Pool(args.workers) as pool:
        for d, moved, bad in pool.starmap(move_dir, [(d, args.dry_run) for d in dirs]):
            tot_moved += moved
            tot_bad += bad
            if bad:
                print("UNMATCHED", bad, d, flush=True)
    print(f"{'dry-run ' if args.dry_run else ''}dirs {len(dirs)}, files {'to move' if args.dry_run else 'moved'} "
          f"{tot_moved}, unmatched {tot_bad}")
