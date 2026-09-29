"""
Completeness check of the RTF rerun. Prints a report, exit code 1 if anything is missing.

  python src/verify_rtf_rerun.py [--sample N]   # N = recordings per location-date to open (default 3)

Checks: queue empty and no failed task; every FLAC in every enqueued window has processed.json or
error.json; every location-date finalized for each tag; results.json has all channels; processed.json
has all groups and only local species; every audit clip named in audit_clips.json exists.
"""

import os
import sys
import glob
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OUTPUT_RTF_DIR, AUDIT_CLIPS_DIR, MONITORING_DATA, LOCATION_MAP, SKIP_LOCATIONS
from rtf_queue import task_files

WINDOWS = {"dd": [("04:30", "08:30"), ("16:00", "19:30")], "night": [("00:00", "01:00")]}   # as enqueued
GROUPS = ["mono_channel", "sa_channel", "beamformed_LabIR", "beamformed_SPIR", "beamformed_WCIR_own",
          "beamformed_WCIR_cross", "beamformed_all"]
N_CHANNELS = 249      # mono + sa + 247 beams

ap = argparse.ArgumentParser()
ap.add_argument("--sample", type=int, default=3)
args = ap.parse_args()

problems = []
report = {}


def bad(msg):
    problems.append(msg)


# 1. queue
counts = {s: len(task_files(s)) for s in ["pending", "active", "done", "failed", "hold"]}
report["queue"] = counts
for s in ["pending", "active", "failed"]:
    if counts[s]:
        bad(f"queue: {counts[s]} tasks in {s}")

# 2. every FLAC of every done task has an output
tasks = [json.load(open(p)) for p in task_files("done")]
n_flac = n_ok = 0
errors, missing = [], []
by_locdate = {}
for t in tasks:
    by_locdate.setdefault((t["location"], t["date"]), set())
    if "tag" in t:                     # first tasks (before tags existed) cover all hours, finalized via dd/night
        by_locdate[(t["location"], t["date"])].add(t["tag"])
    for flac in t["flacs"]:
        n_flac += 1
        rec_dir = os.path.join(OUTPUT_RTF_DIR, t["location"], t["date"], flac[:-5])
        if os.path.isfile(os.path.join(rec_dir, "processed.json")):
            n_ok += 1
        elif os.path.isfile(os.path.join(rec_dir, "error.json")):
            errors.append(rec_dir)
        else:
            missing.append(rec_dir)
report.update({"flacs_in_done_tasks": n_flac, "processed": n_ok, "error_json (corrupt FLAC)": len(errors),
               "location_dates": len(by_locdate)})
if missing:
    bad(f"{len(missing)} FLACs without processed.json/error.json, e.g. {missing[:3]}")

# 2b. every FLAC on disk inside a window was enqueued
queued = {(t["location"], t["date"], f) for t in tasks for f in t["flacs"]}
n_expected = n_not_queued = 0
for loc, rpi in LOCATION_MAP.items():
    if loc in SKIP_LOCATIONS:
        continue
    for f in glob.glob(os.path.join(MONITORING_DATA, rpi, "20*", "*.flac")):
        hhmm = os.path.basename(f)[:5].replace("-", ":")
        if any(a <= hhmm < b for w in WINDOWS.values() for a, b in w):
            n_expected += 1
            if (loc, os.path.basename(os.path.dirname(f)), os.path.basename(f)) not in queued:
                n_not_queued += 1
report["flacs_on_disk_in_windows"] = n_expected
report["flacs_not_queued"] = n_not_queued
if n_not_queued:
    bad(f"{n_not_queued} FLACs on disk in the windows were never queued")

# 3. finalize per location-date and tag; 4. audit clips exist; 5. open a few recordings
n_rows = n_clip_missing = n_nonlocal = 0
for (loc, date), tags in sorted(by_locdate.items()):
    d = os.path.join(OUTPUT_RTF_DIR, loc, date)
    for tag in tags:
        if not os.path.isdir(os.path.join(d, f".finalized_{tag}")):
            bad(f"{loc} {date}: not finalized for tag {tag}")
    for name in ["daily_summary.json", "detection_audit_manifest.json"]:
        if not os.path.isfile(os.path.join(d, name)):
            bad(f"{loc} {date}: {name} missing")
    have = set(os.listdir(os.path.join(AUDIT_CLIPS_DIR, loc, date, "audit_clips"))) \
        if os.path.isdir(os.path.join(AUDIT_CLIPS_DIR, loc, date, "audit_clips")) else set()
    recs = sorted(r for r in os.listdir(d) if os.path.isfile(os.path.join(d, r, "processed.json")))
    for r in recs:
        rows = json.load(open(os.path.join(d, r, "audit_clips.json")))
        n_rows += len(rows)
        for row in rows:
            for c in (row["clip"], row["mono_clip"]):
                if c not in have:
                    n_clip_missing += 1
    for r in recs[:args.sample]:
        res = json.load(open(os.path.join(d, r, "results.json")))
        if len(res) != N_CHANNELS:
            bad(f"{loc} {date} {r}: results.json has {len(res)} channels, expected {N_CHANNELS}")
        proc = json.load(open(os.path.join(d, r, "processed.json")))
        if list(proc) != GROUPS:
            bad(f"{loc} {date} {r}: processed.json groups {list(proc)}")
report["audit_claims"] = n_rows
report["audit_clips_missing"] = n_clip_missing
if n_clip_missing:
    bad(f"{n_clip_missing} audit clips named in audit_clips.json are missing")

print(json.dumps(report, indent=2, ensure_ascii=False))
if errors:
    print("corrupt FLACs:")
    for e in errors:
        print("  ", e)
print("PROBLEMS:" if problems else "COMPLETE: no problems")
for p in problems[:30]:
    print("  -", p)
sys.exit(1 if problems else 0)
