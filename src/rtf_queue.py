"""
RTF rerun queue: tasks of up to 5 FLACs from one location-date, shared by every GPU node.

  python src/rtf_queue.py enqueue TAG HH:MM-HH:MM[,...] [LOC DATE]
                                             # one task per 5 FLACs whose start time is in the windows
  python src/rtf_queue.py hold               # pending -> hold (kept, not processed)
  python src/rtf_queue.py worker             # claim tasks until the queue is empty (one per GPU)
  python src/rtf_queue.py status             # short progress report
  python src/rtf_queue.py finalize LOC DATE  # daily summary + audit manifest (workers do this themselves)
  python src/rtf_queue.py sweep              # finalize every (date, loc, tag) whose tasks are all done
  python src/rtf_queue.py requeue            # active tasks of a worker silent > 20 min -> pending

Files under output_rtf/:
  _queue/{pending,active,done,failed,hold}/<date>_<loc>_<tag>_<k>.json   (claim = atomic rename)
  _status/<worker>.json                                        (heartbeat, written after every recording)
  <loc>/<date>/<rec>/{results,processed,paired_detections,threshold_summary,audit_clips}.json
  <loc>/<date>/{daily_summary,detection_audit_manifest}.{json,md}, corrupted_files.json
Clips: AUDIT_CLIPS_DIR/<loc>/<date>/audit_clips/ (ephemeral)
"""

import os
import sys
import glob
import json
import time
import socket
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (OUTPUT_RTF_DIR, MONITORING_DATA, LOCATION_MAP, SKIP_LOCATIONS, DEFAULT_THRESHOLDS,
                    AUDIT_CLIPS_DIR, EPHEM_DIR)

TASK_SIZE = 5
Q = os.path.join(OUTPUT_RTF_DIR, "_queue")
STATUS = os.path.join(OUTPUT_RTF_DIR, "_status")
STATES = ["pending", "active", "done", "failed", "hold"]


def task_files(state, prefix=""):
    return sorted(glob.glob(os.path.join(Q, state, f"{prefix}*.json")))


def enqueue(tag, windows, only_loc=None, only_date=None):
    """windows: 'HH:MM-HH:MM,...' on the FLAC start time (file name HH-MM-SS_...)."""
    spans = [w.split("-") for w in windows.split(",")]
    for s in STATES:
        os.makedirs(os.path.join(Q, s), exist_ok=True)
    known = {os.path.basename(p).split(".")[0] for s in STATES for p in task_files(s)}
    n_new = 0
    for loc, rpi in LOCATION_MAP.items():
        if loc in SKIP_LOCATIONS or (only_loc and loc != only_loc):
            continue
        for date_dir in sorted(glob.glob(os.path.join(MONITORING_DATA, rpi, "20*"))):
            date = os.path.basename(date_dir)
            if only_date and date != only_date:
                continue
            flacs = sorted(os.path.basename(f) for f in glob.glob(os.path.join(date_dir, "*.flac")))
            flacs = [f for f in flacs if any(a <= f[:5].replace("-", ":") < b for a, b in spans)]
            chunks = [flacs[i:i + TASK_SIZE] for i in range(0, len(flacs), TASK_SIZE)]
            for k, chunk in enumerate(chunks):
                name = f"{date}_{loc}_{tag}_{k:03d}"
                if name in known:
                    continue
                with open(os.path.join(Q, "pending", name + ".json"), "w") as f:
                    json.dump({"location": loc, "date": date, "tag": tag, "flacs": chunk,
                               "n_tasks_date": len(chunks)}, f)
                n_new += 1
    print(f"enqueued {n_new} new tasks ({tag}: {windows})")


def hold():
    os.makedirs(os.path.join(Q, "hold"), exist_ok=True)
    for p in task_files("pending"):
        os.rename(p, os.path.join(Q, "hold", os.path.basename(p)))
    print(f"hold: {len(task_files('hold'))} tasks")


def claim(worker):
    for p in task_files("pending"):
        dst = os.path.join(Q, "active", os.path.basename(p)[:-5] + f".{worker}.json")
        try:
            os.rename(p, dst)          # atomic: only one worker wins
            return dst
        except FileNotFoundError:
            continue
    return None


def heartbeat(worker, **info):
    os.makedirs(STATUS, exist_ok=True)
    info.update({"worker": worker, "time": time.time()})
    tmp = os.path.join(STATUS, f".{worker}.tmp")
    with open(tmp, "w") as f:
        json.dump(info, f)
    os.replace(tmp, os.path.join(STATUS, f"{worker}.json"))


def finalize(loc, date):
    from pair_and_recap import evaluate_threshold_counts, format_markdown_table

    out = os.path.join(OUTPUT_RTF_DIR, loc, date)
    recs = sorted(d for d in os.listdir(out) if os.path.isfile(os.path.join(out, d, "processed.json")))
    daily = {}
    for rec in recs:
        processed = json.load(open(os.path.join(out, rec, "processed.json")))
        for m, sp_dict in processed.items():
            for sp, info in sp_dict.items():
                daily.setdefault(m, {}).setdefault(sp, {"conf_list": []})["conf_list"].extend(info["conf_list"])
    summary = evaluate_threshold_counts(daily, DEFAULT_THRESHOLDS)
    json.dump(summary, open(os.path.join(out, "daily_summary.json"), "w"), indent=4)

    corrupted = []
    for p in glob.glob(os.path.join(out, "*", "error.json")):
        corrupted.append(json.load(open(p)))
    with open(os.path.join(out, "daily_summary.md"), "w") as f:
        f.write(f"# Daily Detection Summary (RTF): {loc} ({date})\n\n")
        f.write(f"Successfully processed : {len(recs)}\nCorrupted & skipped   : {len(corrupted)}\n\n")
        f.write(format_markdown_table(summary, DEFAULT_THRESHOLDS) + "\n")
    if corrupted:
        json.dump(corrupted, open(os.path.join(out, "corrupted_files.json"), "w"), indent=4)

    # audit manifest: one row per (group, species, window) claim, pointing at the shared clips
    clips = os.path.join(AUDIT_CLIPS_DIR, loc, date, "audit_clips")
    mac = f"/Volumes/ri322/ephemeral/{os.path.relpath(clips, EPHEM_DIR)}"
    items = []
    for rec in recs:
        for r in json.load(open(os.path.join(out, rec, "audit_clips.json"))):
            r["audio_paths"] = {"cx3_wav": f"{clips}/{r['clip']}", "cx3_mono_wav": f"{clips}/{r['mono_clip']}",
                                "mac_wav": f"{mac}/{r['clip']}", "mac_mono_wav": f"{mac}/{r['mono_clip']}"}
            items.append(r)
    json.dump(items, open(os.path.join(out, "detection_audit_manifest.json"), "w"), indent=4, ensure_ascii=False)
    with open(os.path.join(out, "detection_audit_manifest.md"), "w") as f:
        f.write(f"# Detection Audit Manifest (RTF): {loc} ({date})\n\n"
                f"{len(items)} claims, clips in `{clips}`\n\n"
                "| rec | window (s) | group | species | channel | conf | mono conf |\n|---|---|---|---|---|---|---|\n")
        for r in items:
            f.write(f"| {r['rec']} | {r['window_seconds'][0]:.0f}-{r['window_seconds'][1]:.0f} | {r['group']} | "
                    f"{r['species']} | {r['channel']} | {r['conf']:.3f} | {r['mono_conf'] or '-'} |\n")
    print(f"finalized {loc} {date}: {len(recs)} recordings, {len(items)} audit claims")


def worker():
    import sea_gpu
    name = f"{socket.gethostname()}_{os.environ.get('PBS_JOBID', 'nojob').split('.')[0]}"
    n_recs, audio_s, busy_s = 0, 0.0, 0.0
    while True:
        task = claim(name)
        if task is None:
            heartbeat(name, state="idle (queue empty)", n_recs=n_recs, audio_h=round(audio_s / 3600, 2))
            print("queue empty, exiting")
            return
        t = json.load(open(task))
        loc, date = t["location"], t["date"]
        heartbeat(name, state="busy", task=os.path.basename(task), rec="(claimed)", n_recs=n_recs,
                  audio_h=round(audio_s / 3600, 2), x_realtime=round(audio_s / max(busy_s, 1e-9), 1))
        try:
            for flac in t["flacs"]:
                rec = flac[:-5]
                out_rec = os.path.join(OUTPUT_RTF_DIR, loc, date, rec)
                if os.path.isfile(os.path.join(out_rec, "processed.json")):
                    continue
                t0 = time.time()
                r = sea_gpu.process_recording(os.path.join(MONITORING_DATA, LOCATION_MAP[loc], date, flac),
                                              loc, date, out_rec)
                if "error" in r:
                    json.dump(r["error"], open(os.path.join(out_rec, "error.json"), "w"), indent=4)
                else:
                    n_recs += 1
                    audio_s += r["dur_s"]
                busy_s += time.time() - t0
                heartbeat(name, state="busy", task=os.path.basename(task), rec=rec, last=r, n_recs=n_recs,
                          audio_h=round(audio_s / 3600, 2), x_realtime=round(audio_s / max(busy_s, 1e-9), 1))
            os.rename(task, os.path.join(Q, "done", os.path.basename(task).split(".")[0] + ".json"))
        except Exception:
            err = traceback.format_exc()
            print(err)
            with open(os.path.join(Q, "failed", os.path.basename(task).split(".")[0] + ".json"), "w") as f:
                json.dump({**t, "worker": name, "error": err}, f)
            os.remove(task)
            continue

        try_finalize(loc, date, t.get("tag"))


def try_finalize(loc, date, tag):
    """Finalize (date, loc) once all tasks of this tag are done; mkdir lock per tag is atomic."""
    if tag is None:
        return
    prefix = f"{date}_{loc}_{tag}_"
    done = task_files("done", prefix)
    if task_files("pending", prefix) or task_files("active", prefix) or not done:
        return
    if len(done) != json.load(open(done[0]))["n_tasks_date"]:
        return
    lock = os.path.join(OUTPUT_RTF_DIR, loc, date, f".finalized_{tag}")
    try:
        os.mkdir(lock)
    except FileExistsError:
        return
    try:
        finalize(loc, date)
    except Exception:
        os.rmdir(lock)          # let a later sweep retry
        raise


def sweep():
    for p in task_files("done"):
        t = json.load(open(p))
        try_finalize(t["location"], t["date"], t.get("tag"))


def status():
    counts = {s: len(task_files(s)) for s in STATES}
    total = sum(counts.values()) - counts["hold"]
    print(f"tasks {counts['done']}/{total} done ({100 * counts['done'] / max(total, 1):.1f}%) | "
          f"active {counts['active']} | pending {counts['pending']} | failed {counts['failed']} | hold {counts['hold']}")
    per = {}
    for s in ["done", "pending", "active"]:
        for p in task_files(s):
            loc = os.path.basename(p).split("_")[1]
            per.setdefault(loc, [0, 0])[1] += 1
            per[loc][0] += s == "done"
    print("per location: " + " | ".join(f"{loc} {d}/{n}" for loc, (d, n) in sorted(per.items())))
    now = time.time()
    for p in sorted(glob.glob(os.path.join(STATUS, "*.json"))):
        h = json.load(open(p))
        age = now - h["time"]
        print(f"  {h['worker']}: {h['state']} {h.get('rec', '')} | recs {h['n_recs']} audio {h['audio_h']} h "
              f"x{h.get('x_realtime', '-')} | {age / 60:.0f} min ago{'  <-- STALE' if age > 900 and h['state'] == 'busy' else ''}")
    for p in task_files("failed")[:3]:
        print("  FAILED", os.path.basename(p), json.load(open(p))["error"].strip().splitlines()[-1][:150])


def requeue():
    now = time.time()
    for p in task_files("active"):
        worker = os.path.basename(p).split(".", 1)[1][:-5]
        hb = os.path.join(STATUS, f"{worker}.json")
        if not os.path.isfile(hb) or now - json.load(open(hb))["time"] > 1200:
            os.rename(p, os.path.join(Q, "pending", os.path.basename(p).split(".")[0] + ".json"))
            print("requeued", os.path.basename(p))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "enqueue":
        enqueue(*sys.argv[2:6])
    elif cmd == "hold":
        hold()
    elif cmd == "sweep":
        sweep()
    elif cmd == "worker":
        worker()
    elif cmd == "status":
        status()
    elif cmd == "requeue":
        requeue()
    elif cmd == "finalize":
        finalize(sys.argv[2], sys.argv[3])
