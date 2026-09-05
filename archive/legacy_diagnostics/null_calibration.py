#!/usr/bin/env python
"""The winner's curse floor, measured on windows that contain no bird.

Section 4a of the draft email promises this. The noise reference clips are
bird-free by construction, and they are cut from the same instants on every
beam, so they form matched windows exactly like the recordings do.

The prototype is built from one half of the clips and the other half is scored
against it, so the two are disjoint and nothing is compared with itself. Taking
the maximum over beams on those windows cannot be finding a bird, so whatever
lift appears is the floor that any reported oracle number sits on.
"""
import argparse, json, os, re, sys
from collections import defaultdict
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_schema import CONDITIONS, audits_dir, bacpipe_meta_dir
from spatial_clustering import l2_normalize


def load_clips(meta_dir):
    """noise_<date>_<cond>_<beam>_embeddings.npy -> per condition, per beam."""
    out = defaultdict(dict)
    if not os.path.isdir(meta_dir):
        return out
    for name in sorted(os.listdir(meta_dir)):
        if not (name.startswith("noise_") and name.endswith("_embeddings.npy")):
            continue
        stem = name[len("noise_"):-len("_embeddings.npy")]
        m = re.match(r"(\d{4}-\d{2}-\d{2})_(" + "|".join(CONDITIONS) + r")_(.+)$", stem)
        if not m:
            continue
        date_str, cond, beam = m.groups()
        try:
            arr = np.load(os.path.join(meta_dir, name), mmap_mode="r")
        except Exception:
            continue
        if arr.ndim != 2 or len(arr) < 4:
            continue
        out[(date_str, cond)][beam] = np.asarray(arr, dtype=np.float32)
    return out


def group_of(beam):
    if beam.startswith("LabIR"):
        return "LabIR"
    if beam.startswith("SPIR"):
        return "SPIR"
    if beam.startswith("SBF"):
        return "SBF"
    return beam


def run(location, model):
    meta_dir = bacpipe_meta_dir(location, model)
    clips = load_clips(meta_dir)
    if not clips:
        return {"model": model, "available": False, "reason": "tidak ada noise embedding"}
    out = {"model": model, "available": True, "scopes": []}
    for (date_str, cond), beams in sorted(clips.items()):
        if "mono" not in beams:
            continue
        n = min(len(v) for v in beams.values())
        if n < 6:
            continue
        half = n // 2
        for group in ("LabIR", "SPIR", "SBF"):
            members = sorted(b for b in beams if group_of(b) == group)
            if len(members) < 3:
                continue
            # prototype from the first half, scoring on the second half
            def dist(beam, rows):
                v = l2_normalize(beams[beam][rows])
                proto = l2_normalize(beams[beam][:half].mean(axis=0, keepdims=True))[0]
                return 1.0 - v @ proto
            rows = np.arange(half, n)
            mono = dist("mono", rows)
            M = np.stack([dist(b, rows) for b in members], axis=1)   # window x beam
            D = M - mono[:, None]
            rng = np.random.default_rng(0)
            out["scopes"].append({
                "date": date_str, "condition": cond, "group": group,
                "n_windows": int(len(rows)), "n_beams": len(members),
                "median_beam_lift": round(float(np.median(D)), 5),
                "mean_beam_lift": round(float(D.mean()), 5),
                "oracle_max_lift": round(float(D.max(axis=1).mean()), 5),
                "random_beam_lift": round(float(
                    D[np.arange(len(rows)), rng.integers(0, D.shape[1], len(rows))].mean()), 5),
                "oracle_minus_median": round(float(D.max(axis=1).mean() - np.median(D)), 5),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--models", nargs="+",
                    default=["birdnet", "avesecho_passt", "birdnet_v3", "perch_bird",
                             "perch_v2", "vggish", "biolingual", "protoclr"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(audits_dir(a.location), "null_calibration.json")
    res = {"location": a.location, "results": []}
    for m in a.models:
        try:
            r = run(a.location, m)
        except Exception as e:
            r = {"model": m, "available": False, "reason": str(e)[:200]}
        res["results"].append(r)
        print(f"[{m}] {'ok, %d scope' % len(r.get('scopes', [])) if r.get('available') else r.get('reason','')}",
              flush=True)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
