"""Measure the refinement budget of the SBF coarse grid.

Reads the 37 direction coarse scores that already exist, finds which windows
carry a peak that is confirmed against its neighbours, and counts how many new
10 degree directions the second beamforming pass would have to render. Nothing
is rendered here; this only sizes the job.
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ANALYSIS_OUTPUT
from embedding_io import load_embeddings_from_dir
from embedding_schema import audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir
from beam_confidence import load_head, confidence
from permutation_null import build_windows

# the coarse grid: six elevation rings of six azimuths, plus the zenith
RING = [1, 3, 5, 7, 9, 11]
ELEV = {1: -45, 3: -20, 5: 0, 7: 20, 9: 45, 11: 75, 12: 90}
AZ = [0, 60, 120, 180, 240, 300]
# speakers measured between the coarse rings, for refining in elevation
BETWEEN = {1: [2], 3: [2, 4], 5: [4, 6], 7: [6, 8], 9: [8, 10], 11: [10, 12], 12: []}
TAG = re.compile(r"SBF\(S(\d\d)_(\d\d\d)\)")


def parse(beam):
    m = TAG.search(beam)
    return (int(m.group(1)), int(m.group(2))) if m else None


def neighbours(d):
    s, a = d
    if s == 12:
        return [(11, x) for x in AZ]
    out = [(s, (a + 60) % 360), (s, (a - 60) % 360)]
    i = RING.index(s)
    if i > 0:
        out.append((RING[i - 1], a))
    if i < len(RING) - 1:
        out.append((RING[i + 1], a))
    else:
        out.append((12, 0))
    return out


def refine_set(d):
    """The new directions a second pass would need around one confirmed peak."""
    s, a = d
    if s == 12:
        return set()                      # a zenith peak has no azimuth to refine
    out = {(s, (a + 10 * j) % 360) for j in range(-5, 6) if j != 0}
    out |= {(t, a) for t in BETWEEN[s]}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--location", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--model", default="birdnet")
    p.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()

    emb_dir = bacpipe_embeddings_dir(ANALYSIS_OUTPUT, a.location, a.model)
    _emb, X, _y, flat, _names = load_embeddings_from_dir(
        emb_dir, methods=["bf_SBF"], date_filter=[a.date],
        source_tag="bacpipe:" + a.model,
        meta_dir=bacpipe_meta_dir(a.location, a.model))
    if len(X) == 0:
        print("no bf_SBF embeddings for", a.location, a.date, a.model)
        return 1
    C, _, _, _ = confidence(X, load_head(a.repo, a.model), a.seed)

    # reuse the shared keying so the window identity matches every other tool
    raw, _order = build_windows(flat, C, ["bf_SBF"])
    win = {}
    for key, by_method in raw.items():
        got = {}
        for tag, c in by_method.get("bf_SBF", {}).items():
            d = parse(tag or "")
            if d is not None:
                got[d] = c
        if got:
            win[key] = got
    full = {k: v for k, v in win.items() if len(v) == 37}
    print("windows", len(win), "complete at 37 directions", len(full))
    if not full:
        return 1

    report = {"location": a.location, "date": a.date, "model": a.model,
              "windows": len(full), "margins": {}}
    for kmar in (0.0, 0.5, 1.0, 2.0):
        n_any = 0
        n_peaks = []
        per_file_union = {}
        for key, v in full.items():
            vals = np.array(list(v.values()))
            mad = float(np.median(np.abs(vals - np.median(vals)))) or 1e-9
            peaks = []
            for d, c in v.items():
                nb = max(v[x] for x in neighbours(d) if x in v)
                if c - nb > kmar * mad:
                    peaks.append((c, d))
            peaks.sort(reverse=True)
            n_peaks.append(len(peaks))
            if peaks:
                n_any += 1
            u = per_file_union.setdefault(key[0], {1: set(), 2: set(), 3: set()})
            for K in (1, 2, 3):
                for _, d in peaks[:K]:
                    u[K] |= refine_set(d)
        arr = np.array(n_peaks)
        unions = {K: [len(u[K]) for u in per_file_union.values()] for K in (1, 2, 3)}
        entry = {
            "windows_with_a_confirmed_peak_pct": round(100.0 * n_any / len(full), 1),
            "peaks_per_window_mean": round(float(arr.mean()), 2),
            "peaks_per_window_max": int(arr.max()),
            "new_directions_per_file": {
                str(K): {"median": int(np.median(unions[K])),
                         "max": int(np.max(unions[K])),
                         "mean": round(float(np.mean(unions[K])), 1)}
                for K in (1, 2, 3)},
            "files": len(per_file_union),
        }
        report["margins"][str(kmar)] = entry
        nd = entry["new_directions_per_file"]
        print("margin", kmar,
              "|", entry["windows_with_a_confirmed_peak_pct"], "% windows have a peak",
              "|", entry["peaks_per_window_mean"], "peaks/window",
              "| new dirs/file median K=1", nd["1"]["median"],
              "K=2", nd["2"]["median"], "K=3", nd["3"]["median"],
              "| max K=2", nd["2"]["max"])

    out = a.out or os.path.join(audits_dir(a.location),
                                a.date + "_sbf_peaks_" + a.model + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
