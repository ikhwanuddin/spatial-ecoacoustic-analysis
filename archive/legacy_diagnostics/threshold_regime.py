#!/usr/bin/env python
"""Where does the advantage change hands, and how big is it on each side?

Windows are ordered by their mono confidence, which stands in for how strong
the source already is without any array processing. The crossing point is the
mono level at which the paired median of beamforming minus signal averaging
passes through zero. Below it beamforming should be reported; above it signal
averaging should be.

Also tests the user's hypothesis about the steering grid: if a source falls
between two steering directions, the top two beams should score almost the
same. The ratio of the second best beam to the best is reported for the
windows beamforming wins and the windows it loses.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir
from permutation_null import IncompletePairing, METHODS, build_windows, matrix_for
from beam_confidence import load_head, confidence, lagged_index

N_BINS = 10


def crossing(centres, values):
    """First mono level where the paired median stops favouring beamforming."""
    for i in range(len(values) - 1):
        if values[i] > 0 >= values[i + 1]:
            x0, x1, y0, y1 = centres[i], centres[i + 1], values[i], values[i + 1]
            if y0 == y1:
                return float(x1)
            return float(x0 + (x1 - x0) * y0 / (y0 - y1))
    if values and values[0] <= 0:
        return float("-inf")      # signal averaging wins everywhere
    return float("inf")           # beamforming wins everywhere


def run(data_dir, location, date_str, model, seed=0):
    src = bacpipe_embeddings_dir(data_dir, location, model)
    meta = bacpipe_meta_dir(location, model)
    _e, X, _y, fm, mn = load_embeddings_from_dir(
        src, methods=METHODS, date_filter=[date_str],
        source_tag=f"bacpipe:{model}", meta_dir=meta)
    if len(X) == 0:
        return {"model": model, "date": date_str, "available": False}
    head = load_head(os.path.dirname(os.path.abspath(__file__)), model)
    conf, _t, _a, _b = confidence(X, head, seed)
    del X
    win, order = build_windows(fm, conf, mn)
    spir = matrix_for(win, order, "bf_SPIR")
    sa = matrix_for(win, order, "sa", min_beams=1)
    if spir is None or sa is None:
        return {"model": model, "date": date_str, "available": False,
                "reason": "bf_SPIR atau sa tidak terpasang"}
    sa_idx = {k: i for i, k in enumerate(sa["keys"])}
    pr = [(i, sa_idx[k]) for i, k in enumerate(spir["keys"]) if k in sa_idx]
    si = np.array([a for a, _ in pr]); ai = np.array([b for _, b in pr])
    keys = [spir["keys"][i] for i in si]
    D = spir["D"][si]
    sa_lift = sa["D"][ai, 0]
    mono = np.array([next(iter(win[k]["mono"].values())) for k in keys])

    Db = D - D.mean(axis=0, keepdims=True)
    prev = lagged_index(keys, 1)
    ok = prev >= 0
    rows = np.arange(len(keys))
    theta = np.zeros(len(keys), dtype=np.int64)
    theta[ok] = Db[prev[ok]].argmax(axis=1)
    bf_lift = D[rows, theta]
    rng = np.random.default_rng(seed + 7)
    rand_lift = D[rows, rng.integers(0, D.shape[1], len(keys))]

    # how sharp is the winning direction: best beam over runner up
    srt = np.sort(Db, axis=1)
    top1, top2 = srt[:, -1], srt[:, -2]
    sharp = np.where(np.abs(top1) > 1e-12, (top1 - top2) / np.abs(top1), np.nan)

    m, s, b, r, sh = mono[ok], sa_lift[ok], bf_lift[ok], rand_lift[ok], sharp[ok]
    diff = b - s
    edges = np.percentile(m, np.linspace(0, 100, N_BINS + 1)); edges[-1] += 1e-9
    centres, meds, bins = [], [], []
    for i in range(N_BINS):
        sel = (m >= edges[i]) & (m < edges[i + 1])
        if sel.sum() < 20:
            continue
        centres.append(float(m[sel].mean())); meds.append(float(np.median(diff[sel])))
        bins.append({"bin": i + 1, "n": int(sel.sum()),
                     "mono_mean": round(float(m[sel].mean()), 4),
                     "diff_median": round(float(np.median(diff[sel])), 5),
                     "bf_wins_pct": round(100.0 * float((diff[sel] > 0).mean()), 1)})
    thr = crossing(centres, meds)

    def block(sel, label):
        if sel.sum() < 20:
            return {"label": label, "n": int(sel.sum())}
        return {"label": label, "n": int(sel.sum()),
                "share_of_windows_pct": round(100.0 * float(sel.mean()), 1),
                "bf_vs_mono_median": round(float(np.median(b[sel])), 5),
                "sa_vs_mono_median": round(float(np.median(s[sel])), 5),
                "bf_minus_sa_median": round(float(np.median(diff[sel])), 5),
                "bf_wins_pct": round(100.0 * float((diff[sel] > 0).mean()), 1),
                "bf_vs_random_median": round(float(np.median(b[sel] - r[sel])), 5),
                "sharpness_median": round(float(np.nanmedian(sh[sel])), 4)}

    below = m < thr if np.isfinite(thr) else (np.ones_like(m, bool) if thr > 0
                                             else np.zeros_like(m, bool))
    return {"model": model, "date": date_str, "available": True,
            "n_windows": int(ok.sum()),
            "mono_median": round(float(np.median(m)), 4),
            "crossing_mono_level": (round(thr, 4) if np.isfinite(thr) else
                                    ("beamforming menang di semua level" if thr > 0
                                     else "sa menang di semua level")),
            "weak_regime": block(below, "di bawah ambang"),
            "strong_regime": block(~below, "di atas ambang"),
            "sharpness_when_bf_wins": round(float(np.nanmedian(sh[diff > 0])), 4),
            "sharpness_when_bf_loses": round(float(np.nanmedian(sh[diff <= 0])), 4),
            "bins": bins}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("ANALYSIS_OUTPUT", "."))
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["birdnet", "avesecho_passt"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(audits_dir(a.location), "sa_vs_bf_threshold.json")
    res = {"location": a.location, "n_bins": N_BINS, "results": []}
    for d in a.dates:
        for mo in a.models:
            try:
                r = run(a.data_dir, a.location, d, mo)
            # An incomplete pairing is a protocol violation, not a model that
            # happens to fail. It must stop the run so the missing data gets made.
            except IncompletePairing:
                raise
            except Exception as e:
                r = {"model": mo, "date": d, "available": False, "reason": str(e)[:200]}
            res["results"].append(r)
            print(f"[{mo} {d}] {'ok' if r.get('available') else r.get('reason','kosong')}",
                  flush=True)
            with open(a.out, "w") as f:      # write as we go, the sweep is long
                json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
