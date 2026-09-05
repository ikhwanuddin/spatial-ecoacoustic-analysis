#!/usr/bin/env python
"""Where does signal averaging win, and where does beamforming win?

Every window carries a mono confidence, which stands in for how strong the
source already is on the omnidirectional channel. Windows are grouped into
quintiles of that level, and inside each group signal averaging is compared
with the unbiased beamforming selector.

If beamforming wins in the low groups and signal averaging in the high ones,
the useful statement is not "one method is better" but "each method belongs to
a different part of the range".
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir
from permutation_null import METHODS, build_windows, matrix_for
from beam_confidence import load_head, confidence, lagged_index

N_BINS = 5


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(np.float64)
    ry = np.argsort(np.argsort(y)).astype(np.float64)
    return float(np.corrcoef(rx, ry)[0, 1])


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
    pairs = [(i, sa_idx[k]) for i, k in enumerate(spir["keys"]) if k in sa_idx]
    si = np.array([a for a, _ in pairs]); ai = np.array([b for _, b in pairs])
    keys = [spir["keys"][i] for i in si]
    D = spir["D"][si]
    sa_lift = sa["D"][ai, 0]
    mono = np.array([next(iter(win[k]["mono"].values())) for k in keys])

    Db = D - D.mean(axis=0, keepdims=True)
    prev = lagged_index(keys, 1)
    ok = prev >= 0
    theta = np.zeros(len(keys), dtype=np.int64)
    theta[ok] = Db[prev[ok]].argmax(axis=1)
    bf_lift = D[np.arange(len(keys)), theta]

    m, s, b = mono[ok], sa_lift[ok], bf_lift[ok]
    diff = b - s
    edges = np.percentile(m, np.linspace(0, 100, N_BINS + 1))
    edges[-1] += 1e-9
    bins = []
    for i in range(N_BINS):
        sel = (m >= edges[i]) & (m < edges[i + 1])
        if sel.sum() < 20:
            continue
        bins.append({
            "bin": i + 1, "n": int(sel.sum()),
            "mono_range": [round(float(edges[i]), 4), round(float(edges[i + 1]), 4)],
            "mono_mean": round(float(m[sel].mean()), 4),
            "sa_median": round(float(np.median(s[sel])), 5),
            "bf_median": round(float(np.median(b[sel])), 5),
            "diff_median": round(float(np.median(diff[sel])), 5),
            "bf_wins_pct": round(100.0 * float((diff[sel] > 0).mean()), 1),
        })
    return {
        "model": model, "date": date_str, "available": True,
        "n_windows": int(ok.sum()),
        "spearman_mono_vs_diff": round(spearman(m, diff), 4),
        "spearman_mono_vs_sa_lift": round(spearman(m, s), 4),
        "spearman_mono_vs_bf_lift": round(spearman(m, b), 4),
        "bins": bins,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("ANALYSIS_OUTPUT", "."))
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["birdnet", "avesecho_passt"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(audits_dir(a.location), "sa_vs_bf_by_mono_level.json")
    res = {"location": a.location, "n_bins": N_BINS, "results": []}
    for d in a.dates:
        for mo in a.models:
            r = run(a.data_dir, a.location, d, mo)
            res["results"].append(r)
            print(f"[{mo} {d}] {'ok' if r.get('available') else r.get('reason','kosong')}",
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
