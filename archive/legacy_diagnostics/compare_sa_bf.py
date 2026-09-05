#!/usr/bin/env python
"""Per-window distribution of signal averaging against beamforming.

Both are scored as detector confidence minus the confidence of mono on the
same window. The beamforming column uses the unbiased selector: the beam is
chosen by the previous window inside the same recording.

The question this answers is whether signal averaging really is the better
method or whether its mean is carried by a small number of extreme windows.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import (audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir)
from permutation_null import METHODS, build_windows, matrix_for
from beam_confidence import load_head, confidence, lagged_index

Q = (1, 5, 25, 50, 75, 95, 99)


def describe(v, label):
    q = np.percentile(v, Q)
    tail_hi = v[v >= np.percentile(v, 95)].sum() / v.sum() * 100 if v.sum() else float("nan")
    trimmed = v[(v > np.percentile(v, 5)) & (v < np.percentile(v, 95))].mean()
    return {
        "label": label, "n": int(len(v)),
        "mean": round(float(v.mean()), 5),
        "median": round(float(np.median(v)), 5),
        "trimmed_mean_5_95": round(float(trimmed), 5),
        "pct_above_zero": round(100.0 * float((v > 0).mean()), 1),
        "quantiles": {str(k): round(float(x), 5) for k, x in zip(Q, q)},
        "share_of_mean_from_top5pct": round(float(tail_hi), 1),
        "sd": round(float(v.std()), 5),
    }


def run(data_dir, location, date_str, model, seed=0):
    src = bacpipe_embeddings_dir(data_dir, location, model)
    meta = bacpipe_meta_dir(location, model)
    _e, X, _y, fm, mn = load_embeddings_from_dir(
        src, methods=METHODS, date_filter=[date_str],
        source_tag=f"bacpipe:{model}", meta_dir=meta)
    if len(X) == 0:
        return {"model": model, "date": date_str, "available": False}
    head = load_head(os.path.dirname(os.path.abspath(__file__)), model)
    conf, _top, _a, _b = confidence(X, head, seed)
    del X
    win, order = build_windows(fm, conf, mn)

    spir = matrix_for(win, order, "bf_SPIR")
    sa = matrix_for(win, order, "sa", min_beams=1)
    if spir is None or sa is None:
        return {"model": model, "date": date_str, "available": False,
                "reason": "bf_SPIR atau sa tidak terpasang"}

    # keep only the windows both methods have, in the same order
    sa_idx = {k: i for i, k in enumerate(sa["keys"])}
    keep = [(i, sa_idx[k]) for i, k in enumerate(spir["keys"]) if k in sa_idx]
    if len(keep) < 50:
        return {"model": model, "date": date_str, "available": False,
                "reason": f"hanya {len(keep)} window bersama"}
    si = np.array([a for a, _ in keep]); ai = np.array([b for _, b in keep])
    keys = [spir["keys"][i] for i in si]
    D = spir["D"][si]                       # window x beam, lift vs mono
    sa_lift = sa["D"][ai, 0]

    Db = D - D.mean(axis=0, keepdims=True)  # per-beam nightly mean removed
    prev = lagged_index(keys, 1)
    ok = prev >= 0
    theta = np.zeros(len(keys), dtype=np.int64)
    theta[ok] = Db[prev[ok]].argmax(axis=1)
    bf_lift = D[np.arange(len(keys)), theta]

    v_sa, v_bf = sa_lift[ok], bf_lift[ok]
    diff = v_bf - v_sa
    out = {
        "model": model, "date": date_str, "available": True,
        "n_windows": int(ok.sum()), "n_recordings": int(len({k[0] for k in keys})),
        "sa": describe(v_sa, "sa"),
        "bf_spir_sel_prev": describe(v_bf, "bf_SPIR dipilih dari t-1"),
        "paired_bf_minus_sa": {
            "mean": round(float(diff.mean()), 5),
            "median": round(float(np.median(diff)), 5),
            "pct_bf_wins": round(100.0 * float((diff > 0).mean()), 1),
            "quantiles": {str(k): round(float(x), 5)
                          for k, x in zip(Q, np.percentile(diff, Q))},
        },
        "oracle_note": {
            "bf_oracle_mean": round(float(D.max(axis=1).mean()), 5),
            "n_beams": int(D.shape[1]),
        },
    }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("ANALYSIS_OUTPUT", "."))
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["birdnet", "avesecho_passt"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(audits_dir(a.location), "sa_vs_bf_spir_distribution.json")

    res = {"location": a.location, "results": []}
    for d in a.dates:
        for m in a.models:
            r = run(a.data_dir, a.location, d, m)
            res["results"].append(r)
            print(f"[{m} {d}] {'ok' if r.get('available') else r.get('reason','kosong')}",
                  flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
