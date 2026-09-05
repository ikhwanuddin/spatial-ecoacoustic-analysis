#!/usr/bin/env python
"""Is a sector better than a single steering direction?

The user's hypothesis: if a bird sits between two steering directions, picking
the single best beam throws away the half of the source that landed in the
neighbour. Aggregating a small group of beams should then beat the single one.

Tested in the score domain, so no audio is re-rendered. The group is chosen
using the previous window and scored on the current one, exactly as the single
beam selector is, so every row below is free of selection bias and directly
comparable. k = 1 reproduces the current method.

Two ways of grouping:
  top-k        the k highest scoring beams of the previous window, any direction
  neighbour-k  the best beam of the previous window plus its k-1 nearest beams
               in the steering grid, which is what a widened beam would cover
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir
from permutation_null import IncompletePairing, METHODS, build_windows, matrix_for
from beam_confidence import load_head, confidence, lagged_index

KS = (1, 2, 3, 5, 8)


def beam_coords(tag):
    """(azimuth, other) from a steering tag. SPIR carries distance, LabIR elevation."""
    m = re.match(r"SPIR\d?\((\d+)m_(\d+)", tag)
    if m:
        return float(m.group(2)), float(m.group(1))
    m = re.match(r"LabIR\(S(\d+)_(\d+)\)", tag)
    if m:
        return float(m.group(2)), float(m.group(1))
    return None


def neighbour_table(beams, k_max):
    """For each beam, its nearest beams in the steering grid, itself first."""
    xy = [beam_coords(b) for b in beams]
    n = len(beams)
    if any(c is None for c in xy):
        return None
    az = np.array([c[0] for c in xy]); ot = np.array([c[1] for c in xy])
    # azimuth is periodic, the other axis is scaled to a comparable spread
    spread = max(ot.max() - ot.min(), 1.0)
    order = np.empty((n, k_max), dtype=np.int64)
    for i in range(n):
        daz = np.abs(az - az[i]); daz = np.minimum(daz, 360.0 - daz) / 180.0
        dot = np.abs(ot - ot[i]) / spread
        dist = np.sqrt(daz ** 2 + dot ** 2)
        # SPIR1 and SPIR2 overlap at 180 degrees for several ranges, so a beam
        # can tie with itself at distance zero. Force the beam itself first.
        dist[i] = -1.0
        order[i] = np.argsort(dist)[:k_max]
    return order


def run(data_dir, location, date_str, model, method="bf_SPIR", seed=0):
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
    win, order_map = build_windows(fm, conf, mn)
    got = matrix_for(win, order_map, method)
    sa = matrix_for(win, order_map, "sa", min_beams=1)
    if got is None:
        return {"model": model, "date": date_str, "available": False,
                "reason": f"{method} tidak terpasang"}
    D, keys, beams = got["D"], got["keys"], got["beams"]
    Db = D - D.mean(axis=0, keepdims=True)
    prev = lagged_index(keys, 1)
    ok = prev >= 0
    rows = np.arange(len(keys))
    nb = neighbour_table(beams, max(KS))

    out = {"model": model, "date": date_str, "method": method, "available": True,
           "n_windows": int(ok.sum()), "n_beams": len(beams), "topk": {}, "neighbour": {}}

    if sa is not None:
        sa_idx = {k: i for i, k in enumerate(sa["keys"])}
        aligned = np.array([sa_idx.get(k, -1) for k in keys])
        good = ok & (aligned >= 0)
        out["sa_median"] = round(float(np.median(sa["D"][aligned[good], 0])), 5)

    ranks = np.argsort(-Db, axis=1)          # best first, per window
    for k in KS:
        pick = np.zeros((len(keys), k), dtype=np.int64)
        pick[ok] = ranks[prev[ok], :k]        # chosen by the previous window
        v = np.take_along_axis(D, pick, axis=1).mean(axis=1)[ok]
        out["topk"][str(k)] = {"median": round(float(np.median(v)), 5),
                               "mean": round(float(v.mean()), 5),
                               "win_pct": round(100.0 * float((v > 0).mean()), 1)}
        if nb is not None:
            best = np.zeros(len(keys), dtype=np.int64)
            best[ok] = ranks[prev[ok], 0]
            grp = nb[best][:, :k]
            v2 = np.take_along_axis(D, grp, axis=1).mean(axis=1)[ok]
            out["neighbour"][str(k)] = {"median": round(float(np.median(v2)), 5),
                                        "mean": round(float(v2.mean()), 5),
                                        "win_pct": round(100.0 * float((v2 > 0).mean()), 1)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("ANALYSIS_OUTPUT", "."))
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["birdnet", "avesecho_passt"])
    ap.add_argument("--methods", nargs="+", default=["bf_SPIR", "bf_LabIR"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(audits_dir(a.location), "sector_aggregation.json")
    res = {"location": a.location, "ks": list(KS), "results": []}
    for d in a.dates:
        for mo in a.models:
            for me in a.methods:
                try:
                    r = run(a.data_dir, a.location, d, mo, me)
                # An incomplete pairing is a protocol violation, not a model that
                # happens to fail. It must stop the run so the missing data gets made.
                except IncompletePairing:
                    raise
                except Exception as e:
                    r = {"model": mo, "date": d, "method": me,
                         "available": False, "reason": str(e)[:200]}
                res["results"].append(r)
                print(f"[{mo} {d} {me}] {'ok' if r.get('available') else r.get('reason','')}",
                      flush=True)
                with open(a.out, "w") as f:
                    json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
