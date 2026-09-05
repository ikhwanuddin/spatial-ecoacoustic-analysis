#!/usr/bin/env python
"""How much does the paired difference move when the prototype is swapped?

The draft email to Lorenzo puts an absolute prototype drift next to a paired
effect and admits that is not a fair comparison, because a paired difference
cancels part of any shift common to both terms. This measures the fair version:
recompute method-minus-mono while deliberately scoring every window against the
prototype of a condition it does not belong to, and see how far the paired
number itself travels.

The window pairing, the beams and the mono baseline are identical across runs.
Only the prototype changes, so the spread below is attributable to it alone.
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import (CONDITIONS, audits_dir, bacpipe_embeddings_dir,
                              bacpipe_meta_dir, beam_tag_from_name,
                              noise_group_for_method, resolve_noise_vector)
from spatial_clustering import load_noise_embeddings, l2_normalize
from permutation_null import IncompletePairing, METHODS, BF_METHODS, build_windows, matrix_for


def distance_with_condition(X, flat_meta, method_names, noise_vectors, forced_cond):
    """Cosine distance from the prototype of `forced_cond`, whatever the window is."""
    Xn = l2_normalize(X)
    dist = np.full(len(X), np.nan, dtype=np.float32)
    buckets = {}
    for idx, meta in enumerate(flat_meta):
        method = meta.get("method")
        if method not in method_names:
            continue
        wav = str(meta.get("wav", ""))
        key = (noise_group_for_method(method), beam_tag_from_name(wav, method),
               meta.get("date"))
        buckets.setdefault(key, []).append(idx)
    for (group, beam, date_str), idxs in buckets.items():
        nvec, _ = resolve_noise_vector(noise_vectors, forced_cond, group, beam, date_str)
        if nvec is None:
            continue
        rows = np.asarray(idxs, dtype=int)
        dist[rows] = 1.0 - np.dot(Xn[rows], np.asarray(nvec, dtype=np.float32))
    return dist


def run(data_dir, location, date_str, model):
    src = bacpipe_embeddings_dir(data_dir, location, model)
    meta = bacpipe_meta_dir(location, model)
    _e, X, _y, fm, mn = load_embeddings_from_dir(
        src, methods=METHODS, date_filter=[date_str],
        source_tag=f"bacpipe:{model}", meta_dir=meta)
    if len(X) == 0:
        return {"model": model, "date": date_str, "available": False}
    nv = load_noise_embeddings(meta, expected_dim=X.shape[1])
    if not nv:
        return {"model": model, "date": date_str, "available": False,
                "reason": "tidak ada noise reference"}

    out = {"model": model, "date": date_str, "available": True, "per_method": {}}
    per_cond = {}
    for cond in CONDITIONS:
        d = distance_with_condition(X, fm, mn, nv, cond)
        if not np.isfinite(d).any():
            continue
        win, order = build_windows(fm, d, mn)
        entry = {}
        for method in BF_METHODS + ["sa"]:
            got = matrix_for(win, order, method, min_beams=1)
            if got is None:
                continue
            D = got["D"]
            entry[method] = {"delta_mean": round(float(D.mean()), 5),
                             "delta_best_beam": round(float(D.max(axis=1).mean()), 5),
                             "n_windows": int(D.shape[0]), "n_beams": int(D.shape[1])}
        if entry:
            per_cond[cond] = entry
    del X

    for method in BF_METHODS + ["sa"]:
        vals = [(c, e[method]["delta_mean"]) for c, e in per_cond.items() if method in e]
        best = [(c, e[method]["delta_best_beam"]) for c, e in per_cond.items() if method in e]
        if len(vals) < 2:
            continue
        v = np.array([x for _, x in vals]); b = np.array([x for _, x in best])
        out["per_method"][method] = {
            "delta_mean_by_assumed_condition": {c: x for c, x in vals},
            "delta_mean_spread": round(float(v.max() - v.min()), 5),
            "delta_mean_sd": round(float(v.std()), 5),
            "delta_mean_median": round(float(np.median(v)), 5),
            "spread_over_effect": (round(float((v.max() - v.min()) / abs(np.median(v))), 2)
                                   if abs(np.median(v)) > 1e-9 else None),
            "delta_best_beam_spread": round(float(b.max() - b.min()), 5),
            "sign_flips": bool((v > 0).any() and (v < 0).any()),
        }
    out["conditions_tested"] = sorted(per_cond)
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
        a.out = os.path.join(audits_dir(a.location), "prototype_swap_stability.json")
    res = {"location": a.location, "results": []}
    for d in a.dates:
        for m in a.models:
            try:
                r = run(a.data_dir, a.location, d, m)
            # An incomplete pairing is a protocol violation, not a model that
            # happens to fail. It must stop the run so the missing data gets made.
            except IncompletePairing:
                raise
            except Exception as e:
                r = {"model": m, "date": d, "available": False, "reason": str(e)[:200]}
            res["results"].append(r)
            print(f"[{m} {d}] {'ok' if r.get('available') else r.get('reason','')}", flush=True)
            with open(a.out, "w") as f:
                json.dump(res, f, indent=2)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
