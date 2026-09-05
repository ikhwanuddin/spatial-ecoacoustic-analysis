#!/usr/bin/env python
"""Label-free null tests for beam selection.

Three diagnostics. None of them needs a bird label, and all of them run on the
embeddings that are already cached, so nothing has to be re-embedded.

  A. beam-column permutation
     Inside one window the beam deltas are centred, then each beam column is
     shuffled across windows. That keeps every beam's own distribution but
     destroys which directions happen to co-occur inside a single window. Take
     the maximum again. What the observed maximum has over the permuted one is
     the part the array actually found rather than the part the max operator
     produces on its own.

  B. bias-corrected lift
     Each beam's mean delta over the whole night is subtracted before the
     maximum is taken, so a direction that simply sits further from its own
     noise reference cannot win by default.

  C. selected-direction persistence
     How often theta*(t+1) equals theta*(t), against the rate expected if the
     winner were drawn at random from the same marginal distribution.

Usage:
  python permutation_null.py --date 2026-04-30 --models birdnet_v3 --out out.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from config import ANALYSIS_OUTPUT
from embedding_io import load_embeddings_from_dir
from embedding_schema import (
    audits_dir,
    DEFAULT_METHODS,
    bacpipe_embeddings_dir,
    bacpipe_meta_dir,
    beam_tag_from_name,
    condition_from_wav,
    noise_group_for_method,
    resolve_noise_vector,
)
from spatial_clustering import convert_numpy_types, l2_normalize, load_noise_embeddings

# bf_SBF is the coarse search grid and is not in DEFAULT_METHODS, so it is
# appended here. Dates without it simply report the method as unavailable.
class IncompletePairing(RuntimeError):
    """Raised when a window is missing any stream it must be compared against.

    The analysis protocol requires every window to carry mono and every beam of
    the method under test. A missing stream is a rendering or embedding job that
    did not finish, and the answer is to finish it. Dropping the window silently
    changes which acoustic events the comparison covers; forcing the comparison
    pairs different events. Neither is acceptable, so the run stops here.
    """


METHODS = list(DEFAULT_METHODS) + ["bf_SBF"]
BF_METHODS = ["bf_LabIR", "bf_SPIR", "bf_SBF"]


def point_noise_distance(X, flat_meta, method_names, noise_vectors):
    """Cosine distance of every point from the reference for its own beam."""
    X_norm = l2_normalize(X)
    dist = np.full(len(X), np.nan, dtype=np.float32)
    buckets: Dict[Any, List[int]] = {}
    for idx, meta in enumerate(flat_meta):
        method = meta.get("method")
        if method not in method_names:
            continue
        wav = meta.get("wav", "")
        cond = meta.get("condition") or condition_from_wav(wav)
        buckets.setdefault(
            (cond, noise_group_for_method(method), beam_tag_from_name(wav, method),
             meta.get("date")), []
        ).append(idx)
    for (cond, group, beam, date_str), idxs in buckets.items():
        nvec, _ = resolve_noise_vector(noise_vectors, cond, group, beam, date_str)
        if nvec is None or len(nvec) != X.shape[1]:
            continue
        rows = np.asarray(idxs, dtype=int)
        dist[rows] = 1.0 - np.dot(X_norm[rows], nvec)
    return dist


def build_windows(flat_meta, dist, method_names):
    """window key -> {method -> {beam -> distance}}, plus the window start time."""
    windows: Dict[Any, Dict[str, Dict[Any, float]]] = {}
    order: Dict[Any, Any] = {}
    for idx, meta in enumerate(flat_meta):
        if not np.isfinite(dist[idx]):
            continue
        method = meta.get("method")
        if method not in method_names:
            continue
        wav = str(meta.get("wav", ""))
        src = meta.get("source_recording") or wav.split("_mono")[0].split("_sa")[0] \
            .split("_LabIR")[0].split("_SPIR")[0].split("_SBF")[0]
        start = round(float(meta.get("start_sec", 0.0)), 2)
        key = (src, start, round(float(meta.get("end_sec", 0.0)), 2))
        beam = beam_tag_from_name(wav, method)
        windows.setdefault(key, {}).setdefault(method, {})[beam] = float(dist[idx])
        order[key] = (src, start)
    return windows, order


def matrix_for(windows, order, method, min_share=0.9, min_beams=3,
               require_complete=True):
    """(n_windows, n_beams) delta-vs-mono matrix, kept only for complete windows."""
    counts: Counter = Counter()
    for v in windows.values():
        for beam in v.get(method, {}):
            counts[beam] += 1
    if not counts:
        return None
    n_total = len(windows)
    beams = sorted(b for b, c in counts.items() if c >= min_share * n_total)
    if len(beams) < min_beams:
        return None

    # A method that was never rendered for this date is not an incomplete
    # pairing, it is simply not applicable, so it returns None further down.
    # A method present for some windows but not all is incomplete, and that
    # stops the run.
    n_with_method = sum(1 for v in windows.values() if v.get(method))
    if n_with_method == 0:
        return None

    keys, rows, base = [], [], []
    missing_method, missing_mono, missing_beams = [], [], []
    for key, v in windows.items():
        got = v.get(method, {})
        mono = v.get("mono", {})
        if not got:
            if mono:
                missing_method.append(key)
            continue
        if not mono:
            missing_mono.append(key); continue
        gaps = [b for b in beams if b not in got]
        if gaps:
            missing_beams.append((key, gaps)); continue
        keys.append(key)
        rows.append([got[b] for b in beams])
        base.append(next(iter(mono.values())))

    if (missing_mono or missing_beams or missing_method) and require_complete:
        lines = [f"{method}: pasangan window tidak lengkap, analisis dibatalkan.",
                 f"  {len(keys)} window lengkap, {len(missing_method) + len(missing_mono) + len(missing_beams)} tidak."]
        if missing_method:
            lines.append(f"  {len(missing_method)} window punya mono tapi tidak punya "
                         f"{method} sama sekali, contoh: {missing_method[:3]}")
        if missing_mono:
            lines.append(f"  {len(missing_mono)} window tanpa mono, contoh: "
                         f"{missing_mono[:3]}")
        if missing_beams:
            k, gaps = missing_beams[0]
            lines.append(f"  {len(missing_beams)} window kekurangan beam, contoh: "
                         f"{k} kurang {gaps[:4]}"
                         + (f" dan {len(gaps) - 4} lagi" if len(gaps) > 4 else ""))
        lines.append("  Lengkapi render atau embedding yang kurang, lalu jalankan ulang.")
        raise IncompletePairing("\n".join(lines))

    if len(rows) < 20:
        return None

    idx = sorted(range(len(keys)), key=lambda i: order[keys[i]])
    keys = [keys[i] for i in idx]
    M = np.asarray(rows, dtype=np.float64)[idx]
    b = np.asarray(base, dtype=np.float64)[idx]
    return {"beams": beams, "keys": keys, "D": M - b[:, None]}


def permutation_null(D, n_perm, rng):
    """A. shuffle each beam column across windows, keep the window shape only.

    Each window is centred and scaled to unit spread first, so a window that
    simply has a stronger profile than another cannot lift the shuffled maximum
    by lending its large values to a flat window. What is left is peakiness:
    does one direction stand out inside a window more than a reshuffle gives.
    """
    C = D - D.mean(axis=1, keepdims=True)
    sd = C.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    C = C / sd
    observed = float(np.mean(C.max(axis=1)))
    n_win, n_beams = C.shape
    perms = np.empty(n_perm, dtype=np.float64)
    for p in range(n_perm):
        Cp = np.empty_like(C)
        for j in range(n_beams):
            Cp[:, j] = C[rng.permutation(n_win), j]
        perms[p] = float(np.mean(Cp.max(axis=1)))
    return {
        "observed_peakiness": round(observed, 5),
        "permuted_mean": round(float(perms.mean()), 5),
        "permuted_sd": round(float(perms.std()), 5),
        "structural_gain": round(observed - float(perms.mean()), 5),
        "p_value": round(float((1 + np.sum(perms >= observed)) / (n_perm + 1)), 4),
        "n_perm": n_perm,
    }


def bias_corrected(D):
    """B. remove each beam's own average before the maximum."""
    Db = D - D.mean(axis=0, keepdims=True)
    return Db, {
        "raw_best_lift": round(float(np.mean(D.max(axis=1))), 5),
        "raw_best_win_pct": round(float(np.mean(D.max(axis=1) > 0) * 100), 1),
        "raw_median_lift": round(float(np.mean(np.median(D, axis=1))), 5),
        "bias_corrected_best_lift": round(float(np.mean(Db.max(axis=1))), 5),
        "bias_corrected_best_win_pct": round(float(np.mean(Db.max(axis=1) > 0) * 100), 1),
    }


def persistence(Db, keys, beam_names):
    """C. does the winning direction stay put between consecutive windows?"""
    theta = Db.argmax(axis=1)
    n_beams = Db.shape[1]
    counts = np.bincount(theta, minlength=n_beams).astype(np.float64)
    p = counts / counts.sum()
    def repeat_at(lag):
        same, pairs = 0, 0
        for i in range(lag, len(theta)):
            if keys[i][0] != keys[i - lag][0]:
                continue
            pairs += 1
            same += int(theta[i] == theta[i - lag])
        return (round(100.0 * same / pairs, 1) if pairs else None), pairs

    lag_curve = {}
    for lag in (1, 2, 3, 5, 10, 20, 50, 100, 200, 500, 1000):
        pct, n = repeat_at(lag)
        if pct is not None:
            lag_curve[str(lag)] = pct
    pct1, pairs = repeat_at(1)
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])))
    return {
        "n_beams": n_beams,
        "observed_repeat_pct": pct1,
        "chance_repeat_pct": round(100.0 * float(np.sum(p ** 2)), 1),
        "repeat_pct_by_lag": lag_curve,
        "n_consecutive_pairs": pairs,
        "entropy_nats": round(entropy, 3),
        "entropy_if_uniform": round(float(np.log(n_beams)), 3),
        "top_beam_share_pct": round(100.0 * float(p.max()), 1),
        "top_beams": [{"beam": beam_names[i], "share_pct": round(100.0 * float(p[i]), 1)}
                      for i in np.argsort(-p)[:3]],
    }


def run_model(data_dir, location, date_str, model, n_perm, seed):
    source_dir = bacpipe_embeddings_dir(data_dir, location, model)
    meta_dir = bacpipe_meta_dir(location, model)
    _emb, X, _y, flat_meta, method_names = load_embeddings_from_dir(
        source_dir, methods=METHODS, date_filter=[date_str],
        source_tag=f"bacpipe:{model}", meta_dir=meta_dir,
    )
    if len(X) == 0:
        return {"model": model, "available": False, "reason": "no embeddings"}
    noise_vectors = load_noise_embeddings(meta_dir, expected_dim=X.shape[1])
    if not noise_vectors:
        return {"model": model, "available": False, "reason": "no noise references"}

    dist = point_noise_distance(X, flat_meta, method_names, noise_vectors)
    windows, order = build_windows(flat_meta, dist, method_names)
    del X

    rng = np.random.default_rng(seed)
    out = {"model": model, "available": True, "dim": None, "methods": {}}
    for method in BF_METHODS:
        got = matrix_for(windows, order, method)
        if got is None:
            out["methods"][method] = {"available": False}
            continue
        D = got["D"]
        Db, lifts = bias_corrected(D)
        out["methods"][method] = {
            "available": True,
            "n_windows": int(D.shape[0]),
            "n_beams": int(D.shape[1]),
            "lifts": lifts,
            "permutation": permutation_null(D, n_perm, rng),
            "persistence": persistence(Db, got["keys"], got["beams"]),
        }
    return out



def default_out(location, date_str, models, tool="permutation_null"):
    """Analysis output belongs in the audits dir, never in a new HOME folder."""
    tag = models[0] if len(models) == 1 else "all"
    return os.path.join(audits_dir(location), f"{date_str}_{tool}_{tag}.json")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default=ANALYSIS_OUTPUT)
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--date", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="Defaults to the audits dir in embedding_schema.")
    args = ap.parse_args()
    if not args.out:
        args.out = default_out(args.location, args.date, args.models)

    results = []
    for model in args.models:
        print(f"== {model} {args.date} ==", flush=True)
        try:
            res = run_model(args.data_dir, args.location, args.date, model,
                            args.n_perm, args.seed)
        # An incomplete pairing is a protocol violation, not a model that
        # happens to fail. It must stop the run so the missing data gets made.
        except IncompletePairing:
            raise
        except Exception as exc:  # keep the other models going
            res = {"model": model, "available": False, "reason": repr(exc)}
        results.append(res)
        for method, m in res.get("methods", {}).items():
            if not m.get("available"):
                continue
            pm = m["permutation"]
            ps = m["persistence"]
            print(f"  {method}: peakiness obs {pm['observed_peakiness']:.4f} "
                  f"perm {pm['permuted_mean']:.4f} gain {pm['structural_gain']:+.4f} "
                  f"p={pm['p_value']} | repeat {ps['observed_repeat_pct']}% "
                  f"vs chance {ps['chance_repeat_pct']}%", flush=True)

    payload = {"location": args.location, "date": args.date,
               "n_perm": args.n_perm, "models": results}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(convert_numpy_types(payload), fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
