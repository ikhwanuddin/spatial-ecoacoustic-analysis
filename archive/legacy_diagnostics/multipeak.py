#!/usr/bin/env python
"""Do two well separated directions light up at once, and do both persist?

A second peak that is far from the first, in a window, is a candidate second
bird. Noise produces such pairs too, so the test is persistence: a real pair
should still be there in the next window, a chance pair should not.

Angles come from the steering tags. LabIR carries a true azimuth and elevation,
so the separation below is a real angle on the sphere. SPIR carries azimuth and
focus distance, so its separation is azimuth only and is reported separately.
"""
import argparse, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir
from permutation_null import IncompletePairing, METHODS, build_windows, matrix_for
from beam_confidence import load_head, confidence, lagged_index

ELEV = {1: -45, 2: -30, 3: -20, 4: -10, 5: 0, 6: 10,
        7: 20, 8: 30, 9: 45, 10: 60, 11: 75, 12: 90}
SEPARATION_DEG = 90.0


def angles(tag):
    m = re.match(r"LabIR\(S(\d+)_(\d+)\)", tag)
    if m:
        return float(m.group(2)), float(ELEV.get(int(m.group(1)), 0.0))
    m = re.match(r"SPIR\d?\((\d+)m_(\d+)", tag)
    if m:
        return float(m.group(2)), 0.0
    return None


def sphere_sep(a, b):
    """Great circle angle between two (azimuth, elevation) pairs, degrees."""
    az1, el1 = np.radians(a); az2, el2 = np.radians(b)
    c = np.sin(el1) * np.sin(el2) + np.cos(el1) * np.cos(el2) * np.cos(az1 - az2)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def run(data_dir, location, date_str, model, method, seed=0):
    src = bacpipe_embeddings_dir(data_dir, location, model)
    meta = bacpipe_meta_dir(location, model)
    _e, X, _y, fm, mn = load_embeddings_from_dir(
        src, methods=METHODS, date_filter=[date_str],
        source_tag=f"bacpipe:{model}", meta_dir=meta)
    if len(X) == 0:
        return {"model": model, "date": date_str, "method": method, "available": False}
    head = load_head(os.path.dirname(os.path.abspath(__file__)), model)
    conf, _t, _a, _b = confidence(X, head, seed)
    del X
    win, order = build_windows(fm, conf, mn)
    got = matrix_for(win, order, method)
    if got is None:
        return {"model": model, "date": date_str, "method": method,
                "available": False, "reason": f"{method} tidak terpasang"}
    D, keys, beams = got["D"], got["keys"], got["beams"]
    coords = [angles(b) for b in beams]
    if any(c is None for c in coords):
        return {"model": model, "date": date_str, "method": method,
                "available": False, "reason": "tag arah tidak terbaca"}
    n_b = len(beams)
    sep = np.zeros((n_b, n_b))
    for i in range(n_b):
        for j in range(n_b):
            sep[i, j] = sphere_sep(coords[i], coords[j])

    Db = D - D.mean(axis=0, keepdims=True)
    rank = np.argsort(-Db, axis=1)
    first = rank[:, 0]
    # second peak: highest scoring beam that is far from the first
    second = np.full(len(keys), -1, dtype=np.int64)
    for w in range(len(keys)):
        far = np.where(sep[first[w]] >= SEPARATION_DEG)[0]
        if len(far):
            second[w] = far[np.argmax(Db[w, far])]
    has2 = (second >= 0) & (Db[np.arange(len(keys)), np.maximum(second, 0)] > 0)

    prev = lagged_index(keys, 1)
    ok = prev >= 0

    def repeat(idx, mask):
        m = ok & mask & (idx >= 0) & (idx[np.maximum(prev, 0)] >= 0)
        if m.sum() < 50:
            return None, 0
        same = idx[m] == idx[prev[m]]
        return round(100.0 * float(same.mean()), 1), int(m.sum())

    p1, n1 = repeat(first, np.ones(len(keys), bool))
    p2, n2 = repeat(second, has2)
    # chance rate from how often each direction is chosen at all
    c1 = np.bincount(first, minlength=n_b) / len(first)
    s2 = second[has2]
    c2 = (np.bincount(s2, minlength=n_b) / len(s2)) if len(s2) else np.zeros(n_b)
    return {
        "model": model, "date": date_str, "method": method, "available": True,
        "n_windows": int(len(keys)), "n_beams": n_b,
        "separation_deg": SEPARATION_DEG,
        "pct_windows_with_far_second_peak": round(100.0 * float(has2.mean()), 1),
        "second_peak_strength_vs_first": round(float(np.median(
            Db[has2, second[has2]] / np.maximum(Db[has2, first[has2]], 1e-9))), 3)
            if has2.sum() else None,
        "first_peak_repeat_pct": p1, "first_peak_chance_pct": round(100.0 * float((c1 ** 2).sum()), 1),
        "second_peak_repeat_pct": p2, "second_peak_chance_pct": round(100.0 * float((c2 ** 2).sum()), 1),
        "n_pairs_first": n1, "n_pairs_second": n2,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("ANALYSIS_OUTPUT", "."))
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--dates", nargs="+", required=True)
    ap.add_argument("--models", nargs="+", default=["birdnet", "avesecho_passt"])
    ap.add_argument("--methods", nargs="+", default=["bf_LabIR", "bf_SPIR"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if not a.out:
        a.out = os.path.join(audits_dir(a.location), "multipeak_persistence.json")
    res = {"location": a.location, "results": []}
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
