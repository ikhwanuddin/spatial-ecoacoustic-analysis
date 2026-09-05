#!/usr/bin/env python
"""Per-file dominant beam: is the winning direction fixed across the night?

Reuses the loaders in permutation_null. For each recording file we take the beam
that wins most often inside that file, after removing each beam's own nightly
mean so a constant offset cannot win. If the same direction dominates a file at
19:00 and a file at 04:00, the source is geographic. If it moves, the scene is.
"""
import argparse, json, os, sys
from collections import Counter
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir
from spatial_clustering import load_noise_embeddings
from permutation_null import (METHODS, BF_METHODS, point_noise_distance,
                              build_windows, matrix_for)


def stability(D, keys, beams):
    Db = D - D.mean(axis=0, keepdims=True)      # drop each beam's nightly mean
    theta = Db.argmax(axis=1)

    files = {}
    for k, t in zip(keys, theta):
        files.setdefault(k[0], []).append(int(t))
    per_file = []
    for src in sorted(files):
        c = Counter(files[src])
        top, n = c.most_common(1)[0]
        per_file.append({"file": os.path.basename(str(src)), "n_windows": len(files[src]),
                         "dominant_beam": str(beams[top]),
                         "dominant_share_pct": round(100.0 * n / len(files[src]), 1)})
    dom = [f["dominant_beam"] for f in per_file]
    cc = Counter(dom)
    same = sum(1 for i in range(1, len(dom)) if dom[i] == dom[i - 1])
    p = np.array([v / len(dom) for v in cc.values()])
    return {
        "n_beams": int(D.shape[1]), "n_windows": int(D.shape[0]), "n_files": len(per_file),
        "n_distinct_dominant": len(cc),
        "global_dominant": cc.most_common(1)[0][0],
        "global_dominant_file_share_pct": round(100.0 * cc.most_common(1)[0][1] / len(dom), 1),
        "adjacent_file_agree_pct": round(100.0 * same / max(1, len(dom) - 1), 1),
        "chance_agree_pct": round(100.0 * float(np.sum(p ** 2)), 1),
        "dominant_counts": cc.most_common(8),
        "per_file": per_file,
    }



def default_out(location, date_str, models, tool="beam_stability"):
    """Analysis output belongs in the audits dir, never in a new HOME folder."""
    tag = models[0] if len(models) == 1 else "all"
    return os.path.join(audits_dir(location), f"{date_str}_beam_stability_{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=os.environ.get("ANALYSIS_OUTPUT", "."))
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--date", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--out", default=None,
                    help="Defaults to the audits dir in embedding_schema.")
    a = ap.parse_args()
    if not a.out:
        a.out = default_out(a.location, a.date, a.models)

    result = {"location": a.location, "date": a.date, "models": []}
    for model in a.models:
        source_dir = bacpipe_embeddings_dir(a.data_dir, a.location, model)
        meta_dir = bacpipe_meta_dir(a.location, model)
        _e, X, _y, flat_meta, method_names = load_embeddings_from_dir(
            source_dir, methods=METHODS, date_filter=[a.date],
            source_tag=f"bacpipe:{model}", meta_dir=meta_dir)
        if len(X) == 0:
            result["models"].append({"model": model, "available": 0, "reason": "no embeddings"})
            continue
        nv = load_noise_embeddings(meta_dir, expected_dim=X.shape[1])
        if not nv:
            result["models"].append({"model": model, "available": 0, "reason": "no noise refs"})
            continue
        dist = point_noise_distance(X, flat_meta, method_names, nv)
        windows, order = build_windows(flat_meta, dist, method_names)
        del X
        entry = {"model": model, "available": 1, "methods": {}}
        for m in BF_METHODS:
            got = matrix_for(windows, order, m)
            if got is None:
                entry["methods"][m] = {"available": 0}
                continue
            entry["methods"][m] = stability(got["D"], got["keys"], got["beams"])
        result["models"].append(entry)
        print(f"[{model}] done", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
