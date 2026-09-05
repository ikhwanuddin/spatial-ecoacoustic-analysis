#!/usr/bin/env python3
"""Separability of biophony from background, per rendering method.

Labels come from the mono recording so no beamformed method is favoured.
A window counts as biophony when a detector window overlapping it holds bird
events, and as background only when every overlapping detector window was
accepted as clean. Anything ambiguous is left out.

Reported per method:
  d'   separation of the two label groups in units of their own spread
  AUC  probability that a random biophony window outscores a random background one
"""
import glob, json, os, re, sys
import numpy as np

sys.path.insert(0, os.path.expanduser("~/spatial-ecoacoustic-analysis"))
from embedding_io import load_embeddings_from_dir              # noqa: E402
from embedding_schema import (bacpipe_embeddings_dir, bacpipe_meta_dir,  # noqa: E402
                              beam_tag_from_name, condition_from_wav,
                              noise_group_for_method, resolve_noise_vector)
from spatial_clustering import l2_normalize, load_noise_embeddings   # noqa: E402

DATA = "/rds/general/user/ri322/ephemeral/sea-work"
LOC, DATE = "2A400", "2026-04-21"
LABEL_DIR = f"{DATA}/noise_auto_review/{LOC}/{DATE}/_labels_mono"


def recording_key(name):
    base = os.path.basename(str(name))
    for sep in ("_mono", "_sa", "_LabIR", "_SPIR"):
        base = base.split(sep)[0]
    return base


def load_labels():
    """recording -> list of (start, end, is_bird, is_clean)."""
    out = {}
    for path in sorted(glob.glob(f"{LABEL_DIR}/*/*_temporal_noise_detection.json")):
        res = json.load(open(path))
        key = recording_key(os.path.basename(res["input"]))
        out[key] = [(w["start_sec"], w["end_sec"],
                     bool(w.get("n_bird_events", 0)), bool(w.get("candidate")))
                    for w in res["windows"]]
    return out


def label_window(spans, start, end):
    """Biophony when any overlapping detector window holds bird events.

    Background is the natural complement: not one overlapping window holds a
    bird event. Requiring every overlap to also be an accepted noise candidate
    was too strict -- a 3 s embedding window spans about four 2 s detector
    windows, so almost nothing qualified.
    """
    hit = [s for s in spans if s[0] < end and start < s[1]]
    if not hit:
        return None
    if any(s[2] for s in hit):
        return "biophony"
    return "background"


def dprime_auc(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return None, None
    sd = np.sqrt((pos.var(ddof=1) + neg.var(ddof=1)) / 2.0)
    d = float((pos.mean() - neg.mean()) / sd) if sd > 0 else float("nan")
    # AUC via rank statistic (Mann-Whitney), ties shared
    allv = np.concatenate([pos, neg])
    order = allv.argsort()
    ranks = np.empty(len(allv), float)
    ranks[order] = np.arange(1, len(allv) + 1)
    i = 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    auc = float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))
    return round(d, 3), round(auc, 4)


def spearman(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3:
        return None
    def rank(v):
        order = v.argsort(); r = np.empty(len(v), float); r[order] = np.arange(len(v))
        return r
    ra, rb = rank(a), rank(b)
    ra -= ra.mean(); rb -= rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return round(float((ra * rb).sum() / denom), 3) if denom else None


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "birdnet"
    labels = load_labels()
    emb_dir = bacpipe_embeddings_dir(DATA, LOC, model)
    meta_dir = bacpipe_meta_dir(LOC, model)
    import contextlib, io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        _e, X, y, meta, methods = load_embeddings_from_dir(
            emb_dir, date_filter=[DATE], source_tag=model, meta_dir=meta_dir)
        noise = load_noise_embeddings(meta_dir, expected_dim=X.shape[1])
    Xn = l2_normalize(X)

    # score every embedding, then collect per (method, time window)
    per_window = {}
    for i, method in enumerate(methods):
        for idx in np.flatnonzero(y == i):
            rec = meta[idx]
            wav = rec.get("wav", "")
            key = recording_key(wav)
            spans = labels.get(key)
            if not spans:
                continue
            start, end = float(rec.get("start_sec", 0)), float(rec.get("end_sec", 0))
            tag = label_window(spans, start, end)
            if tag is None:
                continue
            vec, _k = resolve_noise_vector(
                noise, rec.get("condition") or condition_from_wav(wav),
                noise_group_for_method(method), beam_tag_from_name(wav, method))
            if vec is None or len(vec) != X.shape[1]:
                continue
            n_bird = sum(1 for s in spans if s[0] < end and start < s[1] and s[2])
            slot = per_window.setdefault((method, key, round(start, 2)), {"tag": tag, "n_bird": n_bird, "v": []})
            slot["v"].append(1.0 - float(np.dot(Xn[idx], vec)))

    print(f"model: {model} | label dari mono, {len(labels)} rekaman\n")
    print(f"{'metode':10s} {'agregasi':9s} {'beam':>5s} {'n bio':>6s} {'n latar':>8s} "
          f"{'d-prime':>8s} {'AUC':>7s} {'rho':>6s}")
    for method in methods:
        slots = [v for (m, _r, _s), v in per_window.items() if m == method]
        if not slots:
            continue
        n_beam = int(np.median([len(v["v"]) for v in slots]))
        for name, fn in (("best", np.max), ("median", np.median), ("mean", np.mean)):
            pos = [fn(v["v"]) for v in slots if v["tag"] == "biophony"]
            neg = [fn(v["v"]) for v in slots if v["tag"] == "background"]
            d, auc = dprime_auc(pos, neg)
            rho = spearman([fn(v["v"]) for v in slots], [v["n_bird"] for v in slots])
            print(f"{method:10s} {name:9s} {n_beam:5d} {len(pos):6d} {len(neg):8d} "
                  f"{'—' if d is None else f'{d:8.3f}'} {'—' if auc is None else f'{auc:7.4f}'} "
                  f"{'—' if rho is None else f'{rho:6.3f}'}")
            if n_beam == 1:
                break


if __name__ == "__main__":
    main()
