#!/usr/bin/env python
"""Beam confidence: score every beam with a detector head, not with a prototype.

BirdNET's classifier is a single dense layer on the same 1024-d embedding we
already cache, so confidence = sigmoid(emb @ W + b), max over classes. No audio,
no prototype, no reference to drift.

Two criteria are then available per window: noise distance (prototype) and
detector confidence (no prototype). The point of this script is to select the
beam with one and report the other, which removes the winner's curse from the
reported number, and to ask how often the two disagree.
"""
import argparse, json, os, sys
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_io import load_embeddings_from_dir
from embedding_schema import (audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir,
                              condition_from_wav)
from spatial_clustering import load_noise_embeddings, l2_normalize
from permutation_null import (METHODS, BF_METHODS, point_noise_distance,
                              build_windows, matrix_for, permutation_null,
                              persistence)

HEADS = {"birdnet": "keras", "avesecho_passt": "passt"}


def _birdnet_head(repo):
    m = tf.keras.models.load_model(
        os.path.join(repo, "bacpipe/checkpoints/birdnet/birdnetv2.4.keras"), compile=False)
    W, b = m.get_layer("CLASS_DENSE_LAYER").get_weights()
    return {"kind": "linear", "W": np.asarray(W, np.float32), "b": np.asarray(b, np.float32)}


def _passt_head(repo):
    """AvesEcho is a PaSST: LayerNorm, then the mean of two linear heads."""
    import torch
    sd = torch.load(os.path.join(repo, "bacpipe/checkpoints/avesecho_passt/best_model_passt.pt"),
                    map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    g = lambda k: np.asarray(sd[k].detach().cpu().numpy(), np.float32)
    return {"kind": "passt",
            "ln_w": g("net.head.0.weight"), "ln_b": g("net.head.0.bias"),
            "W": g("net.head.1.weight").T, "b": g("net.head.1.bias"),
            "Wd": g("net.head_dist.weight").T, "bd": g("net.head_dist.bias")}


def load_head(repo, model):
    return {"keras": _birdnet_head, "passt": _passt_head}[HEADS[model]](repo)


def logits(head, Xb):
    if head["kind"] == "linear":
        return Xb @ head["W"] + head["b"]
    m = Xb.mean(axis=1, keepdims=True)
    v = Xb.var(axis=1, keepdims=True)
    z = (Xb - m) / np.sqrt(v + 1e-6) * head["ln_w"] + head["ln_b"]
    return 0.5 * ((z @ head["W"] + head["b"]) + (z @ head["Wd"] + head["bd"]))


def confidence(X, head, seed, batch=20000):
    """max sigmoid over classes, per row.

    Also returns the max over two disjoint halves of the class set. Selecting a
    beam with half A and scoring it with half B gives a confidence number that
    selection never saw, which is the only way to use the detector as both the
    selector and the score without the winner's curse.
    """
    n_cls = head["W"].shape[1]
    perm = np.random.default_rng(seed).permutation(n_cls)
    A, B = perm[: n_cls // 2], perm[n_cls // 2:]
    out = np.empty(len(X), dtype=np.float32)
    ca = np.empty(len(X), dtype=np.float32)
    cb = np.empty(len(X), dtype=np.float32)
    top = np.empty(len(X), dtype=np.int32)
    for i in range(0, len(X), batch):
        z = logits(head, np.asarray(X[i:i + batch], dtype=np.float32))
        sig = lambda v: 1.0 / (1.0 + np.exp(-v))
        j = z.argmax(axis=1)
        out[i:i + batch] = sig(z[np.arange(len(j)), j])
        top[i:i + batch] = j
        ca[i:i + batch] = sig(z[:, A].max(axis=1))
        cb[i:i + batch] = sig(z[:, B].max(axis=1))
    return out, top, ca, cb


def reference_free_scores(X, flat_meta, method_names):
    """Two selection scores that need no prototype and no detector.

    far_from_mono: how far this beam sits from the mono rendering of the same
      window. A beam that hears something the omnidirectional channel does not
      should sit further away.
    far_from_beams: how far this beam sits from the average of all beams in the
      same window, so a single odd direction stands out.
    """
    Xn = l2_normalize(X)
    mono_of, beams_of = {}, {}
    for idx, meta in enumerate(flat_meta):
        method = meta.get("method")
        if method not in method_names:
            continue
        wav = str(meta.get("wav", ""))
        src = meta.get("source_recording") or wav.split("_mono")[0].split("_sa")[0] \
            .split("_LabIR")[0].split("_SPIR")[0]
        key = (src, round(float(meta.get("start_sec", 0.0)), 2),
               round(float(meta.get("end_sec", 0.0)), 2))
        if method == "mono":
            mono_of[key] = idx
        elif method in BF_METHODS:
            beams_of.setdefault((key, method), []).append(idx)

    far_mono = np.full(len(X), np.nan, dtype=np.float32)
    far_beams = np.full(len(X), np.nan, dtype=np.float32)
    # mono is the baseline these scores are measured against, so it sits at zero
    for idx in mono_of.values():
        far_mono[idx] = 0.0
        far_beams[idx] = 0.0
    for (key, _method), idxs in beams_of.items():
        rows = np.asarray(idxs, dtype=int)
        V = Xn[rows]
        c = V.mean(axis=0)
        n = np.linalg.norm(c)
        if n > 0:
            far_beams[rows] = 1.0 - V @ (c / n)
        mi = mono_of.get(key)
        if mi is not None:
            far_mono[rows] = 1.0 - V @ Xn[mi]
    return far_mono, far_beams


def lagged_index(keys, lag):
    """For each window, the row `lag` windows earlier in the same recording."""
    prev = np.full(len(keys), -1, dtype=np.int64)
    for i in range(lag, len(keys)):
        if keys[i][0] == keys[i - lag][0]:
            prev[i] = i - lag
    return prev


def lag_sweep(C, Db_by_name, keys, rng, lags=(1, 2, 3, 5, 10, 20, 50, 75)):
    """Pick the beam using an earlier window, score it on this one.

    The score never saw the choice, so there is no winner's curse. If the
    advantage decays with lag, the structure is an acoustic event. If it stays
    flat, the structure is fixed geometry.
    """
    n, k = C.shape
    rows = np.arange(n)
    rand_idx = rng.integers(0, k, n)
    out = {}
    # control: choose the beam from a window in a different recording. Whatever
    # advantage survives that is fixed geometry, not an event in this window.
    src = np.array([k[0] for k in keys])
    other = np.empty(n, dtype=np.int64)
    for i in range(n):
        j = int(rng.integers(0, n))
        for _ in range(8):
            if src[j] != src[i]:
                break
            j = int(rng.integers(0, n))
        other[i] = j
    out["_other_file_control"] = {}
    base_all = C[rows, rand_idx]
    for name, Db in Db_by_name.items():
        if Db is None:
            continue
        v = C[rows, Db[other].argmax(axis=1)]
        out["_other_file_control"][name] = {
            "mean_lift": round(float(v.mean()), 5),
            "win_pct": round(100.0 * float((v > 0).mean()), 1),
            "gain_over_random": round(float(v.mean() - base_all.mean()), 5)}

    for lag in lags:
        prev = lagged_index(keys, lag)
        ok = prev >= 0
        if ok.sum() < 200:
            continue
        entry = {"n": int(ok.sum())}
        base = C[rows, rand_idx][ok]
        entry["random_beam"] = {"mean_lift": round(float(base.mean()), 5),
                                "win_pct": round(100.0 * float((base > 0).mean()), 1)}
        for name, Db in Db_by_name.items():
            if Db is None:
                continue
            theta = np.zeros(n, dtype=np.int64)
            theta[ok] = Db[prev[ok]].argmax(axis=1)
            v = C[rows, theta][ok]
            entry[name] = {"mean_lift": round(float(v.mean()), 5),
                           "win_pct": round(100.0 * float((v > 0).mean()), 1),
                           "gain_over_random": round(float(v.mean() - base.mean()), 5)}
            if lag == lags[0]:
                entry[name]["ci95_gain"] = block_bootstrap(
                    v - base, np.array([k[0] for k in keys])[ok], rng)
        out[str(lag)] = entry
    return out


def block_bootstrap(diff, groups, rng, n_boot=2000):
    """95 percent interval, resampling whole recordings rather than windows.

    Windows inside one recording are not independent, so a window-level interval
    would be far too narrow. The recording is the unit that repeats.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    by_g = [diff[inv == i] for i in range(len(uniq))]
    means = np.array([g.mean() for g in by_g])
    sizes = np.array([len(g) for g in by_g], dtype=np.float64)
    draws = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        draws[i] = float(np.average(means[pick], weights=sizes[pick]))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    return {"lo": round(float(lo), 5), "hi": round(float(hi), 5),
            "n_recordings": int(len(uniq))}


def by_condition(C, Cb, keys, rng, lag=1):
    """The unbiased lag-1 estimator, split by time of day."""
    n, k = C.shape
    rows = np.arange(n)
    rand_idx = rng.integers(0, k, n)
    prev = lagged_index(keys, lag)
    theta = np.zeros(n, dtype=np.int64)
    ok = prev >= 0
    theta[ok] = Cb[prev[ok]].argmax(axis=1)
    cond = np.array([condition_from_wav(str(kk[0])) or "unknown" for kk in keys])
    sel = C[rows, theta]
    base = C[rows, rand_idx]
    out = {}
    for c in sorted(set(cond)):
        m = ok & (cond == c)
        if m.sum() < 100:
            continue
        out[c] = {"n": int(m.sum()),
                  "sel_by_conf": round(float(sel[m].mean()), 5),
                  "win_pct": round(100.0 * float((sel[m] > 0).mean()), 1),
                  "random_beam": round(float(base[m].mean()), 5),
                  "gain_over_random": round(float(sel[m].mean() - base[m].mean()), 5)}
    return out


def ladder(C, CA, CB, Dn, extra, keys, seed):
    """C, CA, CB: window x beam confidence, all classes / half A / half B.

    Dn is the same shape for noise distance, or None. Every entry is already a
    difference against mono on the same window, so mono is 0 by construction.
    Every selector below is scored on CB, which it did not choose with, except
    the oracle rows which are there to show the ceiling and the bias.
    """
    rng = np.random.default_rng(seed)
    n, k = C.shape
    rows = np.arange(n)
    Cb = C - C.mean(axis=0, keepdims=True)
    CAb = CA - CA.mean(axis=0, keepdims=True)
    out = {}

    def rep(name, vals, mask=None):
        v = vals if mask is None else vals[mask]
        out[name] = {"mean_lift": round(float(np.mean(v)), 5),
                     "win_pct": round(100.0 * float(np.mean(v > 0)), 1),
                     "n": int(len(v))}

    # --- matched control: one beam, chosen without looking at anything ---
    rand_idx = rng.integers(0, k, n)
    rep("random_beam", C[rows, rand_idx])
    rep("random_beam_scoreB", CB[rows, rand_idx])
    rep("fixed_best_beam", C[:, int(C.mean(axis=0).argmax())])

    # --- biased ceiling, shown for reference only ---
    rep("oracle_max", C.max(axis=1))
    rep("oracle_max_bias_corrected", Cb[rows, Cb.argmax(axis=1)])

    # --- unbiased selector 1: pick with half the classes, score with the other ---
    theta_a = CAb.argmax(axis=1)
    rep("sel_class_split_scoreB", CB[rows, theta_a])

    # --- unbiased selector 2: pick with an earlier window, score this one ---
    Dnb = (Dn - Dn.mean(axis=0, keepdims=True)) if Dn is not None else None
    out["by_condition"] = by_condition(C, Cb, keys, np.random.default_rng(seed + 2))
    sel_maps = {"sel_by_conf": Cb, "sel_by_noise_distance": Dnb}
    for name, E in extra.items():
        if E is not None:
            sel_maps["sel_by_" + name] = E - E.mean(axis=0, keepdims=True)
    out["lag_sweep"] = lag_sweep(C, sel_maps, keys, np.random.default_rng(seed + 1))

    # --- unbiased selector 3: reference-free, no prototype and no detector ---
    for name, E in extra.items():
        if E is None:
            continue
        Eb = E - E.mean(axis=0, keepdims=True)
        rep("sel_" + name, C[rows, Eb.argmax(axis=1)])

    # --- unbiased selector 4: pick with the prototype, score with the detector ---
    if Dn is not None:
        theta_n = Dnb.argmax(axis=1)
        theta_c = Cb.argmax(axis=1)
        rep("sel_noise_distance", C[rows, theta_n])
        p_n = np.bincount(theta_n, minlength=k) / n
        p_c = np.bincount(theta_c, minlength=k) / n
        out["criterion_agreement"] = {
            "noise_vs_conf_agree_pct": round(100.0 * float(np.mean(theta_n == theta_c)), 2),
            "chance_pct": round(100.0 * float(np.dot(p_n, p_c)), 2),
            "rank_corr_of_beam_means": round(float(np.corrcoef(
                np.argsort(np.argsort(C.mean(0))),
                np.argsort(np.argsort(Dn.mean(0))))[0, 1]), 3),
        }
    # how self-consistent the detector is across its own class halves
    p_a = np.bincount(theta_a, minlength=k) / n
    p_c2 = np.bincount(Cb.argmax(axis=1), minlength=k) / n
    out["class_split_agreement"] = {
        "agree_pct": round(100.0 * float(np.mean(theta_a == Cb.argmax(axis=1))), 2),
        "chance_pct": round(100.0 * float(np.dot(p_a, p_c2)), 2),
    }
    return out


def run(repo, data_dir, location, date_str, model, n_perm, seed):
    source_dir = bacpipe_embeddings_dir(data_dir, location, model)
    meta_dir = bacpipe_meta_dir(location, model)
    _e, X, _y, flat_meta, method_names = load_embeddings_from_dir(
        source_dir, methods=METHODS, date_filter=[date_str],
        source_tag=f"bacpipe:{model}", meta_dir=meta_dir)
    if len(X) == 0:
        return {"model": model, "available": False, "reason": "no embeddings"}

    head = load_head(repo, model)
    if X.shape[1] != head["W"].shape[0]:
        return {"model": model, "available": False,
                "reason": f"dim {X.shape[1]} != head {head['W'].shape[0]}"}
    conf, top, cA, cB = confidence(X, head, seed)

    nv = load_noise_embeddings(meta_dir, expected_dim=X.shape[1])
    dist = point_noise_distance(X, flat_meta, method_names, nv) if nv else None
    far_mono, far_beams = reference_free_scores(X, flat_meta, method_names)
    del X

    win_c, order = build_windows(flat_meta, conf, method_names)
    win_a = build_windows(flat_meta, cA, method_names)[0]
    win_b = build_windows(flat_meta, cB, method_names)[0]
    win_d = build_windows(flat_meta, dist, method_names)[0] if dist is not None else None
    win_fm = build_windows(flat_meta, far_mono, method_names)[0]
    win_fb = build_windows(flat_meta, far_beams, method_names)[0]

    out = {"model": model, "available": True, "has_noise_reference": bool(nv),
           "conf_mean_all_rows": round(float(conf.mean()), 5),
           "conf_p95_all_rows": round(float(np.percentile(conf, 95)), 5),
           "n_distinct_top_class": int(len(np.unique(top))),
           "methods": {}}

    for method in BF_METHODS:
        got = matrix_for(win_c, order, method)
        if got is None:
            out["methods"][method] = {"available": False}
            continue
        C = got["D"]
        entry = {"available": True, "n_windows": int(C.shape[0]),
                 "n_beams": int(C.shape[1])}
        def aligned(w):
            g = matrix_for(w, order, method)
            if g is None or g["keys"] != got["keys"] or g["beams"] != got["beams"]:
                return None
            return g["D"]
        CA, CB = aligned(win_a), aligned(win_b)
        Dn = aligned(win_d) if win_d is not None else None
        extra = {"far_from_mono": aligned(win_fm), "far_from_beams": aligned(win_fb)}
        if CA is None or CB is None:
            out["methods"][method] = {"available": False, "reason": "class halves misaligned"}
            continue
        entry["ladder"] = ladder(C, CA, CB, Dn, extra, got["keys"], seed)
        if C.shape[1] > 1:
            rng = np.random.default_rng(seed)
            entry["permutation"] = permutation_null(C, n_perm, rng)
            entry["persistence"] = persistence(
                C - C.mean(axis=0, keepdims=True), got["keys"], got["beams"])
        out["methods"][method] = entry

    # sa renders one stream, so there is no beam to choose. Its lift against
    # mono is the honest competitor to any beamforming number, and CONTEXT.md
    # 5.1 records it beating both beamformers under noise distance.
    out["methods"]["sa"] = single_stream(win_c, order, seed)
    return out


def single_stream(windows, order, seed, method="sa"):
    got = matrix_for(windows, order, method, min_beams=1)
    if got is None:
        return {"available": False, "reason": "no paired windows"}
    v = got["D"][:, 0]
    rng = np.random.default_rng(seed + 3)
    return {
        "available": True,
        "n_windows": int(len(v)),
        "n_beams": 1,
        "lift": {"mean_lift": round(float(v.mean()), 5),
                 "win_pct": round(100.0 * float((v > 0).mean()), 1),
                 "ci95_lift": block_bootstrap(
                     v, np.array([k[0] for k in got["keys"]]), rng)},
    }



def default_out(location, date_str, models, tool="beam_confidence"):
    """Analysis output belongs in the audits dir, never in a new HOME folder."""
    tag = models[0] if len(models) == 1 else "all"
    return os.path.join(audits_dir(location), f"{date_str}_beam_confidence_{tag}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--data-dir", default=os.environ.get("ANALYSIS_OUTPUT", "."))
    ap.add_argument("--location", default="2A400")
    ap.add_argument("--date", required=True)
    ap.add_argument("--models", nargs="+", default=["birdnet"])
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None,
                    help="Defaults to the audits dir in embedding_schema.")
    a = ap.parse_args()
    if not a.out:
        a.out = default_out(a.location, a.date, a.models)

    res = {"location": a.location, "date": a.date, "n_perm": a.n_perm, "models": []}
    for model in a.models:
        res["models"].append(run(a.repo, a.data_dir, a.location, a.date,
                                 model, a.n_perm, a.seed))
        print(f"[{model}] done", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
