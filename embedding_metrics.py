"""
Embedding-based metrics for spatial method comparison.

Focus: method separability and LabIR direction consistency — not species ID.

Usage:
  python embedding_metrics.py --location 2A400 --date 2026-04-21 \\
      --embeddings /Volumes/WD2TB/sea-data/2A400/embeddings
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import ANALYSIS_OUTPUT
from embedding_io import load_embeddings_from_dir, load_location_embeddings
from embedding_schema import audits_dir, legacy_flat_embeddings_dir


def _l2_normalize(X: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


def pairwise_cosine_mean(X: np.ndarray, max_pairs: int = 5000, seed: int = 0) -> float:
    """Mean cosine similarity over a subsample of pairs (upper triangle)."""
    n = len(X)
    if n < 2:
        return float("nan")
    Xn = _l2_normalize(X.astype(np.float32))
    # full matrix if small
    if n <= 80:
        S = Xn @ Xn.T
        iu = np.triu_indices(n, k=1)
        return float(np.mean(S[iu]))
    rng = np.random.default_rng(seed)
    # sample pairs
    i = rng.integers(0, n, size=max_pairs)
    j = rng.integers(0, n, size=max_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    sims = np.sum(Xn[i] * Xn[j], axis=1)
    return float(np.mean(sims))


def centroid_cosine(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return float("nan")
    ca = _l2_normalize(a.mean(axis=0, keepdims=True))[0]
    cb = _l2_normalize(b.mean(axis=0, keepdims=True))[0]
    return float(np.dot(ca, cb))


def method_separability(
    X: np.ndarray,
    y_method: np.ndarray,
    method_names: Sequence[str],
    max_points: int = 2000,
    seed: int = 0,
) -> Dict[str, Any]:
    """Within-method vs between-method cohesion (comparable cosine stats).

    Primary metrics (same scale):
      - mean cosine of points → own method centroid (within cohesion)
      - mean cosine of points → other method centroids (between attraction)
    Also reports pairwise within and centroid–centroid matrix.
    """
    rng = np.random.default_rng(seed)
    by_m: Dict[str, np.ndarray] = {}
    for i, name in enumerate(method_names):
        mask = y_method == i
        if np.any(mask):
            by_m[name] = X[mask]

    names = list(by_m.keys())
    centroids: Dict[str, np.ndarray] = {}
    for name, arr in by_m.items():
        c = arr.mean(axis=0)
        centroids[name] = c / (np.linalg.norm(c) + 1e-9)

    within = {}
    within_cohesion_vals = []
    between_attraction_vals = []
    for name, arr in by_m.items():
        if len(arr) > max_points:
            idx = rng.choice(len(arr), size=max_points, replace=False)
            A = arr[idx]
        else:
            A = arr
        An = _l2_normalize(A.astype(np.float32))
        own = An @ centroids[name]
        others = [centroids[o] for o in names if o != name]
        if others:
            other_mat = np.stack(others, axis=1)  # D × (M-1)
            # mean cosine to each other centroid, then mean
            bet = An @ other_mat  # N × (M-1)
            bet_mean = float(np.mean(bet))
        else:
            bet_mean = float("nan")
        w_pair = pairwise_cosine_mean(A, seed=seed)
        within[name] = {
            "n": int(len(arr)),
            "mean_pairwise_cosine": w_pair,
            "mean_cosine_to_own_centroid": float(np.mean(own)),
            "mean_cosine_to_other_centroids": bet_mean,
            "mean_l2_norm": float(np.mean(np.linalg.norm(arr, axis=1))),
        }
        within_cohesion_vals.append(float(np.mean(own)))
        if not np.isnan(bet_mean):
            between_attraction_vals.append(bet_mean)

    centroid = {a: {b: None for b in names} for a in names}
    between_cent = []
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            c = float(np.dot(centroids[a], centroids[b]))
            centroid[a][b] = c
            if i < j:
                between_cent.append(c)

    return {
        "within_method": within,
        "centroid_cosine": centroid,
        "mean_within_centroid_cohesion": (
            float(np.mean(within_cohesion_vals)) if within_cohesion_vals else None
        ),
        "mean_between_centroid_attraction": (
            float(np.mean(between_attraction_vals)) if between_attraction_vals else None
        ),
        "separability_gap": (
            float(np.mean(within_cohesion_vals) - np.mean(between_attraction_vals))
            if within_cohesion_vals and between_attraction_vals
            else None
        ),
        "mean_between_centroid_cosine": float(np.mean(between_cent)) if between_cent else None,
        "note": (
            "separability_gap = mean(cos to own centroid) − mean(cos to other centroids); "
            "positive ⇒ methods form tighter clusters than cross-method attraction"
        ),
    }


def direction_consistency(
    X: np.ndarray,
    flat_meta: List[dict],
    method_filter: Optional[str] = "bf_LabIR",
    max_per_bin: int = 400,
    seed: int = 0,
) -> Dict[str, Any]:
    """LabIR direction structure via centroid cohesion (comparable stats).

    For each azimuth bin: mean cos(point, own az centroid) vs mean cos(point, other az centroids).
    """
    rng = np.random.default_rng(seed)
    rows: List[Tuple[np.ndarray, Any, Any]] = []
    for i, meta in enumerate(flat_meta):
        if i >= len(X):
            break
        if method_filter and meta.get("method") != method_filter:
            continue
        az = meta.get("azimuth")
        el = meta.get("elevation")
        if az is None:
            continue
        rows.append((X[i], az, el))

    if len(rows) < 10:
        return {
            "n": len(rows),
            "error": "too few directed embeddings",
            "method_filter": method_filter,
        }

    by_az: Dict[Any, List[np.ndarray]] = defaultdict(list)
    by_el: Dict[Any, List[np.ndarray]] = defaultdict(list)
    for vec, az, el in rows:
        by_az[az].append(vec)
        if el is not None:
            by_el[el].append(vec)

    def _subsample(arrs: List[np.ndarray]) -> np.ndarray:
        A = np.stack(arrs, axis=0)
        if len(A) > max_per_bin:
            idx = rng.choice(len(A), size=max_per_bin, replace=False)
            A = A[idx]
        return A

    az_keys = sorted(by_az.keys(), key=lambda x: (x is None, x))
    cents = {}
    stacks = {}
    for az in az_keys:
        A = _subsample(by_az[az])
        stacks[az] = A
        c = A.mean(axis=0)
        cents[az] = c / (np.linalg.norm(c) + 1e-9)

    within_az = {}
    cohesion = []
    attraction = []
    between_pairs = []
    for az in az_keys:
        An = _l2_normalize(stacks[az].astype(np.float32))
        own = An @ cents[az]
        others = [cents[o] for o in az_keys if o != az]
        if others:
            bet = An @ np.stack(others, axis=1)
            bet_m = float(np.mean(bet))
        else:
            bet_m = float("nan")
        within_az[str(az)] = {
            "n": int(len(by_az[az])),
            "mean_cosine_to_own_centroid": float(np.mean(own)),
            "mean_cosine_to_other_centroids": bet_m,
            "mean_pairwise_cosine": pairwise_cosine_mean(stacks[az], seed=seed),
        }
        cohesion.append(float(np.mean(own)))
        if not np.isnan(bet_m):
            attraction.append(bet_m)

    for i, a in enumerate(az_keys):
        for b in az_keys[i + 1 :]:
            between_pairs.append(
                {
                    "az_a": a,
                    "az_b": b,
                    "centroid_cosine": float(np.dot(cents[a], cents[b])),
                }
            )

    within_el = {}
    for el, arrs in by_el.items():
        A = _subsample(arrs)
        within_el[str(el)] = {
            "n": int(len(arrs)),
            "mean_pairwise_cosine": pairwise_cosine_mean(A, seed=seed),
        }

    return {
        "method_filter": method_filter,
        "n": len(rows),
        "n_azimuth_bins": len(by_az),
        "n_elevation_bins": len(by_el),
        "within_azimuth": within_az,
        "between_azimuth_centroids": between_pairs[:40],
        "mean_within_azimuth_cohesion": float(np.mean(cohesion)) if cohesion else None,
        "mean_between_azimuth_attraction": (
            float(np.mean(attraction)) if attraction else None
        ),
        "direction_gap": (
            float(np.mean(cohesion) - np.mean(attraction))
            if cohesion and attraction
            else None
        ),
        "mean_between_azimuth_centroid_cosine": (
            float(np.mean([p["centroid_cosine"] for p in between_pairs]))
            if between_pairs
            else None
        ),
        "within_elevation": within_el,
        "azimuth_values": [str(k) for k in az_keys],
        "note": (
            "direction_gap = mean(cos to own az centroid) − mean(cos to other az centroids)"
        ),
    }


def noise_distance_by_method(
    X: np.ndarray,
    y_method: np.ndarray,
    method_names: Sequence[str],
    emb_dir: str,
) -> Dict[str, Any]:
    """Mean cosine distance to noise mean vector if noise_*.npy present."""
    from embedding_io import list_embedding_files  # noqa: F401

    if not os.path.isdir(emb_dir):
        return {}
    noise_vectors: Dict[str, np.ndarray] = {}
    for fname in os.listdir(emb_dir):
        if not (fname.startswith("noise_") and fname.endswith("_embeddings.npy")):
            continue
        group = fname.replace("noise_", "").replace("_embeddings.npy", "")
        noise = np.load(os.path.join(emb_dir, fname)).astype(np.float32)
        if noise.ndim == 1:
            noise = noise.reshape(1, -1)
        nmean = _l2_normalize(noise).mean(axis=0)
        noise_vectors[group] = nmean / (np.linalg.norm(nmean) + 1e-9)

    # Each method is scored against its own noise group. This is essential
    # for mono/SA versus directional BF and also works for bacpipe outputs.
    out: Dict[str, Any] = {}
    for i, name in enumerate(method_names):
        group = name.replace("bf_", "", 1)
        nmean = noise_vectors.get(group)
        mask = y_method == i
        if nmean is None or not np.any(mask):
            continue
        Xm = _l2_normalize(X[mask])
        sims = Xm @ nmean
        key = f"{name}__vs_noise_{group}"
        out[key] = {
            "method": name,
            "noise_group": group,
            "n": int(mask.sum()),
            "mean_cosine_to_noise": float(np.mean(sims)),
            "mean_cosine_distance": float(1.0 - np.mean(sims)),
        }

    mono = out.get("mono__vs_noise_mono")
    if mono:
        mono_distance = mono["mean_cosine_distance"]
        for result in out.values():
            result["delta_vs_mono"] = float(
                result["mean_cosine_distance"] - mono_distance
            )
    return out


def run_metrics(
    *,
    emb_dir: Optional[str],
    location: Optional[str],
    data_dir: str,
    methods: List[str],
    date_filter: Optional[List[str]],
    direction_method: str,
    backends: Optional[List[str]],
) -> Dict[str, Any]:
    if emb_dir:
        print(f"Loading from {emb_dir}")
        emb_map, X, y_method, flat_meta, method_names = load_embeddings_from_dir(
            emb_dir, methods=methods, date_filter=date_filter, source_tag="direct"
        )
        load_dir = emb_dir
    else:
        loc = location or "2A400"
        packed = load_location_embeddings(
            data_dir,
            loc,
            methods=methods,
            date_filter=date_filter,
            backends=backends or ["legacy"],
        )
        X = packed["X"]
        y_method = packed["y_method"]
        flat_meta = packed["flat_meta"]
        method_names = packed["methods"]
        load_dir = packed["dirs"][0][0] if packed["dirs"] else legacy_flat_embeddings_dir(
            data_dir, loc
        )
        print(f"Sources: {packed['per_source']}")

    if len(X) == 0:
        raise SystemExit("No embeddings loaded")

    print(f"Loaded X={X.shape}")
    report: Dict[str, Any] = {
        "n_embeddings": int(len(X)),
        "embedding_dim": int(X.shape[1]),
        "methods": method_names,
        "dates": date_filter,
        "method_separability": method_separability(X, y_method, method_names),
        "direction_consistency": direction_consistency(
            X, flat_meta, method_filter=direction_method
        ),
        "noise_distance": noise_distance_by_method(X, y_method, method_names, load_dir),
    }
    return report


def _markdown(report: Dict[str, Any]) -> str:
    ms = report["method_separability"]
    dc = report["direction_consistency"]
    lines = [
        "# Embedding metrics",
        "",
        f"- N embeddings: {report['n_embeddings']} (dim={report['embedding_dim']})",
        f"- Methods: {', '.join(report['methods'])}",
        f"- Dates: {report.get('dates')}",
        "",
        "## Method separability",
        f"- Mean cos → own method centroid: {ms.get('mean_within_centroid_cohesion')}",
        f"- Mean cos → other method centroids: {ms.get('mean_between_centroid_attraction')}",
        f"- Separability gap: {ms.get('separability_gap')}",
        f"- Centroid–centroid mean cosine: {ms.get('mean_between_centroid_cosine')}",
        f"- Note: {ms.get('note')}",
        "",
    ]
    for name, w in ms.get("within_method", {}).items():
        lines.append(
            f"- **{name}**: n={w['n']}, "
            f"own={w['mean_cosine_to_own_centroid']:.4f}, "
            f"other={w['mean_cosine_to_other_centroids']:.4f}, "
            f"pair={w['mean_pairwise_cosine']:.4f}, "
            f"‖e‖={w['mean_l2_norm']:.3f}"
        )
    lines += [
        "",
        "## Direction consistency (LabIR)",
        f"- Filter method: {dc.get('method_filter')}",
        f"- N directed: {dc.get('n')}  az bins={dc.get('n_azimuth_bins')}  "
        f"el bins={dc.get('n_elevation_bins')}",
        f"- Mean cos → own az centroid: {dc.get('mean_within_azimuth_cohesion')}",
        f"- Mean cos → other az centroids: {dc.get('mean_between_azimuth_attraction')}",
        f"- Direction gap: {dc.get('direction_gap')}",
        f"- Centroid–centroid mean az cosine: "
        f"{dc.get('mean_between_azimuth_centroid_cosine')}",
    ]
    if dc.get("error"):
        lines.append(f"- Note: {dc['error']}")
    nd = report.get("noise_distance") or {}
    if nd:
        lines += ["", "## Noise distance"]
        for k, v in nd.items():
            lines.append(
                f"- {k}: mean_cos_to_noise={v['mean_cosine_to_noise']:.4f} "
                f"(dist={v['mean_cosine_distance']:.4f}, n={v['n']})"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description="Embedding metrics for spatial methods")
    p.add_argument("--embeddings", default=None, help="Single embeddings directory")
    p.add_argument("--location", default=None)
    p.add_argument("--data-dir", default=ANALYSIS_OUTPUT)
    p.add_argument("--date", default=None, help="Comma-separated dates")
    p.add_argument(
        "--methods",
        default="bf_LabIR,bf_SPIR,sa,mono",
    )
    p.add_argument("--direction-method", default="bf_LabIR")
    p.add_argument(
        "--backends",
        default="legacy",
        help="When using --location: birdnet_native,bacpipe,legacy",
    )
    p.add_argument(
        "--bacpipe-models",
        default=None,
        help="Comma list under embeddings/bacpipe/ (with --backends bacpipe)",
    )
    p.add_argument(
        "--tag",
        default=None,
        help="Output filename tag (default: date or multi)",
    )
    p.add_argument("--output-dir", default=None)
    args = p.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    dates = [d.strip() for d in args.date.split(",")] if args.date else None
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    bp_models = (
        [m.strip() for m in args.bacpipe_models.split(",") if m.strip()]
        if args.bacpipe_models
        else None
    )

    # Direct path: --embeddings points at one model dir
    emb_dir = args.embeddings
    if emb_dir is None and args.location and backends == ["bacpipe"] and bp_models and len(bp_models) == 1:
        from embedding_schema import bacpipe_embeddings_dir

        emb_dir = bacpipe_embeddings_dir(args.data_dir, args.location, bp_models[0])

    report = run_metrics(
        emb_dir=emb_dir,
        location=args.location if emb_dir is None else None,
        data_dir=args.data_dir,
        methods=methods,
        date_filter=dates,
        direction_method=args.direction_method,
        backends=backends if emb_dir is None else None,
    )
    if bp_models and emb_dir:
        report["bacpipe_model"] = bp_models[0]
    report["embeddings_dir"] = emb_dir or args.embeddings

    loc = args.location or "unknown"
    out_dir = args.output_dir
    if not out_dir:
        if args.location:
            out_dir = audits_dir(args.data_dir, args.location)
        elif args.embeddings:
            out_dir = os.path.join(args.embeddings, "audits")
        else:
            out_dir = "."
    os.makedirs(out_dir, exist_ok=True)
    tag = args.tag or (dates[0] if dates and len(dates) == 1 else "multi")
    if bp_models and len(bp_models) == 1:
        tag = f"{tag}_bacpipe_{bp_models[0]}"
    elif emb_dir and "bacpipe" in (emb_dir or ""):
        tag = f"{tag}_" + os.path.basename(emb_dir.rstrip(os.sep))
    json_path = os.path.join(out_dir, f"{tag}_embedding_metrics.json")
    md_path = os.path.join(out_dir, f"{tag}_embedding_metrics.md")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
    md = _markdown(report)
    with open(md_path, "w") as f:
        f.write(md)
    print(md)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
