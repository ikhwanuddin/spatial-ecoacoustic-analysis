#!/usr/bin/env python3
"""
Spatial Embedding Clustering and Matched Pairwise Direction Evaluation.

Evaluates spatial bioacoustic audio embeddings (beamforming LabIR/SPIR vs. SA vs. Mono)
under a matched-window pairwise experimental design.

Key methodologies:
1. Matched-Window Direction Selection:
   Performs temporal alignment across recording windows (src, start_sec, end_sec)
   and selects the optimal beamforming direction theta*(t) that maximizes noise distance,
   eliminating the spatial sample imbalance and angular averaging bias.
2. High-Dimensional Clustering:
   Runs HDBSCAN directly on native high-dimensional embedding space using cosine metric,
   avoiding 2D projection density and distance distortion.
3. Non-Parametric Paired Hypothesis Testing:
   Runs Wilcoxon signed-rank test and computes effect sizes on delta = d_noise(BF) - d_noise(mono).
4. Interactive Standalone Visualisation:
   Generates a self-contained HTML dashboard with Plotly charts.
"""

from __future__ import annotations

import os

# Prevent GPFS NFS lock stalls on HPC clusters
os.environ["NUMBA_CACHE_DIR"] = f"/tmp/numba_{os.environ.get('USER', 'sea')}"
os.environ["NUMBA_DISABLE_JIT_CACHE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import html
import json
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from scipy.stats import wilcoxon
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

try:
    from config import ANALYSIS_OUTPUT
except ImportError:
    ANALYSIS_OUTPUT = "/rds/general/user/ri322/home/sea-data"

CACHE_SUFFIX = "_spatial_cache.json"


def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    elif isinstance(obj, (np.floating, float)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """L2-normalize rows of matrix X safely."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return X / norms


# ── Noise Reference ─────────────────────────────────────

def load_noise_embeddings(emb_dir: str) -> Optional[Dict[str, np.ndarray]]:
    """Load noise reference embeddings if available.

    Returns:
        Dict group_name -> (D,) normalized mean noise vector per group, or None.
    """
    if not os.path.isdir(emb_dir):
        return None
    noise_paths = [
        f for f in os.listdir(emb_dir)
        if f.startswith("noise_") and f.endswith("_embeddings.npy")
    ]
    if not noise_paths:
        return None

    noise_vectors: Dict[str, np.ndarray] = {}
    for fname in sorted(noise_paths):
        group = fname.replace("noise_", "").replace("_embeddings.npy", "")
        path = os.path.join(emb_dir, fname)
        emb = np.load(path).astype(np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if len(emb) == 0:
            continue
        # L2-normalize individual noise embeddings, then average and re-normalize
        emb_norm = l2_normalize(emb)
        mean_vec = np.mean(emb_norm, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        noise_vectors[group] = mean_vec.astype(np.float32)
        print(f"  noise_{group}: {len(emb)} embeddings -> mean vector (dim={len(mean_vec)})")

    return noise_vectors if noise_vectors else None


# ── Matched-Window Direction Selection ──────────────────

def align_and_select_matched_windows(
    X: np.ndarray,
    y_method: np.ndarray,
    method_names: List[str],
    flat_meta: List[Dict[str, Any]],
    noise_vectors: Optional[Dict[str, np.ndarray]] = None,
    selection_mode: str = "max_noise_distance",
) -> Dict[str, Any]:
    """Align temporal windows across methods and select the optimal beamforming direction.

    Eliminates spatial sample imbalance by selecting the steering angle theta*(t)
    with maximum separation from the noise prototype for each matched time-window.
    """
    method_map = {name: i for i, name in enumerate(method_names)}
    X_norm = l2_normalize(X)

    # Compute noise distance per point if noise reference exists
    noise_distances = np.zeros(len(X), dtype=np.float32)
    if noise_vectors:
        for name, i in method_map.items():
            noise_group = name.replace("bf_", "")
            nvec = noise_vectors.get(noise_group)
            if nvec is not None:
                mask = (y_method == i)
                if np.any(mask):
                    sims = np.dot(X_norm[mask], nvec)
                    noise_distances[mask] = 1.0 - sims

    # Group by temporal window: (src_recording, round(start_sec, 2), round(end_sec, 2))
    windows: Dict[Tuple[str, float, float], Dict[str, List[Dict[str, Any]]]] = {}
    for idx, meta in enumerate(flat_meta):
        m = meta.get("method")
        if not m or m not in method_map:
            continue
        wav = meta.get("wav", "")
        # Strip method-specific suffixes to obtain source recording prefix
        src = (
            meta.get("source_recording")
            or wav.split("_mono")[0].split("_sa")[0].split("_LabIR")[0].split("_SPIR")[0]
        )
        s = round(float(meta.get("start_sec", 0.0)), 2)
        e = round(float(meta.get("end_sec", 0.0)), 2)
        key = (src, s, e)
        if key not in windows:
            windows[key] = {}
        if m not in windows[key]:
            windows[key][m] = []
        windows[key][m].append({
            "idx": idx,
            "dist": float(noise_distances[idx]),
            "azimuth": meta.get("azimuth"),
            "elevation": meta.get("elevation"),
            "wav": wav,
            "meta": meta,
        })

    # Filter for fully matched windows where all required methods are present
    required_methods = [m for m in ["mono", "sa", "bf_LabIR", "bf_SPIR"] if m in method_map]
    matched_keys = [k for k, v in windows.items() if all(m in v for m in required_methods)]

    selected_indices: List[int] = []
    paired_records: List[Dict[str, Any]] = []
    beam_distribution: Dict[str, Dict[str, int]] = {
        "bf_LabIR": {"azimuths": {}, "elevations": {}},
        "bf_SPIR": {"azimuths": {}, "elevations": {}},
    }

    for key in sorted(matched_keys):
        mdata = windows[key]
        rec: Dict[str, Any] = {
            "source": key[0],
            "start_sec": key[1],
            "end_sec": key[2],
            "methods": {},
            "deltas_vs_mono": {},
        }

        # Mono and SA
        if "mono" in mdata:
            chosen_mono = mdata["mono"][0]
            selected_indices.append(chosen_mono["idx"])
            rec["methods"]["mono"] = chosen_mono
        if "sa" in mdata:
            chosen_sa = mdata["sa"][0]
            selected_indices.append(chosen_sa["idx"])
            rec["methods"]["sa"] = chosen_sa

        # Beamforming LabIR
        if "bf_LabIR" in mdata:
            chosen_lab = max(mdata["bf_LabIR"], key=lambda x: x["dist"])
            selected_indices.append(chosen_lab["idx"])
            rec["methods"]["bf_LabIR"] = chosen_lab
            az = str(chosen_lab.get("azimuth"))
            el = str(chosen_lab.get("elevation"))
            beam_distribution["bf_LabIR"]["azimuths"][az] = (
                beam_distribution["bf_LabIR"]["azimuths"].get(az, 0) + 1
            )
            beam_distribution["bf_LabIR"]["elevations"][el] = (
                beam_distribution["bf_LabIR"]["elevations"].get(el, 0) + 1
            )

        # Beamforming SPIR
        if "bf_SPIR" in mdata:
            chosen_spir = max(mdata["bf_SPIR"], key=lambda x: x["dist"])
            selected_indices.append(chosen_spir["idx"])
            rec["methods"]["bf_SPIR"] = chosen_spir
            az = str(chosen_spir.get("azimuth"))
            el = str(chosen_spir.get("elevation"))
            beam_distribution["bf_SPIR"]["azimuths"][az] = (
                beam_distribution["bf_SPIR"]["azimuths"].get(az, 0) + 1
            )
            beam_distribution["bf_SPIR"]["elevations"][el] = (
                beam_distribution["bf_SPIR"]["elevations"].get(el, 0) + 1
            )

        # Deltas vs mono
        if "mono" in rec["methods"]:
            mono_d = rec["methods"]["mono"]["dist"]
            for mname, mval in rec["methods"].items():
                if mname != "mono":
                    rec["deltas_vs_mono"][mname] = round(mval["dist"] - mono_d, 5)

        paired_records.append(rec)

    # Compute statistical summary
    n_windows = len(paired_records)
    summary_stats: Dict[str, Any] = {
        "n_matched_windows": n_windows,
        "n_total_matched_embeddings": len(selected_indices),
        "per_method": {},
        "hypothesis_tests": {},
        "beam_distribution": beam_distribution,
    }

    if n_windows > 0:
        for mname in required_methods:
            dists = [r["methods"][mname]["dist"] for r in paired_records if mname in r["methods"]]
            summary_stats["per_method"][mname] = {
                "mean_noise_distance": float(np.mean(dists)),
                "std_noise_distance": float(np.std(dists)),
                "median_noise_distance": float(np.median(dists)),
            }

        # Pairwise tests against mono
        if "mono" in required_methods:
            mono_dists = np.array([r["methods"]["mono"]["dist"] for r in paired_records])
            for mname in [m for m in required_methods if m != "mono"]:
                m_dists = np.array([r["methods"][mname]["dist"] for r in paired_records])
                deltas = m_dists - mono_dists
                win_count = int(np.sum(deltas > 0))
                win_rate = float(win_count / n_windows * 100.0)

                # Wilcoxon signed-rank test
                p_value = None
                if HAS_SCIPY and n_windows >= 5 and np.any(deltas != 0):
                    try:
                        res = wilcoxon(m_dists, mono_dists, alternative="greater")
                        p_value = float(res.pvalue)
                    except Exception:
                        p_value = None

                # Cliff's Delta effect size
                cdelta = float(np.mean(deltas > 0) - np.mean(deltas < 0))

                summary_stats["hypothesis_tests"][mname] = {
                    "mean_delta_vs_mono": float(np.mean(deltas)),
                    "median_delta_vs_mono": float(np.median(deltas)),
                    "win_count": win_count,
                    "win_rate_pct": round(win_rate, 1),
                    "wilcoxon_p_value": p_value,
                    "cliffs_delta": round(cdelta, 3),
                    "is_significant_p05": bool(p_value is not None and p_value < 0.05),
                }

    selected_indices_arr = np.array(selected_indices, dtype=np.int64)
    return {
        "matched_indices": selected_indices_arr,
        "paired_records": paired_records,
        "summary_stats": summary_stats,
    }


# ── High-Dimensional HDBSCAN ────────────────────────────

def run_hdbscan(
    X: np.ndarray,
    min_cluster_size: int = 5,
    min_samples: int = 3,
    metric: str = "cosine",
) -> np.ndarray:
    """Cluster X directly in native high-dimensional embedding space with HDBSCAN.

    Uses cosine distance metric on L2-normalized embeddings to preserve
    true acoustic geometry without 2D projection distortion.
    """
    X_norm = l2_normalize(X.astype(np.float64))
    cluster_size = min(min_cluster_size, max(2, len(X)))
    samples = max(1, min(min_samples, cluster_size // 2))

    try:
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=cluster_size,
            min_samples=samples,
            metric="euclidean",
            prediction_data=True,
        )
    except ImportError:
        from sklearn.cluster import HDBSCAN
        clusterer = HDBSCAN(
            min_cluster_size=cluster_size,
            min_samples=samples,
            metric="euclidean",
        )
    return clusterer.fit_predict(X_norm)


# ── 2D Projection (t-SNE / PCA / UMAP for Visualisation) ────────

def run_umap(
    X: np.ndarray,
    n_components: int = 2,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Reduce X to 2D strictly for visual rendering in HTML.

    Uses sklearn TSNE / PCA for fast, resilient non-linear projection on HPC clusters
    without numba threading stalls on GPFS shared filesystems.
    """
    n_pts = len(X)
    if n_pts < 4:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)

    perp = max(2, min(n_neighbors, (n_pts - 1) // 3))
    try:
        from sklearn.manifold import TSNE
        return TSNE(
            n_components=2,
            perplexity=perp,
            random_state=random_state,
            init="pca",
            learning_rate="auto",
        ).fit_transform(X)
    except Exception:
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X)


# ── Statistics & Shared Clusters ─────────────────────────

def compute_intra_method_cosine(
    embeddings: Dict[str, np.ndarray],
    method_names: List[str],
) -> List[float]:
    """Compute mean pairwise cosine similarity per method."""
    results = []
    for method in method_names:
        Xm = embeddings.get(method)
        if Xm is None or Xm.shape[0] < 2:
            results.append(0.0)
            continue
        X_norm = l2_normalize(Xm)
        n = X_norm.shape[0]
        sum_vec = np.sum(X_norm, axis=0)
        total = float(np.dot(sum_vec, sum_vec))
        mean_cos = (total - n) / (n * (n - 1))
        results.append(round(mean_cos, 4))
    return results


def compute_shared_cluster_analysis(
    X: np.ndarray,
    cluster_labels: np.ndarray,
    y_method: np.ndarray,
    method_names: List[str],
    flat_meta: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute shared cluster stratification and angular diversity."""
    unique_clusters = sorted(set(cluster_labels))
    cluster_info: List[Dict[str, Any]] = []
    
    tier_counts = {
        "tier_1_all_methods": 0,
        "tier_2_three_methods": 0,
        "tier_3_two_methods": 0,
        "tier_4_bf_only": 0,
        "tier_5_mono_sa_only": 0,
        "noise": int(np.sum(cluster_labels == -1)),
    }

    for cid in unique_clusters:
        if cid == -1:
            continue
        mask = (cluster_labels == cid)
        methods_present = {}
        azimuths = set()
        elevations = set()
        
        for i, m in enumerate(method_names):
            cnt = int(np.sum(mask & (y_method == i)))
            if cnt > 0:
                methods_present[m] = cnt

        # Metadata for cluster points
        c_meta = [flat_meta[idx] for idx in np.where(mask)[0]]
        for item in c_meta:
            if item.get("azimuth") is not None:
                azimuths.add(item["azimuth"])
            if item.get("elevation") is not None:
                elevations.add(item["elevation"])

        n_methods = len(methods_present)
        has_bf = ("bf_LabIR" in methods_present) or ("bf_SPIR" in methods_present)
        has_baseline = ("mono" in methods_present) or ("sa" in methods_present)

        if n_methods == 4:
            tier = 1
            tier_counts["tier_1_all_methods"] += 1
        elif n_methods == 3:
            tier = 2
            tier_counts["tier_2_three_methods"] += 1
        elif n_methods == 2:
            tier = 3
            tier_counts["tier_3_two_methods"] += 1
        elif has_bf and not has_baseline:
            tier = 4
            tier_counts["tier_4_bf_only"] += 1
        else:
            tier = 5
            tier_counts["tier_5_mono_sa_only"] += 1

        cluster_info.append({
            "cluster_id": int(cid),
            "size": int(np.sum(mask)),
            "tier": tier,
            "methods_present": methods_present,
            "n_methods": n_methods,
            "angular_diversity": {
                "unique_azimuths": len(azimuths),
                "azimuths": sorted(list(azimuths)),
                "unique_elevations": len(elevations),
                "elevations": sorted(list(elevations)),
            },
        })

    return {
        "summary": tier_counts,
        "clusters": cluster_info,
    }


def compute_stats(
    X: np.ndarray,
    y_method: np.ndarray,
    method_names: List[str],
    cluster_labels: np.ndarray,
    embeddings: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, Any]:
    """Compute overall clustering and per-method statistics."""
    total = len(X)
    n_noise = int(np.sum(cluster_labels == -1))
    unique_c = set(cluster_labels) - {-1}
    noise_pct = round(100.0 * n_noise / total, 1) if total > 0 else 0.0

    per_method = []
    emb_dict = embeddings or {}
    if not emb_dict and len(method_names) > 0:
        for i, m in enumerate(method_names):
            emb_dict[m] = X[y_method == i]

    intra_cos = compute_intra_method_cosine(emb_dict, method_names)

    for i, method in enumerate(method_names):
        mask = (y_method == i)
        n_pts = int(np.sum(mask))
        if n_pts == 0:
            continue
        c_sub = cluster_labels[mask]
        n_m_noise = int(np.sum(c_sub == -1))
        m_noise_pct = round(100.0 * n_m_noise / n_pts, 1) if n_pts > 0 else 0.0
        n_unique_c = len(set(c_sub) - {-1})
        per_method.append({
            "method": method,
            "n_embeddings": n_pts,
            "n_in_cluster": n_pts - n_m_noise,
            "n_noise": n_m_noise,
            "noise_pct": m_noise_pct,
            "n_unique_clusters": n_unique_c,
            "cosine_sim": intra_cos[i] if i < len(intra_cos) else 0.0,
        })

    per_cluster = []
    for cid in sorted(unique_c):
        mask = (cluster_labels == cid)
        by_method = {}
        for i, method in enumerate(method_names):
            by_method[method] = int(np.sum(mask & (y_method == i)))
        per_cluster.append({
            "cluster_id": int(cid),
            "size": int(np.sum(mask)),
            "by_method": by_method,
        })

    return {
        "overview": {
            "total_embeddings": total,
            "n_clusters": len(unique_c),
            "n_noise": n_noise,
            "noise_pct": noise_pct,
            "embedding_dim": int(X.shape[1]),
        },
        "per_method": per_method,
        "per_cluster": per_cluster,
        "snr_proxy": {
            "cosine_similarity": [
                {"method": pm["method"], "value": pm["cosine_sim"]}
                for pm in per_method
            ],
            "method_order": method_names,
        },
    }


def compute_noise_distance(
    X: np.ndarray,
    y_method: np.ndarray,
    method_names: List[str],
    cluster_labels: np.ndarray,
    noise_vectors: Dict[str, np.ndarray],
    shared_analysis: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute cosine distance from noise reference prototype for each method."""
    method_to_noise: Dict[str, str] = {}
    for method in method_names:
        noise_group = method.replace("bf_", "")
        if noise_group in noise_vectors:
            method_to_noise[method] = noise_group

    if not method_to_noise:
        return {"available": False}

    X_norm = l2_normalize(X)
    noise_sim = np.zeros(len(X), dtype=np.float32)

    for i, method in enumerate(method_names):
        mask = (y_method == i)
        if method in method_to_noise and np.any(mask):
            noise_vec = noise_vectors[method_to_noise[method]]
            noise_sim[mask] = np.dot(X_norm[mask], noise_vec)

    per_method = []
    for i, method in enumerate(method_names):
        mask = (y_method == i)
        if np.sum(mask) > 0:
            mean_sim = float(np.mean(noise_sim[mask]))
            per_method.append({
                "method": method,
                "mean_noise_sim": round(mean_sim, 4),
                "mean_noise_distance": round(1.0 - mean_sim, 4),
            })

    mono_entry = next((p for p in per_method if p["method"] == "mono"), None)
    if mono_entry:
        mono_dist = mono_entry["mean_noise_distance"]
        for p in per_method:
            p["delta_vs_mono"] = round(p["mean_noise_distance"] - mono_dist, 4)

    return {
        "available": True,
        "per_method": per_method,
    }


# ── Cache IO ─────────────────────────────────────────────

def save_cache(cache_path: str, emb_dir: str, umap_2d: np.ndarray,
               y_method: np.ndarray, method_names: List[str],
               cluster_labels: np.ndarray, stats: Dict[str, Any]):
    """Serialize analysis results to JSON."""
    cache = {
        "emb_dir": emb_dir,
        "umap_2d": umap_2d.tolist(),
        "method_labels": [int(x) for x in y_method],
        "method_names": method_names,
        "cluster_labels": [int(x) for x in cluster_labels],
        "stats": convert_numpy_types(stats),
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)
    print(f"  Saved cache to {cache_path}")


def load_cache(cache_path: str):
    """Load cached analysis results."""
    with open(cache_path, "r", encoding="utf-8") as f:
        cache = json.load(f)
    return (
        np.array(cache["umap_2d"]),
        np.array(cache["method_labels"]),
        cache["method_names"],
        np.array(cache["cluster_labels"]),
        cache["stats"],
        cache["emb_dir"],
    )
