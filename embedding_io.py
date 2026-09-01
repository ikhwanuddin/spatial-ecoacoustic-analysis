"""
Load dense embeddings from native BirdNET, bacpipe, or legacy flat layouts.

Filename convention:
  {date}_{method}.npy          on ephemeral scratch
  {date}_{method}_meta.json    in HOME, passed as *meta_dir*

The vectors and their metadata live on two different filesystems, so every
reader takes both directories.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from config import expected_beam_tags
from embedding_schema import (
    beam_tag_from_name,
    DEFAULT_METHODS,
    bacpipe_embeddings_dir,
    birdnet_embeddings_dir,
    embeddings_root,
    legacy_flat_embeddings_dir,
    meta_basename,
)


def _parse_date_method(fname: str) -> Optional[Tuple[str, str]]:
    if not fname.endswith(".npy"):
        return None
    if fname.startswith("noise_"):
        return None
    stem = fname[: -len(".npy")]
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return None
    date_str, method = parts[0], parts[1]
    # dates are YYYY-MM-DD
    if len(date_str) != 10 or date_str[4] != "-":
        return None
    return date_str, method


def list_embedding_files(
    emb_dir: str,
    methods: Optional[Sequence[str]] = None,
    date_filter: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Return (path, date, method) for embedding npy files in *emb_dir*."""
    if not os.path.isdir(emb_dir):
        return []
    methods_set = set(methods) if methods else None
    dates_set = set(date_filter) if date_filter else None
    out: List[Tuple[str, str, str]] = []
    for fname in sorted(os.listdir(emb_dir)):
        parsed = _parse_date_method(fname)
        if not parsed:
            continue
        date_str, method = parsed
        if methods_set is not None and method not in methods_set:
            continue
        if dates_set is not None and date_str not in dates_set:
            continue
        out.append((os.path.join(emb_dir, fname), date_str, method))
    return out


def load_embeddings_from_dir(
    emb_dir: str,
    methods: Optional[Sequence[str]] = None,
    date_filter: Optional[Sequence[str]] = None,
    source_tag: Optional[str] = None,
    meta_dir: Optional[str] = None,
    allowed_beams: Optional[Sequence[str]] = None,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, List[dict], List[str]]:
    """Load embeddings from a single directory (cluster_poc-compatible return).

    Returns:
        embeddings: method → (N, D)
        X: stacked (K, D)
        y_method: (K,) method index
        flat_meta: list of meta dicts (len K); injects source/backend if missing
        methods: ordered method names present
    """
    methods = list(methods) if methods else list(DEFAULT_METHODS)
    files = list_embedding_files(emb_dir, methods=methods, date_filter=date_filter)
    # The beamformed audio on disk can predate a configuration change, so rows
    # whose steering direction is no longer configured are dropped here rather
    # than quietly inflating every per-method statistic.
    allowed = expected_beam_tags() if allowed_beams is None else set(allowed_beams)
    dropped = 0

    all_emb: Dict[str, List[np.ndarray]] = {m: [] for m in methods}
    all_meta: Dict[str, List[dict]] = {m: [] for m in methods}
    dim: Optional[int] = None

    for emb_path, date_str, method in files:
        emb = np.load(emb_path).astype(np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        if dim is None:
            dim = int(emb.shape[1])
        meta_path = os.path.join(meta_dir or emb_dir, meta_basename(date_str, method))
        meta: List[dict] = []
        if os.path.isfile(meta_path):
            with open(meta_path, "r") as f:
                meta = json.load(f)
        # Align meta length loosely
        if len(meta) < len(emb):
            meta = meta + [{} for _ in range(len(emb) - len(meta))]
        elif len(meta) > len(emb):
            meta = meta[: len(emb)]
        for rec in meta:
            if source_tag and "backend" not in rec:
                rec = dict(rec)
                rec.setdefault("source_dir", source_tag)
            if source_tag:
                rec.setdefault("source_tag", source_tag)
            rec.setdefault("date", date_str)
            rec.setdefault("method", method)

        if allowed:
            keep = [
                i for i, rec in enumerate(meta)
                if (tag := beam_tag_from_name(rec.get("wav", ""), method)) is None
                or tag == method or tag in allowed
            ]
            if len(keep) < len(meta):
                dropped += len(meta) - len(keep)
                rows = np.asarray(keep, dtype=int)
                emb = emb[rows]
                meta = [meta[i] for i in keep]

        all_emb[method].append(emb)
        all_meta[method].extend(meta)
        print(f"  [{source_tag or emb_dir}] {os.path.basename(emb_path)}: {len(emb)}")

    if dropped:
        print(f"  [beam filter] dropped {dropped} embeddings whose beam is not in the current config")

    embeddings: Dict[str, np.ndarray] = {}
    d = dim or 1024
    for m in methods:
        if all_emb[m]:
            embeddings[m] = np.concatenate(all_emb[m], axis=0)
        else:
            embeddings[m] = np.zeros((0, d), dtype=np.float32)

    stacked: List[np.ndarray] = []
    method_ids: List[np.ndarray] = []
    flat_meta: List[dict] = []
    for i, m in enumerate(methods):
        n = len(embeddings[m])
        if n > 0:
            stacked.append(embeddings[m])
            method_ids.append(np.full(n, i, dtype=np.int16))
            flat_meta.extend(all_meta[m])

    if not stacked:
        return embeddings, np.zeros((0, d), dtype=np.float32), np.array([], dtype=np.int16), [], methods

    X = np.concatenate(stacked, axis=0).astype(np.float32)
    y_method = np.concatenate(method_ids)
    return embeddings, X, y_method, flat_meta, methods


def resolve_embedding_dirs(
    data_dir: str,
    location: str,
    *,
    embeddings_path: Optional[str] = None,
    backends: Optional[Sequence[str]] = None,
    bacpipe_models: Optional[Sequence[str]] = None,
) -> List[Tuple[str, str]]:
    """Return list of (directory, source_tag).

    backends: subset of {'birdnet_native', 'bacpipe', 'legacy'} or None = auto.
    If embeddings_path is set, only that directory is used.
    """
    if embeddings_path:
        tag = os.path.basename(os.path.rstrip(embeddings_path, os.sep)) or "custom"
        return [(embeddings_path, tag)]

    backends_set = set(backends) if backends else None
    out: List[Tuple[str, str]] = []

    def want(name: str) -> bool:
        return backends_set is None or name in backends_set

    bird_dir = birdnet_embeddings_dir(data_dir, location)
    if want("birdnet_native") and os.path.isdir(bird_dir) and list_embedding_files(bird_dir):
        out.append((bird_dir, "birdnet_native"))

    if want("bacpipe"):
        models = list(bacpipe_models or [])
        bac_root = os.path.join(embeddings_root(data_dir, location), "bacpipe")
        if models:
            for m in models:
                d = bacpipe_embeddings_dir(data_dir, location, m)
                if os.path.isdir(d) and list_embedding_files(d):
                    out.append((d, f"bacpipe:{m}"))
        elif os.path.isdir(bac_root):
            for m in sorted(os.listdir(bac_root)):
                d = os.path.join(bac_root, m)
                if os.path.isdir(d) and list_embedding_files(d):
                    out.append((d, f"bacpipe:{m}"))

    flat = legacy_flat_embeddings_dir(data_dir, location)
    if want("legacy") and os.path.isdir(flat) and list_embedding_files(flat):
        # Avoid double-count if birdnet/ is empty and all files are flat only
        if not any(tag == "birdnet_native" for _, tag in out):
            out.append((flat, "legacy_flat"))
        elif backends_set and "legacy" in backends_set:
            out.append((flat, "legacy_flat"))

    # Auto: if nothing nested, always try flat
    if not out and os.path.isdir(flat):
        out.append((flat, "legacy_flat"))

    return out


def load_location_embeddings(
    data_dir: str,
    location: str,
    *,
    methods: Optional[Sequence[str]] = None,
    date_filter: Optional[Sequence[str]] = None,
    embeddings_path: Optional[str] = None,
    backends: Optional[Sequence[str]] = None,
    bacpipe_models: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Load and stack embeddings across resolved backends.

    Returns dict with keys:
      X, y_method, methods, flat_meta, source_tags, per_source, dirs
    """
    dirs = resolve_embedding_dirs(
        data_dir,
        location,
        embeddings_path=embeddings_path,
        backends=backends,
        bacpipe_models=bacpipe_models,
    )
    methods = list(methods) if methods else list(DEFAULT_METHODS)

    Xs: List[np.ndarray] = []
    ys: List[np.ndarray] = []
    metas: List[dict] = []
    source_tags: List[str] = []
    per_source: Dict[str, Dict[str, Any]] = {}

    for emb_dir, tag in dirs:
        print(f"Loading embeddings from {emb_dir} ({tag})")
        emb_map, X, y, meta, meths = load_embeddings_from_dir(
            emb_dir, methods=methods, date_filter=date_filter, source_tag=tag
        )
        per_source[tag] = {
            "dir": emb_dir,
            "n": int(len(X)),
            "methods": {m: int(len(emb_map[m])) for m in meths if len(emb_map[m])},
        }
        if len(X) == 0:
            continue
        # Offset method indices only within shared methods list
        Xs.append(X)
        ys.append(y)
        for rec in meta:
            rec = dict(rec)
            rec["source_tag"] = tag
            metas.append(rec)
            source_tags.append(tag)

    if not Xs:
        return {
            "X": np.zeros((0, 1024), dtype=np.float32),
            "y_method": np.array([], dtype=np.int16),
            "methods": methods,
            "flat_meta": [],
            "source_tags": [],
            "per_source": per_source,
            "dirs": dirs,
        }

    dims = {x.shape[1] for x in Xs}
    if len(dims) > 1:
        raise ValueError(
            f"Cannot stack embeddings with different dims {sorted(dims)}. "
            "Load one model/backend at a time (e.g. only bacpipe:perch_bird "
            "or only birdnet_native/legacy BirdNET 1024-d)."
        )

    return {
        "X": np.concatenate(Xs, axis=0),
        "y_method": np.concatenate(ys, axis=0),
        "methods": methods,
        "flat_meta": metas,
        "source_tags": source_tags,
        "per_source": per_source,
        "dirs": dirs,
    }
