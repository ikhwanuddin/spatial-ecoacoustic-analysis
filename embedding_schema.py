"""
Shared embedding layout and metadata schema for the active pipeline.

Two roots. Bulk vectors stay on ephemeral scratch, which is wiped every 30
days; everything small and expensive to recompute is written to HOME.

Ephemeral, under ANALYSIS_OUTPUT:

  {location}/
    {date}/
      bf_LabIR/  bf_SPIR/  sa/  mono/     # method audio (full WAVs)
    emb/
      {model}/                            # e.g. birdnet, perch_bird
        {date}_{method}.npy
    emb_native/
      birdnet/                            # native BirdNET dense windows

HOME, under RESULTS_ROOT (default ~/sea-emb):

  {location}/
    {model}/
      {date}_{method}_meta.json
      {date}_summary.json
      noise_{group}_embeddings.npy
      noise_{group}_meta.json
    audits/                               # FP / silent-chunk reports

Meta JSON is a list of per-window dicts. Required keys (v1):
  wav, method, start_sec, end_sec, model, backend,
  window_sec, slide_sec
Optional:
  azimuth, elevation, embedding_dim
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1

BACKEND_BIRDNET = "birdnet_native"
BACKEND_BACPIPE = "bacpipe"

DEFAULT_METHODS = ("bf_LabIR", "bf_SPIR", "sa", "mono")

# Native BirdNET dense scan (extract_embeddings; signal processing is separate)
BIRDNET_WINDOW_SEC = 3.0
BIRDNET_SLIDE_SEC = 1.5
BIRDNET_EMBEDDING_DIM = 1024
BIRDNET_MODEL_ID = "birdnet"


# HOME root for the small, expensive-to-recompute outputs.
RESULTS_ROOT = os.environ.get(
    "SEA_RESULTS", os.path.join(os.path.expanduser("~"), "sea-emb")
)
DASHBOARDS_ROOT = os.environ.get(
    "SEA_DASHBOARDS", os.path.join(os.path.expanduser("~"), "sea-dashboards")
)


def embeddings_root(data_dir: str, location: str) -> str:
    """Ephemeral root holding the raw embedding vectors."""
    return os.path.join(data_dir, location, "emb")


def birdnet_embeddings_dir(data_dir: str, location: str) -> str:
    """Native BirdNET dense embeddings, kept apart from the bacpipe models."""
    return os.path.join(data_dir, location, "emb_native", "birdnet")


def bacpipe_embeddings_dir(data_dir: str, location: str, model: str) -> str:
    """Ephemeral dir holding one model's .npy vectors."""
    return os.path.join(embeddings_root(data_dir, location), model)


def results_root(location: str) -> str:
    """HOME root for metadata, summaries, noise references and audits."""
    return os.path.join(RESULTS_ROOT, location)


def bacpipe_meta_dir(location: str, model: str) -> str:
    """HOME dir holding one model's meta, summary and noise-reference files."""
    return os.path.join(results_root(location), model)


def audits_dir(location: str) -> str:
    return os.path.join(results_root(location), "audits")


def dashboards_dir(location: str, date_str: str) -> str:
    """HOME dir for one date's HTML dashboard set."""
    return os.path.join(DASHBOARDS_ROOT, location, date_str)


def embedding_basename(date_str: str, method: str) -> str:
    return f"{date_str}_{method}.npy"


def meta_basename(date_str: str, method: str) -> str:
    return f"{date_str}_{method}_meta.json"


def summary_basename(date_str: str) -> str:
    return f"{date_str}_summary.json"


def resolve_birdnet_out_dir(
    data_dir: str,
    location: str,
    embeddings_out: Optional[str] = None,
) -> str:
    """Default to embeddings/birdnet; honour explicit --embeddings-out."""
    if embeddings_out:
        return embeddings_out
    return birdnet_embeddings_dir(data_dir, location)


def legacy_flat_embeddings_dir(data_dir: str, location: str) -> str:
    """Pre-layout path: {location}/embeddings/*.npy (no birdnet/ subfolder)."""
    return embeddings_root(data_dir, location)


def make_window_meta(
    *,
    wav: str,
    method: str,
    start_sec: float,
    end_sec: float,
    model: str = BIRDNET_MODEL_ID,
    backend: str = BACKEND_BIRDNET,
    window_sec: float = BIRDNET_WINDOW_SEC,
    slide_sec: float = BIRDNET_SLIDE_SEC,
    embedding_dim: int = BIRDNET_EMBEDDING_DIM,
    azimuth: Optional[int] = None,
    elevation: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one per-window metadata record (schema v1)."""
    rec: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "wav": wav,
        "method": method,
        "start_sec": round(float(start_sec), 3),
        "end_sec": round(float(end_sec), 3),
        "model": model,
        "backend": backend,
        "window_sec": float(window_sec),
        "slide_sec": float(slide_sec),
        "embedding_dim": int(embedding_dim),
        "azimuth": azimuth,
        "elevation": elevation,
    }
    if extra:
        rec.update(extra)
    return rec


def find_embedding_sources(
    data_dir: str,
    location: str,
    prefer_birdnet_subdir: bool = True,
) -> List[str]:
    """Dirs that may contain {date}_{method}_embeddings.npy files.

    Prefer embeddings/birdnet/, fall back to flat embeddings/ for old runs.
    """
    dirs: List[str] = []
    nested = birdnet_embeddings_dir(data_dir, location)
    flat = legacy_flat_embeddings_dir(data_dir, location)
    if prefer_birdnet_subdir and os.path.isdir(nested):
        dirs.append(nested)
    if os.path.isdir(flat):
        # Only add flat if it actually holds .npy at top level (legacy)
        has_npy = any(
            f.endswith(".npy")
            for f in os.listdir(flat)
            if os.path.isfile(os.path.join(flat, f))
        )
        if has_npy and flat not in dirs:
            dirs.append(flat)
    if not dirs and os.path.isdir(nested):
        dirs.append(nested)
    return dirs
