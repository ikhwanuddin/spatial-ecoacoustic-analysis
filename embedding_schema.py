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
import re
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


# Local-time bins used to scope a noise reference to a time condition.
CONDITION_BINS = (("dawn", 5, 7), ("day", 7, 17), ("dusk", 17, 19))
CONDITIONS = ("dawn", "day", "dusk", "night")


def condition_of_hour(hour: int) -> str:
    for name, lo, hi in CONDITION_BINS:
        if lo <= hour < hi:
            return name
    return "night"


def condition_from_wav(name: str) -> Optional[str]:
    """Time condition from a recording name starting with HH-MM-SS."""
    base = os.path.basename(str(name))
    if len(base) < 2 or not base[:2].isdigit():
        return None
    hour = int(base[:2])
    return condition_of_hour(hour) if 0 <= hour <= 23 else None


BEAM_TAG_PATTERN = re.compile(r"(LabIR\(S\d{2}_\d{3}\)|SPIR[12]\([^)]*\))")


def beam_tag_from_name(name: str, method: Optional[str] = None) -> Optional[str]:
    """Steering direction encoded in a filename, e.g. 'LabIR(S01_000)'.

    mono and sa have no beam, so they fall back to their own group name.
    """
    match = BEAM_TAG_PATTERN.search(os.path.basename(str(name)))
    if match:
        return match.group(1)
    return noise_group_for_method(method) if method else None


def noise_group_for_method(method: str) -> str:
    """bf_LabIR -> LabIR, mono -> mono."""
    return method[3:] if method.startswith("bf_") else method


def noise_key(condition: Optional[str], group: str, date_str: Optional[str] = None) -> str:
    parts = [p for p in (date_str, condition) if p] + [group]
    return "_".join(parts)


def resolve_noise_vector(noise_vectors, condition, group, beam_tag=None, date_str=None):
    """Pick the reference for this window's date, condition, method and direction.

    A beam is measured against noise captured through that same beam. Most
    specific wins, so a reference built from the window's own date is preferred
    over one pooled across dates:
      1. <date>_<condition>_<beam>   same day, same time, same direction
      2. <date>_<condition>_<group>  same day, method pooled over its beams
      3. <condition>_<beam>          pooled across dates, same direction
      4. <condition>_<group>         pooled across dates and beams
      5. <group>                     reference predating condition scoping
    Returns (vector, key_used) or (None, None) — never a silent zero.
    """
    if not noise_vectors:
        return None, None
    candidates = []
    for d in ((date_str,) if date_str else ()) + (None,):
        if condition and beam_tag:
            candidates.append(noise_key(condition, beam_tag, d))
        if condition:
            candidates.append(noise_key(condition, group, d))
    candidates.append(group)
    for key in candidates:
        vec = noise_vectors.get(key)
        if vec is not None:
            return vec, key
    return None, None


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
