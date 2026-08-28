#!/usr/bin/env python3
"""
bacpipe multi-model embedding pilot on existing method WAVs.

Optimized for HPC execution (CUDA GPU acceleration for PyTorch, CPU fallback for TF, VRAM GC).
Does not run beamforming. Reads full WAVs under:
  {data_dir}/{location}/{date}/{method}/h_*/m_*/*.wav

Writes schema-aligned arrays under:
  {data_dir}/{location}/embeddings/bacpipe/{model}/

Usage:
  python experiments/bacpipe/run_pilot.py \
    --location 2A400 --date 2026-04-21 --models all --methods mono,sa,bf_LabIR,bf_SPIR --device auto
"""

from __future__ import annotations

# Fast local caches on HPC compute nodes to prevent GPFS lock stalls
import os
import sys

_USER = os.environ.get("USER", "ri322")
_EPHEM_BASE = f"/rds/general/user/{_USER}/ephemeral"
_EPHEM_TMP = f"{_EPHEM_BASE}/tmp"
_TMP_DIR = _EPHEM_TMP if os.path.isdir(_EPHEM_BASE) else f"/tmp/{_USER}"
try:
    os.makedirs(_TMP_DIR, exist_ok=True)
except Exception:
    pass

os.environ.setdefault("TMPDIR", _TMP_DIR)
os.environ.setdefault("TEMP", _TMP_DIR)
os.environ.setdefault("TMP", _TMP_DIR)
if os.path.isdir(_EPHEM_BASE):
    os.environ.setdefault("HF_HOME", f"{_EPHEM_BASE}/.cache/huggingface")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_EPHEM_BASE}/.cache/huggingface/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{_EPHEM_BASE}/.cache/huggingface/hub")
    os.environ.setdefault("TORCH_HOME", f"{_EPHEM_BASE}/.cache/torch")
os.environ.setdefault("NUMBA_CACHE_DIR", f"{_TMP_DIR}/numba")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"{_TMP_DIR}/torch_ext")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
os.environ.setdefault("ORT_DISABLE_THREAD_AFFINITY", "1")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("ORT_NUM_THREADS", "4")

# NVIDIA CUDA 12.6 NVVM / ptxas for TensorFlow XLA
_CUDA_HOME = os.environ.get("CUDA_HOME", "/rds/easybuild/noarch/apps/software/CUDA/12.6.0")
if os.path.isdir(f"{_CUDA_HOME}/bin"):
    os.environ["PATH"] = f"{_CUDA_HOME}/bin" + os.pathsep + os.environ.get("PATH", "")
if os.path.isfile(f"{_CUDA_HOME}/nvvm/libdevice/libdevice.10.bc"):
    _cur_xla = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir" not in _cur_xla:
        os.environ["XLA_FLAGS"] = f"{_cur_xla} --xla_gpu_cuda_data_dir={_CUDA_HOME}".strip()

# Canonical Ephemeral Storage
_USER = os.environ.get("USER", "ri322")
if "ANALYSIS_OUTPUT" not in os.environ:
    os.environ["ANALYSIS_OUTPUT"] = f"/rds/general/user/{_USER}/ephemeral/sea-work"
if "MONITORING_DATA" not in os.environ:
    os.environ["MONITORING_DATA"] = f"/rds/general/user/{_USER}/ephemeral/monitoring_data"

import argparse
import gc
import gzip
import importlib
import json
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Repo root on path when run as script
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import ANALYSIS_OUTPUT, LOCATION_MAP, RPIID_TO_LOCATION  # noqa: E402
from embedding_schema import (  # noqa: E402
    BACKEND_BACPIPE,
    DEFAULT_METHODS,
    bacpipe_embeddings_dir,
    embedding_basename,
    make_window_meta,
    meta_basename,
    summary_basename,
)

from direction_meta import parse_direction_metadata as _parse_direction_metadata


# ─────────────────────────────────────────────────────────────────────────────
# DEVICE & FRAMEWORK UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_TF_MODELS = {
    "perch_bird",
    "surfperch",
    "google_whale",
    "vggish",
    "birdnet",
}

# TF SavedModels that compile and run on CUDA once ptxas is available.
TF_GPU_MODELS = {
    "perch_bird",
    "surfperch",
    "vggish",
    "birdnet",
}

# Small / fragile TF graphs kept on CPU (XLA/JIT crash risk, little GPU gain).
TF_CPU_ONLY_MODELS = {
    "google_whale",
}


def _model_target_device(model: str, requested_device: str) -> str:
    """Route each model to CUDA when it can actually use the GPU."""
    if requested_device != "cuda":
        return requested_device

    name = model.lower()
    if name in TF_GPU_MODELS:
        print(f"  [Device Router] Model '{model}' is TensorFlow SavedModel -> CUDA")
        return "cuda"

    cpu_set = set(TF_CPU_ONLY_MODELS)
    try:
        import bacpipe
        if hasattr(bacpipe, "TF_MODELS"):
            cpu_set.update(name for name in bacpipe.TF_MODELS if name not in TF_GPU_MODELS)
    except Exception:
        pass

    if name in cpu_set:
        print(f"  [Device Router] Model '{model}' is TensorFlow-based -> CPU")
        return "cpu"

    print(f"  [Device Router] Model '{model}' is PyTorch/ONNX-based -> CUDA")
    return "cuda"


def _configure_frameworks(device: str) -> None:
    """Configure TensorFlow & PyTorch for safe, performant execution."""
    try:
        import tensorflow as tf
        try:
            tf.config.optimizer.set_jit(False)
        except Exception:
            pass
        if device == "cuda":
            gpus = tf.config.list_physical_devices("GPU")
            if gpus:
                for gpu in gpus:
                    try:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    except RuntimeError:
                        pass
                try:
                    tf.config.set_soft_device_placement(True)
                except Exception:
                    pass
        elif device == "cpu":
            try:
                tf.config.set_visible_devices([], "GPU")
            except Exception:
                pass
    except Exception:
        pass


def _resolve_location(location: str) -> str:
    rpiid = LOCATION_MAP.get(location, location)
    return RPIID_TO_LOCATION.get(rpiid, rpiid)


def _model_name_candidates(value: Any) -> List[str]:
    """Extract plausible model names from a bacpipe registry value."""
    if isinstance(value, dict):
        value = list(value.keys())
    if isinstance(value, (set, frozenset)):
        value = list(value)
    if isinstance(value, str):
        value = re.split(r"[,\s]+", value)
    if not isinstance(value, (list, tuple)):
        return []
    names = []
    for item in value:
        if isinstance(item, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", item):
            names.append(item)
    return names


# Core curated avian and bioacoustic foundation models in bacpipe
CURATED_BIRD_MODELS = [
    "birdnet",
    "birdnet_v3",
    "avesecho_passt",
    "perch_bird",
    "perch_v2",
    "biolingual",
    "birdaves_especies",
    "birdmae",
    "audioprotopnet",
    "protoclr",
    "vggish",
]


def discover_models() -> List[str]:
    """Return model names exposed by the installed bacpipe package."""
    try:
        import bacpipe
    except ImportError:
        return list(CURATED_BIRD_MODELS)

    modules = [bacpipe]
    for sub in ("feature_extractors", "models", "registry"):
        try:
            modules.append(importlib.import_module(f"bacpipe.{sub}"))
        except ImportError:
            pass

    found: List[str] = []
    for mod in modules:
        for attr in ("MODELS", "FEATURE_EXTRACTORS", "AVAILABLE_MODELS", "MODEL_REGISTRY"):
            if hasattr(mod, attr):
                for name in _model_name_candidates(getattr(mod, attr)):
                    if name not in found:
                        found.append(name)

    # Fallback to curated list if discovery returns nothing
    if not found:
        return list(CURATED_BIRD_MODELS)
    return found


def _resolve_models(models_arg: List[str]) -> List[str]:
    """Resolve comma-separated strings and the 'all' keyword."""
    expanded: List[str] = []
    for item in models_arg:
        for name in item.split(","):
            name = name.strip()
            if not name:
                continue
            if name.lower() in ("all", "curated", "bird_models"):
                for discovered in discover_models():
                    if discovered not in expanded:
                        expanded.append(discovered)
            elif name not in expanded:
                expanded.append(name)
    return expanded


def _resolve_device(requested: str) -> str:
    if requested in ("cuda", "mps", "cpu"):
        return requested
    try:
        import torch

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            print(f"[Device Auto-Detect] CUDA Active: {gpu_name} ({vram_gb:.1f} GB VRAM)")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("[Device Auto-Detect] Apple Silicon Metal (MPS) Active")
            return "mps"
    except ImportError:
        pass
    print("[Device Auto-Detect] CPU Active")
    return "cpu"


def find_noise_wavs(
    data_dir: str,
    location: str,
    noise_dir: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Return (wav_path, wav_name, noise_group) from noise_references."""
    root = Path(noise_dir) if noise_dir else Path(data_dir) / location / "noise_references"
    if not root.is_dir():
        return []
    found: List[Tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.wav")):
        if path.name.startswith("._"):
            continue
        rel = path.relative_to(root)
        group = rel.parts[0] if len(rel.parts) > 1 else "unknown"
        group = {"LabIR": "LabIR", "SPIR": "SPIR", "sa": "sa", "mono": "mono"}.get(
            group, group
        )
        found.append((str(path), path.name, group))
    return found


def _noise_group_for_method(method: str) -> str:
    return method.replace("bf_", "", 1)


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X.astype(np.float32), axis=1, keepdims=True)
    return X.astype(np.float32) / np.maximum(norms, 1e-9)


def _score_noise_distance(
    by_method_emb: Dict[str, Any],
    noise_embeddings: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Score each method against its own model-space noise reference."""
    distances: Dict[str, Dict[str, Any]] = {}
    for method, chunks in sorted(by_method_emb.items()):
        group = _noise_group_for_method(method)
        noise = noise_embeddings.get(group)
        if noise is None or (not chunks if isinstance(chunks, list) else chunks.size == 0):
            distances[method] = {"noise_group": group, "status": "missing_noise_reference"}
            continue
        if isinstance(chunks, np.ndarray):
            X = chunks.astype(np.float32)
        else:
            X = np.concatenate(chunks, axis=0).astype(np.float32)
        noise_mean = _l2_normalize_rows(noise).mean(axis=0)
        noise_mean /= max(float(np.linalg.norm(noise_mean)), 1e-9)
        cosine = _l2_normalize_rows(X) @ noise_mean
        distances[method] = {
            "noise_group": group,
            "status": "ok",
            "n": int(len(X)),
            "mean_cosine_to_noise": float(np.mean(cosine)),
            "mean_noise_distance": float(1.0 - np.mean(cosine)),
            "std_noise_distance": float(1.0 - cosine).std(),
        }

    mono = distances.get("mono", {})
    mono_distance = mono.get("mean_noise_distance") if mono.get("status") == "ok" else None
    for method, result in distances.items():
        if result.get("status") == "ok" and mono_distance is not None:
            result["delta_vs_mono"] = float(result["mean_noise_distance"] - mono_distance)
        elif method != "mono":
            result["delta_vs_mono"] = None
    return distances


def find_method_wavs(
    data_dir: str,
    location: str,
    date_str: str,
    methods: List[str],
    max_wavs: int = 0,
    max_wavs_per_method: int = 0,
) -> List[Tuple[str, str, str]]:
    """Return list of (wav_path, wav_name, method)."""
    date_dir = os.path.join(data_dir, location, date_str)
    found: List[Tuple[str, str, str]] = []
    for method in methods:
        method_dir = os.path.join(date_dir, method)
        if not os.path.isdir(method_dir):
            continue
        method_files: List[Tuple[str, str, str]] = []
        for root, _dirs, files in os.walk(method_dir):
            for fname in sorted(files):
                if not fname.lower().endswith(".wav"):
                    continue
                if fname.startswith("._"):
                    continue
                method_files.append((os.path.join(root, fname), fname, method))
        if max_wavs_per_method and len(method_files) > max_wavs_per_method:
            method_files = method_files[:max_wavs_per_method]
        found.extend(method_files)
    if max_wavs and len(found) > max_wavs:
        found = found[:max_wavs]
    return found


def _normalize_embeddings(embeddings: Any) -> np.ndarray:
    """Coerce bacpipe return value to float32 (N, D)."""
    if embeddings is None:
        return np.zeros((0, 0), dtype=np.float32)
    if isinstance(embeddings, dict):
        parts = []
        for v in embeddings.values():
            arr = np.asarray(v, dtype=np.float32)
            if arr.size == 0:
                continue
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            elif arr.ndim > 2:
                arr = arr.reshape(arr.shape[0], -1)
            parts.append(arr)
        if not parts:
            return np.zeros((0, 0), dtype=np.float32)
        return np.concatenate(parts, axis=0).astype(np.float32)
    emb = np.asarray(embeddings, dtype=np.float32)
    if emb.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if emb.ndim == 1:
        emb = emb.reshape(1, -1)
    elif emb.ndim > 2:
        emb = emb.reshape(emb.shape[0], -1)
    return emb


def _model_window_sec(em: Any) -> Tuple[Any, Any]:
    """Best-effort (window_sec, slide_sec) from bacpipe model object."""
    window_sec: Any = "model_native"
    slide_sec: Any = "model_native"
    model = getattr(em, "model", None)
    if model is None:
        return window_sec, slide_sec
    for attr in ("segment_length", "sample_duration", "duration", "clip_length"):
        if hasattr(model, attr):
            try:
                val = getattr(model, attr)
                if val is not None:
                    window_sec = float(val)
                    break
            except (TypeError, ValueError):
                pass
    try:
        sr = getattr(model, "sr", None) or getattr(model, "sample_rate", None)
        seg = getattr(model, "segment_length", None) or getattr(
            model, "num_samples", None
        )
        if sr and seg and float(sr) > 100:
            window_sec = float(seg) / float(sr)
    except Exception:
        pass
    for attr in ("hop_length", "hop_size", "stride"):
        if hasattr(model, attr):
            try:
                hop = getattr(model, attr)
                sr = getattr(model, "sr", None) or getattr(model, "sample_rate", None)
                if hop is not None and sr and float(sr) > 0:
                    slide_sec = float(hop) / float(sr) if float(hop) > 20 else float(hop)
                    break
            except (TypeError, ValueError):
                pass
    if isinstance(window_sec, (int, float)) and not isinstance(slide_sec, (int, float)):
        slide_sec = float(window_sec)
    return window_sec, slide_sec


_DEFAULT_CKPT = Path(__file__).resolve().parent / "checkpoints"


def _ensure_checkpoints(models: List[str], checkpoint_dir: Path) -> Path:
    """Download bacpipe model checkpoints if missing (HuggingFace)."""
    import bacpipe

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    try:
        bacpipe.settings.model_base_path = str(checkpoint_dir)
    except Exception:
        pass
    print(f"Ensuring checkpoints in {checkpoint_dir} for {models} …")
    bacpipe.ensure_models_exist(checkpoint_dir, models)
    return checkpoint_dir


def _get_embedder(
    model: str,
    device: str,
    cache: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
):
    """Reuse one Embedder per model (checkpoint load is expensive)."""
    if model in cache:
        return cache[model]
    import bacpipe

    target_dev = _model_target_device(model, device)
    if target_dev == "cpu" and model.lower() in TF_CPU_ONLY_MODELS:
        try:
            import tensorflow as tf
            tf.config.set_visible_devices([], "GPU")
        except Exception:
            pass
    ckpt = Path(checkpoint_dir) if checkpoint_dir else _DEFAULT_CKPT
    try:
        bacpipe.settings.device = target_dev
        bacpipe.settings.model_base_path = str(ckpt)
        bacpipe.settings.run_pretrained_classifier = False
    except Exception:
        pass
    try:
        em = bacpipe.Embedder(model, run_pretrained_classifier=False)
    except TypeError:
        em = bacpipe.Embedder(model)
    if hasattr(em, "model"):
        if hasattr(em.model, "bool_classifier"):
            em.model.bool_classifier = False
        if hasattr(em.model, "run_pretrained_classifier"):
            em.model.run_pretrained_classifier = False
    cache[model] = em
    return em


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK STREAMING & CHECKPOINT HELPERS (LAYER 2)
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_SIZE = 50  # Process and stream to disk in chunks of N WAVs (keeps RAM < 350MB)


def _model_already_done(out_dir: str, date_str: str, methods: List[str]) -> bool:
    """Layer 1: True if all outputs for this model are already complete on disk."""
    summary_path = os.path.join(out_dir, summary_basename(date_str))
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        done_methods = set(summary.get("methods", {}).keys())
        if not set(methods).issubset(done_methods):
            print(f"  [Ckpt-L1] Summary exists but methods are incomplete: "
                  f"{set(methods) - done_methods}")
            return False
        for method in methods:
            emb_path = os.path.join(out_dir, embedding_basename(date_str, method))
            if not os.path.exists(emb_path):
                return False
        return True
    except (json.JSONDecodeError, OSError):
        return False


def _expand_window_meta(
    record: dict,
    model: str,
) -> List[dict]:
    """Deterministically expand one compact WAV-level record into N window metadata dicts."""
    wav = record["wav"]
    method = record["method"]
    n_win = record["n_win"]
    win = record["win"]
    hop = record["hop"]
    azimuth = record.get("az")
    elevation = record.get("el")
    dim = record.get("dim", 0)
    meta = []
    for i in range(n_win):
        start = float(i * hop)
        end = float(start + win)
        meta.append(
            make_window_meta(
                wav=wav,
                method=method,
                start_sec=start,
                end_sec=end,
                model=model,
                backend=BACKEND_BACPIPE,
                window_sec=float(win),
                slide_sec=float(hop),
                embedding_dim=int(dim),
                azimuth=azimuth,
                elevation=elevation,
                extra={"wav_path": record.get("path")},
            )
        )
    return meta


def _save_chunk(
    ckpt_dir: str,
    date_str: str,
    chunk_idx: int,
    buffer_emb: Dict[str, List[np.ndarray]],
    buffer_meta: List[dict],
    processed_count: int,
) -> str:
    """Save one compact compressed FP16 chunk file and update state."""
    if not buffer_meta or not any(buffer_emb.values()):
        print(f"  [Ckpt-L2] Buffer is empty, skipping chunk #{chunk_idx}")
        return ""

    os.makedirs(ckpt_dir, exist_ok=True)
    chunk_filename = f"{date_str}_chunk_{chunk_idx:04d}.npz"
    chunk_path = os.path.join(ckpt_dir, chunk_filename)
    tmp_path = os.path.join(ckpt_dir, f".tmp_{chunk_filename}")

    emb_fp16 = {}
    for m, arr_list in buffer_emb.items():
        if arr_list:
            emb_fp16[m] = np.concatenate(arr_list, axis=0).astype(np.float16)

    meta_json_gz = gzip.compress(json.dumps(buffer_meta).encode("utf-8"))
    np.savez_compressed(
        tmp_path,
        meta_gz=np.frombuffer(meta_json_gz, dtype=np.uint8),
        **emb_fp16,
    )
    os.replace(tmp_path, chunk_path)

    # Update state.json
    state_path = os.path.join(ckpt_dir, f"{date_str}_state.json")
    state = {
        "date": date_str,
        "last_n": processed_count,
        "last_chunk_idx": chunk_idx,
        "methods": list(buffer_emb.keys()),
    }
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)

    sz_mb = os.path.getsize(chunk_path) / (1024**2)
    print(f"  [Ckpt-L2] Chunk #{chunk_idx} saved ({processed_count} WAVs total, {sz_mb:.1f} MB) — memory cleared")
    return chunk_path


def _load_progress(
    out_dir: str,
    date_str: str,
    model: str,
) -> Tuple[int, int]:
    """Layer 2: Check for existing progress. Returns (start_wav_idx, next_chunk_idx).
    Validates chunk files and cleans up any corrupt/empty shards automatically."""
    ckpt_dir = os.path.join(out_dir, ".ckpt")
    if not os.path.isdir(ckpt_dir):
        return 0, 0

    chunk_files = sorted(Path(ckpt_dir).glob(f"{date_str}_chunk_*.npz"))
    valid_chunks = []
    total_wavs = 0
    for cp in chunk_files:
        if cp.stat().st_size < 1024:  # Corrupt / empty stub
            print(f"  [Ckpt-L2] Removing invalid/empty chunk shard: {cp.name}")
            try:
                cp.unlink()
            except OSError:
                pass
            continue
        try:
            with np.load(cp) as l:
                if "meta_gz" in l:
                    meta = json.loads(gzip.decompress(l["meta_gz"].tobytes()).decode("utf-8"))
                    if len(meta) > 0:
                        valid_chunks.append(cp)
                        unique_wavs = len(set(r["wav"] for r in meta if "wav" in r))
                        total_wavs += unique_wavs
        except Exception as e:
            print(f"  [Ckpt-L2] Removing unreadable chunk shard {cp.name}: {e}")
            try:
                cp.unlink()
            except OSError:
                pass

    if valid_chunks:
        next_chunk_idx = len(valid_chunks)
        state_path = os.path.join(ckpt_dir, f"{date_str}_state.json")
        state = {
            "date": date_str,
            "last_n": total_wavs,
            "last_chunk_idx": next_chunk_idx - 1,
        }
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        print(f"  [Ckpt-L2] Resuming from WAV #{total_wavs + 1} ({len(valid_chunks)} valid chunks on disk)")
        return total_wavs, next_chunk_idx

    return 0, 0


# ─────────────────────────────────────────────────────────────────────────────


def _embed_with_bacpipe(
    model: str,
    wav_path: str,
    wav_name: str,
    method: str,
    device: str,
    embedder_cache: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Run one bacpipe Embedder on a single WAV. Returns (embeddings, compact_meta_record)."""
    cache = embedder_cache if embedder_cache is not None else {}
    em = _get_embedder(model, device, cache, checkpoint_dir=checkpoint_dir)

    is_tf = model.lower() in KNOWN_TF_MODELS
    if is_tf:
        embeddings = em.get_embeddings_from_model(wav_path)
    else:
        try:
            import torch
            with torch.inference_mode():
                embeddings = em.get_embeddings_from_model(wav_path)
        except Exception:
            embeddings = em.get_embeddings_from_model(wav_path)

    emb = _normalize_embeddings(embeddings)
    if emb.size == 0:
        return emb, None

    n, dim = emb.shape
    window_sec, slide_sec = _model_window_sec(em)
    azimuth, elevation = _parse_direction_metadata(wav_name)
    hop = slide_sec if isinstance(slide_sec, (int, float)) else (
        window_sec if isinstance(window_sec, (int, float)) else 1.0
    )
    win = window_sec if isinstance(window_sec, (int, float)) else float(hop)

    # 1 lightweight record per WAV (Lazy Metadata) instead of 80 dict objects
    record = {
        "wav": wav_name,
        "method": method,
        "n_win": int(n),
        "win": float(win),
        "hop": float(hop),
        "dim": int(dim),
        "az": azimuth,
        "el": elevation,
        "path": wav_path,
    }
    return emb, record


def run_pilot(
    *,
    location: str,
    date_str: str,
    models: List[str],
    methods: List[str],
    data_dir: str = ANALYSIS_OUTPUT,
    device: str = "auto",
    max_wavs: int = 0,
    max_wavs_per_method: int = 0,
    dry_run: bool = False,
    checkpoint_dir: Optional[Path | str] = None,
    noise_dir: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_loc = _resolve_location(location)
    resolved_models = _resolve_models(models)
    resolved_device = _resolve_device(device)
    ckpt_path = Path(checkpoint_dir) if checkpoint_dir else _DEFAULT_CKPT

    _configure_frameworks(resolved_device)

    method_wavs = find_method_wavs(
        data_dir=data_dir,
        location=resolved_loc,
        date_str=date_str,
        methods=methods,
        max_wavs=max_wavs,
        max_wavs_per_method=max_wavs_per_method,
    )
    noise_wavs = find_noise_wavs(
        data_dir=data_dir, location=resolved_loc, noise_dir=noise_dir
    )

    report: Dict[str, Any] = {
        "location": resolved_loc,
        "date": date_str,
        "device": resolved_device,
        "methods": methods,
        "comparator_methods": [m for m in methods if m != "mono"],
        "n_wavs": len(method_wavs),
        "n_noise_wavs": len(noise_wavs),
        "models": {},
    }

    print(
        f"bacpipe pilot: {resolved_loc} / {date_str} (Requested Device: {resolved_device.upper()}) — "
        f"{len(method_wavs)} method WAVs across {methods}, "
        f"{len(noise_wavs)} noise WAVs, {len(resolved_models)} models: {resolved_models}"
    )

    if dry_run:
        print("Dry run requested. Discovery complete.")
        return report

    if not method_wavs:
        print(f"Warning: No WAVs found in {data_dir}/{resolved_loc}/{date_str} for {methods}")
        return report

    _ensure_checkpoints(resolved_models, ckpt_path)

    for model_idx, model in enumerate(resolved_models, 1):
        target_dev = _model_target_device(model, resolved_device)
        out_dir = bacpipe_embeddings_dir(data_dir, resolved_loc, model)
        os.makedirs(out_dir, exist_ok=True)
        ckpt_dir = os.path.join(out_dir, ".ckpt")

        # ── LAYER 1: Skip already completed models ──────────────────────────
        if _model_already_done(out_dir, date_str, methods):
            print(f"\n[{model_idx}/{len(resolved_models)}] ✅ SKIP {model} "
                  f"— output already complete in {out_dir}")
            sum_path = os.path.join(out_dir, summary_basename(date_str))
            try:
                with open(sum_path, encoding="utf-8") as f:
                    report["models"][model] = json.load(f)
            except Exception:
                pass
            continue
        # ────────────────────────────────────────────────────────────────────

        print(f"\n[{model_idx}/{len(resolved_models)}] Model: {model} (Target Device: {target_dev.upper()})")
        t0 = time.time()

        embedder_cache: Dict[str, Any] = {}
        errors: List[dict] = []

        # ── Fail-fast: pre-initialize embedder once before WAV loop ─────────
        try:
            _get_embedder(model, target_dev, embedder_cache, checkpoint_dir=ckpt_path)
        except Exception as e:
            print(f"  [ERROR] Failed to initialize model '{model}': {e}")
            errors.append({"file": "INITIALIZATION", "method": "init", "error": str(e)})
            report["models"][model] = {
                "out_dir": out_dir,
                "elapsed_sec": time.time() - t0,
                "error_count": 1,
                "errors": errors,
            }
            continue
        # ────────────────────────────────────────────────────────────────────

        # ── LAYER 2: Load previous progress (Incremental Chunk Streaming) ───
        start_idx, next_chunk_idx = _load_progress(out_dir, date_str, model)
        buffer_emb: Dict[str, List[np.ndarray]] = {}
        buffer_meta: List[dict] = []
        chunk_idx = next_chunk_idx
        # ────────────────────────────────────────────────────────────────────

        for i, (wav_path, wav_name, method) in enumerate(method_wavs[start_idx:], start_idx + 1):
            try:
                emb, record = _embed_with_bacpipe(
                    model=model,
                    wav_path=wav_path,
                    wav_name=wav_name,
                    method=method,
                    device=target_dev,
                    embedder_cache=embedder_cache,
                    checkpoint_dir=ckpt_path,
                )
                if emb.size > 0 and record is not None:
                    buffer_emb.setdefault(method, []).append(emb)
                    buffer_meta.append(record)

                # ── Save compact chunk every _CHUNK_SIZE WAVs & purge RAM ───
                if i % _CHUNK_SIZE == 0:
                    _save_chunk(ckpt_dir, date_str, chunk_idx, buffer_emb, buffer_meta, i)
                    chunk_idx += 1
                    buffer_emb.clear()
                    buffer_meta.clear()
                    gc.collect()
                    try:
                        import torch
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except Exception:
                        pass
                # ────────────────────────────────────────────────────────────

                if i % 25 == 0 or i == len(method_wavs):
                    print(f"  Processed {i}/{len(method_wavs)} WAVs ... ({time.time()-t0:.1f}s)")
            except Exception as e:
                errors.append({"file": wav_name, "method": method, "error": str(e)})
                print(f"  [ERROR] {model} on {wav_name}: {e}")

        # Save any trailing items in buffer as final chunk
        if any(buffer_emb.values()):
            _save_chunk(ckpt_dir, date_str, chunk_idx, buffer_emb, buffer_meta, len(method_wavs))
            chunk_idx += 1
            buffer_emb.clear()
            buffer_meta.clear()
            gc.collect()

        # Process noise references
        noise_embeddings: Dict[str, np.ndarray] = {}
        noise_meta: Dict[str, List[dict]] = {}
        noise_errors: List[dict] = []
        for wav_path, wav_name, noise_group in noise_wavs:
            try:
                emb, record = _embed_with_bacpipe(
                    model=model,
                    wav_path=wav_path,
                    wav_name=wav_name,
                    method=f"noise_{noise_group}",
                    device=target_dev,
                    embedder_cache=embedder_cache,
                    checkpoint_dir=ckpt_path,
                )
                if emb.size == 0 or record is None:
                    continue
                if noise_group not in noise_embeddings:
                    noise_embeddings[noise_group] = emb.astype(np.float32)
                else:
                    noise_embeddings[noise_group] = np.concatenate(
                        [noise_embeddings[noise_group], emb.astype(np.float32)], axis=0
                    )
                noise_meta.setdefault(noise_group, []).extend(_expand_window_meta(record, model))
            except Exception as e:
                noise_errors.append({"file": wav_name, "group": noise_group, "error": str(e)})

        model_summary: Dict[str, Any] = {
            "out_dir": out_dir,
            "errors": errors,
            "noise_errors": noise_errors,
            "methods": {},
            "noise_references": {},
            "noise_distance": {},
            "elapsed_sec": round(time.time() - t0, 1),
        }

        # ── Assemble all chunks into final .npy & expand lazy metadata ──────
        all_method_embs: Dict[str, List[np.ndarray]] = {}
        all_method_meta: Dict[str, List[dict]] = {}

        chunk_files = sorted(Path(ckpt_dir).glob(f"{date_str}_chunk_*.npz"))
        print(f"  [Assembly] Streaming {len(chunk_files)} chunks into final outputs...")
        for cp in chunk_files:
            with np.load(cp) as l:
                rec_meta = json.loads(gzip.decompress(l["meta_gz"].tobytes()).decode("utf-8"))
                for rec in rec_meta:
                    m = rec["method"]
                    all_method_meta.setdefault(m, []).extend(_expand_window_meta(rec, model))
                for k in l.files:
                    if k != "meta_gz":
                        all_method_embs.setdefault(k, []).append(l[k])

        for method, chunks in sorted(all_method_embs.items()):
            all_emb = np.concatenate(chunks, axis=0).astype(np.float32)
            meta = all_method_meta.get(method, [])
            emb_path = os.path.join(out_dir, embedding_basename(date_str, method))
            meta_path = os.path.join(out_dir, meta_basename(date_str, method))
            np.save(emb_path, all_emb)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            model_summary["methods"][method] = {
                "n_embeddings": int(len(all_emb)),
                "embedding_dim": int(all_emb.shape[1]) if all_emb.ndim == 2 else 0,
                "embeddings_file": os.path.basename(emb_path),
                "metadata_file": os.path.basename(meta_path),
            }
            print(
                f"  ✓ {method}: {len(all_emb)} × "
                f"{all_emb.shape[1] if all_emb.ndim == 2 else '?'} → {emb_path}"
            )

        for group, emb in sorted(noise_embeddings.items()):
            noise_path = os.path.join(out_dir, f"noise_{group}_embeddings.npy")
            noise_meta_path = os.path.join(out_dir, f"noise_{group}_meta.json")
            np.save(noise_path, emb.astype(np.float32))
            with open(noise_meta_path, "w", encoding="utf-8") as f:
                json.dump(noise_meta.get(group, []), f, indent=2, ensure_ascii=False)
            model_summary["noise_references"][group] = {
                "n_embeddings": int(len(emb)),
                "embedding_dim": int(emb.shape[1]),
                "embeddings_file": os.path.basename(noise_path),
                "metadata_file": os.path.basename(noise_meta_path),
            }

        model_summary["noise_distance"] = _score_noise_distance(
            all_method_embs, noise_embeddings
        )
        for method, result in model_summary["noise_distance"].items():
            if result.get("status") == "ok":
                delta = result.get("delta_vs_mono")
                delta_text = f", Δmono={delta:+.4f}" if delta is not None else ""
                print(
                    f"  noise-distance {method}: "
                    f"{result['mean_noise_distance']:.4f}{delta_text}"
                )

        sum_path = os.path.join(out_dir, summary_basename(date_str))
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)
        report["models"][model] = model_summary
        print(f"  Summary: {sum_path}  ({model_summary['elapsed_sec']}s)")

        # ── Clean up intermediate shards (model completed, Layer 2) ──────────
        _ckpt_dir = os.path.join(out_dir, ".ckpt")
        if os.path.islink(_ckpt_dir):
            os.unlink(_ckpt_dir)
            print(f"  [Ckpt-L2] Symlink .ckpt removed (target left on ephemeral)")
        elif os.path.isdir(_ckpt_dir):
            shutil.rmtree(_ckpt_dir)
            print(f"  [Ckpt-L2] Intermediate chunk files cleaned up")
        # ────────────────────────────────────────────────────────────────────

        # Explicit VRAM & RAM cleanup between models
        embedder_cache.clear()
        all_method_embs.clear()
        all_method_meta.clear()
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


    return report


def write_comparison_report(report: Dict[str, Any], output_dir: str) -> Tuple[str, str]:
    """Write a compact model × method report centred on mono."""
    os.makedirs(output_dir, exist_ok=True)
    stem = f"{report['date']}_bacpipe_comparison"
    json_path = os.path.join(output_dir, f"{stem}.json")
    md_path = os.path.join(output_dir, f"{stem}.md")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    lines = [
        f"# bacpipe comparison — {report['location']} / {report['date']}",
        "",
        "- Baseline: **mono**",
        f"- Comparators: {', '.join(report['comparator_methods'])}",
        f"- Method WAVs: {report['n_wavs']}",
        f"- Noise WAVs: {report['n_noise_wavs']}",
        f"- Device: {report.get('device', 'cpu')}",
        "",
        "Positive Δmono means farther from the model-specific noise reference than mono.",
        "",
        "| Model | Method | N embeddings | Noise distance | Δ vs mono | Status |",
        "|---|---|---:|---:|---:|---|",
    ]
    for model, summary in report.get("models", {}).items():
        scores = summary.get("noise_distance", {})
        for method in report.get("methods", []):
            score = scores.get(method, {})
            if score.get("status") == "ok":
                delta = score.get("delta_vs_mono")
                lines.append(
                    f"| `{model}` | `{method}` | {score.get('n', '')} | "
                    f"{score.get('mean_noise_distance', float('nan')):.6f} | "
                    f"{delta:.6f} | ok |"
                    if delta is not None
                    else
                    f"| `{model}` | `{method}` | {score.get('n', '')} | "
                    f"{score.get('mean_noise_distance', float('nan')):.6f} | — | ok |"
                )
            else:
                lines.append(
                    f"| `{model}` | `{method}` | — | — | — | "
                    f"{score.get('status', 'not_processed')} |"
                )
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return json_path, md_path


def main() -> None:
    p = argparse.ArgumentParser(
        description="bacpipe embedding pilot on existing spatial method WAVs"
    )
    p.add_argument("--location", required=True)
    p.add_argument("--date", required=True, help="Single date YYYY-MM-DD")
    p.add_argument(
        "--models",
        default="all",
        help="Comma-separated bacpipe model names, or 'all'/'bird_models' to run curated bioacoustic models",
    )
    p.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method folder names",
    )
    p.add_argument("--data-dir", default=ANALYSIS_OUTPUT)
    p.add_argument(
        "--device",
        default=os.environ.get("BACPIPE_DEVICE", "auto"),
        help="auto | cuda | cpu | mps",
    )
    p.add_argument(
        "--max-wavs",
        type=int,
        default=0,
        help="Global cap on WAVs (0 = all). Prefer --max-wavs-per-method for balanced pilots.",
    )
    p.add_argument(
        "--max-wavs-per-method",
        type=int,
        default=0,
        help="Cap WAVs per method folder (balanced mono/SA/BF pilots).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="List WAVs only; do not import bacpipe or embed",
    )
    p.add_argument(
        "--noise-dir",
        default=None,
        help="Noise reference root (default: {data_dir}/{location}/noise_references)",
    )
    p.add_argument(
        "--report-json",
        default=None,
        help="Optional additional path to write the full run report JSON",
    )
    p.add_argument(
        "--checkpoint-dir",
        default=str(_DEFAULT_CKPT),
        help="Directory for bacpipe model checkpoints (auto-download)",
    )
    args = p.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    report = run_pilot(
        location=args.location,
        date_str=args.date,
        models=models,
        methods=methods,
        data_dir=args.data_dir,
        device=args.device,
        max_wavs=args.max_wavs,
        dry_run=args.dry_run,
        checkpoint_dir=args.checkpoint_dir,
        max_wavs_per_method=args.max_wavs_per_method,
        noise_dir=args.noise_dir,
    )

    if not args.dry_run:
        default_audit_dir = os.path.join(args.data_dir, args.location, "embeddings", "audits")
        comparison_json, comparison_md = write_comparison_report(report, default_audit_dir)
        print(f"Comparison report written: {comparison_json}")
        print(f"Comparison table written:  {comparison_md}")

    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written: {args.report_json}")


if __name__ == "__main__":
    main()
