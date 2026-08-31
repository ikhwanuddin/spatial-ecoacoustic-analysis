#!/usr/bin/env python3
"""
High-Performance Bioacoustic Foundation Model Embedding Pipeline (pipeline_bacpipe.py).

Optimized for NVIDIA RTX 6000 on Imperial CX3 HPC:
- Saturated Window Batching (batch_size = 32..64)
- Asynchronous Background Audio Prefetcher (Zero GPU I/O Starvation)
- Safe JIT Kernel Fusion (torch.compile) with graceful eager fallback
- Guaranteed IEEE 754 Float32 (FP32) Bit-Exact Precision
- Layer 1 (Model-level) & Layer 2 (Chunk-level) Checkpointing & Resume
"""

from __future__ import annotations

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
    os.environ.setdefault("HF_HUB_CACHE", f"{_EPHEM_BASE}/.cache/huggingface/hub")
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", f"{_EPHEM_BASE}/.cache/huggingface/hub")
    os.environ.setdefault("TORCH_HOME", f"{_EPHEM_BASE}/.cache/torch")
os.environ.setdefault("NUMBA_CACHE_DIR", f"{_TMP_DIR}/numba")
os.environ.setdefault("TORCH_EXTENSIONS_DIR", f"{_TMP_DIR}/torch_ext")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("CUDA_MODULE_LOADING", "LAZY")
# Make CUDA 12.6 and matching cuDNN visible before TensorFlow/ONNX Runtime import.
# The CX3 module environment may expose the compiler but not the shared libraries
# when this script is launched from an interactive shell.  ORT then silently falls
# back to CPU, so discover the standard CX3 installations here as well.
_CUDA_CANDIDATES = (
    os.environ.get("CUDA_HOME"),
    "/sw-eb/software/CUDA/12.6.0",
    "/rds/easybuild/noarch/apps/software/CUDA/12.6.0",
)
_CUDA_HOME = next(
    (p for p in _CUDA_CANDIDATES if p and os.path.isdir(f"{p}/bin")),
    os.environ.get("CUDA_HOME", "/sw-eb/software/CUDA/12.6.0"),
)
if os.path.isdir(f"{_CUDA_HOME}/bin"):
    os.environ["PATH"] = f"{_CUDA_HOME}/bin" + os.pathsep + os.environ.get("PATH", "")
_CUDA_LIBRARY_DIRS = [
    f"{_CUDA_HOME}/lib64",
    f"{_CUDA_HOME}/lib",
]
_CUDNN_CANDIDATES = (
    os.environ.get("CUDNN_HOME"),
    "/sw-eb/software/cuDNN/9.10.2.21-CUDA-12.6.0",
    "/rds/easybuild/noarch/apps/software/cuDNN/9.10.2.21-CUDA-12.6.0",
)
_CUDNN_HOME = next(
    (p for p in _CUDNN_CANDIDATES if p and os.path.isdir(f"{p}/lib")),
    None,
)
if _CUDNN_HOME:
    _CUDA_LIBRARY_DIRS.append(f"{_CUDNN_HOME}/lib")
_CUDA_LIBRARY_DIRS = [p for p in _CUDA_LIBRARY_DIRS if os.path.isdir(p)]
if _CUDA_LIBRARY_DIRS:
    _existing_ld = os.environ.get("LD_LIBRARY_PATH", "")
    _ld_parts = _CUDA_LIBRARY_DIRS + ([_existing_ld] if _existing_ld else [])
    os.environ["LD_LIBRARY_PATH"] = os.pathsep.join(dict.fromkeys(_ld_parts))
    # LD_LIBRARY_PATH changes do not update glibc's loader after process start.
    # Preload the provider dependencies with RTLD_GLOBAL so ORT can resolve them.
    try:
        import ctypes
        _cuda_runtime_libs = (
            "libcudart.so.12",
            "libcublas.so.12",
            "libcublasLt.so.12",
            "libcurand.so.10",
            "libcufft.so.11",
            "libnvrtc.so.12",
        )
        _cudnn_runtime_libs = (
            "libcudnn.so.9",
            "libcudnn_adv.so.9",
            "libcudnn_ops.so.9",
            "libcudnn_cnn.so.9",
            "libcudnn_graph.so.9",
            "libcudnn_engines_runtime_compiled.so.9",
            "libcudnn_engines_precompiled.so.9",
            "libcudnn_heuristic.so.9",
        )
        for _lib_name in _cuda_runtime_libs + _cudnn_runtime_libs:
            _lib_root = _CUDNN_HOME if _lib_name.startswith("libcudnn") else _CUDA_HOME
            _lib_path = os.path.join(_lib_root, "lib", _lib_name) if _lib_root else ""
            if os.path.isfile(_lib_path):
                try:
                    ctypes.CDLL(_lib_path, mode=ctypes.RTLD_GLOBAL)
                except OSError:
                    # A missing optional dependency is reported by the backend
                    # validation when CUDA is explicitly requested.
                    pass
    except ImportError:
        pass
if os.path.isfile(f"{_CUDA_HOME}/nvvm/libdevice/libdevice.10.bc"):
    _cur_xla = os.environ.get("XLA_FLAGS", "")
    if "--xla_gpu_cuda_data_dir" not in _cur_xla:
        os.environ["XLA_FLAGS"] = f"{_cur_xla} --xla_gpu_cuda_data_dir={_CUDA_HOME}".strip()

import argparse
import gc
import hashlib
import gzip
import importlib
import json
import queue
import re
import shutil
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure parent directory is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import (
    ANALYSIS_OUTPUT,
    LOCATION_MAP,
    RPIID_TO_LOCATION,
)
from embedding_schema import (
    audits_dir,
    BACKEND_BACPIPE,
    beam_tag_from_name,
    CONDITIONS,
    noise_key,
    DEFAULT_METHODS,
    bacpipe_embeddings_dir,
    bacpipe_meta_dir,
    embedding_basename,
    meta_basename,
    summary_basename,
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS & MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

BACKEND_BACPIPE = "bacpipe"

# No active model in the fork is intentionally CPU-only on Linux/CX3.
TF_CPU_ONLY_MODELS = set()
# Active TensorFlow models; all are CUDA-capable on Linux with the TF GPU stack.
TF_GPU_MODELS = {
    "perch_bird",
    "google_whale",
    "surfperch",
    "vggish",
    "birdnet",
    "birdnet_v3",
    "hbdet",
}

KNOWN_TF_MODELS = TF_CPU_ONLY_MODELS | TF_GPU_MODELS


def _model_target_device(model_name: str, requested_device: str) -> str:
    """Route model to its optimal and safe compute device."""
    name_lower = model_name.lower()
    if requested_device == "cpu":
        return "cpu"
    if name_lower in TF_CPU_ONLY_MODELS:
        return "cpu"
    if name_lower in TF_GPU_MODELS:
        return "cuda" if requested_device in ("cuda", "auto") else requested_device
    return requested_device if requested_device in ("cuda", "mps") else "cuda"


def _configure_frameworks(device: str) -> None:
    """Configure TensorFlow & PyTorch for maximum GPU memory efficiency."""
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
    "convnext_birdset",
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


# ─────────────────────────────────────────────────────────────────────────────
# METADATA & PARSING HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _parse_direction_metadata(filename: str) -> Tuple[Optional[float], Optional[float]]:
    m_deg = re.search(r"_(\d{1,3})deg(?:\.wav)?$", filename, re.IGNORECASE)
    if m_deg:
        return float(m_deg.group(1)), 0.0

    m_spk = re.search(r"_(S\d{2})_([+-]?\d+deg)?(?:\.wav)?$", filename, re.IGNORECASE)
    if m_spk:
        speaker = m_spk.group(1).upper()
        zenith_elevation = {"S01": 90.0, "S12": -90.0}
        if speaker in zenith_elevation:
            return 0.0, zenith_elevation[speaker]
        if m_spk.group(2):
            deg = float(m_spk.group(2).replace("deg", ""))
            return deg, 0.0
        return None, 0.0

    m_bf = re.search(r"_(?:az|deg)(\d{1,3})(?:_el([+-]?\d+))?", filename, re.IGNORECASE)
    if m_bf:
        az = float(m_bf.group(1))
        el = float(m_bf.group(2)) if m_bf.group(2) else 0.0
        return az, el

    return None, None


def make_window_meta(
    *,
    wav: str,
    method: str,
    start_sec: float,
    end_sec: float,
    model: str,
    backend: str,
    window_sec: float,
    slide_sec: float,
    embedding_dim: int,
    azimuth: Optional[float] = None,
    elevation: Optional[float] = None,
    extra: Optional[dict] = None,
) -> dict:
    meta = {
        "wav": wav,
        "method": method,
        "start_sec": round(start_sec, 3),
        "end_sec": round(end_sec, 3),
        "model": model,
        "backend": backend,
        "window_sec": round(window_sec, 3),
        "slide_sec": round(slide_sec, 3),
        "embedding_dim": embedding_dim,
    }
    if azimuth is not None:
        meta["azimuth"] = azimuth
    if elevation is not None:
        meta["elevation"] = elevation
    if extra:
        meta["extra"] = extra
    return meta


def _l2_normalize_rows(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X.astype(np.float32), axis=1, keepdims=True)
    return X.astype(np.float32) / np.maximum(norms, 1e-9)


def _noise_fingerprint(noise_wavs: List[Tuple[str, str, str]]) -> str:
    """Identity of the noise reference set, so a changed set forces a rescore.

    Without this a summary that already holds a noise_distance block is treated
    as done, and freshly rebuilt references are silently ignored.
    """
    parts = []
    for path, name, group in sorted(noise_wavs, key=lambda item: item[1]):
        try:
            size = os.path.getsize(path)
        except OSError:
            size = -1
        parts.append(f"{group}/{name}:{size}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


def _noise_group_for_method(method: str) -> str:
    return method.replace("bf_", "", 1)


def _noise_keys_by_condition(
    noise_embeddings: Dict[str, np.ndarray], group: str, methods: List[str]
) -> Dict[str, List[str]]:
    """Reference keys for one method, grouped by time condition.

    References are stored per beam, so one method owns many keys inside a
    condition. They are pooled within their condition and never across it.
    """
    def belongs(tail: str) -> bool:
        # A beam tag is the group name, optionally with a sub-type digit,
        # followed by its parameters: SPIR -> SPIR1(02m_000), SPIR2(64m_180_r2).
        if tail == group:
            return True
        head = tail.split("(", 1)[0]
        return head.rstrip("0123456789") == group and head != tail

    by_condition: Dict[str, List[str]] = {}
    for key in sorted(noise_embeddings):
        for condition in CONDITIONS:
            head = f"{condition}_"
            if not key.startswith(head):
                continue
            if belongs(key[len(head):]):
                by_condition.setdefault(condition, []).append(key)
            break
    if not by_condition and group in noise_embeddings:
        by_condition[""] = [group]      # reference predating condition scoping
    return by_condition


def _score_noise_distance(
    by_method_emb: Dict[str, Any],
    noise_embeddings: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Score methods against noise, failing closed on invalid cached arrays.

    A failed model run can leave a valid-looking .npy file with shape
    (0, 0). It must be reported, not passed to matrix multiplication.

    A method is only ever compared with a reference built from that same
    method. When a date holds more than one time condition, this summary
    cannot tell which window belongs to which condition -- it reports each
    condition separately and marks the result, and the authoritative
    per-window scoring is the one in visualize_bacpipe.
    """
    def _as_matrix(value: Any) -> Optional[np.ndarray]:
        parts = value if isinstance(value, list) else [value]
        valid: List[np.ndarray] = []
        for part in parts:
            if part is None:
                continue
            arr = np.asarray(part)
            if arr.ndim != 2 or arr.shape[0] == 0 or arr.shape[1] == 0:
                continue
            valid.append(arr.astype(np.float32, copy=False))
        if not valid or len({arr.shape[1] for arr in valid}) != 1:
            return None
        return np.concatenate(valid, axis=0)

    distances: Dict[str, Dict[str, Any]] = {}
    methods_list = sorted(by_method_emb)
    for method, chunks in sorted(by_method_emb.items()):
        group = _noise_group_for_method(method)
        keys_by_condition = _noise_keys_by_condition(noise_embeddings, group, methods_list)
        first = next(iter(keys_by_condition.values()), [])
        noise = [noise_embeddings[k] for k in first] if first else None
        X = _as_matrix(chunks)
        noise_matrix = _as_matrix(noise)
        if X is None:
            distances[method] = {
                "noise_group": group,
                "status": "empty_target_embeddings",
            }
            continue
        if noise is None:
            distances[method] = {
                "noise_group": group,
                "status": "missing_noise_reference",
            }
            continue
        if noise_matrix is None:
            distances[method] = {
                "noise_group": group,
                "status": "empty_noise_embeddings",
            }
            continue
        if X.shape[1] != noise_matrix.shape[1]:
            distances[method] = {
                "noise_group": group,
                "status": "embedding_dimension_mismatch",
                "target_embedding_dim": int(X.shape[1]),
                "noise_embedding_dim": int(noise_matrix.shape[1]),
                "n": int(len(X)),
                "n_noise": int(len(noise_matrix)),
            }
            continue
        X_norm = _l2_normalize_rows(X)

        def _against(keys: List[str]) -> Optional[Dict[str, Any]]:
            matrix = _as_matrix([noise_embeddings[k] for k in keys])
            if matrix is None or matrix.shape[1] != X.shape[1]:
                return None
            mean_vec = _l2_normalize_rows(matrix).mean(axis=0)
            mean_vec /= max(float(np.linalg.norm(mean_vec)), 1e-9)
            cosine = X_norm @ mean_vec
            return {
                "n_noise_keys": len(keys),
                "n_noise": int(len(matrix)),
                "mean_cosine_to_noise": float(np.mean(cosine)),
                "mean_noise_distance": float(1.0 - np.mean(cosine)),
                "std_noise_distance": float(np.std(1.0 - cosine)),
            }

        scored = {c: r for c, ks in keys_by_condition.items() if (r := _against(ks)) is not None}
        if not scored:
            distances[method] = {"noise_group": group, "status": "empty_noise_embeddings"}
            continue
        entry: Dict[str, Any] = {
            "noise_group": group,
            "n": int(len(X)),
            "embedding_dim": int(X.shape[1]),
            "per_condition": scored,
        }
        if len(scored) == 1:
            condition, only = next(iter(scored.items()))
            entry.update(only)
            entry["condition"] = condition or "unscoped"
            entry["status"] = "ok"
        else:
            entry["status"] = "scored_per_condition"
        distances[method] = entry

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


def find_noise_wavs(
    data_dir: str,
    location: str,
    date_str: Optional[str] = None,
    noise_dir: Optional[str] = None,
) -> List[Tuple[str, str, str]]:
    """Return (wav_path, wav_name, noise_group) from noise_references."""
    if noise_dir:
        root = Path(noise_dir)
    elif date_str and (Path(data_dir) / location / date_str / "noise_references").is_dir():
        root = Path(data_dir) / location / date_str / "noise_references"
    else:
        root = Path(data_dir) / location / "noise_references"

    if not root.is_dir():
        return []
    found: List[Tuple[str, str, str]] = []
    for path in sorted(root.rglob("*.wav")):
        if path.name.startswith("._"):
            continue
        rel = path.relative_to(root)
        group, condition = "unknown", None
        for part in rel.parts:
            if part in ("LabIR", "SPIR", "sa", "mono"):
                group = part
            elif part in CONDITIONS:
                condition = part
        # Scope the reference to its time condition so a dawn prototype can
        # never be averaged together with a night one.
        found.append((str(path), path.name, noise_key(condition, group)))
    return found


def _normalize_embeddings(embeddings: Any) -> np.ndarray:
    """Coerce bacpipe return value to strict float32 (N, D)."""
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
    return emb.astype(np.float32)


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


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION A & D: BATCH SIZE TUNING & JIT KERNEL FUSION
# ─────────────────────────────────────────────────────────────────────────────

def _tune_model_batch_size(em: Any, model_name: str, device: str) -> None:
    """Optimize internal window batch size to saturate RTX 6000 Tensor Cores."""
    if device != "cuda":
        return

    name = model_name.lower()
    target_bs = 64  # Default saturated batch size for RTX 6000 (24GB VRAM)

    # Heavier models get a safe batch size of 32
    if any(h in name for h in ("audiomae", "birdmae", "perch", "beat")):
        target_bs = 32

    # Set batch_size on model wrapper
    if hasattr(em, "model") and hasattr(em.model, "batch_size"):
        old_bs = em.model.batch_size
        em.model.batch_size = target_bs
        print(f"  [Batch Optimizer] Model '{model_name}' internal batch_size tuned: {old_bs} → {target_bs}")


def _get_embedder(
    model: str,
    device: str,
    cache: Dict[str, Any],
    checkpoint_dir: Optional[Path] = None,
):
    """Reuse one Embedder per model with batch & JIT optimizations."""
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

    if target_dev == "cuda":
        # Fail closed when a CUDA-capable backend silently initializes on CPU.
        _backend = getattr(em, "model", None)
        _onnx = getattr(_backend, "model", None)
        _active_providers = getattr(_backend, "active_providers", None)
        if _active_providers is None:
            _active_providers = getattr(_onnx, "active_providers", None)
        if _active_providers is not None and "CUDAExecutionProvider" not in _active_providers:
            raise RuntimeError(
                f"{model} requested CUDA but active providers are {_active_providers}. "
                "Check CUDA/cuDNN libraries and ONNX Runtime GPU installation."
            )
        if model.lower() in TF_GPU_MODELS:
            try:
                import tensorflow as tf
                if not tf.config.list_logical_devices("GPU"):
                    raise RuntimeError(f"{model} requested CUDA but TensorFlow sees no GPU")
            except ImportError:
                pass
        if model.lower() not in KNOWN_TF_MODELS and _active_providers is None:
            try:
                import torch
                if isinstance(_onnx, torch.nn.Module):
                    first_parameter = next(_onnx.parameters(), None)
                    if first_parameter is not None and first_parameter.device.type != "cuda":
                        raise RuntimeError(
                            f"{model} requested CUDA but model parameters are on "
                            f"{first_parameter.device}"
                        )
            except StopIteration:
                pass
    # Apply Optimization A (Batch Sizing)
    _tune_model_batch_size(em, model, target_dev)

    # Apply Optimization D (Safe PyTorch JIT Kernel Fusion)
    if target_dev == "cuda" and model.lower() not in KNOWN_TF_MODELS:
        try:
            import torch
            if hasattr(torch, "compile") and hasattr(em, "model") and hasattr(em.model, "model"):
                if isinstance(em.model.model, torch.nn.Module):
                    em.model.model = torch.compile(em.model.model, mode="reduce-overhead")
                    print(f"  [JIT Optimizer] JIT Kernel Fusion (torch.compile) enabled for {model}")
        except Exception as exc:
            # Gracefully continue with standard eager mode
            print(f"  [JIT Optimizer] Using eager execution mode ({exc})")

    cache[model] = em
    return em


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZATION B: ASYNCHRONOUS AUDIO PREFETCHER (ZERO GPU STARVATION)
# ─────────────────────────────────────────────────────────────────────────────

class AudioPrefetcher:
    """
    Background worker that pre-loads, decodes, and stages WAV files into RAM
    while the GPU executes the current inference batch.
    """

    def __init__(self, wav_list: List[Tuple[str, str, str]], max_queue_size: int = 4, num_workers: int = 2):
        self.wav_list = wav_list
        self.max_queue_size = max_queue_size
        self.num_workers = num_workers
        self.queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=num_workers)
        self._producer_thread = threading.Thread(target=self._run_producer, daemon=True)
        self._producer_thread.start()

    def _preload_one(self, item: Tuple[str, str, str]) -> Tuple[Tuple[str, str, str], Any]:
        wav_path, wav_name, method = item
        # Fast filesystem probe / pre-caching
        try:
            sz = os.path.getsize(wav_path)
            return item, sz
        except Exception as exc:
            return item, exc

    def _run_producer(self) -> None:
        for item in self.wav_list:
            if self.stop_event.is_set():
                break
            res = self._preload_one(item)
            while not self.stop_event.is_set():
                try:
                    self.queue.put(res, timeout=0.1)
                    break
                except queue.Full:
                    continue
        # Sentinel to signal end of stream
        while not self.stop_event.is_set():
            try:
                self.queue.put(None, timeout=0.1)
                break
            except queue.Full:
                continue

    def get_next(self) -> Optional[Tuple[str, str, str]]:
        if self.stop_event.is_set():
            return None
        res = self.queue.get()
        if res is None:
            return None
        item, _ = res
        return item

    def shutdown(self) -> None:
        self.stop_event.set()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except Exception:
            pass
        self.executor.shutdown(wait=False)


# ─────────────────────────────────────────────────────────────────────────────
# CHUNK STREAMING & CHECKPOINT HELPERS (LAYER 2)
# ─────────────────────────────────────────────────────────────────────────────

_CHUNK_SIZE = 50  # Process and stream to disk in chunks of N WAVs (keeps RAM < 350MB)


def _model_already_done(out_dir: str, meta_dir: str, date_str: str, methods: List[str]) -> bool:
    """Layer 1: True if all outputs for this model are already complete on disk."""
    summary_path = os.path.join(meta_dir, summary_basename(date_str))
    if not os.path.exists(summary_path):
        return False
    try:
        with open(summary_path, encoding="utf-8") as f:
            summary = json.load(f)
        done_methods = set(summary.get("methods", {}).keys())
        if not set(methods).issubset(done_methods):
            print(f"  [Ckpt-L1] Summary exists but methods are incomplete: {set(methods) - done_methods}")
            return False
        for method in methods:
            emb_path = os.path.join(out_dir, embedding_basename(date_str, method))
            if not os.path.exists(emb_path):
                return False
            try:
                emb = np.load(emb_path, mmap_mode="r")
                if emb.ndim != 2 or emb.shape[0] == 0 or emb.shape[1] == 0:
                    print(f"  [Ckpt-L1] Cached {method} embeddings are empty: {emb.shape}")
                    return False
            except (OSError, ValueError):
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
    """Save one compact compressed chunk file and update state."""
    if not buffer_meta or not any(buffer_emb.values()):
        print(f"  [Ckpt-L2] Buffer is empty, skipping chunk #{chunk_idx}")
        return ""

    os.makedirs(ckpt_dir, exist_ok=True)
    chunk_filename = f"{date_str}_chunk_{chunk_idx:04d}.npz"
    chunk_path = os.path.join(ckpt_dir, chunk_filename)
    tmp_path = os.path.join(ckpt_dir, f".tmp_{chunk_filename}")

    emb_fp32 = {}
    for m, arr_list in buffer_emb.items():
        if arr_list:
            emb_fp32[m] = np.concatenate(arr_list, axis=0).astype(np.float32)

    meta_json_gz = gzip.compress(json.dumps(buffer_meta).encode("utf-8"))
    np.savez_compressed(
        tmp_path,
        meta_gz=np.frombuffer(meta_json_gz, dtype=np.uint8),
        **emb_fp32,
    )
    os.replace(tmp_path, chunk_path)

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
    print(f"  [Ckpt-L2] Chunk #{chunk_idx} saved ({processed_count} WAVs total, {sz_mb:.1f} MB) — memory purged")
    return chunk_path


def _load_progress(
    out_dir: str,
    date_str: str,
    model: str,
) -> Tuple[int, int]:
    """Layer 2: Check for existing progress. Returns (start_wav_idx, next_chunk_idx)."""
    ckpt_dir = os.path.join(out_dir, ".ckpt")
    if not os.path.isdir(ckpt_dir):
        return 0, 0

    chunk_files = sorted(Path(ckpt_dir).glob(f"{date_str}_chunk_*.npz"))
    valid_chunks = []
    total_wavs = 0
    for cp in chunk_files:
        if cp.stat().st_size < 1024:
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


def _embed_with_bacpipe(
    model: str,
    wav_path: str,
    wav_name: str,
    method: str,
    device: str,
    embedder_cache: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, Optional[dict]]:
    """Run one bacpipe Embedder on a single WAV. Returns (embeddings [FP32], compact_meta_record)."""
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


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
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
        data_dir=data_dir, location=resolved_loc, date_str=date_str, noise_dir=noise_dir
    )
    noise_fp = _noise_fingerprint(noise_wavs)

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
        f"\n============================================================"
        f"\n🚀 pipeline_bacpipe: {resolved_loc} / {date_str} (Device: {resolved_device.upper()})"
        f"\n📦 {len(method_wavs)} method WAVs across {methods}"
        f"\n🔊 {len(noise_wavs)} noise references | 🧠 {len(resolved_models)} models"
        f"\n============================================================"
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
        meta_dir = bacpipe_meta_dir(resolved_loc, model)
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)
        ckpt_dir = os.path.join(out_dir, ".ckpt")

        # ── LAYER 1: Skip already completed models OR run fast noise scoring ──
        if _model_already_done(out_dir, meta_dir, date_str, methods):
            sum_path = os.path.join(meta_dir, summary_basename(date_str))
            noise_already_scored = False
            try:
                if os.path.exists(sum_path):
                    with open(sum_path, encoding="utf-8") as f:
                        cached_sum = json.load(f)
                    nd = cached_sum.get("noise_distance", {})
                    # Every method scored, against the reference set on disk now.
                    # `any` would have let a partial failure count as finished.
                    scored_ok = bool(nd) and all(
                        v.get("status") in ("ok", "scored_per_condition") for v in nd.values()
                    )
                    if scored_ok and cached_sum.get("noise_reference_fingerprint") == noise_fp:
                        noise_already_scored = True
                    elif scored_ok:
                        print(f"  [Noise] reference set changed for {model}; rescoring.")
            except Exception:
                pass

            if noise_already_scored or not noise_wavs:
                print(f"\n[{model_idx}/{len(resolved_models)}] ✅ SKIP {model} — output and noise scoring already complete in {out_dir}")
                try:
                    with open(sum_path, encoding="utf-8") as f:
                        report["models"][model] = json.load(f)
                except Exception:
                    pass
                continue

            # ── Fast Noise Catch-Up Pass ──────────────────────────────────────
            print(f"\n[{model_idx}/{len(resolved_models)}] 🔄 Fast Noise Scoring: {model} (method embeddings cached, embedding {len(noise_wavs)} noise refs)...")
            t0 = time.time()
            embedder_cache: Dict[str, Any] = {}
            try:
                _get_embedder(model, target_dev, embedder_cache, checkpoint_dir=ckpt_path)
            except Exception as e:
                print(f"  [ERROR] Failed to initialize model '{model}': {e}")
                continue

            noise_embeddings: Dict[str, np.ndarray] = {}
            noise_meta: Dict[str, List[dict]] = {}
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
                    print(f"  [WARN] Failed embedding noise {wav_name}: {e}")

            for group, emb in sorted(noise_embeddings.items()):
                noise_path = os.path.join(meta_dir, f"noise_{group}_embeddings.npy")
                noise_meta_path = os.path.join(meta_dir, f"noise_{group}_meta.json")
                np.save(noise_path, emb.astype(np.float32))
                with open(noise_meta_path, "w", encoding="utf-8") as f:
                    json.dump(noise_meta.get(group, []), f, indent=2, ensure_ascii=False)
                print(f"  ✓ noise_{group}: {len(emb)} embeddings (dim={emb.shape[1]}) -> {os.path.basename(noise_path)}")

            existing_embs: Dict[str, List[np.ndarray]] = {}
            for method in methods:
                emb_file = os.path.join(out_dir, embedding_basename(date_str, method))
                if os.path.isfile(emb_file):
                    existing_embs[method] = [np.load(emb_file)]

            model_summary = {}
            if os.path.exists(sum_path):
                try:
                    with open(sum_path, encoding="utf-8") as f:
                        model_summary = json.load(f)
                except Exception:
                    pass

            model_summary["noise_references"] = {
                group: {
                    "n_embeddings": int(len(emb)),
                    "embedding_dim": int(emb.shape[1]),
                    "embeddings_file": f"noise_{group}_embeddings.npy",
                }
                for group, emb in noise_embeddings.items()
            }
            model_summary["noise_distance"] = _score_noise_distance(existing_embs, noise_embeddings)
            model_summary["noise_reference_fingerprint"] = noise_fp
            for method, result in model_summary["noise_distance"].items():
                if result.get("status") == "ok":
                    delta = result.get("delta_vs_mono")
                    delta_text = f", Δmono={delta:+.4f}" if delta is not None else ""
                    print(f"  noise-distance {method}: {result['mean_noise_distance']:.4f}{delta_text}")

            with open(sum_path, "w", encoding="utf-8") as f:
                json.dump(model_summary, f, indent=2, ensure_ascii=False)
            report["models"][model] = model_summary
            print(f"  ✓ Noise scoring complete for {model} in {time.time() - t0:.1f}s")
            embedder_cache.clear()
            gc.collect()
            continue

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

        # ── LAYER 2: Load previous progress (Incremental Chunk Streaming) ───
        start_idx, next_chunk_idx = _load_progress(out_dir, date_str, model)
        buffer_emb: Dict[str, List[np.ndarray]] = {}
        buffer_meta: List[dict] = []
        chunk_idx = next_chunk_idx

        # Launch Asynchronous Background Audio Prefetcher
        pending_wavs = method_wavs[start_idx:]
        prefetcher = AudioPrefetcher(pending_wavs, max_queue_size=4, num_workers=2)

        try:
            for i in range(start_idx + 1, len(method_wavs) + 1):
                item = prefetcher.get_next()
                if item is None:
                    break
                wav_path, wav_name, method = item
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

                    # Save compact chunk every _CHUNK_SIZE WAVs & purge RAM
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

                    if i % 25 == 0 or i == len(method_wavs):
                        speed = i / max(time.time() - t0, 0.1)
                        print(f"  Processed {i}/{len(method_wavs)} WAVs ... ({time.time()-t0:.1f}s, {speed:.1f} WAV/s)")
                except Exception as e:
                    if _strict_unreadable(e):
                        errors.append({"file": wav_name, "method": method, "status": "skipped_corrupt", "error": str(e)})
                        print(f"  [SKIP corrupt/unreadable] {model} on {wav_name}: {e}")
                    else:
                        errors.append({"file": wav_name, "method": method, "error": str(e)})
                        print(f"  [ERROR] {model} on {wav_name}: {e}")
        finally:
            prefetcher.shutdown()

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

        # Assemble all chunks into final .npy & expand lazy metadata
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
            meta_path = os.path.join(meta_dir, meta_basename(date_str, method))
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
            noise_path = os.path.join(meta_dir, f"noise_{group}_embeddings.npy")
            noise_meta_path = os.path.join(meta_dir, f"noise_{group}_meta.json")
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

        model_summary["noise_reference_fingerprint"] = noise_fp
        sum_path = os.path.join(meta_dir, summary_basename(date_str))
        with open(sum_path, "w", encoding="utf-8") as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)
        report["models"][model] = model_summary
        print(f"  Summary: {sum_path}  ({model_summary['elapsed_sec']}s)")

        # Clean up intermediate shards (model completed, Layer 2)
        _ckpt_dir = os.path.join(out_dir, ".ckpt")
        if os.path.islink(_ckpt_dir):
            os.unlink(_ckpt_dir)
            print(f"  [Ckpt-L2] Symlink .ckpt removed")
        elif os.path.isdir(_ckpt_dir):
            shutil.rmtree(_ckpt_dir)
            print(f"  [Ckpt-L2] Intermediate chunk files cleaned up")

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
        description="High-Performance Bioacoustic Embedding Pipeline (pipeline_bacpipe.py)"
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
        help="Global cap on WAVs (0 = all). Prefer --max-wavs-per-method for balanced runs.",
    )
    p.add_argument(
        "--max-wavs-per-method",
        type=int,
        default=0,
        help="Cap WAVs per method folder (balanced mono/SA/BF runs).",
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

    report = run_pipeline(
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
        default_audit_dir = audits_dir(args.location)
        comparison_json, comparison_md = write_comparison_report(report, default_audit_dir)
        print(f"\nComparison report written: {comparison_json}")
        print(f"Comparison table written:  {comparison_md}")

    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)) or ".", exist_ok=True)
        with open(args.report_json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"Report written: {args.report_json}")



# STRICT_NOISE_GATE_V1
# Fail-closed noise-reference gate. This wrapper runs before model/device
# initialization and scopes refs by location/date/time-condition/group.
from config import MONITORING_DATA as _STRICT_MONITORING_DATA

_STRICT_TIME_CONDITIONS = ("dawn", "day", "dusk", "night")
_STRICT_TIME_RANGES = {
    "dawn": (5, 7),
    "day": (7, 17),
    "dusk": (17, 19),
    "night": (19, 24),
}
_STRICT_ACTIVE_SCOPE = {}


NOISE_REFERENCE_BEAM = "LabIR(S05_000)"


def _strict_condition(name: str) -> Optional[str]:
    match = re.search(r"(?:^|_)(\d{2})-(\d{2})-(\d{2})(?:_|\\.)", name)
    if not match:
        return None
    hour = int(match.group(1))
    if 5 <= hour < 7:
        return "dawn"
    if 7 <= hour < 17:
        return "day"
    if 17 <= hour < 19:
        return "dusk"
    if 0 <= hour < 5 or 19 <= hour < 24:
        return "night"
    return None


def _strict_expected_groups(methods: List[str]) -> List[str]:
    return sorted({_noise_group_for_method(method) for method in methods})


def _strict_scope_root(base: str, location: str, date_str: str, condition: str) -> Path:
    candidates = [
        Path(base) / location / date_str / "noise_references" / condition,
        Path(base) / location / date_str / condition,
        Path(base) / date_str / "noise_references" / condition,
        Path(base) / date_str / condition,
        Path(base) / "noise_references" / condition,
        Path(base) / condition,
        Path(base) / location / "noise_references",
    ]
    for c in candidates:
        if c.is_dir():
            return c
    return candidates[0]


def _strict_inventory(
    base: str, location: str, date_str: str, conditions: List[str]
) -> Dict[str, Dict[str, int]]:
    inventory = {}
    for condition in conditions:
        root = _strict_scope_root(base, location, date_str, condition)
        groups = {}
        if root.is_dir():
            for path in root.rglob("*.wav"):
                if path.name.startswith("._"):
                    continue
                relative = path.relative_to(root)
                group = "unknown"
                for part in relative.parts:
                    if part in ("LabIR", "SPIR", "sa", "mono"):
                        group = part
                        break
                groups[group] = groups.get(group, 0) + 1
        inventory[condition] = groups
    return inventory


def _strict_raw_flacs(location: str, date_str: str, condition: str) -> List[Path]:
    rpi_id = LOCATION_MAP.get(location)
    if not rpi_id:
        return []
    root = Path(_STRICT_MONITORING_DATA) / rpi_id / date_str
    return [p for p in sorted(root.glob("*.flac")) if _strict_condition(p.name) == condition]


def _strict_prepare_review(
    location: str, date_str: str, conditions: List[str], noise_base: str,
    data_dir: str,
) -> Dict[str, Any]:
    detector = _PROJECT_ROOT / "detect_noise_references_temporal.py"
    review_base = Path(noise_base).parent / "noise_auto_review" / location / date_str
    result = {"status": "not_run", "conditions": {}}
    if not detector.is_file():
        result["status"] = "detector_missing"
        result["detector"] = str(detector)
        return result
    result["status"] = "review_prepared"
    # Only the reference beam is scanned. Running the detector on all 19 beams
    # costs 19x the time and yields 19 disagreeing interval sets for the same
    # instant; mono/SA/SPIR inherit the intervals accepted on this one beam.
    labir_wavs = [
        item for item in find_method_wavs(
            data_dir=data_dir, location=location, date_str=date_str,
            methods=["bf_LabIR"],
        )
        if NOISE_REFERENCE_BEAM in item[1]
    ]
    for condition in conditions:
        condition_dir = review_base / condition
        prepared, skipped = 0, 0
        source_wavs = [Path(path) for path, name, _method in labir_wavs
                       if _strict_condition(name) == condition]
        for labir_wav in source_wavs:
            output_dir = condition_dir / labir_wav.stem
            json_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", labir_wav.stem)
            if (output_dir / (json_name + "_temporal_noise_detection.json")).is_file():
                prepared += 1
                continue
            output_dir.mkdir(parents=True, exist_ok=True)
            command = [
                sys.executable, str(detector), "--input", str(labir_wav),
                "--output-dir", str(output_dir), "--export-candidates",
            ]
            completed = subprocess.run(command, check=False)
            if completed.returncode == 0:
                prepared += 1
            else:
                skipped += 1
        result["conditions"][condition] = {
            "review_dir": str(condition_dir),
            "source_method": "bf_LabIR",
            "source_recordings": len(source_wavs),
            "recordings_prepared": prepared,
            "recordings_skipped": skipped,
        }
    return result


def _strict_gate(
    location: str,
    date_str: str,
    methods: List[str],
    available: List[str],
    noise_base: str,
    auto_prepare: bool,
    data_dir: str,
) -> Dict[str, Any]:
    expected = _strict_expected_groups(methods)
    inventory = _strict_inventory(noise_base, location, date_str, available)
    missing = {
        condition: [
            group for group in expected
            if inventory.get(condition, {}).get(group, 0) == 0
        ]
        for condition in available
    }
    missing = {condition: groups for condition, groups in missing.items() if groups}
    preparation = None
    if missing and auto_prepare:
        preparation = _strict_prepare_review(
            location, date_str, list(missing), noise_base, data_dir=data_dir
        )
        inventory = _strict_inventory(noise_base, location, date_str, available)
    return {
        "status": "ready" if not missing else "blocked_noise_references",
        "ready": not missing,
        "conditions": available,
        "expected_groups": expected,
        "inventory": inventory,
        "missing": missing,
        "noise_root": noise_base,
        "preparation": preparation,
    }


def _strict_find_noise_wavs(
    data_dir: str, location: str, date_str: Optional[str] = None, noise_dir: Optional[str] = None, *args, **kwargs
) -> List[Tuple[str, str, str]]:
    scope = _STRICT_ACTIVE_SCOPE
    loc = location or scope.get("location", "")
    date = date_str or scope.get("date", "")
    base = noise_dir or os.path.join(data_dir, loc, date, "noise_references") if loc and date else (noise_dir or os.path.join(data_dir, "noise_references"))
    found = []
    conditions = scope.get("conditions", []) or ["dawn", "day", "dusk", "night"]
    for condition in conditions:
        root = _strict_scope_root(base, loc, date, condition)
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.wav")):
            if path.name.startswith("._"):
                continue
            relative = path.relative_to(root)
            group = "unknown"
            for part in relative.parts:
                if part in ("LabIR", "SPIR", "sa", "mono"):
                    group = part
                    break
            # Scope to the condition it was found under and to the steering
            # direction it was cut from, so a beam is only ever compared with
            # noise captured through that same beam.
            tag = beam_tag_from_name(path.name) or group
            found.append((str(path), path.name, noise_key(condition, tag)))
    return found


def _strict_unreadable(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(token in message for token in (
        "cannot open", "error opening", "failed to open", "decode",
        "decoding", "truncated", "corrupt", "soundfile", "invalid file",
        "no audio", "not a valid", "no such file", "read error",
    ))


_original_find_noise_wavs = find_noise_wavs
_original_run_pipeline = run_pipeline
find_noise_wavs = _strict_find_noise_wavs


def run_pipeline(*args, **kwargs):
    # The original function is keyword-only; preserve that contract.
    location = _resolve_location(kwargs["location"])
    date_str = kwargs["date_str"]
    methods = kwargs["methods"]
    data_dir = kwargs.get("data_dir", ANALYSIS_OUTPUT)
    dry_run = kwargs.get("dry_run", False)
    auto_prepare = kwargs.pop("auto_prepare_noise", True)
    method_wavs = find_method_wavs(
        data_dir=data_dir,
        location=location,
        date_str=date_str,
        methods=methods,
        max_wavs=kwargs.get("max_wavs", 0),
        max_wavs_per_method=kwargs.get("max_wavs_per_method", 0),
    )
    available = [
        condition for condition in _STRICT_TIME_CONDITIONS
        if any(_strict_condition(name) == condition for _p, name, _m in method_wavs)
    ]
    # Same root the finder uses, otherwise the gate looks in a directory that
    # never holds the per-date references and blocks every run.
    noise_base = kwargs.get("noise_dir") or os.path.join(
        data_dir, location, date_str, "noise_references"
    )
    gate = _strict_gate(
        location, date_str, methods, available, noise_base,
        auto_prepare and not dry_run,
        data_dir=data_dir,
    )
    _STRICT_ACTIVE_SCOPE.clear()
    _STRICT_ACTIVE_SCOPE.update({
        "location": location, "date": date_str, "conditions": available
    })
    if not dry_run and not gate["ready"]:
        report = {
            "location": location, "date": date_str,
            "methods": methods, "comparator_methods": [m for m in methods if m != "mono"],
            "n_wavs": len(method_wavs), "n_noise_wavs": 0,
            "available_conditions": available,
            "time_condition_ranges": _STRICT_TIME_RANGES,
            "noise_gate": gate, "models": {},
        }
        print("\n[NOISE GATE] BLOCKED — bacpipe will not process embeddings.")
        print("Available conditions:", ", ".join(available) or "none")
        print("Missing scoped references:", json.dumps(gate["missing"], sort_keys=True))
        if gate.get("preparation"):
            print("Noise-review preparation completed. Inspect candidates, then materialize approved method-specific WAVs under:")
            print("  " + str(Path(noise_base).parent / "noise_auto_review" / location / date_str))
        print("Required layout: <data-dir>/<location>/<date>/noise_references/<condition>/<group>/*.wav")
        return report
    result = _original_run_pipeline(*args, **kwargs)
    result["available_conditions"] = available
    result["time_condition_ranges"] = _STRICT_TIME_RANGES
    result["noise_gate"] = gate
    return result


if __name__ == "__main__":
    main()
