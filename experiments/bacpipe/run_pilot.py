#!/usr/bin/env python3
"""
bacpipe multi-model embedding pilot on existing method WAVs.

Does not run beamforming. Reads full WAVs under:
  {data_dir}/{location}/{date}/{method}/h_*/m_*/*.wav

Writes schema-aligned arrays under:
  {data_dir}/{location}/embeddings/bacpipe/{model}/

Usage:
  python experiments/bacpipe/run_pilot.py \\
    --location 2A400 --date 2026-04-26 --models all --methods mono,sa,bf_LabIR,bf_SPIR --dry-run
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import sys
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


def discover_models() -> List[str]:
    """Return model names exposed by the installed bacpipe package.

    bacpipe releases have used more than one registry name. Inspect the public
    registry/settings modules instead of making Perch a special case. The
    documented models remain a conservative fallback for older releases.
    """
    try:
        import bacpipe
    except ImportError:
        return []

    modules = [bacpipe]
    for module_name in ("settings", "models", "model_registry", "constants"):
        try:
            modules.append(importlib.import_module(f"bacpipe.{module_name}"))
        except (ImportError, ModuleNotFoundError):
            continue

    names: List[str] = []
    registry_words = ("model", "embed", "checkpoint", "available", "registry")
    registry_callables = ("list_models", "available_models", "get_available_models", "get_model_names")
    for module in modules:
        for function_name in registry_callables:
            function = getattr(module, function_name, None)
            if callable(function):
                try:
                    names.extend(_model_name_candidates(function()))
                except Exception:
                    pass
        for attr_name in dir(module):
            if not any(word in attr_name.lower() for word in registry_words):
                continue
            try:
                value = getattr(module, attr_name)
            except Exception:
                continue
            names.extend(_model_name_candidates(value))

    # Older bacpipe versions document these names but do not expose a registry.
    if not names:
        names = ["birdnet", "perch_bird"]
    return list(dict.fromkeys(names))


def _resolve_models(models: List[str]) -> List[str]:
    if any(model.lower() == "all" for model in models):
        discovered = discover_models()
        if not discovered:
            raise RuntimeError("No bacpipe models were discovered")
        return discovered
    return list(dict.fromkeys(models))


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
    by_method_emb: Dict[str, List[np.ndarray]],
    noise_embeddings: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    """Score each method against its own model-space noise reference.

    The report is centred on mono: each available comparator includes a delta
    from mono, where positive means farther from noise.
    """
    distances: Dict[str, Dict[str, Any]] = {}
    for method, chunks in sorted(by_method_emb.items()):
        group = _noise_group_for_method(method)
        noise = noise_embeddings.get(group)
        if noise is None or not chunks:
            distances[method] = {"noise_group": group, "status": "missing_noise_reference"}
            continue
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
            "std_noise_distance": float(np.std(1.0 - cosine)),
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
    """Return list of (wav_path, wav_name, method).

    Caps:
      max_wavs_per_method — applied per method first (balanced multi-method pilots)
      max_wavs — global cap after merging
    """
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
    # samples + sr
    try:
        sr = getattr(model, "sr", None) or getattr(model, "sample_rate", None)
        seg = getattr(model, "segment_length", None) or getattr(
            model, "num_samples", None
        )
        if sr and seg and float(sr) > 100:  # likely samples not seconds
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
        # default non-overlapping if hop unknown
        slide_sec = float(window_sec)
    return window_sec, slide_sec


# Default checkpoint dir under experiments/bacpipe/ (stable absolute path)
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

    ckpt = Path(checkpoint_dir) if checkpoint_dir else _DEFAULT_CKPT
    try:
        bacpipe.settings.device = device
        bacpipe.settings.model_base_path = str(ckpt)
    except Exception:
        pass
    # CWD-relative paths in bacpipe resolve better if we chdir temporarily
    # is fragile; prefer absolute model_base_path already set.
    em = bacpipe.Embedder(model)
    cache[model] = em
    return em


def _embed_with_bacpipe(
    model: str,
    wav_path: str,
    wav_name: str,
    method: str,
    device: str,
    embedder_cache: Optional[Dict[str, Any]] = None,
    checkpoint_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, List[dict]]:
    """Run one bacpipe Embedder on a single WAV; return (N, D), meta list."""
    cache = embedder_cache if embedder_cache is not None else {}
    em = _get_embedder(model, device, cache, checkpoint_dir=checkpoint_dir)
    embeddings = em.get_embeddings_from_model(wav_path)
    emb = _normalize_embeddings(embeddings)
    if emb.size == 0:
        return emb, []

    n, dim = emb.shape
    window_sec, slide_sec = _model_window_sec(em)

    azimuth, elevation = _parse_direction_metadata(wav_name)
    meta: List[dict] = []
    hop = slide_sec if isinstance(slide_sec, (int, float)) else (
        window_sec if isinstance(window_sec, (int, float)) else 1.0
    )
    win = window_sec if isinstance(window_sec, (int, float)) else float(hop)
    for i in range(n):
        start = float(i * hop)
        end = float(start + win)
        meta.append(
            make_window_meta(
                wav=wav_name,
                method=method,
                start_sec=start,
                end_sec=end,
                model=model,
                backend=BACKEND_BACPIPE,
                window_sec=float(win),
                slide_sec=float(hop) if isinstance(hop, (int, float)) else 0.0,
                embedding_dim=int(dim),
                azimuth=azimuth,
                elevation=elevation,
                extra={
                    "window_sec_note": None
                    if isinstance(window_sec, (int, float))
                    else str(window_sec),
                    "wav_path": wav_path,
                },
            )
        )
    return emb, meta


def run_pilot(
    *,
    location: str,
    date_str: str,
    models: List[str],
    methods: List[str],
    data_dir: str,
    device: str,
    max_wavs: int,
    dry_run: bool,
    checkpoint_dir: Optional[str] = None,
    max_wavs_per_method: int = 0,
    noise_dir: Optional[str] = None,
) -> Dict[str, Any]:
    location = _resolve_location(location)
    requested_models = list(models)
    noise_wavs = find_noise_wavs(data_dir, location, noise_dir=noise_dir)
    wavs = find_method_wavs(
        data_dir,
        location,
        date_str,
        methods,
        max_wavs=max_wavs,
        max_wavs_per_method=max_wavs_per_method,
    )
    ckpt = Path(checkpoint_dir) if checkpoint_dir else _DEFAULT_CKPT

    report: Dict[str, Any] = {
        "location": location,
        "date": date_str,
        "n_wavs": len(wavs),
        "models": {},
        "models_requested": requested_models,
        "methods": methods,
        "baseline_method": "mono",
        "comparator_methods": [m for m in methods if m != "mono"],
        "data_dir": data_dir,
        "noise_dir": str(noise_dir or Path(data_dir) / location / "noise_references"),
        "n_noise_wavs": len(noise_wavs),
        "checkpoint_dir": str(ckpt),
    }

    print(f"Location: {location}")
    print(f"Date:     {date_str}")
    print(f"WAVs:     {len(wavs)}  (methods={methods})")
    print(f"Models:   {requested_models}")
    print("Baseline: mono")
    print(f"Device:   {device}")
    print(f"Noise WAVs: {len(noise_wavs)}")
    if dry_run:
        for p, name, m in wavs[:20]:
            print(f"  [{m}] {name}")
        if len(wavs) > 20:
            print(f"  … {len(wavs) - 20} more")
        report["dry_run"] = True
        return report

    if not wavs:
        print("No WAVs found — run pipeline_signal_processing.py first.")
        return report

    try:
        import bacpipe  # noqa: F401
    except ImportError:
        print(
            "bacpipe is not installed in this environment.\n"
            "  pip install -r experiments/bacpipe/requirements.txt\n"
            "or use experiments/bacpipe/.venv (see experiments/bacpipe/README.md).",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        models = _resolve_models(requested_models)
    except Exception as e:
        print(f"Model discovery failed: {e}", file=sys.stderr)
        return report
    report["models_discovered"] = models
    print(f"Resolved models: {models}")

    for model in models:
        try:
            _ensure_checkpoints([model], ckpt)
        except Exception as e:
            print(f"Checkpoint ensure failed for {model}: {e}", file=sys.stderr)
            print("Will still try Embedder (may auto-download).", file=sys.stderr)

    embedder_cache: Dict[str, Any] = {}

    for model in models:
        print(f"\n══ Model: {model} ══")
        out_dir = bacpipe_embeddings_dir(data_dir, location, model)
        os.makedirs(out_dir, exist_ok=True)

        by_method_emb: Dict[str, List[np.ndarray]] = {}
        by_method_meta: Dict[str, List[dict]] = {}
        noise_embeddings: Dict[str, np.ndarray] = {}
        noise_meta: Dict[str, List[dict]] = {}
        t0 = time.time()
        errors = 0
        noise_errors = 0

        for i, (wav_path, wav_name, method) in enumerate(wavs, 1):
            print(f"  [{i}/{len(wavs)}] {method} / {wav_name}", flush=True)
            try:
                emb, meta = _embed_with_bacpipe(
                    model,
                    wav_path,
                    wav_name,
                    method,
                    device,
                    embedder_cache=embedder_cache,
                    checkpoint_dir=ckpt,
                )
            except Exception as e:
                errors += 1
                print(f"    ⚠️  failed: {e}", file=sys.stderr)
                continue
            if emb.size == 0:
                continue
            by_method_emb.setdefault(method, []).append(emb)
            by_method_meta.setdefault(method, []).extend(meta)

        # Re-embed the same noise WAVs in this model's space. Noise vectors
        # from another backend are never reused.
        for noise_path, noise_name, noise_group in noise_wavs:
            print(f"  [noise/{noise_group}] {noise_name}", flush=True)
            try:
                emb, meta = _embed_with_bacpipe(
                    model,
                    noise_path,
                    noise_name,
                    f"noise_{noise_group}",
                    device,
                    embedder_cache=embedder_cache,
                    checkpoint_dir=ckpt,
                )
            except Exception as e:
                noise_errors += 1
                print(f"    ⚠️  noise failed: {e}", file=sys.stderr)
                continue
            if emb.size == 0:
                continue
            noise_embeddings.setdefault(noise_group, np.zeros((0, emb.shape[1]), dtype=np.float32))
            noise_embeddings[noise_group] = np.concatenate(
                [noise_embeddings[noise_group], emb.astype(np.float32)], axis=0
            )
            noise_meta.setdefault(noise_group, []).extend(meta)

        model_summary: Dict[str, Any] = {
            "out_dir": out_dir,
            "errors": errors,
            "noise_errors": noise_errors,
            "methods": {},
            "noise_references": {},
            "noise_distance": {},
            "elapsed_sec": round(time.time() - t0, 1),
        }

        for method, chunks in sorted(by_method_emb.items()):
            all_emb = np.concatenate(chunks, axis=0).astype(np.float32)
            meta = by_method_meta.get(method, [])
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
            by_method_emb, noise_embeddings
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
        f.write("\\n".join(lines) + "\\n")
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
        help="Comma-separated bacpipe model names, or 'all' to discover the installed registry",
    )
    p.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
        help="Comma-separated method folder names",
    )
    p.add_argument("--data-dir", default=ANALYSIS_OUTPUT)
    p.add_argument(
        "--device",
        default=os.environ.get("BACPIPE_DEVICE", "cpu"),
        help="cpu | mps | cuda",
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
