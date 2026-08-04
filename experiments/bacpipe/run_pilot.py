#!/usr/bin/env python3
"""
bacpipe multi-model embedding pilot on existing method WAVs.

Does not run beamforming. Reads full WAVs under:
  {data_dir}/{location}/{date}/{method}/h_*/m_*/*.wav

Writes schema-aligned arrays under:
  {data_dir}/{location}/embeddings/bacpipe/{model}/

Usage:
  python experiments/bacpipe/run_pilot.py \\
    --location 2A400 --date 2026-04-22 --models birdnet,perch_bird --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
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
) -> Dict[str, Any]:
    location = _resolve_location(location)
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
        "methods": methods,
        "data_dir": data_dir,
        "checkpoint_dir": str(ckpt),
    }

    print(f"Location: {location}")
    print(f"Date:     {date_str}")
    print(f"WAVs:     {len(wavs)}  (methods={methods})")
    print(f"Models:   {models}")
    print(f"Device:   {device}")
    if dry_run:
        for p, name, m in wavs[:20]:
            print(f"  [{m}] {name}")
        if len(wavs) > 20:
            print(f"  … {len(wavs) - 20} more")
        report["dry_run"] = True
        return report

    if not wavs:
        print("No WAVs found — run pipeline_embeddings.py Phase 1 first.")
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
        _ensure_checkpoints(models, ckpt)
    except Exception as e:
        print(f"Checkpoint ensure failed: {e}", file=sys.stderr)
        print("Will still try Embedder (may auto-download).", file=sys.stderr)

    embedder_cache: Dict[str, Any] = {}

    for model in models:
        print(f"\n══ Model: {model} ══")
        out_dir = bacpipe_embeddings_dir(data_dir, location, model)
        os.makedirs(out_dir, exist_ok=True)

        by_method_emb: Dict[str, List[np.ndarray]] = {}
        by_method_meta: Dict[str, List[dict]] = {}
        t0 = time.time()
        errors = 0

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

        model_summary: Dict[str, Any] = {
            "out_dir": out_dir,
            "errors": errors,
            "methods": {},
            "elapsed_sec": round(time.time() - t0, 1),
        }

        for method, chunks in sorted(by_method_emb.items()):
            all_emb = np.concatenate(chunks, axis=0).astype(np.float32)
            meta = by_method_meta.get(method, [])
            emb_path = os.path.join(out_dir, embedding_basename(date_str, method))
            meta_path = os.path.join(out_dir, meta_basename(date_str, method))
            np.save(emb_path, all_emb)
            with open(meta_path, "w") as f:
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

        sum_path = os.path.join(out_dir, summary_basename(date_str))
        with open(sum_path, "w") as f:
            json.dump(model_summary, f, indent=2, ensure_ascii=False)
        report["models"][model] = model_summary
        print(f"  Summary: {sum_path}  ({model_summary['elapsed_sec']}s)")

    return report


def main() -> None:
    p = argparse.ArgumentParser(
        description="bacpipe embedding pilot on existing spatial method WAVs"
    )
    p.add_argument("--location", required=True)
    p.add_argument("--date", required=True, help="Single date YYYY-MM-DD")
    p.add_argument(
        "--models",
        default="birdnet,perch_bird",
        help="Comma-separated bacpipe model names",
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
        "--report-json",
        default=None,
        help="Optional path to write full run report JSON",
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
    )

    if args.report_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.report_json)) or ".", exist_ok=True)
        with open(args.report_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Report written: {args.report_json}")


if __name__ == "__main__":
    main()
