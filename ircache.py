"""
Precomputed Impulse Response steering-vector cache.

Each IR file is loaded once, resampled to target fs, RFFT'd to
framelen size, and normalised (relative to channel 0).

The resulting complex128 matrix [n_channels × n_freq_bins] is saved
as a .npz file keyed by the IR's (param, degree, rep).

This avoids re-loading and re-processing each IR every time
Beamformer runs — especially useful for the full 432-direction LabIR
grid across many FLAC files.
"""

import os
import time
import numpy as np
import librosa

from typing import Dict, Optional, Tuple
from config import (
    IR_BASE_PATH,
    FS_TARGET,
    FS_IR_ORIGINAL,
    FRAME_LEN_SEC,
    IR_TYPES,
    IRType,
)

CACHE_ROOT = os.path.join(os.path.dirname(__file__), "ir_cache")


def _cache_key(ir_type_name: str, param: int, degrees: int, rep: Optional[int] = None) -> str:
    """Return a deterministic cache key string."""
    base = f"{ir_type_name}_p{param}_d{degrees:03d}"
    if rep is not None:
        base += f"_r{rep}"
    return base


def _compute_irr(ir_path: str, framelen: int) -> np.ndarray:
    """
    Load IR, resample to FS_TARGET, RFFT, normalise.

    Returns:
        complex128 array [n_channels, n_freq_bins].
    """
    ir_orig, _ = librosa.load(ir_path, sr=FS_IR_ORIGINAL, mono=False)
    ir = librosa.resample(ir_orig, orig_sr=FS_IR_ORIGINAL, target_sr=FS_TARGET)
    IR = np.fft.rfft(ir, n=framelen, axis=-1)
    IRR = IR / (IR[0, :] + 1e-12)   # normalise by reference channel
    return IRR.astype(np.complex128)


class IRCache:
    """
    Precompute and cache IR steering vectors for a given IRType.

    Usage:
        cache = IRCache("LabIR")
        cache.build()             # first time — precompute all
        irr = cache.load("LabIR", speaker=1, degrees=0)

    The cache directory structure:
        ir_cache/LabIR/
            LabIR_p1_d000.npz
            LabIR_p1_d010.npz
            ...
    """

    def __init__(self, ir_type_name: str, framelen: Optional[int] = None):
        if ir_type_name not in IR_TYPES:
            raise KeyError(f"Unknown IR type: {ir_type_name}")

        self.ir_type: IRType = IR_TYPES[ir_type_name]
        self.name = ir_type_name
        self.ir_folder = os.path.join(IR_BASE_PATH, self.ir_type.folder)
        self.cache_dir = os.path.join(CACHE_ROOT, ir_type_name)

        if framelen is None:
            framelen = int(FRAME_LEN_SEC * FS_TARGET)
        self.framelen = framelen
        self.n_freq = framelen // 2 + 1

        os.makedirs(self.cache_dir, exist_ok=True)

    # ── Path helpers ─────────────────────────────────────────

    def _ir_path(self, param: int, degrees: int, rep: Optional[int] = None) -> str:
        """Absolute path to the raw IR .wav file."""
        fmt = {}
        if self.ir_type.param_label == "speaker":
            fmt["speaker"] = param
        else:
            fmt["distance"] = param
        fmt["degrees"] = degrees
        if rep is not None:
            fmt["rep"] = rep
        filename = self.ir_type.ir_filename_pattern.format(**fmt)
        return os.path.join(self.ir_folder, filename)

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, f"{key}.npz")

    # ── Cache operations ─────────────────────────────────────

    def is_cached(self, param: int, degrees: int, rep: Optional[int] = None) -> bool:
        key = _cache_key(self.name, param, degrees, rep)
        return os.path.isfile(self._cache_path(key))

    def load(self, param: int, degrees: int, rep: Optional[int] = None) -> np.ndarray:
        """
        Load a cached steering vector.

        Returns:
            complex128 [n_channels, n_freq_bins]
        """
        key = _cache_key(self.name, param, degrees, rep)
        path = self._cache_path(key)
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Cache miss: {key} — run .build() first"
            )
        data = np.load(path)
        irr = data["irr"]
        return irr

    def compute_one(self, param: int, degrees: int, rep: Optional[int] = None) -> np.ndarray:
        """Compute and cache a single steering vector."""
        ir_path = self._ir_path(param, degrees, rep)
        if not os.path.isfile(ir_path):
            raise FileNotFoundError(f"IR file not found: {ir_path}")

        key = _cache_key(self.name, param, degrees, rep)
        cache_path = self._cache_path(key)

        # Return cached if valid
        if os.path.isfile(cache_path):
            return self.load(param, degrees, rep)

        # Compute
        irr = _compute_irr(ir_path, self.framelen)
        np.savez_compressed(cache_path, irr=irr)
        return irr

    def build(self, force: bool = False, verbose: bool = True) -> int:
        """
        Precompute all steering vectors for this IRType.

        Args:
            force:   Recompute even already-cached entries.
            verbose: Print progress.

        Returns:
            Number of steering vectors cached.
        """
        total = 0
        combo_count = 0

        start = time.time()

        for param in self.ir_type.param_values:
            for deg in self.ir_type.degree_values:
                reps = self.ir_type.rep_values or [None]
                for rep in reps:
                    combo_count += 1
                    key = _cache_key(self.name, param, deg, rep)
                    path = self._cache_path(key)

                    if not force and os.path.isfile(path):
                        continue

                    ir_path = self._ir_path(param, deg, rep)
                    if not os.path.isfile(ir_path):
                        continue

                    try:
                        irr = _compute_irr(ir_path, self.framelen)
                        np.savez_compressed(path, irr=irr)
                        total += 1
                    except Exception as e:
                        if verbose:
                            print(f"  ⚠ Failed {key}: {e}")

        elapsed = time.time() - start
        if verbose:
            existing = combo_count - total
            print(
                f"  ✓ IR cache [{self.name}]: {total} new, {existing} already cached "
                f"({combo_count} total combos, {elapsed:.1f}s)"
            )

        return total

    def stats(self) -> Dict:
        """Return cache statistics."""
        cached = 0
        total_size = 0
        for fname in os.listdir(self.cache_dir):
            if fname.endswith(".npz"):
                cached += 1
                total_size += os.path.getsize(os.path.join(self.cache_dir, fname))
        return {
            "ir_type": self.name,
            "cached_files": cached,
            "total_size_bytes": total_size,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
        }


# ── Convenience: build all caches ────────────────────────────

def build_all_caches(force: bool = False):
    """Precompute steering vectors for all IR types."""
    print("\n🔧 Building IR steering-vector caches ...")
    for name in IR_TYPES:
        cache = IRCache(name)
        cache.build(force=force)
        st = cache.stats()
        print(f"    {st['cached_files']} files, {st['total_size_mb']} MB")
    print("  ✓ All IR caches built.\n")
