"""
Impulse Response-based Multidirectional Beamforming.
No filtering — raw 6-channel audio goes directly through beamforming.

Supports three IR types:
- LabIR:  Virtual sound environment, 12 elevations x 36 azimuths
- SPIR1:  Silwood Park forest, 4 distances x 6 azimuths
- SPIR2:  Silwood Park forest, 7 distances x 1 azimuth x 3 reps

Uses precomputed IR steering-vector cache (ircache.py) to avoid
re-loading and re-processing each raw .wav IR on every run.
"""

import os
import time
import numpy as np
import librosa
import soundfile as sf
from typing import Optional, Union

from config import (
    FS_TARGET,
    N_CHANNELS_EXPECTED,
    FRAME_LEN_SEC,
    IRType,
)
from ircache import IRCache


class Beamformer:
    """Multidirectional beamforming using impulse response steering vectors."""

    def __init__(
        self,
        flac_path: str,
        output_dir: str,
        ir_type_or_name: Union[IRType, str],
    ):
        self.flac_path = flac_path
        self.output_dir = output_dir
        self.fs = FS_TARGET

        if isinstance(ir_type_or_name, str):
            self.ir_type_name = ir_type_or_name
            self.ir_cache = IRCache(ir_type_or_name)
            self.ir_type = self.ir_cache.ir_type
        else:
            self.ir_type = ir_type_or_name
            self.ir_type_name = ir_type_or_name.name
            self.ir_cache = IRCache(self.ir_type_name)

        self.base_name = os.path.splitext(os.path.basename(flac_path))[0]

        # STFT geometry
        self.framelen = int(FRAME_LEN_SEC * self.fs)
        self.frameinc = self.framelen // 2

        # Load audio (no filtering)
        print(f"  Reading audio: {flac_path}")
        self.raw, _ = librosa.load(flac_path, sr=self.fs, mono=False)

        # Precompute STFT (shared across all directions)
        print(f"  Computing STFT ...")
        self.X_stft = librosa.stft(
            self.raw,
            n_fft=self.framelen,
            hop_length=self.frameinc,
            window="hamming",
        )
        self.Y = np.transpose(self.X_stft, (1, 0, 2))
        del self.raw
        os.makedirs(self.output_dir, exist_ok=True)

    def _compute_weights(self, Rxx: np.ndarray, IRR: np.ndarray) -> np.ndarray:
        """MVDR beamforming weights."""
        n_channels, n_freq, n_frame = IRR.shape
        Rxx = Rxx.astype(np.complex128)
        W = np.empty((n_freq, n_channels, n_frame), dtype=np.complex128)
        for ifreq in range(n_freq):
            invRd = np.linalg.solve(Rxx, IRR[:, ifreq, :])
            projection = np.zeros(n_frame, dtype=np.complex128)
            for iframe in range(n_frame):
                projection[iframe] = np.dot(IRR[:, ifreq, iframe].conj(), invRd[:, iframe])
            W[ifreq, :, :] = np.conj(invRd / projection)
        return W

    def _process_one(self, speaker: int, degrees: int, rep: Optional[int] = None) -> bool:
        """Beamform towards one direction using cached steering vector.

        Returns:
            True on success, False if skipped (cache/IR file missing).
        """
        try:
            IRR = self.ir_cache.load(speaker, degrees, rep)
        except FileNotFoundError:
            print(f"    SKIP (not in cache): p={speaker} d={degrees}")
            return False

        print(f"    Beamforming: p={speaker} d={degrees} ...")
        IRR = IRR.reshape(IRR.shape[0], IRR.shape[1], 1)
        n_channel = IRR.shape[0]
        Rxx = np.eye(n_channel)
        W = self._compute_weights(Rxx, IRR)
        Z = np.sum(W * self.Y, axis=1, keepdims=True)
        del W
        z = librosa.istft(Z[:, 0, :], hop_length=self.frameinc, window="hamming")
        del Z
        zmax = np.max(np.abs(z))
        if zmax > 0:
            z = np.real(z) / zmax

        fmt_kwargs = {}
        if self.ir_type.param_label == "speaker":
            fmt_kwargs["speaker"] = speaker
        else:
            fmt_kwargs["distance"] = speaker
        fmt_kwargs["degrees"] = degrees
        if rep is not None:
            fmt_kwargs["rep"] = rep

        out_suffix = self.ir_type.output_suffix_pattern.format(**fmt_kwargs)
        out_filename = f"{self.base_name}_{out_suffix}.wav"
        out_path = os.path.join(self.output_dir, out_filename)
        sf.write(out_path, (z * 32767).clip(-32768, 32767).astype("int16"), self.fs, subtype="PCM_16")
        print(f"      -> {out_filename}")
        return True

    def run(self):
        """Run beamforming for ALL (param x degree) combinations.

        Returns:
            (output_count, skip_count) — total successful vs skipped outputs.
        """
        start = time.time()
        total = 0
        skipped = 0
        zenith = self.ir_type.zenith_speakers or set()
        for param in self.ir_type.param_values:
            if param in zenith:
                degrees = [0]
            else:
                degrees = self.ir_type.degree_values
            for deg in degrees:
                reps = self.ir_type.rep_values or [None]
                for rep in reps:
                    ok = self._process_one(param, deg, rep)
                    total += 1
                    if not ok:
                        skipped += 1
        elapsed = time.time() - start
        if skipped > 0:
            print(f"  ⚠ Beamforming: {total - skipped}/{total} outputs in {elapsed:.1f}s "
                  f"({skipped} skipped — IR cache/raw files missing)")
        else:
            print(f"  Beamforming complete: {total} outputs in {elapsed:.1f}s")
        return (total - skipped, skipped)
