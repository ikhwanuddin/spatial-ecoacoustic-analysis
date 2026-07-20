"""
Impulse Response-based Multidirectional Beamforming.

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
import scipy.signal as signal
import librosa
import soundfile as sf
from typing import Optional, Union

from config import (
    FS_TARGET,
    N_CHANNELS_EXPECTED,
    FRAME_LEN_SEC,
    IRType,
)
from audio_utils import safe_load_audio
from ircache import IRCache


class Beamformer:
    """
    Multidirectional beamforming using impulse response steering vectors.

    For a given FLAC recording and IR type, produces one WAV output per
    (param, degree) combination — or (param, degree, rep) for SPIR2.

    Steering vectors are loaded from precomputed cache (fast) rather
    than raw .wav IR files (slow).
    """

    def __init__(
        self,
        flac_path: str,
        output_dir: str,
        ir_type_or_name: Union[IRType, str],
    ):
        """
        Args:
            flac_path:       Absolute path to the input .flac file.
            output_dir:      Directory where beamformed WAVs are saved.
            ir_type_or_name: IRType dataclass or string like "LabIR".
        """
        self.flac_path = flac_path
        self.output_dir = output_dir
        self.fs = FS_TARGET

        # Resolve IR type
        if isinstance(ir_type_or_name, str):
            self.ir_type_name = ir_type_or_name
            self.ir_cache = IRCache(ir_type_or_name)
            self.ir_type = self.ir_cache.ir_type
        else:
            self.ir_type = ir_type_or_name
            self.ir_type_name = ir_type_or_name.name
            self.ir_cache = IRCache(self.ir_type_name)

        self.fc_high = self.ir_type.fc_high
        self.fc_low = self.ir_type.fc_low
        self.base_name = os.path.splitext(os.path.basename(flac_path))[0]

        # Build high-pass filter
        nyq = self.fs / 2
        Wp = self.ir_type.fc_high / nyq
        Ws = Wp / 2
        Rp, Rs = 3, 40
        self.n_hp, self.Wn_hp = signal.buttord(Wp, Ws, Rp, Rs)
        self.b_hp, self.a_hp = signal.butter(self.n_hp, self.Wn_hp, btype="high")

        # Optionally build low-pass filter
        if self.ir_type.use_dual_filter:
            Wp_lp = self.ir_type.fc_low / nyq
            Ws_lp = Wp_lp * 1.2
            self.n_lp, self.Wn_lp = signal.buttord(Wp_lp, Ws_lp, Rp, Rs)
            self.b_lp, self.a_lp = signal.butter(self.n_lp, self.Wn_lp, btype="low")

        # STFT geometry (fixed for all IRs of this type)
        self.framelen = int(FRAME_LEN_SEC * self.fs)
        self.frameinc = self.framelen // 2

        # Load and filter audio
        print(f"  Reading audio: {flac_path}")
        self.raw = safe_load_audio(flac_path, sr=self.fs, mono=False)
        self.filtered_audio = self._apply_filters(self.raw)

        # Precompute STFT of filtered audio (shared across all directions)
        print(f"  Computing STFT ...")
        self.X_stft = librosa.stft(
            self.filtered_audio,
            n_fft=self.framelen,
            hop_length=self.frameinc,
            window="hamming",
        )  # -> [freq, channel, frame]  or [channel, freq, frame]?

        # librosa.stft with multi-channel returns [n_channels, n_freq, n_frames]
        # Make it [n_freq, n_channels, n_frames] to match weight layout
        self.Y = np.transpose(self.X_stft, (1, 0, 2))
        # self.Y: [n_freq, n_channels, n_frames]

        del self.filtered_audio, self.raw

        os.makedirs(self.output_dir, exist_ok=True)

    # ---- Filtering -------------------------------------------------

    def _apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Apply HP (and optionally LP) filter to multi-channel audio."""
        filtered = signal.filtfilt(self.b_hp, self.a_hp, audio, axis=-1)
        if self.ir_type.use_dual_filter:
            filtered = signal.filtfilt(self.b_lp, self.a_lp, filtered, axis=-1)
        return filtered

    # ---- Beamforming weights computation ---------------------------

    def _compute_weights(self, Rxx: np.ndarray, IRR: np.ndarray) -> np.ndarray:
        """
        MVDR beamforming weights.

        IRR:  [n_channels, n_freq, n_frame]  (n_frame = 1 for cached IR)
        Rxx:  [n_channels, n_channels]       (identity -> delay-and-sum)

        Returns W: [n_freq, n_channels, n_frame]  complex128
        """
        n_channels, n_freq, n_frame = IRR.shape
        Rxx = Rxx.astype(np.complex128)
        W = np.empty((n_freq, n_channels, n_frame), dtype=np.complex128)

        for ifreq in range(n_freq):
            invRd = np.linalg.solve(Rxx, IRR[:, ifreq, :])
            projection = np.zeros(n_frame, dtype=np.complex128)
            for iframe in range(n_frame):
                projection[iframe] = np.dot(
                    IRR[:, ifreq, iframe].conj(), invRd[:, iframe]
                )
            W[ifreq, :, :] = np.conj(invRd / projection)
        return W

    # ---- Single direction processing ------------------------------

    def _process_one(self, speaker: int, degrees: int, rep: Optional[int] = None):
        """
        Beamform the filtered audio towards one direction.

        Uses precomputed steering vector from cache.
        """
        try:
            IRR = self.ir_cache.load(speaker, degrees, rep)
        except FileNotFoundError:
            print(f"    SKIP (not in cache): p={speaker} d={degrees}")
            return

        print(f"    Beamforming: p={speaker} d={degrees} ...")

        # IRR from cache is [n_channels, n_freq]
        # Expand to [n_channels, n_freq, 1] (single-frame constant steering)
        IRR = IRR.reshape(IRR.shape[0], IRR.shape[1], 1)
        n_channel = IRR.shape[0]

        Rxx = np.eye(n_channel)

        # Compute weights: [n_freq, n_channels, 1]
        W = self._compute_weights(Rxx, IRR)

        # Apply: Z = sum(W * Y, over channels) -> [1, n_freq, n_frames]
        # W: [n_freq, n_channels, 1] broadcasts over n_frames
        Z = np.sum(W * self.Y, axis=1, keepdims=True)

        del W

        # ISTFT -> time domain
        z = librosa.istft(
            Z[:, 0, :], hop_length=self.frameinc, window="hamming"
        )
        del Z

        # Normalise
        zmax = np.max(np.abs(z))
        if zmax > 0:
            z = np.real(z) / zmax

        # Build output filename
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
        sf.write(
            out_path,
            (z * 32767).clip(-32768, 32767).astype("int16"),
            self.fs,
            subtype="PCM_16",
        )
        print(f"      -> {out_filename}")

    # ---- Run all combinations -------------------------------------

    def run(self):
        """Run beamforming for ALL (param x degree) combinations."""
        start = time.time()
        total = 0

        zenith = self.ir_type.zenith_speakers or set()

        for param in self.ir_type.param_values:
            if param in zenith:
                degrees = [0]
            else:
                degrees = self.ir_type.degree_values

            for deg in degrees:
                reps = self.ir_type.rep_values or [None]
                for rep in reps:
                    self._process_one(param, deg, rep)
                    total += 1

        elapsed = time.time() - start
        print(f"  Beamforming complete: {total} outputs in {elapsed:.1f}s")
