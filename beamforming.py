"""
Impulse Response-based Multidirectional Beamforming.

Supports three IR types:
- LabIR:  Virtual sound environment, 12 elevations × 36 azimuths
- SPIR1:  Silwood Park forest, 4 distances × 6 azimuths
- SPIR2:  Silwood Park forest, 7 distances × 1 azimuth × 3 reps
"""

import os
import time
import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf
from typing import Optional

from config import (
    FS_TARGET,
    FS_IR_ORIGINAL,
    N_CHANNELS_EXPECTED,
    FRAME_LEN_SEC,
    IRType,
)


class Beamformer:
    """
    Multidirectional beamforming using impulse response steering vectors.

    For a given FLAC recording and IR type, produces one WAV output per
    (param, degree) combination — or (param, degree, rep) for SPIR2.
    """

    def __init__(
        self,
        flac_path: str,
        output_dir: str,
        ir_type: IRType,
        ir_base_path: str,
    ):
        """
        Args:
            flac_path:   Absolute path to the input .flac file.
            output_dir:  Directory where beamformed WAVs are saved.
            ir_type:     IRType dataclass with all IR parameters.
            ir_base_path: Path to MAARU-Impulse-Response/ root.
        """
        self.flac_path = flac_path
        self.output_dir = output_dir
        self.ir_type = ir_type
        self.ir_folder = os.path.join(ir_base_path, ir_type.folder)

        self.fs = FS_TARGET
        self.fs_ir_original = FS_IR_ORIGINAL
        self.fc_high = ir_type.fc_high
        self.fc_low = ir_type.fc_low

        # Derive base name from FLAC path (without .flac extension)
        self.base_name = os.path.splitext(os.path.basename(flac_path))[0]

        # Build high-pass filter
        nyq = self.fs / 2
        Wp = ir_type.fc_high / nyq
        Ws = Wp / 2
        Rp, Rs = 3, 40
        self.n_hp, self.Wn_hp = signal.buttord(Wp, Ws, Rp, Rs)
        self.b_hp, self.a_hp = signal.butter(self.n_hp, self.Wn_hp, btype="high")

        # Optionally build low-pass filter
        if ir_type.use_dual_filter:
            Wp_lp = ir_type.fc_low / nyq
            Ws_lp = Wp_lp * 1.2
            self.n_lp, self.Wn_lp = signal.buttord(Wp_lp, Ws_lp, Rp, Rs)
            self.b_lp, self.a_lp = signal.butter(self.n_lp, self.Wn_lp, btype="low")

        # Load and filter audio
        print(f"  Reading audio: {flac_path}")
        self.raw, _ = librosa.load(flac_path, sr=self.fs, mono=False)
        self.filtered_audio = self._apply_filters(self.raw)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

    # ── Filtering ──────────────────────────────────────────────

    def _apply_filters(self, audio: np.ndarray) -> np.ndarray:
        """Apply HP (and optionally LP) filter to multi-channel audio."""
        filtered = signal.filtfilt(self.b_hp, self.a_hp, audio, axis=-1)
        if self.ir_type.use_dual_filter:
            filtered = signal.filtfilt(self.b_lp, self.a_lp, filtered, axis=-1)
        return filtered

    # ── Beamforming weights computation ────────────────────────

    def _compute_weights(self, Rxx: np.ndarray, IRR: np.ndarray) -> np.ndarray:
        """
        Compute MVDR-style beamforming weights.
        IRR: [n_channels, n_freq, n_frame]
        Rxx: [n_channels, n_channels]  (identity → delay-and-sum)
        Returns W: [n_freq, n_channels, n_frame] complex128
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

    # ── Single IR processing ───────────────────────────────────

    def _process_one(self, speaker: int, degrees: int, rep: Optional[int] = None):
        """
        Beamform the filtered audio using one IR file.

        The IR filename is constructed from ir_filename_pattern using
        the appropriate placeholder names (speaker / distance, degrees, rep).
        """
        # Build IR filename
        fmt_kwargs = {}
        if self.ir_type.param_label == "speaker":
            fmt_kwargs["speaker"] = speaker
        else:
            fmt_kwargs["distance"] = speaker
        fmt_kwargs["degrees"] = degrees
        if rep is not None:
            fmt_kwargs["rep"] = rep

        try:
            ir_file = self.ir_type.ir_filename_pattern.format(**fmt_kwargs)
        except KeyError as e:
            print(f"  ⚠  Skipping: missing key {e} in pattern")
            return

        ir_path = os.path.join(self.ir_folder, ir_file)
        if not os.path.exists(ir_path):
            print(f"  ⚠  IR not found, skipping: {ir_path}")
            return

        print(f"    Beamforming: {ir_file} ...")

        # Load and resample IR
        ir_orig, _ = librosa.load(ir_path, sr=self.fs_ir_original, mono=False)
        ir = librosa.resample(ir_orig, orig_sr=self.fs_ir_original, target_sr=self.fs)

        # STFT of filtered audio
        framelen = int(FRAME_LEN_SEC * self.fs)
        frameinc = framelen // 2
        X = librosa.stft(
            self.filtered_audio, n_fft=framelen, hop_length=frameinc, window="hamming"
        )  # → [freq, channel, frame]

        if X.ndim == 2:
            X = X[np.newaxis, :, :]  # add channel dim if mono

        # FFT of IR → relative transfer function
        IR = np.fft.rfft(ir, n=framelen, axis=-1)
        n_channel = IR.shape[0]
        IRR = IR / (IR[0, :] + 1e-12)  # normalise by ref channel

        if IRR.ndim == 2:
            IRR = IRR.reshape(IRR.shape[0], IRR.shape[1], 1)

        Rxx = np.eye(n_channel)  # Identity — delay-and-sum

        # Compute weights and apply
        W = self._compute_weights(Rxx, IRR)
        Y = np.transpose(X, (1, 0, 2))  # [channel, freq, frame]
        Z = np.sum(W * Y, axis=1, keepdims=True)

        del W, Y, X, IR, IRR

        # ISTFT → time-domain signal
        z = librosa.istft(Z[:, 0, :], hop_length=frameinc, window="hamming")
        del Z

        # Normalise
        zmax = np.max(np.abs(z))
        if zmax > 0:
            z = np.real(z) / zmax

        # Build output filename suffix
        out_fmt = {**fmt_kwargs}
        out_suffix = self.ir_type.output_suffix_pattern.format(**out_fmt)
        out_filename = f"{self.base_name}_{out_suffix}.wav"
        out_path = os.path.join(self.output_dir, out_filename)
        sf.write(out_path, (z * 32767).clip(-32768, 32767).astype("int16"), self.fs, subtype="PCM_16")
        print(f"      → {out_filename}")

        del z, ir, ir_orig

    # ── Run all combinations ───────────────────────────────────

    def run(self):
        """Run beamforming for ALL (param × degree) combinations."""
        start = time.time()
        total = 0

        zenith = self.ir_type.zenith_speakers or set()

        for param in self.ir_type.param_values:
            # Zenith speakers (e.g. S12 in LabIR) only have 0° azimuth.
            if param in zenith:
                degrees = [0]
            else:
                degrees = self.ir_type.degree_values

            for deg in degrees:
                if self.ir_type.rep_values is not None:
                    for rep in self.ir_type.rep_values:
                        self._process_one(param, deg, rep)
                        total += 1
                else:
                    self._process_one(param, deg)
                    total += 1

        elapsed = time.time() - start
        print(f"  ✓ Beamforming complete: {total} outputs in {elapsed:.1f}s")

        # Free memory
        del self.filtered_audio, self.raw
