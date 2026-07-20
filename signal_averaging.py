"""
Direct Sum Signal Averaging.

Sums all 6 channels and divides by 6 to produce a single-channel
output. Applies a high-pass filter at configurable cutoff.
"""

import os
import time
import numpy as np
import scipy.signal as signal
import librosa
import soundfile as sf

from config import FS_TARGET, N_CHANNELS_EXPECTED
from audio_utils import safe_load_audio


class SignalAverager:
    """6-channel → 1-channel direct-sum averaging."""

    def __init__(
        self,
        flac_path: str,
        output_dir: str,
        fc_high: int = 1000,
    ):
        self.flac_path = flac_path
        self.output_dir = output_dir
        self.fs = FS_TARGET
        self.base_name = os.path.splitext(os.path.basename(flac_path))[0]

        # High-pass filter
        nyq = self.fs / 2
        Wp = fc_high / nyq
        Ws = Wp / 2
        Rp, Rs = 3, 40
        self.n, self.Wn = signal.buttord(Wp, Ws, Rp, Rs)
        self.b, self.a = signal.butter(self.n, self.Wn, btype="high")

        # Load
        print(f"  Reading audio: {flac_path}")
        self.raw = safe_load_audio(flac_path, sr=self.fs, mono=False)

        if self.raw.shape[0] != N_CHANNELS_EXPECTED:
            raise ValueError(
                f"Expected {N_CHANNELS_EXPECTED} channels, got {self.raw.shape[0]}"
            )

        # Filter
        self.filtered = signal.filtfilt(self.b, self.a, self.raw, axis=-1)

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """Sum channels and save."""
        start = time.time()

        output = np.sum(self.filtered, axis=0) / float(N_CHANNELS_EXPECTED)

        # Prevent clipping
        amax = np.max(np.abs(output))
        if amax > 1.0:
            output = output / amax

        out_path = os.path.join(self.output_dir, f"{self.base_name}_sa.wav")
        sf.write(out_path, (output * 32767).clip(-32768, 32767).astype("int16"), self.fs, subtype="PCM_16")
        elapsed = time.time() - start
        print(f"  ✓ Signal averaging: {out_path} ({elapsed:.1f}s)")

        del self.raw, self.filtered, output
