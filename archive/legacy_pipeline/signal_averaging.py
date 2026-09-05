"""
Direct Sum Signal Averaging.
Sums all 6 channels and divides by 6 to produce a single-channel output.
No filtering applied.
"""

import os
import time
from typing import Optional
import numpy as np
import librosa
import soundfile as sf

from config import FS_TARGET, N_CHANNELS_EXPECTED
from audio_loader import load_audio_robust


class SignalAverager:
    """6-channel -> 1-channel direct-sum averaging (no filter)."""

    def __init__(
        self,
        flac_path: str,
        output_dir: str,
        raw_audio: Optional[np.ndarray] = None,
    ):
        self.flac_path = flac_path
        self.output_dir = output_dir
        self.fs = FS_TARGET
        self.base_name = os.path.splitext(os.path.basename(flac_path))[0]

        # Load audio with resilient recovery if corrupt
        if raw_audio is not None:
            self.raw = raw_audio
        else:
            print(f"  Reading audio: {flac_path}")
            self.raw, _ = load_audio_robust(flac_path, target_sr=self.fs, expected_channels=N_CHANNELS_EXPECTED)

        if self.raw.shape[0] != N_CHANNELS_EXPECTED:
            raise ValueError(
                f"Expected {N_CHANNELS_EXPECTED} channels, got {self.raw.shape[0]}"
            )

        os.makedirs(self.output_dir, exist_ok=True)

    def run(self):
        """Sum channels and save."""
        start = time.time()

        output = np.sum(self.raw, axis=0) / float(N_CHANNELS_EXPECTED)

        # Prevent clipping
        amax = np.max(np.abs(output))
        if amax > 1.0:
            output = output / amax

        out_path = os.path.join(self.output_dir, f"{self.base_name}_sa.wav")
        sf.write(out_path, (output * 32767).clip(-32768, 32767).astype("int16"), self.fs, subtype="PCM_16")
        elapsed = time.time() - start
        print(f"  ✓ Signal averaging: {out_path} ({elapsed:.1f}s)")

        del self.raw, output
