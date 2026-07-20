"""
Audio I/O utilities — safe multi-channel loading for malformed/corrupt FLAC files.

Handles two classes of problem files:

1. FLAC files with broken STREAMINFO header (total-samples = INT64_MAX)
   - soundfile allocates impossible arrays. Fixed by block-wise reading.

2. FLAC files with corrupt/sync-lost stream data
   - libsndfile fails entirely ("flac decoder lost sync"). Fixed by falling
     back to ffmpeg, which is more tolerant of stream corruption.

Strategy: try librosa -> try soundfile blocks -> try ffmpeg subprocess.
"""

import os
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import librosa


def _decode_via_ffmpeg(path: str, sr: int, n_channels: int = 6) -> np.ndarray:
    """
    Decode audio via ffmpeg subprocess to interleaved float32 raw PCM,
    then reshape to (n_channels, n_samples).

    ffmpeg often handles corrupt FLAC streams that crash libsndfile.
    """

    with tempfile.NamedTemporaryFile(suffix=".raw", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        cmd = [
            "ffmpeg", "-v", "error", "-i", path,
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", str(n_channels),
            "-ar", str(sr),
            "-y", tmp_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"ffmpeg decode failed: {result.stderr[:500]}"
            )

        raw = np.fromfile(tmp_path, dtype=np.float32)
        if raw.size == 0:
            raise RuntimeError(f"ffmpeg decoded 0 samples from {path}")

        n_samples = raw.size // n_channels
        audio = raw[:n_samples * n_channels].reshape(n_samples, n_channels)
        return audio.T  # (n_channels, n_samples)

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def safe_load_audio(path: str, sr: int, mono: bool = False) -> np.ndarray:
    """
    Load a multi-channel audio file safely with three fallback strategies.

    Returns:
        np.ndarray: shape (n_channels, n_samples) for mono=False,
                    shape (n_samples,) for mono=True.
    """
    # ── Attempt 1: librosa.load (works for >99% of files) ──────
    try:
        audio, _ = librosa.load(path, sr=sr, mono=mono)
        return audio
    except ValueError as e:
        if "array is too big" not in str(e):
            raise
    except Exception:
        pass

    # ── Attempt 2: block-wise soundfile (broken STREAMINFO) ────
    try:
        print("  (block-wise read for malformed FLAC header)")
        with sf.SoundFile(path) as f:
            blocks = []
            for block in f.blocks(blocksize=65536, dtype="float32", always_2d=True):
                blocks.append(block)

        if not blocks:
            raise RuntimeError(f"No audio data read from {path}")

        audio = np.concatenate(blocks, axis=0)

        if f.samplerate != sr:
            print(f"  Resampling {f.samplerate} to {sr} Hz ...")
            audio = librosa.resample(audio.T, orig_sr=f.samplerate, target_sr=sr).T

        if mono:
            return np.mean(audio, axis=1)
        return audio.T

    except Exception:
        pass

    # ── Attempt 3: ffmpeg subprocess (corrupt stream) ──────────
    print("  (ffmpeg fallback for corrupt FLAC stream)")
    audio = _decode_via_ffmpeg(path, sr)

    if mono:
        return np.mean(audio, axis=0)
    return audio
