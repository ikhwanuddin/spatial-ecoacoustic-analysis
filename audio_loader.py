"""
Robust audio loading and automatic on-the-fly recovery for FLAC files.
Handles sync loss, corrupted headers, and truncated audio streams gracefully.
"""

import glob
import os
import shutil
import subprocess
import tempfile
from typing import List, Optional, Tuple

import librosa
import numpy as np
import soundfile as sf

_FFMPEG_CANDIDATES = [
    "ffmpeg",
    "/sw-eb/software/FFmpeg/4.4.2-GCCcore-11.3.0/bin/ffmpeg",
    "/sw-eb/software/FFmpeg/4.3.2-GCCcore-11.2.0/bin/ffmpeg",
    "/sw-eb/software/FFmpeg/4.3.2-GCCcore-10.3.0/bin/ffmpeg",
    "/opt/homebrew/bin/ffmpeg",
    "/usr/bin/ffmpeg",
    "/usr/local/bin/ffmpeg",
]

_FLAC_CANDIDATES = [
    "flac",
    "/opt/homebrew/bin/flac",
    "/usr/bin/flac",
    "/usr/local/bin/flac",
]


def find_executable(candidates: List[str]) -> Optional[str]:
    """Find the first available executable from a list of candidate paths."""
    for c in candidates:
        if os.path.isabs(c) and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
        found = shutil.which(c)
        if found:
            return found
    # Try dynamic glob for HPC environment modules
    for pattern in [
        "/sw-eb/software/FFmpeg/*/bin/ffmpeg",
        "/sw-eb/software/FLAC/*/bin/flac",
        "/rds/easybuild/*/software/FFmpeg/*/bin/ffmpeg",
    ]:
        matches = sorted(glob.glob(pattern), reverse=True)
        if matches and os.access(matches[0], os.X_OK):
            return matches[0]
    return None


def repair_flac_to_wav(
    flac_path: str, output_wav: str, target_sr: int = 16000
) -> bool:
    """Attempt to repair/decode a corrupted FLAC file into a clean WAV file."""
    ffmpeg_bin = find_executable(_FFMPEG_CANDIDATES)
    if ffmpeg_bin:
        try:
            cmd = [
                ffmpeg_bin,
                "-y",
                "-err_detect",
                "ignore_err",
                "-i",
                flac_path,
                "-ar",
                str(target_sr),
                "-c:a",
                "pcm_s16le",
                output_wav,
            ]
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if os.path.isfile(output_wav) and os.path.getsize(output_wav) > 1024:
                return True
        except Exception as e:
            print(f"    [Auto-Recovery] FFmpeg repair attempt failed: {e}")

    flac_bin = find_executable(_FLAC_CANDIDATES)
    if flac_bin:
        try:
            cmd = [flac_bin, "-F", "-d", flac_path, "-o", output_wav]
            subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if os.path.isfile(output_wav) and os.path.getsize(output_wav) > 1024:
                return True
        except Exception as e:
            print(f"    [Auto-Recovery] FLAC tool repair attempt failed: {e}")

    return False


def load_audio_robust(
    flac_path: str,
    target_sr: int = 16000,
    expected_channels: int = 6,
    mono: bool = False,
) -> Tuple[np.ndarray, bool]:
    """Loads FLAC audio with automatic recovery for corrupted or truncated files.

    Returns:
        (audio_array, was_repaired)
        where audio_array has shape (channels, samples) or (samples,) if mono=True.
    """
    # 1. Normal fast path
    try:
        data, sr = sf.read(flac_path, dtype="float32", always_2d=True)
        if sr != target_sr:
            data = librosa.resample(data.T, orig_sr=sr, target_sr=target_sr).T
        audio = data.T if not mono else np.mean(data, axis=1)
        if not mono and audio.shape[0] == expected_channels:
            return audio, False
        elif not mono and audio.shape[0] != expected_channels:
            if audio.shape[1] == expected_channels:
                return audio.T, False
        elif mono:
            return audio, False
    except Exception as orig_exc:
        print(f"  ⚠️ Decode warning on {os.path.basename(flac_path)}: {orig_exc}")
        print(f"  🛠️  Attempting automatic on-the-fly audio recovery...")

    # 2. Resilient recovery path
    tmp_base = os.environ.get(
        "TMPDIR",
        os.path.expanduser(
            "~/ephemeral/tmp" if os.path.isdir(os.path.expanduser("~/ephemeral")) else "/tmp"
        ),
    )
    os.makedirs(tmp_base, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(flac_path))[0]
    tmp_wav = os.path.join(tmp_base, f"_repaired_{base_name}_{os.getpid()}.wav")

    try:
        ok = repair_flac_to_wav(flac_path, tmp_wav, target_sr=target_sr)
        if ok:
            data, sr = sf.read(tmp_wav, dtype="float32", always_2d=True)
            audio = data.T if not mono else np.mean(data, axis=1)
            duration_sec = (
                audio.shape[1] / target_sr if audio.ndim > 1 else len(audio) / target_sr
            )
            n_ch = audio.shape[0] if audio.ndim > 1 else 1

            if not mono and n_ch != expected_channels:
                raise ValueError(
                    f"Recovered audio has {n_ch} channels, expected {expected_channels}"
                )

            if duration_sec < 0.5:
                raise ValueError(
                    f"Recovered audio is too short ({duration_sec:.2f}s < 0.5s)"
                )

            print(
                f"  ✅ Auto-recovery successful: {duration_sec:.1f}s salvaged ({n_ch} channels)"
            )
            return audio, True
        else:
            raise RuntimeError(
                f"Auto-recovery failed: unable to decode audio from {flac_path}"
            )
    finally:
        if os.path.exists(tmp_wav):
            try:
                os.remove(tmp_wav)
            except OSError:
                pass
