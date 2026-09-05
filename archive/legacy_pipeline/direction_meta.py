"""Parse azimuth/elevation from spatial method WAV filenames.

Lightweight — no BirdNET/TF imports (safe for bacpipe venv).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

# LabIR speaker → elevation (degrees); matches config.LABIR_SPEAKER_ELEVATION
LABIR_SPEAKER_ELEVATION = {
    1: -45,
    5: 0,
    9: 45,
    12: 90,
}

_LABIR_RE = re.compile(r"LabIR\(S(\d{2})_(\d{3})\)")
_SPIR1_RE = re.compile(r"SPIR1\((\d{2})m_(\d{3})\)")
_SPIR2_RE = re.compile(r"SPIR2\((\d{2})m_(\d{3})_r(\d)\)")


def parse_direction_metadata(wav_name: str) -> Tuple[Optional[int], Optional[int]]:
    """Return (azimuth, elevation) in degrees, or (None, None)."""
    m = _LABIR_RE.search(wav_name)
    if m:
        speaker = int(m.group(1))
        azimuth = int(m.group(2))
        elevation = LABIR_SPEAKER_ELEVATION.get(speaker)
        return (azimuth, elevation)

    m = _SPIR1_RE.search(wav_name)
    if m:
        return (int(m.group(2)), 0)

    m = _SPIR2_RE.search(wav_name)
    if m:
        return (int(m.group(2)), 0)

    return (None, None)
