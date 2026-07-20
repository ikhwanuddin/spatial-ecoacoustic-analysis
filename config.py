"""
Configuration for spatial-ecoacoustic-analysis pipeline.
All paths, location mappings, and IR type definitions.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ============================================================
# BASE PATHS
# ============================================================

# Project root (python code lives here)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# External data volume
HD_DATA = "/Volumes/HD Data"

# Raw monitoring data (FLAC recordings)
MONITORING_DATA = os.path.join(HD_DATA, "monitoring_data")

# Analysis output (beamforming, signal averaging, BirdNET results)
ANALYSIS_OUTPUT = os.path.join(HD_DATA, "sea-data")

# Impulse Response files
IR_BASE_PATH = os.path.join(os.path.dirname(PROJECT_ROOT), "MAARU-Impulse-Response")

# ============================================================
# LOCATION → RPiID MAPPING
# ============================================================
LOCATION_MAP: Dict[str, str] = {
    "S0":    "RPiID-000000003bdd60a1",   # ends 60a1
    "2D400": "RPiID-00000000058096e0",   # ends 96e0
    "Q0":    "RPiID-000000005acf5969",   # ends 5969
    "2A400": "RPiID-0000000091668b26",   # ends 8b26
    "O0":    "RPiID-000000009c3f398b",   # ends 398b
    "2B400": "RPiID-00000000a1e24a04",   # ends 4a04
}

# Reverse lookup
RPIID_TO_LOCATION: Dict[str, str] = {v: k for k, v in LOCATION_MAP.items()}

# ============================================================
# IR TYPE DEFINITIONS
# ============================================================

@dataclass
class IRType:
    """Defines configuration for one impulse response type."""
    name: str                          # "LabIR", "SPIR1", "SPIR2"
    folder: str                        # Subfolder under MAARU-Impulse-Response/
    use_dual_filter: bool = False      # HP only or HP+LP
    fc_high: int = 1000                # High-pass cutoff
    fc_low: int = 4000                 # Low-pass cutoff (only if use_dual_filter)
    fs_target: int = 16000
    fs_ir_original: int = 48000

    # Iteration parameters
    param_label: str = "speaker"       # "speaker" or "distance"
    param_values: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    degree_values: List[int] = field(default_factory=lambda: [0, 60, 120, 180, 240, 300])
    rep_values: Optional[List[int]] = None  # Only for SPIR2
    zenith_speakers: Optional[set] = None   # Speakers with no azimuth variation

    # Filename patterns
    ir_filename_pattern: str = ""      # e.g. "Lab_IR_S{speaker:02d}_{degrees:03d}.wav"
    output_suffix_pattern: str = ""    # e.g. "LabIR(S{speaker}_{degrees:03d})"


# ── LabIR: 12 elevations × 36 azimuths (step 10°) ──────────────────
LAB_IR = IRType(
    name="LabIR",
    folder="Lab_IR",
    use_dual_filter=True,
    fc_high=1000,
    fc_low=4000,
    param_label="speaker",
    # Default: all 12 speakers, 0-350 step 10.
    # S12 (zenith / speaker directly above) has no azimuth variation,
    # so it is handled specially in the beamformer.
    param_values=list(range(1, 13)),
    degree_values=list(range(0, 360, 10)),
    zenith_speakers={12},
    ir_filename_pattern="Lab_IR_S{speaker:02d}_{degrees:03d}.wav",
    output_suffix_pattern="LabIR(S{speaker:02d}_{degrees:03d})",
)

# ── SPIR1: 4 distances × 6 azimuths (step 60°) ─────────────────────
SP_IR1 = IRType(
    name="SPIR1",
    folder="SP_IR1",
    use_dual_filter=False,
    fc_high=1000,
    param_label="distance",
    param_values=[2, 4, 8, 16],
    degree_values=[0, 60, 120, 180, 240, 300],
    ir_filename_pattern="SP_IR_{distance:02d}m_{degrees:03d}.wav",
    output_suffix_pattern="SPIR1({distance:02d}m_{degrees:03d})",
)

# ── SPIR2: 7 distances × 1 azimuth × 3 reps ────────────────────────
SP_IR2 = IRType(
    name="SPIR2",
    folder="SP_IR2",
    use_dual_filter=False,
    fc_high=1000,
    param_label="distance",
    param_values=[1, 2, 4, 8, 16, 32, 64],
    degree_values=[180],
    rep_values=[1, 2, 3],
    ir_filename_pattern="{distance:02d}m_180_{rep}.wav",
    output_suffix_pattern="SPIR2({distance:02d}m_180_r{rep})",
)

IR_TYPES: Dict[str, IRType] = {
    "LabIR": LAB_IR,
    "SPIR1": SP_IR1,
    "SPIR2": SP_IR2,
}

# ============================================================
# PROTOTYPE SUBSETS (for quicker testing)
# ============================================================
# Reduced parameter sets for prototyping to avoid excessive computation.
# When pipeline is mature, switch to full sets.

PROTOTYPE_IR_SUBSETS = {
    "LabIR": IRType(
        name="LabIR",
        folder="Lab_IR",
        use_dual_filter=True,
        fc_high=1000,
        fc_low=4000,
        param_label="speaker",
        param_values=[1, 5, 9, 12],         # 4 elevations
        degree_values=[0, 60, 120, 180, 240, 300],  # 6 azimuths
        zenith_speakers={12},
        ir_filename_pattern="Lab_IR_S{speaker:02d}_{degrees:03d}.wav",
        output_suffix_pattern="LabIR(S{speaker:02d}_{degrees:03d})",
    ),
    "SPIR1": SP_IR1,   # Use full (only 24 combos)
    "SPIR2": SP_IR2,   # Use full (only 21 combos)
}

# ============================================================
# BIRDNET CONFIG
# ============================================================
LOCATION_COORDS = {
    "waycanguk": {"lat": -5.6585004, "lon": 104.4046997},
    "silwood":   {"lat": 51.409111,  "lon": -0.637820},
}

BIRDNET_MIN_CONF = 0.4
BIRDNET_OVERLAP = 0.0

# ============================================================
# AUDIO CONFIG
# ============================================================
FS_TARGET = 16000
FS_IR_ORIGINAL = 48000
N_CHANNELS_EXPECTED = 6

# STFT params
FRAME_LEN_SEC = 0.02
