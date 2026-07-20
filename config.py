"""
Configuration for spatial-ecoacoustic-analysis pipeline.
All paths, location mappings, and IR type definitions.

Paths can be overridden via environment variables for Colab:
  export MONITORING_DATA=/drive/MyDrive/monitoring_data
  export ANALYSIS_OUTPUT=/drive/MyDrive/sea-data
  export IR_BASE_PATH=/drive/MyDrive/MAARU-Impulse-Response
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ============================================================
# BASE PATHS (with env var overrides for Colab)
# ============================================================

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# External data volume (HDD) — Mac Mini default
HD_DATA = "/Volumes/HD Data"

# Raw monitoring data (FLAC recordings)
# Set MONITORING_DATA env var for GDrive: /drive/MyDrive/monitoring_data
MONITORING_DATA = os.environ.get("MONITORING_DATA", os.path.join(HD_DATA, "monitoring_data"))

# SSD volume — Mac Mini default
SSD_DATA = "/Volumes/WD2TB"

# Analysis output
# Set ANALYSIS_OUTPUT env var for GDrive: /drive/MyDrive/sea-data
ANALYSIS_OUTPUT = os.environ.get("ANALYSIS_OUTPUT", os.path.join(SSD_DATA, "sea-data"))

# Impulse Response files
# Set IR_BASE_PATH env var for GDrive: /drive/MyDrive/MAARU-Impulse-Response
IR_BASE_PATH = os.environ.get(
    "IR_BASE_PATH",
    os.path.join(os.path.dirname(PROJECT_ROOT), "MAARU-Impulse-Response"),
)

# ============================================================
# LOCATION -> RPiID MAPPING
# ============================================================
LOCATION_MAP: Dict[str, str] = {
    "S0":    "RPiID-000000003bdd60a1",
    "2D400": "RPiID-00000000058096e0",
    "Q0":    "RPiID-000000005acf5969",
    "2A400": "RPiID-0000000091668b26",
    "O0":    "RPiID-000000009c3f398b",
    "2B400": "RPiID-00000000a1e24a04",
}

RPIID_TO_LOCATION: Dict[str, str] = {v: k for k, v in LOCATION_MAP.items()}

# ============================================================
# IR TYPE DEFINITIONS
# ============================================================

@dataclass
class IRType:
    """Defines configuration for one impulse response type."""
    name: str
    folder: str
    use_dual_filter: bool = False
    fc_high: int = 1000
    fc_low: int = 4000
    fs_target: int = 16000
    fs_ir_original: int = 48000
    param_label: str = "speaker"
    param_values: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    degree_values: List[int] = field(default_factory=lambda: [0, 60, 120, 180, 240, 300])
    rep_values: Optional[List[int]] = None
    zenith_speakers: Optional[set] = None
    ir_filename_pattern: str = ""
    output_suffix_pattern: str = ""


LAB_IR = IRType(
    name="LabIR", folder="Lab_IR", use_dual_filter=True,
    fc_high=1000, fc_low=4000, param_label="speaker",
    param_values=list(range(1, 13)), degree_values=list(range(0, 360, 10)),
    zenith_speakers={12},
    ir_filename_pattern="Lab_IR_S{speaker:02d}_{degrees:03d}.wav",
    output_suffix_pattern="LabIR(S{speaker:02d}_{degrees:03d})",
)

SP_IR1 = IRType(
    name="SPIR1", folder="SP_IR1", use_dual_filter=False,
    fc_high=1000, param_label="distance",
    param_values=[2, 4, 8, 16], degree_values=[0, 60, 120, 180, 240, 300],
    ir_filename_pattern="SP_IR_{distance:02d}m_{degrees:03d}.wav",
    output_suffix_pattern="SPIR1({distance:02d}m_{degrees:03d})",
)

SP_IR2 = IRType(
    name="SPIR2", folder="SP_IR2", use_dual_filter=False,
    fc_high=1000, param_label="distance",
    param_values=[1, 2, 4, 8, 16, 32, 64], degree_values=[180],
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
# PRODUCTION SUBSETS (for 2A400 Colab batch processing)
# ============================================================

PRODUCTION_IR_SUBSETS = {
    "LabIR": IRType(
        name="LabIR", folder="Lab_IR", use_dual_filter=True,
        fc_high=1000, fc_low=4000, param_label="speaker",
        param_values=[1, 5, 9, 12], degree_values=[0, 60, 120, 180, 240, 300],
        zenith_speakers={12},
        ir_filename_pattern="Lab_IR_S{speaker:02d}_{degrees:03d}.wav",
        output_suffix_pattern="LabIR(S{speaker:02d}_{degrees:03d})",
    ),
    "SPIR1": SP_IR1,
    "SPIR2": IRType(
        name="SPIR2", folder="SP_IR2", use_dual_filter=False,
        fc_high=1000, param_label="distance",
        param_values=[1, 2, 4, 8, 16, 32, 64], degree_values=[180],
        rep_values=[2],  # Only repetition 2 (user requested)
        ir_filename_pattern="{distance:02d}m_180_{rep}.wav",
        output_suffix_pattern="SPIR2({distance:02d}m_180_r{rep})",
    ),
}

# Alias for backward compatibility
PROTOTYPE_IR_SUBSETS = PRODUCTION_IR_SUBSETS

# ============================================================
# BIRDNET CONFIG
# ============================================================

BIRDNET_FP16_MODEL = True

LOCATION_COORDS = {
    "waycanguk": {"lat": -5.6585004, "lon": 104.4046997},
    "silwood":   {"lat": 51.409111,  "lon": -0.637820},
}

SITE_COORDS: Dict[str, Dict[str, float]] = {
    "S0":    {"lat": -5.6585004, "lon": 104.4046997},
    "2D400": {"lat": -5.6585004, "lon": 104.4046997},
    "Q0":    {"lat": -5.6585004, "lon": 104.4046997},
    "2A400": {"lat": -5.6585004, "lon": 104.4046997},
    "O0":    {"lat": -5.6585004, "lon": 104.4046997},
    "2B400": {"lat": -5.6585004, "lon": 104.4046997},
}

BIRDNET_MIN_CONF = 0.4
BIRDNET_OVERLAP = 0.0

# ============================================================
# AUDIO CONFIG
# ============================================================
FS_TARGET = 16000
FS_IR_ORIGINAL = 48000
N_CHANNELS_EXPECTED = 6
FRAME_LEN_SEC = 0.02
