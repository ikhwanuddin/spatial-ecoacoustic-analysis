"""
Configuration for Spatial Ecoacoustic Analysis (SEA).
Clean, direct, practical (KISS). No backward-compatibility bloat.
"""

import os

# ============================================================
# FILESYSTEM PATHS (CX3 HPC)
# ============================================================
USER = os.environ.get("USER", "ri322")

# Permanent storage in HOME (Code, Config, Outputs, Visualizations)
HOME_DIR = f"/rds/general/user/{USER}/home"
PROJECT_ROOT = os.path.join(HOME_DIR, "spatial-ecoacoustic-analysis")
IR_BASE_PATH = os.path.join(HOME_DIR, "MAARU-Impulse-Response")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Ephemeral storage (30-day purge, Scratch renders & Raw audio)
EPHEM_DIR = f"/rds/general/user/{USER}/ephemeral"
MONITORING_DATA = os.path.join(EPHEM_DIR, "monitoring_data")
SCRATCH_DIR = os.path.join(EPHEM_DIR, "sea-scratch")

# ============================================================
# AUDIO & DSP PARAMETERS
# ============================================================
FS_TARGET = 16000          # 16 kHz sampling rate across all experiments
FS_IR_ORIGINAL = 48000     # Raw IR sampling rate
HIGH_PASS_CUTOFF = 500     # 500 Hz Butterworth high-pass filter (per paper)

# STFT parameters (20 ms window, 10 ms hop)
FRAME_LEN_SEC = 0.02
FRAME_LEN = int(FRAME_LEN_SEC * FS_TARGET)  # 320 samples
HOP_LEN = FRAME_LEN // 2                   # 160 samples

# Onset-aligned IR window parameters (KISS: 64 samples centered on peak)
IR_WINDOW_LEN = 64
IR_PRE_PEAK_SAMPLES = 16   # 1 ms before peak

# BirdNET evaluation
WINDOW_LEN_SEC = 3.0       # 3.0 seconds decision window
WINDOW_HOP_SEC = 3.0       # Non-overlapping windows (overlap = 0)

# ============================================================
# RECORDER TO LOCATION MAPPING
# ============================================================
LOCATION_MAP = {
    "2A400": "RPiID-0000000091668b26",
    "2D400": "RPiID-00000000058096e0",
    "S0":    "RPiID-000000003bdd60a1",
    "Q0":    "RPiID-000000005acf5969",
    "O0":    "RPiID-000000009c3f398b",
    "2B400": "RPiID-00000000a1e24a04",
}
RPIID_TO_LOCATION = {v: k for k, v in LOCATION_MAP.items()}

# ============================================================
# CANON BEAM SUBSETS (Standard Grids)
# ============================================================
# LabIR: 19 beams (S01, S05, S09 x 6 azimuths + S12 zenith x 1)
LABIR_SPEAKERS = [1, 5, 9, 12]
LABIR_DEGREES = [0, 60, 120, 180, 240, 300]

# SPIR: 31 beams (SPIR1 4 distances x 6 azimuths = 24; SPIR2 7 distances x 1 azimuth = 7)
SPIR1_DISTANCES = [2, 4, 8, 16]
SPIR1_DEGREES = [0, 60, 120, 180, 240, 300]
SPIR2_DISTANCES = [1, 2, 4, 8, 16, 32, 64]
SPIR2_DEGREES = [180]
SPIR2_REP = 2
