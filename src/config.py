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
RTF_BASE_PATH = os.path.join(HOME_DIR, "MAARU-IR-RTF")   # steering vectors (RTF .npz)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")
QUEUE_DIR = os.path.join(PROJECT_ROOT, "queue")
TELEMETRY_DIR = os.path.join(PROJECT_ROOT, "telemetry")

# Ephemeral storage (30-day purge, Scratch renders & Raw audio)
EPHEM_DIR = f"/rds/general/user/{USER}/ephemeral"
MONITORING_DATA = os.path.join(EPHEM_DIR, "monitoring_data")
SCRATCH_DIR = os.path.join(EPHEM_DIR, "sea-scratch")

# ============================================================
# AUDIO & DSP PARAMETERS
# ============================================================
FS_TARGET = 16000          # 16 kHz sampling rate across all experiments
HIGH_PASS_CUTOFF = 500     # 500 Hz Butterworth high-pass filter (per paper)

# STFT parameters (20 ms window, 10 ms hop)
FRAME_LEN_SEC = 0.02
FRAME_LEN = int(FRAME_LEN_SEC * FS_TARGET)  # 320 samples
HOP_LEN = FRAME_LEN // 2                   # 160 samples

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

# O0: every recording is empty (mic cable came loose), so it is left out of SEA / beamforming.
SKIP_LOCATIONS = {"O0"}

# Microphones: every array has a 6-mic hexagonal ring on CH0-5 (default: ReSpeaker 6-Mic, 6 ch FLAC).
# 2B400 is a Sipeed 6+1 (8 ch FLAC): CH6 is the on-board beamformed output, CH7 the centre mic.
# Any FLAC can be steered with any RTF using the 6 ring mics; only an 8 ch FLAC with an 8 ch (Sipeed)
# RTF also uses the centre mic (7 mics).
RING_MICS = [0, 1, 2, 3, 4, 5]
CENTRE_MIC = 7
FLAC_CHANNELS = {
    "2B400": 8,
}

# ============================================================
# CANON BEAM SUBSETS (Standard Grids)
# ============================================================
# LabIR: 133 beams (S01-S11 = 11 elevations -45..75 deg x 12 azimuths every 30 deg + S12 zenith x 1)
LABIR_SPEAKERS = list(range(1, 13))
LABIR_DEGREES = list(range(0, 360, 30))

# SPIR: 30 beams (SPIR1 all 23 measured positions, 4 distances x 6 azimuths without 8 m / 60 deg;
# SPIR2 7 distances x 1 azimuth = 7)
# WCIRown: the Way Canguk RTFs of the same location (12 per set);
# WCIRcross: the Way Canguk RTFs of every other location. All locations get all 247 beams.
SPIR2_DISTANCES = [1, 2, 4, 8, 16, 32, 64]
SPIR2_DEGREES = [180]
SPIR2_REP = 2

# ============================================================
# EVALUATION THRESHOLDS
# ============================================================
DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]
