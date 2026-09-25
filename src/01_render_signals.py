"""
Module 1: Signal Processing & Multidirectional Beamforming.
Renders raw multi-channel FLAC recordings into Mono, Signal Averaging (SA),
and matched-filter beamforming steered by measured RTFs (LabIR, SPIR, WCIR).

Vectorized matrix contraction (einsum) & multi-core parallel ISTFT (KISS).
Includes automatic FLAC integrity verification and repair with graceful skip reporting.
"""

import os
import glob
import time
import argparse
import subprocess
import multiprocessing
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa
from typing import Tuple, Optional, Dict, Any

from config import (
    FS_TARGET,
    HIGH_PASS_CUTOFF,
    FRAME_LEN,
    HOP_LEN,
    RTF_BASE_PATH,
    MONITORING_DATA,
    SCRATCH_DIR,
    LABIR_SPEAKERS,
    LABIR_DEGREES,
    SPIR2_DISTANCES,
    SPIR2_REP,
    LOCATION_MAP,
    MIC_CHANNELS,
)

# In-memory steering vector cache for fast reuse across files
_BEAMS = {}   # location -> (catalog, W tensor)


def format_size(num_bytes: int) -> str:
    val = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if abs(val) < 1024.0:
            return f"{val:3.1f} {unit}"
        val /= 1024.0
    return f"{val:.1f} TB"


def load_and_verify_flac(flac_path: str, scratch_dir: str, n_channels: int = 6) -> Tuple[Optional[np.ndarray], Optional[int], Optional[Dict[str, Any]]]:
    """
    Loads n_channels FLAC audio, attempting automatic repair if corrupted.
    Returns:
        (audio_array, sample_rate, None) if successful
        (None, None, error_info_dict) if unrecoverable and must be skipped.
    """
    if not os.path.isfile(flac_path):
        return None, None, {
            "flac_path": flac_path,
            "file_name": os.path.basename(flac_path),
            "file_size_bytes": 0,
            "file_size_human": "0 B",
            "initial_error": "File not found",
            "repair_attempts": [],
            "decision": "SKIPPED",
        }

    file_size = os.path.getsize(flac_path)
    if file_size == 0:
        return None, None, {
            "flac_path": flac_path,
            "file_name": os.path.basename(flac_path),
            "file_size_bytes": 0,
            "file_size_human": "0 B",
            "initial_error": "File is completely empty (0 bytes)",
            "repair_attempts": [],
            "decision": "SKIPPED",
        }

    initial_error = None
    try:
        audio_raw, sr = sf.read(flac_path, dtype="float32")
        duration_sec = len(audio_raw) / sr if sr > 0 else 0.0
        if audio_raw.ndim == 2 and audio_raw.shape[1] == n_channels and duration_sec >= 3.0:
            return audio_raw, sr, None
        elif duration_sec < 3.0:
            initial_error = f"Audio is severely truncated ({duration_sec:.2f}s < 3.0s minimum window)"
        else:
            shape_str = str(getattr(audio_raw, "shape", None))
            initial_error = f"Invalid audio shape: {shape_str} (expected {n_channels} channels)"
    except Exception as e:
        initial_error = str(e)

    # Attempt automatic repair via ffmpeg error-tolerance stream reconstruction
    repair_details = []
    ffmpeg_bin = os.path.expanduser("~/miniforge3/envs/sea/bin/ffmpeg")
    if not os.path.exists(ffmpeg_bin):
        ffmpeg_bin = "ffmpeg"

    repaired_path = os.path.join(scratch_dir, "repaired_audio.flac")
    cmd = [
        ffmpeg_bin, "-y", "-v", "error",
        "-err_detect", "ignore_err",
        "-i", flac_path,
        "-c:a", "flac",
        repaired_path
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if os.path.exists(repaired_path) and os.path.getsize(repaired_path) > 0:
            rep_audio, rep_sr = sf.read(repaired_path, dtype="float32")
            rep_dur = len(rep_audio) / rep_sr if rep_sr > 0 else 0.0
            if rep_audio.ndim == 2 and rep_audio.shape[1] == n_channels and rep_dur >= 3.0:
                print(f"   ⚠️ REPAIRED corrupted FLAC via ffmpeg ({file_size} bytes -> {len(rep_audio)} samples, {rep_dur:.2f}s)")
                return rep_audio, rep_sr, None
            else:
                rep_shape_str = str(getattr(rep_audio, "shape", None))
                repair_details.append(f"ffmpeg produced invalid audio: shape={rep_shape_str}, duration={rep_dur:.2f}s (< 3.0s)")
                try: os.remove(repaired_path)
                except Exception: pass
        else:
            repair_details.append(f"ffmpeg failed: {res.stderr.strip()}")
    except Exception as e:
        repair_details.append(f"ffmpeg exception: {str(e)}")

    # Secondary attempt via flac CLI -F decode forcing
    flac_bin = os.path.expanduser("~/miniforge3/envs/sea/bin/flac")
    if os.path.exists(flac_bin):
        wav_temp = os.path.join(scratch_dir, "repaired_temp.wav")
        cmd_flac = [flac_bin, "-d", "-F", "-f", "-o", wav_temp, flac_path]
        try:
            res_flac = subprocess.run(cmd_flac, capture_output=True, text=True, timeout=30)
            if os.path.exists(wav_temp) and os.path.getsize(wav_temp) > 0:
                rep_audio, rep_sr = sf.read(wav_temp, dtype="float32")
                rep_dur = len(rep_audio) / rep_sr if rep_sr > 0 else 0.0
                if rep_audio.ndim == 2 and rep_audio.shape[1] == n_channels and rep_dur >= 3.0:
                    print(f"   ⚠️ REPAIRED corrupted FLAC via flac CLI ({file_size} bytes, {rep_dur:.2f}s)")
                    try: os.remove(wav_temp)
                    except Exception: pass
                    return rep_audio, rep_sr, None
                else:
                    repair_details.append(f"flac CLI produced invalid audio: shape={getattr(rep_audio, 'shape', None)}, duration={rep_dur:.2f}s (< 3.0s)")
                    try: os.remove(wav_temp)
                    except Exception: pass
            else:
                repair_details.append(f"flac CLI decoding failed: {res_flac.stderr.strip()}")
        except Exception as e:
            repair_details.append(f"flac CLI exception: {str(e)}")

    return None, None, {
        "flac_path": flac_path,
        "file_name": os.path.basename(flac_path),
        "file_size_bytes": file_size,
        "file_size_human": format_size(file_size),
        "initial_error": initial_error,
        "repair_attempts": repair_details,
        "decision": "SKIPPED",
    }


def butter_highpass_filter(data: np.ndarray, cutoff: float = HIGH_PASS_CUTOFF, fs: int = FS_TARGET, order: int = 5) -> np.ndarray:
    """Apply zero-phase Butterworth high-pass filter across all channels."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def get_rtf_weights(npz_path: str, channels: list) -> np.ndarray:
    """
    Matched-filter weights from a measured RTF (relative to CH0, 161 bins on the STFT grid).
    Returns:
        W: complex128 array of shape (n_channels, n_freq_bins)
    """
    d = np.load(npz_path)
    assert int(d["fs"]) == FS_TARGET and int(d["nfft"]) == FRAME_LEN, npz_path
    rtf = d["rtf"][channels].astype(np.complex128)

    # Matched filter weights: W = conj(RTF) / sum|RTF|^2 (bins without sweep content are all 0 -> W = 0)
    power = np.sum(np.abs(rtf) ** 2, axis=0, keepdims=True) + 1e-12
    return np.conj(rtf) / power


def build_beam_catalog(location: str):
    """List of (output_filename, rtf_path) for a location.
    LabIR and SPIR were measured with the ReSpeaker 6-Mic, so only ReSpeaker locations get them."""
    beams = []

    if location not in MIC_CHANNELS:
        # 1. LabIR (133 beams)
        for spk in LABIR_SPEAKERS:
            degrees_list = [0] if spk == 12 else LABIR_DEGREES
            for deg in degrees_list:
                beams.append((f"LabIR(S{spk:02d}_{deg:03d}).wav",
                              f"{RTF_BASE_PATH}/Lab/RTF/LAB_RTF_S{spk:02d}_{deg:03d}.npz"))

        # 2. SPIR1 (23 beams, every measured position)
        for path in sorted(glob.glob(f"{RTF_BASE_PATH}/SilwoodPark/SP1/RTF/SP_RTF_*.npz")):
            pos = os.path.basename(path)[len("SP_RTF_"):-len(".npz")]
            beams.append((f"SPIR1({pos}).wav", path))

        # 3. SPIR2 (7 beams)
        for dist in SPIR2_DISTANCES:
            beams.append((f"SPIR2({dist:02d}m_180_r{SPIR2_REP}).wav",
                          f"{RTF_BASE_PATH}/SilwoodPark/SP2/RTF/SP_RTF_{dist:02d}m_180_{SPIR2_REP}.npz"))

    # 4. WCIR: Way Canguk RTFs of this location (12 per set)
    for path in sorted(glob.glob(f"{RTF_BASE_PATH}/WayCanguk/RTF/WC_RTF_{location}_*.npz")):
        pos = os.path.basename(path)[len(f"WC_RTF_{location}_"):-len(".npz")].split("_", 1)[1]   # drop device
        beams.append((f"WCIR({pos}).wav", path))

    return beams


def get_beam_weights_tensor(location: str):
    """Retrieve or build the stacked steering weights tensor (n_beams, n_channels, 161)."""
    if location not in _BEAMS:
        channels = MIC_CHANNELS.get(location, list(range(6)))
        catalog = build_beam_catalog(location)
        W = np.stack([get_rtf_weights(path, channels) for _, path in catalog], axis=0)
        _BEAMS[location] = (catalog, W)
    return _BEAMS[location]


def _worker_istft_save(task):
    """Worker function for parallel ISTFT inversion and WAV writing."""
    spec, out_path, hop_len, fs = task
    z = librosa.istft(spec, hop_length=hop_len, window='hamming')
    z = z / (np.max(np.abs(z)) + 1e-12)
    sf.write(out_path, z.astype(np.float32), fs)


def render_single_flac(flac_path: str, output_dir: str, location: str, render_beams: bool = True, workers: int = 8) -> Tuple[bool, Optional[Dict[str, Any]]]:
    """
    Render a single multi-channel FLAC file into Mono, SA, and Beamformed WAV files.
    Uses vectorized matrix contraction (einsum) and multi-core parallel ISTFT.
    Returns:
        (True, None) if successful.
        (False, error_info_dict) if corrupted and unrecoverable.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load and verify multi-channel audio (with automatic repair if corrupted)
    channels = MIC_CHANNELS.get(location, list(range(6)))
    n_file_channels = 8 if location in MIC_CHANNELS else 6
    audio_raw, sr, error_info = load_and_verify_flac(flac_path, output_dir, n_file_channels)
    if audio_raw is None:
        return False, error_info
    audio_raw = audio_raw[:, channels]

    if sr != FS_TARGET:
        audio_raw = librosa.resample(audio_raw.T, orig_sr=sr, target_sr=FS_TARGET).T

    # 2. High-pass filter
    audio_filt = butter_highpass_filter(audio_raw, cutoff=HIGH_PASS_CUTOFF, fs=FS_TARGET)

    # 3. Render Mono (Channel 0)
    mono_audio = audio_filt[:, 0]
    mono_audio = mono_audio / (np.max(np.abs(mono_audio)) + 1e-12)
    mono_path = os.path.join(output_dir, "mono.wav")
    sf.write(mono_path, mono_audio.astype(np.float32), FS_TARGET)

    # 4. Render Signal Averaging (SA: mean of all mic channels)
    sa_audio = np.mean(audio_filt, axis=1)
    sa_audio = sa_audio / (np.max(np.abs(sa_audio)) + 1e-12)
    sa_path = os.path.join(output_dir, "sa.wav")
    sf.write(sa_path, sa_audio.astype(np.float32), FS_TARGET)

    if not render_beams:
        return True, None

    # 5. Multidirectional Beamforming in STFT domain
    catalog, W_tensor = get_beam_weights_tensor(location)

    # Compute STFT for each channel: shape (n_channels, n_freq, n_frames)
    X = np.stack([
        librosa.stft(audio_filt[:, ch], n_fft=FRAME_LEN, hop_length=HOP_LEN, window='hamming')
        for ch in range(audio_filt.shape[1])
    ], axis=0)

    # Vectorized matrix contraction: Z[k, f, t] = sum_c W[k, c, f] * X[c, f, t]
    Z_all = np.einsum('kcf,cft->kft', W_tensor, X)

    # Parallel ISTFT and WAV saving
    tasks = [
        (Z_all[i], os.path.join(output_dir, catalog[i][0]), HOP_LEN, FS_TARGET)
        for i in range(len(catalog))
    ]
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(_worker_istft_save, tasks)

    return True, None


def main():
    parser = argparse.ArgumentParser(description="Render multi-channel FLAC to Mono, SA, and Beamformed WAVs.")
    parser.add_argument("--location", default="2A400", help="Location code (e.g. 2A400)")
    parser.add_argument("--date", default="2026-04-22", help="Date string YYYY-MM-DD")
    parser.add_argument("--max-files", type=int, default=1, help="Max files to process (0 = all)")
    parser.add_argument("--file-pattern", default=None, help="Specific file substring to match")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes for parallel ISTFT")
    args = parser.parse_args()

    rpi_id = LOCATION_MAP.get(args.location, args.location)
    flac_dir = os.path.join(MONITORING_DATA, rpi_id, args.date)
    flac_files = sorted(glob.glob(os.path.join(flac_dir, "*.flac")))

    if not flac_files:
        print(f"❌ No FLAC files found in {flac_dir}")
        return

    if args.file_pattern:
        flac_files = [f for f in flac_files if args.file_pattern in os.path.basename(f)]

    if args.max_files > 0:
        flac_files = flac_files[:args.max_files]

    target_scratch = os.path.join(SCRATCH_DIR, args.location, args.date)
    os.makedirs(target_scratch, exist_ok=True)

    print(f"🚀 Processing {len(flac_files)} files for {args.location} on {args.date} (Workers: {args.workers})")
    print(f"📁 Source: {flac_dir}")
    print(f"💾 Target scratch: {target_scratch}")

    t0 = time.time()
    for idx, flac in enumerate(flac_files, 1):
        rec_name = os.path.splitext(os.path.basename(flac))[0]
        out_folder = os.path.join(target_scratch, rec_name)
        print(f"[{idx}/{len(flac_files)}] Rendering {rec_name}...")
        t_start = time.time()
        ok, err = render_single_flac(flac, out_folder, args.location, render_beams=True, workers=args.workers)
        if ok:
            print(f"    ✓ Done in {time.time() - t_start:.2f}s ({len(os.listdir(out_folder))} WAV files created)")
        else:
            print(f"    ❌ Skipped corrupted FLAC ({err['file_size_human']}): {err['initial_error']}")

    print(f"\n🏁 Finished rendering in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
