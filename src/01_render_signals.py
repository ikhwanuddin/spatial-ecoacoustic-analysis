"""
Module 1: Signal Processing & Multidirectional Beamforming.
Renders raw 6-channel FLAC recordings into Mono, Signal Averaging (SA),
and Onset-Aligned Filter-and-Sum Beamforming (LabIR and SPIR).

Vectorized matrix contraction (einsum) & multi-core parallel ISTFT (KISS).
"""

import os
import glob
import time
import argparse
import multiprocessing
import numpy as np
import scipy.signal as signal
import soundfile as sf
import librosa

from config import (
    FS_TARGET,
    HIGH_PASS_CUTOFF,
    FRAME_LEN,
    HOP_LEN,
    IR_WINDOW_LEN,
    IR_PRE_PEAK_SAMPLES,
    IR_BASE_PATH,
    MONITORING_DATA,
    SCRATCH_DIR,
    LABIR_SPEAKERS,
    LABIR_DEGREES,
    SPIR1_DISTANCES,
    SPIR1_DEGREES,
    SPIR2_DISTANCES,
    SPIR2_DEGREES,
    SPIR2_REP,
    LOCATION_MAP,
)

# In-memory steering vector cache for fast reuse across files
_IR_WEIGHTS_CACHE = {}
_BEAM_CATALOG = None
_W_TENSOR = None


def butter_highpass_filter(data: np.ndarray, cutoff: float = HIGH_PASS_CUTOFF, fs: int = FS_TARGET, order: int = 5) -> np.ndarray:
    """Apply zero-phase Butterworth high-pass filter across all channels."""
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return signal.filtfilt(b, a, data, axis=0)


def get_onset_steering_weights(ir_path: str) -> np.ndarray:
    """
    Load raw IR, align exactly on direct arrival peak, and compute matched-filter weights.
    Returns:
        W: complex128 array of shape (6, n_freq_bins)
    """
    if ir_path in _IR_WEIGHTS_CACHE:
        return _IR_WEIGHTS_CACHE[ir_path]

    if not os.path.exists(ir_path):
        raise FileNotFoundError(f"IR file not found: {ir_path}")

    # Load multi-channel IR
    ir_raw, sr = sf.read(ir_path)
    if ir_raw.shape[0] < ir_raw.shape[1]:  # Ensure shape is (n_samples, n_channels)
        ir_raw = ir_raw.T

    # Resample to 16 kHz if necessary
    if sr != FS_TARGET:
        ir_resampled = librosa.resample(ir_raw.T, orig_sr=sr, target_sr=FS_TARGET).T
    else:
        ir_resampled = ir_raw

    # Find direct arrival peak on reference channel (channel 0)
    peak_idx = int(np.argmax(np.abs(ir_resampled[:, 0])))

    # Align window centered/starting at direct arrival
    start_idx = max(0, peak_idx - IR_PRE_PEAK_SAMPLES)
    end_idx = start_idx + IR_WINDOW_LEN
    ir_window = ir_resampled[start_idx:end_idx, :]

    if ir_window.shape[0] < IR_WINDOW_LEN:
        pad_width = IR_WINDOW_LEN - ir_window.shape[0]
        ir_window = np.pad(ir_window, ((0, pad_width), (0, 0)), mode='constant')

    # Apply half-Hann taper to avoid hard truncation edge
    taper = signal.windows.tukey(IR_WINDOW_LEN, alpha=0.25)[:, None]
    ir_tapered = ir_window * taper

    # FFT zero-padded to STFT frame length (FRAME_LEN = 320)
    # Shape: (n_channels, n_freq) = (6, 161)
    IR = np.fft.rfft(ir_tapered.T, n=FRAME_LEN, axis=-1)

    # Relative steering vector normalized by reference channel
    IRR = IR / (IR[0:1, :] + 1e-12)

    # Matched filter weights: W = conj(IRR) / sum|IRR|^2
    power = np.sum(np.abs(IRR)**2, axis=0, keepdims=True) + 1e-12
    W = np.conj(IRR) / power

    _IR_WEIGHTS_CACHE[ir_path] = W.astype(np.complex128)
    return _IR_WEIGHTS_CACHE[ir_path]


def build_beam_catalog():
    """Build list of (output_filename, ir_filepath) for all canonical 50 beams."""
    beams = []

    # 1. LabIR (19 beams)
    labir_folder = os.path.join(IR_BASE_PATH, "Lab_IR")
    for spk in LABIR_SPEAKERS:
        degrees_list = [0] if spk == 12 else LABIR_DEGREES
        for deg in degrees_list:
            fname = f"LabIR(S{spk:02d}_{deg:03d}).wav"
            ir_file = f"Lab_IR_S{spk:02d}_{deg:03d}.wav"
            beams.append((fname, os.path.join(labir_folder, ir_file)))

    # 2. SPIR1 (24 beams)
    spir1_folder = os.path.join(IR_BASE_PATH, "SP_IR1")
    for dist in SPIR1_DISTANCES:
        for deg in SPIR1_DEGREES:
            fname = f"SPIR1({dist:02d}m_{deg:03d}).wav"
            ir_file = f"SP_IR_{dist:02d}m_{deg:03d}.wav"
            beams.append((fname, os.path.join(spir1_folder, ir_file)))

    # 3. SPIR2 (7 beams)
    spir2_folder = os.path.join(IR_BASE_PATH, "SP_IR2")
    for dist in SPIR2_DISTANCES:
        fname = f"SPIR2({dist:02d}m_180_r{SPIR2_REP}).wav"
        ir_file = f"{dist:02d}m_180_{SPIR2_REP}.wav"
        beams.append((fname, os.path.join(spir2_folder, ir_file)))

    return beams


def get_beam_weights_tensor():
    """Retrieve or build the stacked steering weights tensor (50, 6, 161)."""
    global _BEAM_CATALOG, _W_TENSOR
    if _W_TENSOR is None:
        _BEAM_CATALOG = build_beam_catalog()
        weights = [get_onset_steering_weights(ir_path) for _, ir_path in _BEAM_CATALOG]
        _W_TENSOR = np.stack(weights, axis=0)
    return _BEAM_CATALOG, _W_TENSOR


def _worker_istft_save(task):
    """Worker function for parallel ISTFT inversion and WAV writing."""
    spec, out_path, hop_len, fs = task
    z = librosa.istft(spec, hop_length=hop_len, window='hamming')
    z = z / (np.max(np.abs(z)) + 1e-12)
    sf.write(out_path, z.astype(np.float32), fs)


def render_single_flac(flac_path: str, output_dir: str, render_beams: bool = True, workers: int = 8):
    """
    Render a single 6-channel FLAC file into Mono, SA, and Beamformed WAV files.
    Uses vectorized matrix contraction (einsum) and multi-core parallel ISTFT.
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load multi-channel audio
    audio_raw, sr = sf.read(flac_path)
    if sr != FS_TARGET:
        audio_raw = librosa.resample(audio_raw.T, orig_sr=sr, target_sr=FS_TARGET).T

    # 2. High-pass filter
    audio_filt = butter_highpass_filter(audio_raw, cutoff=HIGH_PASS_CUTOFF, fs=FS_TARGET)

    # 3. Render Mono (Channel 0)
    mono_audio = audio_filt[:, 0]
    mono_audio = mono_audio / (np.max(np.abs(mono_audio)) + 1e-12)
    mono_path = os.path.join(output_dir, "mono.wav")
    sf.write(mono_path, mono_audio.astype(np.float32), FS_TARGET)

    # 4. Render Signal Averaging (SA: mean of 6 channels)
    sa_audio = np.mean(audio_filt, axis=1)
    sa_audio = sa_audio / (np.max(np.abs(sa_audio)) + 1e-12)
    sa_path = os.path.join(output_dir, "sa.wav")
    sf.write(sa_path, sa_audio.astype(np.float32), FS_TARGET)

    if not render_beams:
        return

    # 5. Multidirectional Beamforming in STFT domain
    catalog, W_tensor = get_beam_weights_tensor()

    # Compute STFT for each channel: shape (6, n_freq, n_frames)
    X = np.stack([
        librosa.stft(audio_filt[:, ch], n_fft=FRAME_LEN, hop_length=HOP_LEN, window='hamming')
        for ch in range(6)
    ], axis=0)

    # Vectorized matrix contraction: Z[k, f, t] = sum_c W[k, c, f] * X[c, f, t]
    Z_all = np.einsum('kcf, cft -> kft', W_tensor, X)

    # Parallel ISTFT and WAV saving
    tasks = [
        (Z_all[i], os.path.join(output_dir, catalog[i][0]), HOP_LEN, FS_TARGET)
        for i in range(len(catalog))
    ]
    with multiprocessing.Pool(processes=workers) as pool:
        pool.map(_worker_istft_save, tasks)


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
        render_single_flac(flac, out_folder, render_beams=True, workers=args.workers)
        print(f"    ✓ Done in {time.time() - t_start:.2f}s ({len(os.listdir(out_folder))} WAV files created)")

    print(f"\n🏁 Finished rendering in {time.time() - t0:.2f}s")


if __name__ == "__main__":
    main()
