"""Can a 3 second window be beamformed on its own and come out sample identical?

Everything ABD pass 2 rests on this. The answer decides whether refinement can be
done per window at per-recording cost, or not at all.

What the renderer actually does (beamforming.py):

    X = stft(raw, n_fft=320, hop=160, hamming)      320 samples = 20 ms
    W = conj(IRR) / sum_c |IRR_c|^2                 Rxx is the identity, so this
                                                    is matched filtering, not MVDR
    Z = sum_c W_c * X_c                             pointwise, per frame
    z = istft(Z)
    z = z / max|z|                                  WHOLE FILE, PER DIRECTION
    write int16

Two consequences drive the tests below.

1. `Z = W * X` is pointwise in the frame axis, so frame t of the output depends on
   frame t of the input and nothing else. The footprint of a sample is one 20 ms
   frame, not the length of the impulse response — `_compute_irr` truncates every
   IR to 320 samples with `np.fft.rfft(ir, n=framelen)`. A window can therefore be
   reproduced from a short slice, and the guard needed is tens of milliseconds.

2. The peak normalisation is over the whole 4 minute file and is different for
   every direction. A window rendered on its own does not know that number. This
   is the part that can sink the scheme.

usage: abd_identity_test.py [<location> <date>]
"""

import glob
import os
import re
import sys

import numpy as np
import librosa
import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FRAME_LEN_SEC, FS_TARGET, N_CHANNELS_EXPECTED, IR_TYPES
from ircache import IRCache
from audio_loader import load_audio_robust
from beamforming import Beamformer

ANALYSIS_OUTPUT = os.environ.get(
    "ANALYSIS_OUTPUT", "/rds/general/user/ri322/ephemeral/sea-work")
MONITORING_DATA = os.environ.get(
    "MONITORING_DATA", "/rds/general/user/ri322/ephemeral/monitoring_data")
RPIID = "RPiID-0000000091668b26"          # 2A400

FRAMELEN = int(FRAME_LEN_SEC * FS_TARGET)  # 320
HOP = FRAMELEN // 2                        # 160
WINDOW_SEC = 3.0
WINDOW_N = int(WINDOW_SEC * FS_TARGET)     # 48000, an exact multiple of HOP


def find_rendered(location, date, tag="SBF(S05_000)"):
    """One rendered wav for a known direction, and the FLAC it came from."""
    root = os.path.join(ANALYSIS_OUTPUT, location, date, "bf_SBF")
    hits = sorted(glob.glob(os.path.join(root, "*", "*", f"*_{tag}.wav")))
    if not hits:
        raise SystemExit(f"no rendered wav under {root} for {tag}")
    wav = hits[len(hits) // 2]
    base = os.path.basename(wav)[: -len(f"_{tag}.wav")]
    flacs = sorted(glob.glob(
        os.path.join(MONITORING_DATA, RPIID, "**", base + ".flac"), recursive=True))
    if not flacs:
        raise SystemExit(f"no source FLAC named {base}.flac under {MONITORING_DATA}")
    return wav, flacs[0], base


def whole_file(flac_path, speaker, degrees):
    """Re-run the renderer's own path and keep the intermediates it throws away."""
    raw, _ = load_audio_robust(
        flac_path, target_sr=FS_TARGET, expected_channels=N_CHANNELS_EXPECTED)
    X = librosa.stft(raw, n_fft=FRAMELEN, hop_length=HOP, window="hamming")
    Y = np.transpose(X, (1, 0, 2))                       # (n_freq, n_chan, n_frame)

    irr = IRCache("SBF").load(speaker, degrees, None)
    irr = irr.reshape(irr.shape[0], irr.shape[1], 1)
    bf = Beamformer.__new__(Beamformer)                  # weights only, no I/O
    W = bf._compute_weights(np.eye(irr.shape[0]), irr)   # (n_freq, n_chan, 1)

    Z = np.sum(W * Y, axis=1, keepdims=True)
    z = librosa.istft(Z[:, 0, :], hop_length=HOP, window="hamming")
    return raw, Y, W, Z[:, 0, :], np.real(z)


def to_int16(z, zmax):
    return (np.real(z) / zmax * 32767).clip(-32768, 32767).astype("int16")


def window_from_stft(Z, guard_frames, s0, s1):
    """Reconstruct samples [s0, s1) from a frame slice of the whole file Z.

    librosa's stft is center=True, so frame t is centred on sample t*HOP and the
    frame slice starting at t_lo reconstructs, with center=False, from sample
    t_lo*HOP - FRAMELEN//2.
    """
    t_lo = max(0, s0 // HOP - 1 - guard_frames)
    t_hi = min(Z.shape[1] - 1, s1 // HOP + 1 + guard_frames)
    sub = Z[:, t_lo:t_hi + 1]
    zsub = np.real(librosa.istft(sub, hop_length=HOP, window="hamming", center=False))
    origin = t_lo * HOP - FRAMELEN // 2
    off = s0 - origin
    if off < 0 or off + (s1 - s0) > len(zsub):
        return None
    return zsub[off:off + (s1 - s0)]


def window_from_audio(raw, W, guard_samples, s0, s1):
    """Reconstruct samples [s0, s1) from a slice of the raw audio, no cached STFT."""
    a = max(0, s0 - guard_samples)
    b = min(raw.shape[1], s1 + guard_samples)
    Xs = librosa.stft(raw[:, a:b], n_fft=FRAMELEN, hop_length=HOP, window="hamming")
    Ys = np.transpose(Xs, (1, 0, 2))
    Zs = np.sum(W * Ys, axis=1, keepdims=True)
    zs = np.real(librosa.istft(Zs[:, 0, :], hop_length=HOP, window="hamming"))
    off = s0 - a
    if off + (s1 - s0) > len(zs):
        return None
    return zs[off:off + (s1 - s0)]


def report(name, ref, got):
    if got is None:
        print(f"    {name:28s} NOT ENOUGH SAMPLES")
        return
    d = np.abs(ref - got)
    scale = np.max(np.abs(ref)) or 1.0
    print(f"    {name:28s} max_abs={d.max():.3e}  rel={d.max()/scale:.3e}  "
          f"int16_ulp={d.max()/scale*32767:.4f}")


def main():
    location = sys.argv[1] if len(sys.argv) > 2 else "2A400"
    date = sys.argv[2] if len(sys.argv) > 2 else "2026-04-30"
    speaker, degrees = 5, 0

    wav, flac, base = find_rendered(location, date)
    print(f"recording : {base}")
    print(f"rendered  : {wav}")
    print(f"source    : {flac}")
    print(f"framelen  : {FRAMELEN} samples ({FRAME_LEN_SEC*1000:.0f} ms), hop {HOP}")

    disk, sr = sf.read(wav, dtype="int16")
    print(f"on disk   : {len(disk)} samples at {sr} Hz")

    raw, Y, W, Z, z = whole_file(flac, speaker, degrees)
    zmax = np.max(np.abs(z))
    mine = to_int16(z, zmax)
    n = min(len(disk), len(mine))
    print(f"recomputed: {len(mine)} samples, zmax={zmax:.6f}")

    print("\n[1] whole file re-render vs the wav on disk")
    diff = np.abs(disk[:n].astype(np.int64) - mine[:n].astype(np.int64))
    print(f"    identical={bool(diff.max() == 0)}  max_int16_diff={diff.max()}  "
          f"n_differing={int((diff > 0).sum())} of {n}")

    n_windows = len(z) // WINDOW_N
    picks = [0, n_windows // 2, n_windows - 1] if n_windows >= 3 else [0]
    print(f"\n[2] per window reconstruction, {n_windows} windows, testing {picks}")
    for k in picks:
        s0, s1 = k * WINDOW_N, (k + 1) * WINDOW_N
        ref = z[s0:s1]
        print(f"  window #{k}  samples [{s0}, {s1})")
        for g in (0, 1, 2, 4, 8):
            report(f"stft-slice guard={g} frames", ref, window_from_stft(Z, g, s0, s1))
        for gs in (FRAMELEN, 2 * FRAMELEN, 8 * FRAMELEN):
            report(f"audio-slice guard={gs} smp", ref,
                   window_from_audio(raw, W, gs, s0, s1))

    print("\n[3] the peak normalisation, which is per direction and whole file")
    ir_type = IR_TYPES["SBF"]
    zenith = ir_type.zenith_speakers or set()
    peaks = {}
    for p in ir_type.param_values:
        for d in ([0] if p in zenith else ir_type.degree_values):
            try:
                irr = IRCache("SBF").load(p, d, None)
            except FileNotFoundError:
                continue
            irr = irr.reshape(irr.shape[0], irr.shape[1], 1)
            bf = Beamformer.__new__(Beamformer)
            Wd = bf._compute_weights(np.eye(irr.shape[0]), irr)
            zd = np.real(librosa.istft(
                np.sum(Wd * Y, axis=1, keepdims=True)[:, 0, :],
                hop_length=HOP, window="hamming"))
            peaks[(p, d)] = float(np.max(np.abs(zd)))
    vals = np.array(list(peaks.values()))
    print(f"    directions measured : {len(vals)}")
    print(f"    whole file peak     : min={vals.min():.5f} med={np.median(vals):.5f} "
          f"max={vals.max():.5f}  max/min={vals.max()/vals.min():.3f}")

    wpk = []
    for k in range(n_windows):
        wpk.append(np.max(np.abs(z[k*WINDOW_N:(k+1)*WINDOW_N])))
    wpk = np.array(wpk)
    print(f"    for S05_000, window peak / file peak: min={wpk.min()/zmax:.4f} "
          f"med={np.median(wpk)/zmax:.4f} max={wpk.max()/zmax:.4f}")
    print("    A window normalised by its own peak is scaled by the reciprocal of")
    print("    that ratio, so it is NOT the same signal the renderer would write.")


if __name__ == "__main__":
    main()
