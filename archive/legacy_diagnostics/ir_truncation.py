"""What does the 320 sample steering vector keep, and what does it throw away?

`ircache._compute_irr` builds the steering vector with

    IR = np.fft.rfft(ir, n=framelen)      framelen = FRAME_LEN_SEC * FS_TARGET = 320

and numpy truncates the input when n is shorter than it, so everything after the
first 320 samples (20 ms at 16 kHz) of every impulse response is discarded.

FRAME_LEN_SEC = 0.02 is a bare constant in config.py's AUDIO CONFIG block with no
comment, is referenced nowhere in docs/, and the repo history is grafted, so there
is no recorded reason for it. It is the STFT frame length; the IR truncation is a
side effect of reusing it as the rfft size.

Three questions this answers:
  1. Does the direct arrival even fall inside the first 20 ms? If it does not, the
     steering vector is wrong, not merely simplified.
  2. How much IR energy is kept?
  3. Does the truncated transfer function still match the full one where birds are,
     1-4 kHz?

usage: ir_truncation.py [n_files]
"""

import glob
import os
import sys

import librosa
import numpy as np

from config import FRAME_LEN_SEC, FS_TARGET, FS_IR_ORIGINAL, IR_BASE_PATH

FRAMELEN = int(FRAME_LEN_SEC * FS_TARGET)     # 320
LAB_IR = os.path.join(IR_BASE_PATH, "Lab_IR")


def analyse(path):
    ir_orig, _ = librosa.load(path, sr=FS_IR_ORIGINAL, mono=False)
    ir = librosa.resample(ir_orig, orig_sr=FS_IR_ORIGINAL, target_sr=FS_TARGET)
    n_ch, n = ir.shape

    peaks = [int(np.argmax(np.abs(ir[c]))) for c in range(n_ch)]
    energy = (ir ** 2).sum(axis=1)
    kept = (ir[:, :FRAMELEN] ** 2).sum(axis=1)
    frac = kept / np.where(energy > 0, energy, 1.0)

    # relative transfer function, the thing beamforming actually uses:
    # truncated at 320 versus the full IR evaluated on the same 161 bins
    trunc = np.fft.rfft(ir, n=FRAMELEN)
    trunc = trunc / (trunc[0, :] + 1e-12)
    nfull = 1 << int(np.ceil(np.log2(n)))
    full = np.fft.rfft(ir, n=nfull)
    full = full / (full[0, :] + 1e-12)
    bins_full = np.arange(nfull // 2 + 1) * FS_TARGET / nfull
    bins_tr = np.arange(FRAMELEN // 2 + 1) * FS_TARGET / FRAMELEN
    idx = np.searchsorted(bins_full, bins_tr)
    idx = np.clip(idx, 0, full.shape[1] - 1)
    full_on_tr = full[:, idx]

    band = (bins_tr >= 1000) & (bins_tr <= 4000)
    # channel 0 is 1 by construction in both, so compare channels 1..5
    a, b = trunc[1:, band], full_on_tr[1:, band]
    mag_db = 20 * np.log10(np.abs(a) + 1e-12) - 20 * np.log10(np.abs(b) + 1e-12)
    phase_deg = np.degrees(np.angle(a / (b + 1e-12)))
    phase_deg = (phase_deg + 180) % 360 - 180

    return dict(
        name=os.path.basename(path), n=n, dur_ms=n / FS_TARGET * 1000,
        peaks=peaks, frac=frac,
        mag_med=float(np.median(np.abs(mag_db))), mag_max=float(np.abs(mag_db).max()),
        ph_med=float(np.median(np.abs(phase_deg))), ph_max=float(np.abs(phase_deg).max()),
    )


def main():
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 6
    files = sorted(glob.glob(os.path.join(LAB_IR, "Lab_IR_S*.wav")))
    if not files:
        raise SystemExit(f"no IRs under {LAB_IR}")
    print(f"{len(files)} IR files in {LAB_IR}")
    print(f"truncation length: {FRAMELEN} samples = {FRAMELEN/FS_TARGET*1000:.1f} ms "
          f"at {FS_TARGET} Hz\n")

    step = max(1, len(files) // k)
    rows = [analyse(f) for f in files[::step][:k]]

    print("IR length and where the direct arrival sits")
    print(f"{'file':26s} {'len':>7s} {'dur_ms':>8s} {'peak sample per channel':>34s}")
    for r in rows:
        print(f"{r['name']:26s} {r['n']:7d} {r['dur_ms']:8.1f} "
              f"{str(r['peaks']):>34s}")

    print("\nfraction of IR energy inside the first 320 samples")
    print(f"{'file':26s} {'min':>8s} {'median':>8s} {'max':>8s}")
    for r in rows:
        f = r["frac"]
        print(f"{r['name']:26s} {f.min():8.4f} {np.median(f):8.4f} {f.max():8.4f}")

    print("\ntruncated vs full relative transfer function, 1-4 kHz, channels 1-5")
    print(f"{'file':26s} {'|dB| med':>9s} {'|dB| max':>9s} {'phase med':>10s} {'phase max':>10s}")
    for r in rows:
        print(f"{r['name']:26s} {r['mag_med']:9.3f} {r['mag_max']:9.3f} "
              f"{r['ph_med']:10.2f} {r['ph_max']:10.2f}")

    allpk = [p for r in rows for p in r["peaks"]]
    print(f"\ndirect arrival inside the kept window: "
          f"{'YES' if max(allpk) < FRAMELEN else 'NO'} "
          f"(latest peak at sample {max(allpk)} of {FRAMELEN})")


if __name__ == "__main__":
    main()
