"""Exactly what is inside the 320 samples the steering vector is built from."""
import glob
import os

import librosa
import numpy as np

from config import FRAME_LEN_SEC, FS_TARGET, FS_IR_ORIGINAL, IR_BASE_PATH, IR_TYPES

FRAMELEN = int(FRAME_LEN_SEC * FS_TARGET)


def show(folder, pattern, label):
    files = sorted(glob.glob(os.path.join(IR_BASE_PATH, folder, pattern)))
    if not files:
        print(f"{label}: nothing matched {pattern} under {folder}")
        return
    print(f"\n===== {label}  ({len(files)} files, showing 3) =====")
    for path in files[:: max(1, len(files) // 3)][:3]:
        raw, sr0 = librosa.load(path, sr=None, mono=False)
        ir, _ = librosa.load(path, sr=FS_IR_ORIGINAL, mono=False)
        ir = librosa.resample(ir, orig_sr=FS_IR_ORIGINAL, target_sr=FS_TARGET)
        if ir.ndim == 1:
            ir = ir[None, :]
        pk_all = np.abs(ir).max()
        pk_win = np.abs(ir[:, :FRAMELEN]).max()
        e_all = float((ir ** 2).sum())
        e_win = float((ir[:, :FRAMELEN] ** 2).sum())
        onset = int(np.argmax(np.abs(ir).max(axis=0)))
        print(f"\n{os.path.basename(path)}")
        print(f"  native            : {raw.shape} at {sr0} Hz")
        print(f"  at 16 kHz         : {ir.shape}, {ir.shape[1]/FS_TARGET*1000:.1f} ms")
        print(f"  direct arrival at : sample {onset} = {onset/FS_TARGET*1000:.1f} ms")
        print(f"  peak, whole IR    : {pk_all:.6e}")
        print(f"  peak, first 320   : {pk_win:.6e}   ({20*np.log10(pk_win/pk_all):+.1f} dB)")
        print(f"  energy fraction   : {e_win/e_all:.3e}   "
              f"({10*np.log10(e_win/e_all):+.1f} dB of the total)")

        # the steering vector as beamforming actually builds it
        IR = np.fft.rfft(ir, n=FRAMELEN)
        IRR = IR / (IR[0, :] + 1e-12)
        mag = np.abs(IRR[1:, :])
        print(f"  |IRR| ch1-5       : med {np.median(mag):.3f}  p99 {np.percentile(mag, 99):.1f}"
              f"  max {mag.max():.1f}")
        # for a genuine steering vector between microphones a few cm apart the
        # inter-channel magnitude ratio should sit near 1
        print(f"  share of bins with |IRR| outside [0.5, 2]: "
              f"{float(((mag < 0.5) | (mag > 2)).mean()):.1%}")

        # what an onset-aligned window would have kept instead
        al = ir[:, onset - 8: onset - 8 + FRAMELEN]
        if al.shape[1] == FRAMELEN:
            IRa = np.fft.rfft(al, n=FRAMELEN)
            IRRa = IRa / (IRa[0, :] + 1e-12)
            maga = np.abs(IRRa[1:, :])
            print(f"  if aligned on onset: |IRR| med {np.median(maga):.3f}, "
                  f"outside [0.5,2] {float(((maga < 0.5) | (maga > 2)).mean()):.1%}, "
                  f"energy fraction {float((al**2).sum())/e_all:.3f}")


print(f"steering vector window = {FRAMELEN} samples = {FRAMELEN/FS_TARGET*1000:.1f} ms at {FS_TARGET} Hz")
show(IR_TYPES["LabIR"].folder, "Lab_IR_S*.wav", "LabIR (Lab_IR)")
show(IR_TYPES["SPIR1"].folder, "SP_IR_*.wav", "SPIR1 (SP_IR1)")
show(IR_TYPES["SPIR2"].folder, "*m_180_*.wav", "SPIR2 (SP_IR2)")
