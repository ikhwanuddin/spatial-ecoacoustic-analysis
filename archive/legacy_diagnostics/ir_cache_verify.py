"""Is the .npz on disk really the first-320-samples version, or was it built by
some earlier code path? The renders used the cache, so the cache is what counts.
"""
import glob
import os

import librosa
import numpy as np

from config import FRAME_LEN_SEC, FS_TARGET, FS_IR_ORIGINAL, IR_BASE_PATH

FRAMELEN = int(FRAME_LEN_SEC * FS_TARGET)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ir_cache")


def build(path, start):
    ir, _ = librosa.load(path, sr=FS_IR_ORIGINAL, mono=False)
    ir = librosa.resample(ir, orig_sr=FS_IR_ORIGINAL, target_sr=FS_TARGET)
    onset = int(np.argmax(np.abs(ir).max(axis=0)))
    s = 0 if start == "head" else max(0, onset - 8)
    seg = ir[:, s:s + FRAMELEN]
    IR = np.fft.rfft(seg, n=FRAMELEN)
    return (IR / (IR[0, :] + 1e-12)).astype(np.complex128), onset


for sub, irfile, key in (
    ("SBF", "Lab_IR_S05_000.wav", "SBF_p5_d000"),
    ("LabIR", "Lab_IR_S01_060.wav", "LabIR_p1_d060"),
):
    npz = os.path.join(CACHE, sub, key + ".npz")
    if not os.path.isfile(npz):
        cand = glob.glob(os.path.join(CACHE, sub, "*.npz"))[:3]
        print(f"{npz} missing; found e.g. {[os.path.basename(c) for c in cand]}")
        continue
    z = np.load(npz)
    arr = z[z.files[0]]
    head, onset = build(os.path.join(IR_BASE_PATH, "Lab_IR", irfile), "head")
    algn, _ = build(os.path.join(IR_BASE_PATH, "Lab_IR", irfile), "onset")
    print(f"\n{sub}/{key}  cached shape {arr.shape}  keys {z.files}")
    print(f"  direct arrival at sample {onset}")
    for label, ref in (("first 320 samples", head), ("onset aligned", algn)):
        d = np.abs(arr - ref).max()
        print(f"  vs {label:20s}: max_abs_diff {d:.3e}  "
              f"{'MATCH' if d < 1e-9 else ''}")
