"""Decisive test: is the cached steering vector a direction, or is it noise?

A steering vector describes how one direction's wavefront hits six microphones a
few centimetres apart. Two directions 10 degrees apart must therefore produce
nearly the same vector, and the similarity must fall smoothly as the angle grows.
Noise cannot do that: every IR file's pre-arrival segment is its own independent
noise realisation, so similarity would sit at the chance level for every angle,
including 10 degrees.

Compares, for one speaker ring, the vector built the way `ircache._compute_irr`
builds it (first 320 samples) against one built from a window aligned on the
direct arrival.
"""
import os

import librosa
import numpy as np

from config import FRAME_LEN_SEC, FS_TARGET, FS_IR_ORIGINAL, IR_BASE_PATH

FRAMELEN = int(FRAME_LEN_SEC * FS_TARGET)
SPEAKER = 5
AZIMUTHS = list(range(0, 360, 10))


def steering(ir, start):
    seg = ir[:, start:start + FRAMELEN]
    if seg.shape[1] < FRAMELEN:
        seg = np.pad(seg, ((0, 0), (0, FRAMELEN - seg.shape[1])))
    IR = np.fft.rfft(seg, n=FRAMELEN)
    return IR / (IR[0, :] + 1e-12)


def coherence(a, b):
    """Magnitude of the normalised complex inner product over channels 1-5, all bins."""
    x, y = a[1:].ravel(), b[1:].ravel()
    return float(np.abs(np.vdot(x, y)) / (np.linalg.norm(x) * np.linalg.norm(y)))


trunc, aligned, onsets = {}, {}, {}
for az in AZIMUTHS:
    path = os.path.join(IR_BASE_PATH, "Lab_IR", f"Lab_IR_S{SPEAKER:02d}_{az:03d}.wav")
    if not os.path.isfile(path):
        continue
    ir, _ = librosa.load(path, sr=FS_IR_ORIGINAL, mono=False)
    ir = librosa.resample(ir, orig_sr=FS_IR_ORIGINAL, target_sr=FS_TARGET)
    onset = int(np.argmax(np.abs(ir).max(axis=0)))
    onsets[az] = onset
    trunc[az] = steering(ir, 0)
    aligned[az] = steering(ir, max(0, onset - 8))

print(f"speaker S{SPEAKER:02d}, {len(trunc)} azimuths, window {FRAMELEN} samples "
      f"({FRAMELEN/FS_TARGET*1000:.0f} ms)")
print(f"direct arrival sample: min {min(onsets.values())} max {max(onsets.values())}\n")

ref = 0
print("similarity to azimuth 000, as the angle grows")
print(f"{'angle':>6s} {'cached (first 320)':>20s} {'onset aligned':>15s}")
for az in AZIMUTHS:
    if az not in trunc:
        continue
    ang = min(az - ref, 360 - (az - ref))
    print(f"{ang:6d} {coherence(trunc[ref], trunc[az]):20.4f} "
          f"{coherence(aligned[ref], aligned[az]):15.4f}")

pairs_t = [coherence(trunc[a], trunc[b]) for i, a in enumerate(AZIMUTHS)
           for b in AZIMUTHS[i+1:] if a in trunc and b in trunc]
pairs_a = [coherence(aligned[a], aligned[b]) for i, a in enumerate(AZIMUTHS)
           for b in AZIMUTHS[i+1:] if a in aligned and b in aligned]
adj_t = [coherence(trunc[AZIMUTHS[i]], trunc[AZIMUTHS[i+1]])
         for i in range(len(AZIMUTHS)-1) if AZIMUTHS[i] in trunc and AZIMUTHS[i+1] in trunc]
adj_a = [coherence(aligned[AZIMUTHS[i]], aligned[AZIMUTHS[i+1]])
         for i in range(len(AZIMUTHS)-1) if AZIMUTHS[i] in aligned and AZIMUTHS[i+1] in aligned]

print(f"\n{'':22s} {'adjacent (10 deg)':>18s} {'all pairs':>12s}")
print(f"{'cached (first 320)':22s} {np.mean(adj_t):18.4f} {np.mean(pairs_t):12.4f}")
print(f"{'onset aligned':22s} {np.mean(adj_a):18.4f} {np.mean(pairs_a):12.4f}")
print("\nA real steering vector: adjacent >> all pairs.")
print("Noise: adjacent indistinguishable from all pairs.")
