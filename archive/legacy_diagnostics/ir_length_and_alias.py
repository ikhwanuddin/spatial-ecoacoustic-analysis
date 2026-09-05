"""How long should the steering filter be, and what does the current geometry cost?

Two measurements, both on real data rather than rules of thumb.

A. How much time does the array actually need? The inter-microphone delays set the
   floor, and the decay after the direct arrival sets the useful ceiling.

B. The renderer multiplies one 320-sample STFT frame by a 320-point frequency
   response. Linear convolution of a 320 tap filter with a 320 sample block needs
   639 points, so the frame-wise product is a CIRCULAR convolution and the filter
   tail wraps onto the head of its own frame. This measures that error against
   true linear convolution, and whether a shorter filter removes it.
"""
import os

import librosa
import numpy as np

from config import FRAME_LEN_SEC, FS_TARGET, FS_IR_ORIGINAL, IR_BASE_PATH

FRAMELEN = int(FRAME_LEN_SEC * FS_TARGET)     # 320
HOP = FRAMELEN // 2
LAB = os.path.join(IR_BASE_PATH, "Lab_IR")


def load_ir(speaker, az):
    ir, _ = librosa.load(os.path.join(LAB, f"Lab_IR_S{speaker:02d}_{az:03d}.wav"),
                         sr=FS_IR_ORIGINAL, mono=False)
    return librosa.resample(ir, orig_sr=FS_IR_ORIGINAL, target_sr=FS_TARGET)


def subsample_peak(x, i):
    if i <= 0 or i >= len(x) - 1:
        return float(i)
    a, b, c = x[i - 1], x[i], x[i + 1]
    d = a - 2 * b + c
    return float(i) if d == 0 else float(i) - 0.5 * (c - a) / d


print("=" * 68)
print("A. inter-microphone delays, measured from the IRs themselves")
print("=" * 68)
spread, azdelay = [], {}
for az in range(0, 360, 30):
    ir = load_ir(5, az)
    onset = int(np.argmax(np.abs(ir).max(axis=0)))
    seg = ir[:, onset - 32:onset + 288]
    ref = seg[0]
    dl = []
    for c in range(seg.shape[0]):
        xc = np.correlate(seg[c], ref, mode="full")
        i = int(np.argmax(np.abs(xc)))
        dl.append(subsample_peak(np.abs(xc), i) - (len(ref) - 1))
    azdelay[az] = dl
    spread.append(max(dl) - min(dl))
    print(f"  az {az:3d}  delay vs ch0 (samples): "
          f"[{', '.join(f'{d:+.2f}' for d in dl)}]  spread {max(dl)-min(dl):.2f}")
s = np.array(spread)
print(f"\n  spread across channels: min {s.min():.2f} med {np.median(s):.2f} "
      f"max {s.max():.2f} samples")
print(f"  = {s.max()/FS_TARGET*1e6:.0f} us, i.e. {s.max()/FS_TARGET*343*100:.1f} cm "
      f"of path difference at 343 m/s")
print(f"  a pure delay-and-sum steering vector needs about {int(np.ceil(s.max()))+2} taps.")

print()
print("=" * 68)
print("B. energy decay after the direct arrival: how long is the IR useful?")
print("=" * 68)
ir = load_ir(5, 0)
onset = int(np.argmax(np.abs(ir).max(axis=0)))
tail = ir[:, onset:]
e = (tail ** 2).sum(axis=0)
cum = np.cumsum(e) / e.sum()
print(f"{'ms after arrival':>17s} {'taps':>6s} {'cumulative energy':>18s}")
for ms in (1, 2, 4, 8, 20, 50, 100):
    n = int(ms * FS_TARGET / 1000)
    if n < len(cum):
        print(f"{ms:17d} {n:6d} {cum[n-1]:18.4f}")

print()
print("=" * 68)
print("C. circular vs linear convolution, the cost of filter length = frame length")
print("=" * 68)
irr_seg = ir[:, onset - 8:onset - 8 + FRAMELEN]
IR = np.fft.rfft(irr_seg, n=FRAMELEN)
IRR = IR / (IR[0, :] + 1e-12)
W = np.conj(IRR) / (np.abs(IRR) ** 2).sum(axis=0)      # 6 x 161, the renderer's weights
h_full = np.fft.irfft(W, n=FRAMELEN, axis=-1)          # 6 x 320 taps

rng = np.random.default_rng(0)
x = rng.standard_normal((6, 10 * FS_TARGET)) * 0.05    # 10 s, 6 channels

def stft_domain(x, Wf):
    X = librosa.stft(x, n_fft=FRAMELEN, hop_length=HOP, window="hamming")
    Y = np.transpose(X, (1, 0, 2))
    Z = np.sum(Wf.T[:, :, None] * Y, axis=1)
    return np.real(librosa.istft(Z, hop_length=HOP, window="hamming"))

def linear(x, h):
    n = x.shape[1] + h.shape[1] - 1
    nf = 1 << int(np.ceil(np.log2(n)))
    Y = np.fft.rfft(x, n=nf) * np.fft.rfft(h, n=nf)
    return np.fft.irfft(Y.sum(axis=0), n=nf)[: x.shape[1]]

def err(a, b, skip=2000):
    a, b = a[skip:-skip], b[skip:-skip]
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    return 20 * np.log10(np.linalg.norm(a - b) / np.linalg.norm(b) + 1e-300)

print(f"{'filter taps':>12s} {'window':>10s} {'energy kept':>12s} {'error vs linear conv':>22s}")
for taps in (320, 160, 64, 32):
    h = h_full.copy()
    if taps < FRAMELEN:
        w = np.ones(taps)
        k = max(2, taps // 4)
        w[-k:] = np.hanning(2 * k)[k:]                  # half-Hann fade out
        h = h[:, :taps] * w
        label = f"hann fade"
    else:
        label = "rectangular"
    kept = (h ** 2).sum() / (h_full ** 2).sum()
    Wf = np.fft.rfft(h, n=FRAMELEN)
    d = err(stft_domain(x, Wf), linear(x, h))
    print(f"{taps:12d} {label:>10s} {kept:12.4f} {d:19.1f} dB")

print()
print("The 320 tap row is the renderer as it stands. Its own frequency response")
print("is applied to a frame of the same length, so the block product cannot be a")
print("linear convolution and the error is the time aliasing that follows.")
