"""One variable: onset alignment. Everything else left exactly as it is.

The steering vectors are rebuilt in two ways and nothing else differs — same 320
sample window, same rectangular truncation, same unregularised division by
channel 0, same n_fft, same weights formula, same istft, no normalisation
(BirdNET's front end is scale invariant, measured, so scale cannot confound this).

  head    samples 0..319, which is what `ir_cache` holds today
  onset   samples onset-8..onset+311, the same window moved to the direct arrival

The discriminator needs no threshold and no ground truth. Steer across 36
azimuths 10 degrees apart and look at the shape of the confidence curve. A real
beam pattern is smooth in azimuth, because two directions 10 degrees apart
overlap heavily, so the curve has positive lag-1 autocorrelation. A bank of
direction-blind filters gives a white curve, whose expected lag-1 autocorrelation
is -1/(n-1) = -0.029.

Detection counts and beam contrast are reported alongside, but the
autocorrelation is the measurement that decides.

usage: onset_ab_test.py [<location> <date> <speaker>]
"""
import glob
import os
import sys

import librosa
import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FRAME_LEN_SEC, FS_TARGET, FS_IR_ORIGINAL, IR_BASE_PATH, N_CHANNELS_EXPECTED
from audio_loader import load_audio_robust
from beamforming import Beamformer
from beam_confidence import load_head, confidence

FRAMELEN = int(FRAME_LEN_SEC * FS_TARGET)
HOP = FRAMELEN // 2
WIN_N = 3 * FS_TARGET
BN_SR, BN_LEN = 48000, 144000
REPO = os.path.dirname(os.path.abspath(__file__))
CKPT = os.path.join(REPO, "bacpipe", "checkpoints", "birdnet")
AZIMUTHS = list(range(0, 360, 10))
THRESHOLDS = (0.4, 0.5, 0.65)


def steering(speaker, az, mode):
    path = os.path.join(IR_BASE_PATH, "Lab_IR", f"Lab_IR_S{speaker:02d}_{az:03d}.wav")
    ir, _ = librosa.load(path, sr=FS_IR_ORIGINAL, mono=False)
    ir = librosa.resample(ir, orig_sr=FS_IR_ORIGINAL, target_sr=FS_TARGET)
    start = 0 if mode == "head" else max(0, int(np.argmax(np.abs(ir).max(axis=0))) - 8)
    seg = ir[:, start:start + FRAMELEN]
    if seg.shape[1] < FRAMELEN:
        seg = np.pad(seg, ((0, 0), (0, FRAMELEN - seg.shape[1])))
    IR = np.fft.rfft(seg, n=FRAMELEN)
    return (IR / (IR[0, :] + 1e-12)).astype(np.complex128)


def circular_lag1(curve):
    """Lag-1 autocorrelation on the circle, since azimuth wraps."""
    c = curve - curve.mean()
    d = (c ** 2).sum()
    return 0.0 if d == 0 else float((c * np.roll(c, 1)).sum() / d)


def main():
    loc = sys.argv[1] if len(sys.argv) > 3 else "2A400"
    date = sys.argv[2] if len(sys.argv) > 3 else "2026-04-30"
    speaker = int(sys.argv[3]) if len(sys.argv) > 3 else 5

    flacs = sorted(glob.glob(os.path.join(
        os.environ["MONITORING_DATA"], "RPiID-0000000091668b26", date, "*.flac")))
    if not flacs:
        raise SystemExit(f"no FLAC for {date}")
    flac = flacs[len(flacs) // 2]
    print(f"recording : {flac}")
    print(f"speaker   : S{speaker:02d}, {len(AZIMUTHS)} azimuths at 10 degrees")

    raw, _ = load_audio_robust(flac, target_sr=FS_TARGET,
                               expected_channels=N_CHANNELS_EXPECTED)
    X = librosa.stft(raw, n_fft=FRAMELEN, hop_length=HOP, window="hamming")
    Y = np.transpose(X, (1, 0, 2))
    n_win = raw.shape[1] // WIN_N
    print(f"windows   : {n_win}\n")

    model = tf.keras.models.load_model(
        os.path.join(CKPT, "birdnetv2.4.keras"), compile=False)
    embeds = tf.keras.Model(inputs=model.input, outputs=model.layers[-3].output)
    prep = tf.saved_model.load(
        os.path.join(CKPT, "BirdNET_Preprocessor")).signatures["serving_default"]
    head = load_head(REPO, "birdnet")

    bf = Beamformer.__new__(Beamformer)
    eye = np.eye(N_CHANNELS_EXPECTED)
    out = {}

    for mode in ("head", "onset"):
        scores = np.zeros((n_win, len(AZIMUTHS)), dtype=np.float64)
        for j, az in enumerate(AZIMUTHS):
            irr = steering(speaker, az, mode)
            W = bf._compute_weights(eye, irr.reshape(irr.shape[0], irr.shape[1], 1))
            z = np.real(librosa.istft(
                np.sum(W * Y, axis=1, keepdims=True)[:, 0, :],
                hop_length=HOP, window="hamming"))
            z48 = librosa.resample(z, orig_sr=FS_TARGET, target_sr=BN_SR)
            clips = np.zeros((n_win, BN_LEN), dtype=np.float32)
            for k in range(n_win):
                seg = z48[k * BN_LEN:(k + 1) * BN_LEN]
                clips[k, :len(seg)] = seg
            emb = []
            for i in range(0, n_win, 256):
                mel = prep(tf.convert_to_tensor(clips[i:i + 256], dtype=tf.float32))["concatenate"]
                emb.append(embeds(mel, training=False).numpy())
            C, _t, _a, _b = confidence(np.vstack(emb), head, 0)
            scores[:, j] = C
            print(f"  {mode:5s} az {az:3d}  mean conf {C.mean():.4f}  max {C.max():.4f}",
                  flush=True)
        out[mode] = scores

    print("\n" + "=" * 70)
    print("Is the confidence curve smooth in azimuth?")
    print("=" * 70)
    print(f"white curve would give lag-1 autocorrelation "
          f"{-1.0/(len(AZIMUTHS)-1):+.3f}\n")
    print(f"{'variant':>8s} {'lag-1 autocorr':>22s} {'windows with r>0':>18s}")
    for mode in ("head", "onset"):
        r = np.array([circular_lag1(out[mode][k]) for k in range(n_win)])
        print(f"{mode:>8s} {r.mean():+11.3f} +/- {r.std():.3f} "
              f"{float((r > 0).mean()):17.1%}")

    print("\nbeam contrast, max minus median over the 36 azimuths")
    print(f"{'variant':>8s} {'mean':>10s} {'median':>10s}")
    for mode in ("head", "onset"):
        c = out[mode].max(axis=1) - np.median(out[mode], axis=1)
        print(f"{mode:>8s} {c.mean():10.4f} {np.median(c):10.4f}")

    print("\ndetections, plain rule: max over the 36 azimuths per window")
    print(f"{'variant':>8s} " + " ".join(f"{t:>8}" for t in THRESHOLDS) +
          f" {'distinct winning az':>21s}")
    for mode in ("head", "onset"):
        m = out[mode].max(axis=1)
        w = out[mode].argmax(axis=1)
        cells = " ".join(f"{int((m >= t).sum()):8d}" for t in THRESHOLDS)
        print(f"{mode:>8s} {cells} {len(set(w.tolist())):21d}")

    np.savez(os.path.join(os.environ.get("TMPDIR", "/tmp"),
                          f"onset_ab_{date}_S{speaker:02d}.npz"),
             head=out["head"], onset=out["onset"], azimuths=AZIMUTHS)
    print("\nA higher lag-1 autocorrelation for `onset` is the fix working.")
    print("Both near zero would mean onset alignment alone is not enough.")


if __name__ == "__main__":
    main()
