"""Does the detector care about absolute clip level?

The renderer peak-normalises every beam output over the whole file and per
direction (beamforming.py lines 113-115), and the whole-file peak differs across
directions by up to a factor 1.83 on 2026-04-30. bacpipe reads wavs with
librosa.load, which scales int16 to [-1, 1] and normalises nothing further, so
that factor reaches the model.

Two things follow, and both hinge on one question: is BirdNET's front end scale
invariant?

  If yes  - the per-direction normalisation is harmless, and an ABD pass 2 window
            may be scaled however is convenient.
  If no   - the existing beam comparison carries a level confound of up to 1.83x,
            and a pass 2 window normalised by its own peak is not the signal a
            whole-file render would have produced.

Tested on BirdNET's frozen preprocessor, which is where any normalisation would
live. If the mel spectrogram already differs with scale, everything downstream
does too.

usage: scale_invariance_test.py <wav path>
"""

import sys

import librosa
import numpy as np
import tensorflow as tf

PREPROCESSOR = ("/rds/general/user/ri322/home/spatial-ecoacoustic-analysis/"
                "bacpipe/checkpoints/birdnet/BirdNET_Preprocessor")
SR = 48000
LENGTH = 144000          # 3.0 s at 48 kHz


def main():
    wav = sys.argv[1]
    audio, _ = librosa.load(wav, sr=SR, mono=True)
    clip = audio[:LENGTH]
    if len(clip) < LENGTH:
        clip = np.pad(clip, (0, LENGTH - len(clip)))
    print(f"clip      : {wav}")
    print(f"peak      : {np.abs(clip).max():.6f}   rms {np.sqrt((clip**2).mean()):.6f}")

    loaded = tf.saved_model.load(PREPROCESSOR)
    fn = loaded.signatures["serving_default"]

    def spec(x):
        return fn(tf.convert_to_tensor(x[None, :], dtype=tf.float32))["concatenate"].numpy()

    base = spec(clip)
    print(f"mel shape : {base.shape}  range [{base.min():.4f}, {base.max():.4f}]")

    print("\nA front end that keeps absolute level would scale its own output with")
    print("the input. One that normalises leaves max and mean where they are, and")
    print("only numerical residue moves. The spread over a 400x input range is the")
    print("test, not the absolute size of the difference.\n")
    print(f"{'scale':>8s} {'mel_max':>9s} {'mel_mean':>9s} {'max_abs_diff':>13s} {'rel':>10s}")
    print(f"{1.0:8.3f} {base.max():9.4f} {base.mean():9.4f} {0.0:13.6f} {0.0:10.3e}")
    rels = []
    for s in (0.05, 0.1, 1.0 / 1.83, 0.5, 2.0, 20.0):
        got = spec(clip * s)
        d = np.abs(got - base)
        rel = d.max() / (np.abs(base).max() or 1.0)
        rels.append(rel)
        print(f"{s:8.3f} {got.max():9.4f} {got.mean():9.4f} {d.max():13.6f} {rel:10.3e}")
    print()
    if max(rels) < 1e-2:
        print(f"VERDICT: scale invariant to {max(rels):.1e} relative over a 400x range.")
        print("The front end normalises. The per-direction whole-file peak")
        print("normalisation is divided out again before the model sees anything, so")
        print("it is not a confound, and an ABD pass 2 window may be scaled however")
        print("is convenient.")
    else:
        print(f"VERDICT: level dependent, up to {max(rels):.1e} relative.")


if __name__ == "__main__":
    main()
