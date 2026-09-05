#!/usr/bin/env python3
"""Is the habitat noise prototype stable enough to pool across dates?

Between-date movement is meaningless on its own, so it is compared against the
movement you get inside a single date by splitting its recordings in half.
If between is about the same as within, the dates are exchangeable and pooling
adds real independent samples. If between is much larger, pooling would blend
two different background conditions.

Both are then read against the effect the study measures: delta vs mono ran
0.007 to 0.024.
"""
import glob, json, os, re, sys, itertools
import numpy as np

H = os.path.expanduser("~/sea-emb/2A400")
rng = np.random.default_rng(0)


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n else v


def proto(rows):
    m = rows / np.linalg.norm(rows, axis=1, keepdims=True)
    return unit(m.mean(axis=0))


def load(model, date, condition):
    """Rows for one condition, plus the (start, stop) span of each beam file."""
    rows, spans, at = [], [], 0
    for p in sorted(glob.glob(f"{H}/{model}/noise_{date}_{condition}_*_embeddings.npy")):
        arr = np.load(p).astype(np.float64)
        if not len(arr):
            continue
        rows.append(arr); spans.append((at, at + len(arr))); at += len(arr)
    return (np.concatenate(rows, axis=0), spans) if rows else (None, None)


def within_date(rows, per_beam, n_rep=40):
    """Prototype movement inside one date.

    Concatenating the accepted intervals into one file per beam threw away the
    source recording in the metadata, so the split is by position instead: each
    beam's windows are in recording order, so first half against second half is
    an early-part against late-part comparison of the same night.
    """
    early, late = [], []
    for start, stop in per_beam:
        n = stop - start
        if n < 4:
            continue
        mid = start + n // 2
        early.append(rows[start:mid]); late.append(rows[mid:stop])
    if not early:
        return None
    a, b = np.concatenate(early, axis=0), np.concatenate(late, axis=0)
    temporal = 1.0 - float(np.dot(proto(a), proto(b)))

    # random halves: pure sampling noise, no temporal structure
    idx = np.arange(len(rows))
    rand = []
    for _ in range(n_rep):
        rng.shuffle(idx)
        h = len(idx) // 2
        rand.append(1.0 - float(np.dot(proto(rows[idx[:h]]), proto(rows[idx[h:]]))))
    return temporal, float(np.mean(rand))


def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "birdnet"
    dates = sorted({m.group(1) for p in glob.glob(f"{H}/{model}/noise_*_embeddings.npy")
                    if (m := re.search(r"noise_(\d{4}-\d{2}-\d{2})_", os.path.basename(p)))})
    print(f"model {model} | tanggal tersedia: {', '.join(dates) or 'belum ada'}\n")
    for condition in ("dawn", "day", "dusk", "night"):
        have = {}
        for d in dates:
            rows, spans = load(model, d, condition)
            if rows is not None and len(rows) >= 4:
                have[d] = (rows, spans)
        if not have:
            continue
        print(f"── {condition} ──")
        for d, (rows, spans) in sorted(have.items()):
            w = within_date(rows, spans)
            if w is None:
                print(f"   {d}  {len(rows):5d} vektor  (terlalu sedikit)")
            else:
                print(f"   {d}  {len(rows):5d} vektor   dalam-hari awal-vs-akhir: {w[0]:.5f}"
                      f"   derau sampel murni: {w[1]:.5f}")
        if len(have) < 2:
            print("   (butuh >=2 tanggal untuk perbandingan antar hari)\n")
            continue
        print("   antar hari:")
        for a, b in itertools.combinations(sorted(have), 2):
            d = 1.0 - float(np.dot(proto(have[a][0]), proto(have[b][0])))
            gap = abs((np.datetime64(a) - np.datetime64(b)).astype(int))
            print(f"     {a} vs {b}  jarak {d:.5f}   selisih {gap} hari")
        print()


if __name__ == "__main__":
    main()
