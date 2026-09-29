"""
RTF rerun: one recording end to end in memory on the GPU, no scratch WAVs.

FLAC -> mic channels -> HPF -> STFT (CPU)
     -> per group of beams: einsum with RTF weights -> ISTFT -> peak norm -> x3 resample -> BirdNET (GPU)
     -> results.json (raw), processed.json (location filter), threshold_summary, paired_detections,
        audit clips: every beam processed.json chose + mono, on ephemeral (config.clip_dir)

Same maths as render_signals.py + birdnet_infer.run_birdnet_gpu (librosa STFT/ISTFT with a Hamming
window, scipy resample_poly(3, 1), float32 WAV values); only the device and float precision differ.
"""

import os
import sys
import json
import time
from datetime import datetime

import numpy as np
import scipy.signal
import soundfile as sf
import librosa

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import birdnet_infer
if not birdnet_infer.is_gpu_available():
    raise RuntimeError(f"no GPU or BirdNET GPU model missing in {birdnet_infer.CKPT_DIR}")
from birdnet_infer import tf, get_birdnet_gpu_model   # loads the CUDA libs first
from config import (FS_TARGET, FRAME_LEN, HOP_LEN, HIGH_PASS_CUTOFF, BIRDNET_MIN_CONF, DEFAULT_THRESHOLDS,
                    MONITORING_DATA, LOCATION_MAP, FLAC_CHANNELS, clip_dir)
from render_signals import load_and_verify_flac, butter_highpass_filter, get_beam_weights_tensor, mic_channels
from extract_detections import process_results_file, local_species
from pair_and_recap import pair_methods, evaluate_threshold_counts, format_markdown_table

BEAM_GROUP = 16          # beams rendered + scored together on the GPU
BATCH = 256              # BirdNET windows per forward pass
WIN = 144000             # 3 s at 48 kHz
AUDIT_MIN_CONF = BIRDNET_MIN_CONF
AUDIT_GROUPS = ["beamformed_LabIR", "beamformed_SPIR", "beamformed_WCIR_own", "beamformed_WCIR_cross",
                "beamformed_all"]

_WINDOW = scipy.signal.get_window("hamming", FRAME_LEN, fftbins=True)          # = librosa 'hamming'
_RESAMPLE_H = scipy.signal.firwin(61, 1.0 / 3, window=("kaiser", 5.0)) * 3     # = resample_poly(x, 3, 1)


def _istft(Z):
    """librosa.istft(Z, hop_length=HOP_LEN, window='hamming') for a (k, f, t) complex64 tensor."""
    n_frames = Z.shape[-1]
    frames = tf.signal.irfft(tf.transpose(Z, [0, 2, 1]), fft_length=[FRAME_LEN])    # (k, t, n_fft)
    frames = frames * tf.constant(_WINDOW, tf.float32)
    y = tf.signal.overlap_and_add(frames, HOP_LEN)                                   # (k, n_fft + hop*(t-1))
    wss = librosa.filters.window_sumsquare(window="hamming", n_frames=n_frames, win_length=FRAME_LEN,
                                           n_fft=FRAME_LEN, hop_length=HOP_LEN, dtype=np.float32)
    tiny = np.finfo(np.float32).tiny
    wss = np.where(wss > tiny, wss, 1.0).astype(np.float32)
    y = y / tf.constant(wss)
    start = FRAME_LEN // 2
    return y[:, start:start + HOP_LEN * (n_frames - 1)]


def _resample3(y):
    """scipy.signal.resample_poly(y, 3, 1) for a (k, n) float32 tensor."""
    k, n = y.shape
    up = tf.reshape(tf.stack([y, tf.zeros_like(y), tf.zeros_like(y)], axis=-1), [k, 3 * n])
    h = tf.constant(_RESAMPLE_H[::-1].copy()[:, None, None], tf.float32)              # conv1d = correlation
    out = tf.nn.conv1d(tf.pad(up, [[0, 0], [30, 30]])[:, :, None], h, stride=1, padding="VALID")
    return out[:, :, 0]


def _peak_norm(y):
    return y / (tf.reduce_max(tf.abs(y), axis=1, keepdims=True) + 1e-12)


def _birdnet(sig48, names, model, prep_fn, labels, results):
    """Score (k, n) 48 kHz signals in 3 s windows; append detections >= BIRDNET_MIN_CONF to results."""
    n_win = sig48.shape[1] // WIN
    wins = tf.reshape(sig48[:, :n_win * WIN], [-1, WIN])                               # (k * n_win, WIN)
    probs = []
    for i in range(0, wins.shape[0], BATCH):
        probs.append(tf.sigmoid(model(prep_fn(wins[i:i + BATCH])["concatenate"], training=False)).numpy())
    probs = np.vstack(probs).reshape(len(names), n_win, -1)
    for name, p in zip(names, probs):
        dets = results.setdefault(name, [])
        for k, c in zip(*np.where(p >= BIRDNET_MIN_CONF)):
            dets.append({"common_name": labels[c], "confidence": round(float(p[k, c]), 4),
                         "start_time": float(k * 3.0), "end_time": float((k + 1) * 3.0)})


def clip_name(rec_name: str, start: float, channel: str) -> str:
    return f"{rec_name}_{start:.1f}s_{channel}"


def audit_rows(processed: dict, rec_name: str) -> list:
    """One row per (group, species, window) claim in processed.json at >= AUDIT_MIN_CONF."""
    mono = {(sp, st): c for sp, i in processed["mono_channel"].items()
            for c, st in zip(i["conf_list"], i["start_time_list"])}
    rows = []
    for g in AUDIT_GROUPS:
        for sp, i in processed[g].items():
            for c, st, ch in zip(i["conf_list"], i["start_time_list"], i["primary_channel_list"]):
                if c >= AUDIT_MIN_CONF:
                    rows.append({"rec": rec_name, "group": g, "species": sp, "window_seconds": [st, st + 3.0],
                                 "channel": ch, "conf": c, "mono_conf": mono.get((sp, st)),
                                 "clip": clip_name(rec_name, st, ch), "mono_clip": clip_name(rec_name, st, "mono.wav")})
    return rows


def _prepare(flac_path: str, location: str, err_dir: str):
    """FLAC -> (mono, sa, X) exactly as used for BirdNET and the clips, or (None, None, error dict)."""
    audio, sr, err = load_and_verify_flac(flac_path, err_dir, FLAC_CHANNELS.get(location, 6))
    if audio is None:
        return None, None, err
    audio = audio[:, mic_channels(location)]
    if sr != FS_TARGET:
        audio = librosa.resample(audio.T, orig_sr=sr, target_sr=FS_TARGET).T
    filt = butter_highpass_filter(audio, cutoff=HIGH_PASS_CUTOFF, fs=FS_TARGET)
    mono = filt[:, 0] / (np.max(np.abs(filt[:, 0])) + 1e-12)
    sa = np.mean(filt, axis=1)
    sa = sa / (np.max(np.abs(sa)) + 1e-12)
    X = np.stack([librosa.stft(filt[:, c], n_fft=FRAME_LEN, hop_length=HOP_LEN, window="hamming")
                  for c in range(filt.shape[1])]).astype(np.complex64)
    return mono, sa, X


def _renderer(location: str, X):
    """(catalog, render) where render(idx) = peak-normalised 16 kHz signals of beams idx."""
    catalog, W = get_beam_weights_tensor(location)
    Xg = tf.constant(X)

    def render(idx):
        Z = tf.einsum("kcf,cft->kft", tf.constant(W[idx].astype(np.complex64)), Xg)
        return _peak_norm(_istft(Z))

    return catalog, render


def _write_clips(rows, clips_dir, mono, catalog, render):
    """Mono clip + beam clip per row (1 s buffer each side); existing files are kept. Returns files written."""
    os.makedirs(clips_dir, exist_ok=True)
    written = 0

    def cut(name, st, sig):
        nonlocal written
        a, b = int(max(0.0, st - 1.0) * FS_TARGET), int((st + 3.0 + 1.0) * FS_TARGET)
        dst = os.path.join(clips_dir, name)
        if not os.path.isfile(dst):
            sf.write(dst, np.asarray(sig[a:b], dtype=np.float32), FS_TARGET, subtype="PCM_16")
            written += 1

    for r in rows:
        cut(r["mono_clip"], r["window_seconds"][0], mono)
    need = sorted({i for i, c in enumerate(catalog) if c[0] in {r["channel"] for r in rows}})
    for g in range(0, len(need), BEAM_GROUP):          # render only the beams the clips need
        idx = need[g:g + BEAM_GROUP]
        y = render(idx).numpy()
        sig = {catalog[i][0]: y[j] for j, i in enumerate(idx)}
        for r in rows:
            if r["channel"] in sig:
                cut(r["clip"], r["window_seconds"][0], sig[r["channel"]])
    return written


def regen_clips(flac_path: str, location: str, date_str: str, out_rec_dir: str, clips_dir: str = None) -> dict:
    """Recreate the audit clips of one recording that are missing (same maths as process_recording)."""
    clips_dir = clips_dir or clip_dir(location, date_str, os.path.basename(out_rec_dir))
    rows = json.load(open(os.path.join(out_rec_dir, "audit_clips.json")))
    rows = [r for r in rows if not (os.path.isfile(os.path.join(clips_dir, r["clip"]))
                                    and os.path.isfile(os.path.join(clips_dir, r["mono_clip"])))]
    if not rows:
        return {"written": 0}
    mono, sa, X = _prepare(flac_path, location, out_rec_dir)
    if mono is None:
        return {"error": X}
    catalog, render = _renderer(location, X)
    return {"written": _write_clips(rows, clips_dir, mono, catalog, render)}


def process_recording(flac_path: str, location: str, date_str: str, out_rec_dir: str) -> dict:
    """Run one recording; returns timings. JSON into out_rec_dir, clips into clip_dir(loc, date, rec)."""
    t = {"t0": time.time()}
    os.makedirs(out_rec_dir, exist_ok=True)
    rec_name = os.path.splitext(os.path.basename(flac_path))[0]
    model, prep_fn, labels = get_birdnet_gpu_model()

    mono, sa, X = _prepare(flac_path, location, out_rec_dir)
    if mono is None:
        return {"error": X}
    t["prep"] = time.time()

    results = {}
    plain = tf.constant(np.stack([mono, sa]).astype(np.float32))
    _birdnet(_resample3(plain), ["mono.wav", "sa.wav"], model, prep_fn, labels, results)

    catalog, render = _renderer(location, X)

    for g in range(0, len(catalog), BEAM_GROUP):
        idx = list(range(g, min(g + BEAM_GROUP, len(catalog))))
        _birdnet(_resample3(render(idx)), [catalog[i][0] for i in idx], model, prep_fn, labels, results)
    t["beams"] = time.time()

    # Outputs
    res_path = os.path.join(out_rec_dir, "results.json")
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    processed = process_results_file(res_path, conf_thresh=0.0, allowed=local_species(date_str))
    with open(os.path.join(out_rec_dir, "processed.json"), "w", encoding="utf-8") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)
    mono_d = processed.get("mono_channel", {})
    paired = {f"mono_vs_{g[len('beamformed_'):]}": pair_methods(mono_d, processed.get(g, {}))
              for g in ["beamformed_LabIR", "beamformed_SPIR", "beamformed_WCIR_own", "beamformed_WCIR_cross"]}
    with open(os.path.join(out_rec_dir, "paired_detections.json"), "w", encoding="utf-8") as f:
        json.dump(paired, f, ensure_ascii=False)
    summary = evaluate_threshold_counts(processed, DEFAULT_THRESHOLDS)
    with open(os.path.join(out_rec_dir, "threshold_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)
    with open(os.path.join(out_rec_dir, "threshold_summary.md"), "w") as f:
        f.write(format_markdown_table(summary, DEFAULT_THRESHOLDS) + "\n")

    # Audit clips: every beam processed.json chose (any group, species x window), stored once per
    # beam x window, plus one mono clip per window. 1 s buffer each side.
    rows = audit_rows(processed, rec_name)
    with open(os.path.join(out_rec_dir, "audit_clips.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False)
    _write_clips(rows, clip_dir(location, date_str, rec_name), mono, catalog, render)
    t["out"] = time.time()
    return {"prep_s": round(t["prep"] - t["t0"], 1), "beams_s": round(t["beams"] - t["prep"], 1),
            "out_s": round(t["out"] - t["beams"], 1), "n_beams": len(catalog), "n_clips": len({r["clip"] for r in rows} | {r["mono_clip"] for r in rows}),
            "dur_s": round(len(mono) / FS_TARGET, 1)}


if __name__ == "__main__":
    # python src/sea_gpu.py <location> <date> <flac name> <out_rec_dir>
    loc, date, name, out = sys.argv[1:5]
    flac = os.path.join(MONITORING_DATA, LOCATION_MAP[loc], date, name)
    print(process_recording(flac, loc, date, out))
