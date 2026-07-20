#!/usr/bin/env python3
"""
Core ML experiment v5 — coremltools v5.2 (last version with tflite converter).
"""

import os, sys, time, json, numpy as np, argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wav-dir", default="/Volumes/WD2TB/sea-data/Q0/2026-04-16/beamforming_LabIR")
    p.add_argument("--convert-only", action="store_true")
    p.add_argument("--bench-only", action="store_true")
    args = p.parse_args()

    import birdnetlib.analyzer as ba
    tf16 = os.path.join(os.path.dirname(ba.MODEL_PATH),
        "BirdNET_GLOBAL_6K_V2.4_MData_Model_V2_FP16.tflite")
    cmout = os.path.join(os.path.dirname(__file__), "models", "BirdNET_CoreML.mlmodel")

    print("="*60)
    print("Core ML Experiment v5 (coremltools v5.2)")
    print(f"  TF Lite: {tf16}\n  Core ML: {cmout}")
    print("="*60)

    if not args.bench_only:
        import coremltools as ct
        print(f"  coremltools: {ct.__version__}")
        print(f"  Converting tflite -> Core ML...")
        try:
            mlmodel = ct.convert(tf16, source="auto")
            mlmodel.save(cmout)
            print(f"  Saved: {cmout}")
            spec = mlmodel.get_spec()
            for i, inp in enumerate(spec.description.input):
                print(f"  Input {i}: {inp.name} -> {inp.type}")
            for i, out in enumerate(spec.description.output):
                print(f"  Output {i}: {out.name} -> {out.type}")
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback; traceback.print_exc()
            sys.exit(1)

    if args.convert_only:
        return
    if not os.path.exists(cmout):
        sys.exit("Run --convert-only first")

    # Benchmark
    import coremltools as ct, soundfile as sf, tensorflow.lite as tl

    wavs = sorted([os.path.join(args.wav_dir, f) for f in os.listdir(args.wav_dir)
                  if f.lower().endswith(".wav") and not f.startswith("._")])[:19]
    print(f"\n  Benchmark: {len(wavs)} WAV files")

    # Collect chunks
    chunks = []
    for w in wavs:
        a, sr = sf.read(w)
        for i in range(len(a) // 144000):
            chunks.append(a[i*144000:(i+1)*144000])
    n = len(chunks)
    print(f"  Chunks: {n} ({n/len(wavs):.0f}/file)")

    # Core ML
    ml = ct.models.MLModel(cmout)
    iname = ml.get_spec().description.input[0].name
    print(f"  Core ML input: {iname}")

    # TF Lite
    interp = tl.Interpreter(model_path=tf16, num_threads=1)
    interp.allocate_tensors()
    ii = interp.get_input_details()[0]["index"]
    oi = interp.get_output_details()[0]["index"]

    data0 = np.array([chunks[0]], dtype="float32")

    # TF Lite
    print("\n  -- TF Lite FP16 --")
    interp.resize_tensor_input(ii, [1, len(chunks[0])])
    interp.allocate_tensors()
    interp.set_tensor(ii, data0)
    interp.invoke()
    _ = interp.get_tensor(oi)

    t0 = time.time()
    for c in chunks:
        d = np.array([c], dtype="float32")
        interp.resize_tensor_input(ii, [1, len(c)])
        interp.allocate_tensors()
        interp.set_tensor(ii, d)
        interp.invoke()
        _ = interp.get_tensor(oi)[0]
    tf_sec = time.time() - t0
    print(f"    {tf_sec:.2f}s  |  {tf_sec/n*1000:.1f}ms/chunk  |  {n/tf_sec:.0f} chunks/s")

    # Core ML
    print("\n  -- Core ML --")
    _ = ml.predict({iname: data0})
    t0 = time.time()
    for c in chunks:
        _ = ml.predict({iname: np.array([c], dtype="float32")})
    cm_sec = time.time() - t0
    print(f"    {cm_sec:.2f}s  |  {cm_sec/n*1000:.1f}ms/chunk  |  {n/cm_sec:.0f} chunks/s")

    sp = tf_sec / cm_sec
    print(f"\n  Core ML is {sp:.1f}x {'faster' if sp>1 else 'slower'} than TF Lite FP16")

    out = os.path.join(os.path.dirname(__file__), "coreml_results.json")
    json.dump({"TF_Lite_FP16_ms_per_chunk": round(tf_sec/n*1000, 2),
               "CoreML_ms_per_chunk": round(cm_sec/n*1000, 2),
               "speedup": round(sp, 2)}, open(out, "w"), indent=2)
    print(f"  Saved: {out}")

if __name__ == "__main__":
    main()
