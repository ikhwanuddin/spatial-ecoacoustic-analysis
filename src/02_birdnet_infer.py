"""
Module 2: BirdNET Batch Inference.
Runs BirdNET-Analyzer on all rendered WAV files (Mono, SA, LabIR, SPIR)
in a scratch recording folder, outputting a complete results.json.

Faithful adaptation of Cell 11 from the Silwood notebook.
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

# Ensure ffmpeg in miniforge sea env is in PATH for pydub
ffmpeg_bin = os.path.expanduser("~/miniforge3/envs/sea/bin")
if os.path.exists(ffmpeg_bin) and ffmpeg_bin not in os.environ.get("PATH", ""):
    os.environ["PATH"] = ffmpeg_bin + ":" + os.environ.get("PATH", "")

from birdnetlib import Recording
from birdnetlib.analyzer import Analyzer
from birdnetlib.batch import DirectoryMultiProcessingAnalyzer


def run_birdnet_batch(folder_path: str, date_obj: datetime = None, processes: int = 8):
    """
    Run BirdNET analysis on all WAV files in folder_path.
    Writes results.json in the same folder.
    """
    if date_obj is None:
        date_obj = datetime(2026, 4, 22)

    results_path = os.path.join(folder_path, "results.json")
    results_dict = {}

    def on_complete(recordings):
        for rec in recordings:
            if rec.error:
                print(f"⚠️  Error in {os.path.basename(rec.path)}: {rec.error_message}")
            else:
                file_name = os.path.basename(rec.path)
                results_dict[file_name] = rec.detections

        with open(results_path, "w") as f:
            json.dump(results_dict, f, indent=4)
        print(f"✅ Saved BirdNET results ({len(results_dict)} channels) to: {results_path}")

    analyzer = Analyzer()
    batch = DirectoryMultiProcessingAnalyzer(
        folder_path,
        analyzers=[analyzer],
        date=date_obj,
        min_conf=0.0,      # min_conf=0.0 to capture full confidence spectrum
        overlap=0.0,       # non-overlapping 3-second windows
        processes=processes
    )
    batch.on_analyze_directory_complete = on_complete
    batch.process()
    return results_path


def main():
    parser = argparse.ArgumentParser(description="Run BirdNET batch inference on rendered WAV folder.")
    parser.add_argument("folder", help="Path to recording folder containing rendered WAV files")
    parser.add_argument("--processes", type=int, default=8, help="Number of CPU worker processes")
    args = parser.parse_args()

    if not os.path.isdir(args.folder):
        print(f"❌ Directory not found: {args.folder}")
        return 1

    wav_count = len([f for f in os.listdir(args.folder) if f.endswith(".wav")])
    print(f"🚀 Running BirdNET on {wav_count} WAV files in: {args.folder}")

    t0 = time.time()
    out_json = run_birdnet_batch(args.folder, processes=args.processes)
    print(f"🏁 Finished in {time.time() - t0:.2f}s -> {out_json}")


if __name__ == "__main__":
    main()
