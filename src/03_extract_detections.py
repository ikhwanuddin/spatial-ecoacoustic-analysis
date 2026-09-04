"""
Module 3: Multi-Directional Automated Source Selection.
Processes results.json into processed.json by extracting unique detections
per (species, start_time) with the winning beam direction and max confidence.

Faithful adaptation of Cell 15 from the Silwood notebook.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Any


def extract_unique_channel_detections(results_dict: Dict, channel_pattern: str, conf_thresh: float = 0.0) -> List[Dict]:
    """
    Extract unique detections for a channel subset (e.g. 'LabIR', 'SPIR', 'mono', 'sa').
    For each (species_name, start_time), selects the channel yielding the highest confidence.
    """
    conf_detections = {}

    for channel in results_dict.keys():
        if channel_pattern.lower() in channel.lower():
            for det in results_dict[channel]:
                conf = det.get("confidence", 0.0)
                if conf >= conf_thresh:
                    species_name = det.get("common_name", "Unknown")
                    start_time = round(float(det.get("start_time", 0.0)), 1)
                    prim_key = f"{species_name}_{start_time}"

                    if prim_key not in conf_detections or conf > conf_detections[prim_key].get("confidence", 0.0):
                        det_copy = det.copy()
                        det_copy["start_time"] = start_time
                        det_copy["primary_channel"] = channel
                        conf_detections[prim_key] = det_copy

    return list(conf_detections.values())


def collate_species_stats(detections: List[Dict]) -> Dict[str, Any]:
    """Collate detection events into per-species summary metrics."""
    species_dict = {}

    for det in detections:
        species = det.get("common_name", "Unknown")
        conf = float(det.get("confidence", 0.0))
        start_time = float(det.get("start_time", 0.0))
        chan = det.get("primary_channel", "")

        if species not in species_dict:
            species_dict[species] = {
                "conf_list": [],
                "start_time_list": [],
                "primary_channel_list": [],
            }

        species_dict[species]["conf_list"].append(conf)
        species_dict[species]["start_time_list"].append(start_time)
        if chan:
            species_dict[species]["primary_channel_list"].append(chan)

    # Compute summary statistics
    for species, sdata in species_dict.items():
        confs = sdata["conf_list"]
        sdata["count"] = len(confs)
        if confs:
            sdata["conf_avg"] = round(float(np.mean(confs)), 4)
            sdata["conf_median"] = round(float(np.median(confs)), 4)
            sdata["conf_stdev"] = round(float(np.std(confs)), 4) if len(confs) > 1 else 0.0
            sdata["conf_max"] = round(float(np.max(confs)), 4)
        else:
            sdata["conf_avg"] = sdata["conf_median"] = sdata["conf_stdev"] = sdata["conf_max"] = 0.0

    return species_dict


def process_results_file(results_path: str, conf_thresh: float = 0.0) -> Dict[str, Any]:
    """Process results.json into processed.json structure."""
    with open(results_path, "r") as f:
        results = json.load(f)

    processed = {
        "mono_channel": collate_species_stats(
            extract_unique_channel_detections(results, "mono.wav", conf_thresh)
        ),
        "sa_channel": collate_species_stats(
            extract_unique_channel_detections(results, "sa.wav", conf_thresh)
        ),
        "beamformed_LabIR": collate_species_stats(
            extract_unique_channel_detections(results, "LabIR", conf_thresh)
        ),
        "beamformed_SPIR": collate_species_stats(
            extract_unique_channel_detections(results, "SPIR", conf_thresh)
        ),
        "beamformed_all": collate_species_stats(
            extract_unique_channel_detections(results, "IR", conf_thresh)
        ),
    }
    return processed


def main():
    parser = argparse.ArgumentParser(description="Extract winning beam detections from results.json.")
    parser.add_argument("results_json", help="Path to results.json")
    parser.add_argument("--conf-thresh", type=float, default=0.0, help="Minimum confidence threshold (default: 0.0)")
    parser.add_argument("--out", default=None, help="Output path for processed.json")
    args = parser.parse_args()

    if not os.path.isfile(args.results_json):
        print(f"❌ File not found: {args.results_json}")
        return 1

    out_path = args.out or os.path.join(os.path.dirname(args.results_json), "processed.json")
    processed = process_results_file(args.results_json, conf_thresh=args.conf_thresh)

    with open(out_path, "w") as f:
        json.dump(processed, f, indent=4, ensure_ascii=False)

    print(f"✅ Saved processed detections to: {out_path}")
    print(f"   Mono species: {len(processed['mono_channel'])}")
    print(f"   SA species:   {len(processed['sa_channel'])}")
    print(f"   LabIR species: {len(processed['beamformed_LabIR'])}")
    print(f"   SPIR species:  {len(processed['beamformed_SPIR'])}")


if __name__ == "__main__":
    main()
