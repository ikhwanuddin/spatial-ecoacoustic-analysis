"""
Module 4: Paired Comparison & Multi-Threshold Analysis.
Pairs mono, sa, and beamformed detections per species across all timestamps,
and evaluates detection counts across a sweep of confidence thresholds.

Faithful adaptation of Cell 25 & Cell 19 from the Silwood notebook.
"""

import os
import json
import argparse
import numpy as np
from typing import Dict, List, Any


DEFAULT_THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8]


def pair_methods(method_a_data: Dict[str, Any], method_b_data: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """
    Pairs detections between two methods per species by timestamp.
    Returns dict of species -> list of paired events.
    """
    paired = {}
    all_species = sorted(set(list(method_a_data.keys()) + list(method_b_data.keys())))

    for sp in all_species:
        paired[sp] = []
        a_info = method_a_data.get(sp, {})
        b_info = method_b_data.get(sp, {})

        a_dict = dict(zip(a_info.get("start_time_list", []), a_info.get("conf_list", [])))
        b_dict = dict(zip(b_info.get("start_time_list", []), b_info.get("conf_list", [])))

        all_times = sorted(set(list(a_dict.keys()) + list(b_dict.keys())))
        for t in all_times:
            paired[sp].append({
                "timestamp": t,
                "a_detected": t in a_dict,
                "a_conf": a_dict.get(t, None),
                "b_detected": t in b_dict,
                "b_conf": b_dict.get(t, None),
            })
    return paired


def evaluate_threshold_counts(processed_data: Dict[str, Any], thresholds: List[float]) -> Dict[str, Any]:
    """
    Compute detection counts and unique species counts across candidate thresholds.
    """
    methods = ["mono_channel", "sa_channel", "beamformed_LabIR", "beamformed_SPIR", "beamformed_all"]
    summary = {m: {"total_detections": {}, "species_count": {}} for m in methods}

    for m in methods:
        data = processed_data.get(m, {})
        for t in thresholds:
            t_str = f"{t:.2f}"
            det_count = 0
            species_passed = 0

            for sp, sinfo in data.items():
                confs = [c for c in sinfo.get("conf_list", []) if c >= t]
                if confs:
                    det_count += len(confs)
                    species_passed += 1

            summary[m]["total_detections"][t_str] = det_count
            summary[m]["species_count"][t_str] = species_passed

    return summary


def format_markdown_table(summary: Dict[str, Any], thresholds: List[float]) -> str:
    """Format evaluation summary into a clean GitHub-style Markdown table."""
    lines = [
        "| Threshold | Mono Detections | SA Detections | LabIR Detections | SPIR Detections | BF Gain vs Mono (%) |",
        "|---|---|---|---|---|---|"
    ]
    for t in thresholds:
        t_str = f"{t:.2f}"
        mono_n = summary["mono_channel"]["total_detections"][t_str]
        sa_n = summary["sa_channel"]["total_detections"][t_str]
        labir_n = summary["beamformed_LabIR"]["total_detections"][t_str]
        spir_n = summary["beamformed_SPIR"]["total_detections"][t_str]
        bf_best = max(labir_n, spir_n)

        gain_str = f"+{(bf_best - mono_n) / mono_n * 100:.1f}%" if mono_n > 0 else ("N/A" if bf_best == 0 else "∞")
        lines.append(f"| **{t_str}** | {mono_n} | {sa_n} | {labir_n} | {spir_n} | **{gain_str}** |")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Pair detections and evaluate across confidence thresholds.")
    parser.add_argument("processed_json", help="Path to processed.json")
    parser.add_argument("--thresholds", default=",".join(map(str, DEFAULT_THRESHOLDS)), help="Comma-separated thresholds")
    parser.add_argument("--out-dir", default=None, help="Directory to save paired_detections.json and summary table")
    args = parser.parse_args()

    if not os.path.isfile(args.processed_json):
        print(f"❌ File not found: {args.processed_json}")
        return 1

    thresholds = [float(x.strip()) for x in args.thresholds.split(",") if x.strip()]
    out_dir = args.out_dir or os.path.dirname(args.processed_json)
    os.makedirs(out_dir, exist_ok=True)

    with open(args.processed_json, "r") as f:
        processed = json.load(f)

    # 1. Pair Mono vs Beamformed (LabIR and SPIR)
    paired_labir = pair_methods(processed.get("mono_channel", {}), processed.get("beamformed_LabIR", {}))
    paired_spir = pair_methods(processed.get("mono_channel", {}), processed.get("beamformed_SPIR", {}))

    paired_output = {
        "mono_vs_LabIR": paired_labir,
        "mono_vs_SPIR": paired_spir,
    }
    paired_file = os.path.join(out_dir, "paired_detections.json")
    with open(paired_file, "w") as f:
        json.dump(paired_output, f, indent=4, ensure_ascii=False)
    print(f"✅ Saved paired detections to: {paired_file}")

    # 2. Multi-threshold counts evaluation
    summary = evaluate_threshold_counts(processed, thresholds)
    summary_file = os.path.join(out_dir, "threshold_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=4)

    # 3. Print markdown table
    md_table = format_markdown_table(summary, thresholds)
    md_file = os.path.join(out_dir, "threshold_summary.md")
    with open(md_file, "w") as f:
        f.write(md_table + "\n")

    print("\n📊 Multi-Threshold Detection Counts:")
    print(md_table)


if __name__ == "__main__":
    main()
