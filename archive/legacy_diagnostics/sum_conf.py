#!/usr/bin/env python
"""Summarise beam_confidence outputs."""
import json, glob, os, sys, argparse

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/beam-conf"))
ap.add_argument("--method", default="bf_SPIR")
ap.add_argument("--what", default="lag", choices=["lag", "ladder"])
a = ap.parse_args()

files = sorted(glob.glob(os.path.join(a.dir, "*.json")))
if a.what == "lag":
    hdr = "{:12s} {:14s} {:>4s} {:>6s} {:>9s} {:>9s} {:>9s} {:>7s} {:>9s} {:>9s}".format(
        "date", "model", "lag", "n", "rand", "byConf", "gainConf", "win%", "byNoise", "gainNoise")
    print(hdr); print("-" * len(hdr))
    for f in files:
        d = json.load(open(f))
        for m in d["models"]:
            v = m.get("methods", {}).get(a.method)
            if not v or not v.get("available"):
                continue
            for lag, e in v["ladder"].get("lag_sweep", {}).items():
                if lag.startswith("_"): continue
                c = e.get("sel_by_conf", {}); nd = e.get("sel_by_noise_distance", {})
                print("{:12s} {:14s} {:>4s} {:6d} {:+9.4f} {:+9.4f} {:+9.4f} {:7.1f} {:+9.4f} {:+9.4f}".format(
                    d["date"], m["model"], lag, e["n"],
                    e["random_beam"]["mean_lift"], c.get("mean_lift", 0),
                    c.get("gain_over_random", 0), c.get("win_pct", 0),
                    nd.get("mean_lift", 0), nd.get("gain_over_random", 0)))
else:
    keys = ["random_beam", "fixed_best_beam", "sel_noise_distance",
            "sel_class_split_scoreB", "random_beam_scoreB",
            "oracle_max", "oracle_max_bias_corrected"]
    hdr = "{:12s} {:14s} ".format("date", "model") + " ".join("{:>13s}".format(k[:13]) for k in keys)
    print(hdr); print("-" * len(hdr))
    for f in files:
        d = json.load(open(f))
        for m in d["models"]:
            v = m.get("methods", {}).get(a.method)
            if not v or not v.get("available"):
                continue
            L = v["ladder"]
            row = "{:12s} {:14s} ".format(d["date"], m["model"])
            row += " ".join("{:>13s}".format(
                "{:+.4f}/{:.0f}".format(L[k]["mean_lift"], L[k]["win_pct"]) if k in L else "-")
                for k in keys)
            print(row)
            oc = L.get("lag_sweep", {}).get("_other_file_control")
            if oc:
                print("             other-file control  byConf {:+.4f} (gain {:+.4f})  byNoise {:+.4f}".format(
                    oc.get("sel_by_conf", {}).get("mean_lift", 0),
                    oc.get("sel_by_conf", {}).get("gain_over_random", 0),
                    oc.get("sel_by_noise_distance", {}).get("mean_lift", 0)))
            bc = L.get("by_condition")
            if bc:
                for c, e in bc.items():
                    print("             {:8s} n={:6d}  sel {:+.4f}  rand {:+.4f}  gain {:+.4f}  win {:.1f}%".format(
                        c, e["n"], e["sel_by_conf"], e["random_beam"], e["gain_over_random"], e["win_pct"]))
            ca = L.get("criterion_agreement")
            if ca:
                print("             agree noise-vs-conf {:.2f}% (chance {:.2f}%)  rankcorr {:+.3f}".format(
                    ca["noise_vs_conf_agree_pct"], ca["chance_pct"], ca["rank_corr_of_beam_means"]))
