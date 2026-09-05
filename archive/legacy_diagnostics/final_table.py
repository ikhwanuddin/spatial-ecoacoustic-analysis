#!/usr/bin/env python
"""The headline table: unbiased lag-1 selection vs the random-beam control."""
import json, glob, os, argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default=os.path.expanduser("~/beam-conf2"))
ap.add_argument("--method", default="bf_SPIR")
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(a.dir, "*.json"))):
    d = json.load(open(f))
    for m in d["models"]:
        v = m.get("methods", {}).get(a.method)
        if not v or not v.get("available"):
            continue
        L = v["ladder"]
        e = L.get("lag_sweep", {}).get("1")
        if not e:
            continue
        c = e.get("sel_by_conf", {})
        nd = e.get("sel_by_noise_distance", {})
        ci = c.get("ci95_gain") or {}
        oc = L.get("lag_sweep", {}).get("_other_file_control", {}).get("sel_by_conf", {})
        rows.append(dict(date=d["date"], model=m["model"], n=e["n"],
                         rand=e["random_beam"]["mean_lift"],
                         sel=c.get("mean_lift", np.nan),
                         win=c.get("win_pct", np.nan),
                         gain=c.get("gain_over_random", np.nan),
                         lo=ci.get("lo"), hi=ci.get("hi"), nrec=ci.get("n_recordings"),
                         nd_gain=nd.get("gain_over_random"),
                         other=oc.get("gain_over_random"),
                         has_ref=m.get("has_noise_reference")))

hdr = "{:12s} {:15s} {:>6s} {:>5s} {:>8s} {:>8s} {:>6s} {:>8s} {:>18s} {:>9s} {:>8s}".format(
    "date", "model", "n", "recs", "random", "selected", "win%", "gain", "gain 95% CI",
    "noise-sel", "x-file")
print(hdr); print("-" * len(hdr))
for r in rows:
    ci = "[{:+.4f},{:+.4f}]".format(r["lo"], r["hi"]) if r["lo"] is not None else "-"
    print("{:12s} {:15s} {:6d} {:5s} {:+8.4f} {:+8.4f} {:6.1f} {:+8.4f} {:>18s} {:>9s} {:>8s}".format(
        r["date"], r["model"], r["n"], str(r["nrec"] or "-"), r["rand"], r["sel"], r["win"],
        r["gain"], ci,
        "{:+.4f}".format(r["nd_gain"]) if r["nd_gain"] is not None else "-",
        "{:+.4f}".format(r["other"]) if r["other"] is not None else "-"))

print()
for model in sorted({r["model"] for r in rows}):
    g = np.array([r["gain"] for r in rows if r["model"] == model])
    pos = int((g > 0).sum())
    sig = [r for r in rows if r["model"] == model and r["lo"] is not None and r["lo"] > 0]
    print("{:15s} n_dates={:2d}  gain median {:+.4f}  range [{:+.4f},{:+.4f}]  positive {}/{}  CI excludes 0: {}".format(
        model, len(g), float(np.median(g)), float(g.min()), float(g.max()), pos, len(g), len(sig)))
nd = np.array([r["nd_gain"] for r in rows if r["nd_gain"] is not None])
if len(nd):
    print("\nnoise-distance selection, gain over random: median {:+.4f}  range [{:+.4f},{:+.4f}]  n={}".format(
        float(np.median(nd)), float(nd.min()), float(nd.max()), len(nd)))
oc = np.array([r["other"] for r in rows if r["other"] is not None])
if len(oc):
    print("cross-file control,               gain over random: median {:+.4f}  range [{:+.4f},{:+.4f}]  n={}".format(
        float(np.median(oc)), float(oc.min()), float(oc.max()), len(oc)))
