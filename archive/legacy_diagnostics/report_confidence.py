#!/usr/bin/env python
"""Headline report: unbiased lag-1 selection against the random-beam control.

Reads the beam_confidence JSON in the audits dir. Prints the per-date table,
then the summary twice: over every date, and again with the dates whose
embeddings are only about half covered excluded, so it is visible whether the
conclusion depends on them.
"""
import argparse, glob, json, os, re
import numpy as np

from embedding_schema import audits_dir

SUSPECT = {("birdnet", "2026-04-22"), ("avesecho_passt", "2026-05-04")}

ap = argparse.ArgumentParser()
ap.add_argument("--location", default="2A400")
ap.add_argument("--method", default="bf_SPIR")
a = ap.parse_args()

rows = []
for f in sorted(glob.glob(os.path.join(audits_dir(a.location), "*_beam_confidence_*.json"))):
    d = json.load(open(f))
    for m in d["models"]:
        if not m.get("available"):
            continue
        v = m.get("methods", {}).get(a.method)
        sa = m.get("methods", {}).get("sa", {})
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
        rows.append(dict(
            date=d["date"], model=m["model"], n=e["n"],
            recs=ci.get("n_recordings"),
            rand=e["random_beam"]["mean_lift"],
            sel=c.get("mean_lift"), win=c.get("win_pct"),
            gain=c.get("gain_over_random"), lo=ci.get("lo"), hi=ci.get("hi"),
            nd=nd.get("gain_over_random"), other=oc.get("gain_over_random"),
            sa=(sa.get("lift") or {}).get("mean_lift") if sa.get("available") else None,
            sa_win=(sa.get("lift") or {}).get("win_pct") if sa.get("available") else None,
            suspect=(m["model"], d["date"]) in SUSPECT))

hdr = ("{:12s} {:15s} {:>6s} {:>5s} {:>8s} {:>8s} {:>8s} {:>6s} {:>8s} {:>17s} {:>9s} {:>2s}"
       .format("date", "model", "n", "recs", "random", "sa", "sel t-1", "win%", "gain",
               "gain 95% CI", "noise-sel", ""))
print(hdr); print("-" * len(hdr))
for r in rows:
    ci = "[{:+.4f},{:+.4f}]".format(r["lo"], r["hi"]) if r["lo"] is not None else "-"
    print("{:12s} {:15s} {:6d} {:>5s} {:+8.4f} {:>8s} {:+8.4f} {:6.1f} {:+8.4f} {:>17s} {:>9s} {:>2s}"
          .format(r["date"], r["model"], r["n"], str(r["recs"] or "-"), r["rand"],
                  "{:+.4f}".format(r["sa"]) if r["sa"] is not None else "-",
                  r["sel"], r["win"], r["gain"], ci,
                  "{:+.4f}".format(r["nd"]) if r["nd"] is not None else "-",
                  "!" if r["suspect"] else ""))


def summarise(label, sel):
    print(f"\n--- {label} ---")
    for model in sorted({r["model"] for r in sel}):
        g = np.array([r["gain"] for r in sel if r["model"] == model])
        sig = sum(1 for r in sel if r["model"] == model and r["lo"] is not None and r["lo"] > 0)
        print("  {:15s} n={:2d}  gain median {:+.4f}  range [{:+.4f},{:+.4f}]  positif {}/{}  CI>0 {}"
              .format(model, len(g), float(np.median(g)), float(g.min()), float(g.max()),
                      int((g > 0).sum()), len(g), sig))
    nd = np.array([r["nd"] for r in sel if r["nd"] is not None])
    if len(nd):
        print("  {:15s} n={:2d}  gain median {:+.4f}  range [{:+.4f},{:+.4f}]"
              .format("noise-distance", len(nd), float(np.median(nd)), float(nd.min()), float(nd.max())))
    oc = np.array([r["other"] for r in sel if r["other"] is not None])
    if len(oc):
        print("  {:15s} n={:2d}  gain median {:+.4f}  range [{:+.4f},{:+.4f}]"
              .format("cross-file ctrl", len(oc), float(np.median(oc)), float(oc.min()), float(oc.max())))
    # paired beamforming vs signal averaging, same date and model
    # beamforming against signal averaging, paired on the same date and detector
    for model in sorted({r["model"] for r in sel}) + ["(gabungan)"]:
        pick = sel if model == "(gabungan)" else [r for r in sel if r["model"] == model]
        pairs = [(r["sel"] - r["sa"]) for r in pick
                 if r["sa"] is not None and r["sel"] is not None]
        if not pairs:
            continue
        q = np.array(pairs)
        print("  bf_SPIR-sa  {:16s} n={:2d}  median {:+.4f}  positif {}/{}"
              .format(model, len(q), float(np.median(q)), int((q > 0).sum()), len(q)))


summarise("semua tanggal", rows)
summarise("tanpa birdnet 04-22 dan avesecho 05-04", [r for r in rows if not r["suspect"]])
