import json, glob, os, argparse
ap = argparse.ArgumentParser(); ap.add_argument("--method", default="bf_SPIR")
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(os.path.expanduser("~/beam-stab/*.json"))):
    d = json.load(open(f))
    for m in d["models"]:
        v = m.get("methods", {}).get(a.method)
        if not v or "n_files" not in v:
            continue
        rows.append((d["date"], m["model"], v))
hdr = "{:12s} {:17s} {:>6s} {:>7s} {:>7s} {:>8s} {:>8s} {:>6s} {:>22s}".format(
    "date","model","files","distinc","topshr%","adjagr%","chance%","ratio","global dominant")
print(hdr); print("-"*len(hdr))
for date, model, v in rows:
    r = v["adjacent_file_agree_pct"]/v["chance_agree_pct"] if v["chance_agree_pct"] else 0
    print("{:12s} {:17s} {:6d} {:7d} {:7.1f} {:8.1f} {:8.1f} {:6.2f} {:>22s}".format(
        date, model, v["n_files"], v["n_distinct_dominant"],
        v["global_dominant_file_share_pct"], v["adjacent_file_agree_pct"],
        v["chance_agree_pct"], r, v["global_dominant"][:22]))
# aggregate
import statistics as st
print("\nmedian adjacent-agreement ratio: {:.2f}  (n={})".format(
    st.median([v["adjacent_file_agree_pct"]/v["chance_agree_pct"] for _,_,v in rows if v["chance_agree_pct"]]), len(rows)))
