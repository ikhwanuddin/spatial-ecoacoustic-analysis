import json, glob, os
from collections import defaultdict
agg = defaultdict(lambda: [0, 0.0, 0.0])
print("%-12s %-15s %-7s %7s %9s %9s %9s %7s" % ("date","model","cond","n","sel","rand","gain","win%"))
for f in sorted(glob.glob(os.path.expanduser("~/beam-conf/*.json"))):
    d = json.load(open(f))
    for m in d["models"]:
        v = m.get("methods", {}).get("bf_SPIR")
        if not v or not v.get("available"): continue
        for c, e in v["ladder"].get("by_condition", {}).items():
            print("%-12s %-15s %-7s %7d %+9.4f %+9.4f %+9.4f %7.1f" % (
                d["date"], m["model"], c, e["n"], e["sel_by_conf"],
                e["random_beam"], e["gain_over_random"], e["win_pct"]))
            k = (m["model"], c)
            agg[k][0] += e["n"]; agg[k][1] += e["n"]*e["gain_over_random"]; agg[k][2] += e["n"]*e["sel_by_conf"]
print("\n%-15s %-7s %8s %9s %9s" % ("model","cond","n","mean gain","mean sel"))
for (mo, c), (n, g, s) in sorted(agg.items()):
    print("%-15s %-7s %8d %+9.4f %+9.4f" % (mo, c, n, g/n, s/n))
