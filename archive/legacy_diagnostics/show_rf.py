import json, glob, os, sys
for f in sorted(glob.glob(os.path.expanduser("~/bc_two/*.json"))):
    d = json.load(open(f))
    for m in d["models"]:
        for meth, v in m.get("methods", {}).items():
            if not v.get("available"): continue
            L = v["ladder"]
            print("==", d["date"], m["model"], meth)
            for k in sorted(L):
                if k in ("lag_sweep", "by_condition", "criterion_agreement", "class_split_agreement"):
                    continue
                print("   %-32s %+.4f  win %.1f%%" % (k, L[k]["mean_lift"], L[k]["win_pct"]))
            ls = L.get("lag_sweep", {})
            e = ls.get("1", {})
            print("   -- lag-1 gain over random --")
            for k, val in e.items():
                if isinstance(val, dict) and "gain_over_random" in val:
                    print("   %-32s %+.4f  (lift %+.4f, win %.1f%%)" % (k, val["gain_over_random"], val["mean_lift"], val["win_pct"]))
            oc = ls.get("_other_file_control", {})
            print("   -- other-file control --")
            for k, val in oc.items():
                print("   %-32s %+.4f" % (k, val["gain_over_random"]))
