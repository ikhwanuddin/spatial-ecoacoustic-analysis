import json,glob,os,sys,statistics
date=sys.argv[1]
root=f"/rds/general/user/ri322/home/sea-dashboards/2A400/{date}"
rows=[]
for f in sorted(glob.glob(os.path.join(root,"*.html"))):
    if os.path.basename(f)=="index.html": continue
    h=open(f,encoding="utf-8",errors="replace").read()
    i=h.find("const report = ")
    if i<0: continue
    rep,_=json.JSONDecoder().raw_decode(h[i+len("const report = "):])
    sc=rep["stats"].get("all_beam_analysis",{}).get("selection_comparison")
    if not sc: continue
    m=rep["model"]
    for meth in ("bf_LabIR","bf_SPIR","sa"):
        d=sc.get(meth)
        if not d: continue
        rows.append((m,meth,d["best_beam"]["mean_delta_vs_mono"],d["best_beam"]["win_rate_pct"],
                     d["median_beam"]["mean_delta_vs_mono"],d["median_beam"]["win_rate_pct"],
                     d["selection_effect"]))
print(f"### {date}  n_models={len(set(r[0] for r in rows))}")
print("%-20s%-10s%9s%7s%10s%7s%9s" % ("model","method","best","win%","median","win%","seleff"))
for r in rows:
    print(f"{r[0]:<20}{r[1]:<10}{r[2]:>+9.4f}{r[3]:>7.1f}{r[4]:>+10.4f}{r[5]:>7.1f}{r[6]:>9.4f}")
print()
for meth in ("bf_LabIR","bf_SPIR","sa"):
    b=[r[2] for r in rows if r[1]==meth]; md=[r[4] for r in rows if r[1]==meth]; se=[r[6] for r in rows if r[1]==meth]
    neg=sum(1 for x in md if x<0)
    print(f"{meth:<10} mean best {statistics.mean(b):+.4f} | mean median {statistics.mean(md):+.4f} | median<0 in {neg}/{len(md)} | mean sel.effect {statistics.mean(se):.4f}")
