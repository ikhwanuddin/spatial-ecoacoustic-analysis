"""Roll the per-date detection counts up to condition, day and month tables.

Reads the JSONs `detection_counts.py` writes, one per (location, date, model),
and aggregates them. Nothing is recomputed here: a recap that re-derived its own
numbers could disagree with the per-date files, and then neither would be
trustworthy.

Window counts differ by more than a factor of 300 across these dates (49 windows
on 2026-04-26, 19070 on 2026-05-04), so a raw detection count is not comparable
between rows. Every table therefore carries the rate per 1000 windows alongside
the count, and the count is what gets summed.

Beam counts are read from the files, never assumed: dates rendered before the
SPIR2 rep-2 decision hold 45 SPIR beams and later ones 31, so a row spanning both
reports the range. A wrong denominator has already produced a wrong verdict once.
"""
import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_schema import audits_dir, CONDITIONS

DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


def zero():
    return {"windows": 0, "recordings": 0, "beams": set(),
            "plain": collections.Counter(), "split": collections.Counter(),
            "dir_plain": collections.Counter(), "dir_split": collections.Counter()}


def add(dst, entry, recordings=0):
    dst["windows"] += entry["windows"]
    dst["recordings"] += recordings
    dst["beams"].add(entry["beams"])
    for t, n in entry["detections_plain"].items():
        dst["plain"][t] += n
    for t, n in entry["detections_split"].items():
        dst["split"][t] += n
    for d in entry.get("top_directions_plain", []):
        dst["dir_plain"][d["beam"]] += d["detections"]
    for d in entry.get("top_directions_split", []):
        dst["dir_split"][d["beam"]] += d["detections"]


def beams_label(beams):
    beams = sorted(b for b in beams if b)
    if not beams:
        return "-"
    return str(beams[0]) if len(beams) == 1 else f"{beams[0]}-{beams[-1]}"


def rate(n, windows):
    return 0.0 if not windows else 1000.0 * n / windows


def load(location, model, dates):
    """Per-date reports, newest key order preserved by sorting on the date."""
    pattern = os.path.join(audits_dir(location),
                           f"*_detection_counts_{model}.json")
    out = {}
    for path in sorted(glob.glob(pattern)):
        m = DATE_RE.search(os.path.basename(path))
        if not m:
            continue
        date_str = m.group(0)
        if dates and date_str not in dates:
            continue
        with open(path, encoding="utf-8") as f:
            out[date_str] = json.load(f)
    return out


def table(rows, methods, thresholds, rule, title, first_col):
    """One markdown table: rows x methods, count and rate per 1000 windows."""
    lines = [f"### {title}", ""]
    head = [f"| {first_col} | method | beams | recordings | windows |"]
    for t in thresholds:
        head.append(f" {t} | per 1k |")
    lines.append("".join(head))
    sep = ["|---|---|---:|---:|---:|"] + ["---:|---:|"] * len(thresholds)
    lines.append("".join(sep))
    for label, node in rows:
        for m in methods:
            s = node.get(m)
            if not s or not s["windows"]:
                continue
            cells = [f"| {label} | `{m}` | {beams_label(s['beams'])} | "
                     f"{s['recordings'] or '-'} | {s['windows']} |"]
            for t in thresholds:
                n = s[rule][str(t)]
                cells.append(f" {n} | {rate(n, s['windows']):.1f} |")
            lines.append("".join(cells))
    lines.append("")
    return lines


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--location", required=True)
    p.add_argument("--model", default="birdnet")
    p.add_argument("--dates", default="", help="Comma separated; default every date found")
    p.add_argument("--rule", default="plain", choices=("plain", "split"))
    p.add_argument("--top", type=int, default=8, help="Directions listed per method")
    p.add_argument("--out")
    a = p.parse_args()
    dates = {s.strip() for s in a.dates.split(",") if s.strip()}

    reports = load(a.location, a.model, dates)
    if not reports:
        print(f"no detection_counts JSON for {a.location} / {a.model} "
              f"in {audits_dir(a.location)}")
        return 1

    thresholds, methods = [], []
    for r in reports.values():
        for t in r["thresholds"]:
            if t not in thresholds:
                thresholds.append(t)
        for m in r["daily_total"]:
            if m not in methods:
                methods.append(m)
    thresholds.sort()

    per_day = {}
    per_day_cond = collections.defaultdict(dict)
    per_cond = {c: {m: zero() for m in methods} for c in CONDITIONS}
    per_month = collections.defaultdict(lambda: {m: zero() for m in methods})
    overall = {m: zero() for m in methods}

    for date_str, r in sorted(reports.items()):
        month = date_str[:7]
        n_rec = r.get("recordings", 0)
        day = {m: zero() for m in methods}
        for m, e in r["daily_total"].items():
            add(day[m], e, recordings=n_rec)
            add(per_month[month][m], e, recordings=n_rec)
            add(overall[m], e, recordings=n_rec)
        per_day[date_str] = day
        for cond, node in r.get("per_condition", {}).items():
            slot = per_day_cond[date_str].setdefault(
                cond, {m: zero() for m in methods})
            for m, e in node.items():
                add(slot[m], e)
                if cond in per_cond:
                    add(per_cond[cond][m], e)

    md = [f"# Detection counts recap — {a.location} / {a.model}", "",
          f"Selection rule: **{a.rule}**. "
          f"{len(reports)} dates: {', '.join(sorted(reports))}.", "",
          "Counts are summed; the per-1k column is the rate per 1000 paired "
          "windows, which is the only column comparable down a table, because "
          "window counts differ by more than 300x between these dates.", ""]

    md += table([(d, per_day[d]) for d in sorted(per_day)],
                methods, thresholds, a.rule, "Per day", "date")
    md += table([(c, per_cond[c]) for c in CONDITIONS
                 if any(per_cond[c][m]["windows"] for m in methods)],
                methods, thresholds, a.rule,
                "Per time-of-day condition, all dates", "condition")
    md += table([(k, per_month[k]) for k in sorted(per_month)],
                methods, thresholds, a.rule, "Per month", "month")
    md += table([("all", overall)], methods, thresholds, a.rule,
                "Everything", "scope")

    md += [f"### Winning directions, rule {a.rule}, counted at threshold "
           f"{thresholds[0]}", ""]
    key = "dir_" + a.rule
    for m in methods:
        counts = overall[m][key]
        if not counts:
            continue
        total = sum(counts.values())
        top = counts.most_common(a.top)
        md.append(f"**`{m}`** — {len(counts)} distinct beams won at least once, "
                  f"{total} detections total")
        md.append("")
        md.append("| beam | detections | share |")
        md.append("|---|---:|---:|")
        for b, n in top:
            md.append(f"| `{b}` | {n} | {100.0*n/total:.1f}% |")
        md.append("")
    md += ["A concentrated list means the same beams keep winning across the "
           "whole month, which is what a fixed source looks like, not a bird. "
           "A flat list is what an uninformative beam bank looks like. Neither "
           "reading is evidence on its own — compare against the false-alarm "
           "floor on bird-free windows.", ""]

    out_md = a.out or os.path.join(
        audits_dir(a.location), f"detection_recap_{a.model}_{a.rule}.md")
    out_json = os.path.splitext(out_md)[0] + ".json"

    def dump(node):
        return {m: {"windows": s["windows"], "recordings": s["recordings"],
                    "beams": sorted(s["beams"]),
                    "detections": dict(s[a.rule]),
                    "per_1000_windows": {t: round(rate(s[a.rule][t], s["windows"]), 3)
                                         for t in s[a.rule]},
                    "winning_directions": dict(s[key].most_common())}
                for m, s in node.items() if s["windows"]}

    payload = {
        "location": a.location, "model": a.model, "rule": a.rule,
        "thresholds": thresholds, "dates": sorted(reports),
        "per_day": {d: dump(per_day[d]) for d in sorted(per_day)},
        "per_day_condition": {d: {c: dump(n) for c, n in sorted(cs.items())}
                              for d, cs in sorted(per_day_cond.items())},
        "per_condition": {c: dump(per_cond[c]) for c in CONDITIONS
                          if any(per_cond[c][m]["windows"] for m in methods)},
        "per_month": {k: dump(per_month[k]) for k in sorted(per_month)},
        "overall": dump(overall),
    }

    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print("\n".join(md[:4]))
    for m in methods:
        s = overall[m]
        if not s["windows"]:
            continue
        t0 = str(thresholds[0])
        print(f"  {m:10s} beams {beams_label(s['beams']):>7s}  "
              f"{s['windows']:8d} windows  {s[a.rule][t0]:8d} detections at {t0}  "
              f"({rate(s[a.rule][t0], s['windows']):.1f} per 1k)")
    print("wrote", out_md)
    print("wrote", out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
