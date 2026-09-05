"""Tiered detection counts per method.

Pairs every window across methods, counts how many windows cross a confidence
threshold under each method, and rolls the counts up per recording, then per
time-of-day class, then per day. For every steered method it also records which
direction won each detection.

The threshold is a parameter, not a constant: 0.4 is the Forum Acusticum value,
0.65 was used in the most recent experiment at this site, and BirdNET's authors
state that no single threshold transfers across locations and recording quality.

Two selection rules are reported side by side:

  plain   the beam with the highest confidence, scored by that same confidence
  split   the beam chosen by one disjoint half of the classes, scored by the
          other half, so the reported number was never used to choose
"""
import argparse, collections, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ANALYSIS_OUTPUT
from embedding_io import load_embeddings_from_dir
from embedding_schema import (audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir,
                              condition_from_wav, CONDITIONS)
from beam_confidence import load_head, confidence
from permutation_null import METHODS, BF_METHODS, build_windows, IncompletePairing
from coverage_check import on_disk

DEFAULT_THRESHOLDS = (0.4, 0.5, 0.65)


def expected_beams(windows, method):
    """Every beam tag this method shows anywhere on this date."""
    seen = set()
    for v in windows.values():
        seen |= set(v.get(method, {}))
    return seen


def check_pairing(windows, methods, expect):
    """Refuse to count anything unless every window carries every stream.

    The protocol requires complete pairs. Missing data is completed, never
    dropped and never forced, so this raises instead of filtering.
    """
    missing = []
    for key, v in windows.items():
        for m in methods:
            got = set(v.get(m, {}))
            gaps = expect[m] - got
            if gaps:
                missing.append((key, m, sorted(gaps)[:4], len(gaps)))
    if missing:
        lines = [f"{len(missing)} window/method pairs are incomplete, refusing to count."]
        per_method = collections.Counter(m for _, m, _, _ in missing)
        for m, n in per_method.most_common():
            lines.append(f"  {m}: {n} windows short of {len(expect[m])} beams")
        for key, m, gaps, n in missing[:5]:
            lines.append(f"  e.g. {key[0]} {key[1]:.1f}-{key[2]:.1f}s {m} missing {n}: {gaps}")
        lines.append("Complete the missing renders or embeddings, then re-run.")
        raise IncompletePairing("\n".join(lines))


def check_coverage(location, date_str, windows, methods):
    """Every recording rendered on disk must have contributed windows.

    check_pairing cannot see this failure. A recording that was never embedded
    contributes no window at all, so every window that does exist looks perfectly
    paired while the counts silently rest on fewer recordings. Four model/date
    pairs sat between 11 and 77 percent coverage this way.
    """
    seen = collections.defaultdict(set)
    for (src, _s, _e), v in windows.items():
        for m in methods:
            if v.get(m):
                seen[m].add(src)
    short = []
    for m in methods:
        disk = on_disk(location, date_str, m)
        if not disk:
            continue
        gaps = sorted(disk - seen[m])
        if gaps:
            short.append((m, len(disk), len(seen[m]), gaps))
    if short:
        lines = ["recordings rendered on disk are missing from the embeddings, "
                 "refusing to count."]
        for m, n_disk, n_seen, gaps in short:
            lines.append(f"  {m}: {n_seen} of {n_disk} recordings embedded, "
                         f"{len(gaps)} missing")
            lines.append(f"    e.g. {gaps[:3]}")
        lines.append("Embed the missing recordings, then re-run.")
        raise IncompletePairing("\n".join(lines))


def score_window(v, method, Cw, Aw, Bw):
    """Reported score and winning direction, under both selection rules."""
    beams = sorted(v.get(method, {}))
    c = np.array([Cw[method][b] for b in beams])
    j = int(c.argmax())
    plain = (float(c[j]), beams[j])
    a = np.array([Aw[method][b] for b in beams])
    k = int(a.argmax())
    split = (float(Bw[method][beams[k]]), beams[k])
    return plain, split, len(beams)


def blank(methods, thresholds):
    return {m: {"windows": 0, "beams": 0,
                "plain": {str(t): 0 for t in thresholds},
                "split": {str(t): 0 for t in thresholds},
                "winners_plain": collections.Counter(),
                "winners_split": collections.Counter()}
            for m in methods}


def fold(dst, src):
    for m, s in src.items():
        d = dst[m]
        d["windows"] += s["windows"]
        d["beams"] = max(d["beams"], s["beams"])
        for rule in ("plain", "split"):
            for t, n in s[rule].items():
                d[rule][t] += n
        d["winners_plain"].update(s["winners_plain"])
        d["winners_split"].update(s["winners_split"])


def finish(node, top=0):
    """top=0 keeps every winning direction; the recap sums these, so a truncated
    list would make the monthly direction table a lower bound."""
    out = {}
    for m, s in node.items():
        e = {"windows": s["windows"], "beams": s["beams"],
             "detections_plain": s["plain"], "detections_split": s["split"]}
        if m in BF_METHODS:
            e["top_directions_plain"] = [
                {"beam": b, "detections": n} for b, n in s["winners_plain"].most_common(top or None)]
            e["top_directions_split"] = [
                {"beam": b, "detections": n} for b, n in s["winners_split"].most_common(top or None)]
        out[m] = e
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--location", required=True)
    p.add_argument("--date", required=True)
    p.add_argument("--model", default="birdnet")
    p.add_argument("--thresholds", default=",".join(str(t) for t in DEFAULT_THRESHOLDS))
    p.add_argument("--repo", default=os.path.dirname(os.path.abspath(__file__)))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    a = p.parse_args()
    thresholds = [float(x) for x in a.thresholds.split(",") if x.strip()]

    emb_dir = bacpipe_embeddings_dir(ANALYSIS_OUTPUT, a.location, a.model)
    _e, X, _y, flat, _n = load_embeddings_from_dir(
        emb_dir, methods=METHODS, date_filter=[a.date],
        source_tag="bacpipe:" + a.model,
        meta_dir=bacpipe_meta_dir(a.location, a.model))
    if len(X) == 0:
        print("no embeddings for", a.location, a.date, a.model)
        return 1
    C, _top, cA, cB = confidence(X, load_head(a.repo, a.model), a.seed)

    wins, order = build_windows(flat, C, METHODS)
    winsA, _ = build_windows(flat, cA, METHODS)
    winsB, _ = build_windows(flat, cB, METHODS)

    present = [m for m in METHODS if any(v.get(m) for v in wins.values())]
    expect = {m: expected_beams(wins, m) for m in present}
    print("windows", len(wins), "| methods", ", ".join(
        f"{m} x{len(expect[m])}" for m in present))
    check_coverage(a.location, a.date, wins, present)
    check_pairing(wins, present, expect)

    per_file = {}
    per_cond = {c: blank(present, thresholds) for c in CONDITIONS}
    total = blank(present, thresholds)

    for key in sorted(wins, key=lambda k: order[k]):
        src = key[0]
        v = wins[key]
        node = per_file.setdefault(src, blank(present, thresholds))
        for m in present:
            (pc, pb), (sc, sb), nb = score_window(v, m, v, winsA[key], winsB[key])
            d = node[m]
            d["windows"] += 1
            d["beams"] = nb
            for t in thresholds:
                if pc >= t:
                    d["plain"][str(t)] += 1
                    if t == thresholds[0]:
                        d["winners_plain"][pb] += 1
                if sc >= t:
                    d["split"][str(t)] += 1
                    if t == thresholds[0]:
                        d["winners_split"][sb] += 1

    for src, node in per_file.items():
        cond = condition_from_wav(src) or "unknown"
        if cond in per_cond:
            fold(per_cond[cond], node)
        fold(total, node)

    report = {
        "location": a.location, "date": a.date, "model": a.model,
        "thresholds": thresholds,
        "selection_rules": {
            "plain": "beam with the highest confidence, scored by that same confidence",
            "split": "beam chosen by one half of the classes, scored by the other half",
        },
        "winning_directions_counted_at": thresholds[0],
        "recordings": len(per_file),
        "per_recording": {src: finish(node) for src, node in sorted(per_file.items())},
        "per_condition": {c: finish(per_cond[c]) for c in CONDITIONS
                          if total and per_cond[c][present[0]]["windows"]},
        "daily_total": finish(total),
    }
    out = a.out or os.path.join(audits_dir(a.location),
                                a.date + "_detection_counts_" + a.model + ".json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)

    print()
    print("daily total, threshold", thresholds[0], "  (plain / split)")
    for m in present:
        e = report["daily_total"][m]
        t0 = str(thresholds[0])
        print(f"  {m:10s} x{e['beams']:<3d} {e['windows']:7d} windows  "
              f"{e['detections_plain'][t0]:7d} / {e['detections_split'][t0]:7d}")
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
