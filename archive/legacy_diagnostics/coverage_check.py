"""Every recording rendered on disk must appear in the embeddings.

The pairing protocol checks that a window carries all its streams. It cannot
see a recording that was never embedded at all, because such a recording
contributes no window. This closes that gap.
"""
import argparse, collections, json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import ANALYSIS_OUTPUT
from embedding_io import load_embeddings_from_dir
from embedding_schema import (audits_dir, bacpipe_embeddings_dir, bacpipe_meta_dir)
from permutation_null import METHODS

SUFFIX = re.compile(r"_(mono|sa|LabIR\(S\d\d_\d\d\d\)|SBF\(S\d\d_\d\d\d\)|SPIR[12]\([^)]*\))\.wav$")


def on_disk(location, date_str, method):
    """Distinct source recordings rendered for this method."""
    root = os.path.join(ANALYSIS_OUTPUT, location, date_str, method)
    if not os.path.isdir(root):
        return None
    found = set()
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            if f.endswith(".wav"):
                found.add(SUFFIX.sub("", f))
    return found


def in_embeddings(location, date_str, model):
    emb_dir = bacpipe_embeddings_dir(ANALYSIS_OUTPUT, location, model)
    _e, X, _y, flat, _n = load_embeddings_from_dir(
        emb_dir, methods=METHODS, date_filter=[date_str],
        source_tag="bacpipe:" + model,
        meta_dir=bacpipe_meta_dir(location, model))
    per_method = collections.defaultdict(set)
    for m in flat:
        wav = str(m.get("wav", ""))
        per_method[m.get("method")].add(SUFFIX.sub("", wav))
    return per_method, len(X)


def dates_for(location):
    root = os.path.join(ANALYSIS_OUTPUT, location)
    if not os.path.isdir(root):
        return []
    return sorted(d for d in os.listdir(root)
                  if re.fullmatch(r"\d{4}-\d{2}-\d{2}", d))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--locations", default="2A400,2D400")
    p.add_argument("--models", default="birdnet,avesecho_passt")
    p.add_argument("--out")
    a = p.parse_args()
    locations = [s.strip() for s in a.locations.split(",") if s.strip()]
    models = [s.strip() for s in a.models.split(",") if s.strip()]

    report = {}
    bad = 0
    for loc in locations:
        report[loc] = {}
        disk = {d: {m: on_disk(loc, d, m) for m in METHODS} for d in dates_for(loc)}
        for model in models:
            for date_str in sorted(disk):
                emb, n_rows = in_embeddings(loc, date_str, model)
                if n_rows == 0:
                    continue
                entry = {}
                for method in METHODS:
                    d = disk[date_str].get(method)
                    e = emb.get(method) or set()
                    if not d and not e:
                        continue
                    d = d or set()
                    only_disk = sorted(d - e)
                    only_emb = sorted(e - d)
                    entry[method] = {
                        "recordings_on_disk": len(d),
                        "recordings_in_embeddings": len(e),
                        "missing_from_embeddings": len(only_disk),
                        "examples_missing": only_disk[:3],
                        "in_embeddings_not_on_disk": len(only_emb),
                        "examples_extra": only_emb[:3],
                    }
                    if only_disk or only_emb:
                        bad += 1
                        print(f"MISMATCH {loc} {date_str} {model} {method}: "
                              f"disk {len(d)} emb {len(e)} "
                              f"missing {len(only_disk)} extra {len(only_emb)}")
                report[loc].setdefault(model, {})[date_str] = entry
    out = a.out or os.path.join(audits_dir(locations[0]), "coverage_check.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"mismatches": bad, "detail": report}, f, indent=2)
    print()
    print("mismatched method/date/model combinations:", bad)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
