"""Every recording must carry every beam the current config allows.

The first version of this script flagged any recording whose beam count differed
from the modal count of its date, and then failed 2026-05-18 for exactly the
reason the protocol says to expect: 43 recordings were rendered before the SPIR2
rep-2 decision and hold 45 SPIR beams, while the 68 rendered afterwards hold 31.
Both are correct. `expected_beam_tags()` allows only the 31, so both groups reduce
to the same 31 downstream.

So the test is not "same count as the neighbours" but "carries every allowed
tag". Extra tags are fine — the beam filter drops them. Missing ones are not.

usage: verify_beams.py <location> <date>[,<date>...]
"""
import collections
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import expected_beam_tags

ROOT = os.environ.get("ANALYSIS_OUTPUT",
                      "/rds/general/user/ri322/ephemeral/sea-work")
TAG = re.compile(r"_((?:LabIR|SBF|SPIR1|SPIR2)\([^)]*\))\.wav$")
PLAIN = re.compile(r"_(mono|sa)\.wav$")
PREFIX = {"bf_LabIR": ("LabIR(",), "bf_SBF": ("SBF(",),
          "bf_SPIR": ("SPIR1(", "SPIR2(")}
METHODS = ["mono", "sa", "bf_LabIR", "bf_SPIR", "bf_SBF"]


def scan(method_dir, method):
    """{recording: {beam tags}} for a beamformed method, or {recording: set()}."""
    per = collections.defaultdict(set)
    for dirpath, _dirs, files in os.walk(method_dir):
        for name in files:
            if not name.endswith(".wav") or name.startswith("._"):
                continue
            m = TAG.search(name) or PLAIN.search(name)
            if not m:
                continue
            per[name[: m.start()]].add(m.group(1))
    return per


def main():
    location, dates = sys.argv[1], sys.argv[2].split(",")
    allowed = expected_beam_tags()
    bad = 0
    for date in dates:
        print("=" * 62)
        print(location, date)
        n_source = None
        for method in METHODS:
            per = scan(os.path.join(ROOT, location, date, method), method)
            if not per:
                print(f"  {method:9s} nothing on disk")
                continue
            want = {t for t in allowed if t.startswith(PREFIX[method])} \
                if method in PREFIX else set()
            hist = collections.Counter(len(v) for v in per.values())
            print(f"  {method:9s} recordings={len(per):4d} "
                  f"tags_per_recording={dict(sorted(hist.items()))}"
                  + (f"  required={len(want)}" if want else ""))
            if method == "mono":
                n_source = len(per)
            elif n_source is not None and len(per) != n_source:
                print(f"      SHORT: {len(per)} recordings against mono's {n_source}")
                bad += 1
            if want:
                short = sorted(r for r, got in per.items() if want - got)
                if short:
                    ex = short[0]
                    print(f"      MISSING TAGS in {len(short)} recordings, "
                          f"e.g. {ex} lacks {sorted(want - per[ex])[:4]}")
                    bad += 1
        print("=" * 62)
    print("VERIFY FAIL" if bad else "VERIFY OK")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
