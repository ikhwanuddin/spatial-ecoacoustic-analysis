#!/usr/bin/env python
"""Read-only integrity audit of the inputs and outputs of this pipeline.

Checks, per location and date:
  rendered audio   sa == mono, LabIR == mono * n_labir, SPIR == mono * n_spir
                   where n_labir / n_spir come from the beam tags actually on
                   disk, because dates rendered before the rep-2 decision hold
                   45 SPIR beams and later ones hold 31
  embeddings       every .npy row count matches the audio it came from
  metadata         one meta JSON per (date, method) in HOME, length == rows
  noise refs       each condition carries every group, plus a manifest
  checkpoints      no orphaned .ckpt shards left behind
  audits           every JSON parses
  source           every .py in the repo parses; .bak files are listed

Nothing is written or deleted.
"""
import ast, json, os, re, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

REPO = Path(os.path.expanduser("~/spatial-ecoacoustic-analysis"))
EPH = Path(os.environ.get("ANALYSIS_OUTPUT", "/rds/general/user/ri322/ephemeral/sea-work"))
HOME_RESULTS = Path(os.path.expanduser("~/sea-emb"))
DASH = Path(os.path.expanduser("~/sea-dashboards"))
JOBS = Path(os.path.expanduser("~/sea-jobs"))
METHODS = ("mono", "sa", "bf_LabIR", "bf_SPIR")

problems, notes = [], []
def bad(msg): problems.append(msg)
def note(msg): notes.append(msg)


def beam_tag(name, method):
    m = re.search(r"_(LabIR\([^)]*\)|SPIR\d?\([^)]*\))", name)
    return m.group(1) if m else None


def audio_inventory(loc, date):
    """wav counts and the distinct beam tags per method, straight from disk."""
    out = {}
    for meth in METHODS:
        d = EPH / loc / date / meth
        if not d.is_dir():
            out[meth] = (0, set())
            continue
        n, tags = 0, set()
        for p in d.rglob("*.wav"):
            if p.name.startswith("._"):
                continue
            n += 1
            t = beam_tag(p.name, meth)
            if t:
                tags.add(t)
        out[meth] = (n, tags)
    return out


def check_location(loc):
    root = EPH / loc
    if not root.is_dir():
        note(f"{loc}: tidak ada di ephemeral"); return
    dates = sorted(p.name for p in root.iterdir()
                   if p.is_dir() and re.fullmatch(r"\d{4}-\d{2}-\d{2}", p.name))
    print(f"\n=== {loc}: {len(dates)} tanggal ter-render ===")
    hdr = "{:12s} {:>6s} {:>6s} {:>7s} {:>7s} {:>4s} {:>4s}  {}".format(
        "date", "mono", "sa", "LabIR", "SPIR", "nL", "nS", "audio")
    print(hdr); print("-" * len(hdr))
    beams_by_date = {}
    for date in dates:
        inv = audio_inventory(loc, date)
        nm = inv["mono"][0]
        nL, nS = len(inv["bf_LabIR"][1]), len(inv["bf_SPIR"][1])
        beams_by_date[date] = (nm, nL, nS)
        flags = []
        if nm == 0:
            flags.append("mono kosong")
        else:
            if inv["sa"][0] != nm:
                flags.append(f"sa {inv['sa'][0]}!={nm}")
            if nL and inv["bf_LabIR"][0] != nm * nL:
                flags.append(f"LabIR {inv['bf_LabIR'][0]}!={nm}x{nL}")
            if nS and inv["bf_SPIR"][0] != nm * nS:
                flags.append(f"SPIR {inv['bf_SPIR'][0]}!={nm}x{nS}")
        state = "OK" if not flags else "; ".join(flags)
        if flags:
            bad(f"{loc}/{date} audio: {state}")
        print("{:12s} {:6d} {:6d} {:7d} {:7d} {:4d} {:4d}  {}".format(
            date, nm, inv["sa"][0], inv["bf_LabIR"][0], inv["bf_SPIR"][0], nL, nS, state))
    return beams_by_date


def check_embeddings(loc, beams_by_date):
    emb = EPH / loc / "emb"
    if not emb.is_dir():
        note(f"{loc}: tidak ada emb/"); return
    models = sorted(p.name for p in emb.iterdir() if p.is_dir() and not p.name.startswith("."))
    print(f"\n=== {loc}: embedding, {len(models)} model ===")
    for model in models:
        rows_ok, rows_bad, dates_seen = 0, [], set()
        for p in sorted((emb / model).glob("*_mono.npy")):
            date = p.name.replace("_mono.npy", "")
            dates_seen.add(date)
            try:
                nm = np.load(p, mmap_mode="r").shape[0]
            except Exception as e:
                rows_bad.append(f"{date} mono tidak terbaca: {e}"); continue
            nrec = beams_by_date.get(date)
            nL, nS = (nrec[1], nrec[2]) if nrec else (0, 0)
            for meth, mult in (("sa", 1), ("bf_LabIR", nL), ("bf_SPIR", nS)):
                f = emb / model / f"{date}_{meth}.npy"
                if not f.is_file():
                    rows_bad.append(f"{date} {meth} hilang"); continue
                if not mult:
                    continue
                got = np.load(f, mmap_mode="r").shape[0]
                if got == nm * mult:
                    rows_ok += 1
                else:
                    cov = 100.0 * (got // mult) / nm if nm else 0
                    rows_bad.append(
                        f"{date} {meth} {got} != {nm}x{mult}  cakupan {cov:.1f}%"
                        + ("  (sisa %d, array terpotong)" % (got % mult) if got % mult else ""))
        line = f"  {model:17s} {len(dates_seen):2d} tanggal, {rows_ok} cek baris OK"
        if rows_bad:
            line += f", {len(rows_bad)} MASALAH"
            for b in rows_bad:
                bad(f"{loc}/{model}: {b}")
        print(line)
        # orphaned checkpoints
        ck = emb / model / ".ckpt"
        if ck.is_dir():
            n = len(list(ck.glob("*")))
            if n:
                note(f"{loc}/{model}: {n} file .ckpt tersisa (normal kalau job masih jalan)")


def check_meta(loc):
    root = HOME_RESULTS / loc
    if not root.is_dir():
        note(f"{loc}: tidak ada di sea-emb"); return
    models = sorted(p.name for p in root.iterdir()
                    if p.is_dir() and p.name not in ("audits", "run_reports"))
    print(f"\n=== {loc}: metadata di HOME ===")
    for model in models:
        metas = sorted((root / model).glob("*_meta.json"))
        summaries = sorted((root / model).glob("*_summary.json"))
        mismatch = 0
        for m in metas:
            date_meth = m.name.replace("_meta.json", "")
            if date_meth.startswith("noise_"):
                # noise references pair with <name>_embeddings.npy in HOME,
                # not with a method .npy on ephemeral
                if not (root / model / f"{date_meth}_embeddings.npy").is_file():
                    bad(f"{loc}/{model}: {m.name} tanpa _embeddings.npy")
                continue
            npy = EPH / loc / "emb" / model / f"{date_meth}.npy"
            if not npy.is_file():
                bad(f"{loc}/{model}: meta {m.name} tanpa .npy"); continue
            try:
                with open(m) as f:
                    meta = json.load(f)
            except Exception as e:
                bad(f"{loc}/{model}: {m.name} tidak terparse: {e}"); continue
            n_npy = np.load(npy, mmap_mode="r").shape[0]
            n_meta = len(meta) if isinstance(meta, list) else meta.get("n", -1)
            if isinstance(meta, list) and n_meta != n_npy:
                mismatch += 1
                bad(f"{loc}/{model}: {m.name} {n_meta} entri != {n_npy} baris npy")
        print(f"  {model:17s} {len(metas):3d} meta, {len(summaries):2d} summary"
              + (f", {mismatch} MISMATCH" if mismatch else ""))


def check_noise_refs(loc, beams_by_date):
    print(f"\n=== {loc}: noise reference ===")
    root = EPH / loc
    found = False
    for date in sorted(beams_by_date):
        nr = root / date / "noise_references"
        if not nr.is_dir():
            continue
        found = True
        nL, nS = beams_by_date[date][1], beams_by_date[date][2]
        per = []
        for cond in ("dawn", "day", "dusk", "night"):
            d = nr / cond
            if not d.is_dir():
                continue
            counts = {g: len(list((d / g).glob("*.wav"))) for g in ("LabIR", "SPIR", "sa", "mono")
                      if (d / g).is_dir()}
            miss = [g for g in ("LabIR", "SPIR", "sa", "mono") if counts.get(g, 0) == 0]
            tag = f"{cond}({','.join(f'{k}={v}' for k, v in counts.items())})"
            if miss:
                tag += "!MISSING:" + ",".join(miss)
                bad(f"{loc}/{date} noise ref {cond}: group kosong {miss}")
            if not (d / "manifest.json").is_file() and not (nr / "manifest.json").is_file():
                note(f"{loc}/{date} noise ref {cond}: tanpa manifest")
            per.append(tag)
        print(f"  {date}  " + "  ".join(per) if per else f"  {date}  (kosong)")
    if not found:
        note(f"{loc}: tidak ada noise_references sama sekali")


def check_audits(loc):
    d = HOME_RESULTS / loc / "audits"
    if not d.is_dir():
        note(f"{loc}: tidak ada audits/"); return
    files = sorted(d.glob("*.json"))
    kinds = defaultdict(int)
    for f in files:
        try:
            json.load(open(f))
        except Exception as e:
            bad(f"{loc}/audits: {f.name} tidak terparse: {e}"); continue
        kinds[re.sub(r"^\d{4}-\d{2}-\d{2}_", "", f.name)] = kinds[re.sub(r"^\d{4}-\d{2}-\d{2}_", "", f.name)] + 1
    print(f"\n=== {loc}: audits, {len(files)} JSON, semua terparse kecuali yang dilaporkan ===")
    agg = defaultdict(int)
    for k, v in kinds.items():
        agg[re.sub(r"_[a-z0-9_]+\.json$", "", k)] += v
    for k in sorted(agg):
        print(f"  {k:28s} {agg[k]}")


def check_source():
    print("\n=== repo: source ===")
    py = sorted(REPO.glob("*.py")) + sorted((REPO / "bacpipe").glob("*.py"))
    nbad = 0
    for p in py:
        try:
            ast.parse(open(p, encoding="utf-8").read())
        except SyntaxError as e:
            nbad += 1; bad(f"repo: {p.name} SyntaxError baris {e.lineno}")
    print(f"  {len(py)} file .py, {len(py)-nbad} parse OK")
    baks = sorted(REPO.rglob("*.bak*"))
    if baks:
        print(f"  {len(baks)} backup:")
        for b in baks:
            print(f"    {b.relative_to(REPO)}  {b.stat().st_size} B")


def check_dirs():
    print("\n=== lokasi output ===")
    for name, p in (("sea-emb", HOME_RESULTS), ("sea-dashboards", DASH), ("sea-jobs", JOBS)):
        if p.is_dir():
            n = sum(1 for _ in p.rglob("*") if _.is_file())
            print(f"  {name:16s} {n:5d} file")
        else:
            bad(f"{name} tidak ada")
    stray = [q.name for q in Path(os.path.expanduser("~")).iterdir()
             if q.is_dir() and not q.name.startswith(".")
             and q.name not in {"sea-emb", "sea-dashboards", "sea-jobs",
                                "spatial-ecoacoustic-analysis", "MAARU-Impulse-Response",
                                "ondemand", "R", "miniforge3", "sea-data"}]
    if stray:
        note("folder lain di HOME (bukan lokasi output resmi): " + ", ".join(sorted(stray)))


if __name__ == "__main__":
    locs = sys.argv[1:] or ["2A400", "2D400"]
    for loc in locs:
        b = check_location(loc)
        if b:
            check_embeddings(loc, b)
            check_meta(loc)
            check_noise_refs(loc, b)
            check_audits(loc)
    check_source()
    check_dirs()

    print("\n" + "=" * 70)
    if problems:
        print(f"MASALAH: {len(problems)}")
        for p in problems:
            print("  ! " + p)
    else:
        print("MASALAH: tidak ada")
    if notes:
        print(f"\nCATATAN: {len(notes)}")
        for p in notes:
            print("  - " + p)
