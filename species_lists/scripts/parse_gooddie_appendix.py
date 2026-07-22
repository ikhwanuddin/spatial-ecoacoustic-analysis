#!/usr/bin/env python3
"""
Parse Gooddie (2015) Forktail appendix using PDF word coordinates.

Appendix is a dual-column table. Site columns (WC/S, WT/L/K, DR) are detected
from each page's header row, then each half-row is parsed for binomials + x marks.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import pdfplumber

SCI_RE = re.compile(r"^[A-Z][a-z]+$")
EPITHET_RE = re.compile(r"^[a-z]+$")
IUCN_RE = re.compile(r"^\((CR|EN|VU|NT)\)$")


def line_key(top: float) -> float:
    return round(top / 2.0) * 2.0


def detect_headers(lines: dict[float, list], page_width: float) -> list[dict]:
    """Return list of column geometries: {wc, wt, dr, x_min, x_max} for each table half."""
    headers = []
    for y, ws in lines.items():
        texts = [w["text"] for w in ws]
        if texts.count("WC/S") < 1 or texts.count("DR") < 1:
            continue
        # collect WC/S, WT/L/K, DR word positions
        marks = {"WC/S": [], "WT/L/K": [], "DR": []}
        for w in ws:
            if w["text"] in marks:
                marks[w["text"]].append(w["x0"])
        # pair into left/right by sorting x
        all_wcs = sorted(marks["WC/S"])
        all_wts = sorted(marks["WT/L/K"])
        all_drs = sorted(marks["DR"])
        n = min(len(all_wcs), len(all_wts), len(all_drs))
        for i in range(n):
            wc, wt, dr = all_wcs[i], all_wts[i], all_drs[i]
            # half bounds: from previous mid to next mid
            if i == 0:
                x_min = 0
                x_max = (wc + (all_wcs[1] if n > 1 else page_width)) / 2 if n > 1 else page_width / 2 + 20
                # better: table occupies from left margin to mid
                x_max = page_width / 2 + 5
            else:
                x_min = page_width / 2 - 5
                x_max = page_width + 10
            headers.append({"wc": wc, "wt": wt, "dr": dr, "x_min": x_min, "x_max": x_max})
        if headers:
            break
    return headers


def classify_mark(x0: float, header: dict) -> str | None:
    d = {
        "wc": abs(x0 - header["wc"]),
        "wt": abs(x0 - header["wt"]),
        "dr": abs(x0 - header["dr"]),
    }
    col = min(d, key=d.get)
    if d[col] > 16:
        return None
    return col


def parse_half_line(words: list[dict], header: dict) -> dict | None:
    if not words or not header:
        return None

    marks = {"wc": False, "wt": False, "dr": False}
    content = []
    for w in words:
        if w["text"].lower() == "x" and len(w["text"]) == 1:
            col = classify_mark(w["x0"], header)
            if col:
                marks[col] = True
            continue
        content.append(w)

    if not content:
        return None

    texts = [w["text"] for w in content]
    iucn = ""
    if texts and IUCN_RE.match(texts[-1]):
        iucn = texts[-1].strip("()")
        texts = texts[:-1]
    if not texts:
        return None

    genus_idx = None
    for i in range(len(texts) - 1):
        if SCI_RE.match(texts[i]) and EPITHET_RE.match(texts[i + 1]):
            genus_idx = i
    if genus_idx is None:
        return None

    genus = texts[genus_idx]
    epithet = texts[genus_idx + 1]
    common = " ".join(texts[:genus_idx]).strip()
    common = re.sub(r"\s+", " ", common)
    if len(common) < 3 or common.lower() in {"species", "sites"}:
        return None

    for tok in texts[genus_idx + 2 :]:
        if IUCN_RE.match(tok):
            iucn = tok.strip("()")

    return {
        "common_name_en": common,
        "scientific_name_original": f"{genus} {epithet}",
        "iucn": iucn,
        "at_way_canguk": marks["wc"],
        "at_way_titias": marks["wt"],
        "at_danau_ranau": marks["dr"],
        "sources": ["gooddie2015"],
    }


def parse_pdf(pdf_path: Path) -> list[dict]:
    records: dict[str, dict] = {}
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page_idx in range(8, min(12, len(pdf.pages))):
            page = pdf.pages[page_idx]
            words = page.extract_words(x_tolerance=1.5, y_tolerance=2.5)
            lines: dict[float, list] = defaultdict(list)
            for w in words:
                lines[line_key(w["top"])].append(w)

            headers = detect_headers(lines, page.width)
            if not headers:
                print(f"  page {page_idx+1}: no headers found", file=sys.stderr)
                continue
            print(
                f"  page {page_idx+1}: {len(headers)} table halves "
                f"{[(round(h['wc']), round(h['wt']), round(h['dr'])) for h in headers]}",
                file=sys.stderr,
            )

            for y, ws in sorted(lines.items()):
                ws = sorted(ws, key=lambda w: w["x0"])
                texts = [w["text"] for w in ws]
                joined = " ".join(texts)
                if "WC/S" in texts and "Species" in texts:
                    continue
                if joined.startswith("Sites") or "CHRIS" in joined or "Forktail" in joined:
                    continue
                if "Appendix" in joined or "Site initials" in joined:
                    continue
                if "REFERENCES" in joined or "ACKNOWLEDGEMENTS" in joined:
                    continue

                for header in headers:
                    half = [
                        w
                        for w in ws
                        if header["x_min"] <= w["x0"] < header["x_max"]
                    ]
                    rec = parse_half_line(half, header)
                    if not rec:
                        continue
                    key = rec["scientific_name_original"].lower()
                    if key in records:
                        prev = records[key]
                        prev["at_way_canguk"] |= rec["at_way_canguk"]
                        prev["at_way_titias"] |= rec["at_way_titias"]
                        prev["at_danau_ranau"] |= rec["at_danau_ranau"]
                        if rec["iucn"] and not prev["iucn"]:
                            prev["iucn"] = rec["iucn"]
                        if rec["common_name_en"] and (
                            not prev["common_name_en"]
                            or len(rec["common_name_en"]) > len(prev["common_name_en"])
                        ):
                            prev["common_name_en"] = rec["common_name_en"]
                    else:
                        records[key] = rec

    return sorted(records.values(), key=lambda r: r["scientific_name_original"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pdf",
        type=Path,
        default=Path(
            "/Users/ri322/.grok/sessions/%2FUsers%2Fri322%2Fmacmini/"
            "019f8706-486d-73f1-b983-2f2bd092a020/downloads/1.pdf"
        ),
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.pdf.exists():
        print(f"PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    print(f"Parsing {args.pdf}", file=sys.stderr)
    rows = parse_pdf(args.pdf)
    n_wc = sum(1 for r in rows if r["at_way_canguk"])
    n_wt = sum(1 for r in rows if r["at_way_titias"])
    n_dr = sum(1 for r in rows if r["at_danau_ranau"])
    print(f"Parsed {len(rows)} species | WC={n_wc} WT={n_wt} DR={n_dr}", file=sys.stderr)

    checks = {
        "Argusianus argus": (True, False, False),
        "Rhizothera longirostris": (False, True, False),
        "Carpococcyx viridis": (False, True, False),
        "Pitta schneideri": (False, False, True),
        "Pitta granatina": (True, False, False),
        "Pitta venusta": (False, True, True),
        "Garrulax bicolor": (False, False, True),
        "Rhinoplax vigil": (True, True, True),
        "Blythipicus rubiginosus": (True, True, True),
        "Rollulus rouloul": (True, True, False),
        "Meiglyptes grammithorax": (True, False, True),
        "Synoicus chinensis": (True, False, False),
    }
    by = {r["scientific_name_original"]: r for r in rows}
    ok_n = 0
    for sci, exp in checks.items():
        r = by.get(sci)
        if not r:
            print(f"  MISSING {sci}", file=sys.stderr)
            continue
        got = (r["at_way_canguk"], r["at_way_titias"], r["at_danau_ranau"])
        ok = got == exp
        ok_n += int(ok)
        print(f"  {'OK' if ok else 'FAIL'} {sci}: {got} expected {exp}", file=sys.stderr)
    print(f"Spot-check {ok_n}/{len(checks)}", file=sys.stderr)

    payload = json.dumps(rows, indent=2, ensure_ascii=False)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload + "\n", encoding="utf-8")
        print(f"Wrote {args.out}", file=sys.stderr)
    else:
        print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
