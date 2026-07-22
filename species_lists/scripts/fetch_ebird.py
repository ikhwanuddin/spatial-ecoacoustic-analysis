#!/usr/bin/env python3
"""
Fetch eBird species lists for regions/hotspots near TNBBS / Way Canguk.

Requires: export EBIRD_API_KEY=...

Usage:
  python fetch_ebird.py
  python fetch_ebird.py --region ID-SM-LA
  python fetch_ebird.py --lat -5.689 --lon 104.409 --dist 50
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
OUT = ROOT / "data" / "ebird"


def load_dotenv_files() -> None:
    for path in (PROJECT / ".env", ROOT / ".env", Path.home() / ".config" / "ebird" / "api_key"):
        if not path.exists():
            continue
        if path.name == "api_key":
            key = path.read_text(encoding="utf-8").strip()
            if key and "EBIRD_API_KEY" not in os.environ:
                os.environ["EBIRD_API_KEY"] = key
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip("'").strip('"')
            if k and k not in os.environ:
                os.environ[k] = v


def api_get(path: str, key: str, params: dict | None = None):
    q = f"?{urllib.parse.urlencode(params)}" if params else ""
    url = f"https://api.ebird.org/v2/{path}{q}"
    req = urllib.request.Request(url, headers={"X-eBirdApiToken": key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--region", default="ID-SM-LA", help="eBird region code")
    ap.add_argument("--lat", type=float, default=-5.689)
    ap.add_argument("--lon", type=float, default=104.409)
    ap.add_argument("--dist", type=int, default=50, help="km for nearby hotspots")
    args = ap.parse_args()

    load_dotenv_files()
    key = os.environ.get("EBIRD_API_KEY", "").strip()
    if not key:
        print(
            "Set EBIRD_API_KEY without pasting it in chat:\n"
            "  1) cp .env.example .env   (from spatial-ecoacoustic-analysis/)\n"
            "  2) edit .env → EBIRD_API_KEY=your_key\n"
            "  3) re-run this script\n"
            "Keygen: https://ebird.org/api/keygen",
            file=sys.stderr,
        )
        return 1

    OUT.mkdir(parents=True, exist_ok=True)

    print(f"Region species list: {args.region}")
    codes = api_get(f"product/spplist/{args.region}", key)
    (OUT / f"spplist_{args.region}.json").write_text(
        json.dumps(codes, indent=2), encoding="utf-8"
    )
    print(f"  {len(codes)} taxa → {OUT / f'spplist_{args.region}.json'}")

    print(f"Nearby hotspots: {args.lat},{args.lon} r={args.dist}km")
    try:
        hotspots = api_get(
            "ref/hotspot/geo",
            key,
            {"lat": args.lat, "lng": args.lon, "dist": args.dist, "fmt": "json"},
        )
        (OUT / "hotspots_near_way_canguk.json").write_text(
            json.dumps(hotspots, indent=2), encoding="utf-8"
        )
        print(f"  {len(hotspots)} hotspots")
        # per-hotspot species (can be slow)
        rows = []
        for hs in hotspots[:30]:
            loc = hs.get("locId") or hs.get("locID")
            name = hs.get("locName", "")
            if not loc:
                continue
            try:
                sp = api_get(f"product/spplist/{loc}", key)
            except Exception as e:
                print(f"  skip {loc}: {e}")
                continue
            print(f"  {loc} {name}: {len(sp)}")
            for code in sp:
                rows.append({"locId": loc, "locName": name, "speciesCode": code})
        if rows:
            path = OUT / "hotspot_species.csv"
            with path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=["locId", "locName", "speciesCode"])
                w.writeheader()
                w.writerows(rows)
            print(f"Wrote {path} ({len(rows)} rows)")
    except Exception as e:
        print(f"Hotspot query failed: {e}", file=sys.stderr)

    print("Done. Re-run build_species_lists.py with EBIRD_API_KEY set to merge region list.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
