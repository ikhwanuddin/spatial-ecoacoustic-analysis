#!/usr/bin/env python3
"""
Build Way Canguk species lists for BirdNET Analyzer.

Pipeline:
  1. Load Gooddie (2015) appendix parse (gooddie_raw.json from parse_gooddie_appendix.py)
  2. Enrich with WCRS text mentions from fulltext
  3. Optional eBird API merge (region context; does not auto-expand WC)
  4. Match BirdNET GLOBAL 6K V2.4 labels
  5. Write CSVs + birdnet/species_list_way_canguk.txt
"""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent  # spatial-ecoacoustic-analysis
SCRIPTS = ROOT / "scripts"
SOURCES = ROOT / "sources"
DATA = ROOT / "data"
BIRDNET_OUT = ROOT / "birdnet"
ANDRIYANI_CSV = DATA / "andriyani_2022_species.csv"


def load_dotenv_files() -> None:
    """Load KEY=value from project .env files into os.environ (no overwrite)."""
    for path in (PROJECT / ".env", ROOT / ".env", Path.home() / ".config" / "ebird" / "api_key"):
        if not path.exists():
            continue
        if path.name == "api_key":
            # single-line raw key file
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

GOODDIE_JSON = DATA / "gooddie_raw.json"
GOODDIE_TXT = SOURCES / "gooddie_2015_fulltext.txt"
DEFAULT_PDF = Path(
    "/Users/ri322/.grok/sessions/%2FUsers%2Fri322%2Fmacmini/"
    "019f8706-486d-73f1-b983-2f2bd092a020/downloads/1.pdf"
)
DEFAULT_LABELS = (
    Path(__file__).resolve().parents[2]
    / "venv/lib/python3.11/site-packages/birdnetlib/models/analyzer"
    / "BirdNET_GLOBAL_6K_V2.4_Labels.txt"
)

# scientific name (Gooddie) → modern / BirdNET candidates
ALT_SCI: dict[str, list[str]] = {
    "Accipter trivirgatus": ["Accipiter trivirgatus"],
    "Hierococcyx sparveroides": ["Hierococcyx sparverioides"],
    "Chalcites minutillus": ["Chrysococcyx minutillus"],
    "Chalcites basalis": ["Chrysococcyx basalis"],
    "Picoides moluccensis": ["Yungipicus moluccensis"],
    "Picoides canicapillus": ["Yungipicus canicapillus"],
    "Chrysocolaptes validus": ["Reinwardtipicus validus", "Chrysocolaptes validus"],
    "Meiglyptes grammithorax": ["Meiglyptes tristis", "Meiglyptes grammithorax"],
    "Calorhamphus hayii": ["Caloramphus hayii"],
    "Alcedo peninsulae": ["Alcedo euryzona", "Alcedo peninsulae"],
    "Ramphiculus jambu": ["Ptilinopus jambu"],
    "Amaurornis cinerea": ["Poliolimnas cinereus"],
    "Charadrius mongolus": ["Anarhynchus mongolus"],
    "Charadrius leschenaultii": ["Anarhynchus leschenaultii"],
    "Pitta irena": ["Hydrornis irena", "Pitta irena"],
    "Pitta schneideri": ["Hydrornis schneideri", "Pitta schneideri"],
    "Pitta caerulea": ["Hydrornis caeruleus", "Pitta caerulea"],
    "Pitta granatina": ["Erythropitta granatina", "Pitta granatina"],
    "Pitta venusta": ["Erythropitta venusta", "Pitta venusta"],
    "Philentoma velatum": ["Philentoma velata"],
    "Philentoma pyrhopterum": ["Philentoma pyrhoptera"],
    "Rhinomyias brunneatus": ["Cyornis brunneatus"],
    "Rhinomyias umbratilis": ["Cyornis umbratilis"],
    "Rhinomyias olivacea": ["Cyornis olivaceus"],
    "Copsychus malabaricus": ["Kittacincla malabarica"],
    "Trichixos pyrropyga": ["Copsychus pyrropygus"],
    "Eumyias thalassina": ["Eumyias thalassinus"],
    "Parus major": ["Parus cinereus"],
    "Pycnonotus atriceps": ["Brachypodius melanocephalos", "Pycnonotus atriceps"],
    "Pycnonotus dispar": ["Rubigula dispar"],
    "Pycnonotus squamatus": ["Rubigula squamata"],
    "Pycnonotus cyaniventris": ["Ixodia cyaniventris"],
    "Pycnonotus eutilotus": ["Euptilotus eutilotus"],
    "Alophoixus finschii": ["Iole finschii"],
    "Iole olivacea": ["Iole crypta", "Iole olivacea"],
    "Artamus leucorhynchus": ["Artamus leucorynchus"],
    "Coracina fimbriata": ["Lalage fimbriata"],
    "Pericrocotus flammeus": ["Pericrocotus speciosus"],
    "Chloropsis cochinchinensis": ["Chloropsis moluccensis"],
    "Terpsiphone paradisi": ["Terpsiphone affinis", "Terpsiphone paradisi"],
    "Tephrodornis gularis": ["Tephrodornis virgatus"],
    "Zoothera interpres": ["Geokichla interpres"],
    "Garrulax mitratus": ["Pterorhinus mitratus"],
    "Garrulax lugubris": ["Melanocichla lugubris"],
    "Trichastoma rostratum": ["Pellorneum rostratum"],
    "Trichastoma bicolor": ["Pellorneum bicolor"],
    "Malacocincla malaccensis": ["Pellorneum malaccense"],
    "Stachyridopsis rufifrons": ["Cyanoderma rufifrons"],
    "Stachyridopsis chrysaea": ["Cyanoderma chrysaeum"],
    "Stachyris erythroptera": ["Cyanoderma erythropterum"],
    "Macronous gularis": ["Mixornis gularis"],
    "Macronous ptilosus": ["Macronus ptilosus"],
    "Seicercus castaniceps": ["Phylloscopus castaniceps"],
    "Anthreptes rhodolaema": ["Anthreptes rhodolaemus", "Anthreptes rhodolaema"],
    "Nectarinia jugularis": ["Cinnyris jugularis"],
    "Hypogramma hypogrammicum": ["Kurochkinegramma hypogrammicum"],
    "Leptocoma sperata": ["Leptocoma brasiliana", "Leptocoma sperata"],
    "Arachnothera affinis": ["Arachnothera modesta", "Arachnothera affinis"],
    "Dicaeum concolor": ["Dicaeum minullum"],
    "Prinia atrogularis": ["Prinia superciliaris"],
    "Dupetor flavicollis": ["Ixobrychus flavicollis"],
    "Icthyophaga humilis": ["Haliaeetus humilis", "Icthyophaga humilis"],
    "Icthyophaga ichthyaetus": ["Haliaeetus ichthyaetus", "Icthyophaga ichthyaetus"],
    "Hemicircus sordidus": ["Hemicircus concretus", "Hemicircus sordidus"],
    "Chrysophlegma humii": ["Chrysophlegma mentale", "Chrysophlegma humii"],
    "Calorhamphus hayii": ["Caloramphus hayii", "Caloramphus fuliginosus"],
    "Collocalia esculenta": ["Collocalia esculenta", "Collocalia affinis"],
}


def ensure_gooddie_json() -> list[dict]:
    parser = SCRIPTS / "parse_gooddie_appendix.py"
    pdf = Path(os.environ.get("GOODDIE_PDF", DEFAULT_PDF))
    if not GOODDIE_JSON.exists() or os.environ.get("FORCE_REPARSE"):
        if not pdf.exists():
            print(f"ERROR: Gooddie PDF not found: {pdf}", file=sys.stderr)
            sys.exit(1)
        cmd = [
            sys.executable,
            str(parser),
            "--pdf",
            str(pdf),
            "--out",
            str(GOODDIE_JSON),
        ]
        print("Running", " ".join(cmd))
        subprocess.check_call(cmd)
    return json.loads(GOODDIE_JSON.read_text(encoding="utf-8"))


def merge_andriyani(species: dict[str, dict]) -> int:
    """Merge songbird species named in Andriyani et al. 2022 (SPWC point counts)."""
    if not ANDRIYANI_CSV.exists():
        print("Andriyani CSV not found — skip")
        return 0
    added = 0
    with ANDRIYANI_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            sci = (row.get("scientific_name") or "").strip()
            if not sci:
                continue
            key = sci.lower()
            # also try common synonym keys
            keys = [key]
            if sci == "Rubigula dispar":
                keys.append("pycnonotus dispar")
            if sci == "Kittacincla malabarica":
                keys.append("copsychus malabaricus")
            matched_key = next((k for k in keys if k in species), None)
            if matched_key:
                rec = species[matched_key]
                rec["at_way_canguk"] = True
                if "andriyani2022" not in rec["sources"]:
                    rec["sources"].append("andriyani2022")
                rec["wcrs_text"] = True
            else:
                species[key] = {
                    "common_name_en": row.get("common_name_en") or "",
                    "scientific_name_original": sci,
                    "iucn": "",
                    "at_way_canguk": True,
                    "at_way_titias": False,
                    "at_danau_ranau": False,
                    "sources": ["andriyani2022"],
                    "wcrs_text": True,
                    "wc_from_text_only": False,
                    "common_name_id": row.get("common_name_id") or "",
                }
                added += 1
    print(f"Andriyani 2022: merged named taxa; {added} new species")
    return added


def enrich_wcrs_text(species: dict[str, dict], text: str) -> None:
    """Flag species whose accounts mention WCRS / Present at WCRS."""
    head_re = re.compile(
        r"^(?P<common>[A-Z][A-Za-z'’\-\s]{2,50}?)\s+"
        r"(?P<genus>[A-Z][a-z]+)\s+"
        r"(?P<epithet>[a-z]+)\b"
    )
    wcrs_re = re.compile(r"(?:Present at WCRS|present at WCRS|\bat WCRS\b|WCRS\b)")
    current = None
    for raw in text.splitlines():
        line = re.sub(r"\s+", " ", raw).strip()
        if not line:
            continue
        m = head_re.match(line)
        if m:
            current = f"{m.group('genus')} {m.group('epithet')}".lower()
        if current and wcrs_re.search(line):
            if current in species:
                if not species[current]["at_way_canguk"]:
                    species[current]["at_way_canguk"] = True
                    species[current]["wc_from_text_only"] = True
                src = species[current].setdefault("sources", [])
                if "brickle_wcrs" not in src:
                    src.append("brickle_wcrs")
                species[current]["wcrs_text"] = True


def load_birdnet_labels(path: Path) -> dict[str, str]:
    by_sci: dict[str, str] = {}
    for ln in path.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or "_" not in ln:
            continue
        sci = ln.split("_", 1)[0]
        by_sci.setdefault(sci.lower(), ln)
    return by_sci


def match_birdnet(sci: str, by_sci: dict[str, str]) -> str:
    candidates = [sci] + ALT_SCI.get(sci, [])
    seen = set()
    for c in candidates:
        cl = c.lower()
        if cl in seen:
            continue
        seen.add(cl)
        if cl in by_sci:
            return by_sci[cl]
    return ""


def habitat_tier(rec: dict) -> str:
    if rec["at_way_canguk"]:
        return "lowland"
    if rec["at_danau_ranau"] and not rec["at_way_titias"]:
        return "montane"
    if rec["at_way_titias"]:
        return "mid"
    return "unknown"


def fetch_json(url: str, api_key: str):
    req = urllib.request.Request(url, headers={"X-eBirdApiToken": api_key})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


def merge_ebird(species: dict[str, dict], api_key: str) -> None:
    """Attach eBird Lampung region + nearby hotspot presence (does not expand WC)."""
    tax_rows = fetch_json(
        "https://api.ebird.org/v2/ref/taxonomy/ebird?fmt=json&locale=en", api_key
    )
    tax = {r["speciesCode"]: r for r in tax_rows if "speciesCode" in r}
    codes = fetch_json(
        "https://api.ebird.org/v2/product/spplist/ID-SM-LA", api_key
    )
    print(f"eBird ID-SM-LA: {len(codes)} taxa")
    added = 0

    def upsert(sci: str, com: str, source: str) -> None:
        nonlocal added
        parts = sci.split()
        if len(parts) != 2:
            return
        key = sci.lower()
        if key in species:
            if source not in species[key]["sources"]:
                species[key]["sources"].append(source)
        else:
            species[key] = {
                "common_name_en": com,
                "scientific_name_original": sci,
                "iucn": "",
                "at_way_canguk": False,
                "at_way_titias": False,
                "at_danau_ranau": False,
                "sources": [source],
                "wcrs_text": False,
                "wc_from_text_only": False,
            }
            added += 1

    for code in codes:
        r = tax.get(code)
        if not r:
            continue
        upsert(r.get("sciName") or "", r.get("comName") or "", "ebird_ID-SM-LA")

    # Nearby hotspot cache from fetch_ebird.py
    hotspot_csv = DATA / "ebird" / "hotspot_species.csv"
    if hotspot_csv.exists():
        # lowland-relevant hotspots near WCRS (southern peninsula / TNBBS lowland)
        lowland_locs = {
            "L6349760": "Belimbing",
            "L5865707": "Bengkunat",
            "L6349792": "ForestTrail",
            "L6349779": "DanauMenjukut",
        }
        with hotspot_csv.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                loc = row.get("locId") or ""
                code = row.get("speciesCode") or ""
                r = tax.get(code)
                if not r:
                    continue
                sci = r.get("sciName") or ""
                com = r.get("comName") or ""
                if loc in lowland_locs:
                    upsert(sci, com, f"ebird_{lowland_locs[loc]}")
                else:
                    upsert(sci, com, f"ebird_hotspot_{loc}")
        print(f"  merged hotspot cache: {hotspot_csv}")
    print(f"  new species from eBird only: {added}")


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    load_dotenv_files()
    DATA.mkdir(parents=True, exist_ok=True)
    BIRDNET_OUT.mkdir(parents=True, exist_ok=True)

    raw = ensure_gooddie_json()
    species: dict[str, dict] = {}
    for r in raw:
        key = r["scientific_name_original"].lower()
        species[key] = {
            **r,
            "sources": list(r.get("sources") or ["gooddie2015"]),
            "wcrs_text": False,
            "wc_from_text_only": False,
        }

    if GOODDIE_TXT.exists():
        enrich_wcrs_text(species, GOODDIE_TXT.read_text(encoding="utf-8"))

    merge_andriyani(species)

    api_key = os.environ.get("EBIRD_API_KEY", "").strip()
    if api_key:
        try:
            merge_ebird(species, api_key)
        except Exception as e:
            print(f"WARNING: eBird failed: {e}", file=sys.stderr)
    else:
        print("EBIRD_API_KEY not set — skipping eBird (export key then re-run)")

    labels_path = Path(os.environ.get("BIRDNET_LABELS", DEFAULT_LABELS))
    if not labels_path.exists():
        print(f"ERROR: labels not found: {labels_path}", file=sys.stderr)
        return 1
    by_sci = load_birdnet_labels(labels_path)
    print(f"BirdNET labels: {len(by_sci)}")

    fields = [
        "scientific_name_original",
        "scientific_name_matched",
        "common_name_en",
        "iucn",
        "at_way_canguk",
        "at_way_titias",
        "at_danau_ranau",
        "habitat_tier",
        "wcrs_text",
        "wc_from_text_only",
        "sources",
        "birdnet_label",
        "in_birdnet_v24",
    ]

    rows = []
    for rec in species.values():
        sci = rec["scientific_name_original"]
        lab = match_birdnet(sci, by_sci)
        matched_sci = lab.split("_", 1)[0] if lab else ""
        rows.append(
            {
                "scientific_name_original": sci,
                "scientific_name_matched": matched_sci,
                "common_name_en": rec.get("common_name_en", ""),
                "iucn": rec.get("iucn", ""),
                "at_way_canguk": bool(rec.get("at_way_canguk")),
                "at_way_titias": bool(rec.get("at_way_titias")),
                "at_danau_ranau": bool(rec.get("at_danau_ranau")),
                "habitat_tier": habitat_tier(rec),
                "wcrs_text": bool(rec.get("wcrs_text")),
                "wc_from_text_only": bool(rec.get("wc_from_text_only")),
                "sources": ";".join(rec.get("sources") or []),
                "birdnet_label": lab,
                "in_birdnet_v24": bool(lab),
            }
        )

    rows.sort(key=lambda r: r["scientific_name_original"].lower())
    write_csv(DATA / "master_species.csv", rows, fields)

    wc_rows = [r for r in rows if r["at_way_canguk"]]
    write_csv(DATA / "way_canguk_species.csv", wc_rows, fields)

    matched = [r for r in wc_rows if r["in_birdnet_v24"]]
    unmatched = [r for r in wc_rows if not r["in_birdnet_v24"]]
    write_csv(DATA / "birdnet_matched.csv", matched, fields)
    write_csv(DATA / "not_in_birdnet.csv", unmatched, fields)

    labels = sorted({r["birdnet_label"] for r in matched if r["birdnet_label"]})
    out = BIRDNET_OUT / "species_list_way_canguk.txt"
    out.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")

    print("\n=== Summary ===")
    print(f"Master species:           {len(rows)}")
    print(f"Way Canguk (WC/S) T1:     {len(wc_rows)}")
    print(f"  WCRS text confirmed:    {sum(1 for r in wc_rows if r['wcrs_text'])}")
    print(f"  in BirdNET V2.4:        {len(matched)} ({100*len(matched)/max(len(wc_rows),1):.1f}%)")
    print(f"  not in BirdNET:         {len(unmatched)}")
    print(f"species_list lines:       {len(labels)}")
    print(f"Wrote {out}")

    icons = [
        "Argusianus argus",
        "Eupetes macrocerus",
        "Pitta irena",
        "Pitta granatina",
        "Rhinoplax vigil",
        "Buceros bicornis",
        "Carpococcyx viridis",
    ]
    print("\nSpot-check:")
    for name in icons:
        hits = [r for r in rows if r["scientific_name_original"] == name]
        for h in hits:
            print(
                f"  {h['scientific_name_original']:28} WC={h['at_way_canguk']} "
                f"BN={h['birdnet_label'] or '—'}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
