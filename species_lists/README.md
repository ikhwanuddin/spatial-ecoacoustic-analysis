# Way Canguk bird species lists (BirdNET-ready)

Custom species lists and source tables for **Stasiun Penelitian Way Canguk** (WCRS), Taman Nasional Bukit Barisan Selatan, Sumatra — for use with **BirdNET Analyzer** / `birdnetlib`.

## Quick use

```bash
# BirdNET Analyzer CLI
birdnet_analyzer analyze /path/to/audio \
  --slist species_lists/birdnet/species_list_way_canguk.txt \
  --lat -5.689 --lon 104.409 \
  --min_conf 0.25
```

In the BirdNET Analyzer GUI: **Custom species list** → choose  
`species_lists/birdnet/species_list_way_canguk.txt`.

Label format (required by BirdNET):

```text
Genus species_Common Name
```

## Site context

| Item | Value |
|------|--------|
| Site | Way Canguk Research Station (WCRS) |
| Coordinates | ~5.689°S, 104.409°E |
| Elevation | ~50 m (lowland primary forest) |
| Note | Gooddie column **WC/S** = Way Canguk **+ southern peninsula** (includes some coastal/swamp species) |

## Deliverables

| Path | Description |
|------|-------------|
| `birdnet/species_list_way_canguk.txt` | **Primary** — BirdNET custom list (WC/S species present in model V2.4) |
| `data/way_canguk_species.csv` | Full T1 table (all WC/S, matched + unmatched) |
| `data/master_species.csv` | Full TNBBS appendix (+ optional eBird region) with site flags |
| `data/birdnet_matched.csv` | WC/S species with valid BirdNET labels |
| `data/not_in_birdnet.csv` | WC/S species **not** in BirdNET 6K V2.4 (fine-tune / manual ID candidates) |
| `data/gooddie_raw.json` | Raw parse of Gooddie Appendix 1 |

## Coverage (current build)

Re-run `scripts/build_species_lists.py` to refresh numbers. Typical:

- ~416–430 species in Gooddie TNBBS appendix  
- ~280 marked **WC/S** (Way Canguk + southern peninsula)  
- ~85% of WC/S match BirdNET V2.4 labels  
- Remaining unmatched → `not_in_birdnet.csv` (e.g. some Sunda endemics / rare taxa)

## Sources

1. **Gooddie, C. (2015).** Ornithological records from Bukit Barisan Selatan National Park, Sumatra, Indonesia. *Forktail* 31: 70–81.  
   - Appendix 1 with site columns WC/S, WT/L/K, DR  
   - Incorporates Nick Brickle WCRS records (1997–2009, ~208 species at the station proper)
2. **Species accounts** in the same paper (“Present at WCRS”) — used as confirmation flags (`wcrs_text`)
3. **eBird API** (optional) — Lampung region `ID-SM-LA` merge when `EBIRD_API_KEY` is set  
4. **BirdNET GLOBAL 6K V2.4** labels (via `birdnetlib` package in this repo’s venv)

Planned / not yet merged:

- Andriyani et al. (2022) singing-bird point counts at WCRS (J-BEKH) — PDF access was blocked; merge when available  
- O’Brien & Kinnaird (1996) *Oryx* baseline (276 park-wide)

See `sources/bibliography.md`.

## Rebuild

```bash
cd spatial-ecoacoustic-analysis

# 1) Parse Gooddie PDF (needs pdfplumber — already installed in venv)
./venv/bin/python species_lists/scripts/parse_gooddie_appendix.py \
  --pdf /path/to/Bukit-Barisan.pdf \
  --out species_lists/data/gooddie_raw.json

# 2) Match BirdNET + write lists
./venv/bin/python species_lists/scripts/build_species_lists.py

# 3) Optional eBird
export EBIRD_API_KEY=your_key   # https://ebird.org/api/keygen
./venv/bin/python species_lists/scripts/fetch_ebird.py
./venv/bin/python species_lists/scripts/build_species_lists.py
```

Environment variables:

| Variable | Purpose |
|----------|---------|
| `EBIRD_API_KEY` | eBird API token (prefer `.env`, not chat) |
| `BIRDNET_LABELS` | Override path to `*_Labels.txt` |
| `GOODDIE_PDF` | Override path to Gooddie PDF |
| `FORCE_REPARSE=1` | Force re-parse of PDF even if JSON exists |

### eBird API key (do **not** paste in chat)

```bash
cd spatial-ecoacoustic-analysis
cp .env.example .env
# buka .env di editor, isi: EBIRD_API_KEY=xxxxxxxx
# lalu bilang ke agent "key sudah di .env" — atau jalankan sendiri:
./venv/bin/python species_lists/scripts/fetch_ebird.py
./venv/bin/python species_lists/scripts/build_species_lists.py
```

File `.env` sudah di-ignore git.

## Design choices

1. **Primary list = WC/S only** (not full montane TNBBS) to reduce false positives on lowland MAARU recordings.  
2. Labels must **match BirdNET V2.4 exactly** (scientific + English common name string).  
3. Taxonomy synonyms (splits/lumps since 2015) are mapped in `build_species_lists.py` (`ALT_SCI`).  
4. Species absent from the model stay in CSV for training/fine-tune planning — they are **not** forced into `species_list.txt`.

## Integration with `spatial-ecoacoustic-analysis`

Wired into `birdnet_processor.py` / `run_pipeline.py` via `config.resolve_birdnet_filter`:

| Location | Filter mode |
|----------|-------------|
| Way Canguk plots (`S0`, `2A400`, …) or alias `waycanguk` | **Custom list** `species_list_way_canguk.txt` (lat/lon **not** passed — birdnetlib forbids combining them) |
| Other sites with coords (e.g. `silwood`) | **Geo** lat/lon only (eBird-range model inside BirdNET) |
| Unknown | No location filter (full model + `min_conf`) |

```bash
# normal WC run
python run_pipeline.py --location 2A400 --date 2026-04-20

# force geo-only even at WC (debug)
BIRDNET_USE_SPECIES_LIST=0 python run_pipeline.py --location 2A400 --date ...
```

**What lat/lon does in BirdNET:** not “who the user is”. It feeds a species-range model (eBird-derived) so only taxa considered plausible at that coordinate/week are kept. A literature custom list replaces that function for Way Canguk with a tighter, checklist-based allow-list.

## Caveats

- WC/S includes **southern peninsula** waterbirds/coastal species that may be rare on WCRS trail grid.  
- BirdNET global models under-represent some Sumatran endemics (see `not_in_birdnet.csv`).  
- eBird coverage at WCRS itself is sparse; literature remains the backbone.
