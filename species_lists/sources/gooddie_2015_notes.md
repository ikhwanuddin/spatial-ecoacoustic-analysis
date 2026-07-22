# Notes: Gooddie (2015) Forktail appendix parse

## Paper

Gooddie, C. (2015). Ornithological records from Bukit Barisan Selatan National Park, Sumatra, Indonesia. *Forktail* 31: 70–81.

Local copy: `gooddie_2015_forktail_bbsnp.pdf`

## Site columns (Appendix 1)

| Code | Meaning |
|------|---------|
| WC/S | Way Canguk Research Station **and** southern peninsula (Belimbing, Tampang, Sukaraja, coastal/swamp) |
| WT/L/K | Way Titias / Liwa / Kubuperahu (~850–1100 m, submontane) |
| DR | Danau Ranau montane (~1200–1700 m) |

WCRS coordinates in paper: **5.689°S, 104.409°E**, ~50 m a.s.l.

## Parse method

`scripts/parse_gooddie_appendix.py` uses **pdfplumber** word coordinates:

1. Detect dual-column header row (`WC/S`, `WT/L/K`, `DR`) per page  
2. Assign each `x` mark to nearest column centre within tolerance  
3. Extract binomial + English common name from the same half-row  

Validated against 12 hand-checked species (Great Argus, Ground-Cuckoo, Helmeted Hornbill, etc.) — all correct.

## Relation to Brickle WCRS list

Gooddie cites Nick Brickle’s list of **208** species at WCRS (1997–2009).  
The **WC/S** column is broader (~280 taxa in our parse) because it also includes southern-peninsula records. For strict station-grid work, prefer rows with `wcrs_text=true` in the CSV, or filter coastal families manually.
