# spatial-ecoacoustic-analysis

Spatial (array) ecoacoustic pipeline for MAARU monitoring data: beamforming and baselines → **dense embeddings** for method comparison. Species-ID scoring is **archived** (not the research path).

## Entry points

| Status | Script | Role |
|--------|--------|------|
| **Active** | `pipeline_signal_processing.py` | FLAC → BF + SA + mono signal methods
| **Active** | `cluster_poc.py` | UMAP / HDBSCAN / noise-distance on embeddings |
| **Active** | `process_noise_reference.py` | Noise-reference embeddings for distance metrics |
| **Active** | `extract_embeddings.py` | Embeddings only (WAVs already on disk) |
| **Pilot** | `experiments/bacpipe/run_pilot.py` | Multi-model embeddings via bacpipe |
| **Archive** | `run_pipeline.py` | Species-ID path (results.json / prefilter / conf) |
| **Archive** | `birdnet_processor.py`, `generate_report.py` | Species detection + reporting |

Shared DSP (always keep): `beamforming.py`, `signal_averaging.py`, `ircache.py`, `config.py`, `ir_cache/`.

Domain lists (not the main metric): `species_lists/`.

## Active usage

```bash
# Signal processing only, always regenerate (BF + SA + mono)
# LabIR: S01/S05/S09 at 0,60,120,180,240,300; S12 at 0 automatically
python pipeline_signal_processing.py --location 2A400 \
  --date 2026-04-26 --ir-types LabIR,SPIR1,SPIR2 \
  --labir-speakers S01,S05,S09,S12 \
  --labir-degrees 0,60,120,180,240,300 \
  --max-files 0 --force-signal

# BirdNET/bacpipe are separate downstream workflows on the resulting WAVs.

# Cluster / visualise (legacy flat embeddings still default on disk)
python cluster_poc.py \
  --embeddings /Volumes/WD2TB/sea-data/2A400/embeddings

# Multi-backend resolve (native birdnet/ + bacpipe/ + legacy)
python cluster_poc.py --location 2A400 --backends legacy
python cluster_poc.py --location 2A400 --backends bacpipe --bacpipe-models birdnet

# Offline HTML visualisations from existing bacpipe embeddings.
# Produces one HTML per model, one shared plotly.min.js, index.html,
# manifest.json, and a ZIP ready to download from HPC.
python visualize_bacpipe.py \
  --data-dir /rds/general/user/ri322/home/sea-data \
  --location 2A400 --date 2026-04-26 --models all
# Output: sea-data/2A400/embedding_visuals/2026-04-26/
# ZIP:    sea-data/2A400/embedding_visuals/2A400_embedding_visuals_2026-04-26.zip

# bacpipe multi-model pilot (separate venv)
# Compare embeddings from multiple bioacoustic models on the same existing method WAVs.
# Models may also support species classification, but this workstream first evaluates
# model-space structure / acoustic fingerprints; species labels are not assumed to be
# reliable or required.
# see experiments/bacpipe/README.md

# Embedding method / direction metrics
python embedding_metrics.py --location 2A400 --date 2026-04-21 --backends legacy

# All-model bacpipe comparison (mono is the fixed baseline)
# source experiments/bacpipe/.venv && export PYTHONPATH=$PWD:$PYTHONPATH
# python experiments/bacpipe/run_pilot.py --location 2A400 --date 2026-04-26 \
#   --models all --methods mono,sa,bf_LabIR,bf_SPIR \
#   --noise-dir /Volumes/WD2TB/sea-data/2A400/noise_references
# Reports: sea-data/.../embeddings/audits/2026-04-26_bacpipe_comparison.{json,md}
```

## Output layout

```text
sea-data/{location}/
  {date}/bf_LabIR|bf_SPIR|sa|mono/...
  embeddings/
    birdnet/          # native pipeline (default)
    bacpipe/{model}/  # multi-model pilot
    audits/           # model comparisons, metric/protocol reports, and historical audits
```

Metadata schema: see `embedding_schema.py`.

## Archive species-ID path

Do **not** use species confidence as the primary beamforming metric.  
`run_pipeline.py` and `experiments/silent_chunk_fp_audit.py` remain available for historical/optional audits only; silent-chunk FP auditing is not a current Goal 1 workstream.

## Work plan (integration)

See `docs/WORKPLAN.md`. Short version:

1. Freeze BirdNET embedding path + schema ✅ (this tree)
2. bacpipe pilot on existing WAVs (multi-model)
3. Compare model embeddings / acoustic fingerprints on matched methods and scopes
4. Evaluate embedding-based evidence for beamforming comparisons (extend `cluster_poc` / metrics as justified)
5. Optional: pluggable backend in main pipeline
6. Species-ID and species probing are outside the current research scope

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional multi-model pilot:
# pip install -r experiments/bacpipe/requirements.txt
```

Paths: `config.py` (`MONITORING_DATA`, `ANALYSIS_OUTPUT`, `IR_BASE_PATH`) or env overrides.
