# spatial-ecoacoustic-analysis

Spatial (array) ecoacoustic pipeline for MAARU monitoring data: beamforming and baselines → **dense embeddings** for method comparison. Species-ID scoring is **archived** (not the research path).

## Entry points

| Status | Script | Role |
|--------|--------|------|
| **Active** | `pipeline_embeddings.py` | BF → SA → mono → dense BirdNET embeddings |
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
# Full active pipeline (signal processing + BirdNET embeddings)
python pipeline_embeddings.py --location 2A400 \
  --date 2026-04-22 --ir-types LabIR,SPIR1,SPIR2 --max-files 0

# Cluster / visualise (legacy flat embeddings still default on disk)
python cluster_poc.py \
  --embeddings /Volumes/WD2TB/sea-data/2A400/embeddings

# Multi-backend resolve (native birdnet/ + bacpipe/ + legacy)
python cluster_poc.py --location 2A400 --backends legacy
python cluster_poc.py --location 2A400 --backends bacpipe --bacpipe-models birdnet

# bacpipe multi-model pilot (separate venv)
# Compare embeddings from multiple bioacoustic models on the same existing method WAVs.
# Models may also support species classification, but this workstream first evaluates
# model-space structure / acoustic fingerprints; species labels are not assumed to be
# reliable or required.
# see experiments/bacpipe/README.md

# Embedding method / direction metrics
python embedding_metrics.py --location 2A400 --date 2026-04-21 --backends legacy

# Multi-method bacpipe pilot (balanced per method)
# source experiments/bacpipe/.venv && export PYTHONPATH=$PWD:$PYTHONPATH
# python experiments/bacpipe/run_pilot.py --location 2A400 --date 2026-04-21 \
#   --models birdnet,perch_bird --methods mono,sa,bf_LabIR --max-wavs-per-method 2
# Comparison write-up: sea-data/.../embeddings/audits/2026-04-21_multi_model_comparison.md
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
6. Later and only if needed: linear probing / small classifier for species-related analysis

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional multi-model pilot:
# pip install -r experiments/bacpipe/requirements.txt
```

Paths: `config.py` (`MONITORING_DATA`, `ANALYSIS_OUTPUT`, `IR_BASE_PATH`) or env overrides.
