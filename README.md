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
# see experiments/bacpipe/README.md
# Silent-window FP audit (BirdNET + Way Canguk list by default for 2A400)
python experiments/silent_chunk_fp_audit.py --location 2A400 --date 2026-04-21 \
  --methods mono --max-wavs 2 --with-birdnet --conf-threshold 0.7

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
    audits/           # FP / silent-chunk reports
```

Metadata schema: see `embedding_schema.py`.

## Archive species-ID path

Do **not** use species confidence as the primary beamforming metric.  
`run_pipeline.py` remains for historical runs and optional FP audits only.

## Work plan (integration)

See `docs/WORKPLAN.md`. Short version:

1. Freeze BirdNET embedding path + schema ✅ (this tree)
2. bacpipe pilot on existing WAVs (multi-model)
3. Silent-chunk / FP audit for thread update
4. Embedding-based beamforming metrics (extend `cluster_poc`)
5. Optional: pluggable backend in main pipeline
6. Later: linear probing / small classifier

## Environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Optional multi-model pilot:
# pip install -r experiments/bacpipe/requirements.txt
```

Paths: `config.py` (`MONITORING_DATA`, `ANALYSIS_OUTPUT`, `IR_BASE_PATH`) or env overrides.
