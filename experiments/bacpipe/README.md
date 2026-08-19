# bacpipe pilot

Run **multi-model embeddings** on WAVs already produced by `pipeline_embeddings.py` (or legacy BF outputs). Does not re-run beamforming.

## Why a separate env

bacpipe pulls PyTorch / TensorFlow stacks that can conflict with the main `birdnetlib` venv. Prefer:

```bash
cd /path/to/spatial-ecoacoustic-analysis
python3.11 -m venv experiments/bacpipe/.venv
source experiments/bacpipe/.venv/bin/activate
pip install -r experiments/bacpipe/requirements.txt
# Plus repo root on PYTHONPATH for config + embedding_schema:
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

Or install bacpipe into the main venv if you accept dependency risk.

Checkpoints auto-download to `experiments/bacpipe/checkpoints/` (gitignored)
via HuggingFace `vskode/bacpipe_models`. First run is large (BirdNET tar, etc.).

## Usage

```bash
# List WAVs that would be processed (no models)
python experiments/bacpipe/run_pilot.py \
  --location 2A400 --date 2026-04-22 --dry-run

# Embed with birdnet + perch_bird (downloads checkpoints on first run)
python experiments/bacpipe/run_pilot.py \
  --location 2A400 --date 2026-04-22 \
  --models birdnet,perch_bird \
  --methods bf_LabIR,bf_SPIR,sa,mono \
  --max-wavs 8

# Single model, explicit data root
python experiments/bacpipe/run_pilot.py \
  --location 2A400 --date 2026-03-19 \
  --models perch_bird \
  --data-dir /Volumes/WD2TB/sea-data
```

Outputs:

```text
{data_dir}/{location}/embeddings/bacpipe/{model}/
  {date}_{method}_embeddings.npy
  {date}_{method}_meta.json
  {date}_summary.json
```

## Notes

- Windowing follows each bacpipe model’s native segment length (not always 3 s). Meta records `window_sec` / `slide_sec` from the embedder when available; otherwise `model_native`.
- The main purpose of this pilot is to compare model representations and acoustic fingerprints across the same processing methods. Species labels or classifier confidence are not assumed to be reliable for tropical forest recordings and are not required for the core comparison.
- For scientific comparison of methods (LabIR vs SA vs mono), compare **within the same model**, then optionally compare patterns across models.
- Native BirdNET dense embeddings remain the baseline under `embeddings/birdnet/`.
