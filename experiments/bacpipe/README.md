# bacpipe pilot

Run **multi-model embeddings** on WAVs already produced by `pipeline_signal_processing.py` (or legacy BF outputs). Does not re-run beamforming.

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

# Run every model exposed by the installed bacpipe registry
# mono is the fixed baseline; sa and BF are comparators.
python experiments/bacpipe/run_pilot.py \
  --location 2A400 --date 2026-04-26 \
  --models all \
  --methods mono,sa,bf_LabIR,bf_SPIR \
  --noise-dir /Volumes/WD2TB/sea-data/2A400/noise_references

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
  noise_{group}_embeddings.npy       # same model space as method embeddings
  noise_{group}_meta.json
  {date}_summary.json                # noise distance + Δ versus mono
{data_dir}/{location}/embeddings/audits/
  {date}_bacpipe_comparison.json
  {date}_bacpipe_comparison.md
```

## Notes

- Windowing follows each bacpipe model’s native segment length (not always 3 s). Meta records `window_sec` / `slide_sec` from the embedder when available; otherwise `model_native`.
- The main purpose of this pilot is to compare model representations and acoustic fingerprints across the same processing methods. Species labels or classifier confidence are not used for this experiment.
- `mono` is the fixed baseline. `sa`, `bf_LabIR`, and `bf_SPIR` are comparators; reports include absolute noise distance and Δ versus mono.
- Noise references are re-embedded separately for every model. Cross-model noise vectors are never reused.
- Native BirdNET dense embeddings remain available under `embeddings/birdnet/`, but bacpipe is evaluated as a general multi-model backend.
