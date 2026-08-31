# WORKPLAN — Spatial ecoacoustic analysis (MAARU / Way Canguk)

**Repo:** `spatial-ecoacoustic-analysis`  
**Primary reader:** Rifqi (PhD) — continue in **Zed**  
**Last realigned:** 2026-07-31 (conversation after Grok session; goals restated by user)  
**Status:** Pre-session pipeline + noise-distance story is the scientific spine. bacpipe multi-model is an **early check**, not a rewrite of the PhD question.

---

## 0. How to use this document

- Single source of truth for **why** we work and **what order** to work in.
- Prefer **disk + code** over any chat history if they conflict; then fix this file.
- Agent TUI sessions can drift. Re-read §1 before any new coding session.

---

## 1. Goals (user-authored; do not dilute)

### Goal 1 — Primary (PhD core)

**Test whether signal averaging and beamforming improve usable signal quality relative to the fixed monochannel baseline**, on multi-mic MAARU recordings from **Way Canguk, Lampung, Indonesia**.

Operational meaning of “BF better”:

| Era | Evidence used | Status |
|-----|----------------|--------|
| Earlier | BirdNET species conf / “more trusted species IDs” under BF | **Abandoned as primary metric** — Indonesian soundscape poorly matched to BirdNET species head |
| Current (pre–Grok session) | **Noise reference embeddings + noise distance** on dense BirdNET embeddings (clusters / per-method stats) | **Main success path so far** — user already sees promising results on limited hours of data: BF better than mono/SA |
| Next (early check) | Same *kind* of claim, tested across **all feasible embedding models exposed by bacpipe** | Do `sa`, `bf_LabIR`, and `bf_SPIR` remain better than the fixed `mono` baseline across model spaces? |

**Why this matters:** Mic-array + beamforming is only scientifically defensible for PAM if the gain is not an artefact of one classifier’s species head.

**What Goal 1 is *not*:**

- Not “train a Way Canguk species detector first”
- Not “maximise separability gap of method labels” as the main story (that was an agent-side exploratory metric; secondary only)
- Not abandoning noise-distance work in favour of a new metric zoo

### Species identification — outside current scope

Species annotation, local-expert labelling, species probing, and classifier development are not current goals of this workstream. BirdNET species-ID remains archived context for the pivot to embeddings, not an experiment to develop now.

### Claim clarification (email / BirdNET hallucination)

- High conf (≥ 0.7) discussion referred to **BF chunks**, not mono.
- That motivates distrust of species-conf as BF quality metric; it does **not** redefine Goal 1 as an FP-audit project.
- **FP-on-BF silent audit is not a current workstream.**

---

## 2. Pipeline you already had (before Grok session)

This remains the backbone. Agent work must not replace it without reason.

```text
FLAC (array)
  → mono          (ch0 baseline)
  → sa            (signal average)
  → bf_LabIR      (lab IR beamforming)
  → bf_SPIR       (field SPIR beamforming)
  → signal-method WAVs only (bf_LabIR, bf_SPIR, sa, mono)
  → separate BirdNET/bacpipe workflows on those WAVs
```

Entry points (active):

| Script | Role |
|--------|------|
| `pipeline_signal_processing.py` | FLAC → BF + SA + mono signal-method WAVs |
| `process_noise_reference.py` | Noise reference embeddings |
| `cluster_poc.py` | UMAP / HDBSCAN / dashboard; noise-distance hooks |
| `extract_embeddings.py` | Embed-only if WAVs already exist |

Archive (species-ID era — keep, do not drive Goal 1):

- `run_pipeline.py`, `birdnet_processor.py`, `generate_report.py`, `prefilter.py`
- `report_data/*_detections.json.gz` under sea-data (historical conf)

Data roots (`config.py`):

| Role | Default |
|------|---------|
| Analysis | `/Volumes/WD2TB/sea-data` |
| Raw FLACs | `/Volumes/WD2TB/monitoring_data` (env `MONITORING_DATA`) |

---

## 3. Alignment of Grok session vs your goals (honest audit)

### What fitted Goal 1

| Item | Fit |
|------|-----|
| Keep one repo; bacpipe as adapter not rewrite | Good |
| Archive species-ID as primary BF score | Good |
| Bacpipe integration across available models; mono baseline with SA/BF comparators | Good *as early multi-model check* |
| Preserve native BirdNET embedding path | Good |

### What drifted or needs re-centering

| Item | Issue | Correction going forward |
|------|--------|---------------------------|
| **Separability gap** as headline metric | Answers “do methods look different?” more than “is BF cleaner / higher SNR-like?” | Keep code optional; **do not treat as main claim**. Prefer **noise distance** and related SNR-proxy metrics you already trusted |
| Heavy FP audit tooling | Useful context; easy to steal focus | **No further FP-on-BF work** unless you reopen it |
| Default out path `embeddings/birdnet/` | Schema OK; most existing data still **flat** under `embeddings/*.npy` | Loaders must keep supporting **legacy flat** layout |
| Bacpipe integration smoke test (2 WAV/method) | Fine for smoke, not for Goal 1 conclusion | Scale only after metrics match **your** success criteria (noise distance first) |
| Direction gap on 2 azimuths | Exploratory | Secondary to mono vs SA vs BF noise distance |

### What you already found satisfactory (preserve)

On limited hours of data, with **BirdNET embeddings + noise references + noise distance**:

- BF looks **better** than mono/SA  
- Array PAM looks **promising**  

**Any new multi-model work should try to reproduce that *kind* of conclusion**, not replace it with an unrelated metric narrative.

---

## 4. Goal 1 — planned workstreams (order)

### 4.1 Freeze and document the “happy path” (mono baseline + method comparators)

Priority: so every model comparison has the same fixed `mono` baseline.

1. Commands to regenerate/load embeddings for mono, sa, bf_LabIR, bf_SPIR on a chosen date set.  
2. How noise references were built (`process_noise_reference.py`).  
3. Exactly how noise distance is computed in `cluster_poc` / `embedding_metrics` (cosine to noise mean; distance = 1 − cos).  
4. Which plots/HTML/tables convinced you BF wins.

**Deliverable:** short subsection or link from this file once you freeze a “reference run” date list in Zed (edit this section).

**Reference run (fill in when frozen):**

- Location: `2A400`  
- First sample: `2026-04-26` (3 FLACs, 3 times, 240 seconds each)  
- Raw input: `/Volumes/WD2TB/monitoring_data/RPiID-0000000091668b26/2026-04-26`  
- Analysis output: `/Volumes/WD2TB/sea-data/2A400`  
- Baseline: `mono`  
- Comparators: `sa`, `bf_LabIR`, `bf_SPIR`  
- Metrics: model-specific noise distance and delta versus `mono`  

### 4.2 All-model check via bacpipe (your “selanjutnya”)

**Question:** On the **same WAVs** (`mono` / `sa` / `bf_*`), do all feasible bacpipe embedding models show the comparators as better than the fixed `mono` baseline under the same noise-distance logic?

Implementation already partly scaffolded (session):

| Path | Role |
|------|------|
| `bacpipe/pipeline_bacpipe.py` | Embed existing method WAVs with bacpipe models |
| `bacpipe/.venv` | Isolated env (bacpipe 1.3.3) |
| `embeddings/bacpipe/{model}/` | Bacpipe npy outputs under sea-data |
| `direction_meta.py` | az/el parse without TF |

**Rules for this workstream:**

1. **Do not re-run beamforming** unless audio missing — reuse method WAVs.  
2. **Primary comparison metric = noise distance** (or the same definition you used pre-session), applied **per model**.  
3. Separability gap / UMAP = supporting diagnostics only.  
4. Use `--models all` to discover and try every model exposed by the installed bacpipe version.  
5. Record unavailable, incompatible, or failed models rather than silently dropping them.  
6. Compare every model internally; do not reuse noise vectors across models. Species heads and probing are outside scope.

**Success criterion (early check):**

- For each feasible model, report `mono` as the fixed baseline and `sa`, `bf_LabIR`, `bf_SPIR` as comparators.  
- Report both absolute model-space noise distance and `Δ versus mono`; positive delta means farther from noise.  
- If the pattern differs by model, record the result as model-dependent or inconclusive rather than selecting a preferred model prematurely.

### 4.3 Scale (only after 4.2 metric protocol is agreed)

- More minutes / dates / LabIR directions / SPIR.  
- Matched event windows (same time mono vs BF) if noise-distance alone is too coarse.  
- Multi-date table: method × model × noise distance.

### 4.4 Optional later (not Goal 1 blockers)

- bacpipe official Panel dashboard wired to our layout  
- Bacpipe is an active integrated downstream stage in `bacpipe/pipeline_bacpipe.py`  
- Folder cleanup of archive scripts  

---

## 5. Species-ID work — out of scope

Do not spend current agent cycles on annotation, taxonomy, local-expert labelling, species probing, or species classifiers. Species-ID remains historical context for why the research moved to embedding-based method comparison.

---

## 6. Metric definitions (aligned to Goal 1)

### 6.1 Noise distance (PRIMARY for Goal 1)

**Intent:** Proxy for “how far is this window from a noise-like reference in embedding space?” — closer to the pre-session success story and to SNR intuition than method-label clustering alone.

In code (`embedding_metrics.py` / `cluster_poc` noise hooks):

1. Build or load noise reference embeddings (`noise_*_embeddings.npy` via `process_noise_reference.py`).  
2. Mean noise vector \(n\).  
3. For embeddings of a method: mean \(\cos(x, n)\); report also \(1 - \cos\).  

**Reading (qualitative):**

- **Higher distance to noise** (lower cos to noise mean) ≈ less noise-like in that model’s space.  
- Treat **mono as the fixed baseline**; compare `sa`, `bf_LabIR`, and `bf_SPIR` against it on matched data.
- Report both absolute distance and **Δ versus mono**; positive Δ means farther from noise.
- Always state **which noise group** (noise_mono vs noise_LabIR, etc.).

**Implemented bacpipe rule:** `pipeline_bacpipe.py --models all` re-embeds the same noise WAVs separately for every discovered model and writes model-specific noise files. Cross-model noise vectors are not interchangeable.

### 6.2 Separability gap (SECONDARY diagnostic)

Defined in earlier session:  
\(\mathrm{mean}\cos(x \to c_{\mathrm{own\ method}}) - \mathrm{mean}\cos(x \to c_{\mathrm{other\ methods}})\).

- Useful: “do methods leave different fingerprints?”  
- **Not** the main claim language for “BF improves SNR.”  
- Keep in `embedding_metrics.py`; de-emphasize in papers/talks unless framed as diagnostic.

### 6.3 Direction consistency (SECONDARY)

Az/el structure in LabIR embeddings — interesting for spatial fidelity; not required to state “BF > mono” for Goal 1.

### 6.4 Species confidence (NOT Goal 1 metric)

Archived pipeline only. Contextual for why species-ID failed as BF evaluation.

---

## 7. Paths and environments

| Item | Path |
|------|------|
| Repo | `/Users/ri322/macmini/spatial-ecoacoustic-analysis` |
| Analysis data | `/Volumes/WD2TB/sea-data` |
| Main venv | `venv/` (birdnetlib, metrics, cluster) |
| bacpipe venv | `bacpipe/.venv` |
| bacpipe checkpoints | `bacpipe/checkpoints/` (gitignored) |
| Audits / tables | `…/2A400/embeddings/audits/` |

```bash
# Main
source venv/bin/activate

# bacpipe
source bacpipe/.venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

---

## 8. Code map (what exists after Grok session)

| File | Use for Goal 1? |
|------|------------------|
| `pipeline_signal_processing.py` | Yes — signal-method producer |
| `process_noise_reference.py` | Yes — noise refs |
| `cluster_poc.py` | Yes — visual + noise hooks |
| `embedding_metrics.py` | Yes — but **prioritize noise distance reports** |
| `bacpipe/pipeline_bacpipe.py` | Yes — multi-model early check |
| `embedding_schema.py` / `embedding_io.py` | Support |
| `direction_meta.py` | Support (az/el) |
| `experiments/silent_chunk_fp_audit.py` | Optional context only |
| `run_pipeline.py` etc. | Archive |

---

## 9. Session artefacts (Grok) — status relative to Goal 1

| Artefact | Role now |
|----------|----------|
| `audits/2026-04-21_multi_model_comparison.md` | Exploratory separability; **re-score with noise distance before relying** |
| `embeddings/bacpipe/birdnet|perch_bird/2026-04-21_*` | Bacpipe npy — OK for multi-model experiments if noise protocol added |
| `audits/*silent_fp*` / archive conf stats | Background on species-ID failure; not Goal 1 proof |
| Separability gaps ~0.07 | **Diagnostic only** until noise distance multi-model is run |

---

## 10. Checklist (re-centered)

### Goal 1A — Fixed mono baseline + BirdNET reference

- [x] Pipeline FLAC → mono / sa / bf_* → BirdNET embeddings (pre-session)  
- [x] Noise references + noise distance in cluster workflow (pre-session; user satisfied on limited data)  
- [ ] Freeze `2026-04-26` reference run and document exact plots/numbers  
- [ ] Report `sa`, `bf_LabIR`, and `bf_SPIR` relative to fixed `mono`  
- [ ] Ensure SPIR included whenever array BF is reported

### Goal 1B — All-model bacpipe comparison

- [x] Integrate Bacpipe multi-model pipeline  
- [ ] Discover all models exposed by installed bacpipe version  
- [ ] Re-embed the same noise WAVs separately for every model  
- [ ] Run all feasible models on matched `mono`, `sa`, `bf_LabIR`, `bf_SPIR` WAVs  
- [ ] Produce model × method noise distance and delta-versus-mono table  
- [ ] Decide whether results are consistent, model-dependent, or inconclusive  
- [ ] Scale data only after this protocol is locked

### Goal 1C — Communication

- [ ] Use language about embedding-space noise distance / SNR proxies, not species confidence  
- [ ] Record model failures and protocol limitations alongside successful results  

### Explicitly cancelled / deprioritized

- [x] FP-on-BF silent-chunk campaign as current work  
- [x] Separability gap as primary success metric  
- [x] New monorepo  

---

## 11. Next session in Zed (concrete, small)

1. Use the `2A400 / 2026-04-26` sample as the first reference run.  
2. Run bacpipe with `--models all` and methods `mono,sa,bf_LabIR,bf_SPIR`.  
3. Recompute noise embeddings per model from the same noise WAV set.  
4. Produce `model × method × mean noise distance × Δ versus mono`.  
5. Record unavailable/failed models and only then decide whether more dates are needed.

---

## 12. Understanding check (for future agents / future you)

If a suggestion does not serve this sentence, reject it:

> **Using mono as the fixed baseline, we test whether SA, LabIR beamforming, and SPIR beamforming improve usable signal representations for Indonesian PAM recordings. We measure this primarily with model-specific embedding-space noise distance, first with the existing BirdNET result and then across all feasible models available through bacpipe. Species annotation and probing are outside this workstream.**

---

## 13. Changelog

| Date | Change |
|------|--------|
| 2026-07-30 | Agent scaffolding checklist (bacpipe, schema, FP tools) |
| 2026-07-31 a | Long session log + separability-focused pilot notes |
| 2026-07-31 b | **Realigned to user goals:** Goal 1 = BF SNR/noise-distance story; multi-model bacpipe as early generality check; Goal 2 species deferred; separability demoted; FP-on-BF cancelled; preserve pre-session pipeline success |
| 2026-08-20 | Mono fixed as baseline; SA/LabIR/SPIR are comparators; bacpipe runs all feasible models; raw sample moved to WD2TB; annotation/species probing removed from scope |
