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

**Show that beamforming (BF) improves signal quality for passive acoustic monitoring vs monochannel and signal averaging**, on multi-mic MAARU recordings from **Way Canguk, Lampung, Indonesia**.

Operational meaning of “BF better”:

| Era | Evidence used | Status |
|-----|----------------|--------|
| Earlier | BirdNET species conf / “more trusted species IDs” under BF | **Abandoned as primary metric** — Indonesian soundscape poorly matched to BirdNET species head |
| Current (pre–Grok session) | **Noise reference embeddings + noise distance** on dense BirdNET embeddings (clusters / per-method stats) | **Main success path so far** — user already sees promising results on limited hours of data: BF better than mono/SA |
| Next (early check) | Same *kind* of claim, tested across **multiple embedding models via bacpipe** (esp. **Perch**, plus other models as needed) | Does “BF farther from noise / cleaner structure than mono/SA” hold **beyond BirdNET**? |

**Why this matters:** Mic-array + beamforming is only scientifically defensible for PAM if the gain is not an artefact of one classifier’s species head.

**What Goal 1 is *not*:**

- Not “train a Way Canguk species detector first”
- Not “maximise separability gap of method labels” as the main story (that was an agent-side exploratory metric; secondary only)
- Not abandoning noise-distance work in favour of a new metric zoo

### Goal 2 — Secondary (species ID)

Eventually link processing → **species-level** ecology.

- Expected to need **heavy separate work**: annotation, labelling, local experts / community knowledge at Way Canguk.
- **Do not block Goal 1** on Goal 2.
- Tools (bacpipe probing, small classifiers, 249-species list) stay on the shelf until Goal 1 multi-model story is clearer.

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
  → dense BirdNET embeddings (sliding windows)
  → cluster_poc / noise references / noise distance
```

Entry points (active):

| Script | Role |
|--------|------|
| `pipeline_embeddings.py` | BF + SA + mono + native BirdNET dense embeddings |
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
| Raw FLACs | `/Volumes/HD Data/monitoring_data` (env `MONITORING_DATA`) |

---

## 3. Alignment of Grok session vs your goals (honest audit)

### What fitted Goal 1

| Item | Fit |
|------|-----|
| Keep one repo; bacpipe as adapter not rewrite | Good |
| Archive species-ID as primary BF score | Good |
| bacpipe pilot mono/SA/BF + Perch | Good *as early multi-model check* |
| Preserve native BirdNET embedding path | Good |

### What drifted or needs re-centering

| Item | Issue | Correction going forward |
|------|--------|---------------------------|
| **Separability gap** as headline metric | Answers “do methods look different?” more than “is BF cleaner / higher SNR-like?” | Keep code optional; **do not treat as main claim**. Prefer **noise distance** and related SNR-proxy metrics you already trusted |
| Heavy FP audit tooling | Useful context; easy to steal focus | **No further FP-on-BF work** unless you reopen it |
| Default out path `embeddings/birdnet/` | Schema OK; most existing data still **flat** under `embeddings/*.npy` | Loaders must keep supporting **legacy flat** layout |
| bacpipe pilot tiny (2 WAV/method) | Fine for smoke, not for Goal 1 conclusion | Scale only after metrics match **your** success criteria (noise distance first) |
| Direction gap on 2 azimuths | Exploratory | Secondary to mono vs SA vs BF noise distance |

### What you already found satisfactory (preserve)

On limited hours of data, with **BirdNET embeddings + noise references + noise distance**:

- BF looks **better** than mono/SA  
- Array PAM looks **promising**  

**Any new multi-model work should try to reproduce that *kind* of conclusion**, not replace it with an unrelated metric narrative.

---

## 4. Goal 1 — planned workstreams (order)

### 4.1 Freeze and document the “happy path” (BirdNET + noise)

Priority: so Zed work always has a baseline you already believe.

1. Commands to regenerate/load embeddings for mono, sa, bf_LabIR, bf_SPIR on a chosen date set.  
2. How noise references were built (`process_noise_reference.py`).  
3. Exactly how noise distance is computed in `cluster_poc` / `embedding_metrics` (cosine to noise mean; distance = 1 − cos).  
4. Which plots/HTML/tables convinced you BF wins.

**Deliverable:** short subsection or link from this file once you freeze a “reference run” date list in Zed (edit this section).

**Reference run (fill in when frozen):**

- Location: `2A400` (default site in session)  
- Dates: _TBD by you_  
- Metrics: noise distance by method; optional cluster structure  

### 4.2 Early multi-model check via bacpipe (your “selanjutnya”)

**Question:** On the **same WAVs** (mono / sa / bf_*), do **other embedding models** (especially **Perch**, then others) still show BF “better” under the **same noise-distance (or agreed SNR-proxy) logic**?

Implementation already partly scaffolded (session):

| Path | Role |
|------|------|
| `experiments/bacpipe/run_pilot.py` | Embed existing method WAVs with bacpipe models |
| `experiments/bacpipe/.venv` | Isolated env (bacpipe 1.3.3) |
| `embeddings/bacpipe/{model}/` | Pilot npy outputs under sea-data |
| `direction_meta.py` | az/el parse without TF |

**Rules for this workstream:**

1. **Do not re-run beamforming** unless audio missing — reuse method WAVs.  
2. **Primary comparison metric = noise distance** (or the same definition you used pre-session), applied **per model**.  
3. Separability gap / UMAP = supporting diagnostics only.  
4. Start models: `perch_bird` (and/or `perch_v2` if available), keep `birdnet` via bacpipe only as control vs native.  
5. Expand model list only if it answers: “how model-agnostic is the BF gain?”  
6. Species heads / probing = Goal 2, not here.

**Success criterion (early check):**

- For ≥1 non-BirdNET model (ideally Perch): BF shows **higher distance from noise** (or your agreed “cleaner” direction) than mono and preferably than SA, on a matched subset.  
- Or document **failure**: BF gain is BirdNET-specific → important negative result for Goal 1 generality.

### 4.3 Scale (only after 4.2 metric protocol is agreed)

- More minutes / dates / LabIR directions / SPIR.  
- Matched event windows (same time mono vs BF) if noise-distance alone is too coarse.  
- Multi-date table: method × model × noise distance.

### 4.4 Optional later (not Goal 1 blockers)

- bacpipe official Panel dashboard wired to our layout  
- `--backend bacpipe` inside `pipeline_embeddings.py`  
- Folder cleanup of archive scripts  

---

## 5. Goal 2 — species ID (deferred)

**When to start:** after Goal 1 multi-model noise-distance story is clear enough for a paper section / thesis chapter outline.

**Hard requirements (expected):**

- Annotation protocol  
- Local expertise (Way Canguk / birders)  
- Subset of common species first, not 249 at once  
- Possibly bacpipe linear probing / Burooj-style small classifiers on embeddings that survived Goal 1  

**Do not** spend agent cycles on full taxonomy now.

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
- Compare **bf_* vs mono vs sa** on matched data.  
- Always state **which noise group** (noise_mono vs noise_LabIR, etc.).

**Gap:** Pilot bacpipe folders under `embeddings/bacpipe/*` did **not** automatically include noise files — when scoring Perch/BirdNET bacpipe runs, **reuse legacy noise refs or recompute noise embeddings in that model’s space**. Prefer **noise embeddings from the same model** when comparing models; cross-model noise vectors are not interchangeable.

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
| bacpipe venv | `experiments/bacpipe/.venv` |
| bacpipe checkpoints | `experiments/bacpipe/checkpoints/` (gitignored) |
| Audits / tables | `…/2A400/embeddings/audits/` |

```bash
# Main
source venv/bin/activate

# bacpipe
source experiments/bacpipe/.venv/bin/activate
export PYTHONPATH="$(pwd):$PYTHONPATH"
```

---

## 8. Code map (what exists after Grok session)

| File | Use for Goal 1? |
|------|------------------|
| `pipeline_embeddings.py` | Yes — native pipeline |
| `process_noise_reference.py` | Yes — noise refs |
| `cluster_poc.py` | Yes — visual + noise hooks |
| `embedding_metrics.py` | Yes — but **prioritize noise distance reports** |
| `experiments/bacpipe/run_pilot.py` | Yes — multi-model early check |
| `embedding_schema.py` / `embedding_io.py` | Support |
| `direction_meta.py` | Support (az/el) |
| `experiments/silent_chunk_fp_audit.py` | Optional context only |
| `run_pipeline.py` etc. | Archive |

---

## 9. Session artefacts (Grok) — status relative to Goal 1

| Artefact | Role now |
|----------|----------|
| `audits/2026-04-21_multi_model_comparison.md` | Exploratory separability; **re-score with noise distance before relying** |
| `embeddings/bacpipe/birdnet|perch_bird/2026-04-21_*` | Pilot npy — OK for multi-model experiments if noise protocol added |
| `audits/*silent_fp*` / archive conf stats | Background on species-ID failure; not Goal 1 proof |
| Separability gaps ~0.07 | **Diagnostic only** until noise distance multi-model is run |

---

## 10. Checklist (re-centered)

### Goal 1A — BirdNET + noise (baseline you trust)

- [x] Pipeline FLAC → mono / sa / bf_* → BirdNET embeddings (pre-session)  
- [x] Noise references + noise distance in cluster workflow (pre-session; user satisfied on limited data)  
- [ ] Freeze “reference run” dates + document exact plots/numbers in this file  
- [ ] Ensure SPIR included whenever LabIR is claimed for “array BF” generally  

### Goal 1B — Multi-model early check (bacpipe)

- [x] Scaffold bacpipe pilot + Perch smoke  
- [ ] **Define protocol:** same WAVs, noise embeddings **per model**, distance mono vs sa vs bf  
- [ ] Run protocol on Perch (and BirdNET-via-bacpipe as control)  
- [ ] Decide: BF gain model-agnostic / model-specific / inconclusive  
- [ ] Optionally add 1–2 other bacpipe models if story needs breadth  
- [ ] Scale data only after protocol is locked  

### Goal 1C — Communication

- [ ] Thesis/paper language: BF improves **embedding-space distance from noise / SNR-proxy**, not species conf  
- [ ] Email updates (Vincent et al.): multi-model check as next step; species ID later  

### Goal 2 — Species

- [ ] Deferred: annotation + local experts + probing  

### Explicitly cancelled / deprioritized

- [x] FP-on-BF silent-chunk campaign as current work  
- [x] Separability gap as primary success metric  
- [x] New monorepo  

---

## 11. Next session in Zed (concrete, small)

1. Re-open **your** best pre-session noise-distance result (HTML/cache/plots) and write 5 lines into §4.1 “reference run”.  
2. Specify noise protocol for multi-model:  
   - Option A: recompute noise embeddings with each bacpipe model on the same noise WAVs  
   - Option B: only compare models that can share a defined noise set (document limitation)  
3. Run bacpipe pilot on a **matched** mono/sa/bf set large enough for noise distance (not only 2 files if you want a real check).  
4. Table: `model × method × mean noise distance`.  
5. Only then expand Perch / other models or more dates.

---

## 12. Understanding check (for future agents / future you)

If a suggestion does not serve this sentence, reject it:

> **Beamforming on the MAARU array improves usable signal quality vs mono/SA for Indonesian PAM recordings; we measure that primarily via embedding-space noise distance (and related SNR proxies), first with BirdNET embeddings we already trust, then by checking generality across models (especially Perch) with bacpipe. Species ID is a later, separate investment.**

---

## 13. Changelog

| Date | Change |
|------|--------|
| 2026-07-30 | Agent scaffolding checklist (bacpipe, schema, FP tools) |
| 2026-07-31 a | Long session log + separability-focused pilot notes |
| 2026-07-31 b | **Realigned to user goals:** Goal 1 = BF SNR/noise-distance story; multi-model bacpipe as early generality check; Goal 2 species deferred; separability demoted; FP-on-BF cancelled; preserve pre-session pipeline success |
