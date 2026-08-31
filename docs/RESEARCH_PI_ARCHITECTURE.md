# Research-specific Pi Agent system

## Architecture and minimum viable specification

**Status:** Draft after Mac mini environment inspection  
**Date:** 2026-08-16  
**Scope:** Pi Agent as research collaborator, with the current spatial ecoacoustic pipeline as the first implementation target.

This document specifies the smallest useful research-oriented layer. It does not install an extension, alter scientific criteria, or decide the final metric for the PhD.

---

## 1. Design principles

1. **Pi is the coordinator and research co-worker.** It prepares tasks, gathers evidence, delegates bounded work, checks results, and prepares decisions for Rifqi.
2. **Rifqi retains scientific authority.** Pi must not invent inclusion/exclusion criteria, silently resolve conflicts, select a final interpretation, or promote an exploratory metric into a research claim.
3. **Small tasks are the execution unit.** Each task should have a bounded scope, explicit inputs, expected evidence, stop conditions, and a handoff.
4. **Evidence before interpretation.** Technical observations, computed measurements, methodological reasoning, and scientific interpretation must be separated in outputs.
5. **Reuse before adding infrastructure.** The current Pi/Archimedes environment is the baseline. New extensions require a separate evaluation and user approval.
6. **Full research state stays searchable and archived.** Active task context is a compact pointer-rich packet, not a copy of the entire vault, repository, or chat history.
7. **Research is dynamically exploratory.** Candidate metrics and analyses are hypotheses/evidence sources until Rifqi approves their role. Historical experiments remain part of the research record rather than being silently rewritten.

---

## 2. Inspected environment

### Pi

- Pi Agent: `0.84.2`
- Main executable: `/opt/homebrew/bin/pi`
- Current user configuration: `/Users/ri322/.pi/agent/settings.json`
- Current default model: `github-copilot/gpt-5.6-luna`
- Current user packages:
  - `npm:pi-archimedes` `2.2.0`
  - `npm:pi-web-access` `0.23.0`
  - `npm:pi-context-view` `0.4.2`
  - `npm:pi-markdown-preview` `0.14.1`
  - `npm:pi-ghostty` `1.0.0`
  - diagram-design package
- `pi-subagents` is **not currently installed**.

### Existing Archimedes agents

User-level definitions in `/Users/ri322/.pi/agent/agents/`:

- `scout` — read-only codebase reconnaissance
- `planner` — read-only implementation planning
- `worker` — implementation
- `code-reviewer` — read-only code review
- `researcher` — web/local research with sources
- `phd-research-manager` — user-level PhD research-state management, prioritisation, evidence/provenance checks, and decision packages

The new manager is installed at `/Users/ri322/.pi/agent/agents/phd-research-manager.md` and is read-only by default.

Archimedes currently provides:

- single subagent dispatch;
- parallel subagent dispatch;
- per-agent model and tool configuration;
- agent discovery from user/project scopes;
- live progress and cost display;
- `ask` bridge from children to the parent TUI;
- todo tracking.

Archimedes does **not** currently provide the research-specific contract, risk governance, compact research-state handoff, evidence acceptance protocol, durable mission state, or research audit behaviour required by this project.

### Acoustic-analysis repository

Repository: `/Users/ri322/macmini/spatial-ecoacoustic-analysis`

Current Git state at inspection:

- branch: `main`
- HEAD: `acdb307 Fix BirdNET threading & logs; archive species-ID pipeline`
- one unrelated untracked file: `species_lists/trivia.md`

Active pipeline entry points:

- `pipeline_signal_processing.py`: FLAC → beamforming + signal averaging + mono signal methods;
- `cluster_poc.py`: clustering/dashboard and related diagnostics;
- `process_noise_reference.py`: noise-reference embeddings;
- `extract_embeddings.py`: embedding-only extraction from existing WAVs;
- `bacpipe/pipeline_bacpipe.py`: multi-model embeddings, including `perch_bird`, from existing method WAVs without re-running beamforming;
- `embedding_metrics.py`: method/direction/noise-distance calculations.

Archived path:

- `run_pipeline.py`, `birdnet_processor.py`, `generate_report.py`, and related pre-filter/species-detection code are retained for historical runs and optional audits, not as the current scientific spine.

### Runtime and data

- Main pipeline virtual environment: `spatial-ecoacoustic-analysis/venv/`
  - Python 3.11.15
  - NumPy, SciPy, librosa, soundfile, BirdNET, and TensorFlow available
  - bacpipe not installed there
- Multi-model environment: `spatial-ecoacoustic-analysis/bacpipe/.venv/`
  - Python 3.11.15
  - bacpipe available
- Observed data roots:
  - raw recordings: `/Volumes/HD Data/monitoring_data`
  - analysis outputs: `/Volumes/WD2TB/sea-data`
  - impulse responses: `/Users/ri322/macmini/MAARU-Impulse-Response`
- Existing `2A400` outputs include native BirdNET embeddings, noise references, audits, and a Bacpipe integration report.

No assumption is made here about the Zotero location, Imperial HPC configuration, or the final Way Canguk validation strategy; those remain to be inspected or specified when a task needs them.

---

## 3. Capability comparison

| Requirement | Current Pi + Archimedes | `pi-subagents` candidate | MVP decision |
|---|---|---|---|
| Main Pi coordinator | Yes | Yes | Keep Pi as parent |
| Reusable + task-specific delegation | Basic agent files and per-call task text | Stronger agent management and workflow scripts | Start with existing agents; add research task contracts |
| Parallel work | Yes | Yes | Use only for independent bounded tasks |
| Sequential/chained work | Manual parent turns | `workflowScript` with `runs.run` | Evaluate later |
| Dynamic fanout/branching | Limited to parent choosing tasks | Supported and bounded | Not needed for first audit |
| Durable background runs | Not a current research-state mechanism | Missions, retained children, schedules, status/control | Candidate for later, not installed yet |
| Worktree isolation | No Archimedes worktree mechanism | Supported | Useful for code changes, not needed for read-only audit |
| Human escalation | `ask` bridge | `ask` plus `contact_supervisor`/steering | Existing `ask` is enough for MVP |
| Acceptance/evidence gates | Not built into current subagent contract | Structured acceptance and verification gates | Implement as task output contract first |
| Resource/loop controls | Parent discipline only | Runtime time/tool/turn/spawn controls | Add explicit stop rules now; evaluate runtime controls later |
| Research-state persistence | Obsidian/Git handled manually | Mission state is useful but not a replacement for Obsidian | Keep Obsidian canonical; use mission state only if later adopted |

### Current orchestration decision

`pi-archimedes` is sufficient as the current delegation foundation. It already provides agent discovery, single/parallel dispatch, per-agent tools/models, live progress, cost tracking, child-to-parent questions, and todo visibility.

`pi-subagents` is therefore **not needed for the current system** and will not be installed. Its additional workflow features are not required to start the research-specific layer. If a future verified requirement cannot be met by Archimedes, that decision can be reopened explicitly; it is not part of the MVP.

---

## 4. Research-state and context handoff

### Source-of-truth roles

- **Obsidian:** research question, reasoning, literature, scientific decisions, interpretations, important experiment metadata, unresolved questions, and reporting state.
- **Git:** code, technical configuration, reproducible scripts, and code provenance.
- **Analysis output volume:** generated arrays, reports, dashboards, logs, and run artefacts.
- **Zotero:** approved literature library and bibliographic records.

The compact task context is a **derived handoff**, not a second scientific source of truth.

### Context packet contract

Every non-trivial research task should receive a compact packet with these headings:

```markdown
# Task context

## Research goal and scope
## Scientific constraints and user decisions
## Current state
## Evidence already available
## Task-specific inputs and exact paths
## What Pi may change
## What Pi must not decide
## Expected checks and outputs
## Stop/escalate conditions
## Open questions
## Source pointers
```

The packet should contain facts and links/paths to full evidence, not long pasted transcripts. It should normally remain far below the user's active-context ceiling; no task is allowed to expand context merely because more history is available.

### Handoff output

Each completed task returns:

1. **Observed facts** — commands, paths, versions, counts, and direct results.
2. **Evidence artefacts** — reports, tables, logs, plots, or code changes.
3. **Reasoning** — why the observations matter, clearly labelled as reasoning.
4. **Uncertainty/conflicts** — missing data, incompatible sources, or alternative explanations.
5. **Decision request** — only when user judgement is required.
6. **Next ready task** — one small, concrete action, or an explicit blocker.

Full handoffs may be archived in Obsidian AI Sessions or experiment notes. The active packet should contain only the compact summary and pointers.

---

## 5. Risk and authority model

| Level | Examples | Default action |
|---|---|---|
| R0: read-only factual | Inspect files, Git status, metadata, existing reports, dry-run file discovery | Pi may complete and report |
| R1: bounded technical | Compile checks, small sample runs, reversible local code edits, output-schema checks | Pi may do within exact user-provided scope; report changes |
| R2: consequential technical | Re-running model inference, downloading checkpoints, changing pipeline defaults, writing important experiment artefacts | Prepare a task proposal; require explicit approval if resource/output impact is material |
| R3: scientific | Inclusion/exclusion criteria, ground-truth design, final metric role, interpretation of conflicting results, research direction, manuscript claims | Always escalate with a decision package |
| R4: external/high-cost | Full dataset, HPC jobs, large model downloads, publication/report submission, external communication | Require explicit scope, resource, and approval before execution |

A lower-risk technical task must not smuggle in an R3 decision. For example, implementing a metric is technical; declaring that it is the primary evidence for the PhD is scientific.

### Decision package

When escalation is needed, Pi must provide:

- problem;
- evidence and source paths;
- reasoning;
- at least two viable options where they exist;
- pros and cons;
- recommendation labelled as a recommendation, not a decision;
- consequences;
- exact next actions;
- what remains uncertain.

If sources conflict, Pi reports the conflict and asks Rifqi. It must not select a winner silently.

---

## 6. Metric handling for the current acoustic work

The architecture must not hard-code a final primary metric at this stage.

Noise distance and method separability are related but not identical measurements:

- **Noise distance:** how far a method's windows are from a defined noise reference in embedding space.
- **Method separability:** how distinguishable samples/method centroids are from one another in embedding space.

Both use notions of similarity/distance, and both may contribute to a broader question about signal structure or quality. However, they answer different questions. A method can be far from a noise reference while methods remain overlapping; alternatively, processing can make methods separable for reasons unrelated to improved signal quality. Therefore:

1. treat both as candidate evidence sources;
2. record their definitions, reference sets, sampling/matching, model, and uncertainty;
3. compare them rather than silently collapsing them into one metric;
4. preserve historical exploratory analyses as research history;
5. let Rifqi decide when a metric becomes primary, supporting, or rejected.

The orchestration layer should describe metric tasks as **measurement/evidence evaluation**, not as “optimise the metric”.

For any multi-model task, measurements must be kept within the same embedding model/backend. Noise vectors from one model must not be treated as interchangeable with vectors from another model.

---

## 7. Minimum viable research-oriented capability

### First research capability: `phd-research-manager`

This is a user-level Archimedes agent that manages bounded research-state and project-management tasks. It is not an autonomous PhD manager and does not replace the parent Pi coordinator. It can recommend or prepare a `pipeline-audit` task for the parent to dispatch.

The first domain task capability used by this manager is `pipeline-audit`.

**Input**

- compact research context packet;
- exact repository and data paths;
- date/location/method/model scope;
- explicit allowed resource level;
- requested question.

**Preflight**

- verify paths and virtual environment;
- inspect Git state without changing unrelated work;
- identify code entry points and current schema;
- confirm whether the task is dry-run, bounded run, or full run;
- detect missing or conflicting source-of-truth information.

**Audit dimensions**

1. execution success and reproducibility;
2. output completeness and schema validity;
3. technical correctness of metadata, dimensions, paths, and matching;
4. scientific sensibility checks and edge cases;
5. comparison fairness: same source windows, methods, model/backend, and reference definitions where applicable;
6. resource/runtime and memory risks;
7. provenance: code version, command, environment, input scope, and output locations;
8. uncertainty and alternative explanations.

**Default safety**

- no full 600+ GB run;
- no HPC submission;
- no broadening of data-selection scope;
- no deletion of outputs;
- no model checkpoint download or large inference unless explicitly approved;
- no final scientific interpretation.

**Output**

- concise audit report;
- machine-readable summary when useful;
- list of anomalies and severity;
- evidence pointers;
- one recommended next task;
- decision package if a scientific/resource decision is reached.

### First concrete integration audit

Audit the existing `2026-04-21` multi-model integration and native embedding outputs without re-running inference. Check:

- matched WAV/source-minute coverage across `mono`, `sa`, `bf_LabIR`, and `bf_SPIR`;
- embedding dimensions and metadata consistency;
- whether the Perch/BirdNET integration and native BirdNET outputs are being compared on compatible scopes;
- noise-reference availability and model compatibility;
- definitions and sampling behind the recorded noise-distance and method-separability values;
- missing edge-case checks;
- which next bounded Perch run would reduce uncertainty.

This bounded audit should produce an evidence audit, not a conclusion about whether beamforming is scientifically proven.

---

## 8. Repeatable task lifecycle

```text
load compact context
  → preflight paths, scope, authority, and resources
  → investigate existing evidence
  → run the smallest safe check
  → inspect outputs and edge cases
  → classify uncertainty/risk
  → update handoff and important metadata
  → ask for a decision or propose one next task
```

Loop protection:

- define a maximum number of retries/iterations per task;
- stop when the same failure recurs without new evidence;
- distinguish technical failure from unresolved scientific ambiguity;
- return a failure report rather than silently changing scope;
- never use additional model calls merely to avoid an approval decision.

---

## 9. Adoption plan

### Phase A — use the existing environment

1. Keep Archimedes installed and do not install `pi-subagents` yet.
2. Add a project-scoped research task contract/capability after reviewing this specification.
3. Use the current `researcher`, `scout`, `worker`, and `code-reviewer` only with the compact context and risk rules above.
4. Run the read-only Bacpipe audit.

### Phase B — extend Archimedes with research capabilities

The first implementation is the user-level agent `phd-research-manager`. Its prompt encodes the initial research-management behaviour:

- compact context-packet contract;
- research task brief;
- risk/authority rules;
- evidence-oriented handoff;
- pipeline-audit planning;
- structured decision package;
- loop detection and stop conditions.

Additional domain capabilities can be added as separate prompts or project-scoped agents only when the first manager exposes a concrete gap. No new orchestration extension is required.

### Phase C — expand only after the first bottleneck works

Later capabilities can cover:

- evidence and literature comparison;
- statistical-analysis audit;
- figures/tables with provenance;
- publication/report generation;
- daily and event-driven research-state audits;
- HPC/full-data run management.

They should reuse the same task contract, context packet, risk model, and evidence handoff.

---

## 10. Open decisions that must remain explicit

- exact location and format of the canonical compact research-state packet;
- exact experiment metadata schema in the Obsidian vault;
- exact acceptance threshold for pipeline “scientifically sensible” outputs;
- which matched subset/date(s) are approved for the next Perch run;
- which metric set is appropriate for the next experiment and how it should be interpreted;
- how generalised-lab IR usefulness will eventually be evaluated;
- Way Canguk validation strategy and any ground-truth decision;
- exact Archimedes extension point if the research contracts later need runtime enforcement;
- HPC resource limits and job conventions.

This document intentionally does not resolve those decisions.
