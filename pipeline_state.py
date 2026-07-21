"""
Pipeline State Manager — tracks which processing steps are complete for each FLAC.

Stores a pipeline_state.json on the output volume (WD2TB). On every run,
run_pipeline.py consults this log to skip already-completed steps, enabling
graceful resume after intentional or accidental interruption.

Each entry is keyed by {location}/{date}/{hour}/{base_name} — derived from
the FLAC path and output directory structure.

Two-layer detection:
  1. pipeline_state.json (fast, authoritative cache)
  2. Auto-detection from disk (fallback — checks actual file existence)

State values:
  "pending"  — not yet started
  "partial"  — started but some outputs missing (interrupted mid-step)
  "complete" — all outputs verified on disk
"""

import os
import json
import time
import fcntl
from typing import Dict, List, Optional, Set

from config import ANALYSIS_OUTPUT


# Steps within the pipeline
STEP_BF_PREFIX = "bf_"
STEP_BIRNET_PREFIX = "birdnet_bf_"
STEP_SA = "sa"
STEP_BIRNET_SA = "birdnet_sa"
STEP_MONO = "mono"
STEP_BIRNET_MONO = "birdnet_mono"


def _build_key(location_name: str, date_str: str, hour_subdir: str, base_name: str) -> str:
    """Build the state-key for a single FLAC file."""
    return f"{location_name}/{date_str}/{hour_subdir}/{base_name}"


class PipelineState:
    """
    Loads, queries, and updates the pipeline checkpoint log.

    Thread-safe writes via fcntl file locking (single-process use, but
    future-proof for any concurrency).

    Usage:
        state = PipelineState()
        key = state.make_key(location, date, hour, base_name)

        if not state.is_complete(key, "bf_LabIR"):
            # ... run beamforming LabIR ...
            state.mark_complete(key, "bf_LabIR")
    """

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or os.path.join(ANALYSIS_OUTPUT, "pipeline_state.json")
        self._data: Dict[str, dict] = {}
        self._dirty = False
        self._load()

    # ---------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------

    @staticmethod
    def make_key(location_name: str, date_str: str, hour_subdir: str, base_name: str) -> str:
        """Build the canonical key for a FLAC file's state entry."""
        return _build_key(location_name, date_str, hour_subdir, base_name)

    def get_entry(self, key: str) -> dict:
        """Return the full state dict for a key, or empty dict."""
        return self._data.get(key, {})

    def is_complete(self, key: str, step: str) -> bool:
        """Check if a specific step is marked complete."""
        return self._data.get(key, {}).get(step) == "complete"

    def is_partial(self, key: str, step: str) -> bool:
        """Check if a step was partially done (interrupted)."""
        return self._data.get(key, {}).get(step) == "partial"

    def mark_pending(self, key: str, step: str):
        """Mark a step as pending (not yet started)."""
        self._ensure_entry(key)
        self._data[key][step] = "pending"
        self._dirty = True

    def mark_partial(self, key: str, step: str):
        """Mark a step as partial (in-progress / interrupted)."""
        self._ensure_entry(key)
        self._data[key][step] = "partial"
        self._dirty = True

    def mark_complete(self, key: str, step: str):
        """Mark a step as complete and persist immediately."""
        self._ensure_entry(key)
        self._data[key][step] = "complete"
        self._data[key]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._dirty = True
        self._save()

    def get_pending_steps(self, key: str, expected_steps: List[str]) -> List[str]:
        """Return the subset of expected_steps that are NOT complete."""
        entry = self._data.get(key, {})
        return [s for s in expected_steps if entry.get(s) != "complete"]

    def has_any_complete(self, key: str) -> bool:
        """Return True if the entry exists and has at least one completed step."""
        entry = self._data.get(key, {})
        return any(v == "complete" for v in entry.values())

    # ---------------------------------------------------------------
    # Auto-detection from disk
    # ---------------------------------------------------------------

    def auto_detect_from_disk(
        self,
        location_name: str,
        date_str: str,
        hour_subdir: str,
        base_name: str,
        ir_types: List[str],
        run_sa: bool = True,
        run_birdnet: bool = True,
    ) -> Set[str]:
        """
        Scan output directories to detect which steps are already done.

        Heuristic:
          - BF:  check if any .wav files exist (partial) plus results.json,
                 processed.json (BirdNET done). Full completeness is verified
                 by process_one_flac via _beamforming_complete.
          - SA:  check if {base_name}_sa.wav exists.
          - BN:  check if results.json + processed.json exist.

        Only fills in gaps where pipeline_state.json has no entry.

        Returns set of newly-completed step strings.
        """
        key = self.make_key(location_name, date_str, hour_subdir, base_name)

        # Don't overwrite existing complete entries
        if self.has_any_complete(key):
            return set()

        completed: Set[str] = set()

        for ir_name in ir_types:
            bf_dir = os.path.join(
                ANALYSIS_OUTPUT, location_name, date_str,
                f"beamforming_{ir_name}", hour_subdir,
            )
            if os.path.isdir(bf_dir):
                results_json = os.path.join(bf_dir, f"results_{base_name}.json")
                processed_json = os.path.join(bf_dir, f"processed_{base_name}.json")
                try:
                    wavs = [f for f in os.listdir(bf_dir) if f.endswith(".wav") and f.startswith(base_name)]
                except OSError:
                    wavs = []
                if wavs:
                    bf_step = f"{STEP_BF_PREFIX}{ir_name}"
                    self.mark_partial(key, bf_step)
                    completed.add(bf_step)
                if run_birdnet and os.path.isfile(results_json) and os.path.isfile(processed_json):
                    bn_step = f"{STEP_BIRNET_PREFIX}{ir_name}"
                    self.mark_complete(key, bn_step)
                    completed.add(bn_step)

        # Signal averaging
        sa_dir = os.path.join(
            ANALYSIS_OUTPUT, location_name, date_str,
            "signal_averaging", hour_subdir,
        )
        if run_sa and os.path.isdir(sa_dir) and os.path.isfile(os.path.join(sa_dir, f"{base_name}_sa.wav")):
            self.mark_complete(key, STEP_SA)
            completed.add(STEP_SA)
            if run_birdnet:
                sa_results = os.path.join(sa_dir, f"results_{base_name}.json")
                sa_processed = os.path.join(sa_dir, f"processed_{base_name}.json")
                if os.path.isfile(sa_results) and os.path.isfile(sa_processed):
                    self.mark_complete(key, STEP_BIRNET_SA)
                    completed.add(STEP_BIRNET_SA)

        # Monochannel baseline
        mono_dir = os.path.join(
            ANALYSIS_OUTPUT, location_name, date_str,
            "mono_baseline", hour_subdir,
        )
        mono_file = os.path.join(mono_dir, f"{base_name}_mono.wav")
        if os.path.isfile(mono_file):
            self.mark_complete(key, STEP_MONO)
            completed.add(STEP_MONO)
            if run_birdnet:
                mono_results = os.path.join(mono_dir, f"results_{base_name}.json")
                mono_processed = os.path.join(mono_dir, f"processed_{base_name}.json")
                if os.path.isfile(mono_results) and os.path.isfile(mono_processed):
                    self.mark_complete(key, STEP_BIRNET_MONO)
                    completed.add(STEP_BIRNET_MONO)

        return completed

    # ---------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------

    def summary(self) -> str:
        """Return a one-line human-readable summary."""
        total = len(self._data)
        if total == 0:
            return "No entries in pipeline state."

        fully_complete = 0
        partial = 0
        pending_only = 0

        for entry in self._data.values():
            statuses = {k: v for k, v in entry.items() if k != "last_updated"}
            if not statuses:
                pending_only += 1
            elif all(v == "complete" for v in statuses.values()):
                fully_complete += 1
            else:
                partial += 1

        return (
            f"Pipeline State: {total} entries "
            f"({fully_complete} fully complete, {partial} partial, {pending_only} pending)"
        )

    def detailed_summary(self) -> str:
        """Return a table-formatted summary with every entry."""
        if not self._data:
            return "No entries in pipeline state."

        lines = []
        for key in sorted(self._data.keys()):
            entry = self._data[key]
            statuses = {k: v for k, v in entry.items() if k != "last_updated"}
            status_str = ", ".join(f"{k}={v}" for k, v in sorted(statuses.items()))
            updated = entry.get("last_updated", "")
            lines.append(f"  {key}")
            lines.append(f"    -> {status_str}")
            if updated:
                lines.append(f"    updated: {updated}")
        return "\n".join(lines)

    def clean_stale_keys(self, stale_days: int = 7):
        """Remove entries older than stale_days (based on last_updated)."""
        import time as _time
        cutoff = _time.time() - stale_days * 86400
        stale_keys = []
        for key, entry in self._data.items():
            updated = entry.get("last_updated", "")
            if updated:
                try:
                    ts = _time.mktime(_time.strptime(updated, "%Y-%m-%dT%H:%M:%S"))
                    if ts < cutoff:
                        stale_keys.append(key)
                except (ValueError, OverflowError):
                    pass
        for key in stale_keys:
            del self._data[key]
        if stale_keys:
            self._save()

    def reset_key(self, key: Optional[str] = None):
        """Remove a specific key or clear all state."""
        if key is None or key == "all":
            self._data = {}
        elif key in self._data:
            del self._data[key]
        else:
            print(f"(!) Key not found: {key}")
            return
        self._save()

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _ensure_entry(self, key: str):
        if key not in self._data:
            self._data[key] = {}

    def _load(self):
        """Load state from JSON, gracefully handling missing/corrupt files."""
        if not os.path.isfile(self.state_path):
            self._data = {}
            return

        try:
            with open(self.state_path, "r") as f:
                self._data = json.load(f)
            if not isinstance(self._data, dict):
                self._data = {}
        except (json.JSONDecodeError, IOError):
            print(f"(!) Could not parse {self.state_path} - starting fresh")
            self._data = {}

    def _save(self):
        """Write state to JSON with optional file locking.

        Attempts fcntl.flock for local filesystem safety.
        Falls back to lock-free write on network mounts (SMB/NFS)
        where flock is not supported.
        """
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)

        try:
            with open(self.state_path, "w") as f:
                _locked = False
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _locked = True
                except (IOError, OSError):
                    # flock not supported on this filesystem (SMB, NFS, etc.)
                    pass
                try:
                    json.dump(self._data, f, indent=2, ensure_ascii=False)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    if _locked:
                        try:
                            fcntl.flock(f, fcntl.LOCK_UN)
                        except (IOError, OSError):
                            pass
            self._dirty = False
        except IOError as e:
            print(f"(!) Could not save pipeline state: {e}")

    def save(self):
        """Public save (for explicit flush)."""
        self._save()
