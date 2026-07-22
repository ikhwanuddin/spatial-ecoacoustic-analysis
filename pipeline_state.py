"""
Pipeline State Manager — tracks which processing steps are complete for each FLAC.

Directory structure (v2):
  sea-data/{location}/{date}/
    bf_LabIR/h_23/m_02/    ← chunk WAVs
    bf_SPIR1/h_23/m_02/
    bf_SPIR2/h_23/m_02/
    sa/h_23/m_02/
    mono/h_23/m_02/

Steps:
  bf_{ir_name}       — beamforming + chunk-slicing complete
  birdnet_bf_{ir_name} — BirdNET + processed.json + chunk cleanup done
  sa                  — signal averaging complete
  birdnet_sa          — BirdNET + processed.json on SA done
  mono                — mono baseline complete
  birdnet_mono        — BirdNET + processed.json on mono done
"""

import os
import json
import re
import time
import fcntl
from typing import Dict, List, Optional, Set

from config import ANALYSIS_OUTPUT

STEP_BF_PREFIX = "bf_"
STEP_BIRNET_PREFIX = "birdnet_bf_"
STEP_SA = "sa"
STEP_BIRNET_SA = "birdnet_sa"
STEP_MONO = "mono"
STEP_BIRNET_MONO = "birdnet_mono"

_HM_RE = re.compile(r"^(\d{2})-(\d{2})-\d{2}_dur=")


def _extract_hour_minute(base_name: str) -> tuple:
    m = _HM_RE.match(base_name)
    if m:
        return m.group(1), m.group(2)
    return "00", "00"


class PipelineState:
    """Loads, queries, and updates the pipeline checkpoint log."""

    def __init__(self, state_path: Optional[str] = None):
        self.state_path = state_path or os.path.join(ANALYSIS_OUTPUT, "pipeline_state.json")
        self._data: Dict[str, dict] = {}
        self._dirty = False
        self._load()

    @staticmethod
    def make_key(location_name: str, date_str: str, hour_subdir: str,
                 base_name: str) -> str:
        return f"{location_name}/{date_str}/{hour_subdir}/{base_name}"

    def get_entry(self, key: str) -> dict:
        return self._data.get(key, {})

    def is_complete(self, key: str, step: str) -> bool:
        return self._data.get(key, {}).get(step) == "complete"

    def is_partial(self, key: str, step: str) -> bool:
        return self._data.get(key, {}).get(step) == "partial"

    def mark_pending(self, key: str, step: str):
        self._ensure_entry(key)
        self._data[key][step] = "pending"
        self._dirty = True

    def mark_partial(self, key: str, step: str):
        self._ensure_entry(key)
        self._data[key][step] = "partial"
        self._dirty = True

    def mark_complete(self, key: str, step: str):
        self._ensure_entry(key)
        self._data[key][step] = "complete"
        self._data[key]["last_updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self._dirty = True
        self._save()

    def get_pending_steps(self, key: str, expected_steps: List[str]) -> List[str]:
        entry = self._data.get(key, {})
        return [s for s in expected_steps if entry.get(s) != "complete"]

    def has_any_complete(self, key: str) -> bool:
        entry = self._data.get(key, {})
        return any(v == "complete" for v in entry.values())

    # ---------------------------------------------------------------
    # Auto-detection from disk (v2 paths)
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
        key = self.make_key(location_name, date_str, hour_subdir, base_name)
        if self.has_any_complete(key):
            return set()

        hour_str, minute_str = _extract_hour_minute(base_name)
        completed: Set[str] = set()

        for ir_name in ir_types:
            bf_dir = os.path.join(
                ANALYSIS_OUTPUT, location_name, date_str,
                f"bf_{ir_name}", f"h_{hour_str}", f"m_{minute_str}",
            )
            if os.path.isdir(bf_dir):
                results_json = os.path.join(bf_dir, "results.json")
                processed_json = os.path.join(bf_dir, "processed.json")
                try:
                    chunks = [f for f in os.listdir(bf_dir)
                              if f.startswith("s_") and f.endswith(".wav")
                              and base_name in f]
                except OSError:
                    chunks = []
                if chunks:
                    bf_step = f"{STEP_BF_PREFIX}{ir_name}"
                    # Only mark partial if processed.json not yet done
                    if os.path.isfile(processed_json):
                        self.mark_complete(key, bf_step)
                    else:
                        self.mark_partial(key, bf_step)
                    completed.add(bf_step)
                if run_birdnet and os.path.isfile(results_json) and os.path.isfile(processed_json):
                    bn_step = f"{STEP_BIRNET_PREFIX}{ir_name}"
                    self.mark_complete(key, bn_step)
                    completed.add(bn_step)

        # SA
        sa_dir = os.path.join(
            ANALYSIS_OUTPUT, location_name, date_str,
            "sa", f"h_{hour_str}", f"m_{minute_str}",
        )
        sa_file = os.path.join(sa_dir, f"{base_name}_sa.wav")
        if run_sa and os.path.isfile(sa_file):
            self.mark_complete(key, STEP_SA)
            completed.add(STEP_SA)
            if run_birdnet:
                sa_results = os.path.join(sa_dir, "results.json")
                sa_processed = os.path.join(sa_dir, "processed.json")
                if os.path.isfile(sa_results) and os.path.isfile(sa_processed):
                    self.mark_complete(key, STEP_BIRNET_SA)
                    completed.add(STEP_BIRNET_SA)

        # Mono
        mono_dir = os.path.join(
            ANALYSIS_OUTPUT, location_name, date_str,
            "mono", f"h_{hour_str}", f"m_{minute_str}",
        )
        mono_file = os.path.join(mono_dir, f"{base_name}_mono.wav")
        if os.path.isfile(mono_file):
            self.mark_complete(key, STEP_MONO)
            completed.add(STEP_MONO)
            if run_birdnet:
                mono_results = os.path.join(mono_dir, "results.json")
                mono_processed = os.path.join(mono_dir, "processed.json")
                if os.path.isfile(mono_results) and os.path.isfile(mono_processed):
                    self.mark_complete(key, STEP_BIRNET_MONO)
                    completed.add(STEP_BIRNET_MONO)

        return completed

    # ---------------------------------------------------------------
    # Reporting
    # ---------------------------------------------------------------

    def summary(self) -> str:
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
        if key is None or key == "all":
            self._data = {}
        elif key in self._data:
            del self._data[key]
        else:
            print(f"(!) Key not found: {key}")
            return
        self._save()

    def save(self):
        self._save()

    # ---------------------------------------------------------------
    # Internal
    # ---------------------------------------------------------------

    def _ensure_entry(self, key: str):
        if key not in self._data:
            self._data[key] = {}

    def _load(self):
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
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        try:
            with open(self.state_path, "w") as f:
                _locked = False
                try:
                    fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    _locked = True
                except (IOError, OSError):
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
