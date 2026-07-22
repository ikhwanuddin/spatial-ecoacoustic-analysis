#!/usr/bin/env python3
"""
Generate a lightweight HTML dashboard from processed.json files.

Pure visualisation — no raw detection rows embedded.
Focus: confidence-score comparison across methods, species × method heatmap,
top species ranking, and hourly activity.

BirdNET analysis keeps its pipeline min_conf (typically 0.4). This dashboard
only visualises detections with confidence > DASHBOARD_MIN_CONF (default 0.55)
to suppress the weak ~0.5 band that dominates raw output.

Usage:
    python generate_report.py --location 2A400
    python generate_report.py --location 2A400 --dates 2026-04-20
    python generate_report.py --location 2A400 --dashboard-min-conf 0.55

Output:
    {data_dir}/{location}/{location}_report.html
    {data_dir}/{location}/report_data/{date}_summary.json   (dashboard-filtered)
    {data_dir}/{location}/report_data/{date}_detections.json.gz  (full ≥ pipeline conf)
"""

import argparse
import gzip
import json
import math
import os
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from html import escape
from typing import Any, Dict, List, Optional, Tuple

try:
    from config import ANALYSIS_OUTPUT
except ImportError:
    ANALYSIS_OUTPUT = "/Volumes/WD2TB/sea-data"

# ============================================================
# LABIR SPEAKER -> ELEVATION MAPPING
# ============================================================
LABIR_ELEVATION: Dict[int, int] = {
    1: -45, 2: -30, 3: -20, 4: -10, 5: 0,
    6: 10, 7: 20, 8: 30, 9: 45, 10: 60,
    11: 75, 12: 90,
}

PROCESSING_DIRS = ["bf_LabIR", "bf_SPIR1", "bf_SPIR2", "sa", "mono"]

METHOD_MAP = {
    "bf_LabIR": "LabIR", "bf_SPIR1": "SPIR1",
    "bf_SPIR2": "SPIR2", "sa": "SA", "mono": "Mono",
}

# Display / legend order: baseline → spatial methods
METHOD_ORDER = ["Mono", "SA", "LabIR", "SPIR1", "SPIR2"]

# Dashboard-only floor (strict >). Pipeline BirdNET min_conf stays at 0.4.
DASHBOARD_MIN_CONF = 0.55


def _method_sort_key(method: str) -> Tuple[int, str]:
    try:
        return (METHOD_ORDER.index(method), method)
    except ValueError:
        return (len(METHOD_ORDER), method)

RE_LABIR = re.compile(r"(?:s_\d{3}_)?.*_LabIR\(S(\d+)_(\d+)\)\.wav$")
RE_SPIR1 = re.compile(r"(?:s_\d{3}_)?.*_SPIR1\((\d+)m_(\d+)\)\.wav$")
RE_SPIR2 = re.compile(r"(?:s_\d{3}_)?.*_SPIR2\((\d+)m_180_r(\d+)\)\.wav$")
RE_SA    = re.compile(r"_sa\.wav$")
RE_MONO  = re.compile(r"_mono\.wav$")


# ============================================================
# DETECTION DATA STRUCTURE
# ============================================================
class Detection:
    __slots__ = (
        "date", "hour", "method", "species", "confidence",
        "start_time", "time_str", "distance", "azimuth", "elevation",
        "direction_label", "raw_filename", "source",
    )
    def __init__(self, date, hour, method, species, confidence,
                 start_time, time_str, distance, azimuth, elevation,
                 direction_label, raw_filename, source=""):
        self.date = date
        self.hour = hour
        self.method = method
        self.species = species
        self.confidence = confidence
        self.start_time = start_time
        self.time_str = time_str
        self.distance = distance
        self.azimuth = azimuth
        self.elevation = elevation
        self.direction_label = direction_label
        self.raw_filename = raw_filename
        self.source = source

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date, "hour": self.hour, "method": self.method,
            "species": self.species, "confidence": self.confidence,
            "start_time": self.start_time, "time_str": self.time_str,
            "distance": self.distance, "azimuth": self.azimuth,
            "elevation": self.elevation, "direction_label": self.direction_label,
            "raw_filename": self.raw_filename, "source": self.source,
        }


def filter_for_dashboard(
    detections: List[Detection],
    min_conf: float = DASHBOARD_MIN_CONF,
) -> List[Detection]:
    """Keep only detections strictly above the dashboard confidence floor."""
    return [d for d in detections if d.confidence > min_conf]


# ============================================================
# DIRECTION PARSING
# ============================================================
def _format_time(seconds: float) -> str:
    s = int(seconds)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_direction(filename: str, method: str) -> tuple:
    if method == "LabIR":
        m = RE_LABIR.search(filename)
        if m:
            speaker, azimuth = int(m.group(1)), int(m.group(2))
            elev = LABIR_ELEVATION.get(speaker, "?")
            return ("", f"{azimuth:03d}", f"{elev:+d}",
                    f"S{speaker:02d} elev:{elev:+d}deg az:{azimuth:03d}deg")
        return ("", "", "", filename)
    elif method == "SPIR1":
        m = RE_SPIR1.search(filename)
        if m:
            dist, azimuth = int(m.group(1)), int(m.group(2))
            return (f"{dist}m", f"{azimuth:03d}", "", f"{dist}m az:{azimuth:03d}deg")
        return ("", "", "", filename)
    elif method == "SPIR2":
        m = RE_SPIR2.search(filename)
        if m:
            dist, rep = int(m.group(1)), int(m.group(2))
            return (f"{dist}m", "180", "", f"{dist}m az:180deg rep:{rep}")
        return ("", "", "", filename)
    elif method == "SA":
        return ("", "", "", "omnidirectional")
    elif method == "Mono":
        return ("", "", "", "mono baseline")
    return ("", "", "", filename)


# ============================================================
# DATA COLLECTION
# ============================================================
def find_processed_json_files(
    base_dir: str, location: str,
    date_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str, str]]:
    location_dir = os.path.join(base_dir, location)
    if not os.path.isdir(location_dir):
        print(f"Location directory not found: {location_dir}")
        return []
    found = []
    for entry in sorted(os.listdir(location_dir)):
        date_path = os.path.join(location_dir, entry)
        if not os.path.isdir(date_path):
            continue
        if date_filter and entry not in date_filter:
            continue
        for subdir in PROCESSING_DIRS:
            method_dir = os.path.join(date_path, subdir)
            if not os.path.isdir(method_dir):
                continue
            for hour_entry in sorted(os.listdir(method_dir)):
                if not hour_entry.startswith("h_"):
                    continue
                hour_dir = os.path.join(method_dir, hour_entry)
                if not os.path.isdir(hour_dir):
                    continue
                for minute_entry in sorted(os.listdir(hour_dir)):
                    if not minute_entry.startswith("m_"):
                        continue
                    minute_dir = os.path.join(hour_dir, minute_entry)
                    if not os.path.isdir(minute_dir):
                        continue
                    proc_json = os.path.join(minute_dir, "processed.json")
                    if os.path.isfile(proc_json):
                        method = METHOD_MAP.get(subdir, subdir)
                        found.append((entry, f"{hour_entry}/{minute_entry}", method, proc_json))
    return found


def collect_detections(processed_files: List[Tuple]) -> List[Detection]:
    detections = []
    for date_str, hour_str, method, json_path in processed_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping {json_path}: {e}", file=sys.stderr)
            continue
        is_nested = bool(data) and all(
            isinstance(v, dict) and ("conf_list" in v or "start_time_list" in v
                                     or any(isinstance(vv, dict) for vv in v.values()))
            and not ("conf_list" in v and "start_time_list" in v
                     and isinstance(v.get("conf_list"), list))
            for v in data.values()
        )
        if is_nested:
            for source_base, sp_map in data.items():
                if not isinstance(sp_map, dict):
                    continue
                for species_name, sp_data in sp_map.items():
                    if not isinstance(sp_data, dict):
                        continue
                    detections.extend(_flatten(sp_data, date_str, hour_str, method, species_name, source_base))
        else:
            for species_name, sp_data in data.items():
                detections.extend(_flatten(sp_data, date_str, hour_str, method, species_name, "__legacy__"))
    return detections


def _flatten(sp_data, date_str, hour_str, method, species_name, source_base):
    out = []
    start_times = sp_data.get("start_time_list", [])
    confidences = sp_data.get("conf_list", [])
    channels = sp_data.get("primary_channel_list", [])
    for i in range(len(start_times)):
        conf = round(float(confidences[i]), 3) if i < len(confidences) else 0.0
        ch = channels[i] if i < len(channels) else ""
        start_t = round(float(start_times[i]), 1)
        dist, az, elev, label = parse_direction(ch, method)
        src = source_base if source_base and source_base != "__legacy__" else _infer_source(ch)
        out.append(Detection(
            date=date_str,
            hour=hour_str.replace("h_", "") if "h_" in hour_str else hour_str,
            method=method, species=species_name, confidence=conf,
            start_time=start_t, time_str=_format_time(start_t),
            distance=dist, azimuth=az, elevation=elev,
            direction_label=label, raw_filename=ch, source=src,
        ))
    return out


_REPORT_SOURCE_RE = re.compile(
    r"^(?:s_\d{3}_)?(?P<src>.+?)_(?:LabIR|SPIR1|SPIR2)\([^)]*\)\.wav$"
    r"|^(?:s_\d{3}_)?(?P<src_sa>.+?)_sa\.wav$"
    r"|^(?:s_\d{3}_)?(?P<src_mono>.+?)_mono\.wav$"
)
def _infer_source(wav_name: str) -> str:
    m = _REPORT_SOURCE_RE.match(wav_name)
    if not m:
        return wav_name[:-4] if wav_name.lower().endswith(".wav") else wav_name
    return m.group("src") or m.group("src_sa") or m.group("src_mono")


# ============================================================
# AGGREGATION
# ============================================================
def _percentile(sorted_data, p):
    """Return p-th percentile (0-100) of a sorted list using linear interpolation."""
    if not sorted_data:
        return 0.0
    k = (len(sorted_data) - 1) * p / 100.0
    f = int(k)
    c = k - f
    if f + 1 < len(sorted_data):
        return sorted_data[f] + c * (sorted_data[f + 1] - sorted_data[f])
    return sorted_data[f]

def _r(val, precision=3):
    """Round a float, return 0.0 for None."""
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return 0.0
    return round(float(val), precision)


def _src_sort_key(src: str) -> Tuple[int, int]:
    try:
        parts = src.split("_")[0].split("-")
        return (int(parts[0]), int(parts[1]))
    except Exception:
        return (0, 0)

def _src_label(src: str) -> str:
    try:
        parts = src.split("_")[0].split("-")
        return f"{parts[0]}:{parts[1]}"
    except Exception:
        return src


# Confidence histogram bin edges (BirdNET scores typically ≥ min_conf, ≤ 1.0)
_CONF_HIST_LO = 0.40
_CONF_HIST_HI = 1.00
_CONF_HIST_STEP = 0.05


def _conf_histogram(confs: List[float]) -> List[Dict[str, float]]:
    """Fixed-width bins from 0.40–1.00 for distribution charts."""
    n_bins = int(round((_CONF_HIST_HI - _CONF_HIST_LO) / _CONF_HIST_STEP))
    counts = [0] * n_bins
    for c in confs:
        # Epsilon avoids float edge cases e.g. 0.5 → bin [0.45,0.50)
        idx = int((float(c) - _CONF_HIST_LO + 1e-9) / _CONF_HIST_STEP)
        idx = max(0, min(n_bins - 1, idx))
        counts[idx] += 1
    out = []
    for i, ct in enumerate(counts):
        lo = _CONF_HIST_LO + i * _CONF_HIST_STEP
        hi = lo + _CONF_HIST_STEP
        out.append({"lo": _r(lo, 2), "hi": _r(hi, 2), "count": ct})
    return out


def _parse_labir_speaker(raw_filename: str, direction_label: str) -> Optional[int]:
    m = RE_LABIR.search(raw_filename or "")
    if m:
        return int(m.group(1))
    m2 = re.search(r"S(\d+)", direction_label or "")
    return int(m2.group(1)) if m2 else None


def _labir_aggregate_subset(lab: List[Detection]) -> Dict[str, Any]:
    """Aggregate one LabIR subset into cells / by_azimuth / by_elevation."""
    if not lab:
        return {
            "cells": [], "by_azimuth": [], "by_elevation": [],
            "n_rows": 0, "detections": 0, "n_directions": 0,
        }

    # Detection = unique BirdNET window (source × start_time), not species-inflated
    cell_dets: Dict[Tuple[int, int, Optional[int]], set] = defaultdict(set)
    cell_confs: Dict[Tuple[int, int, Optional[int]], List[float]] = defaultdict(list)
    cell_species: Dict[Tuple[int, int, Optional[int]], set] = defaultdict(set)
    az_dets: Dict[int, set] = defaultdict(set)
    az_confs: Dict[int, List[float]] = defaultdict(list)
    elev_dets: Dict[int, set] = defaultdict(set)
    elev_confs: Dict[int, List[float]] = defaultdict(list)
    all_dets: set = set()

    for d in lab:
        try:
            az = int(d.azimuth)
        except (TypeError, ValueError):
            continue
        elev_s = str(d.elevation).replace("+", "")
        try:
            elev = int(elev_s)
        except (TypeError, ValueError):
            continue
        speaker = _parse_labir_speaker(d.raw_filename, d.direction_label)
        det_key = (d.source, d.start_time)
        key = (az, elev, speaker)
        cell_dets[key].add(det_key)
        cell_confs[key].append(d.confidence)
        cell_species[key].add(d.species)
        az_dets[az].add(det_key)
        az_confs[az].append(d.confidence)
        elev_dets[elev].add(det_key)
        elev_confs[elev].append(d.confidence)
        all_dets.add(det_key)

    cells = []
    for (az, elev, speaker), dets in sorted(
        cell_dets.items(), key=lambda x: (x[0][0], x[0][1])
    ):
        confs = cell_confs[(az, elev, speaker)]
        cells.append({
            "azimuth": az,
            "elevation": elev,
            "speaker": speaker,
            "detections": len(dets),
            "species_count": len(cell_species[(az, elev, speaker)]),
            "conf_avg": _r(statistics.mean(confs)),
            "conf_median": _r(statistics.median(confs)),
            "conf_max": _r(max(confs)),
            "n_rows": len(confs),
        })

    by_azimuth = []
    for az in sorted(az_dets.keys()):
        confs = az_confs[az]
        by_azimuth.append({
            "azimuth": az,
            "detections": len(az_dets[az]),
            "conf_avg": _r(statistics.mean(confs)),
            "conf_median": _r(statistics.median(confs)),
            "n_rows": len(confs),
        })

    by_elevation = []
    for elev in sorted(elev_dets.keys()):
        confs = elev_confs[elev]
        by_elevation.append({
            "elevation": elev,
            "detections": len(elev_dets[elev]),
            "conf_avg": _r(statistics.mean(confs)),
            "conf_median": _r(statistics.median(confs)),
            "n_rows": len(confs),
        })

    return {
        "cells": cells,
        "by_azimuth": by_azimuth,
        "by_elevation": by_elevation,
        "n_rows": len(lab),
        "detections": len(all_dets),
        "n_directions": len(cells),
    }


def build_labir_spatial(detections: List[Detection]) -> Dict[str, Any]:
    """LabIR winning-beam spatial summary: overall + per species.

    Detection = unique BirdNET window (source, start_time) per direction
    (not species-inflated when aggregating overall). Colour scale is absolute
    conf 0.55–1.00 so gradients are comparable across species views.
    """
    lab = [d for d in detections if d.method == "LabIR" and d.azimuth not in ("", None)]
    # Fixed BirdNET-ish scale for colour (dashboard already filters > 0.55)
    color_scale = {"min": 0.55, "max": 1.0, "mode": "absolute"}

    if not lab:
        empty = _labir_aggregate_subset([])
        return {
            **empty,
            "color_scale": color_scale,
            "species_list": [],
            "by_species": {},
        }

    overall = _labir_aggregate_subset(lab)

    by_sp: Dict[str, List[Detection]] = defaultdict(list)
    for d in lab:
        by_sp[d.species].append(d)

    species_list = []
    by_species: Dict[str, Any] = {}
    for sp, rows in by_sp.items():
        agg = _labir_aggregate_subset(rows)
        confs = [d.confidence for d in rows]
        entry_meta = {
            "species": sp,
            "detections": agg["detections"],
            "n_rows": agg["n_rows"],
            "conf_avg": _r(statistics.mean(confs)),
            "conf_max": _r(max(confs)),
        }
        species_list.append(entry_meta)
        by_species[sp] = agg

    # Rank for picker: higher conf first, then more detections
    species_list.sort(key=lambda x: (-x["conf_avg"], -x["detections"], x["species"]))

    return {
        **overall,
        "color_scale": color_scale,
        "species_list": species_list,
        "by_species": by_species,
    }


def build_date_summary(
    detections: List[Detection],
    *,
    dashboard_min_conf: float = DASHBOARD_MIN_CONF,
    raw_row_count: Optional[int] = None,
) -> Dict[str, Any]:
    """Aggregate detections already filtered for the dashboard (conf > min).

    ``raw_row_count`` is optional pre-filter method-row count (for the UI badge).
    """
    meta = {
        "dashboard_min_conf": dashboard_min_conf,
        "filter": f"confidence > {dashboard_min_conf}",
        "raw_method_rows": raw_row_count if raw_row_count is not None else len(detections),
        "dashboard_method_rows": len(detections),
    }

    if not detections:
        return {
            "total_detections": 0, "unique_species": 0,
            "method_confidence": {}, "species_method_heatmap": [],
            "heatmap_methods": [], "top_species": [], "hourly_activity": [],
            "source_breakdown_counts": [], "source_breakdown_confs": [],
            "labir_spatial": build_labir_spatial([]),
            **meta,
        }

    # Unique acoustic events: (species, source, chunk_start) — not inflated by method
    unique_events: set = set()
    for d in detections:
        unique_events.add((d.species, d.source, d.start_time))

    species_unique: Dict[str, set] = defaultdict(set)
    species_confs: Dict[str, List[float]] = defaultdict(list)
    species_methods: Dict[str, set] = defaultdict(set)
    # Detection windows (source × start), not species×window (that inflates hourly bars)
    hourly_dets: Dict[str, set] = defaultdict(set)
    hourly_species: Dict[str, set] = defaultdict(set)
    for d in detections:
        key = (d.source, d.start_time)
        species_unique[d.species].add(key)
        species_confs[d.species].append(d.confidence)
        species_methods[d.species].add(d.method)
        hour = d.hour.split("/")[0] if "/" in d.hour else d.hour
        hourly_dets[hour].add(key)
        hourly_species[hour].add(d.species)

    species_counts = Counter({sp: len(keys) for sp, keys in species_unique.items()})

    # ── Confidence stats per method ───────────────────────
    method_confs: Dict[str, List[float]] = defaultdict(list)
    for d in detections:
        method_confs[d.method].append(d.confidence)

    method_confidence = {}
    sorted_methods = sorted(method_confs.keys(), key=_method_sort_key)
    for method in sorted_methods:
        arr = sorted(method_confs[method])
        method_confidence[method] = {
            "total": len(arr),
            "conf_avg": _r(statistics.mean(arr)),
            "conf_median": _r(statistics.median(arr)),
            "conf_min": _r(min(arr)),
            "conf_max": _r(max(arr)),
            "conf_q25": _r(_percentile(arr, 25)),
            "conf_q75": _r(_percentile(arr, 75)),
            "hist": _conf_histogram(arr),
        }

    # ── Source × Method breakdown ───────────────────────────
    # Counts = unique start-time detections (not species-row inflation)
    src_m_dets: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    src_m_species: Dict[str, Dict[str, set]] = defaultdict(lambda: defaultdict(set))
    src_m_confs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for d in detections:
        src = d.source
        src_m_dets[src][d.method].add(d.start_time)
        src_m_species[src][d.method].add(d.species)
        src_m_confs[src][d.method].append(d.confidence)

    sorted_sources = sorted(src_m_dets.keys(), key=_src_sort_key)

    source_breakdown_counts = []
    source_breakdown_confs = []
    for src in sorted_sources:
        label = _src_label(src)
        entry_c = {"source": src, "label": label, "methods": {}, "species": {}}
        entry_f = {"source": src, "label": label, "methods": {}}
        for method in sorted_methods:
            dets = src_m_dets[src].get(method, set())
            confs = src_m_confs[src].get(method, [])
            entry_c["methods"][method] = len(dets)
            entry_c["species"][method] = len(src_m_species[src].get(method, set()))
            entry_f["methods"][method] = {
                "count": len(dets),
                "species_count": len(src_m_species[src].get(method, set())),
                "conf_avg": _r(statistics.mean(confs)) if confs else 0.0,
                "conf_median": _r(statistics.median(confs)) if confs else 0.0,
            }
        source_breakdown_counts.append(entry_c)
        source_breakdown_confs.append(entry_f)

    # Rank primarily by mean confidence, then unique-event count
    def _species_rank_key(sp: str):
        confs = species_confs[sp]
        return (-statistics.mean(confs), -species_counts[sp], sp)

    top_25 = sorted(species_counts.keys(), key=_species_rank_key)[:25]
    top_species = []
    for sp in top_25:
        confs = species_confs[sp]
        top_species.append({
            "species": sp,
            "count": species_counts[sp],  # unique events (not × methods)
            "raw_count": len(confs),  # method-rows after dashboard filter
            "conf_avg": _r(statistics.mean(confs)),
            "conf_median": _r(statistics.median(confs)),
            "method_count": len(species_methods[sp]),
        })

    # ── Species × Method confidence heatmap (top 15 by conf) ──
    top_15_species = set(top_25[:15])
    sp_method_confs: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for d in detections:
        if d.species in top_15_species:
            sp_method_confs[d.species][d.method].append(d.confidence)

    heatmap = []
    for sp in top_25[:15]:
        row = {
            "species": sp,
            "unique_count": species_counts[sp],  # detections, not sum of method cells
        }
        for method in sorted_methods:
            confs = sp_method_confs[sp].get(method, [])
            if confs:
                row[method] = {
                    "count": len(confs),
                    "conf_avg": _r(statistics.mean(confs)),
                    "conf_median": _r(statistics.median(confs)),
                }
            else:
                row[method] = None
        heatmap.append(row)

    return {
        "total_detections": len(unique_events),
        "unique_species": len(species_counts),
        "method_confidence": method_confidence,
        "species_method_heatmap": heatmap,
        "heatmap_methods": sorted_methods,
        "top_species": top_species,
        # count = unique (source, start_time) detections — not species-inflated rows
        "hourly_activity": [
            {
                "hour": h,
                "count": len(hourly_dets[h]),
                "species_count": len(hourly_species[h]),
            }
            for h in sorted(hourly_dets.keys())
        ],
        "source_breakdown_counts": source_breakdown_counts,
        "source_breakdown_confs": source_breakdown_confs,
        "labir_spatial": build_labir_spatial(detections),
        **meta,
    }


# ============================================================
# JSON EXPORT (external reference, not embedded in HTML)
# ============================================================
def write_report_data(
    detections: List[Detection],
    out_dir: str,
    compress: bool = True,
    dashboard_min_conf: float = DASHBOARD_MIN_CONF,
    summaries: Optional[Dict[str, Any]] = None,
) -> str:
    """Write full detection dumps + dashboard-filtered summaries.

    - ``*_detections.json.gz``: all collected rows (pipeline conf, typically ≥ 0.4)
    - ``*_summary.json``: aggregates after dashboard filter (conf > min), or
      pre-built ``summaries`` if provided
    """
    os.makedirs(out_dir, exist_ok=True)
    by_date: Dict[str, List[Detection]] = defaultdict(list)
    for d in detections:
        by_date[d.date].append(d)
    for date_str, dets in sorted(by_date.items()):
        if summaries and date_str in summaries:
            summary = dict(summaries[date_str])
        else:
            dash = filter_for_dashboard(dets, dashboard_min_conf)
            summary = build_date_summary(
                dash,
                dashboard_min_conf=dashboard_min_conf,
                raw_row_count=len(dets),
            )
        summary["date"] = date_str
        summary_path = os.path.join(out_dir, f"{date_str}_summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        det_list = [d.to_dict() for d in dets]
        det_fname = f"{date_str}_detections.json"
        if compress:
            with gzip.open(os.path.join(out_dir, det_fname + ".gz"), "wt", encoding="utf-8") as f:
                json.dump(det_list, f, ensure_ascii=False)
        else:
            with open(os.path.join(out_dir, det_fname), "w", encoding="utf-8") as f:
                json.dump(det_list, f, ensure_ascii=False)
    return out_dir


# ============================================================
# HTML GENERATION — VISUALISATION-ONLY DASHBOARD
# ============================================================
# Insertion order = legend order (Mono → SA → beamforming family)
METHOD_COLORS = {
    "Mono": "#eab308",
    "SA": "#22c55e",
    "LabIR": "#3b82f6",
    "SPIR1": "#8b5cf6",
    "SPIR2": "#ec4899",
}


def generate_html(
    location: str,
    data_dir: str,
    summaries: Dict[str, Any],
    dashboard_min_conf: float = DASHBOARD_MIN_CONF,
) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    summaries_json = json.dumps(summaries, ensure_ascii=False, separators=(",", ":"))
    summaries_json = summaries_json.replace("<", "\\x3c")
    conf_label = f"conf &gt; {dashboard_min_conf:g}"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Way Canguk &mdash; {escape(location)} BirdNET Dashboard</title>
<style>
:root {{
    --bg: #1a1d23; --bg-card: #21242b; --bg-card-hover: #282d36;
    --bg-input: #2a2d35; --bg-thead: #282c35;
    --border: #2d3240; --border-light: #2a2e38;
    --text: #e5e7eb; --text-muted: #9ca3af; --text-heading: #f3f4f6;
    --text-th: #d1d5db; --text-th-hover: #e5e7eb;
    --accent: #3b82f6;
    --header-from: #1e3a5f; --header-to: #0f2440;
    --conf-high: #22c55e; --conf-med: #eab308; --conf-low: #ef4444;
}}
@media (prefers-color-scheme: light) {{
    :root {{
        --bg: #f3f4f6; --bg-card: #ffffff; --bg-card-hover: #f9fafb;
        --bg-input: #ffffff; --bg-thead: #f1f5f9;
        --border: #d1d5db; --border-light: #e5e7eb;
        --text: #1f2937; --text-muted: #4b5563; --text-heading: #111827;
        --text-th: #374151; --text-th-hover: #111827;
        --accent: #2563eb;
        --header-from: #dbeafe; --header-to: #bfdbfe;
        --conf-high: #16a34a; --conf-med: #ca8a04; --conf-low: #dc2626;
    }}
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{font-size:17px}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);line-height:1.55;padding:1.5rem;max-width:1480px;margin:0 auto;-webkit-font-smoothing:antialiased}}
header{{background:linear-gradient(135deg,var(--header-from),var(--header-to));border:1px solid var(--border);border-radius:10px;padding:1.35rem 2rem;margin-bottom:1.35rem}}
header h1{{font-size:1.55rem;font-weight:700;color:var(--text-heading);letter-spacing:-0.01em}}
header .subtitle{{font-size:0.95rem;color:var(--text-muted);margin-top:0.4rem;line-height:1.45}}

.stats-row{{display:flex;gap:1rem;margin-bottom:1.35rem;flex-wrap:wrap}}
.stat-card{{flex:1;min-width:140px;background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:1rem 1.15rem}}
.stat-card .label{{font-size:0.82rem;text-transform:uppercase;letter-spacing:0.04em;color:var(--text-muted);margin-bottom:0.3rem;font-weight:500}}
.stat-card .value{{font-size:1.45rem;font-weight:700;color:var(--text-heading)}}

.date-selector{{display:flex;gap:0.55rem;margin-bottom:1.1rem;flex-wrap:wrap}}
.date-btn{{background:var(--bg-input);border:1px solid var(--border);border-radius:6px;padding:0.45rem 0.95rem;color:var(--text);font-size:0.95rem;cursor:pointer;transition:all 0.15s}}
.date-btn:hover{{border-color:var(--accent);color:var(--text-heading)}}
.date-btn.active{{background:var(--accent);border-color:var(--accent);color:#fff}}

.card{{background:var(--bg-card);border:1px solid var(--border);border-radius:8px;padding:1.35rem 1.6rem;margin-bottom:1.35rem}}
.card h2{{font-size:1.15rem;font-weight:650;color:var(--text-heading);margin-bottom:0.55rem}}

/* ── Method confidence distribution ─────────────────────── */
.conf-dist{{display:flex;flex-direction:column;gap:0.85rem}}
.conf-split{{display:grid;grid-template-columns:1fr 1fr;gap:1.1rem;align-items:start}}
@media(max-width:960px){{.conf-split{{grid-template-columns:1fr}}}}
.conf-panel{{min-width:0;width:100%}}
.conf-axis-note{{font-size:0.9rem;color:var(--text-muted);margin-bottom:0.35rem;font-weight:500}}
.conf-boxplot-wrap,.conf-hist-wrap{{width:100%;overflow:visible}}
.conf-boxplot-wrap svg,.conf-hist-wrap svg{{display:block;width:100%;height:auto}}
.conf-legend{{display:flex;gap:1rem;flex-wrap:wrap;font-size:0.9rem;color:var(--text-muted);margin-top:0.45rem}}
.conf-legend span{{display:flex;align-items:center;gap:0.35rem}}

/* ── Heatmap table ─────────────────────────────────────── */
.heatmap-wrap{{overflow-x:auto}}
.heatmap{{border-collapse:collapse;font-size:0.95rem;width:100%}}
.heatmap th,.heatmap td{{padding:0.5rem 0.7rem;text-align:center;white-space:nowrap}}
.heatmap th:first-child,.heatmap td:first-child{{text-align:left;font-weight:550;max-width:220px;overflow:hidden;text-overflow:ellipsis}}
.heatmap thead th{{font-weight:650;color:var(--text-th);border-bottom:2px solid var(--border);font-size:0.88rem;text-transform:uppercase;letter-spacing:0.03em}}
.heatmap tbody td{{border-bottom:1px solid var(--border-light);color:var(--text)}}
.heatmap tbody tr:hover{{background:var(--bg-card-hover)}}
.heatmap .nil{{color:var(--text-muted);font-size:0.9rem}}

/* ── Horizontal bars (hourly etc.) ─────────────────────── */
.bar-list{{display:flex;flex-direction:column;gap:0.5rem;max-width:560px}}
.bar-row{{display:flex;align-items:center;gap:0.75rem;font-size:0.95rem}}
.bar-label{{width:72px;text-align:right;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;font-weight:500}}
.bar-track{{flex:1;height:18px;background:var(--bg-input);border-radius:4px;overflow:hidden;position:relative}}
.bar-fill{{height:100%;border-radius:4px;opacity:0.85}}
.bar-value{{width:140px;font-weight:650;color:var(--text-heading);font-size:0.9rem;font-variant-numeric:tabular-nums}}

/* ── Top species: full-width, L→R wrap, card ~ species-name wide ─ */
.species-rank-wrap{{width:100%}}
.species-rank{{
    display:flex;flex-wrap:wrap;align-items:flex-start;align-content:flex-start;
    gap:0.45rem 0.55rem;width:100%;
}}
/* Grow equally across the row so the block spans full width; basis ≈ name+donut */
.sp-rank-item{{
    display:grid;grid-template-columns:1.25rem 58px max-content;
    column-gap:0.4rem;align-items:center;
    flex:1 1 13.5rem;min-width:12.5rem;max-width:100%;
    padding:0.4rem 0.5rem;border-radius:8px;
    border:1px solid var(--border-light);background:var(--bg-input);
    box-sizing:border-box;
}}
.sp-rank-item:hover{{border-color:var(--border);background:var(--bg-card-hover)}}
.sp-rank-num{{font-size:0.8rem;font-weight:700;color:var(--text-muted);font-variant-numeric:tabular-nums}}
.sp-rank-donut{{width:58px;height:58px;display:block;flex-shrink:0}}
.sp-rank-body{{min-width:0;max-width:11rem;display:flex;flex-direction:column;gap:0.1rem}}
.sp-rank-name{{font-size:0.88rem;font-weight:550;color:var(--text-heading);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.sp-rank-meta{{font-size:0.78rem;color:var(--text-muted);font-variant-numeric:tabular-nums}}
.sp-rank-meta strong{{color:var(--text-heading);font-weight:700}}
@media(max-width:520px){{
    .sp-rank-item{{flex-basis:100%;min-width:0;grid-template-columns:1.25rem 58px minmax(0,1fr)}}
    .sp-rank-body{{max-width:none}}
}}

.tooltip{{font-size:0.9rem;color:var(--text-muted);margin:0.15rem 0 0.85rem;line-height:1.45}}

/* ── Source breakdown charts ────────────────────────────── */
.chart-scroll{{overflow-x:auto;overflow-y:hidden}}
.chart-scroll svg{{min-width:500px}}
.chart-bar-group:hover rect{{opacity:1!important}}
.chart-legend{{display:flex;gap:1rem;margin-bottom:0.55rem;flex-wrap:wrap;font-size:0.95rem}}
.chart-legend span{{display:flex;align-items:center;gap:0.35rem}}
.chart-legend .swatch{{width:12px;height:12px;border-radius:2px;flex-shrink:0}}

/* ── LabIR spatial ─────────────────────────────────────── */
.spatial-grid{{display:grid;grid-template-columns:minmax(280px,1fr) minmax(260px,0.9fr);gap:1.25rem;align-items:start}}
@media(max-width:900px){{.spatial-grid{{grid-template-columns:1fr}}}}
.spatial-panel h3{{font-size:1rem;font-weight:600;color:var(--text-heading);margin-bottom:0.45rem}}
.spatial-panel .sub{{font-size:0.88rem;color:var(--text-muted);margin-bottom:0.65rem;line-height:1.4}}
.spatial-polar-wrap{{display:flex;justify-content:center;overflow-x:auto}}
.spatial-stats{{display:flex;gap:0.75rem;flex-wrap:wrap;margin-bottom:0.75rem}}
.spatial-stats .pill{{background:var(--bg-input);border:1px solid var(--border);border-radius:999px;padding:0.3rem 0.75rem;font-size:0.88rem;color:var(--text)}}
.spatial-stats .pill strong{{color:var(--text-heading);font-weight:650}}
.spatial-picker{{display:flex;flex-wrap:wrap;gap:0.45rem;align-items:center;margin-bottom:0.85rem}}
.spatial-picker label{{font-size:0.9rem;color:var(--text-muted);font-weight:500;margin-right:0.15rem}}
.sp-btn{{background:var(--bg-input);border:1px solid var(--border);border-radius:999px;padding:0.35rem 0.85rem;color:var(--text);font-size:0.88rem;cursor:pointer;transition:all 0.12s;max-width:220px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.sp-btn:hover{{border-color:var(--accent);color:var(--text-heading)}}
.sp-btn.active{{background:var(--accent);border-color:var(--accent);color:#fff;font-weight:600}}
.sp-select{{background:var(--bg-input);border:1px solid var(--border);border-radius:8px;padding:0.4rem 0.7rem;color:var(--text);font-size:0.92rem;min-width:200px;max-width:100%}}
.sp-select:focus{{outline:2px solid var(--accent);outline-offset:1px}}
.color-legend{{display:flex;align-items:center;gap:0.5rem;flex-wrap:wrap;margin:0.35rem 0 0.85rem;font-size:0.88rem;color:var(--text-muted)}}
.color-legend .ramp{{display:flex;height:14px;width:160px;border-radius:4px;overflow:hidden;border:1px solid var(--border)}}
.color-legend .ramp span{{flex:1}}
.color-legend .swatch-disc{{width:14px;height:14px;border-radius:3px;border:1px solid var(--border);flex-shrink:0}}

footer{{margin-top:1.75rem;font-size:0.9rem;color:var(--text-muted);text-align:center;line-height:1.5}}

@media(max-width:768px){{
    html{{font-size:16px}}
    body{{padding:0.9rem}}
    .bar-label{{width:120px;font-size:0.88rem}}
    .bar-value{{width:110px;font-size:0.85rem}}
    .stats-row{{gap:0.6rem}}
    .stat-card{{min-width:110px;padding:0.75rem 0.9rem}}
    .stat-card .value{{font-size:1.25rem}}
    header h1{{font-size:1.3rem}}
}}
</style>
</head>
<body>

<header>
    <h1>Way Canguk &mdash; {escape(location)} BirdNET Dashboard</h1>
    <div class="subtitle">Generated: {escape(ts)} &nbsp;|&nbsp; Data: {escape(data_dir)} &nbsp;|&nbsp; Dashboard filter: {conf_label} (pipeline BirdNET min_conf stays 0.4)</div>
</header>

<div class="date-selector" id="dateSelector"></div>

<div class="stats-row" id="statsRow">
    <div class="stat-card"><div class="label">Detections ({conf_label})</div><div class="value" id="statTotal">&mdash;</div></div>
    <div class="stat-card"><div class="label">Species</div><div class="value" id="statSpecies">&mdash;</div></div>
    <div class="stat-card"><div class="label">Methods</div><div class="value" id="statMethods">&mdash;</div></div>
    <div class="stat-card"><div class="label">Hours Active</div><div class="value" id="statHours">&mdash;</div></div>
    <div class="stat-card"><div class="label">Rows kept / raw</div><div class="value" id="statFilter">&mdash;</div></div>
</div>

<div class="card">
    <h2>Confidence Score Distribution by Method</h2>
    <div class="tooltip">Fixed scale 0.40&ndash;1.00. Box = IQR (q25&ndash;q75), line = median, diamond = mean, whiskers = min&ndash;max. Histogram shows mass per bin (why many methods look similar when scores pile near the threshold).</div>
    <div class="conf-dist" id="confRange"></div>
</div>

<div class="card">
    <h2>Species &times; Method Confidence Heatmap</h2>
    <div class="tooltip">Top 15 species ranked by mean confidence (then detection count), after {conf_label}. Cell colour = mean conf on the same fixed scale as LabIR Spatial (0.55&ndash;1.00, blue&rarr;red). Detections = unique (source, time) windows, not summed across methods.</div>
    <div class="heatmap-wrap" id="heatmapContainer"></div>
</div>

<div class="card">
    <h2>LabIR Spatial Directions</h2>
    <div class="tooltip">Winning LabIR beam after {conf_label}. <strong>Bar length / cell number</strong> = BirdNET detections (unique source &times; start window). <strong>Colour</strong> = mean confidence on a fixed scale 0.55&ndash;1.00 (blue = lower, yellow = mid, red = higher). Pick a species or All.</div>
    <div id="labirSpatial"></div>
</div>

<div class="card">
    <h2>Top Species by Confidence</h2>
    <div class="tooltip">Ranked by mean conf (desc), then detections. Cards wrap left→right, top-aligned, spanning full card width. Donut fill = conf 0.55&ndash;1.00 (same scale as LabIR Spatial). Label = conf · detections after {conf_label}.</div>
    <div id="speciesBarList"></div>
</div>

<div class="card">
    <h2>Hourly Activity</h2>
    <div class="tooltip">Bar = unique BirdNET detections per hour (source × start window after conf filter). Not multiplied by species or method. Label also shows species richness that hour.</div>
    <div class="bar-list" id="hourlyBarList"></div>
</div>

<div class="card">
    <h2>Detections by Source Recording (per Method)</h2>
    <div class="tooltip">Each source = one FLAC recording. Bars = unique BirdNET detection windows (start time) after conf filter — not multiplied by species. Hover for count.</div>
    <div class="chart-scroll" id="srcCountChart"></div>
</div>

<div class="card">
    <h2>Confidence Score by Source Recording (per Method)</h2>
    <div class="tooltip">Mean confidence per source and method (after conf filter). Shows whether beamforming improves scores vs mono/SA.</div>
    <div class="chart-scroll" id="srcConfChart"></div>
</div>

<footer>
    Generated {escape(ts)} &nbsp;&middot;&nbsp; Spatial Ecoacoustic Analysis Pipeline &nbsp;&middot;&nbsp;
    Raw data: report_data/
</footer>

<script>
var SUMMARIES = {summaries_json};
var METHOD_COLORS = {json.dumps(METHOD_COLORS, separators=(",", ":"))};
var METHOD_ORDER = {json.dumps(METHOD_ORDER, separators=(",", ":"))};
var currentDate = null;
// Must be assigned before init() runs (var is hoisted as undefined)
var labirSpatialState = {{ spatial: null, selected: '__all__' }};
// Fixed conf colour scale 0.55–1.00: blue (low) → red (high)
var LABIR_CONF_STOPS = [
    {{ t: 0.00, c: [29, 78, 216] }},
    {{ t: 0.25, c: [6, 182, 212] }},
    {{ t: 0.45, c: [34, 197, 94] }},
    {{ t: 0.65, c: [234, 179, 8] }},
    {{ t: 0.85, c: [249, 115, 22] }},
    {{ t: 1.00, c: [220, 38, 38] }}
];

function orderedMethods(methods) {{
    // Prefer canonical order; append any unknown methods at the end
    var set = {{}};
    (methods || []).forEach(function(m) {{ set[m] = true; }});
    var out = [];
    METHOD_ORDER.forEach(function(m) {{ if (set[m]) {{ out.push(m); delete set[m]; }} }});
    Object.keys(set).sort().forEach(function(m) {{ out.push(m); }});
    return out;
}}

function esc(s) {{ return String(s).replace(/&/g,'&amp;').replace(/\\x3c/g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;'); }}

function labirConfColor(conf, scaleMin, scaleMax) {{
    var lo = (typeof scaleMin === 'number') ? scaleMin : 0.55;
    var hi = (typeof scaleMax === 'number') ? scaleMax : 1.0;
    var t = (conf - lo) / (hi - lo || 0.001);
    t = Math.max(0, Math.min(1, t));
    var i = 0;
    while (i < LABIR_CONF_STOPS.length - 2 && t > LABIR_CONF_STOPS[i + 1].t) i++;
    var a = LABIR_CONF_STOPS[i], b = LABIR_CONF_STOPS[i + 1];
    var u = (t - a.t) / (b.t - a.t || 0.001);
    var r = Math.round(a.c[0] + (b.c[0] - a.c[0]) * u);
    var g = Math.round(a.c[1] + (b.c[1] - a.c[1]) * u);
    var bl = Math.round(a.c[2] + (b.c[2] - a.c[2]) * u);
    return 'rgb(' + r + ',' + g + ',' + bl + ')';
}}

// Shared Mean conf colour legend (same wording everywhere)
function confColorLegendHtml(scaleMin, scaleMax) {{
    var lo = (typeof scaleMin === 'number') ? scaleMin : 0.55;
    var hi = (typeof scaleMax === 'number') ? scaleMax : 1.0;
    var html = '\\x3cdiv class="color-legend">';
    html += '\\x3cspan style="font-weight:600;color:var(--text)">Mean conf\\x3c/span>';
    html += '\\x3cspan>' + lo.toFixed(2) + '\\x3c/span>';
    html += '\\x3cdiv class="ramp" title="Fixed scale ' + lo.toFixed(2) + '–' + hi.toFixed(2) + '">';
    for (var ri = 0; ri < 12; ri++) {{
        var rc = lo + (hi - lo) * (ri / 11);
        html += '\\x3cspan style="background:' + labirConfColor(rc, lo, hi) + '">\\x3c/span>';
    }}
    html += '\\x3c/div>';
    html += '\\x3cspan>' + hi.toFixed(2) + '\\x3c/span>';
    html += '\\x3cspan style="margin-left:0.5rem">· blue lower · green mid · red higher\\x3c/span>';
    html += '\\x3c/div>';
    return html;
}}

function renderDateSelector(dates) {{
    var html = '';
    for (var i = 0; i < dates.length; i++) {{
        var d = dates[i];
        var s = SUMMARIES[d];
        html += '\\x3cbutton class="date-btn" data-date="' + esc(d) + '">' +
                esc(d) + ' \\x3cspan style="opacity:0.7">(' + s.total_detections.toLocaleString() + ')\\x3c/span>\\x3c/button>';
    }}
    document.getElementById('dateSelector').innerHTML = html;
    document.querySelectorAll('.date-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            document.querySelectorAll('.date-btn').forEach(function(b) {{ b.classList.remove('active'); }});
            btn.classList.add('active');
            selectDate(btn.dataset.date);
        }});
    }});
    var first = document.querySelector('.date-btn');
    if (first) first.classList.add('active');
}}

function selectDate(d) {{
    currentDate = d;
    renderAll(SUMMARIES[d]);
}}

function renderAll(data) {{
    document.getElementById('statTotal').textContent = data.total_detections.toLocaleString();
    document.getElementById('statSpecies').textContent = data.unique_species;
    document.getElementById('statMethods').textContent = Object.keys(data.method_confidence || {{}}).length;
    document.getElementById('statHours').textContent = (data.hourly_activity || []).length;
    var kept = data.dashboard_method_rows;
    var raw = data.raw_method_rows;
    var filterEl = document.getElementById('statFilter');
    if (filterEl) {{
        if (typeof kept === 'number' && typeof raw === 'number') {{
            filterEl.textContent = kept.toLocaleString() + ' / ' + raw.toLocaleString();
            filterEl.title = 'Dashboard method-rows after conf filter / raw method-rows from processed.json';
        }} else {{
            filterEl.textContent = '—';
        }}
    }}

    renderConfRange(data.method_confidence);
    renderHeatmap(data.species_method_heatmap, data.heatmap_methods);
    renderLabirSpatial(data.labir_spatial);
    renderSpeciesBars(data.top_species);
    renderHourlyBars(data.hourly_activity);
    renderSrcCountChart(data.source_breakdown_counts);
    renderSrcConfChart(data.source_breakdown_confs);
}}

// ── LabIR spatial (species picker + polar + elev + grid) ─
function renderLabirSpatial(spatial) {{
    var container = document.getElementById('labirSpatial');
    if (!container) return;
    if (!spatial || (!(spatial.by_azimuth && spatial.by_azimuth.length) && !(spatial.species_list && spatial.species_list.length))) {{
        container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No LabIR direction data for this date (after conf filter).\\x3c/div>';
        labirSpatialState.spatial = null;
        return;
    }}
    labirSpatialState.spatial = spatial;
    if (labirSpatialState.selected !== '__all__' && !(spatial.by_species && spatial.by_species[labirSpatialState.selected])) {{
        labirSpatialState.selected = '__all__';
    }}
    _paintLabirSpatial();
}}

function _labirViewData() {{
    var spatial = labirSpatialState.spatial;
    if (!spatial) return null;
    if (labirSpatialState.selected === '__all__') {{
        return {{
            label: 'All species',
            by_azimuth: spatial.by_azimuth || [],
            by_elevation: spatial.by_elevation || [],
            cells: spatial.cells || [],
            n_rows: spatial.n_rows || 0,
            detections: spatial.detections || 0,
            n_directions: spatial.n_directions || 0,
        }};
    }}
    var sub = (spatial.by_species || {{}})[labirSpatialState.selected];
    if (!sub) return null;
    return {{
        label: labirSpatialState.selected,
        by_azimuth: sub.by_azimuth || [],
        by_elevation: sub.by_elevation || [],
        cells: sub.cells || [],
        n_rows: sub.n_rows || 0,
        detections: sub.detections || 0,
        n_directions: sub.n_directions || 0,
    }};
}}

function _paintLabirSpatial() {{
    var container = document.getElementById('labirSpatial');
    var spatial = labirSpatialState.spatial;
    if (!container || !spatial) return;
    var view = _labirViewData();
    if (!view) {{
        container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No data for this selection.\\x3c/div>';
        return;
    }}

    var scale = spatial.color_scale || {{ min: 0.55, max: 1.0 }};
    var scaleMin = scale.min, scaleMax = scale.max;
    var speciesList = spatial.species_list || [];
    var selected = labirSpatialState.selected;

    // ── Species picker ──────────────────────────────────
    var html = '\\x3cdiv class="spatial-picker">';
    html += '\\x3clabel>Species\\x3c/label>';
    html += '\\x3cbutton type="button" class="sp-btn' + (selected === '__all__' ? ' active' : '') + '" data-sp="__all__">All species\\x3c/button>';
    var quick = speciesList.slice(0, 10);
    quick.forEach(function(s) {{
        var act = (selected === s.species) ? ' active' : '';
        html += '\\x3cbutton type="button" class="sp-btn' + act + '" data-sp="' + esc(s.species) + '" title="' +
            esc(s.species) + '\\nconf ' + s.conf_avg.toFixed(3) + ' · ' + s.detections + ' det">' +
            esc(s.species) + '\\x3c/button>';
    }});
    html += '\\x3cselect class="sp-select" id="labirSpeciesSelect" title="All LabIR species">';
    html += '\\x3coption value="__all__"' + (selected === '__all__' ? ' selected' : '') + '>All species (' +
        (spatial.detections || 0) + ' det)\\x3c/option>';
    speciesList.forEach(function(s) {{
        html += '\\x3coption value="' + esc(s.species) + '"' +
            (selected === s.species ? ' selected' : '') + '>' +
            esc(s.species) + ' — ' + s.conf_avg.toFixed(3) + ' · ' + s.detections + ' det\\x3c/option>';
    }});
    html += '\\x3c/select>\\x3c/div>';

    html += confColorLegendHtml(scaleMin, scaleMax);

    html += '\\x3cdiv class="spatial-stats">' +
        '\\x3cspan class="pill">View \\x3cstrong>' + esc(view.label) + '\\x3c/strong>\\x3c/span>' +
        '\\x3cspan class="pill">LabIR rows \\x3cstrong>' + view.n_rows.toLocaleString() + '\\x3c/strong>\\x3c/span>' +
        '\\x3cspan class="pill">Detections \\x3cstrong>' + view.detections.toLocaleString() + '\\x3c/strong>\\x3c/span>' +
        '\\x3cspan class="pill">Directions \\x3cstrong>' + view.n_directions.toLocaleString() + '\\x3c/strong>\\x3c/span>' +
        '\\x3c/div>';

    var byAz = view.by_azimuth.slice();
    var byEl = view.by_elevation.slice();
    var cells = view.cells || [];
    var maxDetAz = 1;
    byAz.forEach(function(a) {{ if (a.detections > maxDetAz) maxDetAz = a.detections; }});
    var maxDetEl = 1;
    byEl.forEach(function(e) {{ if (e.detections > maxDetEl) maxDetEl = e.detections; }});

    html += '\\x3cdiv class="spatial-grid">';

    // ── Polar rose ──────────────────────────────────────
    var size = 380;
    var cx = size / 2, cy = size / 2;
    var rMax = 138;
    var rMin = 32;
    html += '\\x3cdiv class="spatial-panel">';
    html += '\\x3ch3>Azimuth rose\\x3c/h3>';
    html += '\\x3cdiv class="sub">0° top, clockwise. <strong>Length</strong> = detections · <strong>colour</strong> = mean conf (fixed 0.55–1.00). White number inside wedge.\\x3c/div>';
    html += '\\x3cdiv class="spatial-polar-wrap">\\x3csvg width="' + size + '" height="' + size + '" viewBox="0 0 ' + size + ' ' + size + '">';

    for (var ring = 1; ring <= 3; ring++) {{
        var rr = rMin + (rMax - rMin) * (ring / 3);
        html += '\\x3ccircle cx="' + cx + '" cy="' + cy + '" r="' + rr + '" fill="none" stroke="var(--border-light)" stroke-width="1"/>';
    }}
    html += '\\x3ccircle cx="' + cx + '" cy="' + cy + '" r="' + rMin + '" fill="var(--bg-input)" stroke="var(--border)" stroke-width="1"/>';

    for (var azg = 0; azg < 360; azg += 60) {{
        var radG = (azg * Math.PI / 180);
        var x2 = cx + Math.sin(radG) * rMax;
        var y2 = cy - Math.cos(radG) * rMax;
        html += '\\x3cline x1="' + cx + '" y1="' + cy + '" x2="' + x2 + '" y2="' + y2 + '" stroke="var(--border-light)" stroke-width="1"/>';
        var lx = cx + Math.sin(radG) * (rMax + 20);
        var ly = cy - Math.cos(radG) * (rMax + 20);
        html += '\\x3ctext x="' + lx + '" y="' + (ly + 4) + '" text-anchor="middle" font-size="13" font-weight="600" fill="var(--text-muted)">' +
            String(azg).padStart(3, '0') + '°\\x3c/text>';
    }}

    function pt(ang, r) {{
        return [cx + Math.sin(ang) * r, cy - Math.cos(ang) * r];
    }}

    byAz.forEach(function(a) {{
        var az = a.azimuth;
        var frac = Math.max(0.12, a.detections / maxDetAz); // min wedge so label fits
        var rOut = rMin + (rMax - rMin) * frac;
        var halfW = 20;
        var a0 = (az - halfW) * Math.PI / 180;
        var a1 = (az + halfW) * Math.PI / 180;
        var p0 = pt(a0, rMin), p1 = pt(a0, rOut), p2 = pt(a1, rOut), p3 = pt(a1, rMin);
        var d = 'M' + p0[0] + ',' + p0[1] +
            ' L' + p1[0] + ',' + p1[1] +
            ' A' + rOut + ',' + rOut + ' 0 0,1 ' + p2[0] + ',' + p2[1] +
            ' L' + p3[0] + ',' + p3[1] +
            ' A' + rMin + ',' + rMin + ' 0 0,0 ' + p0[0] + ',' + p0[1] + ' Z';
        var col = labirConfColor(a.conf_avg, scaleMin, scaleMax);
        html += '\\x3cpath d="' + d + '" fill="' + col + '" opacity="0.95" stroke="rgba(0,0,0,0.25)" stroke-width="1">';
        html += '\\x3ctitle>az ' + String(az).padStart(3,'0') + '°\\ndetections=' + a.detections +
            '\\nmean conf=' + a.conf_avg.toFixed(3) +
            '\\nmedian conf=' + a.conf_median.toFixed(3) + '\\x3c/title>';
        html += '\\x3c/path>';
        // count label INSIDE wedge, white
        var rLab = (rMin + rOut) / 2;
        var mid = pt(az * Math.PI / 180, rLab);
        html += '\\x3ctext x="' + mid[0] + '" y="' + (mid[1] + 5) + '" text-anchor="middle" font-size="14" font-weight="700" fill="#ffffff" stroke="rgba(0,0,0,0.35)" stroke-width="0.6" paint-order="stroke">' +
            a.detections + '\\x3c/text>';
    }});

    html += '\\x3ctext x="' + cx + '" y="' + (cy + 5) + '" text-anchor="middle" font-size="12" font-weight="600" fill="var(--text-muted)">az\\x3c/text>';
    html += '\\x3c/svg>\\x3c/div>\\x3c/div>';

    // ── Elevation ───────────────────────────────────────
    html += '\\x3cdiv class="spatial-panel">';
    html += '\\x3ch3>Elevation (LabIR speakers)\\x3c/h3>';
    html += '\\x3cdiv class="sub">Length = detections · colour = mean conf (same fixed scale). S01 −45° · S05 0° · S09 +45° · S12 +90°.\\x3c/div>';
    if (!byEl.length) {{
        html += '\\x3cdiv style="color:var(--text-muted)">No elevation data.\\x3c/div>';
    }} else {{
        html += '\\x3cdiv class="bar-list">';
        byEl.slice().sort(function(a, b) {{ return b.elevation - a.elevation; }}).forEach(function(e) {{
            var pct = (e.detections / maxDetEl) * 100;
            var elevLabel = (e.elevation > 0 ? '+' : '') + e.elevation + '°';
            var col = labirConfColor(e.conf_avg, scaleMin, scaleMax);
            html += '\\x3cdiv class="bar-row" title="elev ' + elevLabel + '\\ndetections=' + e.detections + '\\nmean conf=' + e.conf_avg.toFixed(3) + '">' +
                '\\x3cdiv class="bar-label" style="width:72px">' + elevLabel + '\\x3c/div>' +
                '\\x3cdiv class="bar-track">\\x3cdiv class="bar-fill" style="width:' + pct + '%;background:' + col + '">\\x3c/div>\\x3c/div>' +
                '\\x3cdiv class="bar-value" style="width:150px">' + e.detections + ' det · ' + e.conf_avg.toFixed(3) + '\\x3c/div>' +
                '\\x3c/div>';
        }});
        html += '\\x3c/div>';
    }}
    html += '\\x3c/div>\\x3c/div>'; // panel + grid

    // ── Az × Elev grid ──────────────────────────────────
    var azSet = {{}}, elSet = {{}};
    cells.forEach(function(c) {{ azSet[c.azimuth] = true; elSet[c.elevation] = true; }});
    // Always show full LabIR ring if overall had them
    (spatial.by_azimuth || []).forEach(function(a) {{ azSet[a.azimuth] = true; }});
    (spatial.by_elevation || []).forEach(function(e) {{ elSet[e.elevation] = true; }});
    var azList = Object.keys(azSet).map(Number).sort(function(a, b) {{ return a - b; }});
    var elList = Object.keys(elSet).map(Number).sort(function(a, b) {{ return b - a; }});
    var cellMap = {{}};
    cells.forEach(function(c) {{ cellMap[c.azimuth + '|' + c.elevation] = c; }});

    html += '\\x3cdiv class="spatial-panel" style="margin-top:1.1rem">';
    html += '\\x3ch3>Azimuth × Elevation grid\\x3c/h3>';
    html += '\\x3cdiv class="sub">White number = detections. Colour = mean conf (fixed scale). Blank = no winning beam for this view.\\x3c/div>';
    if (!azList.length || !elList.length) {{
        html += '\\x3cdiv style="color:var(--text-muted)">No cells.\\x3c/div>';
    }} else {{
        html += '\\x3cdiv class="heatmap-wrap">\\x3ctable class="heatmap">\\x3cthead>\\x3ctr>\\x3cth>Elev \\\\ Az\\x3c/th>';
        azList.forEach(function(az) {{
            html += '\\x3cth>' + String(az).padStart(3, '0') + '°\\x3c/th>';
        }});
        html += '\\x3c/tr>\\x3c/thead>\\x3ctbody>';
        elList.forEach(function(el) {{
            var elevLabel = (el > 0 ? '+' : '') + el + '°';
            html += '\\x3ctr>\\x3ctd style="font-weight:600">' + elevLabel + '\\x3c/td>';
            azList.forEach(function(az) {{
                var c = cellMap[az + '|' + el];
                if (!c) {{
                    html += '\\x3ctd class="nil">&mdash;\\x3c/td>';
                    return;
                }}
                var bg = labirConfColor(c.conf_avg, scaleMin, scaleMax);
                var sp = (c.speaker != null) ? ('S' + String(c.speaker).padStart(2, '0')) : '';
                html += '\\x3ctd style="background:' + bg + ';color:#fff;font-weight:700;text-shadow:0 1px 2px rgba(0,0,0,0.45)" title="' +
                    'az ' + String(az).padStart(3,'0') + '° elev ' + elevLabel +
                    (sp ? (' ' + sp) : '') +
                    '\\ndetections=' + c.detections +
                    '\\nspecies=' + c.species_count +
                    '\\nmean conf=' + c.conf_avg.toFixed(3) +
                    '\\nmax conf=' + c.conf_max.toFixed(3) + '">' +
                    c.detections + '\\x3c/td>';
            }});
            html += '\\x3c/tr>';
        }});
        html += '\\x3c/tbody>\\x3c/table>\\x3c/div>';
    }}
    html += '\\x3c/div>';

    container.innerHTML = html;

    // wire picker
    container.querySelectorAll('.sp-btn').forEach(function(btn) {{
        btn.addEventListener('click', function() {{
            labirSpatialState.selected = btn.getAttribute('data-sp') || '__all__';
            _paintLabirSpatial();
        }});
    }});
    var sel = document.getElementById('labirSpeciesSelect');
    if (sel) {{
        sel.addEventListener('change', function() {{
            labirSpatialState.selected = sel.value || '__all__';
            _paintLabirSpatial();
        }});
    }}
}}

// ── Confidence Distribution (boxplot | histogram side-by-side) ──
function renderConfRange(methods) {{
    var container = document.getElementById('confRange');
    var entries = orderedMethods(Object.keys(methods)).map(function(m) {{
        return [m, methods[m]];
    }});
    if (!entries.length) {{ container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No data\\x3c/div>'; return; }}

    // Fixed BirdNET scale so methods are comparable and IQR collapse is visible
    var scaleMin = 0.40, scaleMax = 1.00;
    var scaleRange = scaleMax - scaleMin;

    function xOf(v) {{
        return ((Math.max(scaleMin, Math.min(scaleMax, v)) - scaleMin) / scaleRange);
    }}

    var rowH = 42;
    var margin = {{ top: 22, right: 14, bottom: 34, left: 64 }};
    // Wide viewBox; SVG scales to 100% of panel width
    var plotW = 520;
    var plotH = entries.length * rowH;
    var svgW = margin.left + plotW + margin.right;
    var svgH = margin.top + plotH + margin.bottom;

    var html = '\\x3cdiv class="conf-split">';

    // ── Left: box / whiskers ─────────────────────────────
    html += '\\x3cdiv class="conf-panel">';
    html += '\\x3cdiv class="conf-axis-note">Box plot (min–IQR–median–mean–max)\\x3c/div>';
    html += '\\x3cdiv class="conf-boxplot-wrap">';
    html += '\\x3csvg viewBox="0 0 ' + svgW + ' ' + svgH + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Confidence box plot by method">';

    for (var t = 0; t <= 6; t++) {{
        var tv = scaleMin + (scaleRange * t / 6);
        var tx = margin.left + xOf(tv) * plotW;
        html += '\\x3cline x1="' + tx + '" y1="' + margin.top + '" x2="' + tx + '" y2="' + (margin.top + plotH) + '" stroke="var(--border-light)" stroke-dasharray="2,3"/>';
        html += '\\x3ctext x="' + tx + '" y="' + (margin.top + plotH + 20) + '" text-anchor="middle" font-size="11" fill="var(--text-muted)">' + tv.toFixed(2) + '\\x3c/text>';
    }}
    var thrX = margin.left + xOf(0.50) * plotW;
    html += '\\x3cline x1="' + thrX + '" y1="' + (margin.top - 4) + '" x2="' + thrX + '" y2="' + (margin.top + plotH) + '" stroke="var(--text-muted)" stroke-width="1" stroke-dasharray="4,3" opacity="0.55"/>';
    html += '\\x3ctext x="' + thrX + '" y="' + (margin.top - 8) + '" text-anchor="middle" font-size="11" fill="var(--text-muted)">0.50\\x3c/text>';

    entries.forEach(function(e, i) {{
        var method = e[0], d = e[1];
        var color = METHOD_COLORS[method] || '#6b7280';
        var cy = margin.top + i * rowH + rowH / 2;
        var xMin = margin.left + xOf(d.conf_min) * plotW;
        var xQ25 = margin.left + xOf(d.conf_q25) * plotW;
        var xMed = margin.left + xOf(d.conf_median) * plotW;
        var xAvg = margin.left + xOf(d.conf_avg) * plotW;
        var xQ75 = margin.left + xOf(d.conf_q75) * plotW;
        var xMax = margin.left + xOf(d.conf_max) * plotW;
        var boxW = Math.max(2, xQ75 - xQ25);
        var boxH = 16;

        html += '\\x3ctext x="' + (margin.left - 8) + '" y="' + (cy + 5) + '" text-anchor="end" font-size="13" font-weight="600" fill="' + color + '">' + esc(method) + '\\x3c/text>';
        html += '\x3cline x1="' + xMin + '" y1="' + cy + '" x2="' + xMax + '" y2="' + cy + '" stroke="' + color + '" stroke-width="1.5" opacity="0.85"/>';
        html += '\\x3cline x1="' + xMin + '" y1="' + (cy - 6) + '" x2="' + xMin + '" y2="' + (cy + 6) + '" stroke="' + color + '" stroke-width="1.5"/>';
        html += '\\x3cline x1="' + xMax + '" y1="' + (cy - 6) + '" x2="' + xMax + '" y2="' + (cy + 6) + '" stroke="' + color + '" stroke-width="1.5"/>';
        html += '\x3crect x="' + xQ25 + '" y="' + (cy - boxH / 2) + '" width="' + boxW + '" height="' + boxH + '" rx="2" fill="' + color + '" opacity="0.85" stroke="' + color + '" stroke-width="1.2"/>';
        html += '\\x3cline x1="' + xMed + '" y1="' + (cy - boxH / 2 - 1) + '" x2="' + xMed + '" y2="' + (cy + boxH / 2 + 1) + '" stroke="' + color + '" stroke-width="2.5"/>';
        var ds = 5;
        html += '\\x3cpolygon points="' + xAvg + ',' + (cy - ds) + ' ' + (xAvg + ds) + ',' + cy + ' ' + xAvg + ',' + (cy + ds) + ' ' + (xAvg - ds) + ',' + cy + '" fill="' + color + '" stroke="var(--bg-card)" stroke-width="1"/>';
        html += '\\x3ctitle>' + esc(method) +
            '\\nmin=' + d.conf_min.toFixed(3) +
            '  q25=' + d.conf_q25.toFixed(3) +
            '  med=' + d.conf_median.toFixed(3) +
            '  avg=' + d.conf_avg.toFixed(3) +
            '  q75=' + d.conf_q75.toFixed(3) +
            '  max=' + d.conf_max.toFixed(3) +
            '\\nn=' + d.total.toLocaleString() + '\\x3c/title>';
    }});

    html += '\\x3c/svg>\\x3c/div>';
    html += '\\x3cdiv class="conf-legend">' +
        '\\x3cspan>whiskers min–max\\x3c/span>' +
        '\\x3cspan>box IQR\\x3c/span>' +
        '\\x3cspan>line median\\x3c/span>' +
        '\\x3cspan>◆ mean\\x3c/span>' +
        '\\x3c/div>';
    html += '\\x3c/div>'; // conf-panel left

    // ── Right: histogram ────────────────────────────────
    html += '\\x3cdiv class="conf-panel">';
    var hasHist = entries.every(function(e) {{ return e[1].hist && e[1].hist.length; }});
    if (hasHist) {{
        var hist = entries[0][1].hist;
        var nBins = hist.length;
        var maxBin = 0;
        entries.forEach(function(e) {{
            e[1].hist.forEach(function(b) {{ maxBin = Math.max(maxBin, b.count); }});
        }});
        var hMargin = {{ top: 22, right: 14, bottom: 42, left: 44 }};
        var hPlotW = 520;
        var hPlotH = Math.max(plotH, 160);
        var hSvgW = hMargin.left + hPlotW + hMargin.right;
        var hSvgH = hMargin.top + hPlotH + hMargin.bottom;
        var groupW = hPlotW / nBins;
        var barW = Math.max(2, (groupW - 3) / entries.length);

        html += '\\x3cdiv class="conf-axis-note">Score histogram (detections per 0.05 bin)\\x3c/div>';
        html += '\\x3cdiv class="conf-hist-wrap">\\x3csvg viewBox="0 0 ' + hSvgW + ' ' + hSvgH + '" preserveAspectRatio="xMidYMid meet" role="img" aria-label="Confidence score histogram">';

        for (var g = 0; g <= 4; g++) {{
            var gy = hMargin.top + hPlotH * (1 - g / 4);
            var gv = Math.round(maxBin * g / 4);
            html += '\\x3cline x1="' + hMargin.left + '" y1="' + gy + '" x2="' + (hMargin.left + hPlotW) + '" y2="' + gy + '" stroke="var(--border-light)" stroke-dasharray="2,3"/>';
            html += '\\x3ctext x="' + (hMargin.left - 6) + '" y="' + (gy + 4) + '" text-anchor="end" font-size="11" fill="var(--text-muted)">' + gv + '\\x3c/text>';
        }}

        hist.forEach(function(bin, bi) {{
            var gx = hMargin.left + bi * groupW;
            entries.forEach(function(e, mi) {{
                var b = e[1].hist[bi];
                var color = METHOD_COLORS[e[0]] || '#6b7280';
                var bh = maxBin > 0 ? (b.count / maxBin) * hPlotH : 0;
                var bx = gx + 1.5 + mi * barW;
                var by = hMargin.top + hPlotH - bh;
                if (b.count === 0) return;
                html += '\\x3crect x="' + bx + '" y="' + by + '" width="' + Math.max(1, barW - 0.4) + '" height="' + Math.max(1, bh) + '" fill="' + color + '" opacity="0.85" rx="1">';
                html += '\\x3ctitle>' + esc(e[0]) + ' [' + bin.lo.toFixed(2) + '&ndash;' + bin.hi.toFixed(2) + ') = ' + b.count + '\\x3c/title>';
                html += '\\x3c/rect>';
            }});
            if (bi % 2 === 0) {{
                html += '\\x3ctext x="' + (gx + groupW / 2) + '" y="' + (hMargin.top + hPlotH + 16) + '" text-anchor="middle" font-size="11" fill="var(--text-muted)">' + bin.lo.toFixed(2) + '\\x3c/text>';
            }}
        }});
        html += '\\x3ctext x="' + (hMargin.left + hPlotW / 2) + '" y="' + (hSvgH - 6) + '" text-anchor="middle" font-size="12" fill="var(--text-muted)">confidence\\x3c/text>';
        html += '\\x3c/svg>\\x3c/div>';
        // method colour legend under hist
        html += '\\x3cdiv class="conf-legend">';
        entries.forEach(function(e) {{
            var color = METHOD_COLORS[e[0]] || '#6b7280';
            html += '\\x3cspan>\\x3cspan style="width:12px;height:12px;border-radius:2px;background:' + color + ';display:inline-block">\\x3c/span>' + esc(e[0]) + '\\x3c/span>';
        }});
        html += '\\x3c/div>';
    }} else {{
        html += '\\x3cdiv style="color:var(--text-muted)">No histogram bins.\\x3c/div>';
    }}
    html += '\\x3c/div>'; // conf-panel right
    html += '\\x3c/div>'; // conf-split

    // Stats table (full width under split)
    html += '\\x3cdiv style="overflow-x:auto;margin-top:0.65rem">\\x3ctable class="heatmap" style="width:auto;min-width:100%">\\x3cthead>\\x3ctr>';
    html += '\\x3cth>Method\\x3c/th>\\x3cth>n\\x3c/th>\\x3cth>min\\x3c/th>\\x3cth>q25\\x3c/th>\\x3cth>median\\x3c/th>\\x3cth>mean\\x3c/th>\\x3cth>q75\\x3c/th>\\x3cth>max\\x3c/th>\\x3c/tr>\\x3c/thead>\\x3ctbody>';
    entries.forEach(function(e) {{
        var method = e[0], d = e[1];
        var color = METHOD_COLORS[method] || '#6b7280';
        html += '\\x3ctr>\\x3ctd style="color:' + color + ';font-weight:600">' + esc(method) + '\\x3c/td>';
        html += '\\x3ctd>' + d.total.toLocaleString() + '\\x3c/td>';
        html += '\\x3ctd>' + d.conf_min.toFixed(3) + '\\x3c/td>';
        html += '\\x3ctd>' + d.conf_q25.toFixed(3) + '\\x3c/td>';
        html += '\\x3ctd style="font-weight:600">' + d.conf_median.toFixed(3) + '\\x3c/td>';
        html += '\\x3ctd>' + d.conf_avg.toFixed(3) + '\\x3c/td>';
        html += '\\x3ctd>' + d.conf_q75.toFixed(3) + '\\x3c/td>';
        html += '\\x3ctd>' + d.conf_max.toFixed(3) + '\\x3c/td>\\x3c/tr>';
    }});
    html += '\\x3c/tbody>\\x3c/table>\\x3c/div>';

    container.innerHTML = html;
}}

// ── Heatmap (colour synced with LabIR Spatial: 0.55–1.00) ─
function renderHeatmap(rows, methods) {{
    var container = document.getElementById('heatmapContainer');
    if (!rows.length) {{ container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No data\\x3c/div>'; return; }}

    // Same absolute scale as LabIR Spatial Directions
    var scaleMin = 0.55, scaleMax = 1.0;
    var allAvgs = [];
    rows.forEach(function(r) {{
        methods.forEach(function(m) {{
            if (r[m]) allAvgs.push(r[m].conf_avg);
        }});
    }});
    var minC = allAvgs.length ? Math.min.apply(null, allAvgs) : scaleMin;
    var maxC = allAvgs.length ? Math.max.apply(null, allAvgs) : scaleMax;

    var html = confColorLegendHtml(scaleMin, scaleMax);

    html += '\\x3ctable class="heatmap">\\x3cthead>\\x3ctr>\\x3cth>Species\\x3c/th>';
    methods.forEach(function(m) {{
        html += '\\x3cth style="color:' + (METHOD_COLORS[m] || '#9ca3af') + '">' + esc(m) + '\\x3c/th>';
    }});
    html += '\\x3cth style="font-weight:400;color:var(--text-muted)">Detections\\x3c/th>\\x3c/tr>\\x3c/thead>\\x3ctbody>';

    rows.forEach(function(r) {{
        // Unique BirdNET detections — do NOT sum method cell counts
        var dets = (typeof r.unique_count === 'number') ? r.unique_count : 0;
        if (!dets) {{
            var seen = 0;
            methods.forEach(function(m) {{ if (r[m]) seen = Math.max(seen, r[m].count); }});
            dets = seen;
        }}
        html += '\\x3ctr>\\x3ctd title="' + esc(r.species) + '">' + esc(r.species) + '\\x3c/td>';
        methods.forEach(function(m) {{
            if (r[m]) {{
                var c = r[m].conf_avg;
                var bg = labirConfColor(c, scaleMin, scaleMax);
                html += '\\x3ctd style="background:' + bg + ';color:#fff;font-weight:700;text-shadow:0 1px 2px rgba(0,0,0,0.45)" title="' +
                        esc(m) + ' | ' + esc(r.species) + '\\nmethod rows=' + r[m].count + '\\navg conf=' + c.toFixed(3) + '">' +
                        c.toFixed(3) + '\\x3c/td>';
            }} else {{
                html += '\\x3ctd class="nil">&mdash;\\x3c/td>';
            }}
        }});
        html += '\\x3ctd style="font-weight:600" title="Unique detections (source × start time)">' + dets + '\\x3c/td>\\x3c/tr>';
    }});

    html += '\\x3c/tbody>\\x3c/table>';
    container.innerHTML = html;
}}

// ── Top species (column-major rank + conf donut 0.55–1.00) ─
function renderSpeciesBars(speciesList) {{
    var container = document.getElementById('speciesBarList');
    if (!speciesList.length) {{ container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No data\\x3c/div>'; return; }}

    // Synced with LabIR Spatial / heatmap
    var scaleMin = 0.55, scaleMax = 1.0;
    var scaleSpan = scaleMax - scaleMin;

    function confDonutSvg(conf, color) {{
        var t = Math.max(0, Math.min(1, (conf - scaleMin) / scaleSpan));
        var size = 58, cx = 29, cy = 29, r = 22, sw = 6.5;
        var circ = 2 * Math.PI * r;
        var dash = t * circ;
        // Track + arc (start at top: rotate -90°)
        return '\\x3csvg class="sp-rank-donut" viewBox="0 0 ' + size + ' ' + size + '" aria-hidden="true">' +
            '\\x3ccircle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="var(--border)" stroke-width="' + sw + '"/>' +
            '\\x3ccircle cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="none" stroke="' + color + '" stroke-width="' + sw + '"' +
            ' stroke-linecap="round" stroke-dasharray="' + dash.toFixed(2) + ' ' + circ.toFixed(2) + '"' +
            ' transform="rotate(-90 ' + cx + ' ' + cy + ')"/>' +
            '\\x3ctext x="' + cx + '" y="' + (cy + 4) + '" text-anchor="middle" font-size="11" font-weight="700" fill="var(--text-heading)">' +
            conf.toFixed(2) + '\\x3c/text>' +
            '\\x3c/svg>';
    }}

    var html = '\\x3cdiv class="species-rank-wrap">';
    html += confColorLegendHtml(scaleMin, scaleMax);

    var show = Math.min(speciesList.length, 20);
    // Flex wrap L→R, top-aligned; items grow to fill each row full-width
    html += '\\x3cdiv class="species-rank">';
    for (var i = 0; i < show; i++) {{
        var sp = speciesList[i];
        var col = labirConfColor(sp.conf_avg, scaleMin, scaleMax);
        var rawNote = (typeof sp.raw_count === 'number') ? ('\\nMethod-rows: ' + sp.raw_count) : '';
        var tip = esc(sp.species) + '\\nmean conf=' + sp.conf_avg.toFixed(3) +
            ' (donut 0.55–1.00)\\ndetections=' + sp.count + rawNote +
            '\\nmethods=' + sp.method_count + '\\nrank=' + (i + 1);

        html += '\\x3cdiv class="sp-rank-item" title="' + tip + '">' +
            '\\x3cdiv class="sp-rank-num">' + (i + 1) + '\\x3c/div>' +
            confDonutSvg(sp.conf_avg, col) +
            '\\x3cdiv class="sp-rank-body">' +
                '\\x3cdiv class="sp-rank-name">' + esc(sp.species) + '\\x3c/div>' +
                '\\x3cdiv class="sp-rank-meta">\\x3cstrong>' + sp.conf_avg.toFixed(3) + '\\x3c/strong> · ' +
                    sp.count.toLocaleString() + ' det\\x3c/div>' +
            '\\x3c/div>' +
            '\\x3c/div>';
    }}
    html += '\\x3c/div>'; // species-rank
    if (speciesList.length > show) {{
        html += '\\x3cdiv style="font-size:0.88rem;color:var(--text-muted);margin-top:0.4rem">' +
                '+ ' + (speciesList.length - show) + ' more species\\x3c/div>';
    }}
    html += '\\x3c/div>'; // species-rank-wrap
    container.innerHTML = html;
}}

// ── Hourly Bars (detection windows, not species×time rows) ──
function renderHourlyBars(hourlyList) {{
    var container = document.getElementById('hourlyBarList');
    if (!hourlyList.length) {{ container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No data\\x3c/div>'; return; }}

    var max = Math.max.apply(null, hourlyList.map(function(h) {{ return h.count; }})) || 1;
    var html = '';
    for (var i = 0; i < hourlyList.length; i++) {{
        var h = hourlyList[i];
        var pct = (h.count / max) * 100;
        var spp = (typeof h.species_count === 'number') ? h.species_count : null;
        var value = h.count.toLocaleString() + ' det';
        if (spp !== null) value += ' &middot; ' + spp + ' spp';
        var tip = 'BirdNET detections (unique source × start): ' + h.count;
        if (spp !== null) tip += '\\nSpecies detected: ' + spp;
        html += '\\x3cdiv class="bar-row">' +
            '\\x3cdiv class="bar-label" title="' + tip + '">' + esc(h.hour) + ':00\\x3c/div>' +
            '\\x3cdiv class="bar-track">\\x3cdiv class="bar-fill" style="width:' + pct + '%;background:hsl(200,60%,55%)">\\x3c/div>\\x3c/div>' +
            '\\x3cdiv class="bar-value" title="' + tip + '">' + value + '\\x3c/div>' +
        '\\x3c/div>';
    }}
    container.innerHTML = html;
}}

// ── Source Count Chart (grouped bars per source) ─────────
function renderSrcCountChart(data) {{
    var container = document.getElementById('srcCountChart');
    if (!data || !data.length) {{ container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No data\x3c/div>'; return; }}

    var methods = orderedMethods(Object.keys(data[0].methods));
    var n = data.length;
    var barW = Math.max(6, Math.min(14, Math.floor(600 / n / methods.length)));
    var gap = 2;
    var groupW = methods.length * (barW + gap);
    var margin = {{ top: 24, right: 20, bottom: 36, left: 36 }};

    // Find max
    var maxCount = 0;
    data.forEach(function(r) {{
        methods.forEach(function(m) {{ maxCount = Math.max(maxCount, r.methods[m]); }});
    }});
    var chartW = n * (groupW + 10) + margin.left + margin.right;
    var chartH = 220;

    var html = '\\x3csvg width="' + chartW + '" height="' + chartH + '" viewBox="0 0 ' + chartW + ' ' + chartH + '" style="display:block">';

    // Y-axis grid
    for (var g = 0; g <= 4; g++) {{
        var y = margin.top + (chartH - margin.top - margin.bottom) * (1 - g / 4);
        html += '\\x3cline x1="' + margin.left + '" y1="' + y + '" x2="' + (chartW - margin.right) + '" y2="' + y + '" stroke="var(--border-light)" stroke-dasharray="3,3"/>';
        var label = Math.round(maxCount * g / 4);
        html += '\\x3ctext x="' + (margin.left - 6) + '" y="' + (y + 4) + '" text-anchor="end" font-size="12" fill="var(--text-muted)">' + label + '\\x3c/text>';
    }}

    data.forEach(function(r, i) {{
        var gx = margin.left + i * (groupW + 10);
        methods.forEach(function(m, j) {{
            var val = r.methods[m];
            if (val === 0) return;
            var h = maxCount > 0 ? (val / maxCount) * (chartH - margin.top - margin.bottom) : 0;
            var bx = gx + j * (barW + gap);
            var by = chartH - margin.bottom - h;
            var color = METHOD_COLORS[m] || '#6b7280';
            var spp = (r.species && typeof r.species[m] === 'number') ? r.species[m] : null;
            var tip = esc(m) + ' | ' + esc(r.label) + ' = ' + val + ' detection' + (val === 1 ? '' : 's');
            if (spp !== null) tip += ' · ' + spp + ' spp';
            html += '\x3cg class="chart-bar-group">' +
                '\x3crect x="' + bx + '" y="' + by + '" width="' + barW + '" height="' + Math.max(1, h) + '" fill="' + color + '" rx="1" opacity="0.85"/>' +
                '\x3ctitle>' + tip + '\x3c/title>\x3c/g>';
        }});
        // X-axis label (horizontal HH:MM)
        if (n <= 40 || i % Math.ceil(n / 30) === 0) {{
            html += '\\x3ctext x="' + (gx + groupW / 2) + '" y="' + (chartH - 12) + '" text-anchor="middle" font-size="12" fill="var(--text-muted)">' + esc(r.label) + '\\x3c/text>';
        }}
    }});

    html += '\\x3c/svg>';

    // Legend
    html += '\\x3cdiv class="chart-legend">';
    methods.forEach(function(m) {{
        html += '\\x3cspan>\x3cdiv class="swatch" style="background:' + (METHOD_COLORS[m] || '#6b7280') + '">\x3c/div>' + esc(m) + '\\x3c/span>';
    }});
    html += '\\x3c/div>';

    container.innerHTML = html;
}}

// ── Source Confidence Chart (mean per method per source) ──
function renderSrcConfChart(data) {{
    var container = document.getElementById('srcConfChart');
    if (!data || !data.length) {{ container.innerHTML = '\\x3cdiv style="color:var(--text-muted)">No data\x3c/div>'; return; }}

    var methods = orderedMethods(Object.keys(data[0].methods));
    var n = data.length;
    var barW = Math.max(6, Math.min(14, Math.floor(600 / n / methods.length)));
    var gap = 2;
    var groupW = methods.length * (barW + gap);
    var margin = {{ top: 24, right: 20, bottom: 36, left: 48 }};

    // Find max avg confidence
    var maxConf = 0;
    var minConf = Infinity;
    data.forEach(function(r) {{
        methods.forEach(function(m) {{
            var a = r.methods[m].conf_avg;
            if (a > 0) {{ maxConf = Math.max(maxConf, a); minConf = Math.min(minConf, a); }}
        }});
    }});
    if (!isFinite(minConf)) {{ minConf = 0.4; maxConf = 1.0; }}
    var yMin = Math.floor(minConf * 10) / 10;
    var yMax = Math.ceil(maxConf * 10) / 10;
    var yr = yMax - yMin || 0.001;

    var chartW = n * (groupW + 10) + margin.left + margin.right;
    var chartH = 220;

    var html = '\\x3csvg width="' + chartW + '" height="' + chartH + '" viewBox="0 0 ' + chartW + ' ' + chartH + '" style="display:block">';

    // Y-axis grid
    for (var g = 0; g <= 4; g++) {{
        var y = margin.top + (chartH - margin.top - margin.bottom) * (1 - g / 4);
        var val = yMin + yr * g / 4;
        html += '\\x3cline x1="' + margin.left + '" y1="' + y + '" x2="' + (chartW - margin.right) + '" y2="' + y + '" stroke="var(--border-light)" stroke-dasharray="3,3"/>';
        html += '\\x3ctext x="' + (margin.left - 6) + '" y="' + (y + 4) + '" text-anchor="end" font-size="12" fill="var(--text-muted)">' + val.toFixed(1) + '\\x3c/text>';
    }}

    data.forEach(function(r, i) {{
        var gx = margin.left + i * (groupW + 10);
        methods.forEach(function(m, j) {{
            var d = r.methods[m];
            if (!d || d.conf_avg === 0) return;
            var h = ((d.conf_avg - yMin) / yr) * (chartH - margin.top - margin.bottom);
            var bx = gx + j * (barW + gap);
            var by = chartH - margin.bottom - h;
            var color = METHOD_COLORS[m] || '#6b7280';
            html += '\x3cg class="chart-bar-group">' +
                '\x3crect x="' + bx + '" y="' + by + '" width="' + barW + '" height="' + Math.max(1, h) + '" fill="' + color + '" rx="1" opacity="0.85"/>' +
                '\x3ctitle>' + esc(m) + ' | ' + esc(r.label) + ' = ' + d.conf_avg.toFixed(3) + '\x3c/title>\x3c/g>';
        }});
        if (n <= 40 || i % Math.ceil(n / 30) === 0) {{
            html += '\\x3ctext x="' + (gx + groupW / 2) + '" y="' + (chartH - 12) + '" text-anchor="middle" font-size="12" fill="var(--text-muted)">' + esc(r.label) + '\\x3c/text>';
        }}
    }});

    html += '\\x3c/svg>';

    // Legend
    html += '\\x3cdiv class="chart-legend">';
    methods.forEach(function(m) {{
        html += '\\x3cspan>\x3cdiv class="swatch" style="background:' + (METHOD_COLORS[m] || '#6b7280') + '">\x3c/div>' + esc(m) + '\\x3c/span>';
    }});
    html += '\\x3c/div>';

    container.innerHTML = html;
}}

// ── Init (after all function declarations + state vars) ──
(function init() {{
    var dates = Object.keys(SUMMARIES).sort();
    if (!dates.length) return;
    renderDateSelector(dates);
    selectDate(dates[0]);
}})();
</script>
</body>
</html>"""
    return html


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Generate a visualisation-only HTML dashboard.")
    parser.add_argument("--location", type=str, required=True)
    parser.add_argument("--data-dir", type=str, default=ANALYSIS_OUTPUT)
    parser.add_argument("--dates", type=str, default=None, help="Comma-separated dates")
    parser.add_argument("--output", type=str, default=None, help="Output HTML file path")
    parser.add_argument("--no-compress", action="store_true", help="Do not gzip detection JSON")
    parser.add_argument("--open", action="store_true", help="Open in browser")
    parser.add_argument(
        "--dashboard-min-conf", type=float, default=DASHBOARD_MIN_CONF,
        help=f"Dashboard only: keep confidence > this value (default {DASHBOARD_MIN_CONF}). "
             "Does not change BirdNET pipeline min_conf (0.4).",
    )

    args = parser.parse_args()
    dash_min = float(args.dashboard_min_conf)

    date_filter = [d.strip() for d in args.dates.split(",")] if args.dates else None

    print(f"Scanning {args.data_dir}/{args.location}/ ...")
    processed_files = find_processed_json_files(args.data_dir, args.location, date_filter)
    print(f"Found {len(processed_files)} processed.json file(s)")

    detections = collect_detections(processed_files)
    print(f"Collected {len(detections)} detection(s) across {len(set(d.species for d in detections))} species")
    print(f"Dashboard filter: confidence > {dash_min:g} (pipeline BirdNET min_conf unchanged)")

    # ── Build per-date summaries (dashboard-filtered) ───
    by_date: Dict[str, List[Detection]] = defaultdict(list)
    for d in detections:
        by_date[d.date].append(d)

    summaries = {}
    for date_str in sorted(by_date.keys()):
        raw = by_date[date_str]
        dash = filter_for_dashboard(raw, dash_min)
        print(
            f"  {date_str}: {len(dash):,} / {len(raw):,} method-rows after filter "
            f"({len(set(d.species for d in dash))} species)"
        )
        summaries[date_str] = build_date_summary(
            dash,
            dashboard_min_conf=dash_min,
            raw_row_count=len(raw),
        )

    # ── Write external JSON ─────────────────────────────
    report_data_dir = os.path.join(args.data_dir, args.location, "report_data")
    write_report_data(
        detections,
        report_data_dir,
        compress=not args.no_compress,
        dashboard_min_conf=dash_min,
        summaries=summaries,
    )
    print(f"Report data written to: {report_data_dir}")

    # ── Generate HTML ───────────────────────────────────
    html = generate_html(args.location, args.data_dir, summaries, dashboard_min_conf=dash_min)

    default_path = os.path.join(args.data_dir, args.location, f"{args.location}_report.html")
    output_path = args.output or default_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    print(f"Report written to: {abs_path}  ({len(html):,} bytes)")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
