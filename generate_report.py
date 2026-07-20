#!/usr/bin/env python3
"""
Generate a self-contained, beautiful HTML daily report from processed.json files.

Usage:
    python generate_report.py --location 2A400
    python generate_report.py --location 2A400 --dates 2026-04-16,2026-04-19

Reads ANALYSIS_OUTPUT from config, or can be overridden with --data-dir.
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from html import escape
from typing import Any, Dict, List, Optional, Tuple

try:
    from config import ANALYSIS_OUTPUT
except ImportError:
    ANALYSIS_OUTPUT = "/Volumes/HD Data/sea-data"

# ============================================================
# LABIR SPEAKER -> ELEVATION MAPPING
# ============================================================
LABIR_ELEVATION: Dict[int, int] = {
    1: -45, 2: -30, 3: -20, 4: -10, 5: 0,
    6: 10, 7: 20, 8: 30, 9: 45, 10: 60,
    11: 75, 12: 90,
}

PROCESSING_DIRS = [
    "beamforming_LabIR",
    "beamforming_SPIR1",
    "beamforming_SPIR2",
    "signal_averaging",
]

RE_LABIR = re.compile(r"_LabIR\(S(\d+)_(\d+)\)\.wav$")
RE_SPIR1 = re.compile(r"_SPIR1\((\d+)m_(\d+)\)\.wav$")
RE_SPIR2 = re.compile(r"_SPIR2\((\d+)m_180_r(\d+)\)\.wav$")
RE_SA    = re.compile(r"_sa\.wav$")


# ============================================================
# DETECTION DATA STRUCTURE
# ============================================================

class Detection:
    """A single BirdNET detection row with parsed direction fields."""
    __slots__ = (
        "date", "method", "species", "confidence",
        "start_time", "time_str",
        "distance", "azimuth", "elevation",
        "direction_label", "raw_filename",
    )

    def __init__(
        self,
        date: str,
        method: str,
        species: str,
        confidence: float,
        start_time: float,
        time_str: str,
        distance: str,
        azimuth: str,
        elevation: str,
        direction_label: str,
        raw_filename: str,
    ):
        self.date = date
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

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "method": self.method,
            "species": self.species,
            "confidence": self.confidence,
            "start_time": self.start_time,
            "time_str": self.time_str,
            "distance": self.distance,
            "azimuth": self.azimuth,
            "elevation": self.elevation,
            "direction_label": self.direction_label,
            "raw_filename": self.raw_filename,
        }


# ============================================================
# DIRECTION PARSING — returns (distance, azimuth, elevation, label)
# ============================================================

def _format_time(seconds: float) -> str:
    """Convert seconds to HH:MM:SS string."""
    s = int(seconds)
    h, m = divmod(s, 3600)
    m, s = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def parse_direction(filename: str, method: str) -> tuple:
    """
    Returns (distance, azimuth, elevation, label).
    Distance / azimuth / elevation are strings; empty if N/A.

    LabIR:  ("", "060", "-45", "S01 elev:-45 az:060")
    SPIR1:  ("2m", "300", "", "2m az:300")
    SPIR2:  ("16m", "180", "", "16m az:180 rep:3")
    SA:     ("", "", "", "omnidirectional")
    """
    if method == "LabIR":
        m = RE_LABIR.search(filename)
        if m:
            speaker = int(m.group(1))
            azimuth = int(m.group(2))
            elev = LABIR_ELEVATION.get(speaker, "?")
            az_s = f"{azimuth:03d}"
            elev_s = f"{elev:+d}"
            label = f"S{speaker:02d} elev:{elev_s}deg az:{az_s}deg"
            return ("", az_s, elev_s, label)
        return ("", "", "", filename)

    elif method == "SPIR1":
        m = RE_SPIR1.search(filename)
        if m:
            dist = int(m.group(1))
            azimuth = int(m.group(2))
            dist_s = f"{dist}m"
            az_s = f"{azimuth:03d}"
            return (dist_s, az_s, "", f"{dist_s} az:{az_s}deg")
        return ("", "", "", filename)

    elif method == "SPIR2":
        m = RE_SPIR2.search(filename)
        if m:
            dist = int(m.group(1))
            rep = int(m.group(2))
            dist_s = f"{dist}m"
            az_s = "180"
            label = f"{dist_s} az:180deg rep:{rep}"
            return (dist_s, az_s, "", label)
        return ("", "", "", filename)

    elif method == "SA":
        return ("", "", "", "omnidirectional")

    return ("", "", "", filename)


# ============================================================
# DATA COLLECTION
# ============================================================

def find_processed_json_files(
    base_dir: str,
    location: str,
    date_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """Walk tree, return [(date_str, method, full_path), ...]."""
    location_dir = os.path.join(base_dir, location)
    if not os.path.isdir(location_dir):
        print(f"Location directory not found: {location_dir}")
        return []

    found: List[Tuple[str, str, str]] = []

    for entry in sorted(os.listdir(location_dir)):
        date_path = os.path.join(location_dir, entry)
        if not os.path.isdir(date_path):
            continue
        if date_filter and entry not in date_filter:
            continue

        for subdir in PROCESSING_DIRS:
            proc_json = os.path.join(date_path, subdir, "processed.json")
            if os.path.isfile(proc_json):
                method_map = {
                    "beamforming_LabIR": "LabIR",
                    "beamforming_SPIR1": "SPIR1",
                    "beamforming_SPIR2": "SPIR2",
                    "signal_averaging": "SA",
                }
                method = method_map.get(subdir, subdir)
                found.append((entry, method, proc_json))

    return found


def collect_detections(processed_files: List[Tuple[str, str, str]]) -> List[Detection]:
    """Read processed.json files, flatten into Detection objects."""
    detections: List[Detection] = []

    for date_str, method, json_path in processed_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping {json_path}: {e}", file=sys.stderr)
            continue

        for species_name, sp_data in data.items():
            start_times = sp_data.get("start_time_list", [])
            confidences = sp_data.get("conf_list", [])
            channels = sp_data.get("primary_channel_list", [])

            for i in range(len(start_times)):
                conf = round(float(confidences[i]), 3) if i < len(confidences) else 0.0
                ch = channels[i] if i < len(channels) else ""
                start_t = round(float(start_times[i]), 1)

                dist, az, elev, label = parse_direction(ch, method)
                time_str = _format_time(start_t)

                detections.append(Detection(
                    date=date_str,
                    method=method,
                    species=species_name,
                    confidence=conf,
                    start_time=start_t,
                    time_str=time_str,
                    distance=dist,
                    azimuth=az,
                    elevation=elev,
                    direction_label=label,
                    raw_filename=ch,
                ))

    return detections


# ============================================================
# HTML GENERATION
# ============================================================

def format_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def escape_str(s: Any) -> str:
    return json.dumps(str(s))


def generate_html(location: str, detections: List[Detection], data_dir: str) -> str:
    """Generate a self-contained HTML report."""

    all_species = sorted(set(d.species for d in detections))
    all_dates = sorted(set(d.date for d in detections))
    total_detections = len(detections)

    species_counts: Dict[str, int] = {}
    for d in detections:
        species_counts[d.species] = species_counts.get(d.species, 0) + 1
    species_sorted = sorted(species_counts.items(), key=lambda x: -x[1])[:30]

    detections_json = json.dumps([d.to_dict() for d in detections], ensure_ascii=False)
    bar_data = json.dumps(
        [{"species": sp, "count": ct} for sp, ct in species_sorted],
        ensure_ascii=False,
    )

    ts = format_timestamp()

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Way Canguk — {escape(location)} BirdNET Detections Report</title>
<style>
/* ── Design tokens (auto dark/light via system preference) ── */
:root {{
    --bg: #1a1d23;
    --bg-card: #21242b;
    --bg-card-hover: #282d36;
    --bg-input: #2a2d35;
    --bg-thead: #282c35;
    --border: #2d3240;
    --border-light: #2a2e38;
    --text: #d1d5db;
    --text-muted: #6b7280;
    --text-heading: #e2e8f0;
    --text-th: #9ca3af;
    --text-th-hover: #cbd5e1;
    --accent: #3b82f6;
    --header-from: #1e3a5f;
    --header-to: #0f2440;
    --conf-high: #22c55e;
    --conf-med:  #eab308;
    --conf-low:  #ef4444;
    --tooltip-bg: #0f172a;
    --tooltip-border: #334155;
}}
@media (prefers-color-scheme: light) {{
    :root {{
        --bg: #f3f4f6;
        --bg-card: #ffffff;
        --bg-card-hover: #f9fafb;
        --bg-input: #ffffff;
        --bg-thead: #f1f5f9;
        --border: #d1d5db;
        --border-light: #e5e7eb;
        --text: #1f2937;
        --text-muted: #9ca3af;
        --text-heading: #111827;
        --text-th: #6b7280;
        --text-th-hover: #374151;
        --accent: #2563eb;
        --header-from: #dbeafe;
        --header-to: #bfdbfe;
        --conf-high: #16a34a;
        --conf-med:  #ca8a04;
        --conf-low:  #dc2626;
        --tooltip-bg: #1e293b;
        --tooltip-border: #475569;
    }}
}}

*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ font-size: 14px; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: var(--bg);
    color: var(--text);
    line-height: 1.5;
    padding: 1.5rem;
    max-width: 1500px;
    margin: 0 auto;
}}

/* ── Header ────────────────────────────────────────────── */
header {{
    background: linear-gradient(135deg, var(--header-from), var(--header-to));
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
}}
header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-heading);
    letter-spacing: -0.02em;
}}
header .subtitle {{
    font-size: 0.8rem;
    color: var(--text-muted);
    margin-top: 0.35rem;
}}

/* ── Stats Cards ───────────────────────────────────────── */
.stats-row {{
    display: flex; gap: 1rem; margin-bottom: 1.25rem; flex-wrap: wrap;
}}
.stat-card {{
    flex: 1; min-width: 140px;
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.25rem;
}}
.stat-card .label {{
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 0.3rem;
}}
.stat-card .value {{
    font-size: 1.35rem; font-weight: 700; color: var(--text-heading);
}}

/* ── Bar Chart ─────────────────────────────────────────── */
.chart-section {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;
}}
.chart-section h2 {{
    font-size: 0.95rem; font-weight: 600; color: var(--text-th);
    margin-bottom: 1rem;
}}
.chart-container {{ width: 100%; overflow-x: auto; }}

/* ── Controls ──────────────────────────────────────────── */
.controls {{
    display: flex; gap: 0.75rem; margin-bottom: 0.75rem;
    flex-wrap: wrap; align-items: center;
}}
.controls input {{
    flex: 1; min-width: 200px;
    background: var(--bg-input); border: 1px solid var(--border);
    border-radius: 6px; padding: 0.5rem 0.75rem;
    color: var(--text); font-size: 0.85rem; outline: none;
    transition: border-color 0.15s;
}}
.controls input:focus {{ border-color: var(--accent); }}
.controls .info {{ font-size: 0.75rem; color: var(--text-muted); }}

/* ── Table ─────────────────────────────────────────────── */
.table-wrap {{
    overflow-x: auto; border-radius: 8px;
    border: 1px solid var(--border);
}}
table {{
    width: 100%; border-collapse: collapse;
    background: var(--bg-card); font-size: 0.78rem;
}}
thead {{ background: var(--bg-thead); }}
th {{
    padding: 0.55rem 0.7rem; text-align: left;
    font-weight: 600; color: var(--text-th);
    cursor: pointer; user-select: none; white-space: nowrap;
    border-bottom: 2px solid var(--border);
    font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.04em;
}}
th:hover {{ color: var(--text-th-hover); background: var(--bg-card-hover); }}
th .arrow {{ margin-left: 3px; font-size: 0.6rem; }}
td {{
    padding: 0.35rem 0.7rem;
    border-bottom: 1px solid var(--border-light);
    white-space: nowrap; color: var(--text);
}}
tbody tr {{ cursor: default; transition: background 0.1s; }}
tbody tr:hover {{ background: var(--bg-card-hover); }}
tbody tr:last-child td {{ border-bottom: none; }}

.conf-high {{ color: var(--conf-high); font-weight: 600; }}
.conf-med  {{ color: var(--conf-med);  font-weight: 600; }}
.conf-low  {{ color: var(--conf-low);  font-weight: 600; }}

/* ── Tooltip on hover ──────────────────────────────────── */
[data-tooltip] {{ position: relative; }}
[data-tooltip]::after {{
    content: attr(data-tooltip);
    position: absolute; bottom: calc(100% + 8px); left: 50%;
    transform: translateX(-50%);
    background: var(--tooltip-bg); color: #e2e8f0;
    border: 1px solid var(--tooltip-border);
    border-radius: 6px; padding: 0.5rem 0.75rem;
    font-size: 0.72rem; white-space: nowrap; z-index: 100;
    pointer-events: none; opacity: 0;
    transition: opacity 0.15s;
}}
[data-tooltip]:hover::after {{ opacity: 1; }}

/* ── Viz suggestions ───────────────────────────────────── */
.suggestions {{
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.25rem 1.5rem; margin-top: 1.5rem;
}}
.suggestions h2 {{
    font-size: 0.95rem; font-weight: 600; color: var(--text-th); margin-bottom: 0.75rem;
}}
.suggestions ul {{
    list-style: disc; padding-left: 1.5rem;
    color: var(--text); font-size: 0.82rem; line-height: 1.7;
}}
.suggestions li strong {{
    color: var(--accent);
}}

/* ── Footer ────────────────────────────────────────────── */
footer {{
    margin-top: 1.5rem; font-size: 0.7rem;
    color: var(--text-muted); text-align: center;
}}

/* ── Responsive ────────────────────────────────────────── */
@media (max-width: 768px) {{
    body {{ padding: 0.75rem; }}
    header {{ padding: 1rem 1.25rem; }}
    .stats-row {{ gap: 0.5rem; }}
    .stat-card {{ min-width: 100px; padding: 0.75rem 1rem; }}
    th, td {{ padding: 0.3rem 0.4rem; font-size: 0.68rem; }}
}}
</style>
</head>
<body>

<header>
    <h1>Way Canguk &mdash; {escape(location)} Daily BirdNET Detections Report</h1>
    <div class="subtitle">Last updated: {escape(ts)} &nbsp;|&nbsp; Data source: {escape(data_dir)}</div>
</header>

<div class="stats-row">
    <div class="stat-card">
        <div class="label">Total Detections</div>
        <div class="value">{total_detections}</div>
    </div>
    <div class="stat-card">
        <div class="label">Unique Species</div>
        <div class="value">{len(all_species)}</div>
    </div>
    <div class="stat-card">
        <div class="label">Dates Covered</div>
        <div class="value">{len(all_dates)}</div>
    </div>
    <div class="stat-card">
        <div class="label">Date Range</div>
        <div class="value" style="font-size:1rem;">{escape(all_dates[0] if all_dates else '—')} &rarr; {escape(all_dates[-1] if all_dates else '—')}</div>
    </div>
    <div class="stat-card">
        <div class="label">Processing Methods</div>
        <div class="value" style="font-size:0.9rem;">LabIR &middot; SPIR1 &middot; SPIR2 &middot; SA</div>
    </div>
</div>

<div class="chart-section">
    <h2>Top Species by Detection Count</h2>
    <div class="chart-container">
        <svg id="barChart" width="100%" height="350"></svg>
    </div>
</div>

<div class="controls">
    <input type="text" id="filterInput" placeholder="Filter detections… (searches all columns)" autocomplete="off">
    <span class="info" id="rowCount"></span>
</div>

<div class="table-wrap">
    <table id="detectionsTable">
        <thead>
            <tr>
                <th data-col="date">Date <span class="arrow"></span></th>
                <th data-col="time_str">Time <span class="arrow"></span></th>
                <th data-col="species">Species <span class="arrow"></span></th>
                <th data-col="confidence">Conf <span class="arrow"></span></th>
                <th data-col="method">Method <span class="arrow"></span></th>
                <th data-col="distance">Dist <span class="arrow"></span></th>
                <th data-col="azimuth">Az <span class="arrow"></span></th>
                <th data-col="elevation">Elev <span class="arrow"></span></th>
            </tr>
        </thead>
        <tbody id="tableBody"></tbody>
    </table>
</div>

<div class="suggestions">
    <h2>Visualization Ideas for Supervisor Discussion</h2>
    <ul>
        <li><strong>Polar Azimuth-Elevation Scatter</strong> — plot each detection on polar coordinates (radius = elevation, angle = azimuth). Colored by species. Visually reveals WHERE in 3D space the birds are vocalising.</li>
        <li><strong>Time-of-Day Activity Heatmap</strong> — X = hour of day (0&ndash;24), Y = species, color intensity = detection count. Reveals diurnal/nocturnal patterns per species.</li>
        <li><strong>Confidence Box-Plot Comparison</strong> — side-by-side box plots comparing mono (SA) vs beamformed confidence scores per species. Quantifies the SNR gain from beamforming.</li>
        <li><strong>Directional Rose Diagram</strong> — circular histogram where each petal = detection count in that azimuth bin. Shows whether the soundscape has directional bias.</li>
        <li><strong>Species Accumulation Curve</strong> — X = cumulative dates, Y = unique species discovered. Shows whether we are reaching saturation or still discovering new species at this site.</li>
        <li><strong>Hourly Confidence Timeline</strong> — scatter plot of confidence vs time-of-day, faceted by species. Identifies optimal recording windows for target species.</li>
        <li><strong>Method Comparison Radar Chart</strong> — per-species radar chart comparing LabIR vs SPIR1 vs SPIR2 vs SA across metrics (max conf, detection count, mean conf). Highlights which processing method works best for which species.</li>
    </ul>
</div>

<footer>
    Generated {escape(ts)} &nbsp;&middot;&nbsp; Spatial Ecoacoustic Analysis Pipeline
</footer>

<script>
// ── Data ──────────────────────────────────────────────────
var DETECTIONS = {detections_json};
var BAR_DATA = {bar_data};

// ── Render Table ──────────────────────────────────────────
var sortCol = 'time_str';
var sortDir = 1;
var filterText = '';

function confidenceClass(conf) {{
    if (conf >= 0.8) return 'conf-high';
    if (conf >= 0.5) return 'conf-med';
    return 'conf-low';
}}

function buildTooltip(r) {{
    // Full detail shown on row hover
    return 'Date: ' + r.date + '\\n' +
           'Time: ' + r.time_str + ' (' + r.start_time.toFixed(1) + 's)\\n' +
           'Species: ' + r.species + '\\n' +
           'Confidence: ' + r.confidence.toFixed(3) + '\\n' +
           'Method: ' + r.method + '\\n' +
           'Direction: ' + (r.direction_label || 'N/A') + '\\n' +
           'Distance: ' + (r.distance || '—') + '\\n' +
           'Azimuth: ' + (r.azimuth || '—') + 'deg\\n' +
           'Elevation: ' + (r.elevation || '—') + 'deg\\n' +
           'File: ' + (r.raw_filename || '—');
}}

function renderTable() {{
    var rows = DETECTIONS.slice();

    if (filterText) {{
        var q = filterText.toLowerCase();
        rows = rows.filter(function(d) {{
            return (d.date + ' ' + d.time_str + ' ' + d.species + ' ' +
                    d.confidence + ' ' + d.method + ' ' + d.azimuth +
                    ' ' + d.elevation + ' ' + d.distance)
                    .toLowerCase().indexOf(q) !== -1;
        }});
    }}

    rows.sort(function(a, b) {{
        var va = a[sortCol], vb = b[sortCol];
        if (sortCol === 'confidence' || sortCol === 'start_time') {{
            return (va - vb) * sortDir;
        }}
        if (sortCol === 'azimuth' || sortCol === 'elevation') {{
            var na = va === '' ? -999 : parseInt(va);
            var nb = vb === '' ? -999 : parseInt(vb);
            return (na - nb) * sortDir;
        }}
        return String(va).localeCompare(String(vb)) * sortDir;
    }});

    var tbody = document.getElementById('tableBody');
    var html = '';
    for (var i = 0; i < rows.length; i++) {{
        var r = rows[i];
        var tip = buildTooltip(r).replace(/"/g, '&quot;').replace(/\\n/g, '&#10;');
        html += '<tr data-tooltip="' + tip + '">' +
            '<td>' + esc(r.date) + '</td>' +
            '<td>' + esc(r.time_str) + '</td>' +
            '<td>' + esc(r.species) + '</td>' +
            '<td class="' + confidenceClass(r.confidence) + '">' + r.confidence.toFixed(3) + '</td>' +
            '<td>' + esc(r.method) + '</td>' +
            '<td>' + (r.distance || '—') + '</td>' +
            '<td>' + (r.azimuth || '—') + '</td>' +
            '<td>' + (r.elevation || '—') + '</td>' +
            '</tr>';
    }}
    tbody.innerHTML = html;
    document.getElementById('rowCount').textContent =
        'Showing ' + rows.length + ' of ' + DETECTIONS.length + ' detections';
}}

function esc(s) {{
    return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}}

// ── Sorting ───────────────────────────────────────────────
function updateArrows() {{
    document.querySelectorAll('th').forEach(function(th) {{
        var arrow = th.querySelector('.arrow');
        arrow.textContent = '';
        if (th.dataset.col === sortCol) {{
            arrow.textContent = sortDir === 1 ? ' ▲' : ' ▼';
        }}
    }});
}}

document.querySelectorAll('th').forEach(function(th) {{
    th.addEventListener('click', function() {{
        var col = th.dataset.col;
        sortCol = col;
        sortDir = (sortCol === col) ? -sortDir : 1;
        // sortDir flip handled by toggle
        if (th.dataset.col === sortCol) {{
            sortDir = -sortDir;
        }} else {{
            sortDir = 1;
        }}
        updateArrows();
        renderTable();
    }});
}});

// ── Filter ────────────────────────────────────────────────
document.getElementById('filterInput').addEventListener('input', function(e) {{
    filterText = e.target.value;
    renderTable();
}});

// ── Bar Chart ─────────────────────────────────────────────
function renderBarChart() {{
    var svg = document.getElementById('barChart');
    var data = BAR_DATA;
    if (!data.length) return;

    var width = svg.parentElement.clientWidth - 32;
    if (width < 400) width = 400;
    svg.setAttribute('viewBox', '0 0 ' + width + ' 350');
    svg.style.width = '100%';

    var margin = {{ top: 10, right: 20, bottom: 120, left: 220 }};
    var chartW = width - margin.left - margin.right;
    var chartH = 350 - margin.top - margin.bottom;
    var maxCount = data[0].count;
    var barH = Math.max(12, Math.min(28, (chartH - 4) / data.length));
    var isDark = !window.matchMedia('(prefers-color-scheme: light)').matches;
    var labelColor = isDark ? '#9ca3af' : '#6b7280';

    var html = '';
    for (var i = 0; i < data.length; i++) {{
        var d = data[i];
        var barW = (d.count / maxCount) * chartW;
        var y = margin.top + i * barH;
        var hue = 210 - (i / data.length) * 160;
        var color = 'hsl(' + hue + ', 55%, 55%)';

        html += '<g>' +
            '<text x="' + (margin.left - 8) + '" y="' + (y + barH * 0.65) + '" ' +
                  'text-anchor="end" font-size="11" fill="' + labelColor + '">' +
                  esc(d.species.substring(0, 30)) + '</text>' +
            '<rect x="' + margin.left + '" y="' + y + '" ' +
                  'width="' + Math.max(2, barW) + '" height="' + (barH - 3) + '" ' +
                  'rx="3" fill="' + color + '" opacity="0.85"/>' +
            '<text x="' + (margin.left + Math.max(2, barW) + 6) + '" ' +
                  'y="' + (y + barH * 0.65) + '" ' +
                  'font-size="10" fill="' + labelColor + '">' + d.count + '</text>' +
            '</g>';
    }}
    svg.innerHTML = html;
}}

// ── Init ──────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', function() {{
    updateArrows();
    renderTable();
    renderBarChart();
}});
window.addEventListener('resize', function() {{
    renderBarChart();
}});
</script>

</body>
</html>"""

    return html


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Generate an HTML daily report from processed.json files."
    )
    parser.add_argument(
        "--location", type=str, required=True,
        help="Location code (e.g. 2A400, Q0, S0)",
    )
    parser.add_argument(
        "--data-dir", type=str, default=ANALYSIS_OUTPUT,
        help=f"Base data directory (default: {ANALYSIS_OUTPUT})",
    )
    parser.add_argument(
        "--dates", type=str, default=None,
        help="Comma-separated dates to include (default: all)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output HTML file path (default: daily_report.html in data dir location folder)",
    )
    parser.add_argument(
        "--open", action="store_true",
        help="Open the report in the default browser after generation",
    )

    args = parser.parse_args()

    # ── Parse date filter ─────────────────────────────────────
    date_filter: Optional[List[str]] = None
    if args.dates:
        date_filter = [d.strip() for d in args.dates.split(",")]

    # ── Find processed files ──────────────────────────────────
    print(f"Scanning {args.data_dir}/{args.location}/ ...")
    processed_files = find_processed_json_files(
        args.data_dir, args.location, date_filter
    )
    print(f"Found {len(processed_files)} processed.json file(s)")

    # ── Collect detections ────────────────────────────────────
    detections = collect_detections(processed_files)
    print(f"Collected {len(detections)} detection(s) across {len(set(d.species for d in detections))} species")

    if not detections:
        print("No detections found — generating empty report.")

    # ── Generate HTML ─────────────────────────────────────────
    html = generate_html(args.location, detections, args.data_dir)

    default_path = os.path.join(args.data_dir, args.location, "daily_report.html")
    output_path = args.output or default_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    print(f"Report written to: {abs_path}")
    print(f"   ({len(html):,} bytes)")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
