#!/usr/bin/env python3
"""
Generate a self-contained, beautiful HTML daily report from processed.json files.

Usage:
    python generate_report.py --location 2A400
    python generate_report.py --location 2A400 --dates 2025-06-01,2025-06-02

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

# ── Try to import config; fall back to hardcoded default ────────────
try:
    from config import ANALYSIS_OUTPUT
except ImportError:
    ANALYSIS_OUTPUT = "/Volumes/HD Data/sea-data"

# ============================================================
# LABIR SPEAKER → ELEVATION MAPPING
# ============================================================
LABIR_ELEVATION: Dict[int, int] = {
    1: -45, 2: -30, 3: -20, 4: -10, 5: 0,
    6: 10, 7: 20, 8: 30, 9: 45, 10: 60,
    11: 75, 12: 90,
}

# Processing subdirectories to search for processed.json
PROCESSING_DIRS = [
    "beamforming_LabIR",
    "beamforming_SPIR1",
    "beamforming_SPIR2",
    "signal_averaging",
]

# Regex patterns for extracting direction info from filenames
RE_LABIR = re.compile(r"_LabIR\(S(\d+)_(\d+)\)\.wav$")
RE_SPIR1 = re.compile(r"_SPIR1\((\d+)m_(\d+)\)\.wav$")
RE_SPIR2 = re.compile(r"_SPIR2\((\d+)m_180_r(\d+)\)\.wav$")
RE_SA    = re.compile(r"_sa\.wav$")

# ============================================================
# DETECTION DATA STRUCTURES
# ============================================================

class Detection:
    """A single BirdNET detection row."""
    __slots__ = (
        "date", "method", "species", "confidence",
        "start_time", "direction_label", "raw_filename",
    )

    def __init__(
        self,
        date: str,
        method: str,
        species: str,
        confidence: float,
        start_time: float,
        direction_label: str,
        raw_filename: str,
    ):
        self.date = date
        self.method = method
        self.species = species
        self.confidence = confidence
        self.start_time = start_time
        self.direction_label = direction_label
        self.raw_filename = raw_filename

    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date,
            "method": self.method,
            "species": self.species,
            "confidence": self.confidence,
            "start_time": self.start_time,
            "direction_label": self.direction_label,
            "raw_filename": self.raw_filename,
        }


# ============================================================
# DIRECTION PARSING
# ============================================================

def parse_direction(filename: str, method: str) -> str:
    """
    Parse a human-readable direction label from a primary_channel filename.

    LabIR:  "S01 elev:-45° az:060°"
    SPIR1:  "2m az:300°"
    SPIR2:  "16m az:180° rep:3"
    SA:     "omnidirectional"
    """
    if method == "LabIR":
        m = RE_LABIR.search(filename)
        if m:
            speaker = int(m.group(1))
            azimuth = int(m.group(2))
            elev = LABIR_ELEVATION.get(speaker, "?")
            return f"S{speaker:02d} elev:{elev:+d}° az:{azimuth:03d}°"
        return filename

    elif method == "SPIR1":
        m = RE_SPIR1.search(filename)
        if m:
            distance = int(m.group(1))
            azimuth = int(m.group(2))
            return f"{distance}m az:{azimuth:03d}°"
        return filename

    elif method == "SPIR2":
        m = RE_SPIR2.search(filename)
        if m:
            distance = int(m.group(1))
            rep = int(m.group(2))
            return f"{distance}m az:180° rep:{rep}"
        return filename

    elif method == "SA":
        return "omnidirectional"

    return filename


# ============================================================
# DATA COLLECTION
# ============================================================

def find_processed_json_files(
    base_dir: str,
    location: str,
    date_filter: Optional[List[str]] = None,
) -> List[Tuple[str, str, str]]:
    """
    Walk the directory tree and find all processed.json files.

    Returns list of (date_str, method, full_path) tuples.
    """
    location_dir = os.path.join(base_dir, location)
    if not os.path.isdir(location_dir):
        print(f"❌ Location directory not found: {location_dir}")
        return []

    found: List[Tuple[str, str, str]] = []

    for entry in sorted(os.listdir(location_dir)):
        date_path = os.path.join(location_dir, entry)
        if not os.path.isdir(date_path):
            continue

        # Optional date filter
        if date_filter and entry not in date_filter:
            continue

        for subdir in PROCESSING_DIRS:
            proc_dir = os.path.join(date_path, subdir)
            proc_json = os.path.join(proc_dir, "processed.json")
            if os.path.isfile(proc_json):
                # Map subdir name to method label
                if subdir == "beamforming_LabIR":
                    method = "LabIR"
                elif subdir == "beamforming_SPIR1":
                    method = "SPIR1"
                elif subdir == "beamforming_SPIR2":
                    method = "SPIR2"
                elif subdir == "signal_averaging":
                    method = "SA"
                else:
                    method = subdir

                found.append((entry, method, proc_json))

    return found


def collect_detections(processed_files: List[Tuple[str, str, str]]) -> List[Detection]:
    """
    Read all processed.json files and flatten into a list of Detection objects.
    Each start_time in each species becomes a separate row.
    """
    detections: List[Detection] = []

    for date_str, method, json_path in processed_files:
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data: Dict[str, Any] = json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            print(f"⚠  Skipping {json_path}: {e}", file=sys.stderr)
            continue

        for species_name, sp_data in data.items():
            start_times = sp_data.get("start_time_list", [])
            confidences = sp_data.get("conf_list", [])
            channels = sp_data.get("primary_channel_list", [])

            n = len(start_times)
            for i in range(n):
                conf = confidences[i] if i < len(confidences) else 0.0
                channel = channels[i] if i < len(channels) else ""
                start_t = start_times[i]

                direction = parse_direction(channel, method)

                detections.append(Detection(
                    date=date_str,
                    method=method,
                    species=species_name,
                    confidence=round(float(conf), 3),
                    start_time=round(float(start_t), 1),
                    direction_label=direction,
                    raw_filename=channel,
                ))

    return detections


# ============================================================
# HTML GENERATION
# ============================================================

def format_timestamp() -> str:
    """Return ISO 8601 UTC timestamp for the report."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def escape_str(s: Any) -> str:
    """JSON-escape a string for safe embedding in JavaScript."""
    return json.dumps(str(s))


def generate_html(
    location: str,
    detections: List[Detection],
    data_dir: str,
) -> str:
    """Generate a self-contained HTML report."""

    # ── Aggregate stats ─────────────────────────────────────────
    all_species = sorted(set(d.species for d in detections))
    all_dates = sorted(set(d.date for d in detections))
    total_detections = len(detections)

    # Species counts for bar chart
    species_counts: Dict[str, int] = {}
    for d in detections:
        species_counts[d.species] = species_counts.get(d.species, 0) + 1
    # Sort by count descending, take top 30
    species_sorted = sorted(species_counts.items(), key=lambda x: -x[1])[:30]

    # ── Detection data as JSON for the table ─────────────────────
    detections_json = json.dumps(
        [d.to_dict() for d in detections],
        ensure_ascii=False,
    )

    # ── Bar chart data as JSON ───────────────────────────────────
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
/* ── Reset & Base ──────────────────────────────────────── */
*, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
html {{ font-size: 14px; }}
body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Helvetica, Arial, sans-serif;
    background: #1a1d23;
    color: #d1d5db;
    line-height: 1.5;
    padding: 1.5rem;
    max-width: 1400px;
    margin: 0 auto;
}}

/* ── Header ────────────────────────────────────────────── */
header {{
    background: linear-gradient(135deg, #1e3a5f 0%, #162d50 50%, #0f2440 100%);
    border: 1px solid #2d3a50;
    border-radius: 10px;
    padding: 1.5rem 2rem;
    margin-bottom: 1.25rem;
}}
header h1 {{
    font-size: 1.5rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -0.02em;
}}
header .subtitle {{
    font-size: 0.8rem;
    color: #7f8ea3;
    margin-top: 0.35rem;
}}

/* ── Stats Cards ───────────────────────────────────────── */
.stats-row {{
    display: flex;
    gap: 1rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
}}
.stat-card {{
    flex: 1;
    min-width: 160px;
    background: #21242b;
    border: 1px solid #2d3240;
    border-radius: 8px;
    padding: 1rem 1.25rem;
}}
.stat-card .label {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #6b7280;
    margin-bottom: 0.3rem;
}}
.stat-card .value {{
    font-size: 1.35rem;
    font-weight: 700;
    color: #e2e8f0;
}}

/* ── Bar Chart Section ──────────────────────────────────── */
.chart-section {{
    background: #21242b;
    border: 1px solid #2d3240;
    border-radius: 8px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
}}
.chart-section h2 {{
    font-size: 0.95rem;
    font-weight: 600;
    color: #9ca3af;
    margin-bottom: 1rem;
}}
.chart-container {{
    width: 100%;
    overflow-x: auto;
}}

/* ── Filter & Controls ──────────────────────────────────── */
.controls {{
    display: flex;
    gap: 0.75rem;
    margin-bottom: 0.75rem;
    flex-wrap: wrap;
    align-items: center;
}}
.controls input {{
    flex: 1;
    min-width: 220px;
    background: #2a2d35;
    border: 1px solid #3a3f4a;
    border-radius: 6px;
    padding: 0.5rem 0.75rem;
    color: #d1d5db;
    font-size: 0.85rem;
    outline: none;
    transition: border-color 0.15s;
}}
.controls input:focus {{ border-color: #4a80b5; }}
.controls .info {{
    font-size: 0.75rem;
    color: #6b7280;
}}

/* ── Table ──────────────────────────────────────────────── */
.table-wrap {{
    overflow-x: auto;
    border-radius: 8px;
    border: 1px solid #2d3240;
}}
table {{
    width: 100%;
    border-collapse: collapse;
    background: #21242b;
    font-size: 0.8rem;
}}
thead {{ background: #282c35; }}
th {{
    padding: 0.6rem 0.85rem;
    text-align: left;
    font-weight: 600;
    color: #9ca3af;
    cursor: pointer;
    user-select: none;
    white-space: nowrap;
    border-bottom: 2px solid #3a3f4a;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}
th:hover {{ color: #cbd5e1; background: #2e333d; }}
th .arrow {{ margin-left: 4px; font-size: 0.65rem; }}
td {{
    padding: 0.4rem 0.85rem;
    border-bottom: 1px solid #2a2e38;
    white-space: nowrap;
    color: #c4c9d1;
}}
tbody tr:hover {{ background: #282d36; }}
tbody tr:last-child td {{ border-bottom: none; }}

.conf-high {{ color: #4ade80; }}
.conf-med  {{ color: #facc15; }}
.conf-low  {{ color: #f87171; }}

/* ── Footer ─────────────────────────────────────────────── */
footer {{
    margin-top: 1.5rem;
    font-size: 0.7rem;
    color: #4b5563;
    text-align: center;
}}

/* ── Responsive ─────────────────────────────────────────── */
@media (max-width: 768px) {{
    body {{ padding: 0.75rem; }}
    header {{ padding: 1rem 1.25rem; }}
    .stats-row {{ gap: 0.5rem; }}
    .stat-card {{ min-width: 120px; padding: 0.75rem 1rem; }}
}}
</style>
</head>
<body>

<header>
    <h1>Way Canguk — {escape(location)} Daily BirdNET Detections Report</h1>
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
        <div class="value" style="font-size:1rem;">{escape(all_dates[0] if all_dates else '—')} → {escape(all_dates[-1] if all_dates else '—')}</div>
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
                <th data-col="method">Method <span class="arrow"></span></th>
                <th data-col="species">Species <span class="arrow"></span></th>
                <th data-col="confidence">Confidence <span class="arrow"></span></th>
                <th data-col="start_time">Start Time (s) <span class="arrow"></span></th>
                <th data-col="direction_label">Direction <span class="arrow"></span></th>
            </tr>
        </thead>
        <tbody id="tableBody"></tbody>
    </table>
</div>

<footer>
    Generated {escape(ts)} &nbsp;·&nbsp; Spatial Ecoacoustic Analysis Pipeline
</footer>

<script>
// ── Data ──────────────────────────────────────────────────
var DETECTIONS = {detections_json};
var BAR_DATA = {bar_data};

// ── Render Table ──────────────────────────────────────────
var sortCol = 'start_time';
var sortDir = 1;  // 1 = asc, -1 = desc
var filterText = '';

function confidenceClass(conf) {{
    if (conf >= 0.8) return 'conf-high';
    if (conf >= 0.5) return 'conf-med';
    return 'conf-low';
}}

function renderTable() {{
    var rows = DETECTIONS.slice();

    // Filter
    if (filterText) {{
        var q = filterText.toLowerCase();
        rows = rows.filter(function(d) {{
            return (d.date + ' ' + d.method + ' ' + d.species + ' ' +
                    d.confidence + ' ' + d.start_time + ' ' + d.direction_label)
                    .toLowerCase().indexOf(q) !== -1;
        }});
    }}

    // Sort
    rows.sort(function(a, b) {{
        var va = a[sortCol], vb = b[sortCol];
        if (typeof va === 'number') return (va - vb) * sortDir;
        return String(va).localeCompare(String(vb)) * sortDir;
    }});

    var tbody = document.getElementById('tableBody');
    var html = '';
    for (var i = 0; i < rows.length; i++) {{
        var r = rows[i];
        html += '<tr>' +
            '<td>' + esc(r.date) + '</td>' +
            '<td>' + esc(r.method) + '</td>' +
            '<td>' + esc(r.species) + '</td>' +
            '<td class="' + confidenceClass(r.confidence) + '">' + r.confidence.toFixed(3) + '</td>' +
            '<td>' + r.start_time.toFixed(1) + '</td>' +
            '<td>' + esc(r.direction_label) + '</td>' +
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
    var headers = document.querySelectorAll('th');
    headers.forEach(function(th) {{
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
        if (sortCol === col) {{
            sortDir = -sortDir;
        }} else {{
            sortCol = col;
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

// ── Bar Chart (inline SVG) ────────────────────────────────
function renderBarChart() {{
    var svg = document.getElementById('barChart');
    var data = BAR_DATA;
    if (!data.length) return;

    var width = svg.parentElement.clientWidth - 32;
    if (width < 400) width = 400;
    svg.setAttribute('viewBox', '0 0 ' + width + ' 350');
    svg.style.width = '100%';
    svg.style.maxWidth = width + 'px';

    var margin = {{ top: 10, right: 20, bottom: 120, left: 220 }};
    var chartW = width - margin.left - margin.right;
    var chartH = 350 - margin.top - margin.bottom;

    var maxCount = data[0].count;
    var barH = Math.max(12, Math.min(28, (chartH - 4) / data.length));

    var html = '';
    for (var i = 0; i < data.length; i++) {{
        var d = data[i];
        var barW = (d.count / maxCount) * chartW;
        var y = margin.top + i * barH;
        var pct = Math.round((d.count / maxCount) * 100);

        // Bar color gradient based on position (cool blues → warm)
        var hue = 210 - (i / data.length) * 160;  // 210° blue → 50° orange
        var color = 'hsl(' + hue + ', 55%, 55%)';

        html += '<g>' +
            '<text x="' + (margin.left - 8) + '" y="' + (y + barH * 0.65) + '" ' +
                  'text-anchor="end" font-size="11" fill="#9ca3af">' +
                  esc(d.species.substring(0, 30)) + '</text>' +
            '<rect x="' + margin.left + '" y="' + y + '" ' +
                  'width="' + Math.max(2, barW) + '" height="' + (barH - 3) + '" ' +
                  'rx="3" fill="' + color + '" opacity="0.85"/>' +
            '<text x="' + (margin.left + Math.max(2, barW) + 6) + '" ' +
                  'y="' + (y + barH * 0.65) + '" ' +
                  'font-size="10" fill="#6b7280">' +
                  d.count + '</text>' +
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
        help="Location code (e.g. '2A400', 'Q0')",
    )
    parser.add_argument(
        "--data-dir", type=str, default=ANALYSIS_OUTPUT,
        help=f"Base data directory (default: {ANALYSIS_OUTPUT})",
    )
    parser.add_argument(
        "--dates", type=str, default=None,
        help="Comma-separated date filter, e.g. '2025-06-01,2025-06-02'",
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

    # ── Validate ────────────────────────────────────────────────
    if not os.path.isdir(args.data_dir):
        print(f"❌ Data directory not found: {args.data_dir}")
        print("   Is the external HDD mounted?")
        sys.exit(1)

    # Parse date filter
    date_filter = None
    if args.dates:
        date_filter = [d.strip() for d in args.dates.split(",") if d.strip()]

    # ── Collect data ────────────────────────────────────────────
    print(f"🔍 Scanning {args.data_dir}/{args.location}/ ...")
    processed_files = find_processed_json_files(
        args.data_dir, args.location, date_filter
    )

    if not processed_files:
        print("❌ No processed.json files found.")
        sys.exit(1)

    print(f"📄 Found {len(processed_files)} processed.json file(s)")

    detections = collect_detections(processed_files)
    print(f"🐦 Collected {len(detections)} detection(s) across {len(set(d.species for d in detections))} species")

    if not detections:
        print("⚠  No detections found — generating empty report.")

    # ── Generate HTML ───────────────────────────────────────────
    html = generate_html(args.location, detections, args.data_dir)

    default_path = os.path.join(args.data_dir, args.location, "daily_report.html")
    output_path = args.output or default_path
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    abs_path = os.path.abspath(output_path)
    print(f"✅ Report written to: {abs_path}")
    print(f"   ({len(html):,} bytes)")

    # ── Open in browser ─────────────────────────────────────────
    if args.open:
        import webbrowser
        webbrowser.open(f"file://{abs_path}")


if __name__ == "__main__":
    main()
