#!/usr/bin/env python3
"""
Multi-model offline visualizer for bacpipe embeddings.

Performs matched-window pairwise spatial direction evaluation and high-dimensional
HDBSCAN clustering, generating a portable standalone HTML report suite for all models.
"""

from __future__ import annotations

import os

# Prevent GPFS NFS lock stalls on HPC clusters
os.environ["NUMBA_CACHE_DIR"] = f"/tmp/numba_{os.environ.get('USER', 'sea')}"
os.environ["NUMBA_DISABLE_JIT_CACHE"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import argparse
import html
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

import narrative as narrative_defaults
from narrative import build_digest, load_or_generate_narrative
from spatial_clustering import (
    align_and_select_matched_windows,
    compute_all_beam_noise_analysis,
    compute_noise_distance,
    compute_shared_cluster_analysis,
    compute_stats,
    convert_numpy_types,
    load_noise_embeddings,
    run_hdbscan,
    run_umap,
)
from config import ANALYSIS_OUTPUT
from embedding_io import load_embeddings_from_dir, list_embedding_files
from embedding_schema import (
    DEFAULT_METHODS,
    bacpipe_embeddings_dir,
    bacpipe_meta_dir,
    dashboards_dir,
    embeddings_root,
)


DEFAULT_METHODS_LIST = list(DEFAULT_METHODS)


def _json_for_html(value: Any) -> str:
    """Serialise JSON safely for embedding inside a script tag."""
    clean_val = convert_numpy_types(value)
    return (
        json.dumps(clean_val, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _date_candidates(model_dir: str) -> List[str]:
    dates = set()
    for _path, date_str, _method in list_embedding_files(model_dir):
        dates.add(date_str)
    return sorted(dates)


def _resolve_date(model_dirs: Sequence[str], requested: Optional[str]) -> str:
    if requested:
        return requested
    dates = set()
    for model_dir in model_dirs:
        dates.update(_date_candidates(model_dir))
    if len(dates) == 1:
        return next(iter(dates))
    if not dates:
        raise RuntimeError("No dated embedding files were found")
    raise RuntimeError(
        "Multiple dates found; specify --date explicitly: " + ", ".join(sorted(dates))
    )


def discover_models(data_dir: str, location: str, date_str: Optional[str]) -> List[str]:
    root = Path(embeddings_root(data_dir, location))
    if not root.is_dir():
        return []
    models = []
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        files = list_embedding_files(str(child), date_filter=[date_str] if date_str else None)
        if files:
            models.append(child.name)
    return models


def _compact_metadata(metadata: Sequence[dict], y_method: np.ndarray,
                      method_names: Sequence[str], clusters: np.ndarray,
                      model: str) -> List[List[Any]]:
    """Return small, stable hover rows instead of embedding absolute HPC paths."""
    rows: List[List[Any]] = []
    for idx, rec in enumerate(metadata):
        rec = rec or {}
        method_idx = int(y_method[idx]) if idx < len(y_method) else -1
        method = method_names[method_idx] if 0 <= method_idx < len(method_names) else rec.get("method", "")
        cluster = int(clusters[idx]) if idx < len(clusters) else -1
        rows.append([
            os.path.basename(str(rec.get("wav", ""))),
            method,
            rec.get("start_sec"),
            rec.get("end_sec"),
            rec.get("azimuth"),
            rec.get("elevation"),
            cluster,
            rec.get("model", model),
        ])
    return rows


def _model_html(
    *,
    model: str,
    source_dir: str,
    date_str: str,
    umap_2d: np.ndarray,
    y_method: np.ndarray,
    method_names: Sequence[str],
    cluster_labels: np.ndarray,
    point_meta: Sequence[Sequence[Any]],
    stats: Dict[str, Any],
    parameters: Dict[str, Any],
    narrative: Optional[Dict[str, Any]] = None,
) -> str:
    """Build one HTML report that references the shared Plotly asset."""
    title = f"Spatial Bioacoustic Embeddings — {model} — {date_str}"
    data = {
        "model": model,
        "date": date_str,
        "source_dir": source_dir,
        "umap": umap_2d.tolist(),
        "methods": [int(x) for x in y_method],
        "method_names": list(method_names),
        "clusters": [int(x) for x in cluster_labels],
        "point_meta": list(point_meta),
        "stats": stats,
        "parameters": parameters,
        "narrative": narrative,
    }

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="plotly.min.js"></script>
<style>
:root { color-scheme: light; --bg:#f8fafc; --card:#ffffff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --primary:#2563eb; --success:#16a34a; --warning:#ca8a04; }
* { box-sizing:border-box; }
body { margin:0; padding:28px; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }
main { max-width:1450px; margin:auto; }
header { margin-bottom:24px; }
h1 { margin:0 0 6px; font-size:26px; font-weight:700; color:#1e293b; }
h2 { margin:32px 0 12px; font-size:18px; font-weight:600; color:#334155; display:flex; align-items:center; gap:8px; }
h3 { margin:0 0 12px; font-size:15px; }
.badge { display:inline-block; font-size:12px; font-weight:600; padding:2px 8px; border-radius:12px; background:#e0e7ff; color:#3730a3; }
.badge-success { background:#dcfce7; color:#166534; }
.badge-warning { background:#fef3c7; color:#92400e; }
.subtitle { color:var(--muted); font-size:13px; margin-bottom:12px; word-break:break-all; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:14px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:18px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.stat-value { font-size:24px; font-weight:700; color:#0f172a; }
.stat-label { color:var(--muted); font-size:12px; margin-top:4px; font-weight:500; text-transform:uppercase; letter-spacing:0.5px; }
.chart { height:520px; }
.small-chart { height:380px; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th,td { padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }
th { color:var(--muted); font-weight:600; background:#f8fafc; font-size:12px; text-transform:uppercase; letter-spacing:0.5px; }
tr:hover { background:#f8fafc; }
code { font-size:12px; background:#f1f5f9; padding:2px 6px; border-radius:4px; font-family:ui-monospace,SFMono-Regular,Menlo,monospace; }
.p-val { font-weight:600; color:#16a34a; }
.note-box { background:#f1f5f9; border-left:4px solid var(--primary); padding:12px 16px; border-radius:6px; margin-bottom:16px; font-size:13px; line-height:1.45; }
#narrative h3 { font-size:15px; margin:0 0 10px; }
#narrative ul { margin:0; padding-left:20px; }
#narrative li { margin-bottom:8px; }
.narrative-foot { color:var(--muted); font-size:12px; margin:14px 0 0; }
.sig-star { color:#e11d48; font-weight:bold; }
@media(max-width:960px) { body { padding:16px; } .two { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
<header>
<h1>__TITLE__</h1>
<div class="subtitle">Source: <code>__SOURCE__</code></div>
</header>

<section class="grid" id="overview"></section>

<section id="narrative-section">
<h2>Interpretation <span class="badge" id="narrative-badge"></span></h2>
<div class="card" id="narrative"></div>
</section>

<h2>Matched Pairwise Performance vs Single-Channel Mono</h2>
<div class="two">
  <div class="card">
    <h3>Separation Gain (&Delta; vs Mono)</h3>
    <div id="plot-gain" class="small-chart"></div>
  </div>
  <div class="card">
    <h3>Hypothesis Testing & Effect Size</h3>
    <div id="pairwise-table"></div>
  </div>
</div>

<h2>High-Dimensional HDBSCAN Clusters & Spatial Steered Angles</h2>
<div class="two">
  <div class="card">
    <h3>Optimal Beamforming Steered Directions</h3>
    <div id="plot-angles" class="small-chart"></div>
  </div>
  <div class="card">
    <h3>Cluster Tier Stratification</h3>
    <div id="plot-tiers" class="small-chart"></div>
  </div>
</div>

<h2>2D Projection — Coloured by Method</h2>
<div class="card"><div id="plot-method" class="chart"></div></div>

<h2>2D Projection — Coloured by Native High-Dimensional Cluster</h2>
<div class="card"><div id="plot-cluster" class="chart"></div></div>

<h2>All-Beam Noise Distance <span class="badge" id="allbeam-badge"></span></h2>
<div class="note-box" id="allbeam-note"></div>
<div class="two">
  <div class="card">
    <h3>Every Beam Scored Against Its Own Reference</h3>
    <div id="allbeam-method"></div>
  </div>
  <div class="card">
    <h3>Does Choosing the Best Beam Explain the Gain?</h3>
    <div id="allbeam-selection"></div>
  </div>
</div>
<div class="card" style="margin-top:16px">
  <h3>Per-Beam Detail</h3>
  <div id="allbeam-detail" style="max-height:420px; overflow:auto"></div>
</div>

<h2>Per-Method Evaluation Summary</h2>
<div class="card" id="method-table"></div>

<h2>Run Information & Provenance</h2>
<div class="card" id="run-info"></div>
</main>

<script>
const report = __REPORT__;
const colors = ['#2563eb','#7c3aed','#16a34a','#ca8a04','#dc2626','#0891b2','#db2777','#4f46e5'];
const clusterColors = ['#ef4444','#3b82f6','#8b5cf6','#22c55e','#eab308','#f97316','#06b6d4','#ec4899','#84cc16','#6366f1','#14b8a6','#f43f5e'];
const plotLayout = {
  margin:{l:48,r:20,t:20,b:48}, hovermode:'closest',
  xaxis:{title:'Projection-1',showgrid:false,zeroline:false},
  yaxis:{title:'Projection-2',showgrid:false,zeroline:false},
  legend:{orientation:'h',y:1.08}, paper_bgcolor:'white', plot_bgcolor:'white'
};
const plotConfig = {responsive:true, displaylogo:false, toImageButtonOptions:{format:'png',filename:report.model+'_spatial_embedding'} };
const hover = '<b>%{customdata[0]}</b><br>Method: %{customdata[1]}<br>Window: %{customdata[2]}–%{customdata[3]} s<br>Azimuth: %{customdata[4]}°<br>Elevation: %{customdata[5]}°<br>Cluster: %{customdata[6]}<br>Model: %{customdata[7]}<extra></extra>';

const ov = report.stats.overview;
const hyp = (report.stats.matched_analysis || {}).summary_stats || {};
const nWin = hyp.n_matched_windows || Math.round(ov.total_embeddings / 4);

const allTests = hyp.hypothesis_tests || {};
function winCard(method, label) {
  const t = allTests[method];
  return (t && t.win_rate_pct !== undefined) ? [label, `${t.win_rate_pct}%`] : null;
}

document.getElementById('overview').innerHTML = [
  ['Matched Windows', nWin.toLocaleString()],
  ['Total Points', ov.total_embeddings.toLocaleString()],
  ['High-Dim Clusters', ov.n_clusters.toLocaleString()],
  ['Noise Ratio', `${ov.noise_pct}%`],
  winCard('bf_LabIR', 'LabIR Win Rate'),
  winCard('bf_SPIR', 'SPIR Win Rate'),
].filter(Boolean).map(x => `<div class="card"><div class="stat-value">${x[1]}</div><div class="stat-label">${x[0]}</div></div>`).join('');

// Interpretation card
const nar = report.narrative;
if (!nar) {
  document.getElementById('narrative-section').style.display = 'none';
} else {
  const isLLM = nar.source && nar.source !== 'deterministic';
  document.getElementById('narrative-badge').textContent = isLLM ? `AI-written · ${nar.source}` : 'rule-based summary';
  const list = items => `<ul>${items.map(s => `<li>${s}</li>`).join('')}</ul>`;
  document.getElementById('narrative').innerHTML =
    `<h3>${nar.headline || ''}</h3>` +
    list(nar.findings || []) +
    ((nar.caveats || []).length ? `<h3 style="margin-top:16px">Caveats</h3>${list(nar.caveats)}` : '') +
    `<p class="narrative-foot">Written from the numbers in this report by ${isLLM ? nar.source : 'a rule-based template'}${nar.generated_at ? ' on ' + nar.generated_at : ''}. Check it against the tables below before quoting it.</p>`;
}

// Pairwise Plot & Table
const tests = hyp.hypothesis_tests || {};
const testMethods = Object.keys(tests);
const deltaValues = testMethods.map(m => tests[m].mean_delta_vs_mono);
const winRates = testMethods.map(m => tests[m].win_rate_pct);

Plotly.newPlot('plot-gain', [{
  x: testMethods,
  y: deltaValues,
  type: 'bar',
  marker: {color: ['#8b5cf6','#16a34a','#0891b2']},
  text: deltaValues.map(v => (v >= 0 ? '+' : '') + v.toFixed(4)),
  textposition: 'outside'
}], {
  margin:{l:48,r:20,t:20,b:40},
  yaxis:{title:'Mean &Delta; vs Mono (Noise Distance)'},
  paper_bgcolor:'white', plot_bgcolor:'white'
}, plotConfig);

let hypTableHtml = `<table><thead><tr><th>Method</th><th>Mean &Delta;</th><th>Win Rate</th><th>Wilcoxon p</th><th>Cliff's &delta;</th></tr></thead><tbody>`;
testMethods.forEach(m => {
  const t = tests[m];
  const pStr = t.wilcoxon_p_value !== null ? (t.wilcoxon_p_value < 0.001 ? '<0.001' : t.wilcoxon_p_value.toFixed(4)) : 'N/A';
  const sig = t.is_significant_p05 ? '<span class="sig-star"> * (p<0.05)</span>' : '';
  hypTableHtml += `<tr><td><b>${m}</b></td><td>${(t.mean_delta_vs_mono >= 0 ? '+' : '') + t.mean_delta_vs_mono.toFixed(4)}</td><td><b>${t.win_rate_pct}%</b></td><td class="p-val">${pStr}${sig}</td><td>${t.cliffs_delta}</td></tr>`;
});
hypTableHtml += `</tbody></table>`;
document.getElementById('pairwise-table').innerHTML = hypTableHtml;

// Steered angles distribution, one trace per beamforming method
const beamAll = hyp.beam_distribution || {};
const beamMethods = Object.keys(beamAll);
const azKeys = [...new Set(beamMethods.flatMap(m => Object.keys(beamAll[m].azimuths || {})))].sort((a,b) => Number(a) - Number(b));
const beamColors = {'bf_LabIR':'#7c3aed', 'bf_SPIR':'#16a34a'};
Plotly.newPlot('plot-angles', beamMethods.map(m => ({
  x: azKeys.map(k => `${k}°`),
  y: azKeys.map(k => (beamAll[m].azimuths || {})[k] || 0),
  name: m,
  type: 'bar',
  marker: {color: beamColors[m] || '#0891b2'}
})), {
  margin:{l:48,r:20,t:20,b:40},
  barmode:'group',
  legend:{orientation:'h',y:1.12},
  xaxis:{title:'Selected Azimuth'},
  yaxis:{title:'Count'},
  paper_bgcolor:'white', plot_bgcolor:'white'
}, plotConfig);

// Cluster Tiers
const tiers = (report.stats.shared_cluster_analysis || {}).summary || {};
Plotly.newPlot('plot-tiers', [{
  labels: ['Tier 1 (All 4)', 'Tier 2 (3 Methods)', 'Tier 3 (2 Methods)', 'Tier 4 (BF Only)', 'Noise'],
  values: [tiers.tier_1_all_methods||0, tiers.tier_2_three_methods||0, tiers.tier_3_two_methods||0, tiers.tier_4_bf_only||0, tiers.noise||0],
  type: 'pie',
  hole: 0.45,
  marker: {colors: ['#22c55e','#3b82f6','#eab308','#ec4899','#94a3b8']}
}], {
  margin:{l:20,r:20,t:20,b:20},
  showlegend: true,
  legend:{orientation:'h',y:-0.1}
}, plotConfig);

function indicesWhere(predicate) { const out=[]; report.umap.forEach((_,i)=>{if(predicate(i)) out.push(i);}); return out; }
function traceForIndices(indices, name, color) {
  return {x:indices.map(i=>report.umap[i][0]), y:indices.map(i=>report.umap[i][1]),
    customdata:indices.map(i=>report.point_meta[i]), name, mode:'markers', type:'scattergl',
    marker:{color,size:6,opacity:0.75}, hovertemplate:hover};
}
const methodTraces = report.method_names.map((name,m) => traceForIndices(indicesWhere(i=>report.methods[i]===m),name,colors[m%colors.length]));
Plotly.newPlot('plot-method', methodTraces, {...plotLayout, height:500, title:'2D Projection by Method'}, plotConfig);

const clusterIds = [...new Set(report.clusters)].sort((a,b)=>a-b);
const clusterTraces = clusterIds.map((cid,j) => traceForIndices(indicesWhere(i=>report.clusters[i]===cid), cid===-1?'Noise':`CL-${String(cid).padStart(3,'0')}`, cid===-1?'#94a3b8':clusterColors[j%clusterColors.length]));
Plotly.newPlot('plot-cluster', clusterTraces, {...plotLayout, height:500, title:'High-Dimensional HDBSCAN Clusters'}, plotConfig);

// All-beam analysis
const ab = report.stats.all_beam_analysis || {};
if (!ab.available) {
  document.getElementById('allbeam-note').textContent = 'No noise reference resolved, so no beam could be scored.';
  document.getElementById('allbeam-badge').textContent = 'unavailable';
} else {
  document.getElementById('allbeam-badge').textContent = `${ab.n_scored.toLocaleString()} of ${ab.n_points.toLocaleString()} embeddings`;
  document.getElementById('allbeam-note').innerHTML =
    `Every steering direction is measured against the reference captured through that same beam, including directions never selected as &theta;*(t). ` +
    `The matched analysis higher up keeps one beam per window; this table keeps all ${ab.n_points.toLocaleString()} of them` +
    (ab.n_unscored ? `, and ${ab.n_unscored.toLocaleString()} stayed unscored.` : '.');

  document.getElementById('allbeam-method').innerHTML =
    `<table><thead><tr><th>Method</th><th>Beams</th><th>N</th><th>Mean Noise Distance</th><th>&Delta; vs Mono</th></tr></thead><tbody>` +
    (ab.per_method || []).map(r => r.n ? `<tr><td><b>${r.method}</b></td><td>${r.n_beams}</td><td>${r.n.toLocaleString()}</td><td>${r.mean_noise_distance.toFixed(4)}</td><td>${r.delta_vs_mono === undefined ? '—' : (r.delta_vs_mono >= 0 ? '+' : '') + r.delta_vs_mono.toFixed(4)}</td></tr>` : `<tr><td><b>${r.method}</b></td><td colspan="4">unscored</td></tr>`).join('') +
    `</tbody></table>`;

  const sel = ab.selection_comparison || {};
  const selRows = Object.keys(sel).map(m => {
    const t = sel[m];
    const cell = o => `${(o.mean_delta_vs_mono >= 0 ? '+' : '') + o.mean_delta_vs_mono.toFixed(4)}<br><span style="color:var(--muted)">${o.win_rate_pct}% win</span>`;
    return `<tr><td><b>${m}</b><br><span style="color:var(--muted)">${t.n_beams_per_window} beams/window</span></td><td>${cell(t.best_beam)}</td><td>${cell(t.median_beam)}</td><td>${cell(t.mean_beam)}</td><td><b>${(t.selection_effect >= 0 ? '+' : '') + t.selection_effect.toFixed(4)}</b></td></tr>`;
  }).join('');
  document.getElementById('allbeam-selection').innerHTML =
    `<table><thead><tr><th>Method</th><th>Best beam</th><th>Median beam</th><th>Mean beam</th><th>Best &minus; Median</th></tr></thead><tbody>${selRows}</tbody></table>` +
    `<p style="color:var(--muted);font-size:12px;margin:10px 0 0">The last column is the part of the advantage that comes from being allowed to pick the best of several beams. Mono and SA have one channel and no choice, so they cannot gain it.</p>`;

  document.getElementById('allbeam-detail').innerHTML =
    `<table><thead><tr><th>Method</th><th>Beam</th><th>Condition</th><th>N</th><th>Mean</th><th>SD</th><th>Reference key</th></tr></thead><tbody>` +
    (ab.per_beam || []).map(r => `<tr><td>${r.method}</td><td><code>${r.beam}</code></td><td>${r.condition || '—'}</td><td>${r.n}</td><td>${r.mean_noise_distance.toFixed(4)}</td><td>${r.std_noise_distance.toFixed(4)}</td><td><code>${r.noise_key}</code></td></tr>`).join('') +
    `</tbody></table>`;
}

const methodRows = report.stats.per_method || [];
const noiseRows = (report.stats.noise_analysis || {}).per_method || [];
const noiseMap = {};
noiseRows.forEach(r => { noiseMap[r.method] = r.mean_noise_distance; });

document.getElementById('method-table').innerHTML = `<table><thead><tr><th>Method</th><th>Embeddings (N)</th><th>In High-Dim Cluster</th><th>Noise %</th><th>Mean Noise Distance</th><th>Intra-method Cosine</th></tr></thead><tbody>${methodRows.map(x=>`<tr><td><b>${x.method}</b></td><td>${x.n_embeddings.toLocaleString()}</td><td>${x.n_in_cluster.toLocaleString()}</td><td>${x.noise_pct}%</td><td>${noiseMap[x.method] !== undefined ? noiseMap[x.method].toFixed(4) : 'N/A'}</td><td>${x.cosine_sim ?? 'N/A'}</td></tr>`).join('')}</tbody></table>`;

document.getElementById('run-info').innerHTML = `<table><tbody>
<tr><th>Model</th><td>${report.model}</td></tr>
<tr><th>Date</th><td>${report.date}</td></tr>
<tr><th>Design Mode</th><td>Matched-window pairwise: every scored window is aligned across all methods, so each method is judged on identical audio</td></tr>
<tr><th>Methods Compared</th><td>${report.method_names.join(', ')} (baseline: mono)</td></tr>
<tr><th>Direction Selection</th><td>&theta;*(t) is the steered direction with the largest cosine distance from this model's noise-reference embeddings</td></tr>
<tr><th>Noise Distance</th><td>Cosine distance to the mean noise-reference vector; not a dB signal-to-noise ratio</td></tr>
<tr><th>High-Dim Clustering</th><td>HDBSCAN on the full ${report.stats.overview.embedding_dim}-D space, cosine metric (min_cluster_size=${report.parameters.min_cluster_size})</td></tr>
<tr><th>2D Projection</th><td>UMAP, visual rendering only; never used for clustering or statistics</td></tr>
<tr><th>Generated At</th><td>${report.parameters.generated_at}</td></tr>
</tbody></table>`;
</script>
</body>
</html>'''

    replacements = {
        "__TITLE__": html.escape(title),
        "__SOURCE__": html.escape(source_dir),
        "__REPORT__": _json_for_html(data),
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def _win_badge(label: str, pct) -> str:
    """Colour-coded win-rate chip. Green >=80, amber 60-80, grey <60."""
    if pct is None:
        return f'<span class="chip chip-na">{html.escape(label)}: n/a</span>'
    cls = "chip-hi" if pct >= 80 else ("chip-mid" if pct >= 60 else "chip-lo")
    return f'<span class="chip {cls}">{html.escape(label)}: {pct}%</span>'


def _method_note_html() -> str:
    """Shared derivation of the numbers shown on every model page."""
    return r'''
<section class="method">
<h2>How these numbers are computed</h2>

<p>Every model page reports the same quantities. They are defined once here.</p>

<h3>1. Habitat noise prototype</h3>
<p>Noise references are cut from vetted background intervals detected on a single
reference beam, <code>LabIR(S05_000)</code>, then the identical time intervals are cut
from every other method so all methods hear the same silence. For a given date,
time condition and beam, the \(K\) reference windows are embedded, L2-normalised,
averaged and re-normalised:</p>
<p>$$\bar{\mathbf{n}} \;=\; \frac{\sum_{k=1}^{K} \hat{\mathbf{n}}_k}{\left\lVert \sum_{k=1}^{K} \hat{\mathbf{n}}_k \right\rVert},
\qquad \hat{\mathbf{n}}_k = \frac{\mathbf{n}_k}{\lVert \mathbf{n}_k \rVert}$$</p>

<h3>2. Noise distance</h3>
<p>For a window embedding \(\mathbf{e}\), scored against the prototype of its own
date, its own time condition and its own beam:</p>
<p>$$d(\mathbf{e}) \;=\; 1 - \frac{\mathbf{e} \cdot \bar{\mathbf{n}}}{\lVert \mathbf{e} \rVert \, \lVert \bar{\mathbf{n}} \rVert}$$</p>
<p>Larger means further from the background. This is a cosine distance in embedding
space, <em>not</em> a signal-to-noise ratio in decibels.</p>

<h3>3. Matched windows</h3>
<p>A time window enters the comparison only if all four methods produced an embedding
for it. Let \(\mathcal{W}\) be that set, \(N = \lvert \mathcal{W} \rvert\). This removes
sample-size imbalance between a 1-channel method and a 31-beam one.</p>

<h3>4. Beam selection</h3>
<p>For a beamformed method with beam set \(\mathcal{B}\), each window is represented by
its single best-scoring beam:</p>
<p>$$\theta^{*}(t) \;=\; \arg\max_{b \in \mathcal{B}} \; d(\mathbf{e}_{t,b}),
\qquad d_{m}(t) \;=\; d\!\left(\mathbf{e}_{t,\theta^{*}(t)}\right)$$</p>
<p>For <code>mono</code> and <code>sa</code>, \(\lvert \mathcal{B} \rvert = 1\) and no choice is made.</p>

<h3>5. Win rate</h3>
<p>Paired against the mono channel of the same window, \(\Delta_{m}(t) = d_{m}(t) - d_{\text{mono}}(t)\):</p>
<p>$$\mathrm{WR}_{m} \;=\; \frac{100}{N} \sum_{t \in \mathcal{W}} \mathbb{1}\!\left[\Delta_{m}(t) > 0\right]$$</p>
<p>So a win rate of 94.2% means the method beat mono on 94.2% of matched windows.
50% is the coin-flip line.</p>

<h3>6. The other metrics</h3>
<p>Mean delta, the average size of the advantage:</p>
<p>$$\bar{\Delta}_{m} \;=\; \frac{1}{N} \sum_{t \in \mathcal{W}} \Delta_{m}(t)$$</p>
<p>Cliff's delta, a sign-based effect size on the paired differences:</p>
<p>$$\delta_{m} \;=\; \frac{1}{N}\sum_{t} \mathbb{1}\!\left[\Delta_{m}(t) > 0\right] \;-\; \frac{1}{N}\sum_{t} \mathbb{1}\!\left[\Delta_{m}(t) < 0\right]$$</p>
<p>Reported \(p\) values come from a one-sided Wilcoxon signed-rank test on the paired
distances, alternative <em>greater</em>. Note that this is the paired-dominance form of
Cliff's delta, computed on matched pairs rather than over all cross-pairs.</p>

<h3>7. What the win rate does not tell you</h3>
<div class="warn">
<p>The win rate above uses \(\theta^{*}(t)\), the beam chosen by maximising the very
quantity that is then reported. Taking a maximum over 19 or 31 candidates raises the
score even when no beam carries real signal, so this figure is an <strong>upper
bound</strong>, not an unbiased estimate.</p>
<p>Each model page therefore also carries <em>Does Choosing the Best Beam Explain the
Gain?</em>, which repeats the comparison using the median and mean beam instead of the
best. The gap between the best-beam and median-beam columns is the size of the
selection effect. Read the two together.</p>
</div>

<h3>8. Sorting</h3>
<p>Cards are ordered by the higher of the two win rates, descending, so the strongest
and weakest models sit at opposite ends of the list.</p>
</section>
'''


def _index_html(manifest: Dict[str, Any]) -> str:
    entries = []
    for item in manifest["models"]:
        ov = item["overview"]
        hyp = (item.get("matched_analysis") or {}).get("summary_stats", {}).get("hypothesis_tests", {})
        spir = hyp.get("bf_SPIR", {}).get("win_rate_pct")
        labir = hyp.get("bf_LabIR", {}).get("win_rate_pct")
        rank = max([v for v in (spir, labir) if v is not None], default=-1.0)
        entries.append((rank, item, ov, spir, labir))
    entries.sort(key=lambda e: e[0], reverse=True)

    cards = []
    for _rank, item, ov, spir, labir in entries:
        href = html.escape(item["html"])
        cards.append(
            f'''<article class="card"><h2><a href="{href}">{html.escape(item["model"])}</a></h2>
<p class="meta">{ov["total_embeddings"]:,} points &middot; {ov["embedding_dim"]}D &middot; {ov["n_clusters"]} clusters</p>
<p class="rates">{_win_badge("SPIR Win Rate", spir)} {_win_badge("LabIR Win Rate", labir)}</p>
<p><a href="{href}">Open evaluation dashboard &rarr;</a></p></article>'''
        )
    cards_html = "\n".join(cards)

    skipped = manifest.get("skipped_models", [])
    skipped_html = ""
    if skipped:
        skipped_html = "<h2>Skipped model directories</h2><ul>" + "".join(
            f"<li><code>{html.escape(item['model'])}</code>: {html.escape(item['reason'])}</li>"
            for item in skipped
        ) + "</ul>"

    location = html.escape(manifest["location"])
    date_str = html.escape(manifest["date"])

    css = '''
:root { --bg:#f8fafc; --card:#fff; --text:#0f172a; --muted:#64748b; --border:#e2e8f0; --primary:#2563eb; }
body { margin:0; padding:32px; background:var(--bg); color:var(--text); font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1100px; margin:auto; }
h1 { margin:0 0 6px; font-size:26px; font-weight:700; }
.subtitle { color:var(--muted); margin-bottom:24px; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:16px; margin-top:16px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:20px; box-shadow:0 1px 3px rgba(0,0,0,0.05); }
.card h2 { margin:0 0 8px; font-size:17px; }
.meta { margin:0 0 10px; color:var(--muted); font-variant-numeric:tabular-nums; }
.rates { margin:0 0 12px; display:flex; flex-wrap:wrap; gap:6px; }
.chip { padding:3px 9px; border-radius:10px; font-weight:600; font-size:12px; white-space:nowrap; font-variant-numeric:tabular-nums; }
.chip-hi { background:#dcfce7; color:#166534; }
.chip-mid { background:#fef3c7; color:#92400e; }
.chip-lo { background:#e2e8f0; color:#475569; }
.chip-na { background:#f1f5f9; color:#94a3b8; }
a { color:var(--primary); text-decoration:none; font-weight:600; }
a:hover { text-decoration:underline; }
.method { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:24px 28px; margin-top:32px; }
.method h2 { margin-top:0; font-size:20px; }
.method h3 { font-size:15px; margin:22px 0 6px; }
.method p { margin:8px 0; }
.method code { background:#f1f5f9; padding:1px 5px; border-radius:4px; font-size:13px; }
.method .warn { background:#fffbeb; border-left:3px solid #f59e0b; padding:12px 16px; border-radius:0 8px 8px 0; margin-top:10px; }
'''

    katex = '''
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.css">
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/katex.min.js"></script>
<script defer src="https://cdnjs.cloudflare.com/ajax/libs/KaTeX/0.16.9/contrib/auto-render.min.js"></script>
<script>
document.addEventListener("DOMContentLoaded", function () {
  if (window.renderMathInElement) {
    renderMathInElement(document.body, {
      delimiters: [
        {left: "$$", right: "$$", display: true},
        {left: "\\\\(", right: "\\\\)", display: false}
      ],
      throwOnError: false
    });
  }
});
</script>'''

    return (
        '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f'<title>Spatial Bioacoustic Evaluation &mdash; {location} &mdash; {date_str}</title>\n'
        f'{katex}\n<style>{css}</style></head><body>\n<main>\n'
        '<h1>Spatial Bioacoustic Evaluation Dashboard</h1>\n'
        f'<div class="subtitle">Location: <b>{location}</b> | Date: <b>{date_str}</b> | '
        'Design: <b>Matched-Window Pairwise Direction Selection</b></div>\n'
        f'<div class="grid">{cards_html}</div>\n'
        f'{skipped_html}\n'
        f'{_method_note_html()}\n'
        '</main></body></html>'
    )


def _write_plotly_asset(output_dir: Path, custom_plotly_js: Optional[str] = None) -> None:
    asset_path = output_dir / "plotly.min.js"
    if custom_plotly_js:
        shutil.copyfile(custom_plotly_js, asset_path)
        return
    if asset_path.is_file() and asset_path.stat().st_size > 0:
        return
    import plotly
    plotly_path = Path(plotly.__file__).parent / "package_data" / "plotly.min.js"
    if plotly_path.is_file():
        shutil.copyfile(plotly_path, asset_path)
        return
    raise RuntimeError("plotly.min.js could not be located; provide --plotly-js explicitly")


def _clean_previous_output(output_dir: Path) -> None:
    generated = {"index.html", "manifest.json", "manifest.zip"}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        with open(manifest_path, encoding="utf-8") as f:
            data = json.load(f)
        for item in data.get("models", []):
            filename = item.get("html")
            if isinstance(filename, str) and filename.endswith(".html"):
                generated.add(filename)
    for filename in generated:
        path = output_dir / filename
        if path.is_file():
            path.unlink()


def build_report(
    *,
    data_dir: str,
    location: str,
    date_str: Optional[str],
    models: Sequence[str],
    methods: Sequence[str],
    output_dir: Path,
    plotly_js: Optional[str],
    umap_n_neighbors: int,
    umap_min_dist: float,
    min_cluster_size: int,
    make_zip: bool,
    clean_output: bool,
    matched_mode: bool = True,
    noise_dir: Optional[str] = None,
    narrative_mode: str = "auto",
    narrative_model: str = narrative_defaults.DEFAULT_MODEL,
    narrative_base_url: str = narrative_defaults.DEFAULT_BASE_URL,
) -> Path:
    model_dirs = [bacpipe_embeddings_dir(data_dir, location, model) for model in models]
    resolved_date = _resolve_date(model_dirs, date_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        _clean_previous_output(output_dir)
    _write_plotly_asset(output_dir, plotly_js)

    manifest: Dict[str, Any] = {
        "schema_version": 2,
        "location": location,
        "date": resolved_date,
        "models": [],
        "models_requested": list(models),
        "skipped_models": [],
        "methods": list(methods),
        "plotly_asset": "plotly.min.js",
    }

    for model in models:
        source_dir = bacpipe_embeddings_dir(data_dir, location, model)
        meta_dir = bacpipe_meta_dir(location, model)
        print(f"\n== {model} ==")
        _embeddings, X_raw, y_raw, flat_meta_raw, method_names = load_embeddings_from_dir(
            source_dir,
            methods=methods,
            date_filter=[resolved_date],
            source_tag=f"bacpipe:{model}",
            meta_dir=meta_dir,
        )
        if len(X_raw) == 0:
            reason = "no embeddings for requested date"
            print(f"  skipped: {reason}")
            manifest["skipped_models"].append({"model": model, "reason": reason})
            continue

        noise_vectors = load_noise_embeddings(meta_dir, noise_dir=noise_dir, expected_dim=X_raw.shape[1] if len(X_raw) > 0 else None)

        if matched_mode and noise_vectors:
            print("  Aligning matched windows and selecting optimal beamforming direction...")
            matched_res = align_and_select_matched_windows(
                X_raw, y_raw, method_names, flat_meta_raw, noise_vectors
            )
            matched_idx = matched_res["matched_indices"]
            X = X_raw[matched_idx]
            y_method = y_raw[matched_idx]
            flat_meta = [flat_meta_raw[i] for i in matched_idx]
            n_win = matched_res["summary_stats"]["n_matched_windows"]
            print(f"  Matched dataset: {len(X)} points across {n_win} windows")
        else:
            matched_res = None
            X = X_raw
            y_method = y_raw
            flat_meta = flat_meta_raw

        # Run native high-dimensional HDBSCAN (cosine metric)
        cluster_size = min(min_cluster_size, max(2, len(X)))
        print(f"  High-Dim HDBSCAN: clustering {len(X)} points (min_cluster_size={cluster_size})...")
        cluster_labels = run_hdbscan(X, min_cluster_size=cluster_size)

        # Run 2D Projection strictly for visualization
        n_neighbors = min(umap_n_neighbors, max(2, len(X) - 1))
        print(f"  2D Projection: n_neighbors={n_neighbors}, min_dist={umap_min_dist}...")
        umap_2d = run_umap(X, n_neighbors=n_neighbors, min_dist=umap_min_dist)

        # Compute stats & noise analysis
        stats = compute_stats(X, y_method, method_names, cluster_labels)
        shared = compute_shared_cluster_analysis(
            X, cluster_labels, y_method, method_names, flat_meta
        )
        stats["shared_cluster_analysis"] = shared
        if matched_res:
            stats["matched_analysis"] = matched_res

        if noise_vectors:
            stats["noise_analysis"] = compute_noise_distance(
                X, y_method, method_names, cluster_labels, noise_vectors,
                shared_analysis=shared, flat_meta=flat_meta,
            )
            # Every beam is scored here, including directions never chosen as
            # theta*(t); the matched analysis above keeps only one beam per window.
            stats["all_beam_analysis"] = compute_all_beam_noise_analysis(
                X_raw, y_raw, method_names, flat_meta_raw, noise_vectors
            )
        else:
            stats["noise_analysis"] = {"available": False}
            stats["all_beam_analysis"] = {"available": False}

        parameters = {
            "design_mode": "matched_window_pairwise" if matched_mode else "raw_all_angles",
            "umap_n_neighbors": n_neighbors,
            "umap_min_dist": umap_min_dist,
            "min_cluster_size": cluster_size,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        digest = build_digest(
            model=model,
            date_str=resolved_date,
            location=location,
            method_names=method_names,
            stats=stats,
            parameters=parameters,
        )
        narrative = load_or_generate_narrative(
            output_dir, model, digest, mode=narrative_mode,
            llm_model=narrative_model, base_url=narrative_base_url,
        )

        point_meta = _compact_metadata(flat_meta, y_method, method_names, cluster_labels, model)
        filename = f"{model}.html"
        (output_dir / filename).write_text(
            _model_html(
                model=model,
                source_dir=source_dir,
                date_str=resolved_date,
                umap_2d=umap_2d,
                y_method=y_method,
                method_names=method_names,
                cluster_labels=cluster_labels,
                point_meta=point_meta,
                stats=stats,
                parameters=parameters,
                narrative=narrative,
            ),
            encoding="utf-8",
        )

        manifest["models"].append(convert_numpy_types({
            "model": model,
            "html": filename,
            "source_dir": source_dir,
            "overview": stats["overview"],
            "matched_analysis": matched_res,
        }))

    (output_dir / "manifest.json").write_text(
        json.dumps(convert_numpy_types(manifest), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "index.html").write_text(_index_html(manifest), encoding="utf-8")

    if make_zip:
        zip_path = output_dir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as arc:
            for item in sorted(output_dir.rglob("*")):
                if item.is_file():
                    arc.write(item, item.relative_to(output_dir))
        print(f"\nWrote portable ZIP archive -> {zip_path}")

    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default=ANALYSIS_OUTPUT, help="Root folder containing sea-data")
    parser.add_argument("--location", default="2A400", help="Location folder name")
    parser.add_argument("--date", default=None, help="Date filter (YYYY-MM-DD); auto-resolved if omitted")
    parser.add_argument("--models", nargs="*", default=[], help="Specific models to visualise; defaults to all")
    parser.add_argument("--methods", nargs="*", default=DEFAULT_METHODS_LIST, help="Audio extraction methods")
    parser.add_argument("--output-dir", default=None, help="Target HTML folder")
    parser.add_argument("--plotly-js", default=None, help="Path to local plotly.min.js file")
    parser.add_argument("--umap-n-neighbors", type=int, default=15, help="Projection neighbour count")
    parser.add_argument("--umap-min-dist", type=float, default=0.1, help="Projection minimum distance")
    parser.add_argument("--min-cluster-size", type=int, default=5, help="HDBSCAN minimum cluster size")
    parser.add_argument("--no-zip", action="store_true", help="Do not create a zip archive")
    parser.add_argument("--no-clean", action="store_true", help="Keep preexisting generated HTML files")
    parser.add_argument("--no-matched", action="store_true", help="Disable matched-window pairwise selection (use raw)")
    parser.add_argument("--noise-dir", default=None, help="Directory containing noise_*.npy embeddings")
    parser.add_argument("--narrative", choices=["auto", "force", "off"], default="auto",
                        help="Interpretation card: auto reuses a cached narrative, force regenerates, off uses the rule-based summary")
    parser.add_argument("--narrative-model", default=narrative_defaults.DEFAULT_MODEL,
                        help="Chat model id used to write the interpretation card")
    parser.add_argument("--narrative-base-url", default=narrative_defaults.DEFAULT_BASE_URL,
                        help="OpenAI-compatible chat completions endpoint")
    args = parser.parse_args()

    models = args.models or discover_models(args.data_dir, args.location, args.date)
    if not models:
        print("No models found to visualise.")
        sys.exit(1)

    model_dirs = [bacpipe_embeddings_dir(args.data_dir, args.location, m) for m in models]
    resolved_date = _resolve_date(model_dirs, args.date)
    output_dir = Path(args.output_dir or dashboards_dir(args.location, resolved_date))

    build_report(
        data_dir=args.data_dir,
        location=args.location,
        date_str=resolved_date,
        models=models,
        methods=args.methods,
        output_dir=output_dir,
        plotly_js=args.plotly_js,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cluster_size=args.min_cluster_size,
        make_zip=not args.no_zip,
        clean_output=not args.no_clean,
        matched_mode=not args.no_matched,
        noise_dir=args.noise_dir,
        narrative_mode=args.narrative,
        narrative_model=args.narrative_model,
        narrative_base_url=args.narrative_base_url,
    )


if __name__ == "__main__":
    main()
