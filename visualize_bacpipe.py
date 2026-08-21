#!/usr/bin/env python3
"""Create offline HTML visualisations from existing bacpipe embeddings.

The script consumes the schema-aligned bacpipe output produced by
experiments/bacpipe/run_pilot.py. It does not run an embedder or download model
checkpoints. One HTML report is produced per model because model spaces and
embedding dimensions must not be mixed.

Example:
    python visualize_bacpipe.py \
        --data-dir /rds/general/user/ri322/home/sea-data \
        --location 2A400 \
        --date 2026-04-26 \
        --models all

The output directory contains one shared plotly.min.js asset, one HTML file per
model, index.html, manifest.json, and (unless --no-zip is supplied) a ZIP next
to the output directory. Model directories are discovered automatically; a
summary-only or incomplete model directory is ignored.
"""

from __future__ import annotations

import argparse
import html
import json
import os

import shutil
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from cluster_poc import (
    compute_noise_distance,
    compute_shared_cluster_analysis,
    compute_stats,
    load_noise_embeddings,
    run_hdbscan,
    run_umap,
)
from config import ANALYSIS_OUTPUT
from embedding_io import load_embeddings_from_dir, list_embedding_files
from embedding_schema import DEFAULT_METHODS, bacpipe_embeddings_dir, embeddings_root


DEFAULT_METHODS_LIST = list(DEFAULT_METHODS)


def _json_for_html(value: Any) -> str:
    """Serialise JSON safely for embedding inside a script tag."""
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
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
    root = Path(embeddings_root(data_dir, location)) / "bacpipe"
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
) -> str:
    """Build one HTML report that references the shared Plotly asset."""
    title = f"bacpipe embeddings — {model} — {date_str}"
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
    }

    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<script src="plotly.min.js"></script>
<style>
:root { color-scheme: light; --bg:#f8fafc; --card:#fff; --text:#172033; --muted:#64748b; --border:#e2e8f0; }
* { box-sizing:border-box; }
body { margin:0; padding:28px; background:var(--bg); color:var(--text); font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; }
main { max-width:1450px; margin:auto; }
h1 { margin:0 0 4px; font-size:26px; }
h2 { margin:28px 0 8px; font-size:19px; }
h3 { margin:0 0 12px; font-size:15px; }
.subtitle, .note { color:var(--muted); }
.subtitle { margin-bottom:22px; word-break:break-all; }
.grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:12px; }
.card { background:var(--card); border:1px solid var(--border); border-radius:12px; padding:16px; box-shadow:0 2px 8px #0f172a0a; }
.stat-value { font-size:23px; font-weight:700; }
.stat-label { color:var(--muted); margin-top:3px; }
.chart { height:530px; }
.small-chart { height:380px; }
.two { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
table { width:100%; border-collapse:collapse; }
th,td { padding:9px 8px; border-bottom:1px solid var(--border); text-align:left; }
th { color:var(--muted); font-weight:600; }
code { font-size:12px; }
@media(max-width:900px) { body { padding:14px; } .two { grid-template-columns:1fr; } }
</style>
</head>
<body>
<main>
<header>
<h1>__TITLE__</h1>
<div class="subtitle">Source: <code>__SOURCE__</code></div>
<p class="note">Each point is one model window. UMAP/HDBSCAN are exploratory visual analyses, not ground truth. Absolute WAV paths are intentionally omitted from this portable report.</p>
</header>

<section class="grid" id="overview"></section>

<h2>UMAP projection — coloured by method</h2>
<div class="card"><div id="plot-method" class="chart"></div></div>

<h2>UMAP projection — coloured by cluster</h2>
<div class="card"><div id="plot-cluster" class="chart"></div></div>

<div class="two">
<section>
<h2>Cluster sizes</h2>
<div class="card"><div id="plot-dist" class="small-chart"></div></div>
</section>
<section>
<h2>Noise distance by method</h2>
<div class="card"><div id="plot-noise" class="small-chart"></div></div>
</section>
</div>

<h2>Per-method summary</h2>
<div class="card" id="method-table"></div>

<h2>Run information</h2>
<div class="card" id="run-info"></div>
</main>
<script>
const report = __REPORT__;
const colors = ['#2563eb','#7c3aed','#16a34a','#ca8a04','#dc2626','#0891b2','#db2777','#4f46e5'];
const clusterColors = ['#ef4444','#3b82f6','#8b5cf6','#22c55e','#eab308','#f97316','#06b6d4','#ec4899','#84cc16','#6366f1','#14b8a6','#f43f5e'];
const plotLayout = {
  height: 510, margin:{l:48,r:20,t:12,b:48}, hovermode:'closest',
  xaxis:{title:'UMAP-1',showgrid:false,zeroline:false},
  yaxis:{title:'UMAP-2',showgrid:false,zeroline:false},
  legend:{orientation:'h',y:1.08}, paper_bgcolor:'white', plot_bgcolor:'white'
};
const plotConfig = {responsive:true, displaylogo:false, toImageButtonOptions:{format:'png',filename:report.model+'_embedding'} };
const hover = '<b>%{customdata[0]}</b><br>Method: %{customdata[1]}<br>Window: %{customdata[2]}–%{customdata[3]} s<br>Azimuth: %{customdata[4]}°<br>Elevation: %{customdata[5]}°<br>Cluster: %{customdata[6]}<br>Model: %{customdata[7]}<extra></extra>';

const ov = report.stats.overview;
document.getElementById('overview').innerHTML = [
  ['Embeddings', ov.total_embeddings.toLocaleString()],
  ['Clusters', ov.n_clusters.toLocaleString()],
  ['Embedding dimension', ov.embedding_dim],
  ['Noise points', `${ov.n_noise.toLocaleString()} (${ov.noise_pct}%)`],
].map(x => `<div class="card"><div class="stat-value">${x[1]}</div><div class="stat-label">${x[0]}</div></div>`).join('');

function indicesWhere(predicate) { const out=[]; report.umap.forEach((_,i)=>{if(predicate(i)) out.push(i);}); return out; }
function traceForIndices(indices, name, color) {
  return {x:indices.map(i=>report.umap[i][0]), y:indices.map(i=>report.umap[i][1]),
    customdata:indices.map(i=>report.point_meta[i]), name, mode:'markers', type:'scattergl',
    marker:{color,size:5,opacity:0.68}, hovertemplate:hover};
}
const methodTraces = report.method_names.map((name,m) => traceForIndices(indicesWhere(i=>report.methods[i]===m),name,colors[m%colors.length]));
Plotly.newPlot('plot-method', methodTraces, {...plotLayout, title:'Click or hover a point for window metadata'}, plotConfig);

const clusterIds = [...new Set(report.clusters)].sort((a,b)=>a-b);
const clusterTraces = clusterIds.map((cid,j) => traceForIndices(indicesWhere(i=>report.clusters[i]===cid), cid===-1?'Noise':`CL-${String(cid).padStart(3,'0')}`, cid===-1?'#94a3b8':clusterColors[j%clusterColors.length]));
Plotly.newPlot('plot-cluster', clusterTraces, {...plotLayout, title:'HDBSCAN cluster assignment'}, plotConfig);

const clusterStats = report.stats.per_cluster || [];
Plotly.newPlot('plot-dist', [{x:clusterStats.map(c=>`CL-${String(c.cluster_id).padStart(3,'0')}`),y:clusterStats.map(c=>c.size),type:'bar',marker:{color:clusterStats.map((_,i)=>clusterColors[i%clusterColors.length])}}], {...plotLayout,height:360, xaxis:{tickangle:-45,title:'Cluster'},yaxis:{title:'Points'},showlegend:false}, plotConfig);

const noise = (report.stats.noise_analysis || {}).per_method || [];
Plotly.newPlot('plot-noise', [{x:noise.map(x=>x.method),y:noise.map(x=>x.mean_noise_distance),type:'bar',marker:{color:noise.map((_,i)=>colors[i%colors.length])},text:noise.map(x=>x.mean_noise_distance.toFixed(4)),textposition:'outside'}], {...plotLayout,height:360,yaxis:{title:'Cosine distance from noise',range:[0,1]},showlegend:false}, plotConfig);

const methodRows = report.stats.per_method || [];
document.getElementById('method-table').innerHTML = `<table><thead><tr><th>Method</th><th>N</th><th>In cluster</th><th>Noise %</th><th>Unique clusters</th><th>Intra-method cosine</th></tr></thead><tbody>${methodRows.map(x=>`<tr><td><b>${x.method}</b></td><td>${x.n_embeddings.toLocaleString()}</td><td>${x.n_in_cluster.toLocaleString()}</td><td>${x.noise_pct}%</td><td>${x.n_unique_clusters}</td><td>${x.cosine_sim ?? 'n/a'}</td></tr>`).join('')}</tbody></table>`;

document.getElementById('run-info').innerHTML = `<table><tbody>
<tr><th>Model</th><td>${report.model}</td></tr>
<tr><th>Date</th><td>${report.date}</td></tr>
<tr><th>UMAP neighbours</th><td>${report.parameters.umap_n_neighbors}</td></tr>
<tr><th>UMAP min distance</th><td>${report.parameters.umap_min_dist}</td></tr>
<tr><th>HDBSCAN min cluster size</th><td>${report.parameters.min_cluster_size}</td></tr>
<tr><th>Generated at</th><td>${report.parameters.generated_at}</td></tr>
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


def _index_html(manifest: Dict[str, Any]) -> str:
    cards = []
    for item in manifest["models"]:
        ov = item["overview"]
        cards.append(
            f'''<article><h2><a href="{html.escape(item["html"])}">{html.escape(item["model"])}</a></h2>
<p>{ov["total_embeddings"]:,} embeddings · {ov["embedding_dim"]} dimensions · {ov["n_clusters"]} clusters</p>
<p><a href="{html.escape(item["html"])}">Open visualisation →</a></p></article>'''
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
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{location} bacpipe embedding visualisations</title>
<style>
body{{font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:#f8fafc;color:#172033;margin:0;padding:32px}}
main{{max-width:960px;margin:auto}}h1{{margin-bottom:4px}}.muted{{color:#64748b}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:16px;margin-top:24px}}
article{{background:white;border:1px solid #e2e8f0;border-radius:12px;padding:18px}}a{{color:#2563eb;text-decoration:none}}a:hover{{text-decoration:underline}}code{{font-size:12px}}
</style></head><body><main>
<h1>{location} bacpipe embedding visualisations</h1>
<p class="muted">Date: {date_str} · {len(manifest["models"])} model reports · generated from existing <code>.npy</code> embeddings; no model inference was performed.</p>
<div class="grid">{cards_html}</div>
{skipped_html}
<h2>Notes</h2><p class="muted">Each model is visualised separately because embedding spaces and dimensions differ. The individual reports reference the shared local <code>plotly.min.js</code> file, so the extracted folder works without internet access.</p>
</main></body></html>'''


def _write_plotly_asset(output_dir: Path, source: Optional[str]) -> Path:
    target = output_dir / "plotly.min.js"
    if source:
        shutil.copyfile(source, target)
        return target
    try:
        from plotly.offline import get_plotlyjs
    except ImportError as exc:
        raise RuntimeError(
            "Plotly is required to create the shared asset. Install it in the analysis environment "
            "or pass --plotly-js PATH to an existing plotly.min.js."
        ) from exc
    target.write_text(get_plotlyjs(), encoding="utf-8")
    return target


def _zip_output(output_dir: Path, zip_path: Path) -> None:
    """Create a portable archive whose extracted root contains index.html."""
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(output_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))


def _clean_previous_output(output_dir: Path) -> None:
    """Remove files generated by an earlier run, preserving unrelated files."""
    manifest_path = output_dir / "manifest.json"
    generated = {"index.html", "manifest.json", "plotly.min.js"}
    if manifest_path.is_file():
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        for item in previous.get("models", []):
            filename = item.get("html") if isinstance(item, dict) else None
            if isinstance(filename, str) and Path(filename).name == filename and filename.endswith(".html"):
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
) -> Path:
    model_dirs = [bacpipe_embeddings_dir(data_dir, location, model) for model in models]
    resolved_date = _resolve_date(model_dirs, date_str)
    output_dir.mkdir(parents=True, exist_ok=True)
    if clean_output:
        _clean_previous_output(output_dir)
    _write_plotly_asset(output_dir, plotly_js)

    manifest: Dict[str, Any] = {
        "schema_version": 1,
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
        print(f"\n== {model} ==")
        _embeddings, X, y_method, flat_meta, method_names = load_embeddings_from_dir(
            source_dir,
            methods=methods,
            date_filter=[resolved_date],
            source_tag=f"bacpipe:{model}",
        )
        if len(X) == 0:
            reason = "no embeddings for requested date"
            print(f"  skipped: {reason}")
            manifest["skipped_models"].append({"model": model, "reason": reason})
            continue

        n_neighbors = min(umap_n_neighbors, max(2, len(X) - 1))
        print(f"  UMAP: {len(X):,} points, dimension {X.shape[1]}, n_neighbors={n_neighbors}")
        umap_2d = run_umap(X, n_neighbors=n_neighbors, min_dist=umap_min_dist)
        cluster_size = min(min_cluster_size, max(2, len(X)))
        cluster_labels = run_hdbscan(
            umap_2d,
            min_cluster_size=cluster_size,
            min_samples=max(1, cluster_size // 2),
        )
        stats = compute_stats(X, y_method, method_names, cluster_labels, embeddings=_embeddings)
        shared = compute_shared_cluster_analysis(
            X, cluster_labels, y_method, method_names, flat_meta
        )
        stats["shared_cluster_analysis"] = shared
        noise_vectors = load_noise_embeddings(source_dir)
        if noise_vectors:
            stats["noise_analysis"] = compute_noise_distance(
                X, y_method, method_names, cluster_labels, noise_vectors,
                shared_analysis=shared,
            )
        else:
            stats["noise_analysis"] = {"available": False}

        parameters = {
            "umap_n_neighbors": n_neighbors,
            "umap_min_dist": umap_min_dist,
            "min_cluster_size": cluster_size,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
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
            ),
            encoding="utf-8",
        )
        manifest["models"].append({
            "model": model,
            "html": filename,
            "source_dir": source_dir,
            "overview": stats["overview"],
            "parameters": parameters,
        })
        print(f"  wrote {output_dir / filename}")

    if not manifest["models"]:
        raise RuntimeError(f"No reports were generated under {output_dir}")
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_dir / "index.html").write_text(_index_html(manifest), encoding="utf-8")

    if make_zip:
        zip_path = output_dir.parent / f"{location}_embedding_visuals_{resolved_date}.zip"
        _zip_output(output_dir, zip_path)
        print(f"\nZIP written to: {zip_path}")
        return zip_path
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Create offline HTML reports from bacpipe embeddings")
    parser.add_argument("--data-dir", default=ANALYSIS_OUTPUT,
                        help="Root containing {location}/embeddings (default: config ANALYSIS_OUTPUT)")
    parser.add_argument("--location", required=True, help="Location code, e.g. 2A400")
    parser.add_argument("--date", default=None, help="Date filter, e.g. 2026-04-26; inferred if unique")
    parser.add_argument("--models", default="all", help="Comma list or all")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS_LIST),
                        help="Comma list of methods to include")
    parser.add_argument("--output-dir", default=None,
                        help="Output folder; default: {data}/{location}/embedding_visuals/{date}")
    parser.add_argument("--plotly-js", default=None,
                        help="Use an existing plotly.min.js instead of importing plotly in Python")
    parser.add_argument("--umap-n-neighbors", type=int, default=30)
    parser.add_argument("--umap-min-dist", type=float, default=0.1)
    parser.add_argument("--min-cluster-size", type=int, default=5)
    parser.add_argument("--no-zip", action="store_true", help="Do not create the ZIP archive")
    parser.add_argument("--clean-output", action="store_true",
                        help="Remove reports/assets from an earlier run before writing")
    args = parser.parse_args()

    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    discovered = discover_models(args.data_dir, args.location, args.date)
    if not discovered:
        raise SystemExit(f"No bacpipe model embeddings found under {embeddings_root(args.data_dir, args.location)}")
    if args.models.lower() == "all":
        models = discovered
    else:
        models = [x.strip() for x in args.models.split(",") if x.strip()]
        missing = [x for x in models if x not in discovered]
        if missing:
            raise SystemExit(f"Requested models not found: {', '.join(missing)}; available: {', '.join(discovered)}")

    # Resolve the date before constructing the default output path.
    resolved_date = _resolve_date(
        [bacpipe_embeddings_dir(args.data_dir, args.location, m) for m in models], args.date
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(args.data_dir) / args.location / "embedding_visuals" / resolved_date
    )
    result = build_report(
        data_dir=args.data_dir,
        location=args.location,
        date_str=resolved_date,
        models=models,
        methods=methods,
        output_dir=output_dir,
        plotly_js=args.plotly_js,
        umap_n_neighbors=args.umap_n_neighbors,
        umap_min_dist=args.umap_min_dist,
        min_cluster_size=args.min_cluster_size,
        make_zip=not args.no_zip,
        clean_output=args.clean_output,
    )
    print(f"\nDone: {result}")


if __name__ == "__main__":
    main()
