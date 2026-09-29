from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from ada.atlas.data.manifests import load_manifest_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an interactive 2D CLS atlas viewer from cached embeddings.")
    parser.add_argument("--query-cache", required=True, type=Path)
    parser.add_argument("--metrics-csv", required=True, type=Path)
    parser.add_argument("--prediction-csv", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-points", default=8000, type=int)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--title", default="CLS Atlas")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    points = build_points(
        query_cache=args.query_cache,
        metrics_csv=args.metrics_csv,
        prediction_csvs=args.prediction_csv,
        max_points=int(args.max_points),
        seed=int(args.seed),
    )
    points_csv = output / "cls_space_points.csv"
    html_path = output / "cls_space.html"
    _write_csv(points_csv, points)
    html_path.write_text(_render_html(points, title=str(args.title)))
    summary = {
        "query_cache": str(args.query_cache),
        "metrics_csv": str(args.metrics_csv),
        "prediction_csvs": [str(path) for path in args.prediction_csv],
        "points_csv": str(points_csv),
        "html": str(html_path),
        "n_points": len(points),
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True))
    print(json.dumps({"html": str(html_path), "points_csv": str(points_csv)}, indent=2, sort_keys=True))


def build_points(
    *,
    query_cache: Path,
    metrics_csv: Path,
    prediction_csvs: list[Path],
    max_points: int,
    seed: int,
) -> list[dict[str, object]]:
    import numpy as np

    rows = load_manifest_csv(Path(query_cache) / "manifest.csv")
    embeddings = np.load(Path(query_cache) / "embeddings.npy", mmap_mode="r").astype("float32", copy=False)
    if embeddings.shape[0] != len(rows):
        raise ValueError("query embeddings and manifest row counts differ")

    n = int(embeddings.shape[0])
    rng = np.random.default_rng(int(seed))
    if max_points > 0 and n > int(max_points):
        selected = np.sort(rng.choice(n, size=int(max_points), replace=False))
    else:
        selected = np.arange(n)

    x = np.asarray(embeddings[selected], dtype=np.float32)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-12)
    coords = _pca2(x)

    metrics = _read_by_sample(metrics_csv)
    predictions = _read_prediction_columns(prediction_csvs)
    points: list[dict[str, object]] = []
    for coord_idx, source_idx in enumerate(selected.tolist()):
        row = rows[int(source_idx)]
        sample_id = row.sample_id
        out: dict[str, object] = {
            "sample_id": sample_id,
            "x": float(coords[coord_idx, 0]),
            "y": float(coords[coord_idx, 1]),
            "class_id": int(row.class_id),
            "class_name": row.class_name,
            "relative_path": row.relative_path,
        }
        out.update(metrics.get(sample_id, {}))
        out.update(predictions.get(sample_id, {}))
        points.append(out)
    return points


def _pca2(x):
    import numpy as np

    if x.shape[0] < 2:
        return np.zeros((x.shape[0], 2), dtype=np.float32)
    centered = x - x.mean(axis=0, keepdims=True)
    _u, _s, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:2].T
    scale = np.maximum(coords.std(axis=0, keepdims=True), 1.0e-12)
    return (coords / scale).astype(np.float32)


def _read_by_sample(path: Path) -> dict[str, dict[str, object]]:
    with Path(path).open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {str(row["sample_id"]): dict(row) for row in reader}


def _read_prediction_columns(paths: list[Path]) -> dict[str, dict[str, object]]:
    out: dict[str, dict[str, object]] = {}
    for path in paths:
        with Path(path).open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sample_id = str(row["sample_id"])
                model_id = str(row["model_id"]).replace("/", "_").replace(" ", "_")
                target = out.setdefault(sample_id, {})
                target[f"{model_id}_correct"] = row.get("correct", "")
                target[f"{model_id}_confidence"] = row.get("max_probability_calibrated") or row.get("max_probability_raw", "")
                target[f"{model_id}_logit_margin"] = row.get("logit_margin", "")
                target[f"{model_id}_predicted_label"] = row.get("predicted_label", "")
    return out


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames: list[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _render_html(points: list[dict[str, object]], *, title: str) -> str:
    numeric_fields = _numeric_fields(points)
    default_color = "class_support_multiscale_percentile" if "class_support_multiscale_percentile" in numeric_fields else "class_id"
    payload = json.dumps(points, separators=(",", ":"))
    fields = json.dumps(numeric_fields)
    escaped_title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escaped_title}</title>
  <style>
    body {{ margin: 0; font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #202124; background: #f7f7f2; }}
    header {{ display: flex; align-items: center; justify-content: space-between; gap: 16px; padding: 14px 18px; border-bottom: 1px solid #d8d7cf; background: #ffffff; }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 700; letter-spacing: 0; }}
    main {{ display: grid; grid-template-columns: minmax(0, 1fr) 340px; min-height: calc(100vh - 57px); }}
    canvas {{ width: 100%; height: calc(100vh - 57px); display: block; background: #ffffff; }}
    aside {{ border-left: 1px solid #d8d7cf; padding: 14px; background: #fbfbf7; overflow: auto; }}
    label {{ display: block; font-size: 12px; font-weight: 700; color: #54524b; margin-bottom: 6px; }}
    select, input {{ width: 100%; box-sizing: border-box; border: 1px solid #b9b7aa; border-radius: 6px; padding: 8px 9px; background: #ffffff; color: #202124; }}
    .control {{ margin-bottom: 14px; }}
    .stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; margin: 14px 0; }}
    .stat {{ border: 1px solid #d8d7cf; border-radius: 6px; padding: 8px; background: #ffffff; }}
    .stat b {{ display: block; font-size: 11px; color: #69665d; }}
    .stat span {{ font-size: 16px; font-weight: 700; }}
    pre {{ white-space: pre-wrap; overflow-wrap: anywhere; border: 1px solid #d8d7cf; border-radius: 6px; padding: 10px; background: #ffffff; font-size: 12px; line-height: 1.35; }}
    @media (max-width: 900px) {{ main {{ grid-template-columns: 1fr; }} canvas {{ height: 62vh; }} aside {{ border-left: 0; border-top: 1px solid #d8d7cf; }} }}
  </style>
</head>
<body>
  <header>
    <h1>{escaped_title}</h1>
    <div style="font-size:12px;color:#69665d;">PCA view of cached CLS tokens</div>
  </header>
  <main>
    <canvas id="plot"></canvas>
    <aside>
      <div class="control">
        <label for="colorField">Color</label>
        <select id="colorField"></select>
      </div>
      <div class="control">
        <label for="search">Search sample/class</label>
        <input id="search" type="search" placeholder="sample id or class name">
      </div>
      <div class="stats">
        <div class="stat"><b>points</b><span id="pointCount"></span></div>
        <div class="stat"><b>visible</b><span id="visibleCount"></span></div>
      </div>
      <pre id="detail">Hover a point.</pre>
    </aside>
  </main>
  <script>
    const points = {payload};
    const numericFields = {fields};
    const defaultColor = {json.dumps(default_color)};
    const canvas = document.getElementById('plot');
    const ctx = canvas.getContext('2d');
    const colorField = document.getElementById('colorField');
    const search = document.getElementById('search');
    const detail = document.getElementById('detail');
    document.getElementById('pointCount').textContent = points.length;
    for (const field of numericFields) {{
      const option = document.createElement('option');
      option.value = field;
      option.textContent = field;
      if (field === defaultColor) option.selected = true;
      colorField.appendChild(option);
    }}
    let bounds = null;
    function resize() {{
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      canvas.width = Math.max(320, Math.floor(rect.width * ratio));
      canvas.height = Math.max(320, Math.floor(rect.height * ratio));
      draw();
    }}
    function visiblePoints() {{
      const q = search.value.trim().toLowerCase();
      if (!q) return points;
      return points.filter(p => String(p.sample_id).toLowerCase().includes(q) || String(p.class_name).toLowerCase().includes(q));
    }}
    function computeBounds(ps) {{
      const xs = ps.map(p => Number(p.x));
      const ys = ps.map(p => Number(p.y));
      return {{ minX: Math.min(...xs), maxX: Math.max(...xs), minY: Math.min(...ys), maxY: Math.max(...ys) }};
    }}
    function project(p) {{
      const pad = 30 * (window.devicePixelRatio || 1);
      const w = canvas.width - 2 * pad;
      const h = canvas.height - 2 * pad;
      const x = pad + (Number(p.x) - bounds.minX) / Math.max(bounds.maxX - bounds.minX, 1e-9) * w;
      const y = pad + (1 - (Number(p.y) - bounds.minY) / Math.max(bounds.maxY - bounds.minY, 1e-9)) * h;
      return [x, y];
    }}
    function colorFor(value, minV, maxV) {{
      const v = Number(value);
      if (!Number.isFinite(v)) return '#c8c7bd';
      const t = Math.max(0, Math.min(1, (v - minV) / Math.max(maxV - minV, 1e-12)));
      const r = Math.round(44 + 204 * t);
      const g = Math.round(123 + 70 * (1 - Math.abs(t - 0.45)));
      const b = Math.round(182 - 150 * t);
      return `rgb(${{r}},${{g}},${{b}})`;
    }}
    function draw() {{
      const ps = visiblePoints();
      document.getElementById('visibleCount').textContent = ps.length;
      bounds = computeBounds(points);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const field = colorField.value;
      const values = ps.map(p => Number(p[field])).filter(Number.isFinite);
      const minV = values.length ? Math.min(...values) : 0;
      const maxV = values.length ? Math.max(...values) : 1;
      for (const p of ps) {{
        const [x, y] = project(p);
        ctx.beginPath();
        ctx.arc(x, y, 3.2 * (window.devicePixelRatio || 1), 0, Math.PI * 2);
        ctx.fillStyle = colorFor(p[field], minV, maxV);
        ctx.globalAlpha = 0.78;
        ctx.fill();
      }}
      ctx.globalAlpha = 1;
    }}
    function nearestPoint(event) {{
      const rect = canvas.getBoundingClientRect();
      const ratio = window.devicePixelRatio || 1;
      const mx = (event.clientX - rect.left) * ratio;
      const my = (event.clientY - rect.top) * ratio;
      let best = null;
      let bestD = Infinity;
      for (const p of visiblePoints()) {{
        const [x, y] = project(p);
        const d = (x - mx) ** 2 + (y - my) ** 2;
        if (d < bestD) {{ bestD = d; best = p; }}
      }}
      return bestD < 400 * ratio * ratio ? best : null;
    }}
    canvas.addEventListener('mousemove', event => {{
      const p = nearestPoint(event);
      if (!p) return;
      const keys = ['sample_id','class_name','class_support_multiscale_percentile','class_margin_nearest','trust_ratio_nearest','local_label_entropy_norm_k10'];
      for (const key of Object.keys(p)) {{
        if (key.endsWith('_correct') || key.endsWith('_confidence') || key.endsWith('_logit_margin')) keys.push(key);
      }}
      detail.textContent = keys.filter(k => k in p).map(k => `${{k}}: ${{p[k]}}`).join('\\n');
    }});
    colorField.addEventListener('change', draw);
    search.addEventListener('input', draw);
    window.addEventListener('resize', resize);
    resize();
  </script>
</body>
</html>
"""


def _numeric_fields(points: list[dict[str, object]]) -> list[str]:
    fields = set()
    for row in points:
        for key, value in row.items():
            try:
                float(value)
            except (TypeError, ValueError):
                continue
            fields.add(key)
    preferred = [
        "class_id",
        "class_support_multiscale_percentile",
        "class_support_k5_percentile",
        "class_support_k10_percentile",
        "class_support_k50_percentile",
        "class_margin_nearest",
        "trust_ratio_nearest",
        "local_label_entropy_norm_k10",
        "local_true_label_fraction_k10",
    ]
    return [field for field in preferred if field in fields] + sorted(fields - set(preferred))


if __name__ == "__main__":
    main()
