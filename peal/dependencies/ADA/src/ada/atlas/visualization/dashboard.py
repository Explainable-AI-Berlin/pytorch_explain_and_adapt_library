from __future__ import annotations

import csv
import html
import json
from pathlib import Path


SCORE_COLUMNS = [
    "support_k5_kth_distance",
    "support_k10_kth_distance",
    "support_k50_kth_distance",
    "class_support_k5_kth_distance",
    "class_support_k10_kth_distance",
    "class_support_k50_kth_distance",
]


def build_e0_dashboard(
    *,
    metrics_json: str | Path,
    joined_csv: str | Path,
    manifest_csv: str | Path | None,
    cache_metadata_json: str | Path | None,
    output_html: str | Path,
) -> Path:
    metrics_path = Path(metrics_json)
    joined_path = Path(joined_csv)
    output_path = Path(output_html)
    metrics = json.loads(metrics_path.read_text())
    sample_info = _sample_info(manifest_csv, cache_metadata_json)
    rows = _dashboard_rows(joined_path, sample_info)
    data = {
        "title": "ADA E0 ImageNet-100 CLS Atlas",
        "metrics": metrics,
        "rows": rows,
        "scoreColumns": SCORE_COLUMNS,
        "sources": {
            "metrics_json": str(metrics_path),
            "joined_csv": str(joined_path),
            "manifest_csv": None if manifest_csv is None else str(manifest_csv),
            "cache_metadata_json": None if cache_metadata_json is None else str(cache_metadata_json),
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    output_path.write_text(_html_template().replace("__DASHBOARD_DATA__", payload))
    return output_path


def _sample_info(
    manifest_csv: str | Path | None,
    cache_metadata_json: str | Path | None,
) -> dict[str, dict[str, str]]:
    if manifest_csv is None:
        return {}
    root = ""
    if cache_metadata_json is not None:
        metadata = json.loads(Path(cache_metadata_json).read_text())
        root = str(metadata.get("root", ""))
    out: dict[str, dict[str, str]] = {}
    with Path(manifest_csv).open("r", newline="") as f:
        for row in csv.DictReader(f):
            image_src = ""
            relative_path = row.get("relative_path", "")
            if root and relative_path:
                image_src = str(Path(root) / relative_path)
            out[row["sample_id"]] = {
                "relative_path": relative_path,
                "class_name": row.get("class_name", ""),
                "image_src": image_src,
            }
    return out


def _dashboard_rows(joined_csv: Path, sample_info: dict[str, dict[str, str]]) -> list[dict]:
    rows = []
    with joined_csv.open("r", newline="") as f:
        for row in csv.DictReader(f):
            info = sample_info.get(row["sample_id"], {})
            confidence = _float(row.get("max_probability_calibrated") or row.get("max_probability_raw"))
            out = {
                "sample_id": row["sample_id"],
                "model_id": row["model_id"],
                "true_label": _int(row["true_label"]),
                "predicted_label": _int(row["predicted_label"]),
                "correct": _int(row["correct"]),
                "confidence": confidence,
                "logit_margin": _float(row["logit_margin"]),
                "entropy": _float(row["entropy_raw"]),
                "nll": _float(row["nll"]),
                "relative_path": info.get("relative_path", ""),
                "class_name": info.get("class_name", ""),
                "image_src": info.get("image_src", ""),
            }
            for column in SCORE_COLUMNS:
                out[column] = _float(row[column])
            rows.append(out)
    return rows


def _float(value: str | None) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def _int(value: str | int) -> int:
    return int(value)


def _html_template() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ADA E0 Atlas Dashboard</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --panel-2: #eef3f2;
      --text: #16201f;
      --muted: #5c6a68;
      --border: #d7dfdd;
      --blue: #246bfe;
      --green: #16835f;
      --amber: #b06417;
      --red: #ba3242;
      --ink: #0f1716;
      --shadow: 0 8px 24px rgba(20, 31, 29, 0.08);
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    .app { min-height: 100vh; display: grid; grid-template-columns: 280px 1fr; }
    aside {
      position: sticky;
      top: 0;
      height: 100vh;
      overflow: auto;
      padding: 18px;
      border-right: 1px solid var(--border);
      background: #fbfcfc;
    }
    main { min-width: 0; padding: 18px 22px 36px; }
    h1 { margin: 0 0 4px; font-size: 20px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 15px; letter-spacing: 0; }
    h3 { margin: 0 0 8px; font-size: 13px; letter-spacing: 0; }
    p { margin: 0; }
    .subtle { color: var(--muted); font-size: 12px; }
    .control { margin-top: 16px; }
    label { display: block; color: var(--muted); font-size: 12px; margin: 0 0 6px; }
    select, input {
      width: 100%;
      border: 1px solid var(--border);
      background: white;
      color: var(--text);
      border-radius: 6px;
      min-height: 34px;
      padding: 6px 8px;
      font: inherit;
    }
    input[type="checkbox"] { width: auto; min-height: 0; }
    .check {
      display: flex;
      align-items: center;
      gap: 8px;
      color: var(--text);
      margin-top: 10px;
      font-size: 13px;
    }
    .topbar {
      display: flex;
      align-items: flex-end;
      justify-content: space-between;
      gap: 18px;
      margin-bottom: 16px;
    }
    .source { text-align: right; max-width: 55ch; }
    .grid { display: grid; gap: 14px; }
    .metrics { grid-template-columns: repeat(6, minmax(120px, 1fr)); }
    .metric {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 12px;
      box-shadow: var(--shadow);
      min-height: 82px;
    }
    .metric .label { color: var(--muted); font-size: 12px; margin-bottom: 7px; }
    .metric .value { font-size: 24px; font-weight: 700; letter-spacing: 0; }
    .metric .note { color: var(--muted); font-size: 11px; margin-top: 3px; }
    .two { grid-template-columns: minmax(0, 1.35fr) minmax(320px, 0.85fr); margin-top: 14px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 8px;
      padding: 14px;
      box-shadow: var(--shadow);
      min-width: 0;
    }
    .chart-wrap { height: 330px; }
    svg { width: 100%; height: 100%; display: block; }
    .legend { display: flex; gap: 14px; flex-wrap: wrap; color: var(--muted); font-size: 12px; margin-top: 10px; }
    .swatch { width: 10px; height: 10px; display: inline-block; border-radius: 2px; margin-right: 5px; }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    th, td {
      text-align: left;
      padding: 7px 6px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }
    th { color: var(--muted); font-weight: 600; }
    .samples {
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }
    .sample {
      display: grid;
      grid-template-columns: 86px 1fr;
      gap: 10px;
      align-items: start;
      border: 1px solid var(--border);
      border-radius: 8px;
      background: var(--panel);
      padding: 8px;
      min-width: 0;
    }
    .sample img {
      width: 86px;
      height: 86px;
      object-fit: cover;
      border-radius: 6px;
      background: var(--panel-2);
      border: 1px solid var(--border);
    }
    .sample .name {
      font-weight: 650;
      overflow-wrap: anywhere;
      margin-bottom: 4px;
    }
    .sample .meta { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .pill {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      margin-top: 6px;
      border: 1px solid var(--border);
      background: #f8faf9;
    }
    .pill.err { color: var(--red); border-color: #e8bcc2; background: #fff5f6; }
    .pill.ok { color: var(--green); border-color: #b8ded1; background: #f2fbf8; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin: 8px 0 4px; }
    .toolbar > div { min-width: 180px; }
    .empty { color: var(--muted); padding: 20px 0; }
    .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; }
    @media (max-width: 1100px) {
      .app { grid-template-columns: 1fr; }
      aside { position: relative; height: auto; border-right: 0; border-bottom: 1px solid var(--border); }
      .metrics { grid-template-columns: repeat(3, minmax(120px, 1fr)); }
      .two { grid-template-columns: 1fr; }
      .source { text-align: left; }
      .topbar { align-items: flex-start; flex-direction: column; }
    }
    @media (max-width: 640px) {
      main { padding: 14px; }
      .metrics { grid-template-columns: repeat(2, minmax(0, 1fr)); }
      .sample { grid-template-columns: 72px 1fr; }
      .sample img { width: 72px; height: 72px; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <h1>ADA E0 Atlas</h1>
      <p class="subtle">ImageNet-100, DINOv2 CLS support, prediction competence map.</p>
      <div class="control">
        <label for="modelSelect">Model</label>
        <select id="modelSelect"></select>
      </div>
      <div class="control">
        <label for="scoreSelect">Support score</label>
        <select id="scoreSelect"></select>
      </div>
      <div class="control">
        <label for="sortSelect">Sample sort</label>
        <select id="sortSelect">
          <option value="weakest">weakest support first</option>
          <option value="confidence">highest confidence first</option>
          <option value="margin">lowest logit margin first</option>
          <option value="nll">highest NLL first</option>
        </select>
      </div>
      <div class="control">
        <label for="limitInput">Sample limit</label>
        <input id="limitInput" type="number" min="6" max="120" step="6" value="36">
      </div>
      <label class="check"><input id="errorsOnly" type="checkbox"> errors only</label>
      <label class="check"><input id="highConfidenceOnly" type="checkbox"> confidence >= 0.9</label>
      <label class="check"><input id="disagreeOnly" type="checkbox"> prediction differs from true label</label>
      <div class="control">
        <label for="searchInput">Search sample/class/path</label>
        <input id="searchInput" type="search" placeholder="sample id, synset, path">
      </div>
      <p class="subtle" style="margin-top: 18px">Larger support distance means weaker local support.</p>
    </aside>
    <main>
      <div class="topbar">
        <div>
          <h1 id="pageTitle">ADA E0 Atlas Dashboard</h1>
          <p class="subtle" id="summaryLine"></p>
        </div>
        <div class="source subtle">
          <span id="sourceLine"></span>
        </div>
      </div>
      <section class="grid metrics" id="metricGrid"></section>
      <section class="grid two">
        <div class="panel">
          <h2>Error And Confidence By Support Decile</h2>
          <div class="chart-wrap"><svg id="decileChart" role="img" aria-label="Error and confidence by support decile"></svg></div>
          <div class="legend">
            <span><span class="swatch" style="background: var(--red)"></span>error rate</span>
            <span><span class="swatch" style="background: var(--blue)"></span>mean confidence</span>
            <span><span class="swatch" style="background: var(--amber)"></span>calibration gap</span>
          </div>
        </div>
        <div class="panel">
          <h2>Error Detection Scores</h2>
          <div class="chart-wrap"><svg id="scoreChart" role="img" aria-label="Error detection AUROC and AUPRC"></svg></div>
        </div>
      </section>
      <section class="grid two">
        <div class="panel">
          <h2>Cross-Fit Risk Models</h2>
          <div id="crossfitTable"></div>
        </div>
        <div class="panel">
          <h2>Support Decile Table</h2>
          <div id="decileTable"></div>
        </div>
      </section>
      <section class="panel" style="margin-top: 14px">
        <h2>Sample Explorer</h2>
        <p class="subtle" id="sampleSummary"></p>
        <div class="samples" id="samples"></div>
      </section>
    </main>
  </div>
  <script id="dashboard-data" type="application/json">__DASHBOARD_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById('dashboard-data').textContent);
    const modelSelect = document.getElementById('modelSelect');
    const scoreSelect = document.getElementById('scoreSelect');
    const sortSelect = document.getElementById('sortSelect');
    const limitInput = document.getElementById('limitInput');
    const errorsOnly = document.getElementById('errorsOnly');
    const highConfidenceOnly = document.getElementById('highConfidenceOnly');
    const disagreeOnly = document.getElementById('disagreeOnly');
    const searchInput = document.getElementById('searchInput');
    const modelNames = Object.keys(data.metrics.models);
    const scoreLabels = {
      support_k5_kth_distance: 'global k5 distance',
      support_k10_kth_distance: 'global k10 distance',
      support_k50_kth_distance: 'global k50 distance',
      class_support_k5_kth_distance: 'class k5 distance',
      class_support_k10_kth_distance: 'class k10 distance',
      class_support_k50_kth_distance: 'class k50 distance'
    };

    for (const model of modelNames) {
      const option = document.createElement('option');
      option.value = model;
      option.textContent = prettyModel(model);
      modelSelect.appendChild(option);
    }
    modelSelect.value = modelNames.includes('dino_linear_probe') ? 'dino_linear_probe' : modelNames[0];
    for (const score of data.scoreColumns) {
      const option = document.createElement('option');
      option.value = score;
      option.textContent = scoreLabels[score] || score;
      scoreSelect.appendChild(option);
    }
    scoreSelect.value = 'class_support_k50_kth_distance';
    for (const el of [modelSelect, scoreSelect, sortSelect, limitInput, errorsOnly, highConfidenceOnly, disagreeOnly, searchInput]) {
      el.addEventListener('input', render);
      el.addEventListener('change', render);
    }

    function prettyModel(model) {
      return model
        .replace('resnet18_imagenet1k_restricted100', 'ResNet18 IN1K restricted-100')
        .replace('dino_linear_probe', 'DINO linear probe')
        .replace('dino_knn_k', 'DINO kNN k=');
    }

    function fmt(value, digits = 3) {
      if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
      return Number(value).toFixed(digits);
    }
    function pct(value, digits = 1) { return `${(100 * Number(value)).toFixed(digits)}%`; }
    function currentRows() {
      const model = modelSelect.value;
      const score = scoreSelect.value;
      const query = searchInput.value.trim().toLowerCase();
      let rows = data.rows.filter(row => row.model_id === model);
      if (errorsOnly.checked) rows = rows.filter(row => row.correct === 0);
      if (highConfidenceOnly.checked) rows = rows.filter(row => row.confidence >= 0.9);
      if (disagreeOnly.checked) rows = rows.filter(row => row.predicted_label !== row.true_label);
      if (query) {
        rows = rows.filter(row =>
          row.sample_id.toLowerCase().includes(query) ||
          String(row.true_label).includes(query) ||
          String(row.predicted_label).includes(query) ||
          row.class_name.toLowerCase().includes(query) ||
          row.relative_path.toLowerCase().includes(query)
        );
      }
      const sortMode = sortSelect.value;
      rows = rows.slice();
      if (sortMode === 'confidence') rows.sort((a, b) => b.confidence - a.confidence);
      else if (sortMode === 'margin') rows.sort((a, b) => a.logit_margin - b.logit_margin);
      else if (sortMode === 'nll') rows.sort((a, b) => b.nll - a.nll);
      else rows.sort((a, b) => b[score] - a[score]);
      return rows;
    }
    function renderMetrics() {
      const model = modelSelect.value;
      const score = scoreSelect.value;
      const report = data.metrics.models[model];
      const scoreReport = report.scores[score] || {};
      const cf = report.crossfit_logistic || {};
      const cfModels = cf.models || {};
      const confidence = cfModels.confidence || {};
      const classModel = cfModels.confidence_plus_global_and_class_support || {};
      const delta = (classModel.auprc ?? 0) - (confidence.auprc ?? 0);
      document.getElementById('summaryLine').textContent =
        `${prettyModel(model)} · ${scoreLabels[score]} · ${report.rows.toLocaleString()} rows`;
      document.getElementById('sourceLine').textContent =
        `${data.sources.metrics_json}`;
      const metrics = [
        ['Accuracy', pct(report.accuracy, 2), `${report.errors} errors`],
        ['Mean confidence', pct(report.mean_confidence, 2), 'selected model'],
        ['Support AUROC', fmt(scoreReport.auroc, 4), scoreLabels[score]],
        ['Support AUPRC', fmt(scoreReport.auprc, 4), 'errors as positives'],
        ['OOF +class AUPRC', fmt(classModel.auprc, 4), `delta ${fmt(delta, 4)}`],
        ['Rows', report.rows.toLocaleString(), 'validation samples']
      ];
      document.getElementById('metricGrid').innerHTML = metrics.map(([label, value, note]) => `
        <div class="metric"><div class="label">${escapeHtml(label)}</div><div class="value">${escapeHtml(value)}</div><div class="note">${escapeHtml(note)}</div></div>
      `).join('');
    }
    function renderDecileChart() {
      const model = modelSelect.value;
      const score = scoreSelect.value;
      const deciles = data.metrics.models[model].deciles[score] || [];
      const svg = document.getElementById('decileChart');
      const w = svg.clientWidth || 700;
      const h = svg.clientHeight || 320;
      const pad = { left: 42, right: 18, top: 16, bottom: 34 };
      const innerW = Math.max(10, w - pad.left - pad.right);
      const innerH = Math.max(10, h - pad.top - pad.bottom);
      const maxY = Math.max(0.05, ...deciles.flatMap(d => [d.error_rate, d.confidence_mean, Math.abs(d.calibration_gap)]));
      const x = i => pad.left + (i + 0.5) * innerW / deciles.length;
      const y = v => pad.top + innerH - (v / maxY) * innerH;
      const barW = innerW / deciles.length * 0.55;
      const bars = deciles.map((d, i) => {
        const bh = pad.top + innerH - y(d.error_rate);
        return `<rect x="${x(i) - barW / 2}" y="${y(d.error_rate)}" width="${barW}" height="${bh}" rx="3" fill="var(--red)" opacity="0.82"></rect>`;
      }).join('');
      const line = (key, color) => deciles.map((d, i) => `${i === 0 ? 'M' : 'L'} ${x(i)} ${y(Math.abs(d[key]))}`).join(' ');
      const labels = deciles.map((d, i) => `<text x="${x(i)}" y="${h - 9}" text-anchor="middle" font-size="11" fill="var(--muted)">${i + 1}</text>`).join('');
      const grid = [0, 0.25, 0.5, 0.75, 1].map(t => {
        const yy = pad.top + innerH - t * innerH;
        const value = maxY * t;
        return `<line x1="${pad.left}" x2="${w - pad.right}" y1="${yy}" y2="${yy}" stroke="var(--border)" stroke-width="1"></line><text x="8" y="${yy + 4}" font-size="11" fill="var(--muted)">${pct(value, 0)}</text>`;
      }).join('');
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      svg.innerHTML = `${grid}${bars}
        <path d="${line('confidence_mean', 'var(--blue)')}" fill="none" stroke="var(--blue)" stroke-width="2.5"></path>
        <path d="${line('calibration_gap', 'var(--amber)')}" fill="none" stroke="var(--amber)" stroke-width="2.5" stroke-dasharray="5 4"></path>
        ${labels}
        <text x="${pad.left}" y="12" font-size="11" fill="var(--muted)">weak support</text>
        <text x="${w - pad.right}" y="12" text-anchor="end" font-size="11" fill="var(--muted)">strong support</text>`;
    }
    function renderScoreChart() {
      const model = modelSelect.value;
      const report = data.metrics.models[model];
      const items = [
        ['confidence', report.scores.confidence_risk],
        ['global k50', report.scores.support_k50_kth_distance],
        ['class k50', report.scores.class_support_k50_kth_distance],
        ['rank avg', report.scores.rankavg_confidence_support_k50_kth_distance]
      ].filter(item => item[1]);
      const svg = document.getElementById('scoreChart');
      const w = svg.clientWidth || 420;
      const h = svg.clientHeight || 320;
      const pad = { left: 48, right: 18, top: 22, bottom: 54 };
      const innerW = w - pad.left - pad.right;
      const innerH = h - pad.top - pad.bottom;
      const groupW = innerW / items.length;
      const barW = Math.min(34, groupW * 0.28);
      const y = v => pad.top + innerH - v * innerH;
      const bars = items.map(([label, score], i) => {
        const cx = pad.left + i * groupW + groupW / 2;
        return `
          <rect x="${cx - barW - 2}" y="${y(score.auroc)}" width="${barW}" height="${pad.top + innerH - y(score.auroc)}" rx="3" fill="var(--green)"></rect>
          <rect x="${cx + 2}" y="${y(score.auprc)}" width="${barW}" height="${pad.top + innerH - y(score.auprc)}" rx="3" fill="var(--blue)"></rect>
          <text x="${cx}" y="${h - 31}" text-anchor="middle" font-size="11" fill="var(--muted)">${escapeHtml(label)}</text>
        `;
      }).join('');
      const grid = [0, 0.25, 0.5, 0.75, 1].map(v => {
        const yy = y(v);
        return `<line x1="${pad.left}" x2="${w - pad.right}" y1="${yy}" y2="${yy}" stroke="var(--border)" stroke-width="1"></line><text x="10" y="${yy + 4}" font-size="11" fill="var(--muted)">${v.toFixed(2)}</text>`;
      }).join('');
      svg.setAttribute('viewBox', `0 0 ${w} ${h}`);
      svg.innerHTML = `${grid}${bars}
        <text x="${pad.left}" y="${h - 10}" font-size="11" fill="var(--green)">AUROC</text>
        <text x="${pad.left + 58}" y="${h - 10}" font-size="11" fill="var(--blue)">AUPRC</text>`;
    }
    function renderTables() {
      const model = modelSelect.value;
      const score = scoreSelect.value;
      const report = data.metrics.models[model];
      const cf = report.crossfit_logistic || {};
      const cfRows = Object.entries(cf.models || {}).map(([name, row]) => `
        <tr><td>${escapeHtml(name.replaceAll('_', ' '))}</td><td>${fmt(row.auroc, 4)}</td><td>${fmt(row.auprc, 4)}</td><td>${fmt(row.brier, 4)}</td><td>${fmt(row.log_loss, 4)}</td></tr>
      `).join('');
      document.getElementById('crossfitTable').innerHTML = cfRows ? `
        <table><thead><tr><th>Risk model</th><th>AUROC</th><th>AUPRC</th><th>Brier</th><th>Log loss</th></tr></thead><tbody>${cfRows}</tbody></table>
        <p class="subtle" style="margin-top:8px">Backend: ${escapeHtml(cf.backend || 'sklearn')}</p>
      ` : '<p class="empty">No cross-fit risk model metrics available.</p>';
      const dRows = (report.deciles[score] || []).map(row => `
        <tr><td>${row.bin}</td><td>${fmt(row.score_mean, 3)}</td><td>${pct(row.error_rate, 1)}</td><td>${pct(row.confidence_mean, 1)}</td><td>${pct(row.calibration_gap, 1)}</td></tr>
      `).join('');
      document.getElementById('decileTable').innerHTML = `
        <table><thead><tr><th>Decile</th><th>Mean dist.</th><th>Error</th><th>Conf.</th><th>Gap</th></tr></thead><tbody>${dRows}</tbody></table>
      `;
    }
    function renderSamples() {
      const rows = currentRows();
      const limit = Math.max(6, Math.min(120, Number(limitInput.value || 36)));
      const visible = rows.slice(0, limit);
      document.getElementById('sampleSummary').textContent = `${visible.length} shown from ${rows.length.toLocaleString()} matching rows`;
      const htmlRows = visible.map(row => {
        const img = row.image_src ? `<img src="${escapeAttr(row.image_src)}" alt="">` : '<div></div>';
        const state = row.correct ? '<span class="pill ok">correct</span>' : '<span class="pill err">error</span>';
        return `<article class="sample">
          ${img}
          <div>
            <div class="name">${escapeHtml(row.class_name || row.relative_path || row.sample_id)}</div>
            <div class="meta mono">${escapeHtml(row.sample_id)}</div>
            <div class="meta">true ${row.true_label} · pred ${row.predicted_label}</div>
            <div class="meta">conf ${pct(row.confidence, 1)} · dist ${fmt(row[scoreSelect.value], 3)}</div>
            ${state}
          </div>
        </article>`;
      }).join('');
      document.getElementById('samples').innerHTML = htmlRows || '<p class="empty">No samples match the current filters.</p>';
    }
    function render() {
      renderMetrics();
      renderDecileChart();
      renderScoreChart();
      renderTables();
      renderSamples();
    }
    function escapeHtml(value) {
      return String(value).replace(/[&<>"']/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[ch]));
    }
    function escapeAttr(value) { return escapeHtml(value); }
    window.addEventListener('resize', () => {
      renderDecileChart();
      renderScoreChart();
    });
    render();
  </script>
</body>
</html>
"""
