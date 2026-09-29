from __future__ import annotations

import csv
import html
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


DEFAULT_SUPPORT_COLUMN = "class_support_k50_kth_distance"


def build_failure_atlas(
    *,
    joined_csv: str | Path,
    output_dir: str | Path,
    manifest_csv: str | Path | None = None,
    cache_metadata_json: str | Path | None = None,
    task_model: str = "dino_linear_probe",
    independent_model: str = "resnet18_imagenet1k_restricted100",
    purity_model: str = "dino_knn_k10",
    support_column: str = DEFAULT_SUPPORT_COLUMN,
    weak_support_fraction: float = 0.10,
    max_exemplars: int = 80,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_info = _sample_info(manifest_csv, cache_metadata_json)
    samples = _pivot_joined_rows(Path(joined_csv), sample_info)
    _annotate_support(samples, support_column=support_column, weak_fraction=weak_support_fraction)
    _annotate_signatures(
        samples,
        task_model=task_model,
        independent_model=independent_model,
        purity_model=purity_model,
        support_column=support_column,
    )
    matrix_rows = [_matrix_row(sample) for sample in samples]
    signature_rows = _signature_summary(samples, support_column=support_column)
    confusion_rows = _confusion_summary(samples, support_column=support_column)
    support_rows = _support_by_signature(samples, support_column=support_column)
    candidate_rows = _actionable_candidates(
        samples,
        task_model=task_model,
        independent_model=independent_model,
        purity_model=purity_model,
        support_column=support_column,
    )
    matched_rows = _matched_controls(
        samples,
        candidates=candidate_rows,
        task_model=task_model,
        independent_model=independent_model,
        support_column=support_column,
    )

    _write_csv(output / "failure_matrix.csv", matrix_rows)
    _write_csv(output / "signature_summary.csv", signature_rows)
    _write_csv(output / "confusion_pairs.csv", confusion_rows)
    _write_csv(output / "support_by_signature.csv", support_rows)
    _write_csv(output / "actionable_candidates.csv", candidate_rows)
    _write_csv(output / "matched_controls.csv", matched_rows)
    report = {
        "joined_csv": str(joined_csv),
        "manifest_csv": None if manifest_csv is None else str(manifest_csv),
        "cache_metadata_json": None if cache_metadata_json is None else str(cache_metadata_json),
        "task_model": task_model,
        "independent_model": independent_model,
        "purity_model": purity_model,
        "support_column": support_column,
        "weak_support_fraction": weak_support_fraction,
        "num_samples": len(samples),
        "num_actionable_candidates": len(candidate_rows),
        "signature_summary": signature_rows,
        "top_confusion_pairs": confusion_rows[:50],
    }
    (output / "failure_atlas_summary.json").write_text(json.dumps(report, indent=2, sort_keys=True))
    _write_html(
        output / "failure_atlas.html",
        report=report,
        candidates=candidate_rows[:max_exemplars],
        matched_controls=matched_rows,
        signature_rows=signature_rows,
        confusion_rows=confusion_rows[:30],
    )
    return output / "failure_atlas_summary.json"


def _sample_info(
    manifest_csv: str | Path | None,
    cache_metadata_json: str | Path | None,
) -> dict[str, dict[str, str]]:
    if manifest_csv is None:
        return {}
    root = ""
    if cache_metadata_json is not None:
        root = str(json.loads(Path(cache_metadata_json).read_text()).get("root", ""))
    out: dict[str, dict[str, str]] = {}
    with Path(manifest_csv).open("r", newline="") as f:
        for row in csv.DictReader(f):
            relative_path = row.get("relative_path", "")
            image_src = str(Path(root) / relative_path) if root and relative_path else ""
            out[row["sample_id"]] = {
                "class_name": row.get("class_name", ""),
                "relative_path": relative_path,
                "image_src": image_src,
            }
    return out


def _pivot_joined_rows(path: Path, sample_info: dict[str, dict[str, str]]) -> list[dict]:
    by_sample: dict[str, dict] = {}
    support_columns = [
        "support_k5_kth_distance",
        "support_k10_kth_distance",
        "support_k50_kth_distance",
        "class_support_k5_kth_distance",
        "class_support_k10_kth_distance",
        "class_support_k50_kth_distance",
    ]
    with path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            sample = by_sample.setdefault(
                row["sample_id"],
                {
                    "sample_id": row["sample_id"],
                    "true_label": int(row["true_label"]),
                    "models": {},
                    "support": {},
                    **sample_info.get(row["sample_id"], {}),
                },
            )
            for column in support_columns:
                sample["support"][column] = float(row[column])
            sample["models"][row["model_id"]] = {
                "predicted_label": int(row["predicted_label"]),
                "correct": int(row["correct"]),
                "error": 1 - int(row["correct"]),
                "confidence": _confidence(row),
                "logit_margin": float(row["logit_margin"]),
                "nll": float(row["nll"]),
            }
    return list(by_sample.values())


def _confidence(row: dict[str, str]) -> float:
    calibrated = row.get("max_probability_calibrated", "")
    return float(calibrated if calibrated else row["max_probability_raw"])


def _annotate_support(samples: Sequence[dict], *, support_column: str, weak_fraction: float) -> None:
    ordered = sorted(samples, key=lambda sample: sample["support"][support_column], reverse=True)
    n = len(ordered)
    weak_count = max(1, int(round(n * weak_fraction)))
    for rank, sample in enumerate(ordered):
        sample["support_rank_weak_first"] = rank + 1
        sample["support_percentile_weak"] = 1.0 - (rank / max(n - 1, 1))
        sample["support_decile_weak_first"] = int((rank * 10) // max(n, 1)) + 1
        sample["weak_support"] = int(rank < weak_count)


def _annotate_signatures(
    samples: Sequence[dict],
    *,
    task_model: str,
    independent_model: str,
    purity_model: str,
    support_column: str,
) -> None:
    for sample in samples:
        task = sample["models"].get(task_model, {})
        independent = sample["models"].get(independent_model, {})
        purity = sample["models"].get(purity_model, {})
        task_error = int(task.get("error", 0))
        independent_error = int(independent.get("error", 0))
        purity_correct = int(purity.get("correct", 0))
        weak = int(sample.get("weak_support", 0))
        if task_error and not independent_error and weak:
            signature = "task_wrong_independent_correct_low_support"
            priority = 1
        elif task_error and not independent_error:
            signature = "task_wrong_independent_correct"
            priority = 2
        elif task_error and independent_error and weak:
            signature = "both_wrong_low_support"
            priority = 3
        elif task_error and independent_error:
            signature = "both_wrong"
            priority = 4
        elif not task_error and independent_error and weak:
            signature = "independent_wrong_task_correct_low_support"
            priority = 5
        elif not task_error and independent_error:
            signature = "independent_wrong_task_correct"
            priority = 6
        elif not task_error and not independent_error and weak:
            signature = "both_correct_low_support"
            priority = 7
        else:
            signature = "both_correct_supported"
            priority = 8
        error_vector = "".join(str(sample["models"][model]["error"]) for model in sorted(sample["models"]))
        sample["signature"] = signature
        sample["signature_priority"] = priority
        sample["task_error"] = task_error
        sample["independent_error"] = independent_error
        sample["purity_correct"] = purity_correct
        sample["error_count"] = sum(model["error"] for model in sample["models"].values())
        sample["error_vector"] = error_vector
        sample["actionability_score"] = _actionability_score(
            sample,
            task_model=task_model,
            independent_model=independent_model,
            purity_model=purity_model,
            support_column=support_column,
        )


def _actionability_score(
    sample: dict,
    *,
    task_model: str,
    independent_model: str,
    purity_model: str,
    support_column: str,
) -> float:
    task = sample["models"].get(task_model, {})
    independent = sample["models"].get(independent_model, {})
    purity = sample["models"].get(purity_model, {})
    return (
        2.0 * int(task.get("error", 0))
        + 1.25 * int(not independent.get("error", 1))
        + 0.75 * int(purity.get("correct", 0))
        + 1.50 * float(sample.get("weak_support", 0))
        + 0.75 * float(sample.get("support_percentile_weak", 0.0))
        + 0.25 * float(independent.get("confidence", 0.0))
        + 0.25 * float(task.get("confidence", 0.0))
    )


def _matrix_row(sample: dict) -> dict:
    out = {
        "sample_id": sample["sample_id"],
        "true_label": sample["true_label"],
        "class_name": sample.get("class_name", ""),
        "relative_path": sample.get("relative_path", ""),
        "image_src": sample.get("image_src", ""),
        "signature": sample["signature"],
        "signature_priority": sample["signature_priority"],
        "error_count": sample["error_count"],
        "error_vector": sample["error_vector"],
        "weak_support": sample["weak_support"],
        "support_decile_weak_first": sample["support_decile_weak_first"],
        "support_percentile_weak": sample["support_percentile_weak"],
        "actionability_score": sample["actionability_score"],
    }
    for column, value in sample["support"].items():
        out[column] = value
    for model_id, model in sorted(sample["models"].items()):
        prefix = _safe_name(model_id)
        out[f"{prefix}_pred"] = model["predicted_label"]
        out[f"{prefix}_error"] = model["error"]
        out[f"{prefix}_confidence"] = model["confidence"]
        out[f"{prefix}_margin"] = model["logit_margin"]
    return out


def _signature_summary(samples: Sequence[dict], *, support_column: str) -> list[dict]:
    by_sig: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        by_sig[sample["signature"]].append(sample)
    rows = []
    for signature, group in by_sig.items():
        rows.append(
            {
                "signature": signature,
                "count": len(group),
                "mean_error_count": _mean(sample["error_count"] for sample in group),
                "task_error_rate": _mean(sample["task_error"] for sample in group),
                "independent_error_rate": _mean(sample["independent_error"] for sample in group),
                "purity_correct_rate": _mean(sample["purity_correct"] for sample in group),
                "weak_support_rate": _mean(sample["weak_support"] for sample in group),
                f"mean_{support_column}": _mean(sample["support"][support_column] for sample in group),
                "mean_actionability_score": _mean(sample["actionability_score"] for sample in group),
            }
        )
    rows.sort(key=lambda row: (-row["mean_actionability_score"], -row["count"]))
    return rows


def _confusion_summary(samples: Sequence[dict], *, support_column: str) -> list[dict]:
    counts: dict[tuple[str, int, int], list[dict]] = defaultdict(list)
    for sample in samples:
        for model_id, model in sample["models"].items():
            if model["error"]:
                counts[(model_id, sample["true_label"], model["predicted_label"])].append(sample)
    rows = []
    for (model_id, true_label, predicted_label), group in counts.items():
        rows.append(
            {
                "model_id": model_id,
                "true_label": true_label,
                "predicted_label": predicted_label,
                "count": len(group),
                "weak_support_count": sum(sample["weak_support"] for sample in group),
                f"mean_{support_column}": _mean(sample["support"][support_column] for sample in group),
                "example_sample_ids": ",".join(sample["sample_id"] for sample in group[:5]),
            }
        )
    rows.sort(key=lambda row: (-row["count"], -row["weak_support_count"], row["model_id"]))
    return rows


def _support_by_signature(samples: Sequence[dict], *, support_column: str) -> list[dict]:
    rows = []
    for signature, group in _groups(samples, key="signature").items():
        values = sorted(sample["support"][support_column] for sample in group)
        rows.append(
            {
                "signature": signature,
                "count": len(group),
                "support_min": values[0],
                "support_p25": _quantile(values, 0.25),
                "support_median": _quantile(values, 0.50),
                "support_p75": _quantile(values, 0.75),
                "support_max": values[-1],
            }
        )
    rows.sort(key=lambda row: (-row["support_median"], -row["count"]))
    return rows


def _actionable_candidates(
    samples: Sequence[dict],
    *,
    task_model: str,
    independent_model: str,
    purity_model: str,
    support_column: str,
) -> list[dict]:
    rows = []
    for sample in samples:
        task = sample["models"].get(task_model, {})
        independent = sample["models"].get(independent_model, {})
        purity = sample["models"].get(purity_model, {})
        if not task.get("error", 0):
            continue
        if not sample.get("weak_support", 0):
            continue
        row = {
            "sample_id": sample["sample_id"],
            "true_label": sample["true_label"],
            "class_name": sample.get("class_name", ""),
            "relative_path": sample.get("relative_path", ""),
            "image_src": sample.get("image_src", ""),
            "signature": sample["signature"],
            "actionability_score": sample["actionability_score"],
            support_column: sample["support"][support_column],
            "support_decile_weak_first": sample["support_decile_weak_first"],
            "task_predicted_label": task.get("predicted_label", ""),
            "task_confidence": task.get("confidence", ""),
            "independent_predicted_label": independent.get("predicted_label", ""),
            "independent_correct": int(not independent.get("error", 1)),
            "independent_confidence": independent.get("confidence", ""),
            "purity_model_correct": purity.get("correct", ""),
            "error_count": sample["error_count"],
            "error_vector": sample["error_vector"],
        }
        rows.append(row)
    rows.sort(
        key=lambda row: (
            int(row["independent_correct"]),
            int(row["purity_model_correct"]) if row["purity_model_correct"] != "" else 0,
            float(row["actionability_score"]),
            float(row[support_column]),
        ),
        reverse=True,
    )
    return rows


def _matched_controls(
    samples: Sequence[dict],
    *,
    candidates: Sequence[dict],
    task_model: str,
    independent_model: str,
    support_column: str,
) -> list[dict]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    sample_by_id = {sample["sample_id"]: sample for sample in samples}
    for sample in samples:
        task = sample["models"].get(task_model, {})
        independent = sample["models"].get(independent_model, {})
        if task.get("correct", 0) and independent.get("correct", 0):
            by_class[sample["true_label"]].append(sample)
    rows = []
    for candidate in candidates:
        source = sample_by_id[candidate["sample_id"]]
        controls = sorted(
            by_class.get(source["true_label"], []),
            key=lambda sample: abs(sample["support"][support_column] - source["support"][support_column]),
        )[:3]
        for rank, control in enumerate(controls, start=1):
            rows.append(
                {
                    "candidate_sample_id": source["sample_id"],
                    "control_rank": rank,
                    "control_sample_id": control["sample_id"],
                    "true_label": control["true_label"],
                    "class_name": control.get("class_name", ""),
                    "relative_path": control.get("relative_path", ""),
                    "image_src": control.get("image_src", ""),
                    support_column: control["support"][support_column],
                    "support_delta": abs(control["support"][support_column] - source["support"][support_column]),
                }
            )
    return rows


def _groups(samples: Sequence[dict], *, key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for sample in samples:
        out[str(sample[key])].append(sample)
    return out


def _mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else 0.0


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    pos = q * (len(values) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def _safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in value).strip("_")


def _write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_html(
    path: Path,
    *,
    report: dict,
    candidates: Sequence[dict],
    matched_controls: Sequence[dict],
    signature_rows: Sequence[dict],
    confusion_rows: Sequence[dict],
) -> None:
    controls_by_candidate: dict[str, list[dict]] = defaultdict(list)
    for row in matched_controls:
        controls_by_candidate[row["candidate_sample_id"]].append(row)
    data = {
        "report": report,
        "candidates": list(candidates),
        "controlsByCandidate": controls_by_candidate,
        "signatureRows": list(signature_rows),
        "confusionRows": list(confusion_rows),
    }
    payload = json.dumps(data, separators=(",", ":"), sort_keys=True).replace("</", "<\\/")
    path.write_text(_html_template().replace("__FAILURE_ATLAS_DATA__", payload))


def _html_template() -> str:
    return r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>ADA Failure Atlas</title>
  <style>
    :root {
      --bg: #f5f7f7;
      --panel: #ffffff;
      --text: #17211f;
      --muted: #60706c;
      --border: #d9e1df;
      --red: #b73143;
      --green: #178262;
      --blue: #286df3;
      --amber: #ad6416;
      --shadow: 0 8px 22px rgba(20, 31, 29, 0.08);
    }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--bg); color: var(--text); font: 14px/1.4 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    header { padding: 20px 24px 12px; }
    main { padding: 0 24px 32px; }
    h1 { margin: 0 0 4px; font-size: 24px; letter-spacing: 0; }
    h2 { margin: 0 0 12px; font-size: 16px; letter-spacing: 0; }
    .subtle { color: var(--muted); font-size: 12px; }
    .grid { display: grid; gap: 14px; }
    .metrics { grid-template-columns: repeat(4, minmax(150px, 1fr)); margin-bottom: 14px; }
    .metric, .panel { background: var(--panel); border: 1px solid var(--border); border-radius: 8px; box-shadow: var(--shadow); }
    .metric { padding: 12px; }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { font-size: 24px; font-weight: 700; margin-top: 4px; }
    .two { grid-template-columns: 1fr 1fr; }
    .panel { padding: 14px; min-width: 0; }
    table { width: 100%; border-collapse: collapse; font-size: 12px; }
    th, td { padding: 7px 6px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
    th { color: var(--muted); font-weight: 650; }
    .toolbar { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; margin-bottom: 12px; }
    select, input { min-height: 34px; border: 1px solid var(--border); border-radius: 6px; background: white; padding: 6px 8px; font: inherit; }
    .cards { display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 12px; }
    .card { border: 1px solid var(--border); border-radius: 8px; padding: 10px; background: white; }
    .pair { display: grid; grid-template-columns: 112px 1fr; gap: 10px; }
    img { width: 112px; height: 112px; object-fit: cover; border-radius: 6px; border: 1px solid var(--border); background: #edf2f1; }
    .name { font-weight: 700; overflow-wrap: anywhere; }
    .meta { color: var(--muted); font-size: 12px; overflow-wrap: anywhere; }
    .pill { display: inline-block; border-radius: 999px; padding: 2px 7px; margin: 6px 4px 0 0; font-size: 11px; border: 1px solid var(--border); }
    .pill.good { color: var(--green); border-color: #b8dfd1; background: #f2fbf8; }
    .pill.bad { color: var(--red); border-color: #e8bcc2; background: #fff5f6; }
    .pill.warn { color: var(--amber); border-color: #ebd0b5; background: #fff8ef; }
    .controls { display: flex; gap: 7px; margin-top: 8px; flex-wrap: wrap; }
    .controls img { width: 54px; height: 54px; }
    @media (max-width: 900px) { .metrics, .two { grid-template-columns: 1fr; } main, header { padding-left: 14px; padding-right: 14px; } }
  </style>
</head>
<body>
  <header>
    <h1>ADA Failure Atlas</h1>
    <p class="subtle" id="subtitle"></p>
  </header>
  <main>
    <section class="grid metrics" id="metrics"></section>
    <section class="grid two">
      <div class="panel">
        <h2>Failure Signatures</h2>
        <div id="signatureTable"></div>
      </div>
      <div class="panel">
        <h2>Top Confusion Pairs</h2>
        <div id="confusionTable"></div>
      </div>
    </section>
    <section class="panel" style="margin-top: 14px">
      <h2>Actionable Candidate Explorer</h2>
      <div class="toolbar">
        <select id="signatureFilter"></select>
        <select id="sortMode">
          <option value="actionability">actionability</option>
          <option value="support">weakest support</option>
          <option value="independent">independent confidence</option>
          <option value="task">task confidence</option>
        </select>
        <input id="search" type="search" placeholder="sample, class, path">
      </div>
      <p class="subtle" id="candidateLine"></p>
      <div class="cards" id="cards"></div>
    </section>
  </main>
  <script id="failure-atlas-data" type="application/json">__FAILURE_ATLAS_DATA__</script>
  <script>
    const data = JSON.parse(document.getElementById('failure-atlas-data').textContent);
    const sigFilter = document.getElementById('signatureFilter');
    const sortMode = document.getElementById('sortMode');
    const search = document.getElementById('search');
    const signatures = ['all', ...new Set(data.candidates.map(row => row.signature))];
    for (const sig of signatures) {
      const opt = document.createElement('option');
      opt.value = sig;
      opt.textContent = sig === 'all' ? 'all signatures' : sig.replaceAll('_', ' ');
      sigFilter.appendChild(opt);
    }
    for (const el of [sigFilter, sortMode, search]) {
      el.addEventListener('input', renderCards);
      el.addEventListener('change', renderCards);
    }
    function pct(v) { return `${(100 * Number(v)).toFixed(1)}%`; }
    function fmt(v, n = 3) { return Number(v).toFixed(n); }
    function esc(v) { return String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])); }
    function renderStatic() {
      const r = data.report;
      document.getElementById('subtitle').textContent = `${r.num_samples.toLocaleString()} samples · task ${r.task_model} · independent ${r.independent_model} · support ${r.support_column}`;
      const topSig = data.signatureRows[0] || {};
      const metrics = [
        ['Samples', r.num_samples.toLocaleString(), 'validation rows collapsed by sample'],
        ['Candidates', r.num_actionable_candidates.toLocaleString(), 'task wrong and weak support'],
        ['Weak support', pct(r.weak_support_fraction), 'largest class-support distances'],
        ['Top signature', topSig.signature ? topSig.signature.replaceAll('_', ' ') : 'n/a', `${topSig.count || 0} samples`]
      ];
      document.getElementById('metrics').innerHTML = metrics.map(([label, value, note]) => `<div class="metric"><div class="label">${esc(label)}</div><div class="value">${esc(value)}</div><div class="subtle">${esc(note)}</div></div>`).join('');
      document.getElementById('signatureTable').innerHTML = `<table><thead><tr><th>Signature</th><th>Count</th><th>Task err</th><th>Ind. err</th><th>kNN ok</th><th>Mean score</th></tr></thead><tbody>${data.signatureRows.map(row => `<tr><td>${esc(row.signature.replaceAll('_', ' '))}</td><td>${row.count}</td><td>${pct(row.task_error_rate)}</td><td>${pct(row.independent_error_rate)}</td><td>${pct(row.purity_correct_rate)}</td><td>${fmt(row.mean_actionability_score, 2)}</td></tr>`).join('')}</tbody></table>`;
      document.getElementById('confusionTable').innerHTML = `<table><thead><tr><th>Model</th><th>True</th><th>Pred</th><th>Count</th><th>Weak</th></tr></thead><tbody>${data.confusionRows.map(row => `<tr><td>${esc(row.model_id)}</td><td>${row.true_label}</td><td>${row.predicted_label}</td><td>${row.count}</td><td>${row.weak_support_count}</td></tr>`).join('')}</tbody></table>`;
    }
    function filteredCandidates() {
      const q = search.value.trim().toLowerCase();
      let rows = data.candidates.slice();
      if (sigFilter.value !== 'all') rows = rows.filter(row => row.signature === sigFilter.value);
      if (q) rows = rows.filter(row =>
        row.sample_id.toLowerCase().includes(q) ||
        String(row.true_label).includes(q) ||
        String(row.task_predicted_label).includes(q) ||
        String(row.independent_predicted_label).includes(q) ||
        row.class_name.toLowerCase().includes(q) ||
        row.relative_path.toLowerCase().includes(q)
      );
      const mode = sortMode.value;
      rows.sort((a, b) => {
        if (mode === 'support') return b[data.report.support_column] - a[data.report.support_column];
        if (mode === 'independent') return Number(b.independent_confidence || 0) - Number(a.independent_confidence || 0);
        if (mode === 'task') return Number(b.task_confidence || 0) - Number(a.task_confidence || 0);
        return b.actionability_score - a.actionability_score;
      });
      return rows;
    }
    function renderCards() {
      const rows = filteredCandidates();
      document.getElementById('candidateLine').textContent = `${Math.min(rows.length, 80)} shown from ${rows.length.toLocaleString()} matching candidates`;
      document.getElementById('cards').innerHTML = rows.slice(0, 80).map(row => {
        const controls = data.controlsByCandidate[row.sample_id] || [];
        return `<article class="card">
          <div class="pair">
            <img src="${esc(row.image_src)}" alt="">
            <div>
              <div class="name">${esc(row.class_name || row.relative_path || row.sample_id)}</div>
              <div class="meta">${esc(row.sample_id)}</div>
              <div class="meta">true ${row.true_label} · task pred ${row.task_predicted_label} · independent pred ${row.independent_predicted_label}</div>
              <span class="pill bad">task wrong</span>
              <span class="pill ${Number(row.independent_correct) ? 'good' : 'bad'}">independent ${Number(row.independent_correct) ? 'correct' : 'wrong'}</span>
              <span class="pill warn">support ${fmt(row[data.report.support_column])}</span>
              <div class="meta" style="margin-top:6px">score ${fmt(row.actionability_score, 2)} · task conf ${fmt(row.task_confidence, 2)} · ind conf ${fmt(row.independent_confidence, 2)}</div>
            </div>
          </div>
          <div class="controls">${controls.map(ctrl => `<img src="${esc(ctrl.image_src)}" title="matched correct control ${esc(ctrl.control_sample_id)}">`).join('')}</div>
        </article>`;
      }).join('');
    }
    renderStatic();
    renderCards();
  </script>
</body>
</html>
"""
