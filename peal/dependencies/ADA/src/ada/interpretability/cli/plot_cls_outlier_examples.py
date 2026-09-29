from __future__ import annotations

import argparse
import csv
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

from ada.atlas.hashing import file_sha1, hash_rows, stable_hash


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot DINO CLS outlier examples with same-class average controls and downstream correctness."
    )
    parser.add_argument("--outlier-csv", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--deletion-evaluation-csv", type=Path, required=True)
    parser.add_argument("--train-image-root", type=Path, required=True)
    parser.add_argument("--validation-image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-images", type=int, default=12)
    parser.add_argument("--average-max-images", type=int, default=96)
    parser.add_argument("--baseline-seed", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=176)
    parser.add_argument(
        "--sort-by",
        choices=("input", "dino_cls_outlier_percentile", "text_margin"),
        default="dino_cls_outlier_percentile",
        help="Rows are joined before sorting. dino_cls_outlier_percentile sorts descending; text_margin sorts ascending.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not args.overwrite:
        raise FileExistsError(f"plot artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    outliers_all = _read_csv(args.outlier_csv)
    manifest = _read_csv(Path(args.manifest_dir) / "samples.csv")
    manifest_by_sample = {str(row["sample_id"]): row for row in manifest}
    controls_by_region = _controls_by_region(manifest)
    outliers = _sort_outliers(outliers_all, manifest_by_sample, str(args.sort_by))[: int(args.max_images)]
    correctness = _baseline_correctness(
        args.deletion_evaluation_csv,
        baseline_seed=int(args.baseline_seed),
        wanted_sample_ids=[str(row["sample_id"]) for row in outliers],
    )

    avg_dir = output / "class_averages"
    avg_dir.mkdir(parents=True, exist_ok=True)
    avg_paths: dict[str, Path] = {}
    for row in outliers:
        region_id = str(row["region_id"])
        if region_id in avg_paths:
            continue
        avg_paths[region_id] = _write_region_average(
            region_id=region_id,
            controls=controls_by_region.get(region_id, []),
            train_image_root=args.train_image_root,
            output_dir=avg_dir,
            image_size=int(args.image_size),
            max_images=int(args.average_max_images),
        )

    plot_rows: list[dict[str, object]] = []
    for row in outliers:
        sample_id = str(row["sample_id"])
        sample = manifest_by_sample.get(sample_id, {})
        pred = correctness.get(sample_id, {})
        support_percentile = _float(sample.get("support_percentile"))
        plot_rows.append(
            {
                **row,
                "dino_cls_outlier_percentile": support_percentile,
                "downstream_model_id": pred.get("model_id", "dino_cls_deletion_probe"),
                "downstream_predicted_label": pred.get("predicted_label", ""),
                "downstream_correct_full_data_seed0": pred.get("correct", ""),
                "downstream_probe_probability": pred.get(
                    "true_class_probability",
                    pred.get("max_probability_calibrated", pred.get("max_probability_raw", "")),
                ),
                "downstream_probe_logit_margin": pred.get(
                    "true_class_logit_margin",
                    pred.get("logit_margin", ""),
                ),
                "class_average_image": str(avg_paths.get(str(row["region_id"]), "")),
                "class_average_source": "same_class_high_support_train_controls",
            }
        )

    panel = output / "cls_outlier_examples_panel.png"
    _draw_panel(
        rows=plot_rows,
        panel_path=panel,
        image_size=int(args.image_size),
    )
    rows_csv = output / "cls_outlier_examples_panel_rows.csv"
    _write_csv(rows_csv, plot_rows)
    report = output / "report.md"
    report.write_text(_markdown_report(plot_rows, panel, rows_csv), encoding="utf-8")

    result_hash = hash_rows(plot_rows, prefix="cls-outlier-panel")
    metadata = {
        "artifact_id": stable_hash(
            {
                "result_hash": result_hash,
                "outlier_csv": str(args.outlier_csv),
                "manifest_dir": str(args.manifest_dir),
                "deletion_evaluation_csv": str(args.deletion_evaluation_csv),
            },
            prefix="cls-outlier-panel-artifact",
        ),
        "result_hash": result_hash,
        "summary": {
            "rows": len(plot_rows),
            "selection": f"Rows selected from outlier_csv after join; sort_by={args.sort_by}.",
            "dino_cls_outlier_percentile": "support_percentile from the frozen eight-region manifest; larger means less local DINOv2-CLS support.",
            "downstream_correctness": "DINOv2-CLS full-data q=1 baseline probe prediction for the selected baseline seed.",
            "class_average": "Pixel average over same-class high-support train controls from the frozen interpretability manifest.",
        },
        "outputs": {
            "panel_png": str(panel),
            "rows_csv": str(rows_csv),
            "report_md": str(report),
        },
        "config": {
            "outlier_csv": str(args.outlier_csv),
            "outlier_csv_sha1": file_sha1(args.outlier_csv),
            "manifest_dir": str(args.manifest_dir),
            "deletion_evaluation_csv": str(args.deletion_evaluation_csv),
            "train_image_root": str(args.train_image_root),
            "validation_image_root": str(args.validation_image_root),
            "max_images": int(args.max_images),
            "average_max_images": int(args.average_max_images),
            "baseline_seed": int(args.baseline_seed),
            "image_size": int(args.image_size),
            "sort_by": str(args.sort_by),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    completed.write_text("ok\n", encoding="utf-8")
    print(json.dumps({"metadata_json": str(output / "metadata.json"), **metadata["outputs"]}, indent=2, sort_keys=True))


def _controls_by_region(rows: Sequence[Mapping[str, str]]) -> dict[str, list[Mapping[str, str]]]:
    preferred: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    fallback: dict[str, list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        region_id = str(row.get("region_id", ""))
        if not region_id:
            continue
        group = str(row.get("analysis_group", ""))
        if group == "same_class_supported_train_control":
            preferred[region_id].append(row)
        elif group == "region_train_member":
            fallback[region_id].append(row)
    out: dict[str, list[Mapping[str, str]]] = {}
    for region_id in sorted(set(preferred) | set(fallback)):
        out[region_id] = preferred.get(region_id) or fallback.get(region_id, [])
    return out


def _sort_outliers(
    rows: Sequence[Mapping[str, str]],
    manifest_by_sample: Mapping[str, Mapping[str, str]],
    sort_by: str,
) -> list[Mapping[str, str]]:
    out = list(rows)
    if sort_by == "input":
        return out
    if sort_by == "text_margin":
        return sorted(out, key=lambda row: _float(row.get("text_margin")))
    if sort_by == "dino_cls_outlier_percentile":
        return sorted(
            out,
            key=lambda row: _float(manifest_by_sample.get(str(row.get("sample_id", "")), {}).get("support_percentile")),
            reverse=True,
        )
    raise ValueError(f"Unsupported sort_by={sort_by!r}")


def _baseline_correctness(path: Path, *, baseline_seed: int, wanted_sample_ids: Sequence[str]) -> dict[str, dict[str, str]]:
    wanted = set(wanted_sample_ids)
    eval_rows = _read_csv(path)
    pred_paths: list[Path] = []
    for row in eval_rows:
        if str(row.get("control_family", "")) != "baseline":
            continue
        if not math.isclose(_float(row.get("retention_level")), 1.0):
            continue
        if int(float(row.get("seed", "-1") or "-1")) != int(baseline_seed):
            continue
        pred = row.get("probe_predictions_csv", "")
        if pred:
            pred_paths.append(Path(str(pred)))

    out: dict[str, dict[str, str]] = {}
    for pred_path in pred_paths:
        for row in _read_csv(pred_path):
            sample_id = str(row.get("sample_id", ""))
            if sample_id in wanted and sample_id not in out:
                out[sample_id] = row
        if len(out) >= len(wanted):
            break
    return out


def _write_region_average(
    *,
    region_id: str,
    controls: Sequence[Mapping[str, str]],
    train_image_root: Path,
    output_dir: Path,
    image_size: int,
    max_images: int,
) -> Path:
    arrays: list[np.ndarray] = []
    for row in controls[: max(1, int(max_images))]:
        rel = str(row.get("relative_path", ""))
        if not rel:
            continue
        path = Path(train_image_root) / rel
        try:
            image = _load_square(path, image_size)
        except (FileNotFoundError, OSError):
            continue
        arrays.append(np.asarray(image, dtype=np.float32))
    out_path = output_dir / f"{region_id}_same_class_average.png"
    if arrays:
        avg = np.clip(np.mean(np.stack(arrays, axis=0), axis=0), 0, 255).astype(np.uint8)
        Image.fromarray(avg, mode="RGB").save(out_path)
    else:
        Image.new("RGB", (image_size, image_size), (238, 238, 238)).save(out_path)
    return out_path


def _draw_panel(*, rows: Sequence[Mapping[str, object]], panel_path: Path, image_size: int) -> None:
    gutter = 16
    text_w = 560
    row_h = image_size + 28
    header_h = 64
    width = image_size * 2 + text_w + gutter * 4
    height = header_h + row_h * len(rows) + gutter
    canvas = Image.new("RGB", (width, height), (250, 250, 248))
    draw = ImageDraw.Draw(canvas)
    font = _font(15)
    small = _font(13)
    title_font = _font(20)
    draw.text((gutter, 18), "DINOv2-CLS Outlier Examples", fill=(24, 24, 24), font=title_font)
    draw.text(
        (gutter, 43),
        "Left: validation image. Middle: same-class high-support train average. Right: support/text/correctness.",
        fill=(88, 88, 88),
        font=small,
    )

    y0 = header_h
    for idx, row in enumerate(rows, start=1):
        val_image = _load_square(Path(str(row["image_path"])), image_size)
        avg_image = _load_square(Path(str(row["class_average_image"])), image_size)
        x_img = gutter
        x_avg = gutter * 2 + image_size
        x_text = gutter * 3 + image_size * 2
        y = y0 + (idx - 1) * row_h
        canvas.paste(val_image, (x_img, y))
        canvas.paste(avg_image, (x_avg, y))
        draw.rectangle((x_img, y, x_img + image_size - 1, y + image_size - 1), outline=(180, 180, 180))
        draw.rectangle((x_avg, y, x_avg + image_size - 1, y + image_size - 1), outline=(180, 180, 180))
        draw.text((x_img, y + image_size + 4), "outlier image", fill=(55, 55, 55), font=small)
        draw.text((x_avg, y + image_size + 4), "class/control average", fill=(55, 55, 55), font=small)
        text_lines = _row_text(idx, row)
        draw.multiline_text((x_text, y + 2), "\n".join(text_lines), fill=(26, 26, 26), font=font, spacing=4)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(panel_path)


def _row_text(idx: int, row: Mapping[str, object]) -> list[str]:
    correct = str(row.get("downstream_correct_full_data_seed0", ""))
    correct_text = "unknown"
    if correct != "":
        correct_text = "yes" if int(float(correct)) == 1 else "no"
    pred = row.get("downstream_predicted_label", "")
    prob = _maybe_float(row.get("downstream_probe_probability"))
    margin = _maybe_float(row.get("downstream_probe_logit_margin"))
    outlier_pct = _maybe_float(row.get("dino_cls_outlier_percentile"))
    text_margin = _maybe_float(row.get("text_margin"))
    repair = _maybe_float(row.get("q25_unique_target_repair_gain"))
    nontarget = _maybe_float(row.get("q25_same_class_nontarget_repair_gain"))
    lines = [
        f"{idx}. {row.get('class_name', '')} / {row.get('pilot_type', '')}",
        f"sample: {row.get('sample_id', '')}",
        f"DINO-CLS outlier percentile: {_fmt(outlier_pct)}",
        f"SigLIP text margin: {_fmt(text_margin)} vs {row.get('top_competing_class_name', '')}",
        f"full-data downstream correct: {correct_text}; pred={pred}",
        f"probe conf={_fmt(prob)}; top/logit margin={_fmt(margin)}",
        f"q=.25 unique repair={_fmt(repair)}; non-target repair={_fmt(nontarget)}",
    ]
    desc = str(row.get("region_descriptors", ""))
    if desc:
        wrapped = textwrap.wrap("descriptors: " + desc, width=64)
        lines.extend(wrapped[:3])
    return lines


def _markdown_report(rows: Sequence[Mapping[str, object]], panel: Path, rows_csv: Path) -> str:
    lines = [
        "# DINOv2-CLS Outlier Example Panel",
        "",
        f"![DINOv2-CLS outlier examples]({panel.name})",
        "",
        "The outlier percentile is the frozen DINOv2-CLS same-class support percentile from the eight-region manifest; larger means less local support. The downstream correctness column uses the full-data q=1 DINOv2-CLS linear probe baseline for seed 0. Rows are sorted by DINOv2-CLS outlier percentile unless configured otherwise.",
        "",
        f"- rows: [{rows_csv.name}]({rows_csv.name})",
        "",
        "## Image Links",
        "",
    ]
    for idx, row in enumerate(rows, start=1):
        lines.append(
            f"{idx}. [{row.get('relative_path', '')}](<{row.get('image_path', '')}>) "
            f"- `{row.get('class_name', '')}`, correct=`{row.get('downstream_correct_full_data_seed0', '')}`, "
            f"outlier_percentile=`{_fmt(_maybe_float(row.get('dino_cls_outlier_percentile')))}`"
        )
    lines.append("")
    return "\n".join(lines)


def _load_square(path: Path, size: int) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.fit(image.convert("RGB"), (size, size), method=Image.Resampling.BICUBIC)


def _font(size: int) -> ImageFont.ImageFont:
    for name in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
    ):
        path = Path(name)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(str(key))
                fieldnames.append(str(key))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def _float(value: object) -> float:
    if value in ("", None):
        return float("nan")
    return float(value)


def _maybe_float(value: object) -> float | None:
    try:
        out = _float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _fmt(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


if __name__ == "__main__":
    main()
