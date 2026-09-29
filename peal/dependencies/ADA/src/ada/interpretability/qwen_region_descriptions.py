from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from ada.atlas.hashing import hash_rows, stable_hash


REGION_PROMPT = """You are auditing two sets of images from the same object class.

Set A and Set B contain different subsets of that class.

Identify visual attributes that are consistently more common in Set A than Set B.
Also identify visual attributes that are consistently more common in Set B than Set A.
Focus only on observable properties:
- viewpoint and pose
- scale and crop
- occlusion and visibility
- number of objects
- lighting and color
- texture
- background and context
- photographic versus non-photographic medium
- possible ambiguity with another object class

Do not speculate about why the images were selected.
Do not mention sparse regions, outliers, model errors, or training data.
Do not report an attribute unless multiple images support it.
For every attribute, cite the image indices that provide evidence.
Return valid JSON only. Do not use markdown fences.
Use this exact schema:
{
  "shared_class_identity": "short phrase",
  "set_a_enriched_attributes": [
    {
      "phrase": "observable attribute",
      "description": "one sentence",
      "evidence_a": ["A1", "A3"],
      "counterexamples_a": [],
      "evidence_b": [],
      "confidence": 0.0
    }
  ],
  "set_b_enriched_attributes": [
    {
      "phrase": "observable attribute",
      "description": "one sentence",
      "evidence_b": ["B1", "B3"],
      "counterexamples_b": [],
      "evidence_a": [],
      "confidence": 0.0
    }
  ],
  "class_ambiguities": [
    {
      "competing_class": "class name or empty string",
      "evidence": ["A2", "B4"],
      "confidence": 0.0
    }
  ],
  "uncertain_observations": []
}
"""

REGION_PROMPT_CANDIDATE = """You are auditing two sets of images from the same object class.

Set A and Set B contain different subsets of that class.

Identify observable visual attributes that appear more common in one set than the other.
Subtle but visible differences are useful; assign lower confidence when the evidence is weak.
If you see only a weak pattern, put it in uncertain_observations instead of pretending it is certain.
Focus on:
- viewpoint and pose
- scale and crop
- occlusion and visibility
- number of objects
- lighting and color
- texture
- background and context
- photographic versus non-photographic medium
- possible ambiguity with another object class

Do not speculate about why the images were selected.
Do not mention sparse regions, outliers, model errors, or training data.
Do not report an attribute unless at least two images support it.
For every attribute, cite the image indices that provide evidence.
Return valid JSON only. Do not use markdown fences.
Use this exact schema:
{
  "shared_class_identity": "short phrase",
  "set_a_enriched_attributes": [
    {
      "phrase": "observable attribute",
      "description": "one sentence",
      "evidence_a": ["A1", "A3"],
      "counterexamples_a": [],
      "evidence_b": [],
      "confidence": 0.0
    }
  ],
  "set_b_enriched_attributes": [
    {
      "phrase": "observable attribute",
      "description": "one sentence",
      "evidence_b": ["B1", "B3"],
      "counterexamples_b": [],
      "evidence_a": [],
      "confidence": 0.0
    }
  ],
  "class_ambiguities": [
    {
      "competing_class": "class name or empty string",
      "evidence": ["A2", "B4"],
      "confidence": 0.0
    }
  ],
  "uncertain_observations": [
    {
      "phrase": "tentative observable difference",
      "direction": "set_a or set_b",
      "evidence": ["A1", "A2"],
      "reason_uncertain": "short reason"
    }
  ]
}
"""


@dataclass(frozen=True)
class QwenRegionDescriptionConfig:
    manifest_dir: Path
    region_cards_csv: Path
    output_dir: Path
    train_image_root: Path
    validation_image_root: Path
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    model_path: Path | None = None
    model_revision: str = ""
    max_regions: int | None = None
    max_tasks: int | None = None
    images_per_set: int = 8
    max_new_tokens: int = 768
    temperature: float = 0.0
    top_p: float = 1.0
    device: str = "auto"
    dtype: str = "auto"
    run_inference: bool = False
    include_swapped_order: bool = False
    include_null_control: bool = False
    deterministic_repeats: int = 1
    prompt_variant: str = "strict"
    overwrite: bool = False


def build_qwen_region_descriptions(config: QwenRegionDescriptionConfig) -> dict[str, object]:
    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"Qwen region description artifact already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    samples = _read_csv(Path(config.manifest_dir) / "samples.csv")
    cards = _read_csv(config.region_cards_csv)
    tasks = _build_tasks(samples, cards, config)
    if config.max_tasks is not None:
        tasks = tasks[: int(config.max_tasks)]
    task_jsonl = output / "qwen_region_tasks.jsonl"
    image_csv = output / "qwen_region_task_images.csv"
    _write_jsonl(task_jsonl, tasks)
    _write_csv(image_csv, _task_image_rows(tasks))

    outputs: list[dict[str, object]] = []
    inference_status = "NOT_REQUESTED"
    metrics: dict[str, object] | None = None
    runtime: dict[str, object] = {}
    if config.run_inference:
        outputs, inference_status, runtime = _run_inference(tasks, config)
        _write_jsonl(output / "qwen_region_hypotheses.jsonl", outputs)
        metrics = _canary_metrics(outputs, tasks, max(1, int(config.deterministic_repeats)))
        (output / "qwen_canary_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8")

    result_hash = hash_rows(tasks, prefix="qwen-region-tasks")
    metadata = {
        "artifact_id": stable_hash(
            {
                "result_hash": result_hash,
                "model_name": config.model_name,
                "model_path": "" if config.model_path is None else str(config.model_path),
                "run_inference": bool(config.run_inference),
            },
            prefix="qwen-region-artifact",
        ),
        "result_hash": result_hash,
        "model": {
            "model_name": config.model_name,
            "model_path": "" if config.model_path is None else str(config.model_path),
            "model_revision": config.model_revision,
            "dtype": config.dtype,
        },
        "summary": {
            "tasks": len(tasks),
            "regions": len({str(task["region_id"]) for task in tasks}),
            "run_inference": bool(config.run_inference),
            "inference_status": inference_status,
            "outputs": len(outputs),
        },
        "runtime": runtime,
        "outputs": {
            "qwen_region_tasks_jsonl": str(task_jsonl),
            "qwen_region_task_images_csv": str(image_csv),
            "qwen_region_hypotheses_jsonl": str(output / "qwen_region_hypotheses.jsonl") if outputs else "",
            "qwen_canary_metrics_json": str(output / "qwen_canary_metrics.json") if metrics is not None else "",
            "report_md": str(output / "report.md"),
        },
        "config": {
            "manifest_dir": str(config.manifest_dir),
            "region_cards_csv": str(config.region_cards_csv),
            "train_image_root": str(config.train_image_root),
            "validation_image_root": str(config.validation_image_root),
            "max_regions": config.max_regions,
            "max_tasks": config.max_tasks,
            "images_per_set": int(config.images_per_set),
            "run_inference": bool(config.run_inference),
            "include_swapped_order": bool(config.include_swapped_order),
            "include_null_control": bool(config.include_null_control),
            "deterministic_repeats": int(config.deterministic_repeats),
            "prompt_variant": str(config.prompt_variant),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    (output / "report.md").write_text(_report(tasks, metadata), encoding="utf-8")
    completed.write_text("ok\n", encoding="utf-8")
    return metadata


def _build_tasks(
    samples: Sequence[Mapping[str, str]],
    cards: Sequence[Mapping[str, str]],
    config: QwenRegionDescriptionConfig,
) -> list[dict[str, object]]:
    by_region: dict[str, list[Mapping[str, str]]] = {}
    for row in samples:
        by_region.setdefault(str(row["region_id"]), []).append(row)
    tasks: list[dict[str, object]] = []
    region_rows = list(cards)
    if config.max_regions is not None:
        region_rows = region_rows[: int(config.max_regions)]
    null_added = False
    for card in region_rows:
        region_id = str(card["region_id"])
        rows = by_region.get(region_id, [])
        region_members = _select_images(rows, "region_train_member", config.train_image_root, int(config.images_per_set))
        controls = _select_images(rows, "same_class_high_support_control", config.train_image_root, int(config.images_per_set))
        retained, deleted, deletion_seed = _select_deleted_retained_images(
            rows,
            config.train_image_root,
            int(config.images_per_set),
        )
        if region_members and controls:
            tasks.append(
                _task(
                    task_id=f"{region_id}:region_vs_control",
                    comparison_type="region_vs_supported_control",
                    region=card,
                    set_a_name="region members",
                    set_a=region_members,
                    set_b_name="same-class supported controls",
                    set_b=controls,
                    prompt=_prompt_for_variant(config.prompt_variant),
                )
            )
            if config.include_swapped_order:
                tasks.append(
                    _task(
                        task_id=f"{region_id}:region_vs_control_swap",
                        comparison_type="region_vs_supported_control_swap",
                        region=card,
                        set_a_name="same-class supported controls",
                        set_a=controls,
                        set_b_name="region members",
                        set_b=region_members,
                        prompt=_prompt_for_variant(config.prompt_variant),
                        extra={"swapped_of": f"{region_id}:region_vs_control"},
                    )
                )
        if retained and deleted:
            tasks.append(
                _task(
                    task_id=f"{region_id}:deleted_vs_retained_seed{deletion_seed}",
                    comparison_type="deleted_vs_retained",
                    region=card,
                    set_a_name="retained q=0.25 anchors",
                    set_a=retained,
                    set_b_name="deleted q=0.25 real examples",
                    set_b=deleted,
                    prompt=_prompt_for_variant(config.prompt_variant),
                    extra={"deletion_seed": deletion_seed},
                )
            )
            if config.include_swapped_order:
                tasks.append(
                    _task(
                        task_id=f"{region_id}:deleted_vs_retained_seed{deletion_seed}_swap",
                        comparison_type="deleted_vs_retained_swap",
                        region=card,
                        set_a_name="deleted q=0.25 real examples",
                        set_a=deleted,
                        set_b_name="retained q=0.25 anchors",
                        set_b=retained,
                        prompt=_prompt_for_variant(config.prompt_variant),
                        extra={
                            "deletion_seed": deletion_seed,
                            "swapped_of": f"{region_id}:deleted_vs_retained_seed{deletion_seed}",
                        },
                    )
                )
        if config.include_null_control and not null_added:
            null_a, null_b = _select_null_split(rows, "region_train_member", config.train_image_root, int(config.images_per_set))
            if null_a and null_b:
                tasks.append(
                    _task(
                        task_id=f"{region_id}:null_random_split",
                        comparison_type="null_random_split",
                        region=card,
                        set_a_name="random same-region split A",
                        set_a=null_a,
                        set_b_name="random same-region split B",
                        set_b=null_b,
                        prompt=_prompt_for_variant(config.prompt_variant),
                    )
                )
                if config.include_swapped_order:
                    tasks.append(
                        _task(
                            task_id=f"{region_id}:null_random_split_swap",
                            comparison_type="null_random_split_swap",
                            region=card,
                            set_a_name="random same-region split B",
                            set_a=null_b,
                            set_b_name="random same-region split A",
                            set_b=null_a,
                            prompt=_prompt_for_variant(config.prompt_variant),
                            extra={"swapped_of": f"{region_id}:null_random_split"},
                        )
                    )
                null_added = True
    return tasks


def _task(
    *,
    task_id: str,
    comparison_type: str,
    region: Mapping[str, str],
    set_a_name: str,
    set_a: Sequence[Mapping[str, object]],
    set_b_name: str,
    set_b: Sequence[Mapping[str, object]],
    prompt: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, object]:
    task = {
        "task_id": task_id,
        "comparison_type": comparison_type,
        "region_id": str(region["region_id"]),
        "class_id": int(region["class_id"]),
        "class_name": str(region["class_name"]),
        "pilot_type": str(region.get("pilot_type", "")),
        "set_a_name": set_a_name,
        "set_b_name": set_b_name,
        "set_a_images": list(set_a),
        "set_b_images": list(set_b),
        "prompt": prompt,
    }
    if extra:
        task.update(dict(extra))
    return task


def _prompt_for_variant(variant: str) -> str:
    key = str(variant or "strict").strip().lower()
    if key in {"strict", "default"}:
        return REGION_PROMPT
    if key in {"candidate", "candidate_discovery", "class_aware"}:
        return REGION_PROMPT_CANDIDATE
    raise ValueError(f"unknown Qwen prompt variant: {variant}")


def _select_deleted_retained_images(
    rows: Sequence[Mapping[str, str]],
    root: Path,
    count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], str]:
    seeds = sorted(
        {
            str(row.get("deletion_seed", ""))
            for row in rows
            if str(row.get("analysis_group", "")) in {"q025_retained_anchor", "q025_deleted_real"}
            and str(row.get("deletion_seed", "")) != ""
        },
        key=_seed_sort_key,
    )
    best: tuple[int, int, str, list[dict[str, object]], list[dict[str, object]]] | None = None
    for seed in seeds:
        retained = _select_images(rows, "q025_retained_anchor", root, count, deletion_seed=seed)
        retained_ids = {str(row["sample_id"]) for row in retained}
        deleted = _select_images(
            rows,
            "q025_deleted_real",
            root,
            count,
            deletion_seed=seed,
            exclude_sample_ids=retained_ids,
        )
        score = (min(len(retained), len(deleted)), len(retained) + len(deleted))
        if len(retained) >= count and len(deleted) >= count:
            return retained, deleted, seed
        if retained and deleted and (best is None or score > (best[0], best[1])):
            best = (score[0], score[1], seed, retained, deleted)
    if best is None:
        return [], [], ""
    return best[3], best[4], best[2]


def _select_null_split(
    rows: Sequence[Mapping[str, str]],
    group: str,
    root: Path,
    count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    selected = _select_images(rows, group, root, count * 2)
    if len(selected) < count * 2:
        return [], []
    return selected[:count], selected[count: count * 2]


def _seed_sort_key(seed: str) -> tuple[int, str]:
    try:
        return int(seed), seed
    except ValueError:
        return 10**9, seed


def _select_images(
    rows: Sequence[Mapping[str, str]],
    group: str,
    root: Path,
    count: int,
    *,
    deletion_seed: str | None = None,
    exclude_sample_ids: set[str] | None = None,
) -> list[dict[str, object]]:
    selected = []
    seen: set[str] = set()
    excluded = set() if exclude_sample_ids is None else set(exclude_sample_ids)
    for row in rows:
        if str(row.get("analysis_group", "")) != group:
            continue
        if deletion_seed is not None and str(row.get("deletion_seed", "")) != str(deletion_seed):
            continue
        sid = str(row["sample_id"])
        if sid in seen or sid in excluded:
            continue
        seen.add(sid)
        selected.append(row)
    selected = sorted(selected, key=lambda row: str(row["sample_id"]))[: int(count)]
    out: list[dict[str, object]] = []
    for idx, row in enumerate(selected, start=1):
        out.append(
            {
                "index": idx,
                "sample_id": str(row["sample_id"]),
                "relative_path": str(row["relative_path"]),
                "path": str(Path(root) / str(row["relative_path"])),
                "analysis_group": group,
                "deletion_seed": str(row.get("deletion_seed", "")),
                "retained_at_q025": str(row.get("retained_at_q025", "")),
                "deleted_at_q025": str(row.get("deleted_at_q025", "")),
            }
        )
    return out


def _run_inference(
    tasks: Sequence[Mapping[str, object]],
    config: QwenRegionDescriptionConfig,
) -> tuple[list[dict[str, object]], str, dict[str, object]]:
    try:
        import sys
        import torch
        import transformers
        from transformers import AutoConfig, Qwen3VLForConditionalGeneration, Qwen3VLProcessor
    except Exception as exc:
        runtime = {"import_error": f"{type(exc).__name__}: {exc}"}
        return [_error_output(task, f"runtime_import_error: {type(exc).__name__}: {exc}") for task in tasks], "FAILED_IMPORT", runtime

    model_source = str(config.model_path) if config.model_path is not None and str(config.model_path) else str(config.model_name)
    runtime = {
        "python_executable": sys.executable,
        "transformers_version": getattr(transformers, "__version__", ""),
        "transformers_path": getattr(transformers, "__file__", ""),
        "model_source": model_source,
        "requested_dtype": str(config.dtype),
        "model_class": "Qwen3VLForConditionalGeneration",
    }
    try:
        hf_config = AutoConfig.from_pretrained(model_source, local_files_only=True)
        runtime["model_type"] = str(getattr(hf_config, "model_type", ""))
        runtime["config_class"] = type(hf_config).__name__
        if str(getattr(hf_config, "model_type", "")) != "qwen3_vl":
            raise ValueError(f"expected model_type='qwen3_vl', got {getattr(hf_config, 'model_type', '')!r}")
        processor = Qwen3VLProcessor.from_pretrained(model_source, local_files_only=True)
        runtime["processor_class"] = type(processor).__name__
    except Exception as exc:
        runtime["preflight_error"] = f"{type(exc).__name__}: {exc}"
        return [_error_output(task, f"preflight_error: {type(exc).__name__}: {exc}") for task in tasks], "FAILED_PREFLIGHT", runtime

    try:
        load_kwargs = _model_load_kwargs(torch, config.dtype)
        model = Qwen3VLForConditionalGeneration.from_pretrained(model_source, local_files_only=True, device_map="auto", **load_kwargs)
    except Exception as exc:
        runtime["model_load_error"] = f"{type(exc).__name__}: {exc}"
        return [_error_output(task, f"model_load_error: {type(exc).__name__}: {exc}") for task in tasks], "FAILED_MODEL_LOAD", runtime
    runtime["input_device"] = str(_model_input_device(model))

    outputs: list[dict[str, object]] = []
    for task in tasks:
        for repeat_index in range(max(1, int(config.deterministic_repeats))):
            try:
                images = _task_images(task)
                prompt = _render_prompt(task)
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": image} for image in images] + [{"type": "text", "text": prompt}],
                    }
                ]
                inputs = _apply_chat_template(processor, messages)
                inputs = inputs.to(_model_input_device(model))
                generation_kwargs = {
                    "max_new_tokens": int(config.max_new_tokens),
                    "do_sample": float(config.temperature) > 0.0,
                }
                if float(config.temperature) > 0.0:
                    generation_kwargs["temperature"] = float(config.temperature)
                    generation_kwargs["top_p"] = float(config.top_p)
                with torch.no_grad():
                    generated = model.generate(**inputs, **generation_kwargs)
                trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated)]
                text = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
                outputs.append(
                    {
                        **_task_header(task),
                        "repeat_index": repeat_index,
                        "status": "OK",
                        "raw_text": text,
                        "parsed_json": _try_parse_json(text),
                    }
                )
            except Exception as exc:
                outputs.append({**_error_output(task, f"inference_error: {type(exc).__name__}: {exc}"), "repeat_index": repeat_index})
    return outputs, "COMPLETED", runtime


def _model_load_kwargs(torch, dtype: str) -> dict[str, object]:
    value = str(dtype).lower()
    if value in ("auto", ""):
        return {"dtype": "auto"}
    return {"dtype": _torch_dtype(torch, value)}


def _model_input_device(model) -> object:
    device = getattr(model, "device", None)
    if device is not None:
        return device
    return next(model.parameters()).device


def _apply_chat_template(processor, messages):
    try:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            add_vision_id=True,
            return_dict=True,
            return_tensors="pt",
        )
    except TypeError:
        return processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )


def _render_prompt(task: Mapping[str, object]) -> str:
    set_a = task["set_a_images"]
    set_b = task["set_b_images"]
    labels_a = [f"A{x['index']}" for x in set_a]
    labels_b = [f"B{x['index']}" for x in set_b]
    lines = [
        str(task["prompt"]),
        "",
        f"Both sets are from the same ImageNet class id: {task['class_name']}",
        f"Set A image indices: {', '.join(labels_a)}",
        f"Set B image indices: {', '.join(labels_b)}",
        "Images are provided in order: all Set A images first, then all Set B images.",
        f"Valid evidence labels are exactly: {', '.join(labels_a + labels_b)}.",
        "Never cite an image label that is not in that exact list.",
    ]
    return "\n".join(lines)


def _task_images(task: Mapping[str, object]):
    from PIL import Image

    images = []
    for image_row in list(task["set_a_images"]) + list(task["set_b_images"]):
        images.append(Image.open(str(image_row["path"])).convert("RGB"))
    return images


def _error_output(task: Mapping[str, object], error: str) -> dict[str, object]:
    return {**_task_header(task), "status": "ERROR", "error": error, "raw_text": "", "parsed_json": None}


def _task_header(task: Mapping[str, object]) -> dict[str, object]:
    header = {
        "task_id": str(task["task_id"]),
        "comparison_type": str(task["comparison_type"]),
        "region_id": str(task["region_id"]),
        "class_id": int(task["class_id"]),
        "class_name": str(task["class_name"]),
        "pilot_type": str(task["pilot_type"]),
    }
    if "deletion_seed" in task:
        header["deletion_seed"] = str(task["deletion_seed"])
    if "swapped_of" in task:
        header["swapped_of"] = str(task["swapped_of"])
    return header


def _try_parse_json(text: str):
    try:
        return json.loads(text)
    except Exception:
        return None


def _canary_metrics(
    outputs: Sequence[Mapping[str, object]],
    tasks: Sequence[Mapping[str, object]],
    expected_repeats: int,
) -> dict[str, object]:
    forbidden = ("sparse", "outlier", "training", "coverage", "model error", "dataset imbalance", "deletion", "restoration")
    task_ids = [str(task["task_id"]) for task in tasks]
    task_lookup = {str(task["task_id"]): task for task in tasks}
    output_keys = [(str(row.get("task_id", "")), int(row.get("repeat_index", -1))) for row in outputs]
    duplicate_pairs = sorted(
        [f"{task_id}#{repeat}" for (task_id, repeat), count in _counts(output_keys).items() if count > 1]
    )
    expected_pairs = {(task_id, repeat) for task_id in task_ids for repeat in range(expected_repeats)}
    observed_pairs = set(output_keys)
    missing_pairs = sorted(f"{task_id}#{repeat}" for task_id, repeat in expected_pairs - observed_pairs)
    parsed = [row for row in outputs if row.get("parsed_json") is not None]
    ok = [row for row in outputs if str(row.get("status", "")) == "OK"]
    by_task: dict[str, list[str]] = {}
    by_task_rows: dict[str, list[Mapping[str, object]]] = {}
    for row in outputs:
        by_task.setdefault(str(row.get("task_id", "")), []).append(str(row.get("raw_text", "")))
        by_task_rows.setdefault(str(row.get("task_id", "")), []).append(row)
    deterministic_matches = {
        task_id: len(set(texts)) <= 1
        for task_id, texts in by_task.items()
        if len(texts) > 1
    }
    attr_rows = []
    null_attr_rows = []
    uncertain_rows = []
    null_uncertain_rows = []
    schema_errors: list[dict[str, object]] = []
    citation_errors: list[dict[str, object]] = []
    possible_truncations: list[dict[str, object]] = []
    for row in outputs:
        parsed_json = row.get("parsed_json")
        if str(row.get("status", "")) == "OK" and parsed_json is None:
            possible_truncations.append(
                {
                    "task_id": row.get("task_id", ""),
                    "repeat_index": row.get("repeat_index", ""),
                    "raw_len": len(str(row.get("raw_text", ""))),
                    "ends_with_brace": str(row.get("raw_text", "")).rstrip().endswith("}"),
                }
            )
        if not isinstance(parsed_json, Mapping):
            continue
        schema_error = _schema_error(parsed_json)
        if schema_error:
            schema_errors.append(
                {
                    "task_id": row.get("task_id", ""),
                    "repeat_index": row.get("repeat_index", ""),
                    "error": schema_error,
                }
            )
        task = task_lookup.get(str(row.get("task_id", "")))
        if task is not None:
            citation_errors.extend(_citation_errors(row, task))
        attrs = _json_attributes(parsed_json)
        uncertain = _json_uncertain_observations(parsed_json)
        attr_rows.extend(attrs)
        uncertain_rows.extend(uncertain)
        if "null_random_split" in str(row.get("comparison_type", "")):
            null_attr_rows.extend(attrs)
            null_uncertain_rows.extend(uncertain)
    citation_rows = [attr for attr in attr_rows if _has_two_citations(attr)]
    uncertain_citation_rows = [attr for attr in uncertain_rows if _has_two_citations(attr)]
    canonical_repeat_matches, evidence_repeat_matches = _repeat_consistency(by_task_rows)
    reversal = _reversal_metrics(outputs)
    return {
        "tasks": len(tasks),
        "expected_repeats": expected_repeats,
        "expected_outputs": len(tasks) * expected_repeats,
        "outputs": len(outputs),
        "ok_outputs": len(ok),
        "unique_task_ids": len({str(row.get("task_id", "")) for row in outputs}),
        "missing_task_ids": sorted(set(task_ids) - {str(row.get("task_id", "")) for row in outputs}),
        "missing_task_repeat_pairs": missing_pairs,
        "duplicate_task_repeat_pairs": duplicate_pairs,
        "json_parse_rate": 0.0 if not outputs else len(parsed) / len(outputs),
        "schema_error_outputs": schema_errors,
        "possible_truncation_outputs": possible_truncations,
        "forbidden_term_outputs": [
            {
                "task_id": row.get("task_id", ""),
                "repeat_index": row.get("repeat_index", ""),
                "terms": [term for term in forbidden if term in str(row.get("raw_text", "")).lower()],
            }
            for row in outputs
            if any(term in str(row.get("raw_text", "")).lower() for term in forbidden)
        ],
        "attributes": len(attr_rows),
        "attribute_two_citation_rate": 0.0 if not attr_rows else len(citation_rows) / len(attr_rows),
        "uncertain_observations": len(uncertain_rows),
        "uncertain_two_citation_rate": 0.0
        if not uncertain_rows
        else len(uncertain_citation_rows) / len(uncertain_rows),
        "candidate_items": len(attr_rows) + len(uncertain_rows),
        "citation_error_count": len(citation_errors),
        "citation_errors": citation_errors[:100],
        "null_task_attributes": len(null_attr_rows),
        "null_task_uncertain_observations": len(null_uncertain_rows),
        "null_task_candidate_items": len(null_attr_rows) + len(null_uncertain_rows),
        "null_nonempty_outputs": sum(
            1
            for row in outputs
            if "null_random_split" in str(row.get("comparison_type", ""))
            and isinstance(row.get("parsed_json"), Mapping)
            and (_json_attributes(row["parsed_json"]) or _json_uncertain_observations(row["parsed_json"]))
        ),
        "deterministic_repeat_matches": deterministic_matches,
        "deterministic_repeat_match_rate": None
        if not deterministic_matches
        else sum(1 for value in deterministic_matches.values() if value) / len(deterministic_matches),
        "canonical_concept_repeat_matches": canonical_repeat_matches,
        "canonical_concept_repeat_match_rate": None
        if not canonical_repeat_matches
        else sum(1 for value in canonical_repeat_matches.values() if value) / len(canonical_repeat_matches),
        "evidence_index_repeat_matches": evidence_repeat_matches,
        "evidence_index_repeat_match_rate": None
        if not evidence_repeat_matches
        else sum(1 for value in evidence_repeat_matches.values() if value) / len(evidence_repeat_matches),
        **reversal,
    }


def _counts(values: Sequence[object]) -> dict[object, int]:
    counts: dict[object, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _schema_error(parsed_json: Mapping[str, object]) -> str:
    required = {
        "shared_class_identity": str,
        "set_a_enriched_attributes": list,
        "set_b_enriched_attributes": list,
        "class_ambiguities": list,
        "uncertain_observations": list,
    }
    missing = [key for key in required if key not in parsed_json]
    wrong = [key for key, typ in required.items() if key in parsed_json and not isinstance(parsed_json[key], typ)]
    if missing or wrong:
        return f"missing={missing}; wrong_type={wrong}"
    return ""


def _citation_errors(row: Mapping[str, object], task: Mapping[str, object]) -> list[dict[str, object]]:
    valid_a = {f"A{image['index']}" for image in task["set_a_images"]}
    valid_b = {f"B{image['index']}" for image in task["set_b_images"]}
    valid_all = valid_a | valid_b
    errors: list[dict[str, object]] = []
    parsed_json = row.get("parsed_json")
    if not isinstance(parsed_json, Mapping):
        return errors
    for attr in _attributes_with_direction(parsed_json) + _uncertain_with_direction(parsed_json):
        for citation in attr["citations"]:
            if citation not in valid_all:
                errors.append({**_output_key(row), "citation": citation, "error": "unknown_image_index"})
        direction = attr["direction"]
        for citation in attr["evidence_a"]:
            if not str(citation).startswith("A") or citation not in valid_a:
                errors.append({**_output_key(row), "citation": citation, "direction": direction, "error": "bad_set_a_evidence"})
        for citation in attr["evidence_b"]:
            if not str(citation).startswith("B") or citation not in valid_b:
                errors.append({**_output_key(row), "citation": citation, "direction": direction, "error": "bad_set_b_evidence"})
    return errors


def _output_key(row: Mapping[str, object]) -> dict[str, object]:
    return {"task_id": row.get("task_id", ""), "repeat_index": row.get("repeat_index", "")}


def _repeat_consistency(
    by_task_rows: Mapping[str, Sequence[Mapping[str, object]]],
) -> tuple[dict[str, bool], dict[str, bool]]:
    canonical: dict[str, bool] = {}
    evidence: dict[str, bool] = {}
    for task_id, rows in by_task_rows.items():
        if len(rows) <= 1:
            continue
        concept_sets = []
        evidence_sets = []
        for row in sorted(rows, key=lambda item: int(item.get("repeat_index", 0))):
            parsed_json = row.get("parsed_json")
            if not isinstance(parsed_json, Mapping):
                continue
            concept_sets.append(_concept_set(parsed_json, include_evidence=False))
            evidence_sets.append(_concept_set(parsed_json, include_evidence=True))
        if len(concept_sets) > 1:
            canonical[task_id] = len({tuple(sorted(value)) for value in concept_sets}) == 1
        if len(evidence_sets) > 1:
            evidence[task_id] = len({tuple(sorted(value)) for value in evidence_sets}) == 1
    return canonical, evidence


def _reversal_metrics(outputs: Sequence[Mapping[str, object]]) -> dict[str, object]:
    repeat0 = {
        str(row.get("task_id", "")): row
        for row in outputs
        if int(row.get("repeat_index", 0)) == 0 and isinstance(row.get("parsed_json"), Mapping)
    }
    pair_rows = []
    matched = 0
    total = 0
    same_direction = 0
    for task_id, swap_row in repeat0.items():
        base_id = str(swap_row.get("swapped_of", ""))
        if not base_id or base_id not in repeat0:
            continue
        base_json = repeat0[base_id]["parsed_json"]
        swap_json = swap_row["parsed_json"]
        if not isinstance(base_json, Mapping) or not isinstance(swap_json, Mapping):
            continue
        base = {item for item in _concept_set(base_json, include_evidence=False) if item[0] in {"A", "B"}}
        swap = {item for item in _concept_set(swap_json, include_evidence=False) if item[0] in {"A", "B"}}
        expected = {(_opposite_direction(direction), phrase) for direction, phrase in base}
        recovered = len(expected & swap)
        retained = len(base & swap)
        matched += recovered
        total += len(expected)
        same_direction += retained
        pair_rows.append(
            {
                "task_id": base_id,
                "swapped_task_id": task_id,
                "base_concepts": len(base),
                "reversed_matches": recovered,
                "same_direction_matches": retained,
                "reversal_rate": None if not expected else recovered / len(expected),
            }
        )
    return {
        "ab_reversal_pairs": pair_rows,
        "ab_reversal_recovered_concepts": matched,
        "ab_reversal_base_concepts": total,
        "ab_reversal_consistency": None if total == 0 else matched / total,
        "ab_reversal_same_direction_leaks": same_direction,
    }


def _opposite_direction(direction: str) -> str:
    if direction == "A":
        return "B"
    if direction == "B":
        return "A"
    return direction


def _json_attributes(parsed_json: Mapping[str, object]) -> list[Mapping[str, object]]:
    attrs: list[Mapping[str, object]] = []
    for key in ("set_a_enriched_attributes", "set_b_enriched_attributes", "class_ambiguities"):
        value = parsed_json.get(key, [])
        if isinstance(value, list):
            attrs.extend(item for item in value if isinstance(item, Mapping))
    return attrs


def _json_uncertain_observations(parsed_json: Mapping[str, object]) -> list[Mapping[str, object]]:
    value = parsed_json.get("uncertain_observations", [])
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, Mapping)]


def _attributes_with_direction(parsed_json: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, direction in (("set_a_enriched_attributes", "A"), ("set_b_enriched_attributes", "B")):
        value = parsed_json.get(key, [])
        if not isinstance(value, list):
            continue
        for item in value:
            if not isinstance(item, Mapping):
                continue
            evidence_a = _citation_values(item, "evidence_a")
            evidence_b = _citation_values(item, "evidence_b")
            rows.append(
                {
                    "direction": direction,
                    "phrase": str(item.get("phrase", "")),
                    "evidence_a": evidence_a,
                    "evidence_b": evidence_b,
                    "citations": evidence_a + evidence_b + _citation_values(item, "evidence") + _citation_values(item, "evidence_images"),
                }
            )
    value = parsed_json.get("class_ambiguities", [])
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                citations = _citation_values(item, "evidence") + _citation_values(item, "evidence_images")
                rows.append(
                    {
                        "direction": "ambiguity",
                        "phrase": str(item.get("competing_class", "")),
                        "evidence_a": [citation for citation in citations if str(citation).startswith("A")],
                        "evidence_b": [citation for citation in citations if str(citation).startswith("B")],
                        "citations": citations,
                    }
                )
    return rows


def _uncertain_with_direction(parsed_json: Mapping[str, object]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for item in _json_uncertain_observations(parsed_json):
        raw_direction = str(item.get("direction", "")).lower()
        if "set_a" in raw_direction or raw_direction == "a":
            direction = "A"
        elif "set_b" in raw_direction or raw_direction == "b":
            direction = "B"
        else:
            direction = "uncertain"
        citations = _citation_values(item, "evidence") + _citation_values(item, "evidence_images")
        rows.append(
            {
                "direction": direction,
                "phrase": str(item.get("phrase", "")),
                "evidence_a": [citation for citation in citations if str(citation).startswith("A")],
                "evidence_b": [citation for citation in citations if str(citation).startswith("B")],
                "citations": citations,
            }
        )
    return rows


def _concept_set(parsed_json: Mapping[str, object], *, include_evidence: bool) -> set[tuple[object, ...]]:
    concepts = set()
    for attr in _attributes_with_direction(parsed_json):
        phrase = _normalize_phrase(str(attr["phrase"]))
        if not phrase:
            continue
        direction = str(attr["direction"])
        if include_evidence:
            concepts.add((direction, phrase, tuple(sorted(str(citation) for citation in attr["citations"]))))
        else:
            concepts.add((direction, phrase))
    return concepts


def _citation_values(attr: Mapping[str, object], key: str) -> list[str]:
    value = attr.get(key, [])
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _normalize_phrase(text: str) -> str:
    lowered = text.lower().strip()
    keep = []
    for ch in lowered:
        if ch.isalnum() or ch.isspace():
            keep.append(ch)
        else:
            keep.append(" ")
    return " ".join("".join(keep).split())


def _has_two_citations(attr: Mapping[str, object]) -> bool:
    citations: list[str] = []
    for key in ("evidence_a", "evidence_b", "evidence", "evidence_images"):
        value = attr.get(key, [])
        if isinstance(value, list):
            citations.extend(str(item) for item in value)
    return len({item for item in citations if item}) >= 2


def _torch_dtype(torch, dtype: str):
    value = str(dtype).lower()
    if value in ("bf16", "bfloat16"):
        return torch.bfloat16
    if value in ("fp16", "float16"):
        return torch.float16
    if value in ("fp32", "float32"):
        return torch.float32
    return "auto"


def _task_image_rows(tasks: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for task in tasks:
        for side in ("set_a_images", "set_b_images"):
            for image in task[side]:
                rows.append(
                    {
                        "task_id": task["task_id"],
                        "comparison_type": task["comparison_type"],
                        "region_id": task["region_id"],
                        "class_id": task["class_id"],
                        "class_name": task["class_name"],
                        "pilot_type": task["pilot_type"],
                        "deletion_seed": task.get("deletion_seed", ""),
                        "swapped_of": task.get("swapped_of", ""),
                        "set": "A" if side == "set_a_images" else "B",
                        **dict(image),
                    }
                )
    return rows


def _report(tasks: Sequence[Mapping[str, object]], metadata: Mapping[str, object]) -> str:
    lines = [
        "# Qwen3-VL Region Description Tasks",
        "",
        f"- artifact: `{metadata['artifact_id']}`",
        f"- tasks: `{len(tasks)}`",
        f"- run inference: `{metadata['summary']['run_inference']}`",
        f"- inference status: `{metadata['summary']['inference_status']}`",
        "",
        "The tasks are blinded set comparisons. DINOv2 region labels and causal outcomes are not included in the prompt text.",
        "",
    ]
    for task in tasks[:8]:
        lines.extend(
            [
                f"## {task['task_id']}",
                "",
                f"- class: `{task['class_name']}`",
                f"- pilot type in metadata: `{task['pilot_type']}`",
                f"- comparison: `{task['comparison_type']}`",
                f"- Set A images: `{len(task['set_a_images'])}`",
                f"- Set B images: `{len(task['set_b_images'])}`",
                "",
            ]
        )
    return "\n".join(lines)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(dict(row), sort_keys=True) + "\n")
