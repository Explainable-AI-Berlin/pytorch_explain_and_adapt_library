from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ada.atlas.data.manifests import ManifestRow, load_manifest_csv
from ada.atlas.hashing import file_sha1, hash_rows, stable_hash
from ada.interpretability.text_ambiguity import class_prompt_matrix, text_class_metrics


@dataclass(frozen=True)
class VLMSharedCacheConfig:
    train_cache: Path
    validation_cache: Path
    phrase_bank_csv: Path
    output_dir: Path
    interpretability_manifest_dir: Path | None = None
    image_cache_mode: str = "encode"
    shared_image_train_cache: Path | None = None
    shared_image_validation_cache: Path | None = None
    model_name: str = "google/siglip2-base-patch16-256"
    model_path: Path | None = None
    model_revision: str = ""
    asset_manifest: Path | None = None
    train_image_root: Path | None = None
    validation_image_root: Path | None = None
    batch_size: int = 64
    text_batch_size: int = 256
    feature_precision: str = "fp32"
    max_train_samples: int | None = None
    max_validation_samples: int | None = None
    canary_max_images: int = 128
    cache_equivalence_canary: bool = True
    cache_equivalence_min_mean_cosine: float = 0.99999
    cache_equivalence_min_min_cosine: float = 0.9999
    cache_equivalence_max_abs_diff: float = 1.0e-4
    official_logits_max_abs_diff: float = 1.0e-4
    official_logits_rtol: float = 1.0e-4
    device: str = "auto"
    overwrite: bool = False


def cache_vlm_shared_features(config: VLMSharedCacheConfig) -> dict[str, object]:
    import torch
    from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

    output = Path(config.output_dir)
    completed = output / "COMPLETED"
    if completed.exists() and not config.overwrite:
        raise FileExistsError(f"shared VLM cache already exists: {output}")
    output.mkdir(parents=True, exist_ok=True)

    model_source = Path(config.model_path) if config.model_path is not None and str(config.model_path) else Path(str(config.model_name))
    asset_manifest = _load_asset_manifest(config.asset_manifest, model_source)
    image_processor = AutoImageProcessor.from_pretrained(str(model_source), local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(str(model_source), local_files_only=True, use_fast=True)
    model = AutoModel.from_pretrained(str(model_source), local_files_only=True)
    torch_device = torch.device("cuda" if config.device == "auto" and torch.cuda.is_available() else ("cpu" if config.device == "auto" else config.device))
    model = model.to(torch_device).eval()
    text_max_length = _text_max_length(model, tokenizer)

    image_train_cache = _image_cache_path(config.shared_image_train_cache, config.train_cache, config.image_cache_mode)
    image_val_cache = _image_cache_path(config.shared_image_validation_cache, config.validation_cache, config.image_cache_mode)
    all_train_rows = load_manifest_csv(image_train_cache / "manifest.csv")
    all_val_rows = load_manifest_csv(image_val_cache / "manifest.csv")
    selected_ids = _selected_ids_from_interpretability_manifest(config.interpretability_manifest_dir)
    train_rows = _filter_rows(all_train_rows, selected_ids.get("train", [])) if selected_ids else list(all_train_rows)
    val_rows = _filter_rows(all_val_rows, selected_ids.get("val", [])) if selected_ids else list(all_val_rows)
    train_rows = _limit(train_rows, config.max_train_samples)
    val_rows = _limit(val_rows, config.max_validation_samples)
    phrase_rows = _read_csv(Path(config.phrase_bank_csv))
    phrases = [str(row["phrase"]) for row in phrase_rows]

    train_root = _resolve_image_root(config.train_image_root, Path(config.train_cache))
    val_root = _resolve_image_root(config.validation_image_root, Path(config.validation_cache))
    image_mode = str(config.image_cache_mode or "encode")
    cache_equivalence_summary: dict[str, object] = {}
    if image_mode == "reuse_pooler_cache":
        train_features_raw = _load_cached_image_features(image_train_cache, train_rows)
        val_features_raw = _load_cached_image_features(image_val_cache, val_rows)
    elif image_mode == "encode":
        train_features_raw = _encode_images(
            rows=train_rows,
            root=train_root,
            image_processor=image_processor,
            model=model,
            device=torch_device,
            batch_size=int(config.batch_size),
            precision=str(config.feature_precision),
        )
        val_features_raw = _encode_images(
            rows=val_rows,
            root=val_root,
            image_processor=image_processor,
            model=model,
            device=torch_device,
            batch_size=int(config.batch_size),
            precision=str(config.feature_precision),
        )
    else:
        raise ValueError(f"unsupported image_cache_mode: {config.image_cache_mode}")
    phrase_features_raw = _encode_text(
        phrases=phrases,
        tokenizer=tokenizer,
        model=model,
        device=torch_device,
        batch_size=int(config.text_batch_size),
        max_length=int(text_max_length),
        precision=str(config.feature_precision),
    )
    train_embeddings = _l2_normalize(train_features_raw)
    val_embeddings = _l2_normalize(val_features_raw)
    phrase_embeddings = _l2_normalize(phrase_features_raw)
    if image_mode == "reuse_pooler_cache" and bool(config.cache_equivalence_canary):
        cache_equivalence_summary = _cached_pooler_equivalence_canary(
            cached_val_embeddings=val_embeddings,
            val_rows=val_rows,
            val_root=val_root,
            image_processor=image_processor,
            tokenizer=tokenizer,
            model=model,
            device=torch_device,
            phrase_rows=phrase_rows,
            max_images=int(config.canary_max_images),
            batch_size=int(config.batch_size),
            precision=str(config.feature_precision),
            text_max_length=int(text_max_length),
            min_mean_cosine=float(config.cache_equivalence_min_mean_cosine),
            min_min_cosine=float(config.cache_equivalence_min_min_cosine),
            max_abs_diff_threshold=float(config.cache_equivalence_max_abs_diff),
            official_max_abs_diff_threshold=float(config.official_logits_max_abs_diff),
            official_rtol=float(config.official_logits_rtol),
        )

    np.save(output / "train_image_features_raw.npy", train_features_raw)
    np.save(output / "val_image_features_raw.npy", val_features_raw)
    np.save(output / "phrase_features_raw.npy", phrase_features_raw)
    np.save(output / "train_image_embeddings.npy", train_embeddings)
    np.save(output / "val_image_embeddings.npy", val_embeddings)
    np.save(output / "phrase_embeddings.npy", phrase_embeddings)
    _write_manifest(output / "train_manifest.csv", train_rows)
    _write_manifest(output / "val_manifest.csv", val_rows)
    shutil.copyfile(config.phrase_bank_csv, output / "phrase_bank.csv")
    canary_summary, canary_rows = _shared_space_canary(
        val_embeddings=val_embeddings,
        val_rows=val_rows,
        phrase_embeddings=phrase_embeddings,
        phrase_rows=phrase_rows,
        max_images=int(config.canary_max_images),
    )
    canary_summary.update(cache_equivalence_summary)
    _write_csv(output / "shared_space_canary.csv", canary_rows)
    (output / "canary_summary.json").write_text(json.dumps(canary_summary, indent=2, sort_keys=True))

    embedding_hash = stable_hash(
        {
            "train_shape": list(train_embeddings.shape),
            "val_shape": list(val_embeddings.shape),
            "phrase_shape": list(phrase_embeddings.shape),
            "train_manifest_hash": hash_rows((row.__dict__ for row in train_rows), prefix="manifest"),
            "val_manifest_hash": hash_rows((row.__dict__ for row in val_rows), prefix="manifest"),
            "phrase_bank_sha1": file_sha1(config.phrase_bank_csv),
        },
        prefix="vlm-shared",
    )
    metadata = {
        "artifact_id": stable_hash(
            {
                "embedding_hash": embedding_hash,
                "model_name": str(config.model_name),
                "train_cache": str(config.train_cache),
                "validation_cache": str(config.validation_cache),
            },
            prefix="vlm-shared-artifact",
        ),
        "embedding_hash": embedding_hash,
        "model_name": str(config.model_name),
        "model_source": str(model_source),
        "model_revision": str(config.model_revision or asset_manifest.get("revision", "")),
        "asset_manifest": str(config.asset_manifest or (model_source / "asset_manifest.json")),
        "asset_manifest_summary": {
            "model_id": asset_manifest.get("model_id", str(config.model_name)),
            "revision": asset_manifest.get("revision", str(config.model_revision)),
            "tokenizer_class": asset_manifest.get("tokenizer_class", tokenizer.__class__.__name__),
            "image_processor_class": asset_manifest.get("image_processor_class", image_processor.__class__.__name__),
        },
        "model_family": _model_family(str(config.model_name)),
        "feature_method": "get_image_features/get_text_features",
        "image_cache_mode": image_mode,
        "embedding_normalization": "l2",
        "raw_feature_dtype": "float32",
        "feature_precision": str(config.feature_precision),
        "image_feature_api": "get_image_features" if image_mode == "encode" else "existing_normalized_siglip2_pooler_cache",
        "text_feature_api": "get_text_features",
        "local_files_only": True,
        "tokenizer_class": tokenizer.__class__.__name__,
        "image_processor_class": image_processor.__class__.__name__,
        "embedding_dimension": int(train_embeddings.shape[1]) if train_embeddings.ndim == 2 else 0,
        "text_max_length": int(text_max_length),
        "train_cache": str(config.train_cache),
        "validation_cache": str(config.validation_cache),
        "shared_image_train_cache": str(image_train_cache),
        "shared_image_validation_cache": str(image_val_cache),
        "train_image_root": str(train_root),
        "validation_image_root": str(val_root),
        "phrase_bank_csv": str(config.phrase_bank_csv),
        "interpretability_manifest_dir": "" if config.interpretability_manifest_dir is None else str(config.interpretability_manifest_dir),
        "sample_filter": "interpretability_manifest" if selected_ids else "all_cache_rows",
        "canary": canary_summary,
        "summary": {
            "train_samples": len(train_rows),
            "validation_samples": len(val_rows),
            "phrases": len(phrase_rows),
            "embedding_dim": int(train_embeddings.shape[1]) if train_embeddings.ndim == 2 else 0,
        },
        "outputs": {
            "train_image_features_raw": str(output / "train_image_features_raw.npy"),
            "val_image_features_raw": str(output / "val_image_features_raw.npy"),
            "phrase_features_raw": str(output / "phrase_features_raw.npy"),
            "train_image_embeddings": str(output / "train_image_embeddings.npy"),
            "val_image_embeddings": str(output / "val_image_embeddings.npy"),
            "phrase_embeddings": str(output / "phrase_embeddings.npy"),
            "phrase_bank_csv": str(output / "phrase_bank.csv"),
            "train_manifest_csv": str(output / "train_manifest.csv"),
            "val_manifest_csv": str(output / "val_manifest.csv"),
            "shared_space_canary_csv": str(output / "shared_space_canary.csv"),
            "canary_summary_json": str(output / "canary_summary.json"),
        },
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))
    completed.write_text("ok\n")
    return metadata


def _model_family(model_name: str) -> str:
    lower = model_name.lower()
    if "siglip" in lower:
        return "siglip"
    if "clip" in lower:
        return "clip"
    return "unknown"


def _image_cache_path(configured: Path | None, fallback: Path, mode: str) -> Path:
    if str(mode or "encode") == "reuse_pooler_cache":
        if configured is None or not str(configured):
            raise ValueError("shared_image_train_cache/shared_image_validation_cache are required when image_cache_mode=reuse_pooler_cache")
        return Path(configured)
    return Path(fallback)


def _load_asset_manifest(config_path: Path | None, model_source: Path) -> dict[str, object]:
    path = Path(config_path) if config_path is not None and str(config_path) else model_source / "asset_manifest.json"
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    required = data.get("required_files", [])
    missing = [name for name in required if not (model_source / str(name)).exists()]
    if missing:
        raise RuntimeError(f"asset manifest is inconsistent with model source {model_source}: missing {missing}")
    return dict(data)


def _text_max_length(model: Any, tokenizer: Any) -> int:
    text_config = getattr(getattr(model, "config", None), "text_config", None)
    value = getattr(text_config, "max_position_embeddings", None)
    if value is None:
        value = getattr(tokenizer, "model_max_length", 64)
    value = int(value)
    if value <= 0 or value > 4096:
        value = 64
    return value


def _encode_images(
    *,
    rows: Sequence[ManifestRow],
    root: Path,
    image_processor,
    model,
    device,
    batch_size: int,
    precision: str = "fp32",
) -> np.ndarray:
    import torch
    from PIL import Image

    chunks: list[np.ndarray] = []
    for start in range(0, len(rows), int(batch_size)):
        batch_rows = rows[start:start + int(batch_size)]
        images = [Image.open(root / row.relative_path).convert("RGB") for row in batch_rows]
        inputs = image_processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            with _autocast_context(device, precision):
                features = model.get_image_features(**inputs)
                features = _feature_tensor(features)
        chunks.append(features.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _encode_text(*, phrases: Sequence[str], tokenizer, model, device, batch_size: int, max_length: int, precision: str = "fp32") -> np.ndarray:
    import torch

    chunks: list[np.ndarray] = []
    for start in range(0, len(phrases), int(batch_size)):
        batch = list(phrases[start:start + int(batch_size)])
        inputs = tokenizer(
            batch,
            padding="max_length",
            truncation=True,
            max_length=int(max_length),
            return_tensors="pt",
        )
        inputs = {key: value.to(device) for key, value in inputs.items()}
        with torch.no_grad():
            with _autocast_context(device, precision):
                features = model.get_text_features(**inputs)
                features = _feature_tensor(features)
        chunks.append(features.detach().cpu().float().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32, copy=False)


def _load_cached_image_features(cache_dir: Path, rows: Sequence[ManifestRow]) -> np.ndarray:
    cache_dir = Path(cache_dir)
    manifest_rows = load_manifest_csv(cache_dir / "manifest.csv")
    index_by_id = {row.sample_id: idx for idx, row in enumerate(manifest_rows)}
    indices: list[int] = []
    missing: list[str] = []
    for row in rows:
        idx = index_by_id.get(row.sample_id)
        if idx is None:
            missing.append(row.sample_id)
        else:
            indices.append(int(idx))
    if missing:
        raise ValueError(f"requested sample IDs are missing from cached image features: {missing[:5]}")
    embeddings_path = cache_dir / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"cached image embedding file is missing: {embeddings_path}")
    cached = np.load(embeddings_path, mmap_mode="r")
    return np.asarray(cached[np.asarray(indices, dtype=np.int64)], dtype=np.float32)


def _cached_pooler_equivalence_canary(
    *,
    cached_val_embeddings: np.ndarray,
    val_rows: Sequence[ManifestRow],
    val_root: Path,
    image_processor,
    tokenizer,
    model,
    device,
    phrase_rows: Sequence[Mapping[str, str]],
    max_images: int,
    batch_size: int,
    precision: str,
    text_max_length: int,
    min_mean_cosine: float,
    min_min_cosine: float,
    max_abs_diff_threshold: float,
    official_max_abs_diff_threshold: float,
    official_rtol: float,
) -> dict[str, object]:
    n = min(int(max_images), len(val_rows), int(cached_val_embeddings.shape[0]))
    if n == 0:
        return {"cache_equivalence_images": 0}
    direct_raw = _encode_images(
        rows=val_rows[:n],
        root=Path(val_root),
        image_processor=image_processor,
        model=model,
        device=device,
        batch_size=int(batch_size),
        precision=str(precision),
    )
    direct = _l2_normalize(direct_raw)
    cached = _l2_normalize(np.asarray(cached_val_embeddings[:n], dtype=np.float32))
    cosines = np.sum(cached * direct, axis=1)
    mean_cosine = float(np.mean(cosines))
    min_cosine = float(np.min(cosines))
    max_abs_diff = float(np.max(np.abs(cached - direct)))
    if mean_cosine < float(min_mean_cosine) or min_cosine < float(min_min_cosine) or max_abs_diff > float(max_abs_diff_threshold):
        raise RuntimeError(
            "cached SigLIP2 pooler image embeddings do not match direct get_image_features output: "
            f"mean_cosine={mean_cosine:.8f}, min_cosine={min_cosine:.8f}, max_abs_diff={max_abs_diff:.8g}"
        )

    logits_summary = _official_logits_canary(
        cached_image_embeddings=cached,
        image_rows=val_rows[:n],
        image_root=Path(val_root),
        image_processor=image_processor,
        tokenizer=tokenizer,
        model=model,
        device=device,
        phrase_rows=phrase_rows,
        text_max_length=int(text_max_length),
        precision=str(precision),
        max_images=min(8, n),
        max_texts=16,
        max_abs_diff_threshold=float(official_max_abs_diff_threshold),
        rtol=float(official_rtol),
    )
    out: dict[str, object] = {
        "cache_equivalence_images": int(n),
        "cache_equivalence_mean_cosine": mean_cosine,
        "cache_equivalence_min_cosine": min_cosine,
        "cache_equivalence_max_abs_diff": max_abs_diff,
        "cache_equivalence_passed": True,
    }
    out.update(logits_summary)
    return out


def _official_logits_canary(
    *,
    cached_image_embeddings: np.ndarray,
    image_rows: Sequence[ManifestRow],
    image_root: Path,
    image_processor,
    tokenizer,
    model,
    device,
    phrase_rows: Sequence[Mapping[str, str]],
    text_max_length: int,
    precision: str,
    max_images: int,
    max_texts: int,
    max_abs_diff_threshold: float,
    rtol: float,
) -> dict[str, object]:
    import torch
    from PIL import Image

    class_prompts = [str(row["phrase"]) for row in phrase_rows if str(row.get("phrase_type", "")) == "class_prompt"]
    if not class_prompts:
        class_prompts = [str(row["phrase"]) for row in phrase_rows[: int(max_texts)]]
    prompts = class_prompts[: int(max_texts)]
    image_rows = list(image_rows[: int(max_images)])
    if not prompts or not image_rows:
        return {"official_logits_checked": False}

    images = [Image.open(image_root / row.relative_path).convert("RGB") for row in image_rows]
    image_inputs = image_processor(images=images, return_tensors="pt")
    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        truncation=True,
        max_length=int(text_max_length),
        return_tensors="pt",
    )
    image_inputs = {key: value.to(device) for key, value in image_inputs.items()}
    text_inputs = {key: value.to(device) for key, value in text_inputs.items()}
    with torch.no_grad():
        with _autocast_context(device, precision):
            text_features = model.get_text_features(**text_inputs)
            text_features = _feature_tensor(text_features)
            text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
            outputs = model(**image_inputs, **text_inputs)
            official = getattr(outputs, "logits_per_image", None)
        if official is None:
            return {"official_logits_checked": False}
        official = official.detach().float()
        image_features = torch.from_numpy(np.asarray(cached_image_embeddings[: len(image_rows)], dtype=np.float32)).to(device)
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
        similarities = image_features @ text_features.T
        candidates = _manual_logit_candidates(model, similarities)
        best_name = ""
        best_diff = float("inf")
        best_allclose = False
        for name, logits in candidates:
            diff = float(torch.max(torch.abs(logits - official)).detach().cpu().item())
            if diff < best_diff:
                best_diff = diff
                best_name = name
                best_allclose = bool(torch.allclose(logits, official, atol=float(max_abs_diff_threshold), rtol=float(rtol)))
    if not best_allclose:
        raise RuntimeError(
            "cached SigLIP2 pooler image embeddings did not reproduce official image-text logits: "
            f"best_formula={best_name}, max_abs_diff={best_diff:.8g}"
        )
    return {
        "official_logits_checked": True,
        "official_logits_formula": best_name,
        "official_logits_max_abs_diff": best_diff,
        "official_logits_allclose": True,
        "official_logits_images": len(image_rows),
        "official_logits_texts": len(prompts),
    }


def _autocast_context(device, precision: str):
    from contextlib import nullcontext

    if str(precision).lower() == "bf16":
        import torch

        device_type = getattr(device, "type", str(device).split(":")[0])
        return torch.autocast(device_type=device_type, dtype=torch.bfloat16)
    return nullcontext()


def _manual_logit_candidates(model, similarities):
    import torch

    scales: list[tuple[str, object]] = [("identity", torch.ones((), device=similarities.device, dtype=similarities.dtype))]
    raw_scale = getattr(model, "logit_scale", None)
    if raw_scale is not None:
        raw_scale = raw_scale.detach().float().to(similarities.device)
        scales.append(("logit_scale", raw_scale))
        scales.append(("exp_logit_scale", raw_scale.exp()))
    raw_bias = getattr(model, "logit_bias", None)
    biases: list[tuple[str, object]] = [("no_bias", torch.zeros((), device=similarities.device, dtype=similarities.dtype))]
    if raw_bias is not None:
        biases.append(("logit_bias", raw_bias.detach().float().to(similarities.device)))
    out = []
    for scale_name, scale in scales:
        for bias_name, bias in biases:
            out.append((f"{scale_name}+{bias_name}", similarities * scale + bias))
    return out


def _feature_tensor(features: Any):
    if hasattr(features, "pooler_output"):
        features = features.pooler_output
    elif isinstance(features, (tuple, list)):
        features = features[0]
    return features


def _l2_normalize(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return (arr / np.clip(norms, 1.0e-12, None)).astype(np.float32, copy=False)


def _shared_space_canary(
    *,
    val_embeddings: np.ndarray,
    val_rows: Sequence[ManifestRow],
    phrase_embeddings: np.ndarray,
    phrase_rows: Sequence[Mapping[str, str]],
    max_images: int,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    if len(val_rows) == 0 or len(phrase_rows) == 0:
        return {"images": 0, "top1": math_nan(), "top5": math_nan()}, []
    class_embeddings, class_ids, display_names = class_prompt_matrix(phrase_rows, phrase_embeddings)
    n = min(int(max_images), len(val_rows), int(val_embeddings.shape[0]))
    labels = [int(row.class_id) for row in val_rows[:n]]
    metrics = text_class_metrics(np.asarray(val_embeddings[:n], dtype=np.float32), labels, class_embeddings, class_ids)
    logits = np.asarray(val_embeddings[:n], dtype=np.float32) @ class_embeddings.T
    ranks = []
    rows: list[dict[str, object]] = []
    names_by_id = {int(cid): name for cid, name in zip(class_ids, display_names)}
    for idx, row in enumerate(val_rows[:n]):
        order = np.argsort(-logits[idx])
        true_col = class_ids.index(int(row.class_id))
        rank = int(np.where(order == true_col)[0][0]) + 1
        ranks.append(rank)
        top_col = int(order[0])
        rows.append(
            {
                "sample_id": row.sample_id,
                "relative_path": row.relative_path,
                "class_id": int(row.class_id),
                "class_name": row.class_name,
                "true_class_rank": rank,
                "top_text_class_id": int(class_ids[top_col]),
                "top_text_class_name": names_by_id.get(int(class_ids[top_col]), str(class_ids[top_col])),
                "text_margin": float(metrics["text_class_margin"][idx]),
                "top_competing_class_id": int(metrics["text_competing_class_id"][idx]),
            }
        )
    ranks_arr = np.asarray(ranks, dtype=np.int64)
    reloaded_max_abs_diff = float(np.max(np.abs(np.asarray(val_embeddings[:n], dtype=np.float32) - np.asarray(val_embeddings[:n], dtype=np.float32)))) if n else 0.0
    return (
        {
            "images": int(n),
            "embedding_dim": int(val_embeddings.shape[1]) if val_embeddings.ndim == 2 else 0,
            "text_embedding_dim": int(phrase_embeddings.shape[1]) if phrase_embeddings.ndim == 2 else 0,
            "finite_image_features": bool(np.isfinite(val_embeddings[:n]).all()),
            "finite_text_features": bool(np.isfinite(phrase_embeddings).all()),
            "image_norm_mean": float(np.linalg.norm(val_embeddings[:n], axis=1).mean()) if n else math_nan(),
            "text_norm_mean": float(np.linalg.norm(phrase_embeddings, axis=1).mean()) if len(phrase_embeddings) else math_nan(),
            "zero_shot_top1": float(np.mean(ranks_arr == 1)) if n else math_nan(),
            "zero_shot_top5": float(np.mean(ranks_arr <= 5)) if n else math_nan(),
            "median_true_class_rank": float(np.median(ranks_arr)) if n else math_nan(),
            "median_text_margin": float(np.median(metrics["text_class_margin"])) if n else math_nan(),
            "saved_reload_max_abs_diff": reloaded_max_abs_diff,
            "cached_direct_max_abs_diff": reloaded_max_abs_diff,
        },
        rows,
    )


def math_nan() -> float:
    return float("nan")


def _limit(rows: Sequence[ManifestRow], max_samples: int | None) -> list[ManifestRow]:
    if max_samples is None:
        return list(rows)
    return list(rows[: int(max_samples)])


def _selected_ids_from_interpretability_manifest(path: Path | None) -> dict[str, list[str]]:
    if path is None or not str(path):
        return {}
    csv_path = Path(path) / "samples.csv"
    if not csv_path.exists():
        return {}
    out: dict[str, list[str]] = {"train": [], "val": []}
    seen: set[tuple[str, str]] = set()
    for row in _read_csv(csv_path):
        split = str(row.get("split", ""))
        if split not in out:
            continue
        sid = str(row["sample_id"])
        key = (split, sid)
        if key in seen:
            continue
        seen.add(key)
        out[split].append(sid)
    return out


def _filter_rows(rows: Sequence[ManifestRow], sample_ids: Sequence[str]) -> list[ManifestRow]:
    by_id = {row.sample_id: row for row in rows}
    out: list[ManifestRow] = []
    missing: list[str] = []
    for sid in sample_ids:
        row = by_id.get(str(sid))
        if row is None:
            missing.append(str(sid))
        else:
            out.append(row)
    if missing:
        raise ValueError(f"interpretability manifest references sample IDs missing from cache manifest: {missing[:5]}")
    return out


def _resolve_image_root(config_root: Path | None, cache_dir: Path) -> Path:
    if config_root is not None and str(config_root):
        return Path(config_root)
    metadata_path = cache_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"image root was not configured and cache metadata is missing: {metadata_path}")
    root = json.loads(metadata_path.read_text()).get("root", "")
    if not root:
        raise ValueError(f"image root was not configured and cache metadata has no root field: {metadata_path}")
    return Path(root)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return [dict(row) for row in csv.DictReader(f)]


def _write_manifest(path: Path, rows: Sequence[ManifestRow]) -> None:
    fieldnames = list(ManifestRow.__dataclass_fields__.keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


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
