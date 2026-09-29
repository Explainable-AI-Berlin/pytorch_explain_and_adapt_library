"""Generate presentation figures for SmoothDiff comparison experiments.

This script replaces the exploratory `smoothdiff copy.ipynb` workflow with a
repeatable command line entry point. It creates figures for:

1. Method comparison: vanilla gradients, SmoothGrad, and SmoothDiff.
2. Sample-size comparison for SmoothGrad, SmoothDiff, and optional manifold-aware variants.
3. Gaussian smoothing hyperparameter comparison.
4. Optional manifold-aware strength, guidance-scale, and inference-step comparisons using Stable Diffusion 3 image-to-image.

Example:
    python smoothdiff_presentation_figures.py \
        --image assets/car.jpg \
        --output-dir presentation_figures \
        --device cuda
"""

from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms

from smoothdiff import replace_nonlinear_layers, set_smoothdiff_layer_mode

try:
    import cmcrameri.cm as cmc

    ATTRIBUTION_CMAP = cmc.batlow
except ImportError:
    ATTRIBUTION_CMAP = plt.cm.magma

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "presentation_figures"
DEFAULT_IMAGE = SCRIPT_DIR / "assets" / "car.jpg"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class AttributionRun:
    method: str
    attribution: np.ndarray
    seconds: float
    n_samples: int | None = None
    params: dict[str, float | int | str] = field(default_factory=dict)


def parse_number_list(
    raw: str, cast: Callable[[str], float | int]
) -> list[float | int]:
    values = []
    for value in raw.split(","):
        value = value.strip()
        if value:
            values.append(cast(value))
    if not values:
        raise argparse.ArgumentTypeError("Expected at least one comma-separated value.")
    return values


def progress(values: Iterable, description: str) -> Iterable:
    if tqdm is None:
        return values
    return tqdm(values, desc=description, leave=False)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def seed_offset(seed: int, *parts: object) -> int:
    text = "::".join(str(part) for part in parts)
    return seed + sum((idx + 1) * ord(char) for idx, char in enumerate(text)) % 100_000


def resolve_path(path: Path) -> Path:
    if path.exists():
        return path
    candidate = SCRIPT_DIR / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find path: {path}")


def image_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def image_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
    return image_preprocess()(image.convert("RGB")).unsqueeze(0).to(device)


def tensor_to_display_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = tensor.detach().cpu().squeeze(0).clone()
    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
    tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0.0, 1.0)
    return tensor.permute(1, 2, 0).numpy()


def load_model(
    model_name: str, weights_mode: str, device: torch.device
) -> tuple[torch.nn.Module, list[str] | None]:
    labels = None
    if model_name == "vgg16":
        weights = None
        if weights_mode == "imagenet":
            weights = torchvision.models.VGG16_Weights.IMAGENET1K_V1
            labels = list(weights.meta.get("categories", []))
        model = torchvision.models.vgg16(weights=weights)
    elif model_name == "resnet18":
        weights = None
        if weights_mode == "imagenet":
            weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1
            labels = list(weights.meta.get("categories", []))
        model = torchvision.models.resnet18(weights=weights)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model.to(device).eval(), labels


def target_score(
    model: torch.nn.Module, inputs: torch.Tensor, target_class: int
) -> torch.Tensor:
    logits = model(inputs)
    return logits[:, target_class].sum()


def choose_target_class(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    requested_class: int | None,
) -> tuple[int, float]:
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        if requested_class is None:
            target_class = int(probabilities.argmax(dim=1).item())
        else:
            target_class = requested_class
        confidence = float(probabilities[0, target_class].item())
    return target_class, confidence


def gradient_for_input(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    model.zero_grad(set_to_none=True)
    inputs_for_grad = inputs.clone().detach().requires_grad_(True)
    score = target_score(model, inputs_for_grad, target_class)
    score.backward()
    return inputs_for_grad.grad.detach().cpu().numpy()


def vanilla_gradient(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
) -> np.ndarray:
    return gradient_for_input(model, inputs, target_class)


def smoothgrad_gaussian(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    n_samples: int,
    std: float,
    batch_size: int,
) -> np.ndarray:
    grads = []
    remaining = n_samples

    for _ in progress(range((n_samples + batch_size - 1) // batch_size), "SmoothGrad"):
        this_batch = min(batch_size, remaining)
        remaining -= this_batch

        batch = inputs.repeat(this_batch, 1, 1, 1)
        noisy = batch + torch.randn_like(batch) * std
        noisy = noisy.detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        target_score(model, noisy, target_class).backward()
        grads.append(noisy.grad.detach().cpu().numpy())

    return np.mean(np.concatenate(grads, axis=0), axis=0, keepdims=True)


def smoothdiff_gaussian(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    n_samples: int,
    std: float,
) -> np.ndarray:
    smooth_model = replace_nonlinear_layers(model).to(inputs.device).eval()

    set_smoothdiff_layer_mode(smooth_model, collect_stats=True, smooth_backward=False)
    with torch.no_grad():
        for _ in progress(range(n_samples), "SmoothDiff"):
            smooth_model(inputs + torch.randn_like(inputs) * std)

    set_smoothdiff_layer_mode(smooth_model, collect_stats=False, smooth_backward=True)
    attribution = gradient_for_input(smooth_model, inputs, target_class)

    del smooth_model
    if inputs.device.type == "cuda":
        torch.cuda.empty_cache()
    return attribution


class ManifoldSampler:
    def __init__(
        self,
        model_id: str,
        device: torch.device,
        prompt: str,
        strength: float,
        guidance_scale: float,
        num_inference_steps: int,
        image_size: int,
    ) -> None:
        try:
            from diffusers import StableDiffusion3Img2ImgPipeline
        except ImportError as exc:
            raise RuntimeError(
                "Manifold-aware sampling requires diffusers and transformers. "
                "Install them or run without --include-manifold."
            ) from exc

        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.pipe = StableDiffusion3Img2ImgPipeline.from_pretrained(
            model_id, torch_dtype=dtype
        )
        self.pipe = self.pipe.to(device)
        self.device = device
        self.prompt = prompt
        self.strength = strength
        self.guidance_scale = guidance_scale
        self.num_inference_steps = num_inference_steps
        self.image_size = image_size

    def sample(
        self,
        image: Image.Image,
        strength: float | None = None,
        guidance_scale: float | None = None,
        num_inference_steps: int | None = None,
        seed: int | None = None,
    ) -> Image.Image:
        generator = None
        if seed is not None:
            generator = torch.Generator(device=self.device).manual_seed(seed)

        output = self.pipe(
            prompt=self.prompt,
            image=image.convert("RGB"),
            strength=self.strength if strength is None else strength,
            height=self.image_size,
            width=self.image_size,
            num_inference_steps=(
                self.num_inference_steps
                if num_inference_steps is None
                else num_inference_steps
            ),
            guidance_scale=(
                self.guidance_scale if guidance_scale is None else guidance_scale
            ),
            generator=generator,
        )
        return output.images[0]


def manifold_smoothgrad(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    target_class: int,
    sampler: ManifoldSampler,
    n_samples: int,
    strength: float,
    seed: int,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
) -> np.ndarray:
    grads = []
    for idx in progress(range(n_samples), "Manifold SmoothGrad"):
        sampled_image = sampler.sample(
            image,
            strength=strength,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed + idx,
        )
        sampled = image_to_tensor(sampled_image, device).detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        target_score(model, sampled, target_class).backward()
        grads.append(sampled.grad.detach().cpu().numpy())

    return np.mean(np.concatenate(grads, axis=0), axis=0, keepdims=True)


def manifold_smoothdiff(
    model: torch.nn.Module,
    image: Image.Image,
    inputs: torch.Tensor,
    device: torch.device,
    target_class: int,
    sampler: ManifoldSampler,
    n_samples: int,
    strength: float,
    seed: int,
    guidance_scale: float | None = None,
    num_inference_steps: int | None = None,
) -> np.ndarray:
    smooth_model = replace_nonlinear_layers(model).to(device).eval()

    set_smoothdiff_layer_mode(smooth_model, collect_stats=True, smooth_backward=False)
    with torch.no_grad():
        for idx in progress(range(n_samples), "Manifold SmoothDiff"):
            sampled_image = sampler.sample(
                image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed + idx,
            )
            sampled = image_to_tensor(sampled_image, device)
            smooth_model(sampled)

    set_smoothdiff_layer_mode(smooth_model, collect_stats=False, smooth_backward=True)
    attribution = gradient_for_input(smooth_model, inputs, target_class)

    del smooth_model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return attribution


def timed_run(
    method: str,
    fn: Callable[[], np.ndarray],
    n_samples: int | None,
    **params: float | int | str,
) -> AttributionRun:
    start = time.perf_counter()
    attribution = fn()
    seconds = time.perf_counter() - start
    return AttributionRun(
        method=method,
        attribution=attribution,
        seconds=seconds,
        n_samples=n_samples,
        params=params,
    )


def pooled_attribution(attribution: np.ndarray) -> np.ndarray:
    attribution = np.asarray(attribution)
    if attribution.ndim == 4:
        attribution = attribution[0]
    if attribution.shape[0] != 3:
        raise ValueError(
            f"Expected attribution with 3 channels, got shape {attribution.shape}"
        )
    return np.linalg.norm(attribution, axis=0)


def normalized_heatmap(attribution: np.ndarray, percentile: float = 99.5) -> np.ndarray:
    pooled = pooled_attribution(attribution)
    scale = np.percentile(np.abs(pooled), percentile)
    if scale <= 1e-12:
        return np.zeros_like(pooled)
    return np.clip(pooled / scale, 0.0, 1.0)


def add_heatmap(ax: plt.Axes, attribution: np.ndarray) -> None:
    ax.imshow(
        normalized_heatmap(attribution), cmap=ATTRIBUTION_CMAP, vmin=0.0, vmax=1.0
    )
    ax.set_xticks([])
    ax.set_yticks([])


def style_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, save_pdf: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    if save_pdf:
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def method_subtitle(run: AttributionRun) -> str:
    parts = []
    if run.n_samples is not None:
        parts.append(f"n={run.n_samples}")
    for key in ("std", "strength", "guidance"):
        if key in run.params:
            parts.append(f"{key}={run.params[key]}")
    if parts:
        return f"{run.method}\n" + ", ".join(parts)
    return run.method


def plot_method_comparison(
    input_image: np.ndarray,
    runs: list[AttributionRun],
    output_dir: Path,
    title: str,
    save_pdf: bool,
) -> None:
    ncols = len(runs) + 1
    fig, axes = plt.subplots(
        1, ncols, figsize=(3.0 * ncols, 3.4), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    axes[0].imshow(input_image)
    axes[0].set_title("Input", fontsize=11, fontweight="bold")
    style_axes(axes[0])

    for ax, run in zip(axes[1:], runs):
        add_heatmap(ax, run.attribution)
        ax.set_title(method_subtitle(run), fontsize=10)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "method_comparison", save_pdf)


def plot_attribution_grid(
    rows: list[str],
    columns: list[str],
    grid: list[list[np.ndarray]],
    output_dir: Path,
    stem: str,
    title: str,
    save_pdf: bool,
) -> None:
    fig_width = max(4.0, 2.25 * len(columns))
    fig_height = max(3.0, 2.25 * len(rows))
    fig, axes = plt.subplots(
        len(rows),
        len(columns),
        figsize=(fig_width, fig_height),
        squeeze=False,
        constrained_layout=True,
    )

    for row_idx, row_name in enumerate(rows):
        for col_idx, col_name in enumerate(columns):
            ax = axes[row_idx, col_idx]
            add_heatmap(ax, grid[row_idx][col_idx])
            if row_idx == 0:
                ax.set_title(col_name, fontsize=10)
            if col_idx == 0:
                ax.set_ylabel(row_name, fontsize=10, fontweight="bold")

    fig.suptitle(title, fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, stem, save_pdf)


def attribution_metrics(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    run: AttributionRun,
) -> dict[str, float | int | str]:
    heat = normalized_heatmap(run.attribution)
    flat = heat.reshape(-1)
    total_energy = float(np.sum(flat) + 1e-12)
    top5_count = max(1, int(0.05 * flat.size))
    energy_top5 = float(
        np.sum(np.partition(flat, -top5_count)[-top5_count:]) / total_energy
    )

    dy = np.abs(np.diff(heat, axis=0)).mean()
    dx = np.abs(np.diff(heat, axis=1)).mean()
    roughness = float(dx + dy)

    top20_count = max(1, int(0.20 * flat.size))
    threshold = np.partition(flat, -top20_count)[-top20_count]
    mask = torch.from_numpy(heat >= threshold).to(inputs.device)
    mask = mask.unsqueeze(0).unsqueeze(0).expand_as(inputs)

    with torch.no_grad():
        base_score = float(model(inputs)[0, target_class].item())
        ablated = torch.where(mask, torch.zeros_like(inputs), inputs)
        ablated_score = float(model(ablated)[0, target_class].item())

    return {
        "method": run.method,
        "n_samples": "" if run.n_samples is None else run.n_samples,
        "runtime_seconds": run.seconds,
        "score_drop_top20": base_score - ablated_score,
        "energy_top5_fraction": energy_top5,
        "roughness": roughness,
        **run.params,
    }


def write_csv(rows: list[dict[str, float | int | str]], path: Path) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_metric_summary(
    metrics: list[dict[str, float | int | str]],
    output_dir: Path,
    save_pdf: bool,
) -> None:
    methods = [str(row["method"]) for row in metrics]
    panels = [
        ("score_drop_top20", "Target-score drop\nhigher is better"),
        ("energy_top5_fraction", "Energy in top 5%\nhigher is sharper"),
        ("roughness", "Heatmap roughness\nlower is smoother"),
        ("runtime_seconds", "Runtime seconds\nlower is faster"),
    ]
    colors = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2"]

    fig, axes = plt.subplots(
        1, len(panels), figsize=(4.0 * len(panels), 3.8), constrained_layout=True
    )
    axes = np.atleast_1d(axes)

    for ax, (key, label) in zip(axes, panels):
        values = [float(row.get(key, np.nan)) for row in metrics]
        ax.bar(
            methods,
            values,
            color=colors[: len(methods)],
            edgecolor="#222222",
            linewidth=0.6,
        )
        ax.set_title(label, fontsize=10)
        ax.tick_params(axis="x", rotation=35)
        ax.grid(axis="y", alpha=0.25)

    fig.suptitle("Method Metrics", fontsize=13, fontweight="bold")
    save_figure(fig, output_dir, "method_metrics", save_pdf)


def run_method_comparison(
    model: torch.nn.Module,
    image: Image.Image,
    inputs: torch.Tensor,
    device: torch.device,
    target_class: int,
    args: argparse.Namespace,
    sampler: ManifoldSampler | None,
) -> list[AttributionRun]:
    runs = [
        timed_run(
            "Vanilla Grad",
            lambda: vanilla_gradient(model, inputs, target_class),
            n_samples=None,
        )
    ]

    set_seed(seed_offset(args.seed, "method", "smoothgrad"))
    runs.append(
        timed_run(
            "SmoothGrad",
            lambda: smoothgrad_gaussian(
                model,
                inputs,
                target_class,
                n_samples=args.method_samples,
                std=args.std,
                batch_size=args.batch_size,
            ),
            n_samples=args.method_samples,
            std=args.std,
        )
    )

    set_seed(seed_offset(args.seed, "method", "smoothdiff"))
    runs.append(
        timed_run(
            "SmoothDiff",
            lambda: smoothdiff_gaussian(
                model,
                inputs,
                target_class,
                n_samples=args.method_samples,
                std=args.std,
            ),
            n_samples=args.method_samples,
            std=args.std,
        )
    )

    if args.include_manifold:
        if sampler is None:
            raise ValueError(
                "Expected a ManifoldSampler when --include-manifold is set."
            )
        manifold_seed = seed_offset(args.seed, "method", "manifold")

        runs.append(
            timed_run(
                "Manifold SmoothGrad",
                lambda: manifold_smoothgrad(
                    model,
                    image,
                    device,
                    target_class,
                    sampler,
                    n_samples=args.manifold_samples,
                    strength=args.strength,
                    seed=manifold_seed,
                ),
                n_samples=args.manifold_samples,
                strength=args.strength,
                guidance=args.guidance_scale,
            )
        )

        runs.append(
            timed_run(
                "Manifold SmoothDiff",
                lambda: manifold_smoothdiff(
                    model,
                    image,
                    inputs,
                    device,
                    target_class,
                    sampler,
                    n_samples=args.manifold_samples,
                    strength=args.strength,
                    seed=manifold_seed + 10_000,
                ),
                n_samples=args.manifold_samples,
                strength=args.strength,
                guidance=args.guidance_scale,
            )
        )

    return runs


def run_sample_size_grid(
    model: torch.nn.Module,
    image: Image.Image,
    inputs: torch.Tensor,
    device: torch.device,
    target_class: int,
    args: argparse.Namespace,
    sampler: ManifoldSampler | None,
) -> tuple[list[str], list[str], list[list[np.ndarray]]]:
    sample_sizes = parse_number_list(args.sample_sizes, int)
    rows = ["SmoothGrad", "SmoothDiff"]
    if args.include_manifold:
        if sampler is None:
            raise ValueError(
                "Expected a ManifoldSampler when --include-manifold is set."
            )
        rows.extend(["Manifold SmoothGrad", "Manifold SmoothDiff"])

    columns = [f"n={value}" for value in sample_sizes]
    grid = []

    for method in rows:
        row = []
        for n_samples in sample_sizes:
            set_seed(seed_offset(args.seed, "sample-size", method, n_samples))
            if method == "SmoothGrad":
                attr = smoothgrad_gaussian(
                    model,
                    inputs,
                    target_class,
                    n_samples=int(n_samples),
                    std=args.std,
                    batch_size=args.batch_size,
                )
            elif method == "SmoothDiff":
                attr = smoothdiff_gaussian(
                    model,
                    inputs,
                    target_class,
                    n_samples=int(n_samples),
                    std=args.std,
                )
            elif method == "Manifold SmoothGrad":
                attr = manifold_smoothgrad(
                    model,
                    image,
                    device,
                    target_class,
                    sampler,
                    n_samples=int(n_samples),
                    strength=args.strength,
                    seed=seed_offset(
                        args.seed, "sample-size-sampler", method, n_samples
                    ),
                )
            elif method == "Manifold SmoothDiff":
                attr = manifold_smoothdiff(
                    model,
                    image,
                    inputs,
                    device,
                    target_class,
                    sampler,
                    n_samples=int(n_samples),
                    strength=args.strength,
                    seed=seed_offset(
                        args.seed, "sample-size-sampler", method, n_samples
                    ),
                )
            else:
                raise ValueError(f"Unsupported sample-size method: {method}")
            row.append(attr)
        grid.append(row)

    return rows, columns, grid


def run_std_grid(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    target_class: int,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[list[np.ndarray]]]:
    std_values = parse_number_list(args.std_values, float)
    rows = ["SmoothGrad", "SmoothDiff"]
    columns = [f"std={value:g}" for value in std_values]
    grid = []

    for method in rows:
        row = []
        for std in std_values:
            set_seed(seed_offset(args.seed, "std", method, std))
            if method == "SmoothGrad":
                attr = smoothgrad_gaussian(
                    model,
                    inputs,
                    target_class,
                    n_samples=args.hyperparam_samples,
                    std=float(std),
                    batch_size=args.batch_size,
                )
            else:
                attr = smoothdiff_gaussian(
                    model,
                    inputs,
                    target_class,
                    n_samples=args.hyperparam_samples,
                    std=float(std),
                )
            row.append(attr)
        grid.append(row)

    return rows, columns, grid


def run_manifold_strength_grid(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    target_class: int,
    sampler: ManifoldSampler,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[list[np.ndarray]]]:
    strength_values = parse_number_list(args.strength_values, float)
    rows = ["Manifold SmoothGrad"]
    columns = [f"strength={value:g}" for value in strength_values]
    grid = [[]]

    for strength in strength_values:
        set_seed(seed_offset(args.seed, "manifold-strength", strength))
        attr = manifold_smoothgrad(
            model,
            image,
            device,
            target_class,
            sampler,
            n_samples=args.manifold_samples,
            strength=float(strength),
            seed=seed_offset(args.seed, "manifold-strength-sampler", strength),
        )
        grid[0].append(attr)

    return rows, columns, grid


def run_manifold_guidance_grid(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    target_class: int,
    sampler: ManifoldSampler,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[list[np.ndarray]]]:
    guidance_values = parse_number_list(args.guidance_values, float)
    rows = ["Manifold SmoothGrad"]
    columns = [f"guidance={value:g}" for value in guidance_values]
    grid = [[]]

    for guidance_scale in guidance_values:
        set_seed(seed_offset(args.seed, "manifold-guidance", guidance_scale))
        attr = manifold_smoothgrad(
            model,
            image,
            device,
            target_class,
            sampler,
            n_samples=args.manifold_samples,
            strength=args.strength,
            seed=seed_offset(args.seed, "manifold-guidance-sampler", guidance_scale),
            guidance_scale=float(guidance_scale),
        )
        grid[0].append(attr)

    return rows, columns, grid


def run_manifold_inference_steps_grid(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
    target_class: int,
    sampler: ManifoldSampler,
    args: argparse.Namespace,
) -> tuple[list[str], list[str], list[list[np.ndarray]]]:
    step_values = parse_number_list(args.num_inference_step_values, int)
    rows = ["Manifold SmoothGrad"]
    columns = [f"steps={value}" for value in step_values]
    grid = [[]]

    for num_inference_steps in step_values:
        set_seed(
            seed_offset(args.seed, "manifold-inference-steps", num_inference_steps)
        )
        attr = manifold_smoothgrad(
            model,
            image,
            device,
            target_class,
            sampler,
            n_samples=args.manifold_samples,
            strength=args.strength,
            seed=seed_offset(
                args.seed, "manifold-inference-steps-sampler", num_inference_steps
            ),
            guidance_scale=args.guidance_scale,
            num_inference_steps=int(num_inference_steps),
        )
        grid[0].append(attr)

    return rows, columns, grid


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create presentation-ready SmoothDiff comparison figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image", type=Path, default=DEFAULT_IMAGE, help="Input image path."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for figures and CSVs.",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--model",
        choices=["vgg16", "resnet18"],
        default="vgg16",
        help="Torchvision classifier.",
    )
    parser.add_argument(
        "--weights",
        choices=["imagenet", "random"],
        default="imagenet",
        help="Classifier weights.",
    )
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help="ImageNet class index to explain. Defaults to top prediction.",
    )
    parser.add_argument(
        "--std", type=float, default=0.5, help="Gaussian smoothing standard deviation."
    )
    parser.add_argument(
        "--method-samples",
        type=int,
        default=32,
        help="Samples for the main method comparison.",
    )
    parser.add_argument(
        "--sample-sizes", default="1,2,4,8,16,32", help="Comma-separated sample sizes."
    )
    parser.add_argument(
        "--std-values",
        default="0.05,0.1,0.25,0.5,1.0",
        help="Comma-separated Gaussian std values.",
    )
    parser.add_argument(
        "--hyperparam-samples",
        type=int,
        default=16,
        help="Samples per hyperparameter setting.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for SmoothGrad gradient averaging.",
    )
    parser.add_argument(
        "--no-metrics", action="store_true", help="Skip metric CSV and bar chart."
    )
    parser.add_argument(
        "--save-pdf", action="store_true", help="Also save PDF versions of each figure."
    )

    parser.add_argument(
        "--include-manifold",
        action="store_true",
        help="Include Stable Diffusion 3 manifold-aware variants.",
    )
    parser.add_argument(
        "--sd3-model-id",
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Diffusers SD3 img2img model id.",
    )
    parser.add_argument(
        "--prompt", default="", help="Prompt for SD3 image-to-image sampling."
    )
    parser.add_argument(
        "--manifold-samples",
        type=int,
        default=8,
        help="Samples for manifold-aware methods.",
    )
    parser.add_argument(
        "--strength", type=float, default=0.3, help="SD3 img2img strength."
    )
    parser.add_argument(
        "--strength-values",
        default="0.1,0.2,0.3,0.4,0.5",
        help="Comma-separated SD3 strength values.",
    )
    parser.add_argument(
        "--guidance-scale", type=float, default=4.0, help="SD3 guidance scale."
    )
    parser.add_argument(
        "--guidance-values",
        default="0.0,1.0,2.0,4.0,7.0,10.0",
        help="Comma-separated SD3 guidance scale values.",
    )
    parser.add_argument(
        "--num-inference-steps", type=int, default=28, help="SD3 inference steps."
    )
    parser.add_argument(
        "--num-inference-step-values",
        default="8,16,28,40,60",
        help="Comma-separated SD3 inference-step values.",
    )
    parser.add_argument(
        "--sd-image-size", type=int, default=512, help="SD3 output image size."
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    image_path = resolve_path(args.image)
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = SCRIPT_DIR / output_dir

    device = torch.device(args.device)
    image = Image.open(image_path).convert("RGB")
    inputs = image_to_tensor(image, device)
    display_image = tensor_to_display_image(inputs)

    model, labels = load_model(args.model, args.weights, device)
    target_class, confidence = choose_target_class(model, inputs, args.target_class)
    target_label = str(target_class)
    if labels and 0 <= target_class < len(labels):
        target_label = labels[target_class]

    prompt = args.prompt or f"a photo of a {target_label.replace('_', ' ')}"
    figure_title = f"{args.model} target: {target_label} ({confidence:.2%})"

    print(f"Input image: {image_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device}")
    print(f"Target class: {target_class} ({target_label}), confidence={confidence:.4f}")

    sampler = None
    if args.include_manifold:
        sampler = ManifoldSampler(
            model_id=args.sd3_model_id,
            device=device,
            prompt=prompt,
            strength=args.strength,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.num_inference_steps,
            image_size=args.sd_image_size,
        )

    method_runs = run_method_comparison(
        model=model,
        image=image,
        inputs=inputs,
        device=device,
        target_class=target_class,
        args=args,
        sampler=sampler,
    )
    plot_method_comparison(
        display_image,
        method_runs,
        output_dir,
        title=f"Method Comparison - {figure_title}",
        save_pdf=args.save_pdf,
    )

    sample_rows, sample_columns, sample_grid = run_sample_size_grid(
        model,
        image,
        inputs,
        device,
        target_class,
        args,
        sampler,
    )
    sample_size_title = f"Sample-Size Comparison - std={args.std:g}"
    if args.include_manifold:
        sample_size_title += f", manifold strength={args.strength:g}"
    plot_attribution_grid(
        sample_rows,
        sample_columns,
        sample_grid,
        output_dir,
        stem="sample_size_comparison",
        title=sample_size_title,
        save_pdf=args.save_pdf,
    )

    std_rows, std_columns, std_grid = run_std_grid(model, inputs, target_class, args)
    plot_attribution_grid(
        std_rows,
        std_columns,
        std_grid,
        output_dir,
        stem="gaussian_std_comparison",
        title=f"Gaussian Smoothing Hyperparameter - n={args.hyperparam_samples}",
        save_pdf=args.save_pdf,
    )

    if args.include_manifold:
        if sampler is None:
            raise ValueError(
                "Expected a ManifoldSampler when --include-manifold is set."
            )
        strength_rows, strength_columns, strength_grid = run_manifold_strength_grid(
            model,
            image,
            device,
            target_class,
            sampler,
            args,
        )
        plot_attribution_grid(
            strength_rows,
            strength_columns,
            strength_grid,
            output_dir,
            stem="manifold_strength_comparison",
            title=f"Manifold-Aware Strength Hyperparameter - n={args.manifold_samples}",
            save_pdf=args.save_pdf,
        )
        guidance_rows, guidance_columns, guidance_grid = run_manifold_guidance_grid(
            model,
            image,
            device,
            target_class,
            sampler,
            args,
        )
        plot_attribution_grid(
            guidance_rows,
            guidance_columns,
            guidance_grid,
            output_dir,
            stem="manifold_guidance_comparison",
            title=(
                "Manifold-Aware Guidance Scale "
                f"- n={args.manifold_samples}, strength={args.strength:g}"
            ),
            save_pdf=args.save_pdf,
        )
        steps_rows, steps_columns, steps_grid = run_manifold_inference_steps_grid(
            model,
            image,
            device,
            target_class,
            sampler,
            args,
        )
        plot_attribution_grid(
            steps_rows,
            steps_columns,
            steps_grid,
            output_dir,
            stem="manifold_inference_steps_comparison",
            title=(
                "Manifold-Aware Inference Steps "
                f"- n={args.manifold_samples}, strength={args.strength:g}, "
                f"guidance={args.guidance_scale:g}"
            ),
            save_pdf=args.save_pdf,
        )

    if not args.no_metrics:
        metric_rows = [
            attribution_metrics(model, inputs, target_class, run) for run in method_runs
        ]
        write_csv(metric_rows, output_dir / "method_metrics.csv")
        plot_metric_summary(metric_rows, output_dir, args.save_pdf)

    print("Generated figures:")
    for path in sorted(output_dir.glob("*.png")):
        print(f"  {path}")


if __name__ == "__main__":
    main()
