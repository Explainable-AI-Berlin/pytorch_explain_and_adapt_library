"""Concrete ``PealDataset`` implementations for tabular and image data.

``SymbolicDataset`` reads a ``data.csv`` of tabular features; ``ImageDataset``
is the shared base of the image datasets and implements the pieces
explainers and adaptors call back into: contrastive collages of
factual/counterfactual pairs, serialization of generated counterfactual
datasets to disk and DINO-based generator metrics (FID, LPIPS, Mahalanobis
outlier score). ``Image2MixedDataset`` loads images with multi-attribute CSV
labels (CelebA style, optional masks/hints, groups, text descriptions) and
``Image2ClassDataset`` loads ImageFolder-style class directories. Datasets
toggle what ``__getitem__`` returns via ``enable_*``/``disable_*`` methods.
"""

import torch
import random
import os
import copy
import numpy as np
import matplotlib

from torchvision.transforms import ToTensor
from PIL import Image
from pathlib import Path
from matplotlib import pyplot as plt
import matplotlib.cm as cm

from peal.data.interfaces import PealDataset
from peal.data.dataset_utils import parse_csv
from peal.global_utils import (
    embed_numberstring,
    high_contrast_heatmap,
    DINOEvaluator,
    generate_ssim_overlay,
)
from peal.generators.interfaces import Generator
from peal.log import get_logger

_log = get_logger(__name__)


# No matplotlib.use("Agg") here: forcing the backend at import time broke
# inline plots in every notebook that imports PEAL, and matplotlib already
# falls back to Agg on machines without a display.

from typing import Union


class SymbolicDataset(PealDataset):
    """
    Tabular (symbolic) dataset read from ``<dataset_path>/data.csv``.

    Every row becomes a float tensor in ``data[key]``; by default the last
    column is the target and all others are inputs, unless ``task_config``
    selects columns via ``x_selection``/``y_selection``. A pandas copy of
    the table is kept in ``df`` (used by ``DiCEExplainer``). The config's
    ``tabular_preprocessing`` list may contain ``"minmax_-1_1"``, which
    scales continuous columns (more than 10 unique values, excluding
    targets and ``confounding_factors``) to ``[-1, 1]``.

    Parameters
    ----------
    mode : str
        Split to load: ``"train"``, ``"val"``, ``"test"`` or ``"all"``.
    config : DataConfig
        Data config; uses ``dataset_path``, ``set_negative_to_zero``,
        ``output_size``, ``confounding_factors`` and
        ``tabular_preprocessing``.
    transform : callable, optional
        Stored but not applied to the tensors. Defaults to ``ToTensor()``.
    task_config : TaskConfig, optional
        Column selection and ``output_channels``.
    **kwargs
        Ignored.

    Attributes
    ----------
    attributes : list of str
        Column names.
    keys : list
        Row identifiers of this split.
    df : pandas.DataFrame
        The (possibly preprocessed) table.
    """

    def __init__(
        self,
        mode,
        config,
        transform=ToTensor(),
        task_config=None,
        **kwargs,
    ):
        """Parse ``data.csv``, build the dataframe and apply tabular preprocessing."""
        self.config = config
        self.transform = transform
        self.task_config = task_config
        data_dir = os.path.join(config.dataset_path, "data.csv")

        self.attributes, self.data, self.keys = parse_csv(
            data_dir=data_dir,
            config=config,
            mode=mode,
            set_negative_to_zero=config.set_negative_to_zero,
        )
        import pandas as pd

        df_data = [self.data[k].numpy() for k in self.keys]
        self.df = pd.DataFrame(df_data, columns=self.attributes)
        self.groups_enabled = False
        self.idx_enabled = False
        self.return_dict = False
        self.hints_enabled = False

        if (
            hasattr(self.config, "tabular_preprocessing")
            and self.config.tabular_preprocessing is not None
        ):
            for preprocessing_step in self.config.tabular_preprocessing:
                if preprocessing_step == "minmax_-1_1":
                    # Determine which columns to NOT scale (targets and confounders)
                    exclude_cols = []
                    if (
                        self.task_config is not None
                        and self.task_config.y_selection is not None
                    ):
                        exclude_cols.extend(self.task_config.y_selection)
                    else:
                        exclude_cols.append(
                            self.attributes[-1]
                        )  # Default target is last column

                    if (
                        hasattr(self.config, "confounding_factors")
                        and self.config.confounding_factors is not None
                    ):
                        exclude_cols.extend(self.config.confounding_factors)

                    exclude_indices = [
                        self.attributes.index(col)
                        for col in exclude_cols
                        if col in self.attributes
                    ]

                    # Convert data dict to tensor for faster metric calculation
                    all_data_tensor = torch.stack(list(self.data.values()))

                    # We only want to scale continuous features, not categorical ones (e.g., 0, 1)
                    # Use a heuristic: features with > 10 unique values are continuous
                    continuous_indices = []
                    for i in range(all_data_tensor.shape[1]):
                        if (
                            i not in exclude_indices
                            and len(torch.unique(all_data_tensor[:, i])) > 10
                        ):
                            continuous_indices.append(i)

                    col_mins = all_data_tensor.min(dim=0)[0]
                    col_maxs = all_data_tensor.max(dim=0)[0]
                    col_ranges = col_maxs - col_mins

                    # Prevent division by zero
                    col_ranges[col_ranges == 0] = 1.0

                    for k in self.keys:
                        # Start with unscaled original data
                        scaled = self.data[k].clone()

                        # Only scale the continuous indices identified
                        for idx in continuous_indices:
                            val = (self.data[k][idx] - col_mins[idx]) / col_ranges[
                                idx
                            ]  # Scale to [0, 1]
                            scaled[idx] = val * 2.0 - 1.0  # Scale to [-1, 1]

                        self.data[k] = scaled

                    # Update DataFrame representation as well
                    df_data = [self.data[k].numpy() for k in self.keys]
                    self.df = pd.DataFrame(df_data, columns=self.attributes)

    def __len__(self):
        """Number of rows in this split."""
        return len(self.keys)

    @property
    def output_size(self):
        """``task_config.output_channels`` when targets are selected, else config."""
        if self.task_config is not None and self.task_config.y_selection is not None:
            return self.task_config.output_channels

        else:
            return self.config.output_size

    def enable_hints(self):
        """Set the ``hints_enabled`` flag (tabular data has no hints to return)."""
        self.hints_enabled = True

    def disable_hints(self):
        """Clear the ``hints_enabled`` flag."""
        self.hints_enabled = False

    def enable_groups(self):
        """Return ``has_confounder`` in dict mode (second confounding factor)."""
        self.groups_enabled = True

    def disable_groups(self):
        """Stop returning ``has_confounder``."""
        self.groups_enabled = False

    def enable_idx(self):
        """Also return the sample index from ``__getitem__``."""
        self.idx_enabled = True

    def disable_idx(self):
        """Stop returning the sample index."""
        self.idx_enabled = False

    def __getitem__(self, idx):
        """
        Return the features and target of row ``idx``.

        Parameters
        ----------
        idx : int
            Position in ``keys``.

        Returns
        -------
        tuple or dict
            ``(x, y)`` with ``x`` of shape ``[num_features]`` and ``y`` a
            scalar (or ``[len(y_selection)]`` for several targets);
            ``(x, [y, idx])`` when indices are enabled. With ``return_dict``
            a dict with ``x``, ``y`` and optional ``index``/``has_confounder``.
        """
        name = self.keys[idx]

        data = self.data[name].clone().detach().to(torch.float32)

        if (
            not self.task_config is None
            and not self.task_config.x_selection is None
            and not len(self.task_config.x_selection) == 0
        ):
            x = torch.zeros([len(self.task_config.x_selection)], dtype=torch.float32)
            for idx, selection in enumerate(self.task_config.x_selection):
                x[idx] = data[self.attributes.index(selection)]

        else:
            x = data[:-1].to(torch.float32)

        if (
            not self.task_config is None
            and not self.task_config.y_selection is None
            and not self.task_config.y_selection is None
        ):
            y = torch.zeros([len(self.task_config.y_selection)])
            for idx, selection in enumerate(self.task_config.y_selection):
                y[idx] = data[self.attributes.index(selection)]

            if y.shape[0] == 1:
                y = y[0]

        else:
            y = data[-1]

        if not self.return_dict:
            if self.idx_enabled:
                return x, [y, idx]
            return x, y

        return_dict = {"x": x, "y": y}
        if self.idx_enabled:
            return_dict["index"] = idx

        if self.groups_enabled:
            if (
                not self.config.confounding_factors is None
                and len(self.config.confounding_factors) >= 2
            ):
                confounder_name = self.config.confounding_factors[1]
                has_confounder = data[self.attributes.index(confounder_name)]
                return_dict["has_confounder"] = has_confounder
            else:
                return_dict["has_confounder"] = 0.0

        return return_dict

    def generate_contrastive_collage(
        self,
        x_list: list,
        x_counterfactual_list: list,
        y_target_list: list,
        y_source_list: list,
        y_list: list,
        y_target_start_confidence_list: list,
        y_target_end_confidence_list: list,
        base_path: str,
        start_idx: int = 0,
        y_original_teacher_list=None,
        y_counterfactual_teacher_list=None,
        feedback_list=None,
        **kwargs: dict,
    ) -> tuple:
        """
        Render bar-chart collages of factual vs. counterfactual feature vectors.

        For every pair a figure with three bar charts (original,
        counterfactual, difference) over the feature names is written to
        ``<base_path>/<zero-padded index>_collage.png``.

        Parameters
        ----------
        x_list, x_counterfactual_list : list of torch.Tensor
            Factual and counterfactual feature vectors.
        y_target_list, y_source_list, y_list : list
            Target class, predicted source class and ground truth per pair.
        y_target_start_confidence_list, y_target_end_confidence_list : list
            Target-class confidence before and after the edit.
        base_path : str
            Output directory, created if missing.
        start_idx : int, optional
            Offset added to the running index in the filenames.
        y_original_teacher_list, y_counterfactual_teacher_list, feedback_list
            Accepted for API compatibility; not drawn for tabular data.
        **kwargs
            Ignored.

        Returns
        -------
        tuple
            ``(attribution_list, collage_paths)`` where attributions are
            ``|cf - x|`` tensors and paths are the written PNG files.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from pathlib import Path

        Path(base_path).mkdir(parents=True, exist_ok=True)
        collage_paths = []
        attribution_list = []

        if self.task_config is not None and self.task_config.x_selection is not None:
            feature_names = self.task_config.x_selection
        else:
            feature_names = self.attributes[:-1]

        for i in range(len(x_list)):
            original = x_list[i].detach().cpu().numpy().flatten()
            cf = x_counterfactual_list[i].detach().cpu().numpy().flatten()
            diff = cf - original
            attribution_list.append(torch.tensor(np.abs(diff)))

            try:
                fig, axes = plt.subplots(3, 1, figsize=(10, 12))

                # Plot Original
                axes[0].bar(feature_names, original, color="#3498db")
                axes[0].set_title(
                    f"Original (Source Class: {int(y_source_list[i])}, Confidence: {float(y_target_start_confidence_list[i]):.2f})",
                    fontweight="bold",
                )
                axes[0].tick_params(axis="x", rotation=45)
                axes[0].grid(axis="y", linestyle="--", alpha=0.7)

                # Plot Counterfactual
                axes[1].bar(feature_names, cf, color="#2ecc71")
                axes[1].set_title(
                    f"Counterfactual (Target Class: {int(y_target_list[i])}, Confidence: {float(y_target_end_confidence_list[i]):.2f})",
                    fontweight="bold",
                )
                axes[1].tick_params(axis="x", rotation=45)
                axes[1].grid(axis="y", linestyle="--", alpha=0.7)

                # Plot Difference
                axes[2].bar(feature_names, diff, color="#e74c3c")
                axes[2].set_title(
                    "Difference (Counterfactual - Original)", fontweight="bold"
                )
                axes[2].tick_params(axis="x", rotation=45)
                axes[2].grid(axis="y", linestyle="--", alpha=0.7)

                plt.tight_layout()

                collage_path = os.path.join(
                    base_path,
                    embed_numberstring(str(start_idx + i)) + "_collage.png",
                )
                plt.savefig(collage_path, dpi=150)
                collage_paths.append(collage_path)
            finally:
                plt.close(fig)

        plt.close("all")
        return attribution_list, collage_paths

    def serialize_dataset(
        self,
        output_dir,
        x_list,
        y_list,
        sample_names=None,
        hint_list=[],
        classifier=None,
    ):
        """
        Write samples as a CSV next to ``output_dir``.

        The file ``<output_dir>.csv`` gets one row per sample with the
        feature columns (``task_config.x_selection`` or all but the last
        attribute) followed by the target column(s). Nothing is written for
        an empty ``x_list``.

        Parameters
        ----------
        output_dir : str
            Directory that is created; the CSV is named ``output_dir + ".csv"``.
        x_list : list of torch.Tensor
            Feature vectors.
        y_list : list
            Scalar targets.
        sample_names, hint_list, classifier
            Accepted for API compatibility; unused for tabular data.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        if len(x_list) == 0:
            _log.info(
                "%s", "Warning: serialize_dataset called with empty x_list. Skipping."
            )
            return

        x = torch.stack(x_list, dim=0).cpu()
        y = torch.stack([torch.tensor([y]) for y in y_list], dim=0).cpu()
        data = torch.cat([x, y], dim=1)
        if (
            not self.task_config is None
            and not self.task_config.x_selection is None
            and not len(self.task_config.x_selection) == 0
        ):
            features = copy.deepcopy(self.task_config.x_selection)

        else:
            features = copy.deepcopy(self.attributes[:-1])

        if (
            not self.task_config is None
            and not self.task_config.y_selection is None
            and not self.task_config.y_selection is None
        ):
            targets = copy.deepcopy(self.task_config.y_selection)

        else:
            targets = copy.deepcopy(self.attributes[-1:])

        np.savetxt(
            output_dir + ".csv",
            data.numpy(),
            delimiter=",",
            header=",".join(features + targets),
            comments="",
        )


class ImageDataset(PealDataset):
    """
    Base class of the image datasets with the explainer-facing callbacks.

    Subclasses provide loading (``__getitem__``, ``__len__``,
    ``output_size``) and the value-range helpers
    ``project_to_pytorch_default``/``project_from_pytorch_default`` of
    ``PealDataset``; this class adds collage rendering, serialization of
    counterfactual datasets and DINO-feature metrics. A ``DINOEvaluator``
    is created lazily in ``dino_eval`` and fitted on this dataset the first
    time a metric is requested.
    """

    def generate_contrastive_collage(
        self,
        x_list: list,
        x_counterfactual_list: list,
        y_target_list: list,
        y_source_list: list,
        y_list: list,
        y_target_start_confidence_list: list,
        y_target_end_confidence_list: list,
        base_path: str,
        start_idx: int = 0,
        y_original_teacher_list=None,
        y_counterfactual_teacher_list=None,
        feedback_list=None,
        hint_list=None,
        idx_to_info=None,
        tracking_level=1,
        history_list=None,
        additional_info=None,
        **kwargs: dict,
    ) -> tuple:
        """
        Render factual/counterfactual/SSIM-difference collages to PNG files.

        Images are mapped to ``[0, 1]`` with ``project_to_pytorch_default``,
        a high-contrast difference heatmap is computed for every pair and,
        if ``tracking_level >= 1``, a three-panel figure titled with the
        labels, confidences, optional latent info, history and teacher
        feedback is saved as ``<base_path>/<zero-padded index>_collage.png``.
        Titles longer than 800 characters are truncated to keep matplotlib
        from failing.

        Parameters
        ----------
        x_list, x_counterfactual_list : list of torch.Tensor
            Images of shape ``[C, H, W]`` in the dataset's value range.
        y_target_list, y_source_list, y_list : list
            Target class, predicted source class and ground truth per pair.
        y_target_start_confidence_list, y_target_end_confidence_list : list
            Target-class confidence before and after the edit.
        base_path : str
            Output directory, created if missing; also stored in
            ``self.base_path``.
        start_idx : int, optional
            Offset added to the running index in the filenames.
        y_original_teacher_list, y_counterfactual_teacher_list, feedback_list
            Teacher verdicts appended to the title when ``feedback_list`` is
            given.
        hint_list : list, optional
            Per-sample hints passed on to ``idx_to_info``.
        idx_to_info : callable, optional
            ``f(x, x_counterfactual, hint) -> str`` for an extra title line.
        tracking_level : int, optional
            Below 1 no figures are written (paths become ``None``).
        history_list : list, optional
            Per-sample optimisation histories printed into the title.
        additional_info : str, optional
            Free text drawn at the bottom of the figure.
        **kwargs
            Ignored.

        Returns
        -------
        tuple
            ``(heatmap_list, collage_paths)``: difference heatmaps as tensors
            and the PNG paths (``None`` where nothing was written).
        """

        Path(base_path).mkdir(parents=True, exist_ok=True)
        self.base_path = base_path
        collage_paths = []
        heatmap_list = []
        from torchvision.transforms import ToPILImage

        to_pil = ToPILImage()
        for i in range(len(x_list)):
            x = self.project_to_pytorch_default(x_list[i])
            counterfactual = self.project_to_pytorch_default(x_counterfactual_list[i])
            heatmap_high_contrast, x_in, counterfactual_rgb = high_contrast_heatmap(
                x, counterfactual
            )

            heatmap_list.append(heatmap_high_contrast)

            if tracking_level >= 1:
                ssim_overlay = generate_ssim_overlay(x, counterfactual)
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                try:
                    # Show images with padding and labels below
                    axes[0].imshow(x_in.permute(1, 2, 0).cpu().numpy())
                    axes[0].axis("off")
                    axes[0].text(
                        0.5,
                        -0.15,
                        "factual",
                        transform=axes[0].transAxes,
                        ha="center",
                        fontweight="bold",
                    )

                    axes[1].imshow(counterfactual_rgb.permute(1, 2, 0).cpu().numpy())
                    axes[1].axis("off")
                    axes[1].text(
                        0.5,
                        -0.15,
                        "counterfactual",
                        transform=axes[1].transAxes,
                        ha="center",
                        fontweight="bold",
                    )

                    axes[2].imshow(ssim_overlay.permute(1, 2, 0).cpu().numpy())
                    axes[2].axis("off")
                    axes[2].text(
                        0.5,
                        -0.15,
                        "SSIM difference",
                        transform=axes[2].transAxes,
                        ha="center",
                        fontweight="bold",
                    )

                    # Robustly build title string with length checks
                    def safe_int_str(val):
                        """Render a label (scalar, one-hot tensor or other) as text."""
                        if torch.is_tensor(val):
                            if val.numel() == 1:
                                return str(int(val.item()))
                            else:
                                return str(int(val.argmax()))
                        try:
                            return str(int(val))
                        except (ValueError, TypeError):
                            return str(val)

                    title_string = "Original: "
                    if len(y_list) > i:
                        title_string += safe_int_str(y_list[i])
                    else:
                        title_string += "?"

                    title_string += " -> Prediction: "
                    if len(y_source_list) > i:
                        title_string += safe_int_str(y_source_list[i])
                    else:
                        title_string += "?"

                    title_string += " -> Target: "
                    if len(y_target_list) > i:
                        title_string += safe_int_str(y_target_list[i])
                    else:
                        title_string += "?"

                    title_string += "\n"

                    start_conf = (
                        float(y_target_start_confidence_list[i])
                        if len(y_target_start_confidence_list) > i
                        else 0.0
                    )
                    end_conf = (
                        float(y_target_end_confidence_list[i])
                        if len(y_target_end_confidence_list) > i
                        else 0.0
                    )

                    title_string += (
                        "Target Confidence: "
                        + str(round(start_conf, 2))
                        + " -> "
                        + str(round(end_conf, 2))
                        + "\n"
                    )
                    if not idx_to_info is None:
                        hint = (
                            hint_list[i]
                            if hint_list is not None and i < len(hint_list)
                            else None
                        )
                        title_string += (
                            idx_to_info(x_list[i], x_counterfactual_list[i], hint)
                            + "\n"
                        )

                    if (
                        history_list is not None
                        and i < len(history_list)
                        and history_list[i] is not None
                    ):
                        hist_val = history_list[i]
                        if isinstance(hist_val, torch.Tensor):
                            title_string += str(hist_val.tolist()) + "\n"
                        elif hist_val:
                            title_string += str(hist_val) + "\n"

                    if not feedback_list is None:
                        title_string += (
                            ", Teacher: "
                            + str(int(y_original_teacher_list[i]))
                            + " -> "
                            + str(int(y_counterfactual_teacher_list[i]))
                            + " -> "
                            + str(feedback_list[i])
                        )

                    # An explainer whose history holds per-step prediction tensors (ACE does)
                    # turns this title into hundreds of lines. matplotlib then shrinks the
                    # figure until the glyph size rounds to zero and FreeType raises
                    # "Could not convert glyph to bitmap (error code 0x62)", which killed the
                    # whole run from a purely cosmetic step. The collage is a visualisation, so
                    # bound the title instead.
                    if len(title_string) > 800:
                        title_string = title_string[:800] + " ... [truncated]"
                    fig.suptitle(title_string)
                    plt.tight_layout()
                    if additional_info is not None:
                        plt.gcf().text(0.1, 0.05, additional_info, ha="left")
                    collage_path = os.path.join(
                        base_path,
                        embed_numberstring(str(start_idx + i)) + "_collage.png",
                    )
                    plt.savefig(collage_path, bbox_inches="tight")
                    _log.info("%s", "Saved collage to " + collage_path)
                    collage_paths.append(collage_path)
                finally:
                    plt.close(fig)

            else:
                collage_paths.append(None)

        plt.close("all")
        return heatmap_list, collage_paths

    def serialize_dataset(
        self,
        output_dir,
        x_list,
        y_list,
        hint_list=[],
        sample_names=None,
        classifier=None,
    ):
        """
        Save images (and optional masks) as a class-folder dataset on disk.

        Creates ``<output_dir>/imgs/<class>/`` (and ``masks/<class>/`` when
        hints are given) for every class, writes each image twice (inside
        its class folder and directly under ``imgs/``) as 8-bit PNG after
        ``project_to_pytorch_default`` and writes an ``ImgPath,Class`` CSV
        to ``<output_dir>/<config.label_rel_path>`` so the result can be
        loaded again as a PEAL dataset.

        Parameters
        ----------
        output_dir : str
            Root of the new dataset.
        x_list : list of torch.Tensor
            Images ``[C, H, W]`` in this dataset's value range.
        y_list : list
            Integer class labels.
        hint_list : list of torch.Tensor, optional
            Masks ``[C, H, W]`` in ``[0, 1]``; saved when non-empty.
        sample_names : list of str
            File stem per sample (required).
        classifier
            Unused.
        """
        # TODO this does not seem very clean
        for class_name in range(max(2, self.output_size)):
            Path(os.path.join(output_dir, "imgs", str(class_name))).mkdir(
                parents=True, exist_ok=True
            )
            if not len(hint_list) == 0:
                Path(os.path.join(output_dir, "masks", str(class_name))).mkdir(
                    parents=True, exist_ok=True
                )

        data = []
        for idx, x in enumerate(x_list):
            class_name = int(y_list[idx])
            x = self.project_to_pytorch_default(x)
            img = Image.fromarray(
                np.array(255 * x.cpu().numpy().transpose(1, 2, 0), dtype=np.uint8)
            )
            img_name = os.path.join(str(class_name), sample_names[idx] + ".png")
            img.save(os.path.join(output_dir, "imgs", img_name))
            img.save(
                os.path.join(
                    output_dir, "imgs", os.path.join(sample_names[idx] + ".png")
                )
            )
            if not len(hint_list) == 0:
                mask = Image.fromarray(
                    np.array(
                        255 * hint_list[idx].cpu().numpy().transpose(1, 2, 0),
                        dtype=np.uint8,
                    )
                )
                mask.save(os.path.join(output_dir, "masks", img_name))

            data.append(
                [
                    img_name,
                    class_name,
                ]
            )

        data = "ImgPath,Class\n" + "\n".join([",".join(map(str, x)) for x in data])
        with open(os.path.join(output_dir, self.config.label_rel_path), "w") as f:
            f.write(data)

    def track_generator_performance(
        self,
        generator: Union[Generator, torch.Tensor],
        batch_size=None,
        num_samples=None,
    ):
        """
        Compute DINO-feature FID and LPIPS for generated images.

        Draws ``num_samples`` images from the generator (or uses the given
        tensor), fits the ``DINOEvaluator`` on this dataset on first use
        (for CelebA a shared reference state under
        ``$PEAL_RUNS/sce_cfkd/results_for_paper/...`` is loaded if present)
        and compares against the first real samples of the dataset.

        Parameters
        ----------
        generator : Generator or torch.Tensor
            Generator with ``sample_x`` or a batch of images ``[B, C, H, W]``.
        batch_size : int, optional
            Sampling batch size; defaults to ``generator.config.batch_size``
            or 1.
        num_samples : int, optional
            Number of generated images evaluated. Defaults to ``batch_size``.

        Returns
        -------
        dict
            ``fid``/``dino_fid``, ``lpips``/``dino_lpips`` and, if
            ``reference_fid`` is set on the dataset, ``quality_score`` in
            ``(0, 1]``.

        Raises
        ------
        NotImplementedError
            For unsupported ``generator`` types.
        """
        if batch_size is None:
            if hasattr(generator, "config"):
                batch_size = generator.config.batch_size

            else:
                batch_size = 1

        if num_samples is None:
            num_samples = batch_size

        if isinstance(generator, torch.Tensor):
            generated_images = torch.clone(generator)

        elif isinstance(generator, Generator):
            # TODO set device
            generated_images = generator.sample_x(batch_size=batch_size).detach()
            while generated_images.shape[0] < num_samples:
                generated_images = torch.cat(
                    [generated_images, generator.sample_x(batch_size=batch_size)], dim=0
                )

        else:
            raise NotImplementedError("Generator type not supported")

        generated_images = generated_images[:num_samples]
        if generated_images.shape[0] == 1:
            generated_images = torch.cat([generated_images, generated_images], dim=0)

        if not hasattr(self, "dino_eval"):
            self.dino_eval = DINOEvaluator()

            # state_path = os.path.join(self.base_path, "dino_evaluation_state.pt")
            celeba_state_path = os.path.join(
                os.environ.get("PEAL_RUNS", "peal_runs"),
                "sce_cfkd/results_for_paper",
                "celeba_blond_hair_natural_best_Results",
                "dino_evaluation_state.pt",
            )
            # The shared CelebA reference state keeps FID comparable across the paper
            # runs, but it lives outside this repo and is not present on every node.
            # Fall back to fitting the evaluator on this dataset so a missing file
            # costs comparability rather than aborting the whole adaptor run.
            if (
                hasattr(self, "config")
                and getattr(self.config, "dataset_class", None) == "CelebADataset"
                and os.path.exists(celeba_state_path)
            ):
                self.dino_eval._load_state(celeba_state_path)
            else:
                self.dino_eval.fit(
                    torch.utils.data.DataLoader(self, batch_size=batch_size),
                )

        dino_fid = self.dino_eval.compute_fid(generated_images)

        # Compute DINO LPIPS against a subset of real dataset samples
        real_images = torch.stack(
            [self[i][0] for i in range(min(len(self), generated_images.shape[0]))]
        ).to(generated_images.device)
        dino_lpips = self.dino_eval.compute_lpips(real_images, generated_images)

        output_dict = {
            "fid": dino_fid,
            "dino_fid": dino_fid,
            "lpips": dino_lpips,
            "dino_lpips": dino_lpips,
        }

        if hasattr(self, "reference_fid") and not self.reference_fid is None:
            quality_score = min(1.0, self.reference_fid / (dino_fid + 1e-8))
            output_dict["quality_score"] = quality_score

        return output_dict

    def calculate_outlier_score(self, x):
        """
        Mahalanobis distance of images to the dataset's DINO feature Gaussian.

        Parameters
        ----------
        x : torch.Tensor
            Batch ``[B, C, H, W]``; its size is also used as the fitting
            batch size on first use.

        Returns
        -------
        dict
            ``{"absolute": scores}`` plus ``"relative"`` (divided by
            ``reference_outlier_scores``) when that attribute is set.
        """
        if not hasattr(self, "dino_eval"):

            self.dino_eval = DINOEvaluator()
            # state_path = os.path.join(self.base_path, "dino_evaluation_state.pt")
            # if os.path.exists(state_path):
            #     self.dino_eval._load_state(state_path)
            # else:
            self.dino_eval.fit(
                torch.utils.data.DataLoader(self, batch_size=x.shape[0]),
                # nothing reloads it (see above), and the default wrote
                # dino_evaluation_state.pt into whatever the working directory was
                save_path=None,
            )

        outlier_scores = {"absolute": self.dino_eval.compute_mahalanobis(x)}

        if (
            hasattr(self, "reference_outlier_scores")
            and self.reference_outlier_scores is not None
        ):
            outlier_scores["relative"] = outlier_scores["absolute"] / (
                self.reference_outlier_scores + 1e-8
            )

        return outlier_scores

    def _initialize_performance_metrics(self):
        """Create and fit the ``DINOEvaluator`` if it does not exist yet."""
        if not hasattr(self, "dino_eval"):
            self.dino_eval = DINOEvaluator()
            self.dino_eval.fit(torch.utils.data.DataLoader(self, batch_size=32))

    def distribution_distance(self, x_list):
        """
        Mean DINO-FID of several image batches against the dataset.

        Parameters
        ----------
        x_list : list
            Batches, each a tensor ``[B, C, H, W]`` or a list of images.

        Returns
        -------
        float
            Average FID over the batches.
        """
        if not hasattr(self, "dino_eval"):
            self.dino_eval = DINOEvaluator()
            self.dino_eval.fit(torch.utils.data.DataLoader(self, batch_size=32))
        fids = []
        for i in range(len(x_list)):
            batch = (
                torch.stack(x_list[i], dim=0)
                if isinstance(x_list[i], (list, tuple))
                else x_list[i]
            )
            fid_score = self.dino_eval.compute_fid(batch)
            fids.append(fid_score)

        return float(np.mean(fids))

    def pair_wise_distance(self, x1, x2):
        """
        Mean DINO-LPIPS between corresponding batches of ``x1`` and ``x2``.

        Parameters
        ----------
        x1, x2 : list
            Equal-length lists of batches (tensor or list of images).

        Returns
        -------
        float
            Average pairwise distance over the batches.
        """
        if not hasattr(self, "dino_eval"):
            self.dino_eval = DINOEvaluator()
        distances = []
        for i in range(len(x1)):
            b1 = (
                torch.stack(x1[i], dim=0) if isinstance(x1[i], (list, tuple)) else x1[i]
            )
            b2 = (
                torch.stack(x2[i], dim=0) if isinstance(x2[i], (list, tuple)) else x2[i]
            )
            dist = self.dino_eval.compute_lpips(b1, b2)
            distances.append(dist)

        return float(np.mean(distances))

    def variance(self, x_list):
        """
        Mean per-sample pixel variance across attempts.

        Parameters
        ----------
        x_list : list of list of torch.Tensor
            ``x_list[j][i]`` is attempt ``j`` for sample ``i``.

        Returns
        -------
        float
            Variance over attempts, averaged over pixels and samples.
        """
        variances = []
        for i in range(len(x_list[0])):
            variance = torch.mean(
                torch.var(
                    torch.stack([x_list[j][i] for j in range(len(x_list))], dim=0),
                    dim=0,
                )
            )
            variances.append(variance)

        return np.mean(variances)

    def flip_rate(self, y_confidence_list):
        """
        Fraction of confidences above 0.5, averaged over samples.

        Parameters
        ----------
        y_confidence_list : list of list of torch.Tensor
            Target-class confidences per sample and attempt.

        Returns
        -------
        float
            Mean flip rate.
        """
        flip_rates = []
        for i in range(len(y_confidence_list)):
            flip_rate = torch.mean((torch.stack(y_confidence_list[i]) > 0.5).float())
            flip_rates.append(flip_rate)

        return np.mean(flip_rates)


class Image2MixedDataset(ImageDataset):
    """
    Image dataset with multi-attribute CSV labels (CelebA-style).

    Images live under ``<root_dir>/<config.x_selection>/`` and are looked up
    by the keys of the label CSV ``<root_dir>/<config.label_rel_path>`` (see
    ``image_name``/``open_image`` for the tolerated filename variants);
    optional masks live under ``<root_dir>/masks/``. Attribute names of the
    form ``A_vs_B`` yield the positive/negative phrases used for text
    descriptions. Targets are the ``task_config.y_selection`` attributes
    (or the first ``config.output_size[0]`` columns); with a ``"ce"``
    criterion the single target is returned as an ``int64`` class index.
    ``config.class_ratios`` subsamples the split to fixed class
    proportions. If the subclass defines ``sample_to_2d_latent``, decision
    boundary and global counterfactual plots become available.

    Parameters
    ----------
    config : DataConfig
        Uses ``dataset_path``, ``label_rel_path``, ``x_selection``,
        ``delimiter``, ``set_negative_to_zero``, ``input_size``,
        ``output_size``, ``class_ratios``, ``has_hints``, ``in_memory`` and
        ``confounding_factors``.
    mode : str
        Split: ``"train"``, ``"val"``, ``"test"`` or ``"all"``.
    root_dir : str, optional
        Dataset root; overrides and rewrites ``config.dataset_path``.
    data_dir : str, optional
        Path of the label CSV. Defaults to ``<root_dir>/<label_rel_path>``.
    transform : callable, optional
        Applied to the PIL image (and, with the same RNG state, to the
        mask). Defaults to ``ToTensor()``.
    task_config : TaskConfig, optional
        Target selection, ``output_channels`` and ``criterions``.
    return_dict : bool, optional
        Return a dict from ``__getitem__`` instead of ``(x, y...)``.

    Attributes
    ----------
    attributes : list of str
        CSV column names; ``attributes_positive``/``attributes_negative``
        hold the description phrases.
    keys : list of str
        Image keys of the (possibly class-balanced or restricted) split.
    data : dict
        Key to label tensor.
    """

    def visualize_decision_boundary(
        self,
        predictor,
        batch_size,
        device,
        path,
        temperature=1.0,
        train_dataloader=None,
        val_dataloaders=[],
        val_weights=[],
        test_dataloader=None,
    ):
        """Generalized decision boundary visualization using sample_to_2d_latent.

        Uses the balanced test dataset to fit a linear model in the 2D latent space,
        then visualizes the decision boundary with train/val/test sample overlays.

        Requires ``self.sample_to_2d_latent`` (returns silently otherwise). A
        logistic regression on the test latents (fitted on the predictor's
        predictions) yields a probability grid that is saved as
        ``<path minus .png>.npy`` and ``..._bounds.npy`` for
        ``global_counterfactual_visualization``; the test plot goes to
        ``decision_boundary_test.png`` next to ``path`` and the train/val
        plot to ``path``. Marker fill shows the prediction, edge the label.

        Parameters
        ----------
        predictor : torch.nn.Module
            Classifier evaluated on ``device``.
        batch_size : int
            Unused; kept for interface compatibility.
        device : torch.device or str
            Device for predictor and latent extraction.
        path : str
            Output PNG of the train/val plot.
        temperature : float, optional
            Unused.
        train_dataloader : DataLoader or DataloaderMixer, optional
            Up to 8 batches are plotted.
        val_dataloaders : list or WeightedDataloaderList, optional
            Validation loaders; ``val_weights`` scales the batch budget.
        val_weights : list of float, optional
        test_dataloader : DataLoader, optional
            Up to 16 batches define the latent range and fit the model.
        """
        if not hasattr(self, "sample_to_2d_latent"):
            return

        from peal.data.dataloaders import DataloaderMixer, WeightedDataloaderList
        from sklearn.linear_model import LogisticRegression

        _log.info("%s", "visualize_decision_boundary (generalized)")

        # Extract 2D latents from a dataloader
        def extract_latents(dataloader, max_batches=8):
            """Return 2D latents, labels and predictions for up to ``max_batches``."""
            latents = []
            y_list = []
            y_pred_list = []
            for batch_idx, batch in enumerate(dataloader):
                if batch_idx >= max_batches:
                    break

                x, y = batch
                if isinstance(y, (list, tuple)):
                    y, hint = y[:2]
                else:
                    hint = None

                if isinstance(x, (list, tuple)):
                    x_tensor = torch.stack(x)
                else:
                    x_tensor = x

                with torch.no_grad():
                    logits = predictor(x_tensor.to(device)).detach().cpu()
                    if logits.shape[-1] == 1:
                        probs = torch.sigmoid(logits)
                        preds = (probs > 0.5).int().flatten()
                    else:
                        preds = logits.argmax(dim=-1)

                for idx in range(len(x)):
                    sample = x[idx]
                    mask = hint[idx] if hint is not None else None
                    with torch.no_grad():
                        latent = self.sample_to_2d_latent(sample.to(device), mask).cpu()
                    latents.append(latent.numpy())
                    y_val = y[idx]
                    if isinstance(y_val, torch.Tensor) and y_val.dim() > 0:
                        y_val = y_val.argmax()
                    y_list.append(int(y_val))
                    y_pred_list.append(int(preds[idx]))

            return np.array(latents), np.array(y_list), np.array(y_pred_list)

        # Step 1: Use the test dataloader to find range and fit linear model
        test_latents, test_y, test_y_pred = np.array([]), np.array([]), np.array([])
        if test_dataloader is not None:
            hints_buffer = test_dataloader.dataset.hints_enabled
            if getattr(self, "hints_enabled", False):
                test_dataloader.dataset.enable_hints()
            else:
                test_dataloader.dataset.disable_hints()
            test_latents, test_y, test_y_pred = extract_latents(
                test_dataloader, max_batches=16
            )
            if hints_buffer:
                test_dataloader.dataset.enable_hints()
            else:
                test_dataloader.dataset.disable_hints()

        if len(test_latents) < 2:
            _log.info("%s", "Not enough test samples for visualize_decision_boundary")
            return

        # Determine latent range from test data
        x_min, x_max = test_latents[:, 0].min(), test_latents[:, 0].max()
        y_min, y_max = test_latents[:, 1].min(), test_latents[:, 1].max()
        margin = 0.1 * max(x_max - x_min, y_max - y_min)
        x_min, x_max = x_min - margin, x_max + margin
        y_min, y_max = y_min - margin, y_max + margin

        # Step 2: Fit a linear model on the 2D latent values (using PREDICTIONS)
        clf = LogisticRegression(max_iter=1000)
        if len(np.unique(test_y_pred)) > 1:
            clf.fit(test_latents, test_y_pred)
        else:
            clf.fit(test_latents, test_y)

        # Generate meshgrid for decision boundary background
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200),
            np.linspace(y_min, y_max, 200),
        )
        grid = np.c_[xx.ravel(), yy.ravel()]
        if hasattr(clf, "predict_proba"):
            probs = clf.predict_proba(grid)[:, 0]
        else:
            probs = clf.decision_function(grid)
        prediction_grid = probs.reshape(xx.shape)

        # Save the prediction grid for global_counterfactual_visualization
        grid_path = path[:-4] + ".npy"
        np.save(grid_path, prediction_grid)

        # Save bounds
        bounds_path = path[:-4] + "_bounds.npy"
        np.save(bounds_path, np.array([x_min, x_max, y_min, y_max]))

        cmap = cm.get_cmap("bwr")

        # Step 3: Visualize test set first
        test_path = os.path.join(os.path.dirname(path), "decision_boundary_test.png")
        plt.figure()
        plt.contourf(xx, yy, prediction_grid, levels=100, cmap=cmap)
        plt.contour(xx, yy, prediction_grid, levels=10, colors="black", linewidths=1.5)
        if len(test_latents) > 0:
            plt.scatter(
                test_latents[:, 0],
                test_latents[:, 1],
                c=np.where(test_y_pred == 1, "blue", "red"),
                marker="o",
                edgecolors=np.where(test_y == 1, "blue", "red"),
                linewidths=1.5,
                label="Test Samples",
                alpha=0.7,
            )
        plt.legend()
        cf_names = getattr(self.config, "confounding_factors", None)
        plt.xlabel(cf_names[0] if cf_names and len(cf_names) >= 1 else "Latent Dim 0")
        plt.ylabel(cf_names[1] if cf_names and len(cf_names) >= 2 else "Latent Dim 1")
        plt.subplots_adjust(right=0.85, top=0.85)
        plt.savefig(test_path, bbox_inches="tight")
        plt.clf()
        _log.info("%s", "visualize_decision_boundary (test) saved under " + test_path)

        # Step 4: Extract train and val latents
        if isinstance(train_dataloader, DataloaderMixer):
            train_hints_buffer = train_dataloader.hints_enabled
            if getattr(self, "hints_enabled", False):
                train_dataloader.enable_hints()
            else:
                train_dataloader.disable_hints()
        elif train_dataloader:
            train_hints_buffer = train_dataloader.dataset.hints_enabled
            if getattr(self, "hints_enabled", False):
                train_dataloader.dataset.enable_hints()
            else:
                train_dataloader.dataset.disable_hints()

        train_latents, train_y, train_y_pred = (
            extract_latents(train_dataloader)
            if train_dataloader
            else (np.array([]), np.array([]), np.array([]))
        )

        if train_dataloader:
            if train_hints_buffer:
                if isinstance(train_dataloader, DataloaderMixer):
                    train_dataloader.enable_hints()
                else:
                    train_dataloader.dataset.enable_hints()
            else:
                if isinstance(train_dataloader, DataloaderMixer):
                    train_dataloader.disable_hints()
                else:
                    train_dataloader.dataset.disable_hints()

        val_latents, val_y, val_y_pred = (np.array([]), np.array([]), np.array([]))
        if isinstance(val_dataloaders, WeightedDataloaderList):
            val_weights = val_dataloaders.weights
            val_dataloaders = val_dataloaders.dataloaders

        for val_idx, val_dataloader in enumerate(val_dataloaders):
            val_hints_buffer = val_dataloader.dataset.hints_enabled
            if getattr(self, "hints_enabled", False):
                val_dataloader.dataset.enable_hints()
            else:
                val_dataloader.dataset.disable_hints()
            max_batches = (
                max(1, int(8 // val_weights[val_idx])) if len(val_weights) > 0 else 8
            )
            val_latents_current, val_y_current, val_y_pred_current = extract_latents(
                val_dataloader, max_batches=max_batches
            )
            if len(val_latents) == 0:
                val_latents = val_latents_current
                val_y = val_y_current
                val_y_pred = val_y_pred_current
            else:
                val_latents = np.concatenate([val_latents, val_latents_current])
                val_y = np.concatenate([val_y, val_y_current])
                val_y_pred = np.concatenate([val_y_pred, val_y_pred_current])
            if val_hints_buffer:
                val_dataloader.dataset.enable_hints()
            else:
                val_dataloader.dataset.disable_hints()

        # Step 5: Create the train+val plot
        plt.figure()
        plt.contourf(xx, yy, prediction_grid, levels=100, cmap=cmap)
        plt.contour(xx, yy, prediction_grid, levels=10, colors="black", linewidths=1.5)

        if len(train_latents) > 0:
            plt.scatter(
                train_latents[:, 0],
                train_latents[:, 1],
                c=np.where(train_y_pred == 1, "blue", "red"),
                marker="^",
                edgecolors=np.where(train_y == 1, "blue", "red"),
                linewidths=1.5,
                label="Train Samples",
                alpha=0.7,
            )

        if len(val_latents) > 0:
            plt.scatter(
                val_latents[:, 0],
                val_latents[:, 1],
                c=np.where(val_y_pred == 1, "blue", "red"),
                marker="s",
                edgecolors=np.where(val_y == 1, "blue", "red"),
                linewidths=1.5,
                label="Val Samples",
                alpha=0.7,
            )

        plt.legend()
        plt.xlabel(cf_names[0] if cf_names and len(cf_names) >= 1 else "Latent Dim 0")
        plt.ylabel(cf_names[1] if cf_names and len(cf_names) >= 2 else "Latent Dim 1")
        plt.subplots_adjust(right=0.85, top=0.85)
        plt.savefig(path, bbox_inches="tight")
        plt.clf()

        _log.info("%s", "visualize_decision_boundary saved under " + path)

    def global_counterfactual_visualization(
        self,
        filename,
        x_list,
        counterfactuals,
        y_target_start_confidence,
        y_target_end_confidence,
        y_list,
        hint_list,
        attempts=1,
    ):
        """Generalized global counterfactual visualization using sample_to_2d_latent.

        Picks up to 10 samples of each of the first two classes seen in
        ``y_list`` (plus their extra attempts), maps factuals and
        counterfactuals to 2D latents and draws arrows between them on top of
        the decision boundary grid saved by ``visualize_decision_boundary``
        (``decision_boundary.npy`` or ``decision_boundary_precise.npy`` in the
        directory of ``filename``). Returns silently without
        ``sample_to_2d_latent``.

        Parameters
        ----------
        filename : str
            Output image path; ``"_precise"`` in the name selects the precise
            grid.
        x_list, counterfactuals : list of torch.Tensor
            Factual and counterfactual images, attempts stacked block-wise.
        y_target_start_confidence, y_target_end_confidence : list
            Target confidences used to color the arrow ends.
        y_list : list
            Ground-truth labels.
        hint_list : list, optional
            Hints forwarded to ``sample_to_2d_latent``.
        attempts : int, optional
            Number of attempts per factual contained in the lists.
        """
        if not hasattr(self, "sample_to_2d_latent"):
            return

        N_all = len(x_list) // attempts

        class_0 = None
        class_1 = None
        for i in range(N_all):
            val = y_list[i]
            if isinstance(val, torch.Tensor):
                val = int(val.item() if val.numel() == 1 else val.argmax())
            if class_0 is None:
                class_0 = val
            elif class_1 is None and val != class_0:
                class_1 = val

        indices_0 = []
        indices_1 = []
        for i in range(N_all):
            val = y_list[i]
            if isinstance(val, torch.Tensor):
                val = int(val.item() if val.numel() == 1 else val.argmax())
            if val == class_0 and len(indices_0) < 10:
                indices_0.append(i)
            elif val == class_1 and len(indices_1) < 10:
                indices_1.append(i)

        base_indices = indices_0 + indices_1

        all_indices = []
        for a in range(attempts):
            all_indices.extend([idx + a * N_all for idx in base_indices])

        x_list = [x_list[i] for i in all_indices]
        counterfactuals = [counterfactuals[i] for i in all_indices]
        y_target_start_confidence = [y_target_start_confidence[i] for i in all_indices]
        y_target_end_confidence = [y_target_end_confidence[i] for i in all_indices]
        y_list = [y_list[i] for i in all_indices]
        if hint_list is not None:
            hint_list = [hint_list[i] for i in all_indices]

        from peal.data.custom_datasets import plot_latents_with_arrows

        y_start_confidence = list(
            map(
                lambda i: abs(y_list[i] - y_target_start_confidence[i]),
                range(len(y_list)),
            )
        )
        y_end_confidence = list(
            map(
                lambda i: abs(y_list[i] - y_target_end_confidence[i]),
                range(len(y_list)),
            )
        )

        original_latents = []
        counterfactual_latents = []
        hints_enabled_buffer = getattr(self, "hints_enabled", False)
        self.hints_enabled = True
        device = x_list[0].device if hasattr(x_list[0], "device") else "cpu"
        for idx in range(len(x_list)):
            x = x_list[idx]
            hint = hint_list[idx] if hint_list is not None else None
            with torch.no_grad():
                orig_latent = self.sample_to_2d_latent(x.to(device), hint)
                cf_latent = self.sample_to_2d_latent(
                    counterfactuals[idx].to(device), hint
                )
            original_latents.append(orig_latent.cpu().tolist())
            counterfactual_latents.append(cf_latent.cpu().tolist())

        self.hints_enabled = hints_enabled_buffer

        is_precise = "_precise" in filename
        grid_name = (
            "decision_boundary_precise.npy" if is_precise else "decision_boundary.npy"
        )
        path = filename.split("/")[:-1] + [grid_name]
        if os.path.exists("/" + str(os.path.join(*path))):
            decision_boundary = np.load("/" + str(os.path.join(*path)))
        else:
            decision_boundary = np.load(str(os.path.join(*path)))

        if is_precise:
            decision_boundary = decision_boundary.T

        bounds_name = grid_name.replace(".npy", "_bounds.npy")
        bounds_path = filename.split("/")[:-1] + [bounds_name]
        if os.path.exists("/" + str(os.path.join(*bounds_path))):
            extent = np.load("/" + str(os.path.join(*bounds_path))).tolist()
        elif os.path.exists(str(os.path.join(*bounds_path))):
            extent = np.load(str(os.path.join(*bounds_path))).tolist()
        else:
            extent = [0, 1, 0, 1]

        cf_names = getattr(self.config, "confounding_factors", None)
        xlabel = cf_names[0] if cf_names and len(cf_names) >= 1 else "Latent Dim 0"
        ylabel = cf_names[1] if cf_names and len(cf_names) >= 2 else "Latent Dim 1"

        plot_latents_with_arrows(
            original_latents,
            counterfactual_latents,
            filename,
            y_start_confidence,
            y_end_confidence,
            decision_boundary,
            extent=extent,
            xlabel=xlabel,
            ylabel=ylabel,
            attempts=attempts,
        )

    def generate_contrastive_collage(
        self,
        x_list: list,
        x_counterfactual_list: list,
        y_target_list: list,
        y_source_list: list,
        y_list: list,
        y_target_start_confidence_list: list,
        y_target_end_confidence_list: list,
        base_path: str,
        idx_to_info=None,
        **kwargs: dict,
    ):
        """
        Collage rendering that adds the 2D latent coordinates to the title.

        If ``idx_to_info`` is not given and the dataset has
        ``sample_to_2d_latent``, a default is built that prints
        ``<factor>: <before> -> <after>`` per latent dimension (named after
        ``config.confounding_factors``). Everything else is delegated to
        ``ImageDataset.generate_contrastive_collage``.

        Returns
        -------
        tuple
            ``(heatmap_list, collage_paths)`` from the base implementation.
        """
        if idx_to_info is None and hasattr(self, "sample_to_2d_latent"):
            cf_names = getattr(self.config, "confounding_factors", None)

            def idx_to_info(x, x_counterfactual, hint):
                """Format the 2D latent shift of one pair as a title line."""
                device = x.device if hasattr(x, "device") else "cpu"
                with torch.no_grad():
                    latent_orig = self.sample_to_2d_latent(x.to(device), hint)
                    latent_cf = self.sample_to_2d_latent(
                        x_counterfactual.to(device), hint
                    )
                parts = []
                for i in range(len(latent_orig)):
                    name = cf_names[i] if cf_names and i < len(cf_names) else f"Dim{i}"
                    parts.append(
                        f"{name}: {round(float(latent_orig[i]), 3)} -> {round(float(latent_cf[i]), 3)}"
                    )
                return ", ".join(parts)

        return super().generate_contrastive_collage(
            x_list=x_list,
            x_counterfactual_list=x_counterfactual_list,
            y_target_list=y_target_list,
            y_source_list=y_source_list,
            y_list=y_list,
            y_target_start_confidence_list=y_target_start_confidence_list,
            y_target_end_confidence_list=y_target_end_confidence_list,
            base_path=base_path,
            idx_to_info=idx_to_info,
            **kwargs,
        )

    def __init__(
        self,
        config,
        mode,
        root_dir=None,
        data_dir=None,
        transform=ToTensor(),
        task_config=None,
        return_dict=False,
    ):
        """Parse the label CSV and optionally load all images into memory."""
        self.config = config
        if root_dir is None:
            self.root_dir = config.dataset_path

        else:
            self.root_dir = root_dir
            self.config.dataset_path = root_dir

        self.transform = transform
        self.task_config = task_config
        self.hints_enabled = False
        self.groups_enabled = False
        self.idx_enabled = False
        self.url_enabled = False
        self.string_description_enabled = False
        self.tokenizer = None
        self.return_dict = return_dict
        self.class_restrictions_enabled = False
        if data_dir is None:
            data_dir = os.path.join(self.root_dir, self.config.label_rel_path)

        if not config.delimiter is None:
            delimiter = config.delimiter

        else:
            delimiter = ","

        self.data_dir = data_dir
        negative_to_zero = getattr(config, "set_negative_to_zero", True)
        self.attributes, self.data, self.keys = parse_csv(
            data_dir,
            config,
            mode,
            key_type="name",
            delimiter=delimiter,
            set_negative_to_zero=negative_to_zero,
        )
        self.attributes_positive = []
        self.attributes_negative = []
        for attribute in self.attributes:
            attribute_values = attribute.split("_vs_")
            if len(attribute_values) == 2:
                self.attributes_positive.append(attribute_values[0])
                self.attributes_negative.append(attribute_values[1])

            else:
                self.attributes_positive.append("Is " + attribute)
                self.attributes_negative.append("Not " + attribute)

        self.task_specific_keys = None
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_keys is None
        ):
            self.set_task_specific_keys()

        if self.config.in_memory:
            self.load_in_memory()

    def load_in_memory(self):
        """
        Cache every image (and mask, if ``config.has_hints``) as a numpy array.

        Images are stored in ``in_memory_images`` under the resolved
        filename and converted to ``"L"`` or ``"RGB"`` according to
        ``config.input_size[0]``; masks go to ``in_memory_masks``.
        """
        _log.info("%s", "load dataset into memory!!")
        self.in_memory_images = {}
        self.in_memory_masks = {}
        for name in self.keys:
            # Cache under the same filename __getitem__ looks up, and resolve it
            # the same way - keys of serialized counterfactual datasets carry no
            # suffix, so a plain open() on the key misses the file entirely.
            name_img = self.image_name(name)
            img_pil = self.open_image(name_img)
            if (
                hasattr(self.config, "input_size")
                and self.config.input_size is not None
                and len(self.config.input_size) > 0
                and self.config.input_size[0] == 1
            ):
                img_pil = img_pil.convert("L")
            else:
                img_pil = img_pil.convert("RGB")
            self.in_memory_images[name_img] = np.array(img_pil)

            if self.config.has_hints:
                option1 = os.path.join(self.root_dir, "masks", name)
                option2 = os.path.join(self.root_dir, "masks", name.split("/")[-1])
                option3 = option2[:-4] + ".png"
                option4 = option1[:-4] + ".png"
                if os.path.exists(option1):
                    mask = Image.open(os.path.join(self.root_dir, "masks", name))

                elif os.path.exists(option2):
                    mask = Image.open(option2)

                elif os.path.exists(option3):
                    mask = Image.open(option3)

                elif os.path.exists(option4):
                    mask = Image.open(option4)

                else:
                    assert (
                        not self.config.has_hints
                    ), "Hints not found despite claim that they exist!"
                    mask = Image.new("RGB", img_pil.size, (0, 0, 0))

                self.in_memory_masks[name] = np.array(mask)

    @property
    def output_size(self):
        """``task_config.output_channels`` when targets are selected, else columns."""
        if self.task_config is not None and self.task_config.y_selection is not None:
            return self.task_config.output_channels

        else:
            return len(self.attributes)

    def __len__(self):
        """Number of keys, after applying ``class_ratios`` balancing if configured."""
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_keys is None
        ):
            self.set_task_specific_keys()

        return len(self.keys)

    def enable_hints(self):
        """Also return the mask from ``<root_dir>/masks`` as ``hint``."""
        self.hints_enabled = True

    def disable_hints(self):
        """Stop returning masks."""
        self.hints_enabled = False

    def enable_groups(self):
        """Also return ``has_confounder`` from the ``confounding_factors`` columns."""
        self.groups_enabled = True

    def disable_groups(self):
        """Stop returning ``has_confounder``."""
        self.groups_enabled = False

    def enable_idx(self):
        """Also return the sample index."""
        self.idx_enabled = True

    def disable_idx(self):
        """Stop returning the sample index."""
        self.idx_enabled = False

    def enable_url(self):
        """Also return the image key as ``url``."""
        self.url_enabled = True

    def disable_url(self):
        """Stop returning the image key."""
        self.url_enabled = False

    def enable_string_description(self):
        """Also return a text ``description`` built from the target attributes."""
        self.string_description_enabled = True

    def disable_string_description(self):
        """Stop returning the text description."""
        self.string_description_enabled = False

    def enable_tokens(self, tokenizer):
        """
        Also return ``tokens``: the description tokenized with ``tokenizer``.

        Parameters
        ----------
        tokenizer : callable
            HuggingFace-style tokenizer with ``model_max_length``; the
            description is padded/truncated to that length.
        """
        self.string_description_enabled_buffer = self.string_description_enabled
        self.enable_string_description()
        self.tokenizer = tokenizer

    def disable_tokens(self):
        """Drop the tokenizer and restore the previous description setting."""
        self.string_description_enabled = self.string_description_enabled_buffer
        self.tokenizer = None

    def enable_class_restriction(self, class_idx):
        """
        Keep only samples whose first selected target equals ``class_idx``.

        The full key list is saved in ``backup_keys``; requires
        ``task_config`` with ``y_selection``.

        Parameters
        ----------
        class_idx : int
            Class value to keep.
        """
        assert not self.task_config is None, "Task config must be set"
        self.backup_keys = copy.deepcopy(self.keys)
        self.keys = []
        for key in self.backup_keys:
            try:
                if (
                    self.task_config is not None
                    and hasattr(self.task_config, "y_selection")
                    and self.task_config.y_selection
                ):
                    sel = self.task_config.y_selection[0]
                    if sel in self.attributes:
                        attr_idx = self.attributes.index(sel)
                        if attr_idx < len(self.data[key]):
                            if int(self.data[key][attr_idx]) == class_idx:
                                self.keys.append(key)
            except Exception:
                pass

        self.class_restrictions_enabled = True

    def disable_class_restriction(self):
        """Restore ``backup_keys`` (note: ``class_restrictions_enabled`` stays True)."""
        if hasattr(self, "backup_keys"):
            self.keys = copy.deepcopy(self.backup_keys)

        self.class_restrictions_enabled = True

    def set_task_specific_keys(self):
        """
        Subsample ``keys`` so the classes follow ``config.class_ratios``.

        Counts samples per class of ``task_config.y_selection[0]``, finds the
        largest multiple of ``class_ratios`` that fits and keeps the first
        matching keys in order; the original list is saved in ``keys_backup``.
        """
        self.task_specific_keys = []
        num_samples_per_class = np.zeros([self.output_size])
        for key in self.keys:
            num_samples_per_class[
                int(
                    self.data[key][
                        self.attributes.index(self.task_config.y_selection[0])
                    ]
                )
            ] += 1

        num_units = num_samples_per_class / np.array(self.config.class_ratios)
        min_units = int(np.min(num_units))
        num_samples_per_class_balanced = min_units * np.array(self.config.class_ratios)
        current_num_samples_per_class = np.zeros([self.output_size])
        for key in self.keys:
            class_idx = int(
                self.data[key][self.attributes.index(self.task_config.y_selection[0])]
            )
            if (
                current_num_samples_per_class[class_idx]
                < num_samples_per_class_balanced[class_idx]
            ):
                self.task_specific_keys.append(key)
                current_num_samples_per_class[class_idx] += 1

        self.keys_backup = copy.deepcopy(self.keys)
        self.keys = self.task_specific_keys

    def image_name(self, name):
        """The image filename a dataset key refers to (keys may omit the suffix)."""
        # ImageNet files are *.JPEG; compare case-insensitively and accept .jpeg
        if not name.lower().endswith((".png", ".jpg", ".jpeg")):
            return name + ".jpg"

        return name

    def open_image(self, name_img):
        """Open an image by filename, tolerating .jpg/.png and zero-padded variants.

        Serialized counterfactual datasets and the raw datasets on disk disagree
        about both the extension and the zero padding, so try the plausible
        spellings before giving up.
        """
        base_dir = os.path.join(self.root_dir, self.config.x_selection)
        candidates = [name_img]
        raw_base = name_img.rsplit(".", 1)[0]
        ext = name_img.rsplit(".", 1)[1] if "." in name_img else "jpg"

        if ext in ["jpg", "JPG"]:
            candidates.append(f"{raw_base}.png")
            candidates.append(f"{raw_base}.PNG")
        elif ext in ["png", "PNG"]:
            candidates.append(f"{raw_base}.jpg")
            candidates.append(f"{raw_base}.JPG")

        if raw_base.isdigit():
            num = int(raw_base)
            for pad_len in [6, 5, 4]:
                candidates.append(f"{num:0{pad_len}d}.png")
                candidates.append(f"{num:0{pad_len}d}.jpg")

        for cand in candidates:
            cand_path = os.path.join(base_dir, cand)
            if os.path.exists(cand_path):
                try:
                    return Image.open(cand_path)
                except Exception:
                    pass

        try:
            return Image.open(os.path.join(base_dir, name_img))
        except Exception:
            return Image.open(os.path.join(base_dir, name_img.replace(".JPG", ".jpg")))

    def __getitem__(self, idx):
        """
        Load one sample.

        The image is opened (from memory or disk), converted to ``L``/``RGB``
        according to ``config.input_size[0]``, transformed and its channel
        count adjusted. Optional entries are added according to the
        ``enable_*`` flags; a mask is transformed with the same RNG state as
        the image so random augmentations match.

        Parameters
        ----------
        idx : int
            Position in ``keys``.

        Returns
        -------
        tuple or dict
            With ``return_dict`` a dict with ``x``, ``y`` and optional
            ``hint``, ``has_confounder``, ``index``, ``url``, ``description``,
            ``tokens``. Otherwise ``(x, y)`` when nothing else is enabled and
            ``(x, [y, extra, ...])`` (values in insertion order) otherwise.
        """
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_keys is None
        ):
            self.set_task_specific_keys()

        name = self.keys[idx]
        name_img = self.image_name(name)

        if self.config.in_memory:
            img = Image.fromarray(self.in_memory_images[name_img])
        else:
            img = self.open_image(name_img)

        if (
            hasattr(self.config, "input_size")
            and self.config.input_size is not None
            and len(self.config.input_size) > 0
            and self.config.input_size[0] == 1
        ):
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        state = torch.get_rng_state()

        img_tensor = self.transform(img)
        if (
            hasattr(self.config, "input_size")
            and self.config.input_size is not None
            and len(self.config.input_size) > 0
        ):
            if img_tensor.shape[0] == 1 and self.config.input_size[0] != 1:
                img_tensor = torch.tile(img_tensor, [self.config.input_size[0], 1, 1])
            elif img_tensor.shape[0] != self.config.input_size[0]:
                img_tensor = img_tensor[: self.config.input_size[0]]
        targets = self.data[name]

        if (
            not self.task_config is None
            and hasattr(self.task_config, "y_selection")
            and not self.task_config.y_selection is None
        ):
            target = torch.zeros([len(self.task_config.y_selection)])
            for i, selection in enumerate(self.task_config.y_selection):
                target[i] = targets[self.attributes.index(selection)]

        else:
            target = torch.tensor(
                targets[: self.config.output_size[0]], dtype=torch.float32
            )
        if (
            not self.task_config is None
            and hasattr(self.task_config, "criterions")
            and "ce" in self.task_config.criterions
        ):
            assert (
                target.shape[0] == 1
            ), "output shape inacceptable for singleclass classification"
            target = target[0].to(torch.int64)

        return_dict = {"x": img_tensor, "y": target}

        if self.hints_enabled:
            if self.config.in_memory:
                mask = Image.fromarray(self.in_memory_masks[name])

            else:
                option1 = os.path.join(self.root_dir, "masks", name)
                option2 = os.path.join(self.root_dir, "masks", name.split("/")[-1])
                option3 = option2[:-4] + ".png"
                option4 = option1[:-4] + ".png"
                if os.path.exists(option1):
                    mask = Image.open(os.path.join(self.root_dir, "masks", name))

                elif os.path.exists(option2):
                    mask = Image.open(option2)

                elif os.path.exists(option3):
                    mask = Image.open(option3)

                elif os.path.exists(option4):
                    mask = Image.open(option4)

                else:
                    assert (
                        not self.config.has_hints
                    ), "Hints not found despite claim that they exist!"
                    mask = Image.new("RGB", img.size, (0, 0, 0))

            torch.set_rng_state(state)
            try:
                mask_tensor = self.transform(mask)
            except Exception:
                mask_tensor = torch.zeros(3, 128, 128)

            if not mask_tensor.shape[0] == 3:

                # TODO very very hacky
                _log.info(
                    "%s",
                    "Mask tensor shape " + str(mask_tensor.shape) + " is not correct",
                )
                mask_tensor = torch.cat([mask_tensor, mask_tensor, mask_tensor])
                mask_tensor = mask_tensor[:3]

            return_dict["hint"] = mask_tensor

        if (
            self.groups_enabled
            and hasattr(self.config, "confounding_factors")
            and self.config.confounding_factors
        ):
            if len(self.config.confounding_factors) == 2:
                conf_factor = self.config.confounding_factors[-1]
                if conf_factor in self.attributes:
                    attr_idx = self.attributes.index(conf_factor)
                    if attr_idx < len(targets):
                        return_dict["has_confounder"] = targets[attr_idx]
            else:
                has_confounder = []
                for factor in self.config.confounding_factors[1:]:
                    if factor in self.attributes:
                        attr_idx = self.attributes.index(factor)
                        if attr_idx < len(targets):
                            has_confounder.append(targets[attr_idx])
                if has_confounder:
                    return_dict["has_confounder"] = has_confounder

        if self.idx_enabled:
            return_dict["index"] = idx

        if self.url_enabled:
            return_dict["url"] = name

        if self.string_description_enabled:
            return_dict["description"] = ""
            if hasattr(self, "task_config"):
                y_selection = self.task_config.y_selection

            else:
                y_selection = self.attributes

            for target_idx, attribute in enumerate(y_selection):
                attribute_idx = self.attributes.index(attribute)
                if (
                    len(y_selection) == 1
                    and target > 0.5
                    or len(y_selection) > 1
                    and target[target_idx] > 0.5
                ):
                    return_dict["description"] += self.attributes_positive[
                        attribute_idx
                    ]

                else:
                    return_dict["description"] += self.attributes_negative[
                        attribute_idx
                    ]

                if not target_idx == len(y_selection) - 1:
                    return_dict["description"] += ", "

            if self.tokenizer is not None:
                return_dict["tokens"] = torch.tensor(
                    self.tokenizer(
                        return_dict["description"],
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                )

        if self.return_dict:
            return return_dict

        else:
            return_list = list(return_dict.values())
            return (
                return_list[0],
                return_list[1:] if len(return_list) > 2 else return_list[1],
            )


class Image2ClassDataset(ImageDataset):
    """
    ImageFolder-style dataset: one sub-directory per class.

    Class names are the sorted directory names under
    ``<root_dir>/<config.x_selection>`` and the class index is the position
    in that list. All ``(class, file)`` pairs are shuffled with seed 0 and
    split by ``config.split`` into train/val/test. Masks, if
    ``config.has_hints``, are expected under ``<root_dir>/masks/[<class>/]``
    and must exist for every image. ``config.class_ratios`` subsamples the
    split to fixed class proportions.

    Parameters
    ----------
    mode : str
        Split: ``"train"``, ``"val"`` or ``"test"``; anything else keeps all.
    config : DataConfig
        Uses ``dataset_path``, ``x_selection``, ``split``, ``has_hints``,
        ``input_size``, ``output_size``, ``class_ratios`` and ``in_memory``.
    root_dir : str, optional
        Dataset root; falls back to ``data_dir`` and then
        ``config.dataset_path``. ``config.dataset_path`` is rewritten to it.
    data_dir : str, optional
        Alternative spelling of ``root_dir``.
    transform : callable, optional
        Applied to image and mask (same RNG state). Defaults to
        ``ToTensor()``.
    task_config : TaskConfig, optional
        Only its presence matters (for ``class_ratios`` balancing).
    return_dict : bool, optional
        Return a dict from ``__getitem__`` instead of ``(x, y...)``.

    Attributes
    ----------
    idx_to_name : list of str
        Class index to directory name.
    urls : list of tuple
        ``(class_name, filename)`` of the current split/restriction.
    """

    def __init__(
        self,
        mode,
        config,
        root_dir=None,
        data_dir=None,
        transform=ToTensor(),
        task_config=None,
        return_dict=False,
    ):
        """Scan the class directories, split them and optionally cache images."""
        self.config = config
        if root_dir is None:
            if data_dir is None:
                root_dir = config.dataset_path

            else:
                root_dir = data_dir

        self.config.dataset_path = root_dir

        if self.config.x_selection is not None:
            self.root_dir = os.path.join(root_dir, self.config.x_selection)
        else:
            self.root_dir = root_dir

        if self.config.has_hints:
            self.mask_dir = os.path.join(root_dir, "masks")
            self.all_urls = []
            self.urls_with_hints = []

        self.hints_enabled = False
        self.url_enabled = False
        self.idx_enabled = False
        self.url_enabled = False
        self.task_config = task_config
        self.transform = transform
        self.return_dict = return_dict
        self.urls = []
        self.idx_to_name = [
            d
            for d in os.listdir(self.root_dir)
            if not d.startswith(".") and os.path.isdir(os.path.join(self.root_dir, d))
        ]
        self.string_description_enabled = False
        self.groups_enabled = False

        self.idx_to_name.sort()
        for target_str in self.idx_to_name:
            files = os.listdir(os.path.join(self.root_dir, target_str))
            files.sort()
            for file in files:
                if file.startswith("."):
                    continue
                self.urls.append((target_str, file))

        random.seed(0)
        random.shuffle(self.urls)

        if mode == "train":
            self.urls = self.urls[: int(config.split[0] * len(self.urls))]

        elif mode == "val":
            self.urls = self.urls[
                int(config.split[0] * len(self.urls)) : int(
                    config.split[1] * len(self.urls)
                )
            ]

        elif mode == "test":
            self.urls = self.urls[int(config.split[1] * len(self.urls)) :]

        if self.config.has_hints:
            self.all_urls = copy.deepcopy(self.urls)
            for target_str, file in self.all_urls:
                if os.path.exists(os.path.join(self.mask_dir, file)) or os.path.exists(
                    os.path.join(self.mask_dir, target_str, file)
                ):
                    self.urls_with_hints.append((target_str, file))

                else:
                    _log.info("%s", os.path.join(self.mask_dir, file))
                    _log.info("%s", os.path.join(self.mask_dir, target_str, file))
                    raise Exception("No hints available!")

        self.task_specific_urls = None
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

        self.class_restriction_enabled = False
        self.backup_urls = copy.deepcopy(self.urls)

        if self.config.in_memory:
            self.load_in_memory()

    def load_in_memory(self):
        """Cache all images (and masks, if ``has_hints``) of the split as arrays."""
        self.in_memory_images = {}
        for target_str, file in self.urls:
            img = Image.open(os.path.join(self.root_dir, target_str, file))
            if (
                hasattr(self.config, "input_size")
                and self.config.input_size is not None
                and len(self.config.input_size) > 0
                and self.config.input_size[0] == 1
            ):
                img = img.convert("L")
            else:
                img = img.convert("RGB")
            self.in_memory_images[os.path.join(target_str, file)] = np.array(img)

        if self.config.has_hints:
            self.in_memory_masks = {}
            for target_str, file in self.urls:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    mask_path = os.path.join(self.mask_dir, file)

                elif os.path.exists(os.path.join(self.mask_dir, target_str, file)):
                    mask_path = os.path.join(self.mask_dir, target_str, file)

                else:
                    raise Exception(
                        os.path.join(self.mask_dir, target_str, file) + " not found!"
                    )

                self.in_memory_masks[os.path.join(target_str, file)] = np.array(
                    Image.open(mask_path)
                )

    def class_idx_to_name(self, class_idx):
        """Return the directory name of class ``class_idx``."""
        return self.idx_to_name[class_idx]

    def enable_hints(self):
        """Return masks as ``mask`` and restrict ``urls`` to samples that have one."""
        if hasattr(self, "urls_with_hints"):
            self.urls = copy.deepcopy(self.urls_with_hints)
        self.hints_enabled = True

    def disable_hints(self):
        """Stop returning masks and restore the full ``urls`` list."""
        if hasattr(self, "all_urls"):
            self.urls = copy.deepcopy(self.all_urls)
        self.hints_enabled = False

    def enable_string_description(self):
        """Also return the class directory name as ``description``."""
        self.string_description_enabled = True

    def disable_string_description(self):
        """Stop returning the description."""
        self.string_description_enabled = False

    def enable_tokens(self, tokenizer):
        """
        Also return ``tokens``: the description tokenized with ``tokenizer``.

        Parameters
        ----------
        tokenizer : callable
            HuggingFace-style tokenizer with ``model_max_length``.
        """
        self.string_description_enabled_buffer = self.string_description_enabled
        self.enable_string_description()
        self.tokenizer = tokenizer

    def disable_tokens(self):
        """Drop the tokenizer and restore the previous description setting."""
        self.string_description_enabled = self.string_description_enabled_buffer
        self.tokenizer = None

    def enable_idx(self):
        """Also return the sample index."""
        self.idx_enabled = True

    def disable_idx(self):
        """Stop returning the sample index."""
        self.idx_enabled = False

    def enable_url(self):
        """Also return the filename as ``url``."""
        self.url_enabled = True

    def disable_url(self):
        """Stop returning the filename."""
        self.url_enabled = False

    @property
    def output_size(self):
        """Number of classes from ``config.output_size`` (int or one-element list)."""
        if isinstance(self.config.output_size, int):
            return self.config.output_size
        if len(self.config.output_size) == 1:
            return self.config.output_size[0]
        return self.config.output_size

    def set_task_specific_urls(self):
        """
        Subsample ``urls`` so the classes follow ``config.class_ratios``.

        Same scheme as ``Image2MixedDataset.set_task_specific_keys``; the
        original list is saved in ``urls_backup``.
        """
        self.task_specific_urls = []
        num_samples_per_class = np.zeros([self.output_size])
        for idx in range(len(self.urls)):
            target_str, file = self.urls[idx]
            target = int(torch.tensor(self.idx_to_name.index(target_str)))
            num_samples_per_class[target] += 1

        num_units = num_samples_per_class / np.array(self.config.class_ratios)
        min_units = int(np.min(num_units))
        num_samples_per_class_balanced = min_units * np.array(self.config.class_ratios)
        current_num_samples_per_class = np.zeros([self.output_size])
        for idx in range(len(self.urls)):
            target_str, file = self.urls[idx]
            class_idx = int(torch.tensor(self.idx_to_name.index(target_str)))
            if (
                current_num_samples_per_class[class_idx]
                < num_samples_per_class_balanced[class_idx]
            ):
                self.task_specific_urls.append((target_str, file))
                current_num_samples_per_class[class_idx] += 1

        self.urls_backup = copy.deepcopy(self.urls)
        self.urls = self.task_specific_urls

    def __len__(self):
        """Number of samples after ``class_ratios`` balancing, if configured."""
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

        return len(self.urls)

    def enable_class_restriction(self, class_idx: Union[int, list[int]]):
        """
        Keep only samples of the given class(es); the full list goes to ``backup_urls``.

        Parameters
        ----------
        class_idx : int or list of int
            Class index or indices to keep.
        """
        self.backup_urls = copy.deepcopy(self.urls)
        self.urls = []
        self.class_restriction_enabled = True

        if isinstance(class_idx, int):
            allowed_classes = [self.idx_to_name[class_idx]]
        else:
            allowed_classes = [self.idx_to_name[idx] for idx in class_idx]

        for url in self.backup_urls:
            if url[0] in allowed_classes:
                self.urls.append(url)
        _log.info(
            "%s %s", "enabled class restriction for target class(es):", str(class_idx)
        )

    def disable_class_restriction(self):
        """Restore ``backup_urls`` and clear the restriction flag."""
        self.urls = copy.deepcopy(self.backup_urls)
        self.class_restriction_enabled = False
        _log.info("%s", "disabled class restriction")

    def enable_groups(self):
        """Also return ``has_confounder`` (always 0 unless a subclass overrides it)."""
        self.groups_enabled = True

    def disable_groups(self):
        """Stop returning ``has_confounder``."""
        self.groups_enabled = False

    def has_confounder(self, filename: str) -> int:
        """
        Confounder presence for a file; placeholder returning 0.

        Subclasses that know their confounder override this. The base
        implementation prints a warning.

        Parameters
        ----------
        filename : str
            Image filename.

        Returns
        -------
        int
            Always 0.
        """
        _log.info("%s", "'has_confounder' not implemented!!!")
        return 0

    def __getitem__(self, idx):
        """
        Load one sample.

        Parameters
        ----------
        idx : int
            Position in ``urls``.

        Returns
        -------
        tuple or dict
            ``x`` is the transformed image ``[C, H, W]`` and ``y`` the class
            index tensor. Optional entries (``mask``, ``index``, ``url``,
            ``description``, ``tokens``, ``has_confounder``) follow the
            ``enable_*`` flags. Returned as a dict with ``return_dict``, as
            ``(x, y)`` when nothing else is enabled, else ``(x, (y, ...))``.
        """
        if (
            not self.task_config is None
            and not self.config.class_ratios is None
            and self.task_specific_urls is None
        ):
            self.set_task_specific_urls()

        target_str, file = self.urls[idx]

        if self.config.in_memory:
            img = Image.fromarray(self.in_memory_images[os.path.join(target_str, file)])

        else:
            img = Image.open(os.path.join(self.root_dir, target_str, file))

        if (
            hasattr(self.config, "input_size")
            and self.config.input_size is not None
            and len(self.config.input_size) > 0
            and self.config.input_size[0] == 1
        ):
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        state = torch.get_rng_state()
        img = self.transform(img)

        if (
            hasattr(self.config, "input_size")
            and self.config.input_size is not None
            and len(self.config.input_size) > 0
        ):
            if img.shape[0] == 1 and self.config.input_size[0] != 1:
                img = torch.tile(img, [self.config.input_size[0], 1, 1])
            elif img.shape[0] != self.config.input_size[0]:
                img = img[: self.config.input_size[0]]

        # target = torch.zeros([len(self.idx_to_name)], dtype=torch.float32)
        # target[self.idx_to_name.index(target_str)] = 1.0
        return_dict = {"x": img}
        target = torch.tensor(self.idx_to_name.index(target_str))
        return_dict["y"] = target

        if self.hints_enabled:
            if self.config.in_memory:
                mask = Image.fromarray(
                    self.in_memory_masks[os.path.join(target_str, file)]
                )

            else:
                if os.path.exists(os.path.join(self.mask_dir, file)):
                    mask_path = os.path.join(self.mask_dir, file)

                elif os.path.exists(os.path.join(self.mask_dir, target_str, file)):
                    mask_path = os.path.join(self.mask_dir, target_str, file)

                else:
                    raise Exception(
                        os.path.join(self.mask_dir, target_str, file) + " not found!"
                    )

                mask = Image.open(mask_path)

            torch.set_rng_state(state)
            mask = self.transform(mask)
            return_dict["mask"] = mask

        if self.idx_enabled:
            return_dict["index"] = idx

        if self.url_enabled:
            return_dict["url"] = os.path.join(*self.urls[idx])

        if self.string_description_enabled:
            return_dict["description"] = target_str

            if self.tokenizer is not None:
                return_dict["tokens"] = torch.tensor(
                    self.tokenizer(
                        return_dict["description"],
                        max_length=self.tokenizer.model_max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).input_ids
                )

        if self.url_enabled:
            return_dict["url"] = file

        if self.groups_enabled:
            return_dict["has_confounder"] = self.has_confounder(file)

        if self.return_dict:
            return return_dict

        elif len(return_dict.values()) == 2:
            return img, list(return_dict.values())[1]

        else:
            return img, tuple(return_dict.values())[1:]
