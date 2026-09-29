"""Base dataset class and the pydantic ``DataConfig`` shared by all PEAL data.

``PealDataset`` extends ``torch.utils.data.Dataset`` with the hooks the rest of
PEAL relies on: mapping samples between the dataset's processed format and the
"pytorch default" format generators work in, rendering original/counterfactual
collages for teachers, and (optional) distance/variance/flip-rate metrics used
by explainer evaluation. ``DataConfig`` is the yaml-facing description of a
dataset (shapes, splits, normalisation, confounder layout, augmentation) that
``dataset_factory`` and every generator/explainer consume.
"""

from typing import Union

import torch
from pydantic import BaseModel, PositiveInt

from torchvision import transforms

from peal.generators.interfaces import Generator


class PealDataset(torch.utils.data.Dataset):
    """Base class of every dataset in PEAL.

    Subclasses (see ``peal.data.datasets`` and ``custom_datasets``) hold a
    ``config`` (``DataConfig``) and usually a ``normalization`` transform. The
    methods defined here are defaults/hooks: most return placeholders and are
    meant to be overridden when a dataset supports the corresponding feature.
    ``project_to_pytorch_default``/``project_from_pytorch_default`` are the
    bridge between the classifier's input format and the generator's image
    format and are used by every counterfactual explainer.
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
        start_idx: int,
        y_counterfactual_teacher_list=None,
        y_original_teacher_list=None,
        feedback_list=None,
        **kwargs: dict,
    ):
        """Render original/counterfactual pairs for teachers and tracking (hook).

        Subclasses write one collage image per pair under ``base_path`` and
        return the paths plus per-sample attribution maps; teachers such as
        ``BaselineTeacher`` or ``Model2ModelTeacher`` call this to show or log
        the counterfactuals they judge.

        Parameters
        ----------
        x_list, x_counterfactual_list : list of torch.Tensor
            Originals and their counterfactuals, in dataset format.
        y_target_list, y_source_list, y_list : list
            Target class of the edit, class predicted for the original, and
            ground-truth label.
        y_target_start_confidence_list, y_target_end_confidence_list : list
            Student confidence in the target class before and after the edit.
        base_path : str
            Directory to write the collages to.
        start_idx : int
            Offset used to number the written files.
        y_counterfactual_teacher_list, y_original_teacher_list : list, optional
            Teacher-model predictions, if available.
        feedback_list : list, optional
            Feedback strings to print onto the collage.

        Returns
        -------
        torch.Tensor
            This base implementation only returns a ``(3, 64, 64)`` zero tensor.
            Overrides return ``(collage_paths, attribution_list)``.
        """
        return torch.zeros([3, 64, 64])

    def serialize_dataset(self, output_dir, x_list, y_list, sample_names=None):
        """
        This function serializes the dataset to a given directory

        Args:
            output_dir (Path): The output directory
            x_list (list): The list of inputs
            y_list (list): The list of labels
            sample_names (list, optional): The list of sample names. Defaults to None.
        """

    def project_to_pytorch_default(self, x):
        """
        This function maps processed data sample back to pytorch default format

        Args:
            x (torch.tensor): The data sample in the processed format

        Returns:
            torch.tensor: The data sample in the pytorch default format
        """
        if hasattr(self, "normalization"):
            x = self.normalization.invert(x)

        return x

    def project_from_pytorch_default(self, x):
        """
        This function maps pytorch default image to the processed format

        Args:
            x (torch.tensor): The data sample in the pytorch default format

        Returns:
            torch.tensor: The data sample in the processed format
        """
        if hasattr(self, "normalization"):
            x = self.normalization(x)

        if list(x.shape[-3:]) != self.config.input_size:
            x = transforms.Resize(self.config.input_size[1:])(x)

        return x

    def track_generator_performance(self, generator: Generator, batch_size=1):
        """
        This function tracks the performance of the generator

        Args:
            generator (Generator): The generator
        """
        return {}

    def distribution_distance(self, x_list):
        """Hook: distance of ``x_list`` to the data distribution (returns None)."""

    def pair_wise_distance(self, x1, x2):
        """Hook: distance between two samples in dataset units (returns None)."""

    def variance(self, x_list):
        """Hook: diversity/variance of a list of samples (returns None)."""

    def flip_rate(self, y_list, y_counterfactual_list):
        """Hook: fraction of counterfactuals whose label differs (returns None)."""


class DataConfig(BaseModel):
    """Pydantic description of a dataset, loadable from yaml.

    The individual fields are documented by the string literals that precede
    them in the source. The fields without such a literal configure diffusion
    based data augmentation and are consumed by ``dataset_factory`` and
    ``peal.data.transformations.DiffusionAugmentation``.

    Parameters
    ----------
    generator : str or object or None
        Generator (config path, config or instance) used for augmentation.
    diffusion_augmented : bool
        If ``True`` the training transform noises every sample with
        ``generator`` and denoises it again (a stochastic augmentation).
    sampling_time_fraction : float
        Fraction of the diffusion schedule to noise to (default 0.3).
    num_discretization_steps : int
        Number of denoising steps used by the augmentation (default 20).
    batch_wise_augmentation : bool
        If ``True`` (default) the per-sample ``DiffusionAugmentation``
        transform is not added to the dataset; augmentation is expected to be
        applied to whole batches elsewhere.
    """

    config_name: str = "DataConfig"
    """
    The type of config. This is necessary to find config class from yaml config
    """
    input_type: str = "image"
    """
    The input type of the data:
    Options: ['image', 'sequence', 'symbolic']
    """
    output_type: str = "singleclass"
    """
    The output type of the data.
    Options: ['singleclass', 'multiclass', 'continuous', 'mixed']
    'mixed' is a hybrid between binary multiclass classification and continuous and
    requires 'output_split' to be set
    """
    input_size: list[PositiveInt] = [3, 128, 128]
    """
    The input size of data.
    For images: [Channels, Height, Width]
    For sequences: [MaxLength, NumTokens]
    For symbolic: [NumVariables]
    """
    output_size: list[PositiveInt] = [2]
    """
    The output size of the model.
    For singleclass: [NumClasses]
    For multiclass: [NumBinaryClasses]
    For continuous: [NumVariables]
    For mixed: [NumBinaryClasses + NumVariables]
    """
    dataset_path: Union[type(None), str] = None
    """
    The path to the dataset.
    """
    dataset_origin_path: Union[type(None), str] = None
    """
    The path to the dataset origin this dataset got derived from.
    """
    num_samples: Union[type(None), int] = None
    """
    The number of samples in the dataset.
    Sometimes important when executing specific experiments.
    """
    dataset_class: Union[type(None), str] = None
    """
    The name of the dataset.
    Only necessary to tell dataset factory which customized dataset class to use
    """
    split: list[float] = [0.8, 0.9]
    """
    The split between train, validation and test set.
    """
    has_hints: Union[type(None), bool] = False
    """
    Whether the dataset contains spatial annotations where the true feature is.
    """
    normalization: Union[type(None), list] = None
    """
    The applied normalization.
    Options: ['mean0std1']
    """
    invariances: list[str] = []
    """
    A list of the invariances exploited for data augmentation:
    Options: ['hflipping', 'vflipping', 'rotation', 'circlecut']
    """
    output_split: Union[type(None), int] = None
    """
    The number of binary multiclass variables in the mixed setting.
    Has to be smaller than output_size.
    """
    downsize: Union[type(None), str] = None
    """
    The way how to downsize an sample if required.
    Options: ['Downsample', 'RandomCrop', 'CenterCrop']
    """
    confounding_factors: Union[type(None), list[str]] = []
    """
    A pair of known confounding factors, one usually being the target.
    This knowledge helps for controlled sampling of confounders for experiments.
    """
    foreground: Union[type(None), list[str]] = None
    """
    The foreground categories to use for the experiment.
    """
    background: Union[type(None), list[str]] = None
    """
    The background contexts to use for the experiment.
    """
    confounder_probability: Union[type(None), float] = None
    """
    The correlation strength of the target and the confounding variable.
    """
    full_confounder_config: Union[type(None), list[float]] = None
    """
    alternative to confounder_probability, specify individual group sizes, e.g. [0.25, 0.25, 0.25, 0.25]
    """
    class_ratios: Union[type(None), list] = None
    """
    The ratio of the classes in the dataset.
    """
    seed: Union[type(None), int] = 0
    """
    The seed the dataset was generated with.
    Only relevant for generated datasets!
    """
    label_noise: Union[type(None), float] = None
    """
    The label noise of a generated dataset.
    Necessary to mimic real dataset behauviour and avoid trivial non-robust solutions.
    """
    set_negative_to_zero: bool = True
    """
    Whether to set negative values to zero.
    """
    font_path: Union[type(None), str] = None
    """
    The path to the font file.
    """
    delimiter: Union[type(None), str] = ","
    """
    The delimiter used for the csv file.
    """
    crop_size: Union[type(None), int] = None
    """
    The number of classes in the dataset.
    """
    dataset_origin_path: Union[type(None), str] = None
    """
    The path of the original dataset.
    """
    inverse: Union[type(None), str] = None
    """
    The path of the original dataset.
    """
    label_rel_path: str = "data.csv"
    """
    Where the label path is relative to the dataset path.
    """
    x_selection: str = "imgs"
    """
    The name of the folder where the images are stored.
    The header of the column in the underlying csv either has to be called like this as well or the path to the images
    has to be in the first column.
    """
    in_memory: bool = False
    """
    Whether to load all datasets into the RAM or not. Careful with big datasets!
    """
    spray_label_file: Union[str, type(None)] = None
    """
    Path to spray label csv file. When set, use spray labels instead of true confounder labels.
    Samples without a spray label will be dropped
    """
    spray_groups_balanced: bool = False
    """
    Whether to re-balance group sizes after dropping samples without a spray label
    """
    generator: Union[type(None), str, object] = None
    diffusion_augmented: bool = False
    sampling_time_fraction: float = 0.3
    num_discretization_steps: int = 20
    batch_wise_augmentation: bool = True
    tabular_preprocessing: Union[type(None), list[str]] = None
    """
    List of preprocessing methods to apply to tabular datasets.
    Options: ``['minmax_-1_1']``
    """
