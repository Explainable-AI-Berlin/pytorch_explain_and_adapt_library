"""Counterfactual Knowledge Distillation (CFKD), PEAL's central repair loop.

CFKD removes a Clever-Hans strategy from a trained classifier by showing a
teacher what the classifier reacts to and finetuning on the verdicts. One
iteration draws a class-balanced pool of factuals, asks the configured
explainer for a counterfactual per sample, has the teacher label each
factual/counterfactual pair as a valid class change ("true") or a change of a
spurious feature ("false"), serialises the pairs with feedback-corrected
labels into a dataset, and finetunes the student on a mixture of that dataset
and the original training data.

The loop repeats for ``finetune_iterations`` rounds. Every round writes its
artefacts under ``base_dir/<iteration>/`` so a crashed run resumes from
``current_iteration`` in ``config.yaml`` instead of recomputing the
counterfactuals, which dominate the cost. Validation statistics are logged to
TensorBoard as ``validation_*`` scalars, and the repaired student is saved as
``model.cpl``.

The method is described in "Mitigating Clever Hans Strategies in Image
Classifiers through Generating Counterexamples"
(https://arxiv.org/pdf/2510.17524); ``reproduction_scripts/reproduce_cfkd_results.sh``
reproduces the paper's numbers.

See Also
--------
peal.adaptors.didae : discovers the directions to correct in a generator
    latent instead of per-sample counterfactuals, and reuses CFKD for step 9.
peal.explainers.counterfactual_explainer : produces the counterfactuals.
peal.teachers : the feedback sources CFKD queries.
"""

import math
import os
import torch
import copy
import shutil
import torchvision
import numpy as np
import platform

from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from torch import nn
from types import SimpleNamespace
from pydantic import PositiveInt
from typing import Union

from peal.architectures.predictors import TorchvisionModel, get_predictor
from peal.global_utils import load_yaml_config, save_yaml_config, cprint
from peal.sparse_dictionaries.interfaces import SparseDictionaryConfig
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.training.loggers import log_images_to_writer
from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer,
    create_dataloaders_from_datasource,
    WeightedDataloaderList,
)
from peal.training.trainers import (
    ModelTrainer,
    calculate_test_accuracy,
    distill_predictor,
)
from peal.explainers.counterfactual_explainer import (
    PerfectFalseCounterfactualConfig,
    flatten_explanations,
)
from peal.visualization.model_comparison import (
    create_comparison,
)
from peal.teachers.segmentation_mask_teacher import SegmentationMaskTeacher
from peal.data.datasets import ImageDataset, Image2MixedDataset
from peal.teachers.teacher_factory import get_teacher
from peal.teachers.interfaces import TeacherInterface
from peal.generators.interfaces import InvertibleGenerator
from peal.generators.generator_factory import get_generator
from peal.training.training_utils import (
    calculate_validation_statistics,
)
from peal.data.interfaces import DataConfig
from peal.generators.interfaces import GeneratorConfig
from peal.training.interfaces import TrainingConfig, PredictorConfig
from peal.architectures.interfaces import TaskConfig
from peal.explainers.interfaces import ExplainerConfig
from peal.explainers.explainer_factory import get_explainer
from peal.explainers.counterfactual_explainer import SCEConfig
from peal.adaptors.interfaces import AdaptorConfig, Adaptor
from peal.log import get_logger

_log = get_logger(__name__)


class CFKDConfig(AdaptorConfig):
    """
    The config template for an running the CFKD adaptor.

    Loaded with ``load_yaml_config(..., CFKDConfig)`` from a yaml whose
    ``adaptor_type`` is ``"CFKD"``. Each field is described by the string
    literal that follows it in the class body, which is what Sphinx renders as
    that field's attribute documentation. The few fields without such a string
    are:

    Parameters
    ----------
    sparse_dictionary : SparseDictionaryConfig, dict or None
        If set, this dictionary is loaded and attached to the generator,
        overriding whatever the generator loaded itself; ``component_indices``
        index into it.
    correct_clusters : list of int
        Cluster indices treated as correct by teachers that judge clusters.
    use_true_counterfactuals : bool
        Whether "true" counterfactuals (labelled with the target class) are
        added to the finetune dataset besides the "false" ones.
    seed : int or None
        Random seed of the run.
    component_indices : list of int or None
        Dictionary components the explainer edits along; when given, the
        explainer's ``component_indices`` and ``num_attempts`` are set from it.

    Notes
    -----
    Fields such as ``current_iteration``, ``feedback_accuracies``,
    ``group_accuracies`` and ``model_path`` are progress logs that CFKD writes
    back into ``base_dir/config.yaml`` after every iteration, which is what
    makes a run resumable.
    """

    adaptor_type: str = "CFKD"
    """
    The adaptor_type for CFKDConfig has to be CFKD to find the CFKDConfig class when loading from a yaml file.
    """
    min_train_samples: PositiveInt = 800
    """
    The minimum number of samples used for finetuning in every iteration.
    The actual number could be higher since not for every sample a counterfactual can be found
    and processing is done in batches.
    """
    max_validation_samples: PositiveInt = 200
    """
    The maximum number of validation samples that are used for tracking stats every iteration.
    """
    max_test_batches: Union[type(None), PositiveInt] = None
    """
    The maximum number of test batches.
    If set to None the test will be done on the full test set.
    """
    finetune_iterations: int = 1
    """
    The number of finetune iterations when executing the adaptor.
    If set to 0 only the explanation and no adaption is done.
    """
    task: Union[TaskConfig, type(None)] = None
    """
    The config of the task the student model shall solve.
    """
    explainer: Union[dict, ExplainerConfig] = SCEConfig()
    """
    The config of the counterfactual explainer that is used.
    All parameters regarding paths, where the generator is from etc in there are overwritten by CFKD and only
    used if the information is not available for CFKD
    """
    training: Union[TrainingConfig, type(None)] = TrainingConfig()
    """
    The config of the training used for finetuning the student model.
    If not set student config can be used.
    """
    data: DataConfig = None
    """
    The config of the data used to create the counterfactuals from.
    """
    test_data: DataConfig = None
    """
    The config of the test data used evaluate the real progress on.
    Often this data has a distribution shift compared to the training data or comes from a totally different data source.
    Hence the option to give its own config.
    If set to None the normal data config is taken.
    """
    student: Union[PredictorConfig, str, type(None)] = None
    """
    The path of the student used.
    Can be either the path to a PyTorch or an onnx model directly or the path to a predictor config or a PredictorConfig
    object.
    """
    teacher: Union[str, dict] = "cluster@8000"
    """
    The type of teacher used.
    """
    generator: Union[GeneratorConfig, type(None)] = None
    """
    The config of the generator used.
    This value will be overwritten if Generator is given via constructor directly.
    If the Generator is not given via constructor and this value is set to None explainer config is searched for
    generator config.
    """
    base_dir: str = "peal_runs/cfkd"
    """
    The base directory where the run of CFKD is stored.
    All the visualizations and the caching is stored here.
    """
    current_iteration: int = 0
    """
    Logging of the current finetune iteration
    """
    continuous_learning: str = "finetune"
    """
    Whether to continue training from the current student model or start training from scratch
    again. Can e.g. be "retrain", which retrains model on original data and counterfactuals from scratch,
    "finetune", which starts of at the weights of the uncorrected student or "deep_feature_reweighting", which
    only finetunes the last layer of the the uncorrected student.
    """
    use_confusion_matrix: bool = False
    """
    Whether to draw samples for counterfactual creation according to the error matrix or not.
    Makes particular sense in the multiclass setting where some classes might be in very
    different modes and one only wants to restrict to connected modes.
    """
    best_feedback_accuracy: float = 0.0
    """
    Logging of the Feedback Accuracy.
    """
    attribution_threshold: float = 0.0
    """
    The attribution threshold when using the SegmentationMask teacher.
    Setting it to 0.0 means that every counterfactual that did on average bigger changes inside than outside the mask
    is considered a True counterfactual and every counterfactual that does not is considered a false counterfactual.
    If it is set higher the bar for True Counterfactual is set higher and if is set lower the bar is set lower as well.
    """
    batch_size: PositiveInt = 1
    """
    What batch_size is used for creating the counterfactuals?
    """
    validation_runs: PositiveInt = 1
    """
    The number validation runs used for evaluating CFKD.
    """
    calculate_group_accuracies: bool = False
    """
    Whether to calculate group accuracies or not. This can only be done if confounding factors are known.
    """
    overwrite: bool = True
    """
    Whether to overwrite the logs and cache intermediate results.
    If overwrite is set to False cached results are loaded. If CFKDConfig is stored as yaml on disk overwrite is 
    automatically set to False so that CFKD can be continued at the last cached result.
    Using this feature dramatically improves ability to debug!
    """
    mixing_ratio: float = 0.5
    """
    How aggressively to change the model based on the counterfactual samples. 0 -> No change, 1 -> Full change
    """
    feedback_accuracies: list = []
    """"
    A list of the feedback accuracies.
    """
    group_accuracies: list = []
    """
    The group accuracies of the model after every finetuning iteration.
    """
    avg_group_accuracies: list = []
    """
    The average group accuracies of the model after every finetuning iteration.
    """
    counterfactual_type: str = "1sided"
    """
    What type of counterfactuals are valid. 1sided means that we can only start from samples with correct prediction,
    2sided also allows that we start from samples with wrong original prediction.
    """
    lazy_feedback: bool = True
    """
    Whether to always give feedback directly after creating validation counterfactuals or whether to wait until
    the next train feedback shall be given as well (which means less interruptions!)
    """
    model_path: str = ""
    """
    The path of the last finetuned model
    """
    is_loaded: bool = False
    """
    Dummy field to be able to use it as a model config!
    """
    generator_performance: dict = {}
    """
    The performance of the generative model measured e.g. in FID score.
    """
    transition_restrictions: Union[list, type(None)] = None
    """
    The restriction to interesting counterfactual transitions.
    Helpful in the case of datasets with a lot of classes and heavy modes like ImageNet.
    """
    clustering_strategy: Union[str, type(None)] = "attempt_nr"
    """
    The clustering strategy used by the counterfactual explainer
    """
    sparse_dictionary: Union[SparseDictionaryConfig, dict, type(None)] = None
    correct_clusters: list = [0]
    use_true_counterfactuals: bool = False
    seed: Union[int, type(None)] = 0
    visualize_latent_sparsity: bool = True
    """
    Whether to visualize the individual latent sparsity scores as collages.
    Each collage contains: factual, counterfactual, target end confidence, sparsity score.
    """
    visualize_latent_diversity: bool = True
    """
    Whether to visualize the individual latent diversity scores as collages.
    Each collage contains: factual, cf1, cf2, target end confidences, diversity score.
    """
    explainer_stats_clusters: list = [0, 1]
    """
    The cluster indices used for calculating explainer stats (sparsity, diversity, feedback metrics).
    Only these clusters are considered when computing latent difference statistics and feedback stats.
    """
    component_indices: Union[list, type(None)] = None


class CFKD(Adaptor):
    """
    This class implements the counterfactual knowledge distillation approach.

    The constructor wires up everything one run needs (student, dataloaders,
    generator, explainer, teacher, sparse dictionary); :meth:`run` executes the
    finetune iterations and returns the corrected student. See the module
    docstring for the algorithm and the per-iteration artifacts.

    Parameters
    ----------
    student : torch.nn.Module, optional
        The classifier to repair. If ``None`` it is built from
        ``adaptor_config.student`` with ``get_predictor``.
    datasource : list or tuple, optional
        Datasets/dataloaders handed to ``create_dataloaders_from_datasource``;
        ``None`` means the data is built from ``adaptor_config.data``.
    generator : InvertibleGenerator, Path or str, optional
        Generator instance or config path; falls back to
        ``adaptor_config.generator``.
    base_dir : str or Path, optional
        Run directory; falls back to ``adaptor_config.base_dir``.
    teacher : str or TeacherInterface, optional
        Teacher spec (e.g. ``"human@8000"``, ``"SegmentationMask"``, a dict
        ``{type: llm}``) or instance; falls back to ``adaptor_config.teacher``.
    adaptor_config : dict, str, Path or CFKDConfig
        The run configuration (yaml path, dict or object).
    overwrite : bool, optional
        Whether cached artifacts in ``base_dir`` are discarded; falls back to
        ``adaptor_config.overwrite``. The stored config always gets
        ``overwrite=False`` so a resumed run reuses its cache.

    Attributes
    ----------
    original_student : torch.nn.Module
        The uncorrected student, kept for the progress visualisation.
    student : torch.nn.Module
        The student being finetuned (a deep copy of the input).
    train_dataloader, val_dataloader, test_dataloader
        The original data; ``val_batch_size`` is forced to the CFKD
        ``batch_size`` because validation batches go straight into the explainer.
    dataloader_mixer : DataloaderMixer
        Original data plus every counterfactual dataset added so far.
    datastack : DataStack
        Per-class sample buffer the batch builder pops factuals from.
    tracked_keys : list of str
        Keys of the explainer output that are collected and cached (extended
        with ``cluster_list``, ``z_difference_list``/``collage_path_list`` at
        ``tracking_level >= 4``, ``hint_list`` and ``idx_list`` as needed).
    data_config, validation_data_config : CFKDConfig
        Copies of the config used to load the serialised counterfactual
        datasets for training and validation respectively.
    """

    def __init__(
        self,
        student: nn.Module = None,
        datasource: Union[list, tuple] = None,
        generator: Union[InvertibleGenerator, Path, str] = None,
        base_dir: Union[str, Path] = None,
        teacher: Union[str, TeacherInterface] = None,
        adaptor_config: Union[
            dict, str, Path, AdaptorConfig
        ] = "<PEAL_BASE>/configs/adaptors/symbolic_cfkd.yaml",
        overwrite: bool = None,
    ):
        """
        Build all components of a CFKD run; see the class docstring for the
        parameters.

        Besides constructing the objects this also mutates ``adaptor_config``:
        ``test_data`` defaults to ``data``, ``in_memory``/``tracking_level``/
        ``transition_restrictions``/``clustering_strategy`` are pushed into the
        explainer config, ``training.val_batch_size`` is set to ``batch_size``
        and ``training.steps_per_epoch`` to one pass over the training set.
        A collage-based teacher (human, web, llm) requires
        ``tracking_level >= 4`` and fails an assertion otherwise.
        """
        # CFKDConfig, not the AdaptorConfig base: run_cfkd.py hands in an already
        # built CFKDConfig object (which load_yaml_config passes through untouched,
        # so the base class was harmless there), but DiDAE step 9 hands in a plain
        # dict, and parsing that as AdaptorConfig drops every CFKD-only field —
        # `explainer` first of all, one line below.
        self.adaptor_config = load_yaml_config(adaptor_config, CFKDConfig)
        if getattr(self.adaptor_config, "component_indices", None) is not None:
            comp_indices = self.adaptor_config.component_indices
            if isinstance(self.adaptor_config.explainer, dict):
                self.adaptor_config.explainer["component_indices"] = comp_indices
                self.adaptor_config.explainer["num_attempts"] = len(comp_indices)
            else:
                self.adaptor_config.explainer.component_indices = comp_indices
                self.adaptor_config.explainer.num_attempts = len(comp_indices)

        self.min_train_samples = (
            self.adaptor_config.explainer.num_attempts
            * self.adaptor_config.min_train_samples
        )
        if self.adaptor_config.test_data is None:
            self.adaptor_config.test_data = self.adaptor_config.data

        self.adaptor_config.data.in_memory = self.adaptor_config.in_memory
        self.adaptor_config.test_data.in_memory = self.adaptor_config.in_memory
        self.adaptor_config.explainer.tracking_level = (
            self.adaptor_config.tracking_level
        )
        self.adaptor_config.explainer.transition_restrictions = (
            self.adaptor_config.transition_restrictions
        )
        if not self.adaptor_config.clustering_strategy is None:
            self.adaptor_config.explainer.clustering_strategy = (
                self.adaptor_config.clustering_strategy
            )

        self.base_dir = (
            base_dir if not base_dir is None else self.adaptor_config.base_dir
        )
        #
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if student is None:
            student, student_config = get_predictor(
                self.adaptor_config.student, device=self.device
            )

        self.overwrite = (
            overwrite if not overwrite is None else self.adaptor_config.overwrite
        )
        self.adaptor_config.overwrite = False
        self.original_student = student
        if isinstance(student, torch.nn.Module):
            self.original_student.eval()
            self.student = copy.deepcopy(student)
            self.student.eval()

        else:
            self.student = student

        teacher = teacher if not teacher is None else self.adaptor_config.teacher

        # kind of dirty, but also very confusing if not done this way since validation batches are fed directly
        # into the explainer and thereby potentially causing VRAM overflows otherwise
        _log.info(
            "%s", "changing val dataloader batch size to equals adapter batch size!!!! "
        )
        self.adaptor_config.training.val_batch_size = self.adaptor_config.batch_size

        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = create_dataloaders_from_datasource(
            datasource=datasource,
            config=self.adaptor_config,
            test_config=self.adaptor_config.test_data,
            enable_hints=bool(teacher == "SegmentationMask"),
        )
        self.joint_validation_dataloader = WeightedDataloaderList([self.val_dataloader])
        self.adaptor_config.data = self.train_dataloader.dataset.config

        #
        explainer_config = load_yaml_config(self.adaptor_config.explainer)
        if hasattr(explainer_config, "num_discretization_steps") and hasattr(
            explainer_config, "sampling_time_fraction"
        ):
            timestep_respacing = int(
                1.0
                / explainer_config.sampling_time_fraction
                * explainer_config.num_discretization_steps
            )

        elif (
            hasattr(explainer_config, "timestep_respacing")
            and not explainer_config.timestep_respacing is None
        ):
            timestep_respacing = explainer_config.timestep_respacing

        else:
            timestep_respacing = None
        self.generator = get_generator(
            generator=(
                generator if not generator is None else self.adaptor_config.generator
            ),
            device=self.device,
            predictor_dataset=self.val_dataloader.dataset,
            timestep_respacing=timestep_respacing,
        )
        if not self.adaptor_config.sparse_dictionary is None:
            # An explicit adaptor_config.sparse_dictionary always wins over whatever the
            # generator loaded from its own config: component_indices are indices into
            # *this* dictionary, so silently keeping the generator's (differently sized)
            # one produces out-of-range / mismatched directions.
            _log.info(
                "%s",
                "CFKD: loading sparse dictionary from "
                "adaptor_config.sparse_dictionary ...",
            )
            self.generator.sparse_dictionary = get_sparse_dictionary(
                self.adaptor_config.sparse_dictionary
            )
            _log.info("%s", "CFKD: sparse dictionary loaded and attached to generator.")

        self.output_size = (
            self.adaptor_config.task.output_channels
            if self.adaptor_config.task.output_channels is not None
            else self.adaptor_config.data.output_size[0]
        )

        #
        outlier_scores_absolute = self.val_dataloader.dataset.calculate_outlier_score(
            next(iter(self.train_dataloader))[0]
        )
        self.val_dataloader.dataset.reference_outlier_scores = torch.mean(
            outlier_scores_absolute["absolute"]
        ).item()
        self.teacher = get_teacher(
            teacher=teacher,
            output_size=self.output_size,
            adaptor_config=self.adaptor_config,
            dataset=self.val_dataloader.dataset,
            device=self.device,
            tracking_level=self.adaptor_config.tracking_level,
        )
        if self.adaptor_config.training.steps_per_epoch is None:
            self.adaptor_config.training.steps_per_epoch = (
                len(self.train_dataloader.dataset)
                // self.adaptor_config.training.train_batch_size
            )

        self.dataloader_mixer = DataloaderMixer(
            self.adaptor_config.training, self.train_dataloader
        )
        self.datastack = DataStack(
            self.dataloader_mixer,
            self.output_size,
            transform=self.val_dataloader.dataset.transform,
        )

        # Teachers that judge rendered collages need `collage_path_list`, and that
        # key is only added to tracked_keys at tracking_level >= 4 (see above).
        # Below that, get_feedback(**tracked_values) raises
        #   TypeError: get_feedback() missing 1 required positional argument:
        #   'collage_path_list'
        # ~20 minutes in, after the counterfactuals have been generated. The
        # check used to cover only `teacher: "human@8000"`, a string; the LLM
        # teacher is configured as a dict ({type: llm}) and slipped past it.
        #
        # This bites in practice because tracking_level >= 4 is also what makes
        # CFKD's accumulation big enough to hit a 16 GiB cgroup: lowering it is
        # the obvious memory mitigation, and it silently disables these teachers.
        needs_collages = (
            isinstance(teacher, str)
            and (teacher[:5] == "human" or teacher[:4] == "web:")
        ) or (
            isinstance(teacher, dict) and teacher.get("type") in ("llm", "human", "web")
        )
        if needs_collages:
            assert self.adaptor_config.tracking_level >= 4, (
                f"tracking_level {self.adaptor_config.tracking_level} is too low for a "
                "collage-based teacher: collage_path_list is only tracked at >= 4. "
                "Raise tracking_level, or use a teacher that does not read collages "
                "(a .cpl Model2ModelTeacher)."
            )

        self.explainer = get_explainer(
            explainer=self.adaptor_config.explainer,
            predictor=self.student,
            generator=self.generator,
            input_type=self.adaptor_config.data.input_type,
            datasource=[self.dataloader_mixer, self.joint_validation_dataloader],
            tracking_level=self.adaptor_config.tracking_level,
        )
        self.logits_to_prediction = lambda logits: logits.argmax(-1)
        self.tracked_keys = [
            "x_counterfactual_list",
            "y_source_list",
            "y_target_list",
            "y_target_end_confidence_list",
            "x_list",
            "y_list",
            "x_attribution_list",
            "y_target_start_confidence_list",
        ]

        # cluster_list has to reach the teacher whenever the *teacher* is the
        # preclustered one, not only when the clustering *strategy* is
        # "preclustered". PreclusteredTeacher.get_feedback requires it, and a
        # DiDAE run pairs that teacher with the default attempt_nr strategy
        # (see configs/didae_experiments/adaptors/square1000x080_didae_preclustered_cfkd.yaml),
        # which otherwise never tracks the key.
        if (
            getattr(self.adaptor_config.explainer, "clustering_strategy", None)
            == "preclustered"
            or self.adaptor_config.teacher == "preclustered"
        ):
            self.tracked_keys.append("cluster_list")

        if self.adaptor_config.tracking_level >= 4:
            self.tracked_keys.extend(
                [
                    "z_difference_list",
                    "collage_path_list",
                ]
            )

        # teacher == "SegmentationMask" or self.adaptor_config.tracking_level > 0:
        if self.adaptor_config.data.has_hints:
            self.hints_enabled = True
            self.tracked_keys.append("hint_list")
            self.train_dataloader.dataset.enable_hints()
            self.val_dataloader.dataset.enable_hints()

        else:
            self.hints_enabled = False

        if isinstance(
            self.explainer.explainer_config, PerfectFalseCounterfactualConfig
        ):
            self.tracked_keys.append("idx_list")
            self.train_dataloader.dataset.enable_idx()
            self.val_dataloader.dataset.enable_idx()
            self.test_dataloader.dataset.enable_idx()

        self.data_config = copy.deepcopy(self.adaptor_config)
        self.data_config.data.split = [1.0, 1.0]
        self.data_config.data.confounding_factors = []
        self.data_config.data.confounder_probability = None
        self.data_config.data.output_type = "singleclass"
        self.data_config.data.output_size = self.train_dataloader.dataset.output_size
        self.data_config.data.delimiter = ","
        self.data_config.data.x_selection = "imgs"
        self.data_config.data.num_samples = self.min_train_samples
        self.data_config.data.dataset_class = None
        self.validation_data_config = copy.deepcopy(self.data_config)
        self.validation_data_config.data.x_selection = "imgs"
        self.validation_data_config.data.num_samples = (
            self.adaptor_config.max_validation_samples
        )
        self.validation_data_config.data.split = [0.0, 1.0]
        self.validation_data_config.training.val_batch_size = 2

    def initialize_run(self):
        """Prepare ``base_dir`` and compute or reload the iteration-0 state.

        On a fresh run (``overwrite`` or no ``logs`` dir yet) the existing
        ``base_dir`` is moved to ``<base_dir>_old_<timestamp>``, the initial
        validation/test (group) accuracies, sample batches and a generator
        sample (with FID) are written to TensorBoard under ``base_dir/logs``,
        ``config.yaml`` and ``platform.txt`` are written and the validation
        counterfactuals of iteration 0 are generated. On a resumed run the
        cached validation values are loaded instead, every counterfactual
        dataset up to ``current_iteration`` is re-added to the dataloader
        mixer and the last ``model.cpl`` becomes the student. Optionally a
        decision-boundary plot (2-D latent datasets, ``tracking_level >= 4``)
        and the progress figure (binary tasks, ``tracking_level >= 6``) are
        rendered.

        Returns
        -------
        tuple
            ``(validation_stats, validation_tracked_values, writer)``: the
            statistics dict of the last completed validation pass, the tracked
            validation counterfactuals and the ``SummaryWriter``.
        """
        cprint("initialize run!!!", self.adaptor_config.tracking_level, 2)
        if self.overwrite:
            # move from self.base_dir to self.base_dir + "_old_" + {date}_{timestamp}
            if os.path.exists(self.base_dir):
                shutil.move(
                    self.base_dir,
                    self.base_dir + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
                )

        if not os.path.exists(os.path.join(self.base_dir, "0")):
            os.makedirs(os.path.join(self.base_dir, "0"))

        boundary_path = os.path.join(self.base_dir, "0", "decision_boundary.png")
        if (
            self.adaptor_config.tracking_level >= 4
            and not os.path.exists(boundary_path)
            and hasattr(
                self.joint_validation_dataloader.dataloaders[0].dataset,
                "sample_to_2d_latent",
            )
        ):
            self.joint_validation_dataloader.dataloaders[
                0
            ].dataset.visualize_decision_boundary(
                self.student,
                self.adaptor_config.training.test_batch_size,
                self.device,
                boundary_path,
                temperature=self.adaptor_config.explainer.temperature,
                train_dataloader=self.dataloader_mixer,
                val_dataloaders=self.joint_validation_dataloader,
                test_dataloader=self.test_dataloader,
            )

        log_dir = os.path.join(self.base_dir, "logs")

        if not os.path.exists(log_dir):
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir)

            hints_enabled_buffer = self.val_dataloader.dataset.hints_enabled
            if hints_enabled_buffer:
                self.val_dataloader.dataset.disable_hints()
            val_accuracy = calculate_test_accuracy(
                self.student,
                self.val_dataloader,
                self.device,
                False,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )
            cprint("val_accuracy: ", self.adaptor_config.tracking_level, 2)
            writer.add_scalar(
                "val_accuracy", val_accuracy, self.adaptor_config.current_iteration
            )
            if hints_enabled_buffer:
                self.val_dataloader.dataset.enable_hints()

            test_accuracy = calculate_test_accuracy(
                self.student,
                self.test_dataloader,
                self.device,
                self.adaptor_config.calculate_group_accuracies,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )
            if self.adaptor_config.calculate_group_accuracies:
                (
                    test_accuracy,
                    group_accuracies,
                    group_distribution,
                    groups,
                    worst_group_accuracy,
                ) = test_accuracy
                for idx in range(len(group_accuracies)):
                    writer.add_scalar(
                        "test_group_accuracy_" + str(idx),
                        group_accuracies[idx],
                        self.adaptor_config.current_iteration,
                    )
                    writer.add_scalar(
                        "test_group_distribution_" + str(idx),
                        group_distribution[idx],
                        self.adaptor_config.current_iteration,
                    )

                writer.add_scalar(
                    "test_worst_group_accuracy",
                    worst_group_accuracy,
                    self.adaptor_config.current_iteration,
                )
                cprint(
                    "group_accuracies: " + str(group_accuracies),
                    self.adaptor_config.tracking_level,
                    2,
                )
                self.adaptor_config.group_accuracies.append(group_accuracies)
                save_yaml_config(
                    self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
                )
                cprint(
                    "group_distribution: " + str(group_distribution),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint(
                    "group_numbers: " + str(groups),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint(
                    "worst_group_accuracy: " + str(worst_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                avg_group_accuracy = float(np.mean(group_accuracies))
                cprint(
                    "avg_group_accuracy: " + str(avg_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                self.adaptor_config.avg_group_accuracies.append(avg_group_accuracy)
                writer.add_scalar(
                    "test_avg_group_accuracy",
                    avg_group_accuracy,
                    self.adaptor_config.current_iteration,
                )

            writer.add_scalar(
                "test_accuracy", test_accuracy, self.adaptor_config.current_iteration
            )
            cprint("log sample batches!", self.adaptor_config.tracking_level, 2)
            log_images_to_writer(self.train_dataloader, writer, "train0")
            log_images_to_writer(self.val_dataloader, writer, "validation0")
            log_images_to_writer(self.test_dataloader, writer, "test")
            cprint("log sample batches done!", self.adaptor_config.tracking_level, 2)

            if (
                isinstance(self.val_dataloader.dataset, ImageDataset)
                and self.adaptor_config.tracking_level >= 4
            ):
                cprint("visualizing sample!!!", self.adaptor_config.tracking_level, 2)
                generator_sample = self.generator.sample_x()
                if not generator_sample is None:

                    torchvision.utils.save_image(
                        generator_sample,
                        os.path.join(self.base_dir, "generator_sample.png"),
                        normalize=True,
                        nrow=int(np.sqrt(generator_sample.shape[0])),
                    )
                    cprint("sample visualized!", self.adaptor_config.tracking_level, 2)
                    # TODO move this back!!!
                    generator_performance = (
                        self.val_dataloader.dataset.track_generator_performance(
                            generator_sample
                        )
                    )
                    cprint(
                        "Generator performance: " + str(generator_performance),
                        self.adaptor_config.tracking_level,
                        2,
                    )
                    writer.add_scalar(
                        "generator_fid",
                        generator_performance["fid"],
                        self.adaptor_config.current_iteration,
                    )
                    self.adaptor_config.generator_performance = generator_performance

                else:
                    # TODO why was a pdb here?
                    cprint(
                        "log sample batches done!",
                        self.adaptor_config.tracking_level,
                        2,
                    )

            else:
                cprint("no visualization!!!", self.adaptor_config.tracking_level, 2)

        else:
            from torch.utils.tensorboard import SummaryWriter

            writer = SummaryWriter(log_dir)

        if not os.path.exists(
            os.path.join(self.base_dir, "0", "validation_tracked_values.npz")
        ):
            assert self.adaptor_config.current_iteration == 0

            save_yaml_config(
                self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
            )

            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            cprint(
                "start generating validation stats!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            (
                validation_tracked_values,
                validation_stats,
            ) = self.retrieve_validation_prestats(finetune_iteration=0)
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        self.adaptor_config.current_iteration,
                    )

            cprint(
                "validation stats generated!!!", self.adaptor_config.tracking_level, 2
            )

        else:
            with open(os.path.join(self.base_dir, "platform.txt"), "w") as f:
                f.write(platform.node())

            cprint(
                "start loading validation stats!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            validation_stats_existed = os.path.exists(
                os.path.join(
                    self.base_dir,
                    str(max(0, self.adaptor_config.current_iteration - 1)),
                    "validation_stats.npz",
                )
            )
            (
                validation_tracked_values,
                validation_prestats,
            ) = self.retrieve_validation_prestats(
                finetune_iteration=max(0, self.adaptor_config.current_iteration - 1)
            )
            validation_stats = self.retrieve_validation_stats(
                finetune_iteration=self.adaptor_config.current_iteration,
                validation_tracked_values=validation_tracked_values,
                validation_prestats=validation_prestats,
            )
            if not validation_stats_existed:
                for key in validation_stats.keys():
                    if isinstance(validation_stats[key], float):
                        writer.add_scalar(
                            "validation_" + key,
                            validation_stats[key],
                            self.adaptor_config.current_iteration,
                        )

            cprint(
                "Create dataloader mixer and add counterfactual datasets!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            self.dataloader_mixer = DataloaderMixer(
                self.adaptor_config.training, self.train_dataloader
            )

            for i in range(1, self.adaptor_config.current_iteration + 1):
                dataset_dir = os.path.join(self.base_dir, str(i), "train_dataset")
                self.dataloader_mixer = self.add_dataset_to_dataloader_mixer(
                    dataloader_old=self.dataloader_mixer,
                    dataset_path=dataset_dir,
                    mixing_ratio=self.adaptor_config.mixing_ratio,
                    writer=writer,
                    finetune_iteration=i,
                )
                cprint(
                    "counterfactual dataset " + str(i) + " added!!!",
                    self.adaptor_config.tracking_level,
                    2,
                )

            self.datastack = DataStack(
                self.dataloader_mixer,
                self.output_size,
                transform=self.val_dataloader.dataset.transform,
            )

            if self.adaptor_config.current_iteration > 0:
                cprint(
                    "load already updated student model!!!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                try:
                    self.student = torch.load(
                        os.path.join(self.adaptor_config.base_dir, "model.cpl"),
                        map_location=self.device,
                    )
                except Exception:
                    self.student = torch.load(
                        os.path.join(self.adaptor_config.base_dir, "model.cpl"),
                        map_location=self.device,
                        weights_only=False,
                    )
                self.explainer.predictor = self.student

        visualization_path = os.path.join(self.base_dir, "visualization.png")
        if (
            self.output_size == 2
            and self.adaptor_config.tracking_level >= 6
            and not os.path.exists(visualization_path)
        ):
            cprint("visualize progress!!!", self.adaptor_config.tracking_level, 2)
            self.visualize_progress([visualization_path])
            cprint("Visualization done!!!", self.adaptor_config.tracking_level, 2)

        cprint("initialization done!!!", self.adaptor_config.tracking_level, 2)
        return validation_stats, validation_tracked_values, writer

    def get_batch(
        self,
        error_matrix: torch.Tensor = None,
        cm_idx_in: int = 0,
    ):
        """Assemble one batch of factuals with (source, target) class pairs.

        Pairs are walked through the flattened ``output_size x output_size``
        confusion-matrix index, skipping the diagonal, or sampled from
        ``error_matrix`` when ``use_confusion_matrix`` is set. Each factual is
        popped from the ``datastack`` of its source class; in ``"1sided"`` mode
        it is kept only when the student predicts it correctly. Hints and
        dataset indices are split off the label when enabled.

        Parameters
        ----------
        error_matrix : torch.Tensor, optional
            Flattened error distribution over class pairs from the validation
            stats; only used with ``use_confusion_matrix``.
        cm_idx_in : int
            Starting offset (in rows) of the confusion-matrix walk, so that
            alternate batches start from different pairs.

        Returns
        -------
        dict
            ``x_list`` (stacked tensor ``[B, ...]``), ``y_list``,
            ``y_target_list`` (tensor ``[B]``), ``y_source_list``,
            ``y_target_start_confidence_list`` (softmax of the tempered logits
            at the target class), ``hint_list`` (zeros unless a
            ``SegmentationMaskTeacher`` is used) and ``idx_list`` (zeros unless
            the explainer is ``PerfectFalseCounterfactualConfig``).
        """
        x_batch = []
        y_source_batch = []
        y_target_batch = []
        y_batch = []
        y_target_start_confidence_batch = []
        hint_batch = []
        idx_batch = []
        sample_idx = 0
        cm_idx = self.output_size * cm_idx_in
        torch.manual_seed(torch.seed())
        if self.adaptor_config.use_confusion_matrix:
            error_distribution = torch.distributions.Categorical(error_matrix)

        while not sample_idx >= self.adaptor_config.batch_size:
            if self.adaptor_config.use_confusion_matrix:
                cm_idx = error_distribution.sample()

            # TODO verify that this is actually balancing itself!
            y_source = int(cm_idx / self.output_size)
            y_target = int(cm_idx % self.output_size)
            while y_source == y_target:
                cm_idx = (cm_idx + 1) % (self.output_size**2)
                y_source = int(cm_idx / self.output_size)
                y_target = int(cm_idx % self.output_size)

            cm_idx = (cm_idx + 1) % (self.output_size**2)

            x, y = self.datastack.pop(int(y_source))

            if self.hints_enabled:
                y_res = y[1:]
                y = y[0]
                if isinstance(self.teacher, SegmentationMaskTeacher):
                    hint = y_res[0]

                if isinstance(
                    self.explainer.explainer_config, PerfectFalseCounterfactualConfig
                ):
                    idx = y_res[-1]

            elif isinstance(
                self.explainer.explainer_config, PerfectFalseCounterfactualConfig
            ):
                idx = y[-1]
                y = y[0]

            logits = (
                self.student(x.to(self.device).unsqueeze(0)).squeeze(0).detach().cpu()
            )
            y_target_start_confidence = torch.nn.Softmax()(
                logits / self.explainer.explainer_config.temperature
            )[y_target]
            prediction = self.logits_to_prediction(logits)
            if (
                not self.adaptor_config.counterfactual_type == "1sided"
                or prediction == y == y_source
            ):
                x_batch.append(x)
                y_source_batch.append(y_source)
                y_target_batch.append(torch.tensor(y_target))
                y_batch.append(y)
                y_target_start_confidence_batch.append(y_target_start_confidence)
                if isinstance(self.teacher, SegmentationMaskTeacher):
                    hint_batch.append(hint)

                else:
                    hint_batch.append(torch.zeros_like(x))

                if isinstance(
                    self.explainer.explainer_config, PerfectFalseCounterfactualConfig
                ):
                    idx_batch.append(idx)

                else:
                    idx_batch.append(0)

                sample_idx += 1

            else:
                pass

        x_batch = torch.stack(x_batch)
        y_target_batch = torch.stack(y_target_batch)
        return {
            "x_list": x_batch,
            "y_list": y_batch,
            "y_target_list": y_target_batch,
            "y_source_list": y_source_batch,
            "y_target_start_confidence_list": y_target_start_confidence_batch,
            "hint_list": hint_batch,
            "idx_list": idx_batch,
        }

    def generate_x_counterfactual_list(
        self,
        error_matrix,
        confidence_score_stats,
        finetune_iteration,
        tracked_keys,
    ):
        """Generate training counterfactuals until ``min_train_samples`` are found.

        Repeatedly builds batches with :meth:`get_batch`, runs
        ``explainer.explain_batch`` on them and accumulates the requested keys.
        Collages go to ``base_dir/<iteration>/collages`` (an existing directory
        is moved aside with an ``_old_<timestamp>`` suffix); the explainer's
        state directory is ``base_dir/<iteration - 1>``.

        Parameters
        ----------
        error_matrix : torch.Tensor
            Passed to :meth:`get_batch`.
        confidence_score_stats : object
            Currently unused.
        finetune_iteration : int
            Iteration whose directory receives the collages.
        tracked_keys : list of str
            Keys of the explainer output to collect.

        Returns
        -------
        dict
            ``{key: list}`` for every key in ``tracked_keys``, with at least
            ``adaptor_config.min_train_samples`` entries each (batches are not
            truncated, so usually a few more).
        """
        cprint("generate x counterfactual list!", self.adaptor_config.tracking_level, 2)
        self.datastack.reset()

        collage_base_path = os.path.join(
            self.base_dir, str(finetune_iteration), "collages"
        )
        if os.path.exists(collage_base_path):
            # move from self.base_dir to self.base_dir + "_old_" + {date}_{timestamp}
            shutil.move(
                collage_base_path,
                collage_base_path + "_old_" + datetime.now().strftime("%Y%m%d_%H%M%S"),
            )

        Path(collage_base_path).mkdir(parents=True, exist_ok=True)

        tracked_values = {key: [] for key in tracked_keys}

        continue_collecting = True
        acceptance_threshold = 0.51

        cprint(
            "Start generating x counterfactual list!!!",
            self.adaptor_config.tracking_level,
            2,
        )
        pbar = tqdm(
            total=int(
                self.adaptor_config.min_train_samples / self.adaptor_config.batch_size
                + 0.99
            )
            * (
                self.adaptor_config.explainer.gradient_steps
                if hasattr(self.adaptor_config.explainer, "gradient_steps")
                else 1
            )
        )
        pbar.stored_values = {}
        pbar.stored_values["n_total"] = 0
        remaining_sample_number = self.min_train_samples
        while continue_collecting:
            num_batches_per_iteration = int(
                1 + remaining_sample_number / self.adaptor_config.batch_size
            )
            if (
                len(list(tracked_values.values())[0])
                >= self.adaptor_config.min_train_samples
            ):
                break

            for i in range(num_batches_per_iteration):
                batch = self.get_batch(error_matrix, cm_idx_in=i % 2)
                values = self.explainer.explain_batch(
                    batch=batch,
                    base_path=collage_base_path,
                    start_idx=len(list(tracked_values.values())[0]),
                    pbar=pbar,
                    mode="Training",
                    explainer_path=os.path.join(
                        self.base_dir, str(finetune_iteration - 1)
                    ),
                )
                for key in tracked_keys:
                    if key in values.keys() and len(values[key]) > 0:
                        tracked_values[key].extend(values[key])

                pbar.stored_values["n_valid"] = (
                    str(len(list(tracked_values.values())[0]))
                    + "/"
                    + str(self.adaptor_config.min_train_samples)
                )
                pbar.stored_values["th"] = acceptance_threshold
                pbar.stored_values["n_total"] += self.adaptor_config.batch_size
                pbar.stored_values["fr"] = (
                    len(list(tracked_values.values())[0])
                    / pbar.stored_values["n_total"]
                )
                remaining_sample_number = self.adaptor_config.min_train_samples - len(
                    list(tracked_values.values())[0]
                )

                if remaining_sample_number <= 0:
                    break

            else:
                continue_collecting = False

        cprint(
            "x counterfactual list generated!!!", self.adaptor_config.tracking_level, 2
        )
        pbar.close()
        return tracked_values

    def retrieve_counterfactual_list(self, validation_stats, finetune_iteration):
        """Training counterfactuals of an iteration, generated or loaded from cache.

        Generates them with :meth:`generate_x_counterfactual_list` and saves
        ``base_dir/<iteration>/tracked_values.npz`` (``tracking_level >= 3``),
        or loads that file (expanding old per-factual layouts with
        ``flatten_explanations``). Then ``collage_path_list`` is rebuilt from
        the ``.png`` files in the collages directory and, if the explainer
        clusters, ``cluster_explanations`` adds the ``clusters<k>`` entries.

        Parameters
        ----------
        validation_stats : dict
            Must contain ``error_matrix`` and ``confidence_score_stats``.
        finetune_iteration : int
            Iteration index (1-based).

        Returns
        -------
        dict
            The tracked values, one list per key.
        """

        tracked_values_path = os.path.join(
            self.base_dir, str(finetune_iteration), "tracked_values.npz"
        )
        if self.overwrite or not os.path.exists(tracked_values_path):
            cprint(
                "Start generating tracked values!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            tracked_values = self.generate_x_counterfactual_list(
                error_matrix=validation_stats["error_matrix"],
                confidence_score_stats=validation_stats["confidence_score_stats"],
                finetune_iteration=finetune_iteration,
                tracked_keys=self.tracked_keys,
            )

            if self.adaptor_config.tracking_level >= 3:
                with open(
                    tracked_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in tracked_values.keys():
                        if len(tracked_values[key]) == 0:
                            continue

                        if isinstance(tracked_values[key][0], torch.Tensor):
                            tracked_values_file[key] = torch.stack(
                                tracked_values[key], dim=0
                            ).numpy()

                        elif isinstance(tracked_values[key][0], int) or isinstance(
                            tracked_values[key][0], float
                        ):
                            tracked_values_file[key] = np.array(tracked_values[key])

                    np.savez(f, **tracked_values_file)

        else:
            with open(
                tracked_values_path,
                "rb",
            ) as f:
                cprint("Load tracked values!!!", self.adaptor_config.tracking_level, 2)
                tracked_values = {}
                tracked_values_file = np.load(f, allow_pickle=True)
                for key in tracked_values_file.keys():
                    tracked_values[key] = list(torch.tensor(tracked_values_file[key]))
                # Same expansion as on the validation cache: a tracked_values.npz
                # written before the layout fix has one row per factual for the
                # per-factual entries and one per counterfactual for the rest, and
                # Model2ModelTeacher.get_feedback indexes both by counterfactual.
                # The expansion is per batch, so it needs the batch size that
                # produced the file, the same one cluster_explanations is given.
                tracked_values = flatten_explanations(
                    tracked_values, batch_size=self.adaptor_config.batch_size
                )

        cprint("Create collage path list!!!", self.adaptor_config.tracking_level, 2)
        collage_path_list = os.listdir(
            os.path.join(self.base_dir, str(finetune_iteration), "collages")
        )
        collage_path_list.sort()
        collage_path_list = list(filter(lambda x: x[-4:] == ".png", collage_path_list))
        tracked_values["collage_path_list"] = list(
            map(
                lambda x: os.path.join(
                    self.base_dir, str(finetune_iteration), "collages", x
                ),
                collage_path_list,
            )
        )
        if self.adaptor_config.explainer.use_clustering and not hasattr(
            tracked_values, "cluster0"
        ):
            tracked_values = self.explainer.cluster_explanations(
                tracked_values,
                self.adaptor_config.batch_size,
                self.adaptor_config.explainer.num_attempts
                * self.adaptor_config.explainer.parallel_attempts,
            )

        return tracked_values

    def retrieve_feedback(self, tracked_values, finetune_iteration, mode):
        """Teacher verdicts for a set of counterfactuals, cached per iteration.

        As a side effect the distilled predictor used by the explainer metrics
        is prepared: with ``calculate_explainer_stats`` the student is
        distilled (``distill_predictor`` with a leaky-softplus activation) into
        ``base_dir/<current_iteration>/distilled_predictor`` or reloaded from
        there; otherwise the student itself is used. The verdicts are written
        to ``base_dir/<iteration>/<mode>_feedback.txt`` and read back from it
        on later runs.

        Parameters
        ----------
        tracked_values : dict
            Tracked counterfactual values, unpacked as keyword arguments of
            ``teacher.get_feedback``.
        finetune_iteration : int
            Iteration whose directory holds the feedback file.
        mode : str
            ``"train"`` or ``"validation"``; also names the teacher's working
            directory ``<mode>_teacher``.

        Returns
        -------
        list of str
            One verdict per counterfactual (``"true"``, ``"false"``, ``"ood"``
            or a teacher-specific marker string).
        """
        # this is only for scientific experiments and could also be sourced out into another file!
        # distill into equivalent model
        predictor_distillation = load_yaml_config(
            "<PEAL_BASE>/configs/sce_experiments/predictors/simple_distillation.yaml",
            PredictorConfig,
        )
        distillation_path = os.path.join(
            self.base_dir,
            str(self.adaptor_config.current_iteration),
            "distilled_predictor",
        )
        distilled_predictor_final = os.path.join(
            distillation_path, "distilled_predictor", "model.cpl"
        )
        if not self.adaptor_config.calculate_explainer_stats:
            self.distilled_predictor = self.student

        elif not os.path.exists(distilled_predictor_final):
            self.distilled_predictor = distill_predictor(
                predictor_distillation,
                distillation_path,
                self.student,
                [self.train_dataloader.dataset, self.val_dataloader.dataset],
                replace_with_activation="leakysoftplus",
                tracking_level=self.adaptor_config.tracking_level,
            )

        else:
            try:
                self.distilled_predictor = torch.load(
                    distilled_predictor_final, map_location=self.device
                )
            except Exception:
                self.distilled_predictor = torch.load(
                    distilled_predictor_final,
                    map_location=self.device,
                    weights_only=False,
                )

        if self.overwrite or not os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), mode + "_feedback.txt")
        ):
            cprint("retrieve feedback!", self.adaptor_config.tracking_level, 2)

            feedback = self.teacher.get_feedback(
                base_dir=os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_teacher"
                ),
                student=self.student,  # TODO including this introduces some inconsistencies that should be tracked!
                num_clusters=self.adaptor_config.explainer.num_attempts
                * self.adaptor_config.explainer.parallel_attempts,
                mode=mode,
                **tracked_values,
            )

            os.makedirs(
                os.path.join(self.base_dir, str(finetune_iteration)), exist_ok=True
            )
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_feedback.txt"
                ),
                "w",
            ) as f:
                f.write("\n".join(feedback))

        else:
            cprint("load feedback!", self.adaptor_config.tracking_level, 2)
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), mode + "_feedback.txt"
                ),
                "r",
            ) as f:
                feedback = f.read().split("\n")
        return feedback

    def ensure_group_accuracies(self):
        """The latest per-group test accuracies, computing them once if needed.

        Returns
        -------
        list or None
            ``adaptor_config.group_accuracies[-1]`` if present; otherwise the
            groups' accuracies freshly measured on the test loader (also
            appended to the config and saved to ``config.yaml``), or ``None``
            when ``calculate_group_accuracies`` is off.
        """
        if len(self.adaptor_config.group_accuracies) > 0:
            return self.adaptor_config.group_accuracies[-1]

        if not self.adaptor_config.calculate_group_accuracies:
            return None

        (
            _,
            group_accuracies,
            group_distribution,
            groups,
            worst_group_accuracy,
        ) = calculate_test_accuracy(
            self.student,
            self.test_dataloader,
            self.device,
            True,
            self.adaptor_config.max_test_batches,
            tracking_level=self.adaptor_config.tracking_level,
        )
        self.adaptor_config.group_accuracies.append(group_accuracies)
        self.adaptor_config.avg_group_accuracies.append(
            float(np.mean(group_accuracies))
        )
        save_yaml_config(
            self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
        )
        cprint(
            "group_accuracies: " + str(group_accuracies),
            self.adaptor_config.tracking_level,
            2,
        )
        cprint(
            "group_distribution: " + str(group_distribution),
            self.adaptor_config.tracking_level,
            2,
        )
        cprint(
            "group_sizes: " + str(groups),
            self.adaptor_config.tracking_level,
            2,
        )
        cprint(
            "worst_group_accuracy: " + str(worst_group_accuracy),
            self.adaptor_config.tracking_level,
            2,
        )
        return group_accuracies

    def calculate_unbiasedness(self, feedback_accuracy_distilled):
        """Unbiasedness score of the explainer from the distilled feedback accuracy.

        ``r_dominant = 1 - mean(two lowest group accuracies)`` estimates how
        often the student uses the shortcut; ``r_actual = 1 -
        feedback_accuracy_distilled`` how often the explainer reveals it. The
        score is ``exp(-|log(r_actual / r_dominant)|)``, i.e. 1 when both
        rates agree.

        Parameters
        ----------
        feedback_accuracy_distilled : float
            Fraction of "true" verdicts among counterfactuals that flip both
            the student and its distilled copy.

        Returns
        -------
        float or None
            The score, ``0.0`` when a rate is zero, ``None`` when group
            accuracies are unavailable or fewer than two groups exist.
        """
        group_accuracies = self.ensure_group_accuracies()
        if group_accuracies is None:
            return None

        group_accuracies = torch.as_tensor(group_accuracies).flatten()
        if len(group_accuracies) < 2:
            cprint(
                "unbiasedness requires accuracies for at least two groups",
                self.adaptor_config.tracking_level,
                1,
            )
            return None

        group_accuracies = group_accuracies.sort().values
        r_dominant = 1 - torch.mean(group_accuracies[:2])
        cprint(
            "r_dominant: " + str(r_dominant),
            self.adaptor_config.tracking_level,
            2,
        )
        r_actual = 1 - float(feedback_accuracy_distilled)
        cprint(
            "r_actual: " + str(r_actual),
            self.adaptor_config.tracking_level,
            2,
        )
        ratio = float(r_actual / r_dominant) if r_dominant > 0 else 0.0
        unbiasedness = float(math.exp(-abs(math.log(ratio)))) if ratio > 0 else 0.0
        cprint(
            "unbiasedness: " + str(unbiasedness),
            self.adaptor_config.tracking_level,
            2,
        )
        return unbiasedness

    def calculate_feedback_stats(self, tracked_values, feedback, finetune_iteration):
        """Explainer metrics from the teacher verdicts of one validation pass.

        Always computes ``flip_rate`` (target confidence >= 0.51, relative to
        ``max_validation_samples`` at iteration 0), ``ood_rate``,
        ``feedback_accuracy`` (fraction of "true" among true+false verdicts on
        correctly classified factuals) and ``counterfactuals_per_second``.
        With several attempts per factual only the highest-confidence attempt
        counts; the attempt layout (interleaved vs sequential) is inferred
        from ``y_list``.

        With ``calculate_explainer_stats`` at iteration 0 the distilled
        predictor is additionally run on every counterfactual and cluster
        member, cluster collages are written to
        ``base_dir/<iteration>/latent_cluster_collages<k>/`` and the
        following are added: ``flip_rate_distilled`` (NAFR: flips shared by
        student and distilled copy), ``non_adversarial_rate``,
        ``feedback_accuracy_distilled``, ``counterfactual_quality`` (dataset
        ``track_generator_performance`` on the non-adversarial flips, with
        quality collages in ``latent_quality_collages/``), ``unbiasedness``
        (when group accuracies exist) and the latent sparsity/diversity stats
        of ``explainer.calculate_latent_difference_stats``.

        Parameters
        ----------
        tracked_values : dict
            Tracked validation counterfactuals; extended in place with
            ``y_target_end_confidence_distilled_list`` and
            ``cluster_confidence_distilled<k>``.
        feedback : list of str
            One verdict per counterfactual.
        finetune_iteration : int
            Iteration the values belong to.

        Returns
        -------
        dict
            Metric name to float.
        """
        n_attempts = self.adaptor_config.explainer.num_attempts
        has_clusters = "clusters0" in tracked_values

        # Determine layout and select highest-confidence attempt per factual
        if not has_clusters and n_attempts > 1 and len(feedback) % n_attempts == 0:
            num_factuals = len(feedback) // n_attempts
            is_sequential = False
            y_val = tracked_values.get("y_list", None)
            if y_val is not None and len(y_val) == len(feedback):
                if num_factuals > 1 and y_val[0] == y_val[num_factuals]:
                    is_sequential = True

            best_feedback = []
            best_y_list = []
            best_y_source_list = []
            best_y_target_end_confidence = []

            for i in range(num_factuals):
                best_conf = -1.0
                best_idx = -1
                for c in range(n_attempts):
                    idx = c * num_factuals + i if is_sequential else i * n_attempts + c
                    if idx < len(tracked_values["y_target_end_confidence_list"]):
                        conf = float(
                            tracked_values["y_target_end_confidence_list"][idx]
                        )
                        if conf > best_conf:
                            best_conf = conf
                            best_idx = idx
                if best_idx != -1:
                    best_feedback.append(feedback[best_idx])
                    best_y_list.append(tracked_values["y_list"][best_idx])
                    best_y_source_list.append(tracked_values["y_source_list"][best_idx])
                    best_y_target_end_confidence.append(
                        tracked_values["y_target_end_confidence_list"][best_idx]
                    )

            feedback_for_metrics = best_feedback
            y_list_for_metrics = best_y_list
            y_source_list_for_metrics = best_y_source_list
            y_target_end_confidence_for_metrics = best_y_target_end_confidence
            num_samples = num_factuals
        else:
            feedback_for_metrics = feedback
            y_list_for_metrics = tracked_values["y_list"]
            y_source_list_for_metrics = tracked_values["y_source_list"]
            y_target_end_confidence_for_metrics = tracked_values[
                "y_target_end_confidence_list"
            ]
            num_samples = len(feedback)

        if finetune_iteration == 0:
            flip_rate_reference = max(
                num_samples, self.adaptor_config.max_validation_samples
            )
        else:
            flip_rate_reference = num_samples

        flipped_samples = list(
            filter(
                lambda x: x >= 0.51,
                y_target_end_confidence_for_metrics[:num_samples],
            )
        )

        flip_rate = (
            len(flipped_samples) / flip_rate_reference
            if flip_rate_reference > 0
            else 0.0
        )
        try:
            ood_rate = (
                len(list(filter(lambda sample: sample == "ood", feedback_for_metrics)))
                / num_samples
            )
        except Exception:
            raise

        num_true_1sided = len(
            list(
                filter(
                    lambda x: x[1] == "true"
                    and y_list_for_metrics[x[0]] == y_source_list_for_metrics[x[0]],
                    enumerate(feedback_for_metrics),
                )
            )
        )
        num_false_1sided = len(
            list(
                filter(
                    lambda x: x[1] == "false"
                    and y_list_for_metrics[x[0]] == y_source_list_for_metrics[x[0]],
                    enumerate(feedback_for_metrics),
                )
            )
        )
        if num_true_1sided + num_false_1sided > 0:
            fa_1sided = num_true_1sided / (num_true_1sided + num_false_1sided)
        else:
            fa_1sided = -1

        feedback_stats = {
            "flip_rate": flip_rate,
            "ood_rate": ood_rate,
            "feedback_accuracy": fa_1sided,
            "counterfactuals_per_second": self.explainer.counterfactuals_per_second,
        }
        cprint(
            "counterfactuals_per_second: "
            + str(self.explainer.counterfactuals_per_second),
            self.adaptor_config.tracking_level,
            2,
        )
        cprint("flip_rate: " + str(flip_rate), self.adaptor_config.tracking_level, 2)

        if self.adaptor_config.calculate_explainer_stats and finetune_iteration == 0:
            # add y_target_end_confidence_distilled_list
            tracked_values["y_target_end_confidence_distilled_list"] = []
            for idx in range(len(tracked_values["x_counterfactual_list"])):
                x = tracked_values["x_counterfactual_list"][idx]
                y = tracked_values["y_target_list"][idx]
                logits = (
                    self.distilled_predictor(x.to(self.device).unsqueeze(0))
                    .squeeze(0)
                    .detach()
                    .cpu()
                )
                if logits.numel() == 1:
                    probs = torch.sigmoid(
                        logits / self.explainer.explainer_config.temperature
                    )
                    y_target_end_confidence = (
                        float(probs[0]) if y == 1 else float(1.0 - probs[0])
                    )
                else:
                    probs = torch.nn.functional.softmax(
                        logits / self.explainer.explainer_config.temperature, dim=-1
                    )
                    y_target_end_confidence = float(probs[y])
                tracked_values["y_target_end_confidence_distilled_list"].append(
                    y_target_end_confidence
                )

            # Use configured cluster indices for explainer stats
            active_cluster_indices = list(self.adaptor_config.explainer_stats_clusters)

            # Compute distilled confidences per cluster for sparsity/diversity gating
            has_clusters_for_distilled = "clusters0" in tracked_values
            if has_clusters_for_distilled:
                n_factuals_clusters = len(tracked_values["clusters0"])
                for c in range(n_attempts):
                    cluster_key = "clusters" + str(c)
                    if cluster_key in tracked_values:
                        distilled_confs_c = []
                        for i in range(len(tracked_values[cluster_key])):
                            cf_sample = tracked_values[cluster_key][i]
                            y_t = tracked_values["y_target_list"][i]
                            logits = (
                                self.distilled_predictor(
                                    cf_sample.to(self.device).unsqueeze(0)
                                )
                                .squeeze(0)
                                .detach()
                                .cpu()
                            )
                            if logits.numel() == 1:
                                probs = torch.sigmoid(
                                    logits / self.explainer.explainer_config.temperature
                                )
                                d_conf = (
                                    float(probs[0])
                                    if y_t == 1
                                    else float(1.0 - probs[0])
                                )
                            else:
                                probs = torch.nn.functional.softmax(
                                    logits
                                    / self.explainer.explainer_config.temperature,
                                    dim=0,
                                )
                                d_conf = float(probs[y_t])
                            distilled_confs_c.append(d_conf)
                        tracked_values["cluster_confidence_distilled" + str(c)] = (
                            distilled_confs_c
                        )

            # Generate cluster collages for all clusters (all samples, regardless of flip)
            iter_dir = os.path.join(self.base_dir, str(finetune_iteration))
            for c in range(n_attempts):
                cluster_key = "clusters" + str(c)
                if cluster_key in tracked_values:
                    self._generate_cluster_collages(
                        tracked_values=tracked_values,
                        cluster_idx=c,
                        base_dir=iter_dir,
                    )

            if has_clusters and n_attempts > 1:
                # For each factual, pick the cluster with the highest original confidence
                best_original_confidence = []
                best_distilled_confidence = []
                best_counterfactual = []
                best_feedback = []
                n_factuals = len(tracked_values["clusters0"])
                for i in range(n_factuals):
                    best_conf = -1.0
                    best_cluster_idx = 0
                    for c in active_cluster_indices:
                        conf_key = "cluster_confidence" + str(c)
                        if conf_key in tracked_values and i < len(
                            tracked_values[conf_key]
                        ):
                            conf = float(tracked_values[conf_key][i])
                            if conf > best_conf:
                                best_conf = conf
                                best_cluster_idx = c
                    best_original_confidence.append(best_conf)

                    # Store the feedback for this specific attempt
                    idx_in_interleaved = i * n_attempts + best_cluster_idx
                    if idx_in_interleaved < len(feedback):
                        best_feedback.append(feedback[idx_in_interleaved])
                    else:
                        best_feedback.append("ood")

                    # Get the distilled confidence for the same cluster's counterfactual
                    cf_key = "clusters" + str(best_cluster_idx)
                    cf_sample = tracked_values[cf_key][i]
                    y = tracked_values["y_target_list"][i]
                    logits = (
                        self.distilled_predictor(cf_sample.to(self.device).unsqueeze(0))
                        .squeeze(0)
                        .detach()
                        .cpu()
                    )
                    if logits.numel() == 1:
                        probs = torch.sigmoid(
                            logits / self.explainer.explainer_config.temperature
                        )
                        dist_conf = float(probs[0]) if y == 1 else float(1.0 - probs[0])
                    else:
                        probs = torch.nn.functional.softmax(
                            logits / self.explainer.explainer_config.temperature, dim=0
                        )
                        dist_conf = float(probs[y])
                    best_distilled_confidence.append(dist_conf)
                    best_counterfactual.append(cf_sample)

                original_flipped = [c > 0.5 for c in best_original_confidence]
                distilled_flipped = [c > 0.5 for c in best_distilled_confidence]

                distilled_conf_list = best_distilled_confidence
                feedback_list = best_feedback
                n_factuals_eval = n_factuals
            else:
                if (
                    not has_clusters
                    and n_attempts > 1
                    and len(feedback) % n_attempts == 0
                ):
                    best_original_confidence = []
                    best_distilled_confidence = []
                    best_counterfactual = []
                    best_feedback = []
                    num_factuals = len(feedback) // n_attempts
                    is_sequential = False
                    y_val = tracked_values.get("y_list", None)
                    if y_val is not None and len(y_val) == len(feedback):
                        if num_factuals > 1 and y_val[0] == y_val[num_factuals]:
                            is_sequential = True

                    for i in range(num_factuals):
                        best_conf = -1.0
                        best_idx = -1
                        for c in active_cluster_indices:
                            idx = (
                                c * num_factuals + i
                                if is_sequential
                                else i * n_attempts + c
                            )
                            if idx < len(
                                tracked_values["y_target_end_confidence_list"]
                            ):
                                conf = float(
                                    tracked_values["y_target_end_confidence_list"][idx]
                                )
                                if conf > best_conf:
                                    best_conf = conf
                                    best_idx = idx
                        if best_idx != -1:
                            best_original_confidence.append(best_conf)
                            best_feedback.append(feedback[best_idx])
                            x = tracked_values["x_counterfactual_list"][best_idx]
                            y = tracked_values["y_target_list"][best_idx]
                            logits = (
                                self.distilled_predictor(x.to(self.device).unsqueeze(0))
                                .squeeze(0)
                                .detach()
                                .cpu()
                            )
                            if logits.numel() == 1:
                                probs = torch.sigmoid(
                                    logits / self.explainer.explainer_config.temperature
                                )
                                dist_conf = (
                                    float(probs[0]) if y == 1 else float(1.0 - probs[0])
                                )
                            else:
                                probs = torch.nn.functional.softmax(
                                    logits
                                    / self.explainer.explainer_config.temperature,
                                    dim=0,
                                )
                                dist_conf = float(probs[y])
                            best_distilled_confidence.append(dist_conf)
                            best_counterfactual.append(x)

                    original_flipped = [c > 0.5 for c in best_original_confidence]
                    distilled_flipped = [c > 0.5 for c in best_distilled_confidence]
                    n_factuals = num_factuals

                    distilled_conf_list = best_distilled_confidence
                    feedback_list = best_feedback
                    n_factuals_eval = num_factuals
                else:
                    # Fallback: single attempt, use the existing lists directly
                    original_flipped = [
                        tracked_values["y_target_end_confidence_list"][i] > 0.5
                        for i in range(num_samples)
                    ]
                    distilled_flipped = [
                        tracked_values["y_target_end_confidence_distilled_list"][i]
                        > 0.5
                        for i in range(num_samples)
                    ]
                    best_counterfactual = [
                        tracked_values["x_counterfactual_list"][i]
                        for i in range(num_samples)
                    ]
                    # The two multi-attempt branches above build these alongside
                    # best_counterfactual; this fallback did not, but the
                    # non-adversarial-quality block below indexes them unconditionally,
                    # so any single-attempt run (i.e. any single-component
                    # component_indices) died here with
                    #   UnboundLocalError: local variable 'best_original_confidence'
                    #   referenced before assignment
                    # after having already generated all 800 counterfactuals and their
                    # feedback. These are the same lists original_flipped /
                    # distilled_flipped are thresholded from, so they stay consistent.
                    best_original_confidence = [
                        tracked_values["y_target_end_confidence_list"][i]
                        for i in range(num_samples)
                    ]
                    best_distilled_confidence = [
                        tracked_values["y_target_end_confidence_distilled_list"][i]
                        for i in range(num_samples)
                    ]
                    n_factuals = num_samples

                    distilled_conf_list = tracked_values[
                        "y_target_end_confidence_distilled_list"
                    ]
                    feedback_list = feedback
                    n_factuals_eval = num_samples

            n_original_flipped = sum(1 for f in original_flipped if f)
            n_both_flipped = sum(
                1
                for i in range(n_factuals)
                if original_flipped[i] and distilled_flipped[i]
            )

            # NAFR: % of dataset with meaningful non-adversarial counterfactuals
            nafr_reference = (
                max(n_factuals, self.adaptor_config.max_validation_samples)
                if finetune_iteration == 0
                else n_factuals
            )
            nafr = n_both_flipped / nafr_reference if nafr_reference > 0 else 0.0
            feedback_stats["flip_rate_distilled"] = float(nafr)
            cprint(
                "flip_rate_distilled (NAFR): " + str(nafr),
                self.adaptor_config.tracking_level,
                2,
            )

            # NA: of originally-flipped samples, how many also flip in distilled
            non_adversarial_rate = (
                n_both_flipped / n_original_flipped if n_original_flipped > 0 else 0.0
            )
            feedback_stats["non_adversarial_rate"] = float(non_adversarial_rate)
            cprint(
                "non_adversarial_rate: " + str(non_adversarial_rate),
                self.adaptor_config.tracking_level,
                2,
            )

            # Now calculate distilled feedback accuracy using the aligned lists
            # Only consider samples where both original and distilled predictors were flipped
            num_true_1sided_distilled = 0
            num_false_1sided_distilled = 0
            for idx in range(n_factuals_eval):
                orig_flipped = (
                    original_flipped[idx] if idx < len(original_flipped) else False
                )
                dist_flipped = float(distilled_conf_list[idx]) > 0.5
                if orig_flipped and dist_flipped:
                    if feedback_list[idx] == "true":
                        num_true_1sided_distilled += 1
                    elif feedback_list[idx] == "false":
                        num_false_1sided_distilled += 1

            if num_true_1sided_distilled + num_false_1sided_distilled > 0:
                fa_1sided_distilled = num_true_1sided_distilled / (
                    num_true_1sided_distilled + num_false_1sided_distilled
                )
            else:
                fa_1sided_distilled = 0.0

            feedback_stats["feedback_accuracy_distilled"] = float(fa_1sided_distilled)
            cprint(
                "feedback_accuracy_distilled: " + str(fa_1sided_distilled),
                self.adaptor_config.tracking_level,
                2,
            )
            # Collect non-adversarial flipped counterfactuals for quality computation
            flipped_cfs = []
            flipped_facs = []
            flipped_orig_confs = []
            flipped_dist_confs = []
            for i in range(n_factuals):
                if original_flipped[i] and distilled_flipped[i]:
                    flipped_cfs.append(best_counterfactual[i])
                    flipped_facs.append(tracked_values["x_list"][i])
                    flipped_orig_confs.append(best_original_confidence[i])
                    flipped_dist_confs.append(best_distilled_confidence[i])

            if len(flipped_cfs) > 0:
                x_counterfactuals_non_adversarial_flips = torch.stack(flipped_cfs)

                # Generate quality collages
                try:
                    self._generate_quality_collages(
                        flipped_facs,
                        flipped_cfs,
                        flipped_orig_confs,
                        flipped_dist_confs,
                        base_dir=os.path.join(self.base_dir, str(finetune_iteration)),
                    )
                except Exception as e:
                    _log.info("%s", f"Failed to generate quality collages: {e}")

                validation_samples = []
                for idx in range(
                    min(
                        self.adaptor_config.max_validation_samples,
                        len(self.val_dataloader.dataset),
                    )
                ):
                    x, y = self.val_dataloader.dataset[idx]
                    validation_samples.append(x.unsqueeze(0))

                validation_samples = torch.cat(validation_samples, dim=0)
                self.train_dataloader.dataset.reference_fid = (
                    self.train_dataloader.dataset.track_generator_performance(
                        validation_samples
                    )["dino_fid"]
                )
                counterfactual_quality = min(
                    1.0,
                    self.train_dataloader.dataset.track_generator_performance(
                        x_counterfactuals_non_adversarial_flips
                    )["quality_score"],
                )
                feedback_stats["counterfactual_quality"] = float(counterfactual_quality)
                cprint(
                    "counterfactual_quality: " + str(counterfactual_quality),
                    self.adaptor_config.tracking_level,
                    2,
                )

            else:
                feedback_stats["counterfactual_quality"] = 0.0
                cprint(
                    "counterfactual_quality: 0.0",
                    self.adaptor_config.tracking_level,
                    2,
                )

            if len(self.adaptor_config.group_accuracies) > 0:
                group_accuracies = torch.tensor(
                    self.adaptor_config.group_accuracies[-1]
                ).flatten()
                group_accuracies_idxs = group_accuracies.argsort()
                r_dominant = (
                    1
                    - (
                        group_accuracies[group_accuracies_idxs[0]]
                        + group_accuracies[group_accuracies_idxs[1]]
                    )
                    / 2
                )
                cprint(
                    "r_dominant: " + str(r_dominant),
                    self.adaptor_config.tracking_level,
                    2,
                )
                r_actual = 1 - fa_1sided_distilled
                cprint(
                    "r_actual: " + str(r_actual),
                    self.adaptor_config.tracking_level,
                    2,
                )
                ratio = float(r_actual / r_dominant) if r_dominant > 0 else 0.0
                if (num_true_1sided_distilled + num_false_1sided_distilled) == 0:
                    unbiasedness = 0.0
                elif ratio > 0:
                    unbiasedness = float(math.exp(-abs(math.log(ratio))))
                else:
                    unbiasedness = 0.0
                feedback_stats["unbiasedness"] = float(unbiasedness)

            tracked_stats = self.explainer.calculate_latent_difference_stats(
                tracked_values,
                explainer_stats_clusters=list(
                    self.adaptor_config.explainer_stats_clusters
                ),
                visualize_latent_sparsity=self.adaptor_config.visualize_latent_sparsity,
                visualize_latent_diversity=self.adaptor_config.visualize_latent_diversity,
                base_dir=os.path.join(self.base_dir, str(finetune_iteration)),
            )
            for key in tracked_stats.keys():
                feedback_stats[key] = tracked_stats[key]

        return feedback_stats

    def _generate_quality_collages(
        self, factuals, counterfactuals, orig_confs, dist_confs, base_dir
    ):
        """
        Generate collages for the counterfactuals used in calculating the quality score.
        """
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        import os
        from pathlib import Path

        collage_dir = os.path.join(base_dir, "latent_quality_collages")
        Path(collage_dir).mkdir(parents=True, exist_ok=True)

        dataset = self.val_dataloader.dataset
        is_image = len(factuals[0].shape) >= 3 and factuals[0].shape[0] in [1, 3]

        for vi, (fac, cf, o_conf, d_conf) in enumerate(
            zip(factuals, counterfactuals, orig_confs, dist_confs)
        ):
            if is_image and hasattr(dataset, "project_to_pytorch_default"):
                factual_vis = dataset.project_to_pytorch_default(fac)
                cf_vis = dataset.project_to_pytorch_default(cf)
            else:
                factual_vis = fac
                cf_vis = cf

            conf_text = f"Original Conf: {o_conf:.4f} | Distilled Conf: {d_conf:.4f}"

            try:
                if is_image:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(
                        factual_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    )
                    axes[0].axis("off")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].imshow(cf_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1))
                    axes[1].axis("off")
                    axes[1].set_title("Counterfactual", fontweight="bold")

                    fig.suptitle(conf_text, fontsize=12, fontweight="bold")
                else:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                    factual_np = factual_vis.cpu().numpy().flatten()
                    cf_np = cf_vis.cpu().numpy().flatten()
                    x_range = range(len(factual_np))

                    axes[0].bar(x_range, factual_np, color="#3498db")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].bar(x_range, cf_np, color="#2ecc71")
                    axes[1].set_title("Counterfactual", fontweight="bold")

                    fig.suptitle(conf_text, fontsize=12, fontweight="bold")

                plt.tight_layout()
                collage_path = os.path.join(collage_dir, f"{vi:07d}_quality.png")
                plt.savefig(collage_path, dpi=150)
            finally:
                plt.close(fig)

        plt.close("all")

    def _generate_cluster_collages(self, tracked_values, cluster_idx, base_dir):
        """
        Generate collages for all samples in a single cluster.
        Each collage contains: factual, counterfactual, original confidence, and distilled confidence.
        No flip filtering — all samples are logged.
        Saves into latent_cluster_collages{cluster_idx}/.
        """
        import matplotlib

        matplotlib.use("Agg")
        from matplotlib import pyplot as plt
        from pathlib import Path

        cluster_key = "clusters" + str(cluster_idx)
        if cluster_key not in tracked_values:
            return

        collage_dir = os.path.join(
            base_dir, "latent_cluster_collages" + str(cluster_idx)
        )
        Path(collage_dir).mkdir(parents=True, exist_ok=True)

        dataset = self.val_dataloader.dataset
        n_samples = len(tracked_values[cluster_key])
        conf_key = "cluster_confidence" + str(cluster_idx)
        dist_conf_key = "cluster_confidence_distilled" + str(cluster_idx)

        for vi in range(n_samples):
            factual = tracked_values["x_list"][vi]
            counterfactual = tracked_values[cluster_key][vi]

            # Get original predictor confidence
            if conf_key in tracked_values and vi < len(tracked_values[conf_key]):
                orig_conf = float(tracked_values[conf_key][vi])
            elif vi < len(tracked_values.get("y_target_end_confidence_list", [])):
                orig_conf = float(tracked_values["y_target_end_confidence_list"][vi])
            else:
                orig_conf = None

            # Get distilled predictor confidence
            if dist_conf_key in tracked_values and vi < len(
                tracked_values[dist_conf_key]
            ):
                dist_conf = float(tracked_values[dist_conf_key][vi])
            elif (
                "y_target_end_confidence_distilled_list" in tracked_values
                and vi < len(tracked_values["y_target_end_confidence_distilled_list"])
            ):
                dist_conf = float(
                    tracked_values["y_target_end_confidence_distilled_list"][vi]
                )
            else:
                dist_conf = None

            # Build title text
            conf_parts = [f"Cluster {cluster_idx}"]
            if orig_conf is not None:
                conf_parts.append(f"Original Conf: {orig_conf:.4f}")
            if dist_conf is not None:
                conf_parts.append(f"Distilled Conf: {dist_conf:.4f}")
            conf_text = " | ".join(conf_parts)

            is_image = len(factual.shape) >= 3 and factual.shape[0] in [1, 3]
            if is_image and hasattr(dataset, "project_to_pytorch_default"):
                factual_vis = dataset.project_to_pytorch_default(factual)
                cf_vis = dataset.project_to_pytorch_default(counterfactual)
            else:
                factual_vis = factual
                cf_vis = counterfactual

            try:
                if is_image:
                    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
                    axes[0].imshow(
                        factual_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1)
                    )
                    axes[0].axis("off")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].imshow(cf_vis.permute(1, 2, 0).cpu().numpy().clip(0, 1))
                    axes[1].axis("off")
                    axes[1].set_title("Counterfactual", fontweight="bold")

                    fig.suptitle(conf_text, fontsize=12, fontweight="bold")
                else:
                    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
                    factual_np = factual_vis.cpu().numpy().flatten()
                    cf_np = cf_vis.cpu().numpy().flatten()
                    x_range = range(len(factual_np))

                    axes[0].bar(x_range, factual_np, color="#3498db")
                    axes[0].set_title("Factual", fontweight="bold")

                    axes[1].bar(x_range, cf_np, color="#2ecc71")
                    axes[1].set_title("Counterfactual", fontweight="bold")

                    fig.suptitle(conf_text, fontsize=12, fontweight="bold")

                plt.tight_layout()
                collage_path = os.path.join(
                    collage_dir, f"{vi:07d}_cluster{cluster_idx}.png"
                )
                plt.savefig(collage_path, dpi=150)
            finally:
                plt.close(fig)

        plt.close("all")
        cprint(
            f"Saved {n_samples} cluster {cluster_idx} collages to {collage_dir}",
            self.adaptor_config.tracking_level,
            2,
        )

    def create_dataset(
        self,
        x_counterfactual_list,
        feedback,
        y_source_list,
        y_target_list,
        finetune_iteration,
        hint_list=None,
        mode="",
        **kwargs,
    ):
        """Serialise judged counterfactuals as a dataset for finetuning.

        "false" counterfactuals are stored with their source class as label
        (the change did not alter the class, so the student must not flip);
        "true" ones with the target class when ``use_true_counterfactuals``
        is set. Sample names encode verdict, source, target and index. The
        dataset is written with ``train_dataloader.dataset.serialize_dataset``
        to ``base_dir/<iteration>/<mode>_dataset`` and skipped if that
        directory already exists.

        Parameters
        ----------
        x_counterfactual_list : list of torch.Tensor
            Counterfactual inputs.
        feedback : list of str
            One verdict per counterfactual; must match in length.
        y_source_list, y_target_list : list
            Source and target class per counterfactual.
        finetune_iteration : int
            Iteration directory to write into.
        hint_list : list of torch.Tensor, optional
            Segmentation hints stored alongside the samples.
        mode : str
            ``"train"`` or ``"validation"``.
        **kwargs
            Other tracked values, ignored.

        Returns
        -------
        str
            Path of the dataset directory.

        Raises
        ------
        Exception
            If ``x_counterfactual_list`` and ``feedback`` differ in length.
        """
        # x_counterfactual_list has 2 counterfactuals per image, generated in batches of 4
        # y_source_list and y_target_list have 1 entry per original image
        if not (len(x_counterfactual_list) == len(feedback)):
            # Previously this dropped into pdb when tracking_level >= 5, which every
            # experiment config sets, so the mismatch was never actually raised.
            raise Exception(
                "mismatch in list lengths while dataset creation: "
                f"{len(x_counterfactual_list)} counterfactuals vs {len(feedback)} feedback entries"
            )

        dataset_dir = os.path.join(
            self.base_dir, str(finetune_iteration), mode + "_dataset"
        )
        if os.path.exists(dataset_dir):
            return dataset_dir
        #
        x_list = []
        hint_list_dataset = []
        y_counterfactual_list = []
        sample_names = []

        # Counterfactuals were generated in batches of 4, with 2 per image
        # batch_size = 4
        # counterfactuals_per_batch = batch_size * 2

        for sample_idx in range(len(feedback)):
            if (
                self.adaptor_config.use_true_counterfactuals
                and feedback[sample_idx] == "true"
            ):
                sample_name = (
                    "true_"
                    + str(int(y_source_list[sample_idx]))
                    + "_to_"
                    + str(int(y_target_list[sample_idx]))
                    + "_"
                    + str(sample_idx)
                )
                x_list.append(x_counterfactual_list[sample_idx])
                if not hint_list is None:
                    hint_list_dataset.append(hint_list[sample_idx])

                y_counterfactual_list.append(int(y_target_list[sample_idx]))
                sample_names.append(sample_name)

            if feedback[sample_idx] == "false":
                sample_name = (
                    "false_"
                    + str(int(y_source_list[sample_idx]))
                    + "_to_"
                    + str(int(y_target_list[sample_idx]))
                    + "_"
                    + str(sample_idx)
                )
                x_list.append(x_counterfactual_list[sample_idx])
                if not hint_list is None:
                    hint_list_dataset.append(hint_list[sample_idx])

                y_counterfactual_list.append(int(y_source_list[sample_idx]))

                sample_names.append(sample_name)

        self.train_dataloader.dataset.serialize_dataset(
            output_dir=dataset_dir,
            x_list=x_list,
            y_list=y_counterfactual_list,
            hint_list=hint_list_dataset,
            sample_names=sample_names,
            classifier=self.student,
        )
        return dataset_dir

    def add_dataset_to_dataloader_mixer(
        self, dataloader_old, dataset_path, mixing_ratio, writer, finetune_iteration
    ):
        """Load a serialised counterfactual dataset and mix it with the old data.

        Parameters
        ----------
        dataloader_old : DataloaderMixer
            The data used so far.
        dataset_path : str
            Directory written by :meth:`create_dataset`.
        mixing_ratio : float
            Weight of the new counterfactual data; the old mixer is appended
            with weight ``1 - mixing_ratio``.
        writer : SummaryWriter
            Sample images of the new data are logged as ``train_<iteration>``.
        finetune_iteration : int
            Used for the log tag.

        Returns
        -------
        DataloaderMixer
            New mixer with ``return_src_internal=True`` and hints enabled when
            the run uses them.
        """
        # TODO adapt batch size so that it matches!
        dataloader, _, _ = create_dataloaders_from_datasource(
            config=self.data_config,
            datasource=dataset_path,
        )
        # import pdb; pdb.set_trace()
        log_images_to_writer(dataloader, writer, "train_" + str(finetune_iteration))
        dataloader = DataloaderMixer(self.adaptor_config.training, dataloader)
        # mixing ratio has to be flipped because in fact the old dataloader is the one appended
        dataloader.append(dataloader_old, weight_added_dataloader=1 - mixing_ratio)
        dataloader.return_src_internal = True
        if self.hints_enabled:
            dataloader.enable_hints()

        return dataloader

    def finetune_student(self, finetune_iteration, dataset_path, writer):
        """Finetune the student on the mixed data of one iteration.

        Loads the validation counterfactual dataset of the iteration (written
        by :meth:`retrieve_validation_stats`) and appends it to the joint
        validation loaders, adds ``dataset_path`` to the dataloader mixer,
        rebuilds the datastack and trains with ``ModelTrainer`` into
        ``base_dir/<iteration>/finetuned_model`` (an existing directory is
        moved aside). ``continuous_learning`` selects ``"finetune"``,
        ``"retrain"`` (fresh resnet18) or ``"deep_feature_reweighting"``
        (last layer only). Hints and dataset indices are disabled during
        training. Afterwards ``model.cpl`` is loaded as the new student and
        handed to the explainer. If the validation split is empty or training
        stops early, ``error_iteration_<n>.txt`` is written and the method
        returns without changing the student.

        Parameters
        ----------
        finetune_iteration : int
            Current iteration.
        dataset_path : str
            Training counterfactual dataset from :meth:`create_dataset`.
        writer : SummaryWriter
            Receives sample images of the new loaders.
        """
        #
        _log.info(
            "%s %s %s",
            "Finetune iteration: " + str(finetune_iteration),
            self.adaptor_config.tracking_level,
            2,
        )
        val_dataset_path = os.path.join(
            self.base_dir, str(finetune_iteration), "validation_dataset"
        )
        _, dataloader_val, _ = create_dataloaders_from_datasource(
            config=self.validation_data_config,
            datasource=val_dataset_path,
        )
        # A counterfactual validation split is small by construction: only "false"
        # verdicts enter it, and a single-component step 9 (DiDAE passing just the
        # confounder direction) halves the pool again because num_attempts ==
        # len(component_indices). Requiring 2 * val_batch_size rejected splits that
        # train perfectly well - e.g. 2 rows, one per class.
        #
        # Everything downstream already tolerates a short split:
        #   - log_images_to_writer swallows a missing batch (try/except continue);
        #   - the validation loop skips any dataloader with len < 1 (trainers.py:645);
        #   - the Logger is handed val_dataloaders[0], which is the ORIGINAL validation
        #     loader - this counterfactual one is appended after it.
        # So require only that the split is non-empty.
        min_val_samples = 1
        if (
            not isinstance(dataloader_val, torch.utils.data.DataLoader)
            or len(dataloader_val.dataset) < min_val_samples
        ):
            open(
                os.path.join(
                    self.adaptor_config.base_dir,
                    "error_iteration_" + str(finetune_iteration) + ".txt",
                ),
                "w",
            ).write("dataloader_val in " + str(finetune_iteration) + " is too empty!")
            return

        self.joint_validation_dataloader.append(dataloader_val)
        log_images_to_writer(
            dataloader_val, writer, "validation_" + str(finetune_iteration)
        )

        #
        if not hasattr(self, "dataloader_mixer"):
            self.dataloader_mixer = DataloaderMixer(
                self.adaptor_config.training, self.train_dataloader
            )
        self.dataloader_mixer = self.add_dataset_to_dataloader_mixer(
            dataloader_old=self.dataloader_mixer,
            dataset_path=dataset_path,
            mixing_ratio=self.adaptor_config.mixing_ratio,
            writer=writer,
            finetune_iteration=finetune_iteration,
        )
        _log.info("%s", "data stacking")
        self.datastack = DataStack(
            self.dataloader_mixer,
            self.output_size,
            transform=self.val_dataloader.dataset.transform,
        )
        if self.overwrite or not os.path.exists(
            os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "finetuned_model",
                "model.cpl",
            )
        ):
            if os.path.exists(
                os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model")
            ):
                shutil.move(
                    os.path.join(
                        self.base_dir, str(finetune_iteration), "finetuned_model"
                    ),
                    os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "finetuned_model_old_"
                        + datetime.now().strftime("%Y%m%d_%H%M%S"),
                    ),
                )

            Path(
                os.path.join(self.base_dir, str(finetune_iteration), "finetuned_model")
            ).mkdir(parents=True, exist_ok=True)
            if self.adaptor_config.continuous_learning == "retrain":
                # TODO this should be changed!
                self.student = TorchvisionModel("resnet18", 2)

            finetune_trainer = ModelTrainer(
                config=copy.deepcopy(self.adaptor_config),
                model=self.student,
                datasource=(self.dataloader_mixer, self.joint_validation_dataloader),
                model_path=os.path.join(
                    self.base_dir, str(finetune_iteration), "finetuned_model"
                ),
                only_last_layer=self.adaptor_config.continuous_learning
                == "deep_feature_reweighting",
            )
            if self.hints_enabled:
                self.dataloader_mixer.disable_hints()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.disable_hints()

            if isinstance(
                self.explainer.explainer_config, PerfectFalseCounterfactualConfig
            ):
                self.dataloader_mixer.disable_idx()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.disable_idx()

            try:
                finetune_trainer.fit(
                    continue_training=True
                )  # bool(self.adaptor_config.continuous_learning != "retrain"))

            except StopIteration:
                _log.info("%s", "Stopping finetune early due to little data!!!")
                _log.info("%s", "Stopping finetune early due to little data!!!")
                _log.info("%s", "Stopping finetune early due to little data!!!")
                open(
                    os.path.join(
                        self.adaptor_config.base_dir,
                        "error_iteration_" + str(finetune_iteration) + ".txt",
                    ),
                    "w",
                ).write(
                    "dataloader_val in " + str(finetune_iteration) + " is too empty!"
                )
                return
            if self.hints_enabled:
                self.dataloader_mixer.enable_hints()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.enable_hints()

            if isinstance(
                self.explainer.explainer_config, PerfectFalseCounterfactualConfig
            ):
                self.dataloader_mixer.enable_idx()
                for val_dataloader in self.joint_validation_dataloader.dataloaders:
                    val_dataloader.dataset.enable_idx()

        try:
            self.student = torch.load(
                os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "finetuned_model",
                    "model.cpl",
                ),
                map_location=self.device,
            )
        except Exception:
            self.student = torch.load(
                os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "finetuned_model",
                    "model.cpl",
                ),
                map_location=self.device,
                weights_only=False,
            )
        _log.info("%s", "loading finetuned model")
        self.explainer.predictor = self.student
        self.explainer.predictor_datasources = [
            self.dataloader_mixer,
            self.joint_validation_dataloader,
        ]

    def visualize_progress(self, paths):
        """Render the before/after comparison figure for a binary task.

        Uses ``create_comparison`` to show test samples with counterfactuals
        of the uncorrected and of the CFKD-corrected student side by side,
        with criteria for class, confounder (if the dataset exposes a
        ``Confounder`` attribute) and both predictions. Ranged explainer
        parameters (two-element lists) are replaced by their midpoint. Two
        images are saved per path: ``<path>`` and ``<path>_success.png`` (the
        latter filtered by a checkbox pattern of expected predictions).

        Parameters
        ----------
        paths : list of str
            ``.png`` output paths.

        Returns
        -------
        PIL.Image.Image
            The unfiltered comparison image.
        """
        task_config_buffer = copy.deepcopy(self.test_dataloader.dataset.task_config)
        # TODO use canonic explainer config!!
        criterions = {}
        if (
            isinstance(self.test_dataloader.dataset, Image2MixedDataset)
            and "Confounder" in self.test_dataloader.dataset.attributes
        ):
            self.test_dataloader.dataset.task_config = SimpleNamespace(
                **{
                    "y_selection": None,
                    "criterions": [],
                }
            )
            criterions["class"] = lambda X, y: int(
                y[
                    self.test_dataloader.dataset.attributes.index(
                        task_config_buffer.y_selection[0]
                    )
                ]
            )
            criterions["confounder"] = lambda X, y: int(
                y[self.test_dataloader.dataset.attributes.index("Confounder")]
            )
            criterions["uncorrected"] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device))
                .squeeze(0)
                .cpu()
                .argmax()
            )
            criterions["cfkd"] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        else:
            criterions["class"] = lambda X, y: int(y)
            criterions["uncorrected"] = lambda X, y: int(
                self.original_student(X.unsqueeze(0).to(self.device))
                .squeeze(0)
                .cpu()
                .argmax()
            )
            criterions["cfkd"] = lambda X, y: int(
                self.student(X.unsqueeze(0).to(self.device)).squeeze(0).cpu().argmax()
            )

        checkbox_dict = {
            "class": torch.tensor([0, 0, 0, 1, 1, 1]),
            "confounder": torch.tensor([1, 1, 1, 0, 0, 0]),
            "uncorrected": torch.tensor([1, 1, 1, 0, 0, 0]),
            "cfkd": torch.tensor([0, 0, 0, 1, 1, 1]),
        }
        # TODO introduce column for teacher
        explainer_config = copy.deepcopy(self.explainer.explainer_config)
        for attribute in self.explainer.explainer_config.__dict__.items():
            if isinstance(attribute[1], list) and len(attribute[1]) == 2:
                setattr(
                    explainer_config,
                    attribute[0],
                    0.5 * (attribute[1][1] + attribute[1][0]),
                )

        tracking_level_buffer = self.explainer.tracking_level
        self.explainer.tracking_level = 0.5
        img_success = create_comparison(
            explainer=self.explainer,
            dataset=self.test_dataloader.dataset,
            criterions=criterions,
            columns={
                "Counterfactual\nExplanation": [
                    "cf",
                    self.original_student,
                    "uncorrected",
                    os.path.join(self.adaptor_config.base_dir, "0"),
                ],
                "CFKD\ncorrected": [
                    "cf",
                    self.student,
                    "cfkd",
                    os.path.join(
                        self.adaptor_config.base_dir,
                        str(self.adaptor_config.current_iteration),
                    ),
                ],
            },
            score_reference_idx=1,
            device=self.device,
            checkbox_dict_in=checkbox_dict,
            batch_size=self.adaptor_config.batch_size,
            max_samples=100,
        )
        for path in paths:
            img_success.save(path.replace(".png", "_success.png"))
            cprint(
                "Saved: " + path.replace(".png", "_success.png"),
                self.adaptor_config.tracking_level,
                2,
            )

        img = create_comparison(
            explainer=self.explainer,
            dataset=self.test_dataloader.dataset,
            criterions=criterions,
            columns={
                "Counterfactual\nExplanation": [
                    "cf",
                    self.original_student,
                    "uncorrected",
                    os.path.join(self.adaptor_config.base_dir, "0"),
                ],
                "CFKD\ncorrected": [
                    "cf",
                    self.student,
                    "cfkd",
                    os.path.join(
                        self.adaptor_config.base_dir,
                        str(self.adaptor_config.current_iteration),
                    ),
                ],
            },
            score_reference_idx=1,
            device=self.device,
            batch_size=self.adaptor_config.batch_size,
            max_samples=100,
        )
        self.explainer.predictor = self.student
        self.explainer.tracking_level = tracking_level_buffer

        for path in paths:
            img.save(path)
            cprint("Saved: " + path, self.adaptor_config.tracking_level, 2)

        self.test_dataloader.dataset.task_config = task_config_buffer
        return img

    def retrieve_validation_prestats(self, finetune_iteration):
        """Validation counterfactuals and pre-feedback statistics of an iteration.

        Runs ``calculate_validation_statistics`` ``validation_runs`` times
        (interpolating ranged explainer parameters between runs) on the joint
        validation loaders and averages the statistics (error matrix,
        confidence score stats, ...). Results are cached as
        ``validation_tracked_values.npz`` and ``validation_prestats.npz`` in
        ``base_dir/<iteration>/`` at ``tracking_level >= 3`` and loaded from
        there otherwise, with ``collage_path_list`` rebuilt from
        ``validation_collages0``. If the explainer clusters, the explanations
        are clustered and stored as ``validation_tracked_cluster_values.npz``;
        2-D latent datasets additionally get ``val_counterfactuals_global.png``.

        Parameters
        ----------
        finetune_iteration : int
            Iteration directory to use.

        Returns
        -------
        tuple
            ``(validation_tracked_values, validation_stats)``.
        """
        validation_values_path = os.path.join(
            self.base_dir, str(finetune_iteration), "validation_tracked_values.npz"
        )
        if self.overwrite or not os.path.exists(validation_values_path):
            cprint(
                "calculate validation tracked values from scratch!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            x_list_collection = []
            x_counterfactual_collection = []
            y_confidence_list = []
            original_explainer_config = copy.deepcopy(self.explainer.explainer_config)
            validation_tracked_values = None
            validation_stats = []
            for i in range(self.adaptor_config.validation_runs):
                cprint(
                    "Validation run: " + str(i), self.adaptor_config.tracking_level, 2
                )
                self.explainer.explainer_config = copy.deepcopy(
                    original_explainer_config
                )
                if self.adaptor_config.validation_runs > 1:
                    for attribute in self.explainer.explainer_config.__dict__.items():
                        if isinstance(attribute[1], list) and len(attribute[1]) == 2:
                            if i == 0:
                                effective_idx = 0

                            else:
                                effective_idx = (
                                    i
                                    / (self.adaptor_config.validation_runs - 1)
                                    * (attribute[1][1] - attribute[1][0])
                                )

                            setattr(
                                self.explainer.explainer_config,
                                attribute[0],
                                attribute[1][0] + effective_idx,
                            )

                validation_collages_base_path = os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "validation_collages" + str(i),
                )
                used_dataloaders = (
                    [self.joint_validation_dataloader.dataloaders[0]]
                    if finetune_iteration == self.adaptor_config.finetune_iterations
                    else self.joint_validation_dataloader.dataloaders
                )
                (
                    validation_tracked_values_current,
                    validation_stats_current,
                ) = calculate_validation_statistics(
                    model=self.student,
                    dataloaders=used_dataloaders,
                    tracked_keys=self.tracked_keys,
                    base_path=validation_collages_base_path,
                    output_size=self.output_size,
                    explainer=self.explainer,
                    device=self.device,
                    logits_to_prediction=self.logits_to_prediction,
                    use_confusion_matrix=self.adaptor_config.use_confusion_matrix,
                    max_validation_samples=self.adaptor_config.max_validation_samples,
                )
                # torch.nn.functional.softmax(
                # self.student(validation_tracked_values_current['x_counterfactual_list'][i]
                # .unsqueeze(0).to('cuda')).squeeze(0))[validation_tracked_values_current['y_target_list'][i]]
                if validation_tracked_values is None:
                    validation_tracked_values = validation_tracked_values_current

                else:
                    for key in validation_tracked_values.keys():
                        validation_tracked_values[key].extend(
                            validation_tracked_values_current[key]
                        )

                validation_stats.append(validation_stats_current)

                if self.adaptor_config.validation_runs > 1:
                    x_list_collection.append(
                        copy.deepcopy(validation_tracked_values_current["x_list"])
                    )
                    x_counterfactual_collection.append(
                        copy.deepcopy(
                            validation_tracked_values_current["x_counterfactual_list"]
                        )
                    )
                    y_confidence_list.append(
                        copy.deepcopy(
                            validation_tracked_values_current[
                                "y_target_end_confidence_list"
                            ]
                        )
                    )

            validation_stats = {
                key: torch.mean(
                    torch.stack(
                        [
                            torch.tensor(validation_stats_current[key])
                            for validation_stats_current in validation_stats
                        ]
                    ),
                    dim=0,
                )
                for key in validation_stats[0].keys()
            }
            self.explainer.explainer_config = original_explainer_config

            if self.adaptor_config.tracking_level >= 3:
                os.makedirs(
                    os.path.join(self.base_dir, str(finetune_iteration)), exist_ok=True
                )
                with open(
                    validation_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in self.tracked_keys:
                        try:
                            val = validation_tracked_values.get(key, None)
                            if val is None:
                                continue
                            if not isinstance(
                                val, (list, tuple, torch.Tensor, np.ndarray)
                            ):
                                tracked_values_file[key] = np.array(val)
                            elif len(val) == 0:
                                continue
                            elif isinstance(val[0], torch.Tensor):
                                tracked_values_file[key] = (
                                    torch.stack(list(val), dim=0).detach().cpu().numpy()
                                )
                            else:
                                tracked_values_file[key] = np.array(val)

                        except Exception as e:
                            _log.info(
                                "%s",
                                f"Failed to log validation stats array for key {key}: {e}",
                            )

                    np.savez(f, **tracked_values_file)

                with open(
                    os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "validation_prestats.npz",
                    ),
                    "wb",
                ) as f:
                    validation_stats_file = {}
                    for key in validation_stats.keys():
                        if isinstance(validation_stats[key], torch.Tensor):
                            validation_stats_file[key] = validation_stats[key].numpy()

                        elif isinstance(validation_stats[key], int) or isinstance(
                            validation_stats[key], float
                        ):
                            validation_stats_file[key] = np.array(validation_stats[key])

                    np.savez(f, **validation_stats_file)

        else:
            # TODO think about this again
            if self.adaptor_config.tracking_level > 0:
                cprint(
                    "load validation tracked values!!!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                with open(
                    validation_values_path,
                    "rb",
                ) as f:
                    validation_tracked_values = {}
                    validation_tracked_value_file = np.load(f, allow_pickle=True)
                    for key in validation_tracked_value_file.keys():
                        if key == "collage_path_list":
                            continue
                        validation_tracked_values[key] = list(
                            torch.tensor(validation_tracked_value_file[key])
                        )
                    # Caches written before the layout fix hold one row per factual
                    # for some entries and one per counterfactual for others. Expand
                    # them on load so a run that was interrupted at clustering or at
                    # teacher feedback resumes instead of regenerating.
                    validation_tracked_values = flatten_explanations(
                        validation_tracked_values,
                        batch_size=self.adaptor_config.batch_size,
                    )

                cprint(
                    "load validation prestats!!!", self.adaptor_config.tracking_level, 2
                )
                with open(
                    os.path.join(
                        self.base_dir,
                        str(finetune_iteration),
                        "validation_prestats.npz",
                    ),
                    "rb",
                ) as f:
                    validation_stats = {}
                    validation_tracked_file = np.load(f, allow_pickle=True)
                    for key in validation_tracked_file.keys():
                        validation_stats[key] = torch.tensor(
                            validation_tracked_file[key]
                        )

            if "collage_path_list" in self.tracked_keys:
                cprint(
                    "recreate validation collage path!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                get_collage_path = lambda x: os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "validation_collages" + str(x),
                )
                idx = 0
                collage_path_list = []
                while os.path.exists(get_collage_path(idx)):
                    collage_paths = os.listdir(get_collage_path(idx))
                    collage_paths.sort()
                    collage_paths = list(
                        filter(lambda x: x[-4:] == ".png", collage_paths)
                    )
                    collage_path_list.extend(collage_paths)
                    idx += 1
                    # TODO this is a bug, but currently not used
                    if idx == 1:
                        break

                validation_tracked_values["collage_path_list"] = list(
                    map(
                        lambda x: os.path.join(
                            self.base_dir,
                            str(finetune_iteration),
                            "validation_collages0",
                            x,
                        ),
                        collage_path_list,
                    )
                )

        if self.adaptor_config.explainer.use_clustering:
            validation_cluster_values_path = os.path.join(
                self.base_dir,
                str(finetune_iteration),
                "validation_tracked_cluster_values.npz",
            )
            """
            TODO loading collage paths does not work yet...
            if os.path.exists(validation_cluster_values_path):
                cprint(
                    "load clustered counterfactual explanations!",
                    self.adaptor_config.tracking_level,
                    2,
                )
                with open(
                    validation_cluster_values_path,
                    "rb",
                ) as f:
                    validation_tracked_values = {}
                    validation_tracked_value_file = np.load(f, allow_pickle=True)
                    for key in validation_tracked_value_file.keys():
                        validation_tracked_values[key] = list(
                            torch.tensor(validation_tracked_value_file[key])
                        )

            else:
            """
            cprint(
                "cluster counterfactual explanations!",
                self.adaptor_config.tracking_level,
                2,
            )
            validation_tracked_values = self.explainer.cluster_explanations(
                validation_tracked_values,
                self.adaptor_config.batch_size,
                self.adaptor_config.explainer.num_attempts
                * self.adaptor_config.explainer.parallel_attempts,
            )
            if self.adaptor_config.tracking_level >= 3:
                with open(
                    validation_cluster_values_path,
                    "wb",
                ) as f:
                    tracked_values_file = {}
                    for key in validation_tracked_values.keys():
                        val = validation_tracked_values[key]
                        if not isinstance(val, (list, tuple, torch.Tensor, np.ndarray)):
                            tracked_values_file[key] = np.array(val)
                        elif len(val) == 0:
                            continue
                        elif isinstance(val[0], torch.Tensor):
                            tracked_values_file[key] = (
                                torch.stack(list(val), dim=0).detach().cpu().numpy()
                            )
                        else:
                            tracked_values_file[key] = np.array(val)

                    np.savez(f, **tracked_values_file)

        if self.adaptor_config.tracking_level >= 4 and hasattr(
            self.joint_validation_dataloader.dataloaders[0].dataset,
            "sample_to_2d_latent",
        ):
            attempts = getattr(self.adaptor_config.explainer, "num_attempts", 1)

            self.joint_validation_dataloader.dataloaders[
                0
            ].dataset.global_counterfactual_visualization(
                os.path.join(
                    self.base_dir,
                    str(finetune_iteration),
                    "val_counterfactuals_global.png",
                ),
                validation_tracked_values["x_list"],
                validation_tracked_values["x_counterfactual_list"],
                validation_tracked_values["y_target_start_confidence_list"],
                validation_tracked_values["y_target_end_confidence_list"],
                validation_tracked_values["y_target_list"],
                validation_tracked_values.get("hint_list"),
                attempts=attempts,
            )
            cprint(
                "global counterfactual visualization saved!!!",
                self.adaptor_config.tracking_level,
                2,
            )

        return validation_tracked_values, validation_stats

    def retrieve_validation_stats(
        self, finetune_iteration, validation_tracked_values, validation_prestats
    ):
        """Complete the validation statistics with teacher feedback.

        If ``base_dir/<iteration>/validation_stats.npz`` exists and
        ``overwrite`` is off it is loaded (back-filling ``unbiasedness`` for
        old files when possible). Otherwise the teacher judges the validation
        counterfactuals, :meth:`calculate_feedback_stats` adds the explainer
        metrics to ``validation_prestats``, the judged counterfactuals are
        serialised as ``<iteration + 1>/validation_dataset`` for the next
        finetuning and the merged dict is saved.

        Parameters
        ----------
        finetune_iteration : int
            Iteration the validation values belong to.
        validation_tracked_values : dict
            From :meth:`retrieve_validation_prestats`.
        validation_prestats : dict
            From :meth:`retrieve_validation_prestats`; extended in place.

        Returns
        -------
        dict
            Statistics including the feedback metrics.
        """
        if not self.overwrite and os.path.exists(
            os.path.join(self.base_dir, str(finetune_iteration), "validation_stats.npz")
        ):
            cprint(
                "load already completed validation stats!!!",
                self.adaptor_config.tracking_level,
                2,
            )
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), "validation_stats.npz"
                ),
                "rb",
            ) as f:
                validation_stats = {}
                validation_tracked_file = np.load(f, allow_pickle=True)
                for key in validation_tracked_file.keys():
                    value = validation_tracked_file[key]
                    if value.ndim == 0:
                        validation_stats[key] = value.item()
                    else:
                        validation_stats[key] = torch.tensor(value)

            should_backfill_unbiasedness = (
                finetune_iteration == 0
                and self.adaptor_config.calculate_explainer_stats
                and self.adaptor_config.calculate_group_accuracies
                and "unbiasedness" not in validation_stats
                and "feedback_accuracy_distilled" in validation_stats
            )
            if should_backfill_unbiasedness:
                unbiasedness = self.calculate_unbiasedness(
                    validation_stats["feedback_accuracy_distilled"]
                )
                if unbiasedness is not None:
                    validation_stats["unbiasedness"] = unbiasedness
                    with open(
                        os.path.join(
                            self.base_dir,
                            str(finetune_iteration),
                            "validation_stats.npz",
                        ),
                        "wb",
                    ) as f:
                        validation_stats_file = {}
                        for key, value in validation_stats.items():
                            if isinstance(value, torch.Tensor):
                                validation_stats_file[key] = value.cpu().numpy()
                            elif isinstance(value, (int, float)):
                                validation_stats_file[key] = np.array(value)
                        np.savez(f, **validation_stats_file)

            cprint("validation stats loaded!!!", self.adaptor_config.tracking_level, 2)
            return validation_stats

        validation_stats = validation_prestats

        validation_feedback = self.retrieve_feedback(
            tracked_values=validation_tracked_values,
            finetune_iteration=finetune_iteration,
            mode="validation",
        )
        validation_feedback_stats = self.calculate_feedback_stats(
            tracked_values=validation_tracked_values,
            feedback=validation_feedback,
            finetune_iteration=finetune_iteration,
        )
        self.create_dataset(
            feedback=validation_feedback,
            finetune_iteration=finetune_iteration + 1,
            mode="validation",
            config=self.validation_data_config,
            **validation_tracked_values,
        )

        for key in validation_feedback_stats.keys():
            validation_stats[key] = validation_feedback_stats[key]

        if self.adaptor_config.tracking_level >= 3:
            with open(
                os.path.join(
                    self.base_dir, str(finetune_iteration), "validation_stats.npz"
                ),
                "wb",
            ) as f:
                validation_stats_file = {}
                for key in validation_stats.keys():
                    if isinstance(validation_stats[key], torch.Tensor):
                        validation_stats_file[key] = validation_stats[key].numpy()

                    elif isinstance(validation_stats[key], int) or isinstance(
                        validation_stats[key], float
                    ):
                        validation_stats_file[key] = np.array(validation_stats[key])

                np.savez(f, **validation_stats_file)

        return validation_stats

    def run(self):
        """
        Run the counterfactual knowledge distillation

        Starting at ``current_iteration + 1`` and up to ``finetune_iterations``,
        every iteration retrieves training counterfactuals and their feedback,
        finalises the previous iteration's validation statistics (logged as
        ``validation_*`` scalars), builds the counterfactual dataset, finetunes
        the student, logs ``val_accuracy``/``test_accuracy`` (plus group
        accuracies and ``gain``, the relative improvement of the mean group
        accuracy, when enabled), computes fresh validation prestats for the
        next iteration, saves ``model.cpl`` and advances ``current_iteration``
        in ``config.yaml``. The post-loop validation pass is skipped unless
        ``PEAL_FULL_VALIDATION_TAIL=1``.

        Returns
        -------
        torch.nn.Module
            The finetuned student.
        """
        cprint(
            "Adaptor Config: " + str(self.adaptor_config),
            self.adaptor_config.tracking_level,
            4,
        )
        validation_prestats, validation_tracked_values, writer = self.initialize_run()

        # iterate over the finetune iterations
        for finetune_iteration in range(
            self.adaptor_config.current_iteration + 1,
            self.adaptor_config.finetune_iterations + 1,
        ):
            cprint(
                "Start retrieving training counterfactuals for iteration "
                + str(finetune_iteration),
                self.adaptor_config.tracking_level,
                2,
            )
            tracked_values = self.retrieve_counterfactual_list(
                validation_stats=validation_prestats,
                finetune_iteration=finetune_iteration,
            )

            feedback = self.retrieve_feedback(
                tracked_values=tracked_values,
                finetune_iteration=finetune_iteration,
                mode="train",
            )
            validation_stats = self.retrieve_validation_stats(
                finetune_iteration=finetune_iteration - 1,
                validation_prestats=validation_prestats,
                validation_tracked_values=validation_tracked_values,
            )
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        finetune_iteration - 1,
                    )

            self.adaptor_config.feedback_accuracies.append(
                validation_stats["feedback_accuracy"]
            )

            dataset_path = self.create_dataset(
                feedback=feedback,
                finetune_iteration=finetune_iteration,
                mode="train",
                config=self.data_config,
                **tracked_values,
            )
            self.finetune_student(
                finetune_iteration=finetune_iteration,
                dataset_path=dataset_path,
                writer=writer,
            )

            hints_enabled_buffer = self.val_dataloader.dataset.hints_enabled
            if hints_enabled_buffer:
                self.val_dataloader.dataset.disable_hints()

            val_accuracy = calculate_test_accuracy(
                self.student,
                self.val_dataloader,
                self.device,
                False,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )
            cprint(
                "val_accuracy: " + str(val_accuracy),
                self.adaptor_config.tracking_level,
                2,
            )
            writer.add_scalar("val_accuracy", val_accuracy, finetune_iteration)
            if hints_enabled_buffer:
                self.val_dataloader.dataset.enable_hints()

            test_accuracy = calculate_test_accuracy(
                self.student,
                self.test_dataloader,
                self.device,
                self.adaptor_config.calculate_group_accuracies,
                self.adaptor_config.max_test_batches,
                tracking_level=self.adaptor_config.tracking_level,
            )
            if self.adaptor_config.calculate_group_accuracies:
                (
                    test_accuracy,
                    group_accuracies,
                    group_distribution,
                    groups,
                    worst_group_accuracy,
                ) = test_accuracy
                for idx in range(len(group_accuracies)):
                    writer.add_scalar(
                        "test_group_accuracy_" + str(idx),
                        group_accuracies[idx],
                        finetune_iteration,
                    )
                    writer.add_scalar(
                        "test_group_distribution_" + str(idx),
                        group_distribution[idx],
                        finetune_iteration,
                    )

                writer.add_scalar(
                    "test_worst_group_accuracy",
                    worst_group_accuracy,
                    finetune_iteration,
                )
                cprint(
                    "group_accuracies: " + str(group_accuracies),
                    self.adaptor_config.tracking_level,
                    2,
                )
                self.adaptor_config.group_accuracies.append(group_accuracies)
                save_yaml_config(
                    self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
                )
                cprint(
                    "group_distribution: " + str(group_distribution),
                    self.adaptor_config.tracking_level,
                    2,
                )
                cprint(
                    "group_sizes: " + str(groups), self.adaptor_config.tracking_level, 2
                )
                cprint(
                    "worst_group_accuracy: " + str(worst_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                avg_group_accuracy = float(np.mean(group_accuracies))
                cprint(
                    "avg_group_accuracy: " + str(avg_group_accuracy),
                    self.adaptor_config.tracking_level,
                    2,
                )
                writer.add_scalar(
                    "test_avg_group_accuracy", avg_group_accuracy, finetune_iteration
                )
                old_avg_group_accuracy = np.mean(
                    self.adaptor_config.group_accuracies[-2]
                )
                gain = (avg_group_accuracy - old_avg_group_accuracy) / (
                    1 - old_avg_group_accuracy
                )
                cprint(
                    "gain: " + str(gain),
                    self.adaptor_config.tracking_level,
                    2,
                )
                writer.add_scalar("gain", gain, finetune_iteration)

            writer.add_scalar("test_accuracy", test_accuracy, finetune_iteration)
            cprint(
                "test_accuracy: " + str(test_accuracy),
                self.adaptor_config.tracking_level,
                2,
            )
            cprint(
                "Start to retrieve validation stats",
                self.adaptor_config.tracking_level,
                2,
            )

            decision_boundary_path = os.path.join(
                self.base_dir, str(finetune_iteration), "decision_boundary.png"
            )
            if (
                hasattr(
                    self.joint_validation_dataloader.dataloaders[0].dataset,
                    "sample_to_2d_latent",
                )
                and not os.path.exists(decision_boundary_path)
                and self.adaptor_config.tracking_level >= 4
            ):
                self.joint_validation_dataloader.dataloaders[
                    0
                ].dataset.visualize_decision_boundary(
                    self.student,
                    self.adaptor_config.training.test_batch_size,
                    self.device,
                    decision_boundary_path,
                    temperature=self.adaptor_config.explainer.temperature,
                    train_dataloader=self.dataloader_mixer,
                    val_dataloaders=self.joint_validation_dataloader,
                    test_dataloader=self.test_dataloader,
                )

            # These prestats are consumed only by the NEXT loop iteration
            # (retrieve_counterfactual_list(validation_stats=validation_prestats, ...)).
            # On the final iteration nothing reads them, and they are expensive: a full
            # distill_predictor plus a fresh round of validation counterfactuals against
            # the just-finetuned student, ~15 min per run here. Every reported metric is
            # already written by then -- `gain` at step finetune_iteration, and all the
            # validation_* Table 1 scalars at step 0 from the pre-finetune pass -- so
            # skipping the last one changes no logged value.
            if finetune_iteration < self.adaptor_config.finetune_iterations:
                (
                    validation_tracked_values,
                    validation_prestats,
                ) = self.retrieve_validation_prestats(
                    finetune_iteration=finetune_iteration
                )

            visualization_path = os.path.join(
                self.base_dir, str(finetune_iteration), "visualization.png"
            )
            if (
                self.output_size == 2
                and self.adaptor_config.tracking_level >= 6
                and not os.path.exists(visualization_path)
            ):
                self.visualize_progress(
                    [
                        visualization_path,
                        os.path.join(self.base_dir, "visualization.png"),
                    ]
                )

            torch.save(self.student, os.path.join(self.base_dir, "model.cpl"))

            self.adaptor_config.current_iteration = (
                self.adaptor_config.current_iteration + 1
            )
            save_yaml_config(
                self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
            )

        # Post-loop validation statistics. This regenerates validation counterfactuals
        # against the finetuned student and re-runs distill_binary_dataset -- ~15 min a
        # run, and it is the bulk of the post-gain tail, NOT the prestats call guarded
        # above (an earlier version of that guard's rationale wrongly attributed the
        # whole tail to the prestats).
        # It writes validation_* at step current_iteration and appends one
        # feedback_accuracy. Nothing reported depends on it: `gain` and test_* are
        # already logged inside the loop, and create_didae_table1.py reads every
        # validation_* tag at step 0, i.e. from the pre-finetune pass.
        # It is also now fed stale prestats: the guard above skips the final
        # retrieve_validation_prestats, so validation_prestats/validation_tracked_values
        # here are the iteration-0 ones. Computing step-1 statistics from iteration-0
        # prestats would be worse than not computing them, so skip it under the same
        # condition rather than leaving the two halves inconsistent.
        # Set PEAL_FULL_VALIDATION_TAIL=1 to restore the original behaviour.
        if os.environ.get("PEAL_FULL_VALIDATION_TAIL", "0") == "1":
            validation_stats = self.retrieve_validation_stats(
                finetune_iteration=self.adaptor_config.current_iteration,
                validation_prestats=validation_prestats,
                validation_tracked_values=validation_tracked_values,
            )
            for key in validation_stats.keys():
                if isinstance(validation_stats[key], float):
                    writer.add_scalar(
                        "validation_" + key,
                        validation_stats[key],
                        self.adaptor_config.current_iteration,
                    )

            self.adaptor_config.feedback_accuracies.append(
                validation_stats["feedback_accuracy"]
            )

        return self.student
