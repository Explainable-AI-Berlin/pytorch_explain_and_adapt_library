"""
DiDAE: Dictionary-based Interpretable Diffusion Autoencoder Explanations.

Implements a 9-step automated counterfactual workflow:
  1) Distill user classifier into reference encoder space (closed-form least squares)
  2) Sweep all SAE directions in latent space
  3) Filter latent-space non-flips
  4) Decode surviving counterfactuals
  5) Filter ambient-space non-flips
  6) Rank directions by success count
  7) Display top-K directions via ClusterTeacher
  8) User labels directions true/false
  9) CFKD fine-tuning on false directions
"""

import copy
import gc
import os
import shutil

import torch
import torchvision
from pathlib import Path
from types import SimpleNamespace
from torch import nn
from torch.utils.data import DataLoader
from peal.log import get_logger

_log = get_logger(__name__)


try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:

        class DummySummaryWriter:
            """No-op stand-in used when neither tensorboard nor tensorboardX imports."""

            def __init__(self, *args, **kwargs):
                """Accept and discard any SummaryWriter constructor arguments."""

            def add_scalar(self, *args, **kwargs):
                """Discard a scalar log call."""

            def add_image(self, *args, **kwargs):
                """Discard an image log call."""

            def close(self):
                """Do nothing; there is no file to flush."""

        SummaryWriter = DummySummaryWriter
from tqdm import tqdm
from typing import Union
from pydantic import PositiveInt

from peal.adaptors.interfaces import AdaptorConfig, Adaptor
from peal.architectures.interfaces import TaskConfig
from peal.architectures.predictors import get_predictor
from peal.data.dataloaders import (
    DataStack,
    DataloaderMixer,
    create_dataloaders_from_datasource,
    WeightedDataloaderList,
)
from peal.data.interfaces import DataConfig
from peal.explainers.interfaces import ExplainerConfig
from peal.explainers.counterfactual_explainer import DAEdistillConfig
from peal.generators.generator_factory import get_generator
from peal.generators.interfaces import GeneratorConfig
from peal.global_utils import load_yaml_config, save_yaml_config
from peal.sparse_dictionaries.interfaces import SparseDictionaryConfig
from peal.sparse_dictionaries.sparse_dictionary_factory import get_sparse_dictionary
from peal.teachers.teacher_factory import get_teacher
from peal.training.interfaces import TrainingConfig, PredictorConfig
from peal.training.trainers import (
    calculate_test_accuracy,
)


class DiDAEConfig(AdaptorConfig):
    """Config for the DiDAE (Dictionary-based Interpretable DAE) adaptor.

    Every field is documented by the string literal that follows it below,
    which is what Sphinx renders as that field's attribute documentation. The
    most important groups are summarised here.

    Parameters
    ----------
    student, generator, sparse_dictionary, teacher, data, test_data, task, training, explainer : config or path
        The components DiDAE wires together: the classifier to repair, the
        diffusion autoencoder whose ``z_sem`` is edited, the dictionary of
        directions swept in that space, the feedback source for step 8, the
        datasets and the CFKD explainer used in step 9.
    probe_sparse_dictionary : config, optional
        Separate dictionary spanning the LASSO basis of the distilled probe
        (step 1); ``None`` uses ``sparse_dictionary``.
    lasso_alpha, lasso_alphas : float, list
        LASSO strength of the probe fit, and extra strengths to report only.
    top_k_directions, max_cf_per_direction : int
        How many ranked directions are shown to the teacher and how many
        factual/counterfactual pairs are kept per direction.
    component_bounds_scale, edit_depth_factor, sweep_atom_subset, linesearch_factors, decode_batch_size : sweep knobs
        Control the empirical ``[c_min, c_max]`` clamp, edit length, atom
        subset, line search and decoding batch of the direction sweep.
    concept_replacement, concept_replacement_candidates, concept_replacement_unique, concept_replacement_unique_strategy : pair mode
        Pair mode: rank (deactivate, activate) concept pairs instead of
        single atoms.
    latent_only, use_generator_data_for_matching : bool
        Stop after the latent half of the workflow; where step 1.5 draws the
        ground-truth attributes from.
    run_cfkd_on_false_directions, cfkd_false_direction_min_share, cfkd_max_false_directions, cfkd_teacher, use_true_counterfactuals, finetune_iterations, min_train_samples, max_validation_samples, counterfactual_type, continuous_learning : step-9 knobs
        Whether and how CFKD finetunes on the directions judged false; the
        first two sample counts also size the discovery sweep pool.
    max_decode_directions, max_decode_per_direction, decode_selection, service_mode
        Decode budgets; forbidden outside ``service_mode`` (see ``__init__``).
    base_dir, overwrite, seed, in_memory, export_onnx, calculate_group_accuracies, max_test_batches, batch_size
        Output directory, caching and evaluation settings.
    n_samples : int
        Deprecated and ignored.
    """

    adaptor_type: str = "DiDAE"
    n_samples: int = 200
    """Deprecated. The discovery sweep now draws min_train_samples +
    max_validation_samples rows the way CFKD seeds its counterfactuals, so this
    knob is ignored; it is kept only so existing configs still load. It used to
    double as a pool switch -- the sweep read the validation split alone unless
    n_samples exceeded it -- which made raising the budget silently change the
    data as well."""
    top_k_directions: int = 10
    """How many top SAE directions to show the user."""
    max_cf_per_direction: int = 5
    """Max representative counterfactual pairs stored per direction."""
    max_export_per_direction: Union[type(None), PositiveInt] = 20
    """Verified flips per direction written as images to
    ``<base_dir>/successful_flips`` (at least max_cf_per_direction); None writes
    every verified flip (the web demo shows them all)."""
    max_decode_directions: Union[type(None), PositiveInt] = None
    """DISABLED (2026-09-14): every latent flip inside the empirical bounds is
    rendered and checked for an ambient flip, exactly like the classical CFKD
    explainer renders every counterfactual. A budgeted sweep (8 "deepest"
    decodes per direction) is not comparable to CFKD and made the discovery of
    Male on CelebA depend on decode noise. Setting either budget raises."""
    max_decode_per_direction: Union[type(None), PositiveInt] = None
    decode_selection: str = "random"
    """Which of a direction's latent flips max_decode_per_direction decodes:
    "random" (unbiased rate estimate) or "deepest" (the flips that landed
    furthest past the boundary, i.e. the ones most likely to survive a decode)."""
    edit_depth_factor: Union[float, None] = None
    """Shorten every rendered edit to this multiple of the smallest step along its
    direction that crosses the distilled probe's boundary (None: full replacement /
    full step to the bound). 2-3 keeps most flips with the RAE's ~0.25 realisation."""
    student_logits_cache: Union[str, None] = None
    """Optional ``.npz`` with ``keys`` (image keys) and ``logits`` (the student's
    two outputs per image), e.g. the web demo's ``peal.web.cache``. Step 1 then
    takes the student's margins from it instead of running the student over the
    train split again; with a missing key it falls back to the student."""
    sweep_atom_subset: Union[list, None] = None
    """Single-atom mode only: render the flips of these dictionary atoms only (a targeted
    experiment, e.g. the confounder atom alone); None = every atom with a latent flip."""
    component_bounds_scale: float = 1.0
    """Widen (>1) or tighten (<1) the empirical [c_min, c_max] clamp about its
    midpoint. That clamp, not the boundary, is what limits the edit in practice.
    1.0 uses the bounds in c_min_and_maxes.txt as written."""
    concept_replacement: bool = False
    """Rank pairs of concepts -- turn one currently-active concept off and one
    currently-inactive concept on -- instead of stepping a single atom toward the
    decision boundary. The single-atom step is limited by the [c_min, c_max]
    clamp rather than by the dictionary; a replacement drives both atoms to the
    bounds themselves, so it gets two atoms' full range and the clamp cannot cut
    it short. Latent-only for now."""
    concept_replacement_candidates: int = 64
    """How many off- and on-candidates the replacement grid scores, M. Cost is
    O(n_samples * M^2), so raising it is cheap."""
    concept_replacement_unique: bool = False
    """Report a one-to-one matching instead of the raw pair ranking: each concept
    may be deactivated at most once and activated at most once. The unconstrained
    ranking repeats its strongest concepts -- on NICO/MSAE seven of the top ten
    all deactivate #2145 crocodile -- so it reads as ten findings when it is
    closer to two. Maximum-weight matching, not a greedy walk down the list."""
    concept_replacement_unique_strategy: str = "greedy"
    """How the one-to-one selection is made. "greedy" walks the ranking by flip
    count and keeps a pair whenever neither concept is spoken for -- the best
    pair survives and each entry is the best remaining claim. "matching" takes
    the maximum-weight one-to-one matching instead, which maximises total flips
    and can drop the single strongest pair to do it (on NICO/MSAE: 718 flips
    without #1, against 698 with it)."""
    run_cfkd_on_false_directions: bool = True
    """Whether step 9 runs at all. False stops after the teacher's verdicts, which
    is the whole of the discovery half and everything a hyperparameter sweep of
    the sweep reads. Step 9 builds a full nested CFKD -- its counterfactual
    generation reached RSS 16 GB here and the Slurm step's cgroup SIGKILLed the
    process mid-stage, so leaving it on costs ~20 min per run and returns
    nothing."""
    cfkd_false_direction_min_share: float = 0.25
    """Step 9 hands CFKD only the false directions whose verified flip count is
    at least this share of the strongest false direction's. Every direction
    given to CFKD takes an equal share of the finetuning counterfactuals
    (num_attempts == len(component_indices)), so a direction the teacher
    marked false on 2 verified flips out of 12 latent ones dilutes the
    confounder's counterfactuals without adding evidence (2026-09-14: five
    false directions at batch 80 each gave gain ~0 where the oracle pair at
    200 gives 0.25). 0 keeps every false direction."""
    cfkd_max_false_directions: Union[type(None), PositiveInt] = None
    """Hard cap on how many false directions step 9 finetunes on (None: no cap)."""
    linesearch_factors: list = ["dynamic"]
    """Linesearch factors for counterfactual generation."""
    decode_batch_size: int = 32
    """Batch size for decoding latent counterfactuals."""
    min_train_samples: PositiveInt = 800
    """Minimum training samples for CFKD finetuning."""
    max_validation_samples: PositiveInt = 200
    """Max validation samples."""
    max_test_batches: Union[type(None), PositiveInt] = None
    """Max test batches."""
    finetune_iterations: int = 1
    """Number of CFKD finetuning iterations after user feedback."""
    task: Union[TaskConfig, type(None)] = None
    """Task config."""
    explainer: Union[dict, ExplainerConfig] = DAEdistillConfig()
    """Explainer config (for CFKD phase)."""
    training: Union[TrainingConfig, type(None)] = TrainingConfig()
    """Training config."""
    data: DataConfig = None
    """Data config."""
    test_data: DataConfig = None
    """Test data config."""
    student: Union[PredictorConfig, str, type(None)] = None
    """Student model path."""
    teacher: Union[str, dict] = "cluster@8000"
    """Teacher interface."""
    generator: Union[GeneratorConfig, type(None)] = None
    """Generator config path."""
    sparse_dictionary: Union[SparseDictionaryConfig, dict, type(None)] = None
    """Sparse dictionary config path."""
    probe_sparse_dictionary: Union[SparseDictionaryConfig, dict, str, type(None)] = None
    """Dictionary whose atoms form the LASSO basis of the distilled probe (step 1)
    when it should differ from `sparse_dictionary`, the direction set of the sweep.
    Measured 2026-09-12 on CelebA/OpenAI-CLIP: a probe fitted in the 40-atom
    Procrustes span (r 0.94) gave 20/259 ambient flips and missed Male, the same
    directions swept under the probe fitted in the 6144-atom MSAE span (r 0.98)
    gave 85/258 and found Male. None = use sparse_dictionary."""
    base_dir: str = "peal_runs/didae"
    """Base directory."""
    current_iteration: int = 0
    """Current iteration tracking."""
    continuous_learning: str = "deep_feature_reweighting"
    """Continuous learning strategy."""
    batch_size: PositiveInt = 200
    """Batch size for counterfactual generation."""
    calculate_group_accuracies: bool = False
    """Calculate group accuracies."""
    lasso_alpha: float = 0.0
    """SAE-Sparse LASSO regularization alpha (0.0 = standard least-squares)."""
    lasso_alphas: Union[list, type(None)] = None
    """Extra LASSO strengths to fit and report alongside lasso_alpha, off the same
    encoding pass, as a distillation-quality vs. support-size table. Reporting
    only: the probe that goes into the sweep is always the one at lasso_alpha."""
    latent_only: bool = False
    """Stop after the latent half of the workflow: distillation (step 1), the
    ground-truth-to-SAE concept matching (step 1.5) and the direction ranking by
    latent-space flips (steps 2-3). Everything past that -- decoding, ambient
    verification, collages, feedback, CFKD -- needs a trained diffusion decoder,
    so this is the mode to run in while the generator is still untrained."""
    use_generator_data_for_matching: bool = True
    """Whether step 1.5 matches ground-truth attributes against SAE concepts on
    the *generator's* dataset. True is right when generator and student share a
    dataset. Set it False when the generator was trained on something else (a
    general-purpose ImageNet autoencoder, say): the attributes to look for are
    the student's, and the generator's dataset does not carry them."""
    overwrite: bool = True
    """Whether to overwrite cached results."""
    feedback_accuracies: list = []
    """Feedback accuracies log."""
    group_accuracies: list = []
    """Group accuracies log."""
    avg_group_accuracies: list = []
    """Average group accuracies log."""
    counterfactual_type: str = "1sided"
    """Counterfactual type."""
    use_true_counterfactuals: bool = False
    """Whether step 9 also finetunes on "true" counterfactuals (CFKD discards them by
    default). With a confounder-only step 9 the "false" verdicts alone can be too few to
    fill a validation batch, which trips CFKD's "too empty" guard and skips finetuning."""
    is_loaded: bool = False
    """Loaded flag."""
    seed: Union[int, type(None)] = 0
    """Seed."""
    in_memory: bool = False
    """In memory dataset."""
    transition_restrictions: Union[list, type(None)] = None
    """Transition restrictions."""
    service_mode: bool = False
    """Service mode (the web demo, peal/web): a public job cannot render every
    latent flip of every direction, so this re-enables the decode budgets
    max_decode_directions / max_decode_per_direction that the research runs
    forbid (see the guard in __init__). Results of a service_mode run are not
    comparable to the paper's CFKD numbers; they are an upper-bound ranking
    ("deepest" selection) meant to surface the strongest shortcut quickly."""
    cfkd_teacher: Union[str, dict, type(None)] = None
    """Teacher used by step 9's CFKD only; None = the same teacher as steps 7-8.
    A direction that a human / LLM judged "false" in step 8 is spurious by that
    verdict, so its counterfactuals need no further per-sample judgement:
    "Baseline:false" accepts every swapped counterfactual as a false one and
    lets step 9 run unattended. Without it a collage-based teacher (web, human,
    llm) would be asked again for every CFKD counterfactual."""
    export_onnx: bool = False
    """Also write the final student as ONNX (model.onnx next to model.cpl).
    Implied when the student was loaded from an .onnx file."""


def _unwrap_dataloader(dataloader):
    """Return the plain DataLoader underneath a (possibly nested) DataloaderMixer."""
    seen = set()
    while isinstance(dataloader, DataloaderMixer):
        if id(dataloader) in seen or not dataloader.dataloaders:
            break
        seen.add(id(dataloader))
        dataloader = dataloader.dataloaders[0]
    return dataloader


class ProbeModel(nn.Module):
    """Linear probe on the generator's ``z_sem`` posing as a binary classifier.

    Wraps the distilled probe ``(w, bias)`` of step 1 so it can be evaluated
    with ``calculate_test_accuracy`` like the student: an image in classifier
    normalisation is converted to generator normalisation, encoded to
    ``z_sem`` and scored as ``s = z @ w + bias``; the output logits are
    ``[-s, s]`` so ``argmax`` gives class 1 when ``s > 0``.

    Parameters
    ----------
    generator : InvertibleGenerator
        Generator with ``encode(x, only_semantic=True)``.
    w : torch.Tensor
        Probe weight of shape ``(encoder_dim,)``.
    bias : float
        Probe bias.
    classifier_to_generator : callable
        Maps classifier-normalised images to generator normalisation.
    device : str or torch.device
        Device the encoding runs on.
    """

    def __init__(self, generator, w, bias, classifier_to_generator, device):
        """Store the probe and the encoder it reads from.

        See the class docstring for the meaning of the arguments.
        """
        super().__init__()
        self.generator = generator
        self.w = w.to(device)
        self.bias = float(bias)
        self.classifier_to_generator = classifier_to_generator
        self.device = device

    def forward(self, x):
        """Return ``(B, 2)`` logits ``[-s, s]`` for a batch of classifier inputs."""
        x_gen = self.classifier_to_generator(x)
        with torch.no_grad():
            z = self.generator.encode(x_gen.to(self.device), only_semantic=True)
            w = self.w.to(device=z.device, dtype=z.dtype)
            s = torch.matmul(z, w) + self.bias
            logits = torch.stack([-s, s], dim=-1)
            return logits


class DiDAE(Adaptor):
    """DiDAE adaptor: discover shortcut directions in a diffusion autoencoder's
    latent with a sparse dictionary, have a teacher judge them, and repair the
    student with CFKD.

    The constructor loads the student, the train/validation/test dataloaders,
    the generator (whose ``sparse_dictionary`` is replaced by
    ``adaptor_config.sparse_dictionary`` when given) and the teacher, and
    writes ``config.yaml`` into ``base_dir``. It refuses a dictionary whose
    ``fitted_on_encoder`` tag does not match the generator's feature space,
    and refuses decode budgets outside ``service_mode``. The workflow itself
    runs in :meth:`run`.

    Parameters
    ----------
    adaptor_config : dict, str, Path or DiDAEConfig
        Loaded with ``load_yaml_config`` into a :class:`DiDAEConfig`.
    **kwargs
        Ignored.

    Attributes
    ----------
    student : torch.nn.Module
        The classifier being explained; replaced by the finetuned one after
        step 9.
    generator : InvertibleGenerator
        Provides ``encode``, ``sweep_all_directions`` and
        ``compute_bipartite_gt_sae_matching``.
    teacher : TeacherInterface
        Judges directions in step 8.
    classifier_to_generator, generator_to_classifier : callable
        Normalisation converters between the two datasets' conventions.
    sweep_found_nothing : bool
        Set by :meth:`run` when the sweep produced no direction.
    """

    def __init__(
        self,
        adaptor_config: Union[dict, str, Path, AdaptorConfig] = None,
        **kwargs,
    ):
        """Load student, data, generator, dictionary and teacher from the config."""
        self.adaptor_config = load_yaml_config(adaptor_config, DiDAEConfig)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # --- Load student predictor ---
        if self.adaptor_config.student is not None:
            self.student, _ = get_predictor(
                self.adaptor_config.student, device=self.device
            )
            if isinstance(self.student, torch.nn.Module):
                self.student.eval()

        # --- Setup base dir ---
        if not self.adaptor_config.service_mode and (
            self.adaptor_config.max_decode_per_direction is not None
            or (
                self.adaptor_config.max_decode_directions is not None
                and not self.adaptor_config.concept_replacement
            )
        ):
            raise ValueError(
                "DiDAE decode budgets are disabled: every latent flip inside the "
                "empirical bounds must be rendered and checked for an ambient flip "
                "(otherwise the sweep is not comparable to CFKD). Remove "
                "max_decode_directions / max_decode_per_direction from the config. "
                "(Only concept_replacement mode may cap the number of PAIRS decoded, "
                "since 256 x 256 candidate pairs x 1000 samples cannot be rendered; "
                "every latent flip of each kept pair is still rendered.)"
            )
        self.base_dir = self.adaptor_config.base_dir
        Path(self.base_dir).mkdir(parents=True, exist_ok=True)

        # --- Setup data ---
        if self.adaptor_config.test_data is None:
            self.adaptor_config.test_data = self.adaptor_config.data

        self.adaptor_config.data.in_memory = self.adaptor_config.in_memory
        self.adaptor_config.test_data.in_memory = self.adaptor_config.in_memory

        self.adaptor_config.training.val_batch_size = self.adaptor_config.batch_size

        (
            self.train_dataloader,
            self.val_dataloader,
            self.test_dataloader,
        ) = create_dataloaders_from_datasource(
            datasource=None,
            config=self.adaptor_config,
            test_config=self.adaptor_config.test_data,
        )
        self.joint_validation_dataloader = WeightedDataloaderList([self.val_dataloader])
        self.adaptor_config.data = self.train_dataloader.dataset.config

        # --- Load generator ---
        self.generator = get_generator(
            generator=self.adaptor_config.generator,
            device=self.device,
            predictor_dataset=self.val_dataloader.dataset,
        )

        # --- Load sparse dictionary ---
        # An explicit adaptor_config.sparse_dictionary always wins over whatever the
        # generator loaded from its own config: component_indices are indices into
        # *this* dictionary, so silently keeping the generator's (differently sized)
        # one produces out-of-range / mismatched directions.
        if self.adaptor_config.sparse_dictionary is not None:
            self.generator.sparse_dictionary = get_sparse_dictionary(
                self.adaptor_config.sparse_dictionary
            )

        # DiDAE edits z_sem and decodes it, so the dictionary has to be a
        # dictionary *of z_sem*. A dictionary fitted on some other 768-d encoder
        # (an OpenCLIP one, say) passes every shape check and yields directions
        # that mean nothing here, so check the tag the fit recorded.
        # fit_sparse_dictionary() records generator.feature_extractor_tag(), which
        # is "diffusion_autoencoder_semantic" for a jointly trained encoder and the
        # encoder name (e.g. "clip:ViT-L/14") when the decoder was trained against
        # a configured frozen encoder -- z_sem IS that encoder's space then, so the
        # tag to compare against is the generator's own, not the literal string.
        sd_now = getattr(self.generator, "sparse_dictionary", None)
        fitted_on = getattr(getattr(sd_now, "config", None), "fitted_on_encoder", None)
        tag_fn = getattr(self.generator, "feature_extractor_tag", None)
        generator_tag = (
            tag_fn() if callable(tag_fn) else "diffusion_autoencoder_semantic"
        )
        if fitted_on is not None and fitted_on != generator_tag:
            raise ValueError(
                f"The sparse dictionary was fitted on '{fitted_on}' activations, but "
                f"this generator's z_sem lives in '{generator_tag}'. DiDAE edits z_sem "
                "and decodes it, so its directions have to live in that space. Both "
                "may be 768-d, which is why nothing else catches this. Fit the "
                "dictionary with the generator config whose encoder the decoder was "
                "trained against, or clear fitted_on_encoder if you are certain the "
                "spaces match."
            )

        # --- Setup teacher ---
        # Same mapping CFKD uses, so the "1sided" acceptance test below agrees
        # with the one that decides which samples CFKD builds counterfactuals from.
        self.logits_to_prediction = lambda logits: logits.argmax(-1)
        self.output_size = (
            self.adaptor_config.task.output_channels
            if self.adaptor_config.task.output_channels is not None
            else self.adaptor_config.data.output_size[0]
        )
        # Model-based teachers score how far a counterfactual drifted out of
        # distribution *relative* to real data, so the validation dataset needs
        # its reference score before the teacher is built - same handshake CFKD
        # does in its own constructor.
        outlier_scores_absolute = self.val_dataloader.dataset.calculate_outlier_score(
            next(iter(self.train_dataloader))[0]
        )
        self.val_dataloader.dataset.reference_outlier_scores = torch.mean(
            outlier_scores_absolute["absolute"]
        ).item()

        self.teacher = get_teacher(
            teacher=self.adaptor_config.teacher,
            output_size=self.output_size,
            adaptor_config=self.adaptor_config,
            dataset=self.val_dataloader.dataset,
            device=self.device,
            tracking_level=self.adaptor_config.tracking_level,
        )

        # --- Normalization helpers ---
        self.classifier_to_generator = (
            lambda x: self.generator.generator_dataset.project_from_pytorch_default(
                self.val_dataloader.dataset.project_to_pytorch_default(x)
            )
        )
        self.generator_to_classifier = (
            lambda x: self.val_dataloader.dataset.project_from_pytorch_default(
                self.generator.generator_dataset.project_to_pytorch_default(x)
            )
        )

        # Save config.yaml at initialization
        save_yaml_config(
            self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
        )

    def _cached_student_margins(self, dataset):
        """Student margins ``logits[:, 1] - logits[:, 0]`` of ``dataset``'s images
        in ``dataset.keys`` order from ``student_logits_cache``, or ``None`` when
        no cache is configured or it does not cover every image."""
        path = getattr(self.adaptor_config, "student_logits_cache", None)
        keys = getattr(dataset, "keys", None)
        if not path or not os.path.isfile(path) or keys is None:
            return None
        if len(keys) != len(dataset):
            return None
        with np.load(path, allow_pickle=False) as z:
            row = {k: i for i, k in enumerate(z["keys"].tolist())}
            logits = z["logits"]
        missing = [k for k in keys if k not in row]
        if missing or logits.ndim != 2 or logits.shape[1] < 2:
            _log.info(
                "%s",
                f"[DiDAE] student_logits_cache does not cover the train split "
                f"({len(missing)} of {len(keys)} images missing); running the student.",
            )
            return None
        idx = np.array([row[k] for k in keys])
        _log.info(
            "%s",
            f"[DiDAE] Step 1 takes the student's margins of {len(keys)} images from {path}.",
        )
        return torch.from_numpy(logits[idx, 1] - logits[idx, 0]).float()

    def _distill_closed_form(self):
        """
        Step 1: Distill user's classifier into reference encoder space
        using closed-form least squares.

        Solves: w*, b* = argmin_{w,b} ||Z w + b - y_pred||^2
        where Z = encoder(x) for all training data,
              y_pred = classifier(x)[:, 1] - classifier(x)[:, 0]
        (signed margin for binary classification)

        Returns:
            w: [encoder_dim] weight vector
            bias: scalar float bias
        """
        probe_dir = os.path.join(self.base_dir, "distilled_probe")
        Path(probe_dir).mkdir(parents=True, exist_ok=True)
        w_path = os.path.join(probe_dir, "w.pt")
        bias_path = os.path.join(probe_dir, "bias.pt")

        if (
            os.path.exists(w_path)
            and os.path.exists(bias_path)
            and not self.adaptor_config.overwrite
        ):
            _log.info(
                "%s", f"[DiDAE] Loading cached distilled probe weights from {w_path}..."
            )
            w = torch.load(w_path, map_location=self.device)
            bias = torch.load(bias_path, map_location=self.device).item()
            return w, bias

        _log.info("%s", "\n[DiDAE] Step 1: Distilling classifier into encoder space...")

        Z_list = []
        y_list = []

        # The adaptor's train dataloader is a DataloaderMixer, whose iterator draws
        # `steps_per_epoch` resampled batches (1000 x 32 here) instead of walking the
        # 800-sample train split once. For a closed-form least-squares fit that is
        # ~40x redundant work on a distribution skewed by the mixer's class balancing,
        # so fit on one clean pass over the underlying dataloader.
        fit_dataloader = _unwrap_dataloader(self.train_dataloader)
        cached_margins = self._cached_student_margins(fit_dataloader.dataset)
        if cached_margins is not None:
            # one unshuffled pass, so batch positions map to dataset.keys
            fit_dataloader = DataLoader(
                fit_dataloader.dataset,
                batch_size=fit_dataloader.batch_size or 32,
                shuffle=False,
                num_workers=int(os.environ.get("PEAL_WEB_NUM_WORKERS", "8")),
            )
        pos = 0

        with torch.no_grad():
            for batch in tqdm(fit_dataloader, desc="Encoding train data"):
                x = batch["x"] if isinstance(batch, dict) else batch[0]
                x_gen = self.classifier_to_generator(x)
                z = self.generator.encode(x_gen.to(self.device), only_semantic=True)

                if cached_margins is not None:
                    y_margin = cached_margins[pos : pos + x.shape[0]]
                    pos += x.shape[0]
                else:
                    logits = self.student(x.to(self.device))
                    if logits.ndim == 2 and logits.shape[1] >= 2:
                        y_margin = logits[:, 1] - logits[:, 0]
                    else:
                        y_margin = logits.squeeze(-1)

                Z_list.append(z.cpu())
                y_list.append(y_margin.cpu())

        Z = torch.cat(Z_list, dim=0).float()  # [N, D]
        y_pred = torch.cat(y_list, dim=0).float()  # [N]

        _log.info(
            "%s",
            f"[DiDAE] Solving least squares: Z [{Z.shape}] @ w + b = y_pred [{y_pred.shape}]",
        )

        Z_dev = Z.to(self.device).float()
        y_dev = y_pred.to(self.device).float()
        Z_mean = Z_dev.mean(dim=0, keepdim=True)
        y_mean = y_dev.mean()

        Z_centered = Z_dev - Z_mean
        y_centered = y_dev - y_mean

        lasso_alpha = getattr(self.adaptor_config, "lasso_alpha", 0.0)
        sd = getattr(self.generator, "sparse_dictionary", None)
        probe_sd_cfg = getattr(self.adaptor_config, "probe_sparse_dictionary", None)
        if probe_sd_cfg is not None:
            sd = get_sparse_dictionary(probe_sd_cfg)
            _log.info(
                "%s",
                "[DiDAE] Probe LASSO basis: probe_sparse_dictionary "
                f"({type(sd).__name__}, {sd.get_components().shape[1]} atoms) instead of "
                "the sweep's dictionary.",
            )

        W_norm = None
        if sd is not None:
            W_dec = sd.get_components().to(self.device).float()
            if W_dec.shape[0] != Z.shape[1] and W_dec.shape[1] == Z.shape[1]:
                W_dec = W_dec.T  # [D, K]

            # Align feature dimension D to match Z.shape[1] (e.g. SpLICE 768 -> 512)
            D_z = Z.shape[1]
            if W_dec.shape[0] != D_z:
                if W_dec.shape[0] < D_z:
                    W_dec = torch.nn.functional.pad(
                        W_dec, (0, 0, 0, D_z - W_dec.shape[0])
                    )
                else:
                    W_dec = W_dec[:D_z, :]

            W_norm = W_dec / (torch.norm(W_dec, dim=0, keepdim=True) + 1e-8)  # [D, K]

        def fit_probe(alpha):
            """Fit the probe at one LASSO strength.

            Returns (w, bias, active_k), where active_k is the size of the
            support in the dictionary's coefficient basis, or None when the fit
            was a plain least-squares one that has no such basis.
            """
            if not (alpha > 0.0 and W_norm is not None):
                # Closed-form least-squares with bias
                Z_b = torch.cat(
                    [Z_dev, torch.ones(Z_dev.shape[0], 1, device=self.device)], dim=1
                )
                sol = torch.linalg.lstsq(Z_b, y_dev.unsqueeze(1)).solution.squeeze(1)
                return sol[:-1], sol[-1].item(), None

            D_z = Z.shape[1]
            effective_alpha = alpha * (512.0 / max(1.0, float(D_z)))
            _log.info(
                "%s",
                f"[DiDAE] Applying SAE-Sparse LASSO regularization (alpha={effective_alpha:.4f}, base_alpha={alpha})...",
            )

            A = torch.matmul(Z_centered, W_norm)  # [N, K]
            N_samples, K_features = A.shape

            # Compute spectral norm L using 5 iterations of power iteration.
            # Seeded: the step size 1/L feeds straight into the soft-threshold,
            # so an unseeded start makes the same alpha return a different
            # support on every fit (834 vs 848 of 32000 directions, measured),
            # which the sweep table above would report as a real difference.
            with torch.no_grad():
                generator = torch.Generator(device=self.device).manual_seed(0)
                v_pow = torch.randn(
                    K_features, 1, device=self.device, generator=generator
                )
                v_pow = v_pow / (torch.norm(v_pow) + 1e-8)
                for _ in range(5):
                    Av = torch.matmul(A, v_pow)
                    v_pow = torch.matmul(A.T, Av)
                    v_pow_norm = torch.norm(v_pow)
                    if v_pow_norm > 1e-8:
                        v_pow = v_pow / v_pow_norm
                L = (v_pow_norm.item() / max(1, N_samples)) + 1e-6

            eta = 1.0 / L
            beta = torch.zeros(K_features, device=self.device)

            def soft_threshold(v, lmbda):
                """ISTA soft-thresholding: shrink ``v`` towards 0 by ``lmbda``."""
                return torch.sign(v) * torch.relu(torch.abs(v) - lmbda)

            for step in range(2500):
                pred = torch.matmul(A, beta)
                residual_vec = pred - y_centered
                grad = torch.matmul(A.T, residual_vec) / N_samples
                beta = soft_threshold(beta - eta * grad, eta * effective_alpha)

            w_fit = torch.matmul(W_norm, beta)  # [D]
            bias_fit = (y_mean - torch.matmul(Z_mean, w_fit).squeeze()).item()
            active_k = (torch.abs(beta) > 1e-4).sum().item()
            _log.info(
                "%s",
                f"[DiDAE] SAE-LASSO distilled probe focused onto {active_k}/{K_features} active SAE directions.",
            )
            return w_fit, bias_fit, active_k

        def fit_quality(w_fit, bias_fit):
            """Score a fitted probe against the student on the held-out split.

            Returns the ``(mse, corr)`` pair between the probe's prediction
            ``Z_dev @ w_fit + bias_fit`` and the student's logit margin ``y_pred``.
            """
            y_hat = (Z_dev @ w_fit + bias_fit).cpu()
            mse = torch.mean((y_hat - y_pred) ** 2).item()
            corr = torch.corrcoef(torch.stack([y_hat, y_pred]))[0, 1].item()
            return mse, corr

        # Optional sweep: fit at several strengths off this one encoding pass and
        # report where the support collapses, so lasso_alpha can be picked on
        # evidence instead of guessed. The sweep only reports -- the probe that
        # goes downstream is always the configured lasso_alpha.
        lasso_alphas = getattr(self.adaptor_config, "lasso_alphas", None)
        if lasso_alphas:
            K_total = W_norm.shape[1] if W_norm is not None else Z.shape[1]
            rows = []
            for alpha in lasso_alphas:
                w_a, bias_a, active_a = fit_probe(float(alpha))
                mse_a, corr_a = fit_quality(w_a, bias_a)
                rows.append((float(alpha), mse_a, corr_a, active_a))
                del w_a
            _log.info(
                "%s", "\n[DiDAE] LASSO sweep (distillation quality vs. support size):"
            )
            _log.info(
                "%s",
                f"  {'alpha':>10}  {'MSE':>12}  {'corr':>8}  {'active':>8} / {K_total}",
            )
            for alpha, mse_a, corr_a, active_a in rows:
                active_str = (
                    str(active_a) if active_a is not None else f"{K_total} (dense)"
                )
                _log.info(
                    "%s",
                    f"  {alpha:>10.5g}  {mse_a:>12.6f}  {corr_a:>8.4f}  {active_str:>8}",
                )
            sweep_path = os.path.join(probe_dir, "lasso_sweep.txt")
            with open(sweep_path, "w") as handle:
                handle.write(
                    "alpha\tmse\tcorrelation\tactive_directions\tn_directions\n"
                )
                for alpha, mse_a, corr_a, active_a in rows:
                    handle.write(
                        f"{alpha}\t{mse_a}\t{corr_a}\t"
                        f"{active_a if active_a is not None else K_total}\t{K_total}\n"
                    )
            _log.info("%s", f"[DiDAE] Wrote LASSO sweep to {sweep_path}")
            _log.info(
                "%s",
                f"[DiDAE] Continuing with the configured lasso_alpha={lasso_alpha}.\n",
            )

        w, bias, _ = fit_probe(lasso_alpha)

        # Compute and log fit quality
        residual, correlation = fit_quality(w, bias)
        _log.info(
            "%s",
            f"[DiDAE] Distillation MSE: {residual:.6f}, correlation: {correlation:.4f}",
        )

        torch.save(w, w_path)
        torch.save(torch.tensor(bias), bias_path)
        _log.info(
            "%s",
            f"[DiDAE] Saved distilled weights and bias (bias={bias:.4f}) to {probe_dir}",
        )

        # Evaluate and log group accuracies of original model vs distilled probe on test set
        self._evaluate_and_log_probe_quality(w, bias, residual, correlation)

        return w, bias

    def _evaluate_and_log_probe_quality(
        self, w, bias=0.0, residual=None, correlation=None
    ):
        """
        Evaluates and logs group accuracies of both the original student model
        and the distilled linear probe on the test set to TensorBoard.
        """
        writer = SummaryWriter(os.path.join(self.base_dir, "logs"))

        if residual is not None and correlation is not None:
            writer.add_scalar("distilled_probe/train_mse", residual, 0)
            writer.add_scalar("distilled_probe/train_correlation", correlation, 0)

        # 1. Evaluate Original Student Model
        _log.info(
            "%s",
            "[DiDAE] Evaluating Original Student Model group accuracies on test dataset...",
        )
        (
            orig_acc,
            orig_group_accs,
            orig_group_dist,
            orig_group_nums,
            orig_worst_acc,
        ) = calculate_test_accuracy(
            model=self.student,
            test_dataloader=self.test_dataloader,
            device=self.device,
            calculate_group_accuracies=True,
            tracking_level=self.adaptor_config.tracking_level,
        )

        writer.add_scalar("original_model/test_overall_accuracy", orig_acc, 0)
        writer.add_scalar("original_model/test_worst_group_accuracy", orig_worst_acc, 0)
        if orig_group_accs is not None:
            for g_idx, g_acc in enumerate(orig_group_accs):
                writer.add_scalar(
                    f"original_model/test_group_{g_idx}_accuracy", g_acc, 0
                )

        # 2. Evaluate Distilled Linear Probe Model
        _log.info(
            "%s",
            "[DiDAE] Evaluating Distilled Linear Probe group accuracies on test dataset...",
        )
        probe_model = ProbeModel(
            generator=self.generator,
            w=w,
            bias=bias,
            classifier_to_generator=self.classifier_to_generator,
            device=self.device,
        )

        (
            probe_acc,
            probe_group_accs,
            probe_group_dist,
            probe_group_nums,
            probe_worst_acc,
        ) = calculate_test_accuracy(
            model=probe_model,
            test_dataloader=self.test_dataloader,
            device=self.device,
            calculate_group_accuracies=True,
            tracking_level=self.adaptor_config.tracking_level,
        )

        writer.add_scalar("distilled_probe/test_overall_accuracy", probe_acc, 0)
        writer.add_scalar(
            "distilled_probe/test_worst_group_accuracy", probe_worst_acc, 0
        )
        if probe_group_accs is not None:
            for g_idx, g_acc in enumerate(probe_group_accs):
                writer.add_scalar(
                    f"distilled_probe/test_group_{g_idx}_accuracy", g_acc, 0
                )

        # 3. Compute Fidelity (% agreement between Original Student and Distilled Probe)
        n_matches = 0
        n_total = 0
        with torch.no_grad():
            for sample in self.test_dataloader:
                x = sample["x"] if isinstance(sample, dict) else sample[0]
                y_stud = self.student(x.to(self.device)).argmax(-1)
                y_probe = probe_model(x.to(self.device)).argmax(-1)
                n_matches += (y_stud == y_probe).sum().item()
                n_total += x.shape[0]

        fidelity = n_matches / max(1, n_total)
        writer.add_scalar("distilled_probe/student_probe_fidelity", fidelity, 0)

        _log.info("%s", "\n" + "=" * 80)
        _log.info(
            "%s",
            "[DiDAE] Distilled Probe vs. Original Model Test Set Group Accuracies:",
        )
        _log.info("%s", "=" * 80)
        if residual is not None and correlation is not None:
            _log.info(
                "%s",
                f"  • Distilled Probe Fit: MSE = {residual:.6f}, Pearson r = {correlation:.4f}",
            )
        _log.info(
            "%s", f"  • Student-Probe Fidelity: {fidelity * 100:.2f}% test agreement"
        )
        _log.info("%s", "-" * 80)
        _log.info(
            "%s",
            f"  [Original Model] Accuracy: Overall = {orig_acc:.4f} | Worst-Group = {orig_worst_acc:.4f}",
        )
        if orig_group_accs is not None:
            for g_idx, g_acc in enumerate(orig_group_accs):
                g_num = (
                    int(orig_group_nums[g_idx]) if orig_group_nums is not None else 0
                )
                _log.info("%s", f"    - Group {g_idx} Acc: {g_acc:.4f} (N={g_num})")
        _log.info("%s", "-" * 80)
        _log.info(
            "%s",
            f"  [Distilled Probe] Accuracy: Overall = {probe_acc:.4f} | Worst-Group = {probe_worst_acc:.4f}",
        )
        if probe_group_accs is not None:
            for g_idx, g_acc in enumerate(probe_group_accs):
                g_num = (
                    int(probe_group_nums[g_idx]) if probe_group_nums is not None else 0
                )
                _log.info("%s", f"    - Group {g_idx} Acc: {g_acc:.4f} (N={g_num})")
        _log.info("%s", "=" * 80 + "\n")

        writer.close()

    def _ensure_component_bounds(self):
        """
        Computes and saves component min/max values (c_min_and_maxes.txt)
        for the active sparse dictionary if not already written to disk.
        """
        sd = getattr(self.generator, "sparse_dictionary", None)
        if sd is None:
            return None

        sd_config = getattr(sd, "config", None)
        sd_base_path = getattr(sd_config, "base_path", None) if sd_config else None

        if not sd_base_path:
            sd_name = (
                getattr(sd_config, "name", None)
                or getattr(sd_config, "sparse_dictionaries_type", None)
                or "sparse_dict"
            )
            sd_base_path = os.path.join(self.base_dir, sd_name)
            if sd_config:
                sd_config.base_path = sd_base_path

        Path(sd_base_path).mkdir(parents=True, exist_ok=True)
        c_min_max_path = os.path.join(sd_base_path, "c_min_and_maxes.txt")

        if os.path.exists(c_min_max_path):
            c_mins_check = []
            with open(c_min_max_path, "r") as file:
                for line in file:
                    parts = line.strip().split("min=")
                    if len(parts) > 1:
                        c_mins_check.append(float(parts[1].split(",")[0]))
            # If all checked mins are >= 0.0, it was generated with non-negative sd.encode solver instead of raw z_sem @ W projections
            if len(c_mins_check) > 0 and not all(m >= 0.0 for m in c_mins_check[:20]):
                _log.info(
                    "%s", f"[DiDAE] Component bounds already exist at {c_min_max_path}"
                )
                return c_min_max_path
            _log.info(
                "%s",
                f"[DiDAE] Existing bounds at {c_min_max_path} contain non-negative (>=0.0) mins. Recomputing raw projection c_min/c_max...",
            )

        _log.info(
            "%s",
            "[DiDAE] Computing component empirical bounds (c_min / c_max) across unpoisoned dataset...",
        )
        W_all = sd.get_components()  # [Dim_dict, K]
        c_mins = None
        c_maxs = None

        train_ds = self.train_dataloader.dataset
        saved_task_config = getattr(train_ds, "task_config", None)
        train_ds.task_config = None

        try:
            with torch.no_grad():
                for idx, batch in enumerate(
                    tqdm(self.train_dataloader, desc="Computing concept bounds")
                ):
                    if idx >= 30:
                        break
                    x = batch[0] if isinstance(batch, (list, tuple)) else batch["x"]
                    x_gen = self.classifier_to_generator(x)
                    z_sem = self.generator.encode(
                        x_gen.to(self.device), only_semantic=True
                    )

                    w_aligned = W_all.to(device=self.device, dtype=z_sem.dtype)
                    if w_aligned.shape[0] != z_sem.shape[1]:
                        if w_aligned.shape[0] < z_sem.shape[1]:
                            w_aligned = torch.nn.functional.pad(
                                w_aligned,
                                (0, 0, 0, z_sem.shape[1] - w_aligned.shape[0]),
                            )
                        else:
                            w_aligned = w_aligned[: z_sem.shape[1], :]

                    c = (
                        z_sem @ w_aligned
                    )  # Compute raw linear projections matching c_factual in sweep_all_directions

                    batch_min = c.min(dim=0).values
                    batch_max = c.max(dim=0).values
                    if c_mins is None:
                        c_mins = batch_min
                        c_maxs = batch_max
                    else:
                        c_mins = torch.minimum(c_mins, batch_min)
                        c_maxs = torch.maximum(c_maxs, batch_max)
        finally:
            train_ds.task_config = saved_task_config

        with open(c_min_max_path, "w") as f:
            for i in range(c_mins.shape[0]):
                f.write(
                    f"Component {i}: min={c_mins[i].item():.4f}, max={c_maxs[i].item():.4f}\n"
                )

        _log.info("%s", f"[DiDAE] Saved component bounds to {c_min_max_path}")
        return c_min_max_path

    @staticmethod
    def _matching_attribute_selection(dataset):
        """Which ground-truth attributes step 1.5 should match SAE concepts against.

        Clearing task_config makes the dataset fall back to
        `targets[: output_size[0]]` -- the *first* few columns of its attribute
        vector, in file order. On Waterbirds that vector is
        ['y', 'split', 'place', 'place_filename'] and output_size is 2, so the
        matching was handed [y, split]: `split` is constant within a split, so it
        matched nothing, and `place` -- the confounder the whole experiment is
        about -- was never looked at.

        The dataset already says which columns are the interesting ones, in
        config.confounding_factors, and already knows how to select columns by
        name (the y_selection branch of __getitem__). Use that instead of the
        positional fallback. A namespace rather than a TaskConfig because
        __getitem__ also branches on task_config.criterions, and this selection
        must not go through the singleclass path that collapses the vector.
        """
        attributes = getattr(dataset, "attributes", None)
        factors = getattr(getattr(dataset, "config", None), "confounding_factors", None)
        if not attributes or not factors:
            return None
        selection = [f for f in factors if f in attributes]
        return selection or None

    def _collect_sweep_samples(
        self,
        target_n=None,
        bypass_task_config=True,
        use_all_dataloaders=False,
        dataloaders=None,
    ):
        """Collect N samples from specified dataloaders (defaults to validation or train+val)."""
        if dataloaders is None:
            dataloaders = (
                [self.train_dataloader, self.val_dataloader]
                if use_all_dataloaders
                else [self.val_dataloader]
            )
        datasets = [dl.dataset for dl in dataloaders if hasattr(dl, "dataset")]

        saved_task_configs = [getattr(ds, "task_config", None) for ds in datasets]
        selection_names = None
        if bypass_task_config:
            for ds in datasets:
                selection = self._matching_attribute_selection(ds)
                if selection is None:
                    ds.task_config = None
                    continue
                if selection_names is None:
                    selection_names = selection
                ds.task_config = SimpleNamespace(y_selection=selection)
            if selection_names is not None:
                _log.info(
                    "%s",
                    f"[DiDAE] Matching against ground-truth attributes {selection_names}.",
                )

        samples_x = []
        samples_y = []
        n_collected = 0
        target = target_n  # None means collect everything

        try:
            for dl in dataloaders:
                for batch in dl:
                    if isinstance(batch, (list, tuple)):
                        x = batch[0]
                        y = batch[1]
                    elif isinstance(batch, dict):
                        x = batch["x"]
                        y = batch["y"]
                    else:
                        raise ValueError(f"Unknown batch format: {type(batch)}")

                    if isinstance(y, (list, tuple)):
                        y = y[0]

                    batch_size = x.shape[0]
                    if target is not None:
                        take = min(batch_size, target - n_collected)
                    else:
                        take = batch_size
                    samples_x.append(x[:take])
                    samples_y.append(
                        y[:take]
                        if isinstance(y, torch.Tensor)
                        else torch.tensor(y[:take])
                    )
                    n_collected += take
                    if target is not None and n_collected >= target:
                        break
                if target is not None and n_collected >= target:
                    break
        finally:
            if bypass_task_config:
                for ds, saved_cfg in zip(datasets, saved_task_configs):
                    ds.task_config = saved_cfg

        x_all = torch.cat(samples_x, dim=0)
        y_all = torch.cat(samples_y, dim=0)
        attr_names = selection_names
        if attr_names is None or (
            y_all.ndim == 2 and len(attr_names) != y_all.shape[1]
        ):
            attr_names = self._resolve_attr_names(y_all)
        return x_all, y_all, attr_names

    def _resolve_attr_names(self, y_all):
        """Find the dataset attribute names matching the collected label width."""

        def extract_all_datasets(obj):
            """Recursively unwrap loaders, subsets and concats into leaf datasets."""
            if obj is None:
                return []
            if hasattr(obj, "attributes") and obj.attributes:
                return [obj]
            if hasattr(obj, "dataset"):
                return extract_all_datasets(obj.dataset)
            if hasattr(obj, "datasets"):
                res = []
                for item in obj.datasets:
                    res.extend(extract_all_datasets(item))
                return res
            if hasattr(obj, "dataloaders"):
                res = []
                for item in obj.dataloaders:
                    res.extend(extract_all_datasets(item))
                return res
            if isinstance(obj, (list, tuple)):
                res = []
                for item in obj:
                    res.extend(extract_all_datasets(item))
                return res
            return [obj]

        target_len = y_all.shape[1] if y_all.ndim == 2 else None
        all_ds_candidates = extract_all_datasets(
            self.val_dataloader
        ) + extract_all_datasets(self.train_dataloader)
        full_attr_names = None
        for candidate in all_ds_candidates:
            if hasattr(candidate, "attributes") and candidate.attributes:
                val = list(candidate.attributes)
                if target_len is None or len(val) == target_len:
                    full_attr_names = val
                    break

        if full_attr_names is None and target_len is not None:
            full_attr_names = [f"Attr_{i}" for i in range(target_len)]

        return full_attr_names

    def _collect_sweep_samples_balanced(self, target_n):
        """
        Draw the sweep's factual samples the way CFKD seeds its counterfactuals.

        CFKD pops class-balanced samples off a DataStack laid over its training
        dataloader and, under counterfactual_type "1sided", keeps only samples the
        student already classifies correctly (`CFKD.get_batch`). DiDAE's discovery
        sweep instead walked the validation loader sequentially, so the two halves
        of the same experiment ran on different data drawn in different ways -- and
        the pool depended on the budget, since the old call site switched from
        val-only to train+val the moment n_samples exceeded the validation split.

        Here the pool is train + validation weighted by split size, so discovery
        sees all min_train_samples + max_validation_samples rows.
        """
        mixer = DataloaderMixer(self.adaptor_config.training, self.train_dataloader)
        # weight_added_dataloader left at None: priorities come out proportional
        # to the two split sizes, so the pool is the union rather than a 50/50 mix.
        mixer.append(self.val_dataloader)
        datastack = DataStack(
            mixer,
            self.output_size,
            transform=self.val_dataloader.dataset.transform,
        )

        one_sided = self.adaptor_config.counterfactual_type == "1sided"
        samples_x, samples_y = [], []
        cm_idx = 0
        rejected = 0
        # A "1sided" pool can be thin: a badly confounded student classifies
        # almost exclusively one class correctly, so stop rather than spin forever
        # once rejections dwarf the budget.
        max_rejections = max(50 * target_n, 5000)

        while len(samples_x) < target_n:
            # Cycle source/target class pairs exactly as CFKD.get_batch does, so
            # the sweep pool has CFKD's class composition rather than the
            # validation split's natural (poisoned) one.
            y_source = int(cm_idx / self.output_size)
            y_target = int(cm_idx % self.output_size)
            while y_source == y_target:
                cm_idx = (cm_idx + 1) % (self.output_size**2)
                y_source = int(cm_idx / self.output_size)
                y_target = int(cm_idx % self.output_size)
            cm_idx = (cm_idx + 1) % (self.output_size**2)

            x, y = datastack.pop(int(y_source))
            y_label = y[0] if isinstance(y, (list, tuple)) else y

            if one_sided:
                with torch.no_grad():
                    logits = (
                        self.student(x.to(self.device).unsqueeze(0))
                        .squeeze(0)
                        .detach()
                        .cpu()
                    )
                prediction = self.logits_to_prediction(logits)
                if not (int(prediction) == int(y_label) == int(y_source)):
                    rejected += 1
                    if rejected > max_rejections:
                        _log.info(
                            "%s",
                            f"[DiDAE] counterfactual_type='1sided' rejected {rejected} "
                            f"samples before the budget was met; sweeping the "
                            f"{len(samples_x)} collected so far.",
                        )
                        break
                    continue

            samples_x.append(x)
            samples_y.append(
                y_label if isinstance(y_label, torch.Tensor) else torch.tensor(y_label)
            )

        x_all = torch.stack(samples_x, dim=0)
        y_all = torch.stack(samples_y, dim=0)
        class_counts = [int((y_all == c).sum()) for c in range(self.output_size)]
        _log.info(
            "%s",
            f"[DiDAE] Sweep pool: {x_all.shape[0]} samples drawn CFKD-style "
            f"(class-balanced over train+val, counterfactual_type="
            f"'{self.adaptor_config.counterfactual_type}', {rejected} rejected). "
            f"Class counts: {class_counts}",
        )
        return x_all, y_all, self._resolve_attr_names(y_all)

    def _generate_direction_collages(self, direction_results, output_dir):
        """
        Generate collage images for each direction showing factual/counterfactual pairs.
        Returns list of collage paths grouped by direction.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        collage_paths_by_direction = []

        for rank, result in enumerate(direction_results):
            d_idx = result["direction_idx"]
            pairs = result["pairs"]
            if len(pairs) == 0:
                continue

            direction_collage_paths = []
            # The sweep returns factuals and counterfactuals in generator
            # normalization, everything below (the dataset collage, its oracle
            # and project_to_pytorch_default) expects predictor normalization.
            x_fac_list = [
                self.generator_to_classifier(p["x_factual"].float()) for p in pairs
            ]
            x_cf_list = [
                self.generator_to_classifier(p["x_counterfactual"].float())
                for p in pairs
            ]
            y_target_list = [p.get("target_class", p.get("cf_class", 0)) for p in pairs]
            y_source_list = [p.get("orig_class", 0) for p in pairs]
            y_list = y_source_list
            y_start_conf_list = [
                p.get("orig_conf", max(0.0, 1.0 - p.get("confidence", 0.5)))
                for p in pairs
            ]
            y_end_conf_list = [p.get("confidence", 0.5) for p in pairs]

            val_dataset = self.val_dataloader.dataset
            dim_name = result.get("dimension_name", f"dir{d_idx}")
            clean_dim = (
                dim_name.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("#", "")
                .replace("/", "_")
            )
            from peal.generators.diffusion_autoencoder import _safe_name

            dir_folder_name = _safe_name(f"rank{rank:03d}_dir{d_idx}_{clean_dim}")
            dir_base_path = os.path.join(output_dir, dir_folder_name)

            if hasattr(val_dataset, "generate_contrastive_collage"):
                try:
                    val_dataset.generate_contrastive_collage(
                        x_list=x_fac_list,
                        x_counterfactual_list=x_cf_list,
                        y_target_list=y_target_list,
                        y_source_list=y_source_list,
                        y_list=y_list,
                        y_target_start_confidence_list=y_start_conf_list,
                        y_target_end_confidence_list=y_end_conf_list,
                        base_path=dir_base_path,
                        start_idx=0,
                    )
                except Exception as e:
                    _log.info(
                        "%s",
                        f"[DiDAE] Warning: dataset generate_contrastive_collage failed: {e}",
                    )

            for pair_idx, pair in enumerate(pairs):
                # Project both to [0,1] range for visualization
                x_fac_vis = val_dataset.project_to_pytorch_default(
                    x_fac_list[pair_idx].unsqueeze(0)
                )
                x_cf_vis = val_dataset.project_to_pytorch_default(
                    x_cf_list[pair_idx].unsqueeze(0)
                )

                # Create side-by-side grid
                grid = torch.cat([x_fac_vis, x_cf_vis], dim=0)
                collage_path = os.path.join(
                    output_dir,
                    f"rank{rank:03d}_dir{d_idx}_pair{pair_idx}_conf{pair['confidence']:.2f}.png",
                )
                torchvision.utils.save_image(grid, collage_path, nrow=2)
                direction_collage_paths.append(collage_path)

            collage_paths_by_direction.append(
                {
                    "direction_idx": d_idx,
                    "rank": rank,
                    "success_count": result["success_count"],
                    "collage_paths": direction_collage_paths,
                }
            )

        return collage_paths_by_direction

    @staticmethod
    def _pair_gt_label(y_samples, pair, fallback):
        """Ground-truth label of the factual sample behind a sweep pair."""
        if y_samples is None or pair is None or "sample_idx" not in pair:
            return fallback

        idx = int(pair["sample_idx"])
        if idx >= len(y_samples):
            return fallback

        y = torch.as_tensor(y_samples[idx]).flatten()
        if y.numel() == 0:
            return fallback

        return int(y.argmax()) if y.numel() > 1 else int(round(float(y[0])))

    def _get_user_feedback(
        self, collage_paths_by_direction, direction_results=None, y_samples=None
    ):
        """
        Step 7-8: Label the top-K directions as "true" (the edit changed the
        class-defining feature) or "false" (it changed a spurious one).

        Two teacher families are supported:
          * collage-based teachers (cluster / human / LLM) judge the rendered
            factual-counterfactual collages;
          * model-based teachers (Model2ModelTeacher, i.e. an oracle .cpl) judge
            the factual/counterfactual *tensors* directly - a direction is
            spurious exactly when the student flipped but the oracle did not.

        Both get the real per-pair labels and confidences from the sweep instead
        of placeholders, and the per-pair verdicts are then majority-aggregated
        per direction. Returns a list of dicts with direction_idx / feedback /
        success_count.
        """
        from peal.teachers.model2model_teacher import Model2ModelTeacher

        pairs_by_direction = {
            r["direction_idx"]: r.get("pairs", []) for r in (direction_results or [])
        }

        # Flatten collages and the pair metadata that belongs to each of them, so
        # the per-collage feedback can be mapped back onto its direction.
        all_paths = []
        all_pairs = []
        direction_slices = []
        for entry in collage_paths_by_direction:
            pairs = pairs_by_direction.get(entry["direction_idx"], [])
            paths = entry["collage_paths"]
            start = len(all_paths)
            for pair_idx, path in enumerate(paths):
                all_paths.append(path)
                all_pairs.append(pairs[pair_idx] if pair_idx < len(pairs) else None)
            direction_slices.append((start, len(all_paths)))
        direction_slices_collage = list(direction_slices)

        from peal.teachers.model2model_teacher import Model2ModelTeacher as _M2M

        if len(all_paths) == 0 and not isinstance(self.teacher, _M2M):
            return []

        y_source_list = [
            int(p["orig_class"]) if p is not None else 0 for p in all_pairs
        ]
        y_target_list = [
            int(p["target_class"]) if p is not None else 0 for p in all_pairs
        ]
        y_end_conf_list = [
            float(p["confidence"]) if p is not None else 0.5 for p in all_pairs
        ]
        y_start_conf_list = [
            (
                float(p.get("orig_conf", 1.0 - float(p["confidence"])))
                if p is not None
                else 0.5
            )
            for p in all_pairs
        ]
        y_list = [
            self._pair_gt_label(y_samples, p, y_source_list[i])
            for i, p in enumerate(all_pairs)
        ]

        if isinstance(self.teacher, Model2ModelTeacher):
            # The oracle judges the images themselves; hand it the real pairs in
            # predictor normalization. mode is neither "train" nor "validation"
            # so the teacher skips its own contrastive-collage dump.
            #
            # It judges EVERY verified flip of the direction, not the top-5
            # collage pairs (2026-09-14): the collage pairs are the highest-
            # confidence flips, i.e. the most extreme edits, and on CelebA those
            # are exactly the Male edits that also changed the hair, so every
            # Male-like direction was voted "true" 5/5 while the classical CFKD
            # teacher calls 70% of all Male flips "false".
            #
            # One direction at a time: materialising the float tensors of all
            # 16k verified flips of 40 directions at once pushed the process
            # over the 16 GiB cgroup (2026-09-14 22:34).
            all_by_direction = {
                r["direction_idx"]: r.get("all_pairs", r.get("pairs", []))
                for r in (direction_results or [])
            }
            use_all = any(
                len(all_by_direction.get(e["direction_idx"], [])) > 0
                for e in collage_paths_by_direction
            )
            feedback_raw, direction_slices = [], []
            n_judged = 0
            for entry, (c_start, c_end) in zip(
                collage_paths_by_direction, direction_slices_collage
            ):
                pairs = (
                    all_by_direction.get(entry["direction_idx"], [])
                    if use_all
                    else [p for p in all_pairs[c_start:c_end] if p is not None]
                )
                start = len(feedback_raw)
                if len(pairs) > 0:
                    y_src = [int(p["orig_class"]) for p in pairs]
                    y_tgt = [int(p["target_class"]) for p in pairs]
                    y_end = [float(p["confidence"]) for p in pairs]
                    y_start = [
                        float(p.get("orig_conf", 1.0 - float(p["confidence"])))
                        for p in pairs
                    ]
                    y_gt = [
                        self._pair_gt_label(y_samples, p, y_src[i])
                        for i, p in enumerate(pairs)
                    ]
                    x_list = [
                        self.generator_to_classifier(p["x_factual"].float())
                        for p in pairs
                    ]
                    x_cf_list = [
                        self.generator_to_classifier(p["x_counterfactual"].float())
                        for p in pairs
                    ]
                    verdicts = self.teacher.get_feedback(
                        x_counterfactual_list=x_cf_list,
                        y_source_list=y_src,
                        x_list=x_list,
                        y_list=y_gt,
                        y_target_end_confidence_list=y_end,
                        y_target_start_confidence_list=y_start,
                        y_target_list=y_tgt,
                        student=self.student,
                        base_dir=self.base_dir,
                        mode="directions",
                    )
                    feedback_raw.extend(list(verdicts))
                    n_judged += len(pairs)
                    del x_list, x_cf_list
                direction_slices.append((start, len(feedback_raw)))
            _log.info(
                "%s",
                f"[DiDAE] Model teacher judged {n_judged} verified flips of "
                f"{len(collage_paths_by_direction)} directions.",
            )
        else:
            feedback_raw = self.teacher.get_feedback(
                num_clusters=len(collage_paths_by_direction),
                collage_path_list=all_paths,
                x_counterfactual_list=(
                    [p["x_counterfactual"].float() for p in all_pairs]
                    if all(p is not None for p in all_pairs)
                    else [torch.zeros(3, 64, 64)] * len(all_paths)
                ),
                y_list=y_list,
                y_source_list=y_source_list,
                y_target_list=y_target_list,
                y_target_end_confidence_list=y_end_conf_list,
                y_target_start_confidence_list=y_start_conf_list,
                x_list=(
                    [p["x_factual"].float() for p in all_pairs]
                    if all(p is not None for p in all_pairs)
                    else [torch.zeros(3, 64, 64)] * len(all_paths)
                ),
                base_dir=self.base_dir,
            )

        # Majority-aggregate the per-pair verdicts onto their direction. Anything
        # that is not a true/false judgement (sentinels like "ood" or "teacher
        # originally wrong!") carries no signal and is ignored.
        direction_feedback = []
        for entry, (start, end) in zip(collage_paths_by_direction, direction_slices):
            verdicts = [
                str(feedback_raw[i]) for i in range(start, min(end, len(feedback_raw)))
            ]
            n_true = sum(1 for v in verdicts if v == "true")
            n_false = sum(1 for v in verdicts if v == "false")
            fb = "false" if n_false > n_true else "true"
            direction_feedback.append(
                {
                    "direction_idx": entry["direction_idx"],
                    "feedback": fb,
                    "success_count": entry["success_count"],
                    "n_true": n_true,
                    "n_false": n_false,
                    "n_pairs": len(verdicts),
                }
            )

        return direction_feedback

    def _run_cfkd_on_false_directions(self, false_direction_indices, writer):
        """
        Step 9: Run CFKD finetuning using counterfactuals from false directions.
        """
        if len(false_direction_indices) == 0:
            _log.info(
                "%s", "[DiDAE] No false directions found. Skipping CFKD finetuning."
            )
            return

        _log.info(
            "%s",
            f"[DiDAE] Step 9: CFKD finetuning on {len(false_direction_indices)} false directions: {false_direction_indices}",
        )

        # Import CFKD and run it with component_indices set to false directions
        from peal.adaptors.counterfactual_knowledge_distillation import CFKD

        # The sweep found these directions under this run's component_bounds_scale;
        # CFKD's "dynamic" linesearch clamps to the same c_min_and_maxes.txt, so it
        # has to widen them identically or the edit it was told to make is out of
        # reach (measured: MSAE 'homme' flips 0/8 at x1 and 3/8 at x3).
        explainer_cfg = copy.deepcopy(self.adaptor_config.explainer)
        if isinstance(explainer_cfg, str):
            explainer_cfg = load_yaml_config(explainer_cfg)
        scale = float(getattr(self.adaptor_config, "component_bounds_scale", 1.0))
        if isinstance(explainer_cfg, dict):
            explainer_cfg["component_bounds_scale"] = scale
        else:
            explainer_cfg.component_bounds_scale = scale

        # Build a CFKD config from our DiDAE config
        # The explainer decodes batch_size x len(component_indices) images at once
        # (num_attempts = one per direction). Two directions at 200 fit the H100;
        # five at 200 asked for 1000 DDPM decodes and ran out of GPU memory. Keep
        # the decode workload at the two-direction level.
        n_dirs = max(2, len(false_direction_indices))
        cfkd_batch_size = max(1, (2 * int(self.adaptor_config.batch_size)) // n_dirs)
        if cfkd_batch_size != self.adaptor_config.batch_size:
            _log.info(
                "%s",
                f"[DiDAE] Step 9 uses batch_size {cfkd_batch_size} for "
                f"{len(false_direction_indices)} directions (config: "
                f"{self.adaptor_config.batch_size}).",
            )

        cfkd_dict = {
            "adaptor_type": "CFKD",
            "batch_size": cfkd_batch_size,
            "tracking_level": self.adaptor_config.tracking_level,
            "min_train_samples": self.adaptor_config.min_train_samples,
            "finetune_iterations": self.adaptor_config.finetune_iterations,
            "max_validation_samples": self.adaptor_config.max_validation_samples,
            "continuous_learning": self.adaptor_config.continuous_learning,
            "calculate_group_accuracies": self.adaptor_config.calculate_group_accuracies,
            "counterfactual_type": self.adaptor_config.counterfactual_type,
            "use_true_counterfactuals": self.adaptor_config.use_true_counterfactuals,
            "max_test_batches": self.adaptor_config.max_test_batches,
            "component_indices": false_direction_indices,
            "data": self.adaptor_config.data,
            "test_data": self.adaptor_config.test_data,
            "student": self.adaptor_config.student,
            "teacher": (
                self.adaptor_config.cfkd_teacher
                if self.adaptor_config.cfkd_teacher is not None
                else self.adaptor_config.teacher
            ),
            "generator": self.adaptor_config.generator,
            "sparse_dictionary": self.adaptor_config.sparse_dictionary,
            "base_dir": os.path.join(self.base_dir, "cfkd_finetuning"),
            "task": self.adaptor_config.task,
            "training": self.adaptor_config.training,
            "explainer": explainer_cfg,
            "seed": self.adaptor_config.seed,
        }

        # Write the equivalent standalone CFKD config first: if this process is
        # killed during step 9 (the 16 GiB cgroup), `run_cfkd.py --config` on it
        # continues from here without repeating discovery.
        standalone_path = os.path.join(self.base_dir, "cfkd_on_false_directions.yaml")
        try:
            save_yaml_config(copy.deepcopy(cfkd_dict), standalone_path)
            _log.info("%s", f"[DiDAE] Step 9 config written to {standalone_path}")
        except Exception as exc:  # never let bookkeeping stop the finetuning
            _log.info("%s", f"[DiDAE] Could not write {standalone_path}: {exc}")

        # CFKD builds its own generator, dictionary, teacher and dataloaders, so
        # for its lifetime this process held two of everything and was SIGKILLed
        # at the cgroup ceiling (measured 14.3 GiB with the slim generator).
        # Nothing after step 9 needs discovery's copies: drop them before CFKD
        # allocates its own. The student is re-loaded by CFKD from its path and
        # replaced by the finetuned one below.
        for attr in (
            "generator",
            "teacher",
            "train_dataloader",
            "val_dataloader",
            "test_dataloader",
            "joint_validation_dataloader",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        cfkd = CFKD(adaptor_config=cfkd_dict)
        self.student = cfkd.run()
        self.student_finetuned = True
        # Post-CFKD evaluation in run() needs the test loader back.
        self.test_dataloader = cfkd.test_dataloader

    def _save_final_student(self):
        """model.cpl for a torch student; model.onnx as well when the student
        came from an ONNX file or export_onnx is set. An onnxruntime closure
        (a graph onnx2torch could not convert) has nothing to save: the original
        file is the model, and step 9 could not have changed it."""
        from peal.architectures.onnx_predictor import OnnxPredictor

        student = self.student
        if not isinstance(student, torch.nn.Module):
            src = getattr(student, "onnx_path", None)
            _log.info(
                "%s",
                "[DiDAE] student is not a torch module (onnxruntime closure"
                + (f" for {src}" if src else "")
                + "); nothing to save.",
            )
            return
        torch.save(student, os.path.join(self.base_dir, "model.cpl"))
        student_spec = self.adaptor_config.student
        from_onnx = isinstance(student, OnnxPredictor) or (
            isinstance(student_spec, str) and student_spec.endswith(".onnx")
        )
        if not (from_onnx or self.adaptor_config.export_onnx):
            return
        onnx_out = os.path.join(self.base_dir, "model.onnx")
        if (
            not getattr(self, "student_finetuned", False)
            and isinstance(student_spec, str)
            and student_spec.endswith(".onnx")
            and os.path.isfile(student_spec)
        ):
            # Step 9 did not touch the student: the uploaded graph IS the result.
            shutil.copy(student_spec, onnx_out)
            _log.info("%s", f"[DiDAE] student unchanged; copied {student_spec}")
            return
        input_shape = getattr(student, "input_shape", None)
        if input_shape is None:
            input_shape = list(self.adaptor_config.data.input_size)
        # In a subprocess: the legacy exporter segfaults in ONNX shape inference
        # on aarch64 torch 2.8 (DGX Spark, 2026-09-28), which no try/except can
        # catch and which used to kill the run after the user's verdicts.
        import subprocess
        import sys

        code = (
            "import sys, torch\n"
            "from peal.architectures.onnx_predictor import export_to_onnx\n"
            "m = torch.load(sys.argv[1], weights_only=False, map_location='cpu')\n"
            "print(export_to_onnx(m, sys.argv[2], input_shape=[int(v) for v in sys.argv[3].split(',')], device='cpu'))\n"
        )
        proc = subprocess.run(
            [
                sys.executable,
                "-c",
                code,
                os.path.join(self.base_dir, "model.cpl"),
                onnx_out,
                ",".join(str(int(v)) if v is not None else "1" for v in input_shape),
            ],
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0 and os.path.isfile(onnx_out):
            _log.info("%s", f"[DiDAE] corrected student exported to {onnx_out}")
        else:
            _log.info(
                "%s",
                f"[DiDAE] ONNX export failed (exit {proc.returncode}); model.cpl is "
                f"saved. {proc.stderr.strip()[-500:]}",
            )

    def run(self):
        """Execute the full 9-step DiDAE workflow and return the (repaired) student.

        Steps, with the artefacts written under ``base_dir``:

        * **Step 1**: distil the student into a linear probe on ``z_sem``
          (``distilled_probe/w.pt``, ``bias.pt``) and compute the empirical
          component bounds ``c_min_and_maxes.txt`` of the dictionary.
        * **Step 1.5**: match ground-truth attributes to dictionary concepts on
          up to 5000 samples of the generator's (or the adaptor's) data via
          ``generator.compute_bipartite_gt_sae_matching``.
        * **Steps 2-6**: draw a CFKD-style class-balanced pool of
          ``min_train_samples + max_validation_samples`` images and call
          ``generator.sweep_all_directions``; the ranked metadata goes to
          ``sweep_results.pt``. With ``latent_only`` the run stops here.
        * **Step 7**: render collages of the top-k directions into
          ``direction_collages/``.
        * **Step 8**: collect true/false verdicts from the teacher (a
          ``Model2ModelTeacher`` judges every direction with a verified flip)
          into ``direction_feedback.txt``; false directions are ranked by
          false flips and pruned by ``cfkd_false_direction_min_share`` and
          ``cfkd_max_false_directions``.
        * **Step 9**: unless disabled, run CFKD on the false directions
          (``cfkd_finetuning/``, standalone config
          ``cfkd_on_false_directions.yaml``) and evaluate the result on the
          test set.

        Scalars are logged to ``logs/`` (TensorBoard) and the final student
        is saved by ``_save_final_student``. An empty sweep sets
        ``self.sweep_found_nothing`` and returns early.

        Returns
        -------
        torch.nn.Module
            The student, finetuned when step 9 ran.
        """
        _log.info("%s", "=" * 80)
        _log.info("%s", "[DiDAE] Starting automated counterfactual workflow")
        _log.info("%s", "=" * 80)

        writer = SummaryWriter(os.path.join(self.base_dir, "logs"))

        # --- Step 1: Distill classifier ---
        w, bias = self._distill_closed_form()

        # Ensure empirical component bounds (c_min_and_maxes.txt) are computed for dynamic line search
        self._ensure_component_bounds()

        # --- Step 1.5: Compute Bipartite GT-to-SAE Matching on generator's balanced dataset for 100% attribute coverage ---
        gen_dataloaders = None
        if not self.adaptor_config.use_generator_data_for_matching:
            _log.info(
                "%s",
                "[DiDAE] Matching concepts on the adaptor's own data "
                "(use_generator_data_for_matching=False).",
            )
        elif (
            hasattr(self.generator, "generator_datasets")
            and self.generator.generator_datasets
        ):
            gen_dataloaders = [
                DataLoader(
                    ds,
                    batch_size=getattr(self.adaptor_config, "batch_size", 200),
                    shuffle=False,
                )
                for ds in self.generator.generator_datasets
            ]
        elif (
            self.adaptor_config.use_generator_data_for_matching
            and hasattr(self.generator, "config")
            and getattr(self.generator.config, "data", None)
        ):
            try:
                from peal.data.dataset_factory import get_datasets

                gen_datasets_tuple = get_datasets(self.generator.config.data)
                gen_dataloaders = [
                    DataLoader(
                        ds,
                        batch_size=getattr(self.adaptor_config, "batch_size", 200),
                        shuffle=False,
                    )
                    for ds in gen_datasets_tuple
                ]
            except Exception as e:
                _log.info(
                    "%s",
                    f"[DiDAE] Warning: Could not instantiate generator datasets: {e}",
                )
                gen_dataloaders = None

        sd = getattr(self.generator, "sparse_dictionary", None)
        vocabulary = None
        matched_results = None
        if sd is not None:
            if hasattr(sd, "get_vocabulary"):
                try:
                    vocabulary = sd.get_vocabulary()
                except Exception:
                    vocabulary = None
            elif hasattr(sd, "vocab"):
                vocabulary = sd.vocab
            elif hasattr(sd, "concept_names"):
                vocabulary = sd.concept_names
            elif hasattr(sd, "component_names"):
                vocabulary = sd.component_names

        matched_results = None
        _log.info(
            "%s",
            "\n[DiDAE] Computing global bipartite GT-to-SAE matching on "
            + ("the generator's" if gen_dataloaders else "the adaptor's")
            + " dataset...",
        )
        x_match, y_match, attr_names_match = self._collect_sweep_samples(
            target_n=5000,
            bypass_task_config=True,
            dataloaders=(
                gen_dataloaders
                if gen_dataloaders
                else [self.train_dataloader, self.val_dataloader]
            ),
        )
        # Convert to generator normalisation and encode chunk by chunk. A
        # generator-normalised copy of the whole pool (2340 x 3x256x256 floats =
        # 1.8 GB for an ImageNet pair) next to x_match itself and the run's ~6 GB
        # baseline is what pushed five of six ImageNet rendering runs over the
        # 16 GiB step cap on 2026-09-15, right after the bounds step.
        with torch.no_grad():
            z_sem_chunks = []
            chunk_size = 64
            for i in range(0, x_match.shape[0], chunk_size):
                x_chunk = self.classifier_to_generator(x_match[i : i + chunk_size]).to(
                    self.generator.device
                )
                z_chunk = self.generator.encode(x_chunk, only_semantic=True)
                z_sem_chunks.append(z_chunk.cpu())
            z_sem_match = torch.cat(z_sem_chunks, dim=0)
            del x_match

            if sd is not None:
                c_sae_match = None
                if hasattr(sd, "encode"):
                    try:
                        c_sae_match = sd.encode(
                            z_sem_match.to(self.generator.device)
                        ).cpu()
                    except Exception:
                        c_sae_match = None

                if c_sae_match is None:
                    W_all = sd.get_components()
                    W = W_all.to(device=self.generator.device, dtype=z_sem_match.dtype)
                    if (
                        W.shape[0] != z_sem_match.shape[1]
                        and W.shape[1] == z_sem_match.shape[1]
                    ):
                        W = W.T
                    D_z = z_sem_match.shape[1]
                    if W.shape[0] != D_z:
                        if W.shape[0] < D_z:
                            W = torch.nn.functional.pad(W, (0, 0, 0, D_z - W.shape[0]))
                        else:
                            W = W[:D_z, :]

                    c_raw = torch.matmul(z_sem_match.to(self.generator.device), W)
                    c_sae_match = (c_raw - c_raw.mean(dim=0, keepdim=True)).relu().cpu()

                matched_results = self.generator.compute_bipartite_gt_sae_matching(
                    y_samples=y_match,
                    c_sae=c_sae_match,
                    attribute_names=attr_names_match,
                    vocabulary=vocabulary,
                    output_dir=self.base_dir,
                )

        # The matching above holds up to 5000 images (x_match, freed right after
        # encoding) plus their embeddings, and nothing below reads them. They stay alive to the end of run()
        # otherwise, because they are locals of this frame -- which matters here:
        # step 9 builds a second full CFKD stack inside this process, and the
        # Slurm step caps every process in the allocation at 16 GiB together.
        del y_match, attr_names_match
        if "z_sem_match" in dir():
            del z_sem_match
        if "c_sae_match" in dir():
            del c_sae_match
        gc.collect()

        # --- Step 2-6: Sweep all directions on the CFKD-style sample pool ---
        # Discovery now uses the same two knobs CFKD does -- min_train_samples for
        # the training pool, max_validation_samples for the validation pool -- and
        # seeds from both, class-balanced, honouring counterfactual_type. n_samples
        # is ignored: it used to be both the budget and, via `> len(val)`, a switch
        # between val-only and train+val, so the two effects could not be separated.
        n_samples = self.adaptor_config.n_samples
        sweep_budget = (
            self.adaptor_config.min_train_samples
            + self.adaptor_config.max_validation_samples
        )
        if n_samples is not None and n_samples != sweep_budget:
            _log.info(
                "%s",
                f"[DiDAE] Ignoring deprecated n_samples={n_samples}; the sweep pool "
                f"is min_train_samples + max_validation_samples = "
                f"{self.adaptor_config.min_train_samples} + "
                f"{self.adaptor_config.max_validation_samples} = {sweep_budget}.",
            )
        x_samples, y_samples, full_attr_names = self._collect_sweep_samples_balanced(
            target_n=sweep_budget,
        )
        _log.info(
            "%s",
            f"\n[DiDAE] Step 2-6: Sweeping all SAE directions with {x_samples.shape[0]} train+validation samples...",
        )
        x_samples_gen = self.classifier_to_generator(x_samples)
        # Only the generator-normalised pool is used from here on; the classifier
        # resolution copy (1.4 GB for an ImageNet pair) would otherwise stay alive
        # through the whole sweep and the CFKD stack. See the matching note above.
        n_sweep_samples = int(x_samples.shape[0])
        del x_samples
        gc.collect()

        data_cfg = getattr(self.adaptor_config, "data", None)
        attr_names = (
            full_attr_names
            or getattr(data_cfg, "confounding_factors", None)
            or getattr(data_cfg, "y_selection", None)
        )

        direction_results = self.generator.sweep_all_directions(
            x_samples=x_samples_gen,
            w=w,
            predictor=self.student,
            linesearch_factors=self.adaptor_config.linesearch_factors,
            decode_batch_size=self.adaptor_config.decode_batch_size,
            max_cf_per_direction=self.adaptor_config.max_cf_per_direction,
            y_samples=y_samples,
            attribute_names=attr_names,
            output_dir=self.base_dir,
            precomputed_matched_results=matched_results,
            explainer_config=self.adaptor_config.explainer,
            latent_only=self.adaptor_config.latent_only,
            bias=bias,
            max_decode_directions=self.adaptor_config.max_decode_directions,
            max_decode_per_direction=self.adaptor_config.max_decode_per_direction,
            decode_selection=self.adaptor_config.decode_selection,
            max_export_per_direction=self.adaptor_config.max_export_per_direction,
            edit_depth_factor=self.adaptor_config.edit_depth_factor,
            sweep_atom_subset=self.adaptor_config.sweep_atom_subset,
            component_bounds_scale=self.adaptor_config.component_bounds_scale,
            concept_replacement=self.adaptor_config.concept_replacement,
            concept_replacement_candidates=(
                self.adaptor_config.concept_replacement_candidates
            ),
            concept_replacement_unique=self.adaptor_config.concept_replacement_unique,
            concept_replacement_unique_strategy=(
                self.adaptor_config.concept_replacement_unique_strategy
            ),
        )

        if len(direction_results) == 0:
            _log.info(
                "%s", "[DiDAE] No successful counterfactual directions found. Aborting."
            )
            _log.info(
                "%s",
                "[DiDAE] Nothing downstream of the sweep ran. Check, in this order:\n"
                "  - the 'Probe margins' line above: N/N or 0/N positive means the "
                "probe puts every sample on one side of its boundary, and a flip "
                "then has to cross from that side alone;\n"
                "  - the 'Distillation MSE / correlation' line: a probe that does "
                "not follow the student has no boundary worth crossing;\n"
                "  - the clamp line: if the median surviving fraction of the "
                "requested step is near zero, the empirical [c_min, c_max] bounds "
                "are what is blocking the edit, not the direction set. "
                "component_bounds_scale widens them.",
            )
            # An empty sweep is a failed run, not a successful one that happened
            # to find nothing: this used to exit 0 and read as success in a
            # reproduction script.
            self.sweep_found_nothing = True
            writer.close()
            return self.student

        # Log sweep statistics
        top_k = min(self.adaptor_config.top_k_directions, len(direction_results))
        if self.adaptor_config.latent_only:
            _log.info(
                "%s",
                f"\n[DiDAE] Top {top_k} directions (ranked by latent-space flips; "
                "ambient verification needs the trained decoder):",
            )
        else:
            _log.info(
                "%s",
                f"\n[DiDAE] Top {top_k} winning directions (ranked by verified ambient flips & target feature change):",
            )
        for i in range(top_k):
            r = direction_results[i]
            dim_name = r.get("dimension_name", f"dim_{r['direction_idx']}")
            latent_flips = r.get("latent_flip_count", r["total_attempted"])
            if self.adaptor_config.latent_only:
                _log.info(
                    "%s",
                    f"  #{i+1}: Index {r['direction_idx']} ({dim_name}) — Latent Flips: "
                    f"{latent_flips}/{n_sweep_samples} "
                    f"(toward c_max: {r.get('increase_flips', '-')}/{r.get('increase_attempts', '-')}, "
                    f"toward c_min: {r.get('decrease_flips', '-')}/{r.get('decrease_attempts', '-')})",
                )
            else:
                _log.info(
                    "%s",
                    f"  #{i+1}: Index {r['direction_idx']} ({dim_name}) — Verified Flips: {r['success_count']}/{r['total_attempted']}, Ambient Flips: {r.get('ambient_flip_count', r['success_count'])}/{r['total_attempted']}, Latent Flips: {latent_flips}/{n_sweep_samples}",
                )
                writer.add_scalar(
                    f"sweep/direction_{r['direction_idx']}_verified_flips",
                    r["success_count"],
                    0,
                )
            writer.add_scalar(
                f"sweep/direction_{r['direction_idx']}_latent_flips",
                latent_flips,
                0,
            )

        # Save sweep results
        sweep_results_path = os.path.join(self.base_dir, "sweep_results.pt")
        # Save metadata (not full tensors) to avoid huge files
        sweep_metadata = [
            {
                "direction_idx": r["direction_idx"],
                "dimension_name": r.get("dimension_name", f"dim_{r['direction_idx']}"),
                "success_count": r["success_count"],
                "ambient_flip_count": r.get("ambient_flip_count", r["success_count"]),
                "latent_flip_count": r.get("latent_flip_count", r["total_attempted"]),
                "total_attempted": r["total_attempted"],
                "increase_flips": r.get("increase_flips"),
                "increase_attempts": r.get("increase_attempts"),
                "decrease_flips": r.get("decrease_flips"),
                "decrease_attempts": r.get("decrease_attempts"),
            }
            for r in direction_results
        ]
        torch.save(sweep_metadata, sweep_results_path)

        if self.adaptor_config.latent_only:
            _log.info(
                "%s",
                f"\n[DiDAE] latent_only: stopping after the latent half of the "
                f"workflow. Wrote the direction ranking to {sweep_results_path} and "
                f"the concept matching to {self.base_dir}. Train the diffusion "
                "autoencoder, then re-run with latent_only: False for decoding, "
                "user feedback and CFKD.",
            )
            save_yaml_config(
                self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
            )
            writer.close()
            return self.student

        # --- Step 7: Generate collages for top-K directions ---
        top_k_results = direction_results[:top_k]
        collage_dir = os.path.join(self.base_dir, "direction_collages")
        _log.info(
            "%s", f"\n[DiDAE] Step 7: Generating collages for top {top_k} directions..."
        )
        collage_paths_by_direction = self._generate_direction_collages(
            top_k_results, collage_dir
        )

        # --- Step 8: User feedback ---
        _log.info(
            "%s", "\n[DiDAE] Step 8: Collecting user feedback via web interface..."
        )
        # A model teacher is a classifier pass, so it judges EVERY direction that
        # produced a verified flip, not just the top_k shown as collages: ranking
        # by verified flips alone favours directions that change the class
        # feature along with the spurious one (on CelebA Mustache/No_Beard edits
        # darken the hair and out-flip the pure Male atom), and those come out
        # "true". The false-flip count then ranks the directions for step 9.
        from peal.teachers.model2model_teacher import Model2ModelTeacher

        judge_results = top_k_results
        judge_entries = collage_paths_by_direction
        if isinstance(self.teacher, Model2ModelTeacher):
            judge_results = [
                r
                for r in direction_results
                if len(r.get("all_pairs", r.get("pairs", []))) > 0
            ]
            shown = {e["direction_idx"] for e in collage_paths_by_direction}
            judge_entries = list(collage_paths_by_direction) + [
                {
                    "direction_idx": r["direction_idx"],
                    "collage_paths": [],
                    "success_count": r["success_count"],
                }
                for r in judge_results
                if r["direction_idx"] not in shown
            ]
            _log.info(
                "%s",
                f"[DiDAE] Model teacher: judging all {len(judge_results)} directions "
                f"with verified flips (collages only for the top {top_k}).",
            )
        direction_feedback = self._get_user_feedback(
            judge_entries,
            direction_results=judge_results,
            y_samples=y_samples,
        )

        # Save feedback
        feedback_path = os.path.join(self.base_dir, "direction_feedback.txt")
        with open(feedback_path, "w") as f:
            for df in direction_feedback:
                f.write(
                    f"direction={df['direction_idx']}, feedback={df['feedback']}, "
                    f"success_count={df['success_count']}, "
                    f"n_true={df.get('n_true', '-')}, n_false={df.get('n_false', '-')}, "
                    f"n_pairs={df.get('n_pairs', '-')}\n"
                )

        # Summarize feedback
        true_dirs = [
            df["direction_idx"] for df in direction_feedback if df["feedback"] == "true"
        ]
        false_dirs = [
            df["direction_idx"]
            for df in direction_feedback
            if df["feedback"] == "false"
        ]
        _log.info("%s", "\n[DiDAE] Feedback summary:")
        _log.info("%s", f"  True directions  ({len(true_dirs)}): {true_dirs}")
        _log.info("%s", f"  False directions ({len(false_dirs)}): {false_dirs}")
        # Step 9 direction selection: weight the teacher's verdict by evidence.
        # Evidence of bias = the number of FALSE flips (student flipped, oracle
        # did not); falls back to the verified count when the teacher gave no
        # per-flip counts (human / collage teachers).
        verified_by_dir = {
            df["direction_idx"]: int(df.get("n_false", 0) or df.get("success_count", 0))
            for df in direction_feedback
        }
        if len(false_dirs) > 1:
            false_dirs = sorted(false_dirs, key=lambda d: -verified_by_dir[d])
            _log.info(
                "%s",
                "[DiDAE] False directions ranked by false flips: "
                + str([(d, verified_by_dir[d]) for d in false_dirs]),
            )
        share = float(
            getattr(self.adaptor_config, "cfkd_false_direction_min_share", 0.0) or 0.0
        )
        if share > 0 and len(false_dirs) > 1:
            top = max(verified_by_dir[d] for d in false_dirs)
            kept = [d for d in false_dirs if verified_by_dir[d] >= share * top]
            dropped = [d for d in false_dirs if d not in kept]
            if dropped:
                _log.info(
                    "%s",
                    f"[DiDAE] Step 9 keeps false directions with >= {share:.2f} x "
                    f"{top} verified flips: {kept}; dropped {dropped} "
                    f"(verified {[verified_by_dir[d] for d in dropped]}).",
                )
            false_dirs = kept
        cap = getattr(self.adaptor_config, "cfkd_max_false_directions", None)
        if cap and len(false_dirs) > int(cap):
            false_dirs = sorted(false_dirs, key=lambda d: -verified_by_dir[d])[
                : int(cap)
            ]
            _log.info(
                "%s",
                f"[DiDAE] Step 9 capped to the {cap} strongest false directions: {false_dirs}",
            )

        writer.add_scalar("feedback/n_true_directions", len(true_dirs), 0)
        writer.add_scalar("feedback/n_false_directions", len(false_dirs), 0)

        # --- Step 9: CFKD on false directions ---
        # CFKD instantiates its own generator, dictionary, student, teacher and
        # dataloaders from the same configs, so for its lifetime this process
        # holds two of everything. Measured on CelebA/ViT-L14: the discovery half
        # sits at 8.2 GiB, and ~90 s into step 9 the process is at the 16 GiB
        # cgroup ceiling and is SIGKILLed. Dropping what discovery no longer
        # needs before handing over is the cheap half of the fix; the real one is
        # to pass the already-loaded generator and dictionary into CFKD instead
        # of letting it build a second copy.
        del x_samples_gen, direction_results, top_k_results
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if len(false_dirs) > 0 and not self.adaptor_config.run_cfkd_on_false_directions:
            _log.info(
                "%s",
                f"[DiDAE] Step 9 skipped (run_cfkd_on_false_directions=False). "
                f"{len(false_dirs)} false direction(s) would have been finetuned on: "
                f"{false_dirs}",
            )
        elif len(false_dirs) > 0:
            _log.info(
                "%s",
                "\n[DiDAE] Step 9: Running CFKD finetuning on false directions...",
            )
            self._run_cfkd_on_false_directions(false_dirs, writer)

            # Evaluate after finetuning
            if self.adaptor_config.calculate_group_accuracies:
                test_result = calculate_test_accuracy(
                    self.student,
                    self.test_dataloader,
                    self.device,
                    True,
                    self.adaptor_config.max_test_batches,
                    tracking_level=self.adaptor_config.tracking_level,
                )
                (
                    test_accuracy,
                    group_accuracies,
                    group_distribution,
                    groups,
                    worst_group_accuracy,
                ) = test_result
                _log.info("%s", f"[DiDAE] Post-CFKD test accuracy: {test_accuracy:.4f}")
                _log.info(
                    "%s", f"[DiDAE] Post-CFKD group accuracies: {group_accuracies}"
                )
                _log.info(
                    "%s",
                    f"[DiDAE] Post-CFKD worst group accuracy: {worst_group_accuracy:.4f}",
                )
                writer.add_scalar("post_cfkd/test_accuracy", test_accuracy, 0)
                writer.add_scalar(
                    "post_cfkd/worst_group_accuracy", worst_group_accuracy, 0
                )
            else:
                test_accuracy = calculate_test_accuracy(
                    self.student,
                    self.test_dataloader,
                    self.device,
                    False,
                    self.adaptor_config.max_test_batches,
                    tracking_level=self.adaptor_config.tracking_level,
                )
                _log.info("%s", f"[DiDAE] Post-CFKD test accuracy: {test_accuracy:.4f}")
                writer.add_scalar("post_cfkd/test_accuracy", test_accuracy, 0)
        else:
            _log.info(
                "%s",
                "[DiDAE] No false directions — classifier appears unbiased for these directions.",
            )

        # Save final model
        self._save_final_student()
        save_yaml_config(
            self.adaptor_config, os.path.join(self.base_dir, "config.yaml")
        )

        writer.close()
        _log.info("%s", "\n" + "=" * 80)
        _log.info("%s", "[DiDAE] Workflow complete!")
        _log.info("%s", "=" * 80)

        return self.student
