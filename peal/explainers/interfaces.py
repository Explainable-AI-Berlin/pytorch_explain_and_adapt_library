"""Base config and interface shared by all PEAL explainers.

``ExplainerConfig`` is the pydantic base model that every concrete explainer
config (counterfactual, no-generator, ...) extends; ``explainer_type`` and
``category`` are the keys the yaml loader uses to pick the concrete pydantic
class. ``ExplainerInterface`` is the minimal duck-typed contract adaptors
such as CFKD rely on: ``explain_batch`` produces explanations for a batch and
``run`` executes the explainer stand-alone.
"""

from typing import Union

from pydantic import BaseModel
from abc import ABC, abstractmethod


class ExplainerConfig(BaseModel):
    """
    Base pydantic config of an explainer.

    Concrete explainers subclass this and add their own fields; the
    attributes documented inline below are common to all of them.
    """

    explainer_type: str
    """
    The type of explanation that shall be used.
    This is necessary to know which pydantic class to use when loading from yaml.
    """
    category: str = "explainer"
    """
    The category of the config. Can not be changed for explainer.
    This is also necessary to identify which pydantic class to use when loading from yaml.
    """
    explanations_dir: str = "explanations"
    """
    The directory where the explanations are stored.
    This only is used if explainer is executed directly and not e.g. executed via CFKD.
    """
    port: int = 8000
    """
    The port the feedback for the explanations shall be given when using the webinterface.
    """
    tracking_level: int = 2
    """
    How many intermediate results are cached an visualized.
    Goes from 0 = None over 1 = caching only to 2 = essential visualizations to 3 = all.
    """
    validate_generator: bool = False
    """
    Whether to sanity check the used generator before creating explanations.
    """
    max_samples: Union[int, None] = None
    """
    The number of samples that counterfactuals are created for.
    If set to None there will be one counterfactual created for every sample in dataset.
    """
    temperature: float = 3.0
    """
    The temperature used for the softmax when creating counterfactuals.
    Can be useful for calibration if confidence goes against 0 or 1 too fast.
    """
    use_clustering: bool = True
    """
    Whether to cluster the explanations and return most salient ones.
    """
    merge_clusters: str = "concatenate"
    """
    How to merge clusters of explanations?
    """
    num_attempts: int = 1
    """
    The number of counterfactuals created for the same sample.
    """
    seed: int = 0
    """
    The seed of all randomness to make results reproducible.
    """
    transition_restrictions: Union[list, type(None)] = None
    """
    The restriction to interesting counterfactual transitions.
    Helpful in the case of datasets with a lot of classes and heavy modes like ImageNet.
    """
    clustering_strategy: str = "attempt_nr"
    parallel_attempts: int = 1
    """
    How many counterfactuals are created for the same factual in parallel.
    """
    component_indices: Union[list, type(None)] = None
    component_bounds_scale: float = 1.0
    """
    Widen (>1) or tighten (<1) the empirical [c_min, c_max] clamp of the "dynamic"
    linesearch about its midpoint, the same knob DiDAE's sweep exposes as
    component_bounds_scale. A direction DiDAE only found at x3 is unreachable for
    CFKD at x1, so DiDAE step 9 forwards its own value here. 1.0 = bounds as written.
    """


class ExplainerInterface(ABC):
    """
    Duck-typed interface every explainer implements.

    Subclasses hold their config in ``explainer_config`` and must override
    ``explain_batch``; the remaining methods are no-op defaults that
    explainers without an interactive feedback loop may leave untouched.
    """

    explainer_config: ExplainerConfig

    @abstractmethod
    def explain_batch(self, batch, **args):
        """
        Explain one batch of samples.

        Parameters
        ----------
        batch : sequence
            A dataset batch, typically ``(x, y, ...)`` tensors.
        **args
            Explainer-specific keyword arguments.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError

    def run(self, oracle_path=None, confounder_oracle_path=None):
        """
        Run the explainer stand-alone (outside of an adaptor).

        Parameters
        ----------
        oracle_path : str, optional
            Path of an oracle classifier used to evaluate the explanations.
        confounder_oracle_path : str, optional
            Path of a confounder oracle used to evaluate the explanations.
        """

    def human_annotate_explanations(self, param):
        """Collect human feedback for previously created explanations (no-op)."""

    def visualize_interpretations(self, feedback, param, param1):
        """Visualize interpreted explanations together with their feedback (no-op)."""
