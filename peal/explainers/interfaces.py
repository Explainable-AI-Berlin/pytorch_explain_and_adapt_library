from typing import Union

from pydantic import BaseModel


class ExplainerConfig(BaseModel):
    """
    This class defines the config of a ExplainerConfig.
    """
    """
    The type of explanation that shall be used.
    This is necessary to know which pydantic class to use when loading from yaml.
    """
    explainer_type: str
    """
    The category of the config. Can not be changed for explainer.
    This is also necessary to identify which pydantic class to use when loading from yaml.
    """
    category: str = 'explainer'
    """
    The directory where the explanations are stored.
    This only is used if explainer is executed directly and not e.g. executed via CFKD.
    """
    explanations_dir: str = 'explanations'
    """
    The port the feedback for the explanations shall be given when using the webinterface.
    """
    port: int = 8000
    """
    How many intermediate results are cached an visualized.
    Goes from 0 = None over 1 = caching only to 2 = essential visualizations to 3 = all.
    """
    tracking_level: int = 2
    """
    Whether to sanity check the used generator before creating explanations.
    """
    validate_generator: bool = False
    """
    The number of samples that counterfactuals are created for.
    If set to None there will be one counterfactual created for every sample in dataset.
    """
    max_samples: Union[int, None] = None
    """
    The temperature used for the softmax when creating counterfactuals.
    Can be useful for calibration if confidence goes against 0 or 1 too fast.
    """
    temperature: float = 3.0
    """
    Whether to cluster the explanations and return most salient ones.
    """
    use_clustering: bool = True
    """
    How to merge clusters of explanations?
    """
    merge_clusters: str = "best"
    """
    The number of counterfactuals created for the same sample.
    """
    num_attempts: int = 1
    """
    The seed of all randomness to make results reproducible.
    """
    seed: int = 0
    """"
    The restriction to interesting counterfactual transitions.
    Helpful in the case of datasets with a lot of classes and heavy modes like ImageNet.
    """
    transition_restrictions: Union[list, type(None)] = None
    clustering_strategy: str = "highest_activation"


class ExplainerInterface:
    explainer_config: ExplainerConfig
    def explain_batch(self, batch, **args):
        raise NotImplementedError

    def run(self, oracle_path=None, confounder_oracle_path=None):
        pass

    def human_annotate_explanations(self, param):
        pass

    def visualize_interpretations(self, feedback, param, param1):
        pass
