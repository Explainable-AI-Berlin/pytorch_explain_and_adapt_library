from typing import Union

from pydantic import BaseModel


class ExplainerConfig(BaseModel):
    """
    This class defines the config of a ExplainerConfig.
    """
    """
    The type of explanation that shall be used.
    Options: ['counterfactual', 'lrp']
    """
    explainer_type: str
    """
    The category of the config.
    """
    category: str = 'explainer'
    """
    The directory where the explanations are stored.
    """
    explanations_dir: str = 'explanations'
    """
    The port the feedback for the explanations shall be given.
    """
    port: int = 8000
    tracking_level: int = 2
    validate_generator: bool = False
    max_samples: Union[int, None] = None


class ExplainerInterface:
    def explain_batch(self, batch, **args):
        raise NotImplementedError

    def run(self, oracle_path=None, confounder_oracle_path=None):
        pass

    def human_annotate_explanations(self, param):
        pass

    def visualize_interpretations(self, feedback, param, param1):
        pass
