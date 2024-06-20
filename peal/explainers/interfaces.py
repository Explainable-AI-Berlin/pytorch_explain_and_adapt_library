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


class ExplainerInterface:
    def explain_batch(self, batch, **args):
        raise NotImplementedError

    def run(self, dataset, oracle_path, confounder_oracle_path):
        pass
