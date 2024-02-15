from typing import Union

from peal.configs.data.data_config import DataConfig
from peal.configs.explainers.explainer_config import ExplainerConfig


class PerfectFalseCounterfactualConfig(ExplainerConfig):
    """
    This class defines the config of a PerfectFalseCounterfactualConfig.
    """

    """
    The type of explanation that shall be used.
    """
    explainer_type: str = "PerfectFalseCounterfactual"
    data: Union[str, DataConfig] = None
    test_data: Union[str, DataConfig] = None
