from typing import Union

from pydantic import BaseModel


class Adaptor:
    def run(self):
        """
        Run the adaptor.
        """
        raise NotImplementedError

class AdaptorConfig(BaseModel):
    """
    The config template for an adaptor.
    """
    """
    The type of adaptor that shall be used.
    This is necessary to know which pydantic class to use when loading from yaml.
    """
    adaptor_type: str
    """
    The category of the config. Can not be changed for adaptor.
    This is also necessary to identify which pydantic class to use when loading from yaml.
    """
    category: str = 'adaptor'
    """
    The seed of all randomness to make results reproducible.
    """
    seed: Union[int, type(None)] = 0
    """
    How many intermediate results are cached an visualized.
    0   -> None
    >=1 -> only progress bars
    >=2 -> caching & prints
    >=3 -> visualizations
    >=4 -> calculation of all values mentioned in the papers
    >=5 -> everything, including expensive visualizations and tracking of values not mentioned by papers
    """
    tracking_level: int = 0