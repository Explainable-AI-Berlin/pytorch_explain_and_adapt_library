from typing import Union

from pydantic import BaseModel


class Adaptor:
    def run(self):
        """
        Run the adaptor.
        """
        raise NotImplementedError

class AdaptorConfig:
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
    seed: Union[int, type(None)] = None
    """
    How many intermediate results are cached an visualized.
    Goes from 0 = None over 1 = caching only to 2 = essential visualizations to 3 = expensive visualizations, 4 = all.
    """
    tracking_level: int = 0