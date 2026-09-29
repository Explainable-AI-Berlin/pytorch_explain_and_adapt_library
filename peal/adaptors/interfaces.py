"""
Base interfaces shared by every PEAL adaptor.

An adaptor is a procedure that repairs (adapts) a trained classifier, e.g.
CFKD, DiDAE, ClArC or GroupDRO. This module holds the abstract ``Adaptor``
class every concrete adaptor derives from and the ``AdaptorConfig`` pydantic
model that carries the config keys common to all of them (``adaptor_type``,
``seed``, ``tracking_level``, ...). Concrete adaptor configs extend it.
"""

from typing import Union

from pydantic import BaseModel
from abc import ABC, abstractmethod


class Adaptor(ABC):
    """
    Abstract base class of all adaptors.

    Concrete adaptors (``CFKD``, ``DiDAE``, ``ClArC``, ``GroupDRO``, ...)
    subclass this and implement ``run``, which performs the whole adaptation
    procedure (explaining the student, collecting feedback, fine-tuning) and
    writes its artifacts to the adaptor's base directory.
    """

    @abstractmethod
    def run(self):
        """
        Run the adaptor.

        Raises
        ------
        NotImplementedError
            Always; subclasses must override this method.
        """
        raise NotImplementedError


class AdaptorConfig(BaseModel):
    """
    The config template for an adaptor.

    Every adaptor config extends this model. The ``adaptor_type`` and
    ``category`` fields tell the yaml loader which pydantic class to
    instantiate; the remaining fields control reproducibility and how much
    intermediate output is produced.

    Parameters
    ----------
    adaptor_type : str
        Name of the adaptor implementation, e.g. ``"CFKD"``.
    category : str
        Always ``"adaptor"``; used to route the yaml to this config family.
    seed : int or None
        Seed for all sources of randomness.
    tracking_level : int
        Verbosity/caching level from 0 (silent) to 5 (everything).
    calculate_explainer_stats : bool
        Whether to compute explainer statistics such as sparsity/diversity.
    in_memory : bool
        Whether datasets are fully loaded into RAM.
    """

    adaptor_type: str
    """
    The type of adaptor that shall be used.
    This is necessary to know which pydantic class to use when loading from yaml.
    """
    category: str = "adaptor"
    """
    The category of the config. Can not be changed for adaptor.
    This is also necessary to identify which pydantic class to use when loading from yaml.
    """
    seed: Union[int, type(None)] = 0
    """
    The seed of all randomness to make results reproducible.
    """
    tracking_level: int = 0
    """
    How many intermediate results are cached an visualized.
    0   -> None
    >=1 -> only progress bars
    >=2 -> prints
    >=3 -> caching
    >=4 -> visualizations
    >=5 -> everything, including expensive visualizations and tracking of values not mentioned by papers
    """
    calculate_explainer_stats: bool = False
    """
    Whether to calculate explainer stats like sparsity, diversity, etc.
    """
    in_memory: bool = False
    """
    Whether to load all datasets into the RAM or not. Careful with big datasets!
    """
