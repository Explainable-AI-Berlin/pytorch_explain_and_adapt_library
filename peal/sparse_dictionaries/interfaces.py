"""Base config and interface shared by every PEAL sparse dictionary.

A sparse dictionary (SAE, Matryoshka SAE, Procrustes/SVD, SpLICE, ...) turns an
encoder activation vector into a sparse code over named components and back.
DiDAE edits a generator's semantic latent along those components, and CFKD can
attach one to its generator via ``adaptor_config.sparse_dictionary``. This module
holds the pydantic config every concrete dictionary config extends and the
abstract ``SparseDictionary`` API that ``sparse_dictionary_factory`` instantiates.
"""

from typing import Union

from pydantic import BaseModel


from peal.data.interfaces import DataConfig
from abc import ABC, abstractmethod


class SparseDictionaryConfig(BaseModel):
    """Common configuration fields of every sparse dictionary.

    Concrete dictionaries subclass this and override ``sparse_dictionaries_type``
    so that ``get_sparse_dictionary`` can resolve the class from a yaml file.

    Parameters
    ----------
    n_components : int or None
        Number of dictionary components (columns of the component matrix).
        Some dictionaries set this themselves after fitting.
    dict_size : int or None
        Size of the learned dictionary for SAE-style dictionaries; treated as
        a synonym of ``n_components`` by several implementations.
    sparse_dictionaries_type : str
        Registry key of the concrete dictionary class.
    category : str
        Fixed to ``"sparse_dictionaries"``; used when resolving default paths.
    base_path, name, run_name : str or None
        Where fitted weights, plots and logs of this dictionary live.
    weights_path : str or None
        Explicit path of the fitted weights; overrides ``base_path``/``name``.
    act_size : int
        Dimensionality of the encoder activations the dictionary is fitted on.
    visualizations_per_component : int
        How many top-activating samples to render per component.
    weights_name : str
        File name of the weights inside ``base_path``.
    batch_size : int
        Batch size used when the dictionary itself iterates over data.
    component_strings : list or None
        Optional human-readable names for the components.
    data : DataConfig, dict, str or None
        The data the dictionary is fitted on (config object, dict or yaml path).
    fitted_on_encoder : str or None
        Identity of the encoder whose activations the dictionary was fitted on;
        see the attribute docstring below.
    """

    n_components: Union[int, None] = None
    dict_size: Union[int, None] = None
    sparse_dictionaries_type: str
    category: str = "sparse_dictionaries"
    base_path: Union[str, None] = None
    name: Union[str, None] = None
    run_name: Union[str, None] = None
    weights_path: Union[str, None] = None
    act_size: int = 512
    visualizations_per_component: int = 100
    weights_name: str = "weights.npz"
    batch_size: int = 16
    component_strings: Union[list, None] = None
    data: Union[DataConfig, dict, str, None] = None
    fitted_on_encoder: Union[str, None] = None
    """
    Which encoder's activations this dictionary was fitted on, stamped at fit
    time. Consumers that edit in the generator's z_sem space (DiDAE) must refuse
    a dictionary fitted somewhere else: an OpenCLIP dictionary and a diffusion
    autoencoder's z_sem are both 768-d, so nothing about the shapes catches the
    mismatch and the edits would simply be meaningless.
    """


class SparseDictionary(ABC):
    """Abstract interface of a sparse dictionary.

    Subclasses hold a ``config`` (a ``SparseDictionaryConfig``) and implement
    fitting from a tensor of activations or from dataloaders, plus
    serialisation. Most concrete implementations additionally offer
    ``encode``/``decompose``, ``get_components`` and ``eval``, which the
    explainers call but which are not part of this minimal base class.

    Attributes
    ----------
    config : SparseDictionaryConfig
        The configuration the dictionary was built from.
    """

    config: SparseDictionaryConfig

    @abstractmethod
    def fit(self, X):
        """Fit the dictionary on a tensor of activations.

        Parameters
        ----------
        X : torch.Tensor
            Activations of shape ``[N, act_size]``.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def fit_from_dataloaders(self, dataloaders):
        """Fit the dictionary by iterating over one or more dataloaders.

        Parameters
        ----------
        dataloaders : list of torch.utils.data.DataLoader
            Loaders yielding ``(x, y)`` batches; implementations run the
            encoder themselves and collect the activations.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def save_on_disk(self, path):
        """Serialise the fitted state to ``path``.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    @abstractmethod
    def load_from_disk(self, path):
        """Restore the fitted state from ``path``.

        Raises
        ------
        NotImplementedError
            Always, in the base class.
        """
        raise NotImplementedError("Subclasses should implement this method.")
