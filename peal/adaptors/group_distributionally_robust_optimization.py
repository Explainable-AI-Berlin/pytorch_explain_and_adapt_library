
from pathlib import Path
from typing import Union
from peal.training.trainers import PredictorConfig
from peal.adaptors.interfaces import Adaptor, AdaptorConfig

from torch import nn


dro_criterions = {
    "ce": nn.CrossEntropyLoss(reduction='none'),
    "bce": nn.BCEWithLogitsLoss(reduction='none'),
    "mse": nn.MSELoss(reduction='none'),
    "mae": nn.MSELoss(reduction='none'),
}


class GroupDROConfig(AdaptorConfig):
    """
    Config template for running the DRO adaptor.
    """

    """
    The config template for an adaptor
    """

    adaptor_type: str = "GroupDRO"
    """
    The config of the predictor to be adapted.
    """
    predictor: PredictorConfig = None
    """
    Parameters for the Gorup_DRO loss computer
    """
    is_robust: bool = False
    alpha: float = None
    gamma: float = 0.1
    adj: float = None
    min_var_weight: float = 0 # I'm guessing this is a float
    step_size: float = 0.01
    normalize_loss: bool = False
    btl: bool = False
    """
    Resets weights if true.
    """
    reset_weights: bool = True
    """
    The path where the model is to be stored. Explicitly overwrites model_path in predictor.
    """
    model_path: str = None
    """
    Sets the seed of the run. Expliticly overwrites the seed parameter in predictor.
    """
    seed: int = 0
    """
    A dict containing all variables that could not be given with the current config structure
    """
    kwargs: dict = {}
    """
    The name of the class.
    """
    __name__: str = "peal.GroupDROConfig"


    def __init__(
        self,
        predictor: Union[dict, PredictorConfig] = None,
        is_robust: bool = False,
        alpha: float = None,
        gamma: float = 0.1,
        adj: float = None,
        min_var_weight: float = 0,
        step_size: float = 0.01,
        normalize_loss: bool = False,
        btl: bool = False,
        reset_weights: bool = True,
        model_path: str = None,
        seed: int = 0,
        **kwargs,
    ):
        """
        The config template for the DRO adaptor.
        Sets the values of the config that are listed above.

        TODO: Run checks to assure all values are filled, including with defaults, if necessary
        Args:
            so weiter und so fort
        """

        # TODO: We are using pydantic to create the config file. Be sure to check that it's written in this style

        if isinstance(predictor, PredictorConfig):
            self.predictor = predictor
        elif isinstance(predictor, dict):
            self.predictor = PredictorConfig(**predictor)
        else:
            raise TypeErorr(f"predictor is of type {type(predictor)}; expecting type dict or PredictorConfig")

        self.is_robust = is_robust
        self.alpha = alpha
        self.gamma = gamma
        self.adj = adj
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.reset_weights = reset_weights
        self.model_path = model_path
        self.seed = seed
        self.kwargs = kwargs


class GroupDRO(Adaptor):
    """ GroupDRO Adaptor """

    def __init__(
        self,
        adaptor_config: Union[
            dict, str, Path, AdaptorConfig
        ] = "<PEAL_BASE>/configs/adaptors/test_dro.yaml",
        model_path=None,
        model=None,
        datasource=None,
        optimizer=None,
        criterions=None,
        logger=None,
        only_last_layer=False,
        unit_test_train_loop=False,
        unit_test_single_sample=False,
        log_frequency=1000,
        gigabyte_vram=None,
        val_dataloader_weights=[1.0],
    ):
        # TODO: Set parameters and settings for the adaptor

        # Transfer required parameters to the class

        # Get logs dir
        # Instantiate logs dir if it doesn't exist

        # Set the global seed

        # Load datasets

        # Load the model

        #

        pass


    def fit(self):

        # TODO: Create setup for the run. Roughly should be the contents of run_expt.py

        pass



    def run_epoch(self):

        # TODO: Should roughly be the contents of the train_epoch function in run_expt.py

        pass