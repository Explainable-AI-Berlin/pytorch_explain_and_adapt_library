
from peal.data.datasets import DataConfig
from peal.adaptors.interfaces import Adaptor, AdaptorConfig
from peal.training.trainers import TrainingConfig
from peal.    import



class DROConfig(AdaptorConfig):
    """
    Config template for running the DRO adaptor.
    """

    """
    The config template for an adaptor
    """
    adaptor_type: str = "GroupDRO"
    """
    The category of the config.
    """
    category: str = 'adaptor'
    """
    The config of the data to train on.
    """
    data: DataConfig = None
    """
    The config of the data used to evaluate real progress on.
    """
    test_data: DataConfig = None
    """
    The config of the trainer used for training the model.
    """
    training: TrainingConfig = TrainingConfig()
    """
    The base directory where the run of GroupDRO is stored.
    """
    base_dir: str = "peal_runs/dro"

    """
    The name of the class.
    """
    __name__: str = "peal.DROConfig"

    def __init__(
        self,
        data: Union[dict, DataConfig] = None,
        test_data: Union[dict, DataConfig] = None
    ):
        """
        The config template for the DRO adaptor.
        Sets the values of the config that are listed above.

        TODO: Run checks to assure all values are filled, including with defaults, if necessary
        Args:
            so weiter und so fort
        """

        # TODO: We are using pydantic to create the config file. Be sure to check that it's written in this style

        if not data is None:
            if isinstance(data, DataConfig):
                self.data = data
            else:
                self.data = DataConfig(**data)

        if not test_data is None:
            if isinstance(test_data, DataConfig):
                self.test_data = test_data
            else:
                self.test_data = DataConfig(**test_data)



class DRO(Adaptor):
    """
    DRO Adaptor docstring.
    """

    # Instantiate the dataset with return_dict=True

    # Pass an instantiated datasets to the model trainer as a list of datasets
    # [train, val, test] under the datasource keyword

    def __init__(
        self,
        datasource: Union[list, tuple] = None,
        adaptor_config: Union[
            dict, str, Path, AdaptorConfig
        ] = "TODO" # TODO: Replace "TODO"
    ):

        self.adaptor_config = load_yaml_config(adaptor_config, AdaptorConfig)




    def run(self):
        """
        Runs GroupDRO method.
        """

        #



        raise NotImplimentedError