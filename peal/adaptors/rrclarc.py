from collections.abc import Iterable

from peal.adaptors.interfaces import Adaptor, AdaptorConfig
from peal.dependencies.rrclarc.experiments.preprocessing.global_collect_relevances_and_activations import \
    run_collect_relevances_and_activations


class RRClArCConfig(AdaptorConfig):
    """
    The config template for a running the RR-ClArC adaptor.
    """

    """
    The config template for an adaptor.
    """
    adaptor_type: str = "RRClArC"

    batch_size: int = 64

    model_path: str

    compute: str = "l2_mean"

    criterion: str = "allrand"

    eval_acc_every_epoch: bool = False

    img_size: int = 224

    lamb: float = 1.0

    layer_name: str = "last_conv"

    loss: str = "cross_entropy"

    lr: float = 2.0e-05

    num_epochs: int = 1

    optimizer: str = "sgd"

    classes: Iterable[int] = [0]


class RRClArC(Adaptor):
    def __init__(self, adaptor_config: RRClArCConfig):
        self.adaptor_config = adaptor_config

    def run(self, *args, **kwargs):

        # preprocessing: collect relevance scores (LRP) and activations
        for class_id in self.adaptor_config.classes:
            run_collect_relevances_and_activations({**self.adaptor_config, 'class_id': class_id})