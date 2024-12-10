import argparse
import os
os.environ['TORCH_USE_CUDA_DSA']="1"
os.environ['CUDA_LAUNCH_BLOCKING']="1"

from peal.adaptors.counterfactual_knowledge_distillation import CFKDConfig
from peal.global_utils import load_yaml_config, add_class_arguments, integrate_arguments, set_random_seed
from peal.adaptors.counterfactual_knowledge_distillation import (
    CFKD,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    add_class_arguments(parser, CFKDConfig)
    args = parser.parse_args()

    adaptor_config = load_yaml_config(args.config, CFKDConfig)
    integrate_arguments(args, adaptor_config, exclude=["config"])
    set_random_seed(adaptor_config.seed)

    cfkd = CFKD(adaptor_config=adaptor_config)
    cfkd.run()


main()
