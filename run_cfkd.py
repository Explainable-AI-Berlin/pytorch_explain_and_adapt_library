import argparse

from peal.configs.adaptors.cfkd_config import CFKDConfig
from peal.global_utils import load_yaml_config, add_class_arguments, integrate_arguments
from peal.adaptors.counterfactual_knowledge_distillation import (
    CounterfactualKnowledgeDistillation,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptor_config", type=str, required=True)
    add_class_arguments(parser, CFKDConfig)
    args = parser.parse_args()

    adaptor_config = load_yaml_config(args.adaptor_config, CFKDConfig)
    integrate_arguments(args, adaptor_config, exclude=["adaptor_config"])

    cfkd = CounterfactualKnowledgeDistillation(adaptor_config=adaptor_config)
    cfkd.run()


main()
