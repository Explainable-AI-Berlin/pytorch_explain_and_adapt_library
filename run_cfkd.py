import argparse

from peal.configs.adaptors.template import AdaptorConfig
from peal.utils import load_yaml_config
from peal.adaptors.counterfactual_knowledge_distillation import CounterfactualKnowledgeDistillation

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adaptor_config", type=str, required=True)
    args = parser.parse_args()

    adaptor_config = load_yaml_config(args.adaptor_config, AdaptorConfig)

    CounterfactualKnowledgeDistillation(adaptor_config).run()