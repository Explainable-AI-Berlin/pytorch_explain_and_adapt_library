"""Run Counterfactual Knowledge Distillation (CFKD) from a YAML config.

Drives the ``CFKD`` adaptor in
``peal/adaptors/counterfactual_knowledge_distillation.py``: load the student
predictor, explain it with counterfactuals from the configured generator and
explainer, collect feedback (oracle, human or SAE based) on the
counterfactuals and finetune the student on them, for ``finetune_iterations``
rounds.

Invocation::

    python run_cfkd.py --config <adaptor.yaml> [--<any CFKDConfig field> value]

Every field of ``CFKDConfig`` is exposed as an optional ``--flag`` that
overrides the YAML value (``add_class_arguments`` / ``integrate_arguments``).
Set ``PEAL_CUDA_SYNC=1`` to make CUDA launches synchronous while debugging a
kernel fault.

Reads the adaptor config and the predictor, data, generator and explainer
configs it references. Writes into ``base_dir``: ``config.yaml``, ``logs/``
(TensorBoard), ``platform.txt``, one ``<iteration>/`` directory per round with
the explainer collages and the ``train_dataset`` / ``validation_dataset`` of
counterfactuals, and the finetuned student as ``model.cpl``.
"""

import argparse
import os

# Synchronous CUDA launches make every kernel error point at its real call site,
# at the cost of serializing every kernel launch. Keep it opt-in: run with
# PEAL_CUDA_SYNC=1 when debugging a CUDA fault.
if os.environ.get("PEAL_CUDA_SYNC", "0") == "1":
    os.environ["TORCH_USE_CUDA_DSA"] = "1"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from peal.adaptors.counterfactual_knowledge_distillation import CFKDConfig
from peal.global_utils import (
    load_yaml_config,
    add_class_arguments,
    integrate_arguments,
    set_random_seed,
)
from peal.adaptors.counterfactual_knowledge_distillation import (
    CFKD,
)


def main():
    """Load the ``CFKDConfig``, apply command-line overrides and run CFKD.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    add_class_arguments(parser, CFKDConfig)
    args = parser.parse_args()
    adaptor_config = load_yaml_config(args.config, CFKDConfig)
    integrate_arguments(args, adaptor_config, exclude=["config"])
    if not adaptor_config.seed is None:
        set_random_seed(adaptor_config.seed)

    cfkd = CFKD(adaptor_config=adaptor_config)
    cfkd.run()


if __name__ == "__main__":
    main()
