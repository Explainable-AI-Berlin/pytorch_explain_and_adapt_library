"""Run any PEAL adaptor from its YAML config.

Generic entry point for the model-correction adaptors in ``peal/adaptors``:
the ``adaptor_type`` field of the config selects the class (``CFKD``,
``DiDAE``, ``ClArC``, ``GroupDRO``, ``GroupDROv2`` or
``ProjectionAdaptor``) and ``peal.adaptors.adaptor_factory.get_adaptor``
instantiates it. ``run_cfkd.py`` and ``run_didae.py`` do the same for one
adaptor each, with per-field command-line overrides.

Reads the adaptor config and the predictor, data, generator and explainer
configs it references. Writes whatever the adaptor produces into its
``base_dir``: ``config.yaml``, logs, per-iteration counterfactual datasets and
the corrected ``model.cpl``.

Example::

    python run_adaptor.py --config configs/<experiment>/adaptors/<run>.yaml \\
        [--seed 0]

``--seed`` overrides the config seed and propagates it to the nested configs.
"""

import argparse

from peal.global_utils import load_yaml_config, propagate_seed, set_random_seed
from peal.adaptors.adaptor_factory import get_adaptor


def main():
    """Parse ``--config`` / ``--seed``, build the adaptor and call ``run()``.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    adaptor_config = load_yaml_config(args.config)
    if args.seed is not None:
        adaptor_config.seed = args.seed
        propagate_seed(adaptor_config)
    set_random_seed(adaptor_config.seed)

    adaptor = get_adaptor(adaptor_config)
    adaptor.run()


if __name__ == "__main__":
    main()
