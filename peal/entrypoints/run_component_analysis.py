"""Visualise every component of a sparse dictionary fitted on a generator.

Drives ``DiffusionAutoencoder.explain_all_components`` in
``peal/generators/diffusion_autoencoder.py``. The generator is loaded from
its config, the sparse dictionary (SAE, MSAE, SVD, ...) is loaded from its run
directory or fitted when no weights exist, and every component in
``[comp_min, comp_max)`` is swept over the validation split with a line
search through the decoder.

Invocation::

    python run_component_analysis.py --config <generator.yaml> \\
        [--sd_config <sparse_dictionary.yaml>] [--is_loaded true]

``--sd_config`` replaces the generator's ``sparse_dictionary`` block and, when
the dictionary config names a ``data`` config, the generator's data config as
well. ``--is_loaded`` is parsed with ``type=bool``, so any non-empty string,
including ``"false"``, is ``True``.

Writes under ``<generator base_path>/<dictionary name>/``:
``c_min_and_maxes.txt``, ``correlations.png`` (component vs. ground-truth
attribute correlations), ``samples.png`` when a latent DDPM exists, and the
per-component ``*_linearsearch.png`` grids and contrastive collages.
"""

import argparse

from peal.global_utils import load_yaml_config, set_random_seed
from peal.generators.generator_factory import get_generator


def main():
    """Load the generator (and dictionary) configs and explain all components.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--is_loaded", type=bool, default=True)
    parser.add_argument("--sd_config", type=str, default=None)
    # add_class_arguments(parser, ModelConfig)
    args = parser.parse_args()

    generator_config = load_yaml_config(args.config)
    if hasattr(generator_config, "is_loaded"):
        generator_config.is_loaded = args.is_loaded

    if not args.sd_config is None:
        sparse_dictionary_config = load_yaml_config(args.sd_config)

        # Force the generator to use this specific sparse dictionary during init
        generator_config.sparse_dictionary = sparse_dictionary_config
        # Also ensure the generator data matches the dictionary data
        if hasattr(sparse_dictionary_config, "data") and sparse_dictionary_config.data:
            from peal.data.interfaces import DataConfig

            generator_config.data = load_yaml_config(
                sparse_dictionary_config.data, DataConfig
            )

    else:
        sparse_dictionary_config = None

    set_random_seed(generator_config.seed)

    generator = get_generator(generator_config)
    generator.explain_all_components(sparse_dictionary_config)


if __name__ == "__main__":
    main()
