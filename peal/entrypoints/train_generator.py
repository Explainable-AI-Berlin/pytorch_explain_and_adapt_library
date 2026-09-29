"""Train a PEAL generator (DDPM, diffusion autoencoder, RAE, ...) from YAML.

Calls ``train_model()`` on the generator class that
``peal.generators.generator_factory.get_generator`` picks from the config's
``generator_type``, e.g. ``DiffusionAutoencoder`` in
``peal/generators/diffusion_autoencoder.py`` or ``DDPMGenerator`` in
``peal/generators/ddpm_generator.py``.

Invocation::

    python train_generator.py --config <generator.yaml> \\
        [--continue_training true] [--is_loaded true] [--seed N] \\
        [--wandb_project peal_sae_analysis]

``--continue_training`` resumes from the checkpoints in the run directory
(it also sets ``is_loaded``); ``--seed`` overrides the config seed. A W&B run
named ``<generator_type>_<dataset_class>`` is opened when the package and an
API key are available; otherwise training proceeds without it.

Reads the generator config and the data config it references. Writes the run
directory ``base_path`` with ``config.yaml``, checkpoints and sample images in
the layout of the respective generator class.
"""

import argparse

from peal.global_utils import load_yaml_config, set_random_seed
from peal.generators.generator_factory import get_generator


def str2bool(v):
    """Parse a command-line boolean such as ``yes``/``no`` or ``1``/``0``.

    Parameters
    ----------
    v : bool or str
        The raw argument value.

    Returns
    -------
    bool
        The parsed value.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``v`` is none of the accepted spellings.
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    """Load the generator config, set up W&B if possible and train.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--is_loaded", type=str2bool, default=False)
    parser.add_argument("--continue_training", type=str2bool, default=False)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--wandb_project", type=str, default="peal_sae_analysis")
    args = parser.parse_args()

    generator_config = load_yaml_config(args.config)
    if hasattr(generator_config, "is_loaded") and args.is_loaded:
        generator_config.is_loaded = args.is_loaded

    if hasattr(args, "continue_training") and args.continue_training:
        generator_config.continue_training = args.continue_training
        generator_config.is_loaded = True

    if args.seed is not None:
        generator_config.seed = args.seed

    set_random_seed(generator_config.seed)

    # W&B Grouping
    data_cfg = (
        load_yaml_config(generator_config.data)
        if hasattr(generator_config, "data") and isinstance(generator_config.data, str)
        else getattr(generator_config, "data", None)
    )
    dataset_variant = (
        getattr(data_cfg, "dataset_class", "dataset") if data_cfg else "dataset"
    )
    group_name = f"foundation/{dataset_variant}"

    try:
        import wandb

        if wandb.run is None:
            gen_type = getattr(generator_config, "generator_type", "generator")
            run_name = f"{gen_type}_{dataset_variant}"
            wandb.init(project=args.wandb_project, name=run_name, group=group_name)
    except ImportError:
        pass
    except Exception as exc:  # e.g. no API key configured on the cluster
        print(
            f"[Train Generator] W&B disabled ({exc.__class__.__name__}); training anyway."
        )

    generator = get_generator(generator_config)
    generator.train_model()


if __name__ == "__main__":
    main()
