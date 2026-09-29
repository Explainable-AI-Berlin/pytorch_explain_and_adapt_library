"""Fit or load a sparse dictionary on a generator's latents and evaluate it.

Drives the sparse-dictionary machinery of the diffusion autoencoder
generator (``load_sparse_dictionary`` / ``fit_sparse_dictionary`` /
``explain_sae`` in ``peal/generators/diffusion_autoencoder.py``, dictionary
classes in ``peal/sparse_dictionaries``) and the ground-truth evaluation in
``peal/sparse_dictionaries/sae_evaluation.py``.

Invocation::

    python run_sae_analysis.py --config <generator.yaml> \\
        [--sd_config <sparse_dictionary.yaml>] [--run_name NAME] \\
        [--top_k K] [--n_components N] [--lr LR] [--explain_components] \\
        [--wandb_project peal_sae_analysis]

Without ``--explain_components`` the dictionary is only loaded, or fitted
when no weights exist. With it, the per-component counterfactual search over
the whole dataset runs as well, which takes hours to days. ``--top_k``,
``--n_components`` and ``--lr`` override the dictionary config. Unknown flags
are ignored, and ``--is_loaded`` is parsed with ``type=bool``, so any
non-empty string is ``True``.

Reads the generator config, its data config and the checkpoints they name.
The dictionary is scored on the validation split, with the test split as a
held-out check, and the results go into the dictionary's ``base_path``: the
fitted weights, ``sae_evaluation_report.txt``, ``sae_evaluation_results.pkl``,
``matching_matrix_diagonalized.png`` and TensorBoard logs in ``sae_logs/``.
The same numbers are logged to W&B when an API key is configured.
"""

import argparse
import os

from peal.global_utils import (
    load_yaml_config,
    set_random_seed,
)
from peal.generators.generator_factory import get_generator
from peal.sparse_dictionaries.sae_evaluation import (
    CELEBA_WELL_DEFINED_20,
    run_sae_eval,
    log_evaluation_to_wandb,
    log_evaluation_to_tensorboard,
)


def main():
    """Fit/load the dictionary, optionally explain it, then evaluate and log.

    Returns
    -------
    None
    """
    parser = argparse.ArgumentParser(
        description="Unified SAE Analysis and Training Framework"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to generator / model config yaml",
    )
    parser.add_argument(
        "--sd_config",
        type=str,
        default=None,
        help="Path to sparse dictionary config yaml",
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Custom run name for SAE analysis"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="peal_sae_analysis",
        help="W&B project name",
    )
    parser.add_argument("--is_loaded", type=bool, default=True)
    parser.add_argument(
        "--explain_components",
        action="store_true",
        help="Also run the per-component counterfactual search after fitting/evaluating the SAE. This walks the full dataset for every latent via the diffusion model and can take hours to days for a large dictionary — off by default.",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=None,
        help="Override the sparse dictionary config's top_k",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=None,
        help="Override the sparse dictionary config's n_components (and dict_size, if set)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override the sparse dictionary config's learning rate",
    )

    args, unknown = parser.parse_known_args()

    generator_config = load_yaml_config(args.config)
    if hasattr(generator_config, "is_loaded"):
        generator_config.is_loaded = args.is_loaded

    if args.sd_config is not None:
        sparse_dictionary_config = load_yaml_config(args.sd_config)
        generator_config.sparse_dictionary = sparse_dictionary_config
        if hasattr(sparse_dictionary_config, "data") and sparse_dictionary_config.data:
            generator_config.data = sparse_dictionary_config.data
    else:
        sparse_dictionary_config = getattr(generator_config, "sparse_dictionary", None)

    if args.run_name:
        if sparse_dictionary_config:
            sparse_dictionary_config.run_name = args.run_name
            sparse_dictionary_config.name = args.run_name

    if sparse_dictionary_config:
        if args.top_k is not None:
            sparse_dictionary_config.top_k = args.top_k
        if args.n_components is not None:
            sparse_dictionary_config.n_components = args.n_components
            if getattr(sparse_dictionary_config, "dict_size", None) is not None:
                sparse_dictionary_config.dict_size = args.n_components
        if args.lr is not None:
            sparse_dictionary_config.lr = args.lr

    set_random_seed(getattr(generator_config, "seed", 42))

    generator = get_generator(generator_config)

    print(
        f"[SAE Analysis] Fitting / Loading SAE: {sparse_dictionary_config.sparse_dictionaries_type if sparse_dictionary_config else 'default'}"
    )
    if args.explain_components:
        generator.explain_sae(sparse_dictionary_config)
    else:
        # Fit/load only — skip find_counterfactual's per-component search, which
        # walks the full dataset through the diffusion model for every latent
        # and is impractical to run to completion (hours to days).
        generator.config.sparse_dictionary = sparse_dictionary_config
        generator.config.sparse_dictionary.act_size = (
            generator.config.encoder_dimensions
        )
        generator.load_sparse_dictionary()
        if generator.sparse_dictionary is None:
            generator.fit_sparse_dictionary()
        if (
            hasattr(generator.sparse_dictionary, "sae")
            and generator.sparse_dictionary.sae is not None
        ):
            generator.sparse_dictionary.sae.eval()

    # Perform ground-truth evaluation & W&B logging
    sd = generator.sparse_dictionary
    if sd is not None:
        val_dataset = (
            generator.generator_datasets[1]
            if hasattr(generator, "generator_datasets")
            and len(generator.generator_datasets) > 1
            else None
        )

        # The SAE is fitted on splits 0 and 1, so split 2 is unseen by both the
        # dictionary and the latent->label matching. It is scored as a held-out
        # check that the matching and thresholds picked on the validation split
        # are not just fitted to it.
        test_dataset = (
            generator.generator_datasets[2]
            if hasattr(generator, "generator_datasets")
            and len(generator.generator_datasets) > 2
            else None
        )

        if val_dataset is not None:
            print(
                "[SAE Analysis] Extracting validation features and ground-truth attributes..."
            )
            try:
                feature_extractor = generator.model.ema_model.encoder
            except AttributeError:
                feature_extractor = generator.encoder

            from peal.sparse_dictionaries import activation_cache

            if activation_cache.enabled():
                # One cache file holds all three splits, shared with the fit above.
                acts = activation_cache.load_or_extract(
                    generator,
                    list(generator.generator_datasets),
                    generator.config.data,
                    feature_extractor,
                    generator.device,
                )
                X, Y = acts[1]
                X_test, Y_test = acts[2] if len(acts) > 2 else (None, None)
            else:
                wanted = [val_dataset] + (
                    [test_dataset] if test_dataset is not None else []
                )
                acts = activation_cache.extract(
                    wanted, feature_extractor, generator.device
                )
                X, Y = acts[0]
                X_test, Y_test = acts[1] if len(acts) > 1 else (None, None)

            if Y is not None:

                # Dynamically retrieve attribute names from dataset
                attribute_names = getattr(val_dataset, "attributes", None)
                if attribute_names is None:
                    attribute_names = [f"Attr_{i}" for i in range(Y.shape[1])]

                # Derive dataset variant name for W&B grouping
                data_cfg = getattr(generator_config, "data", None)
                dataset_variant = (
                    getattr(data_cfg, "dataset_class", "dataset")
                    if data_cfg
                    else "dataset"
                )
                sae_type = (
                    getattr(sparse_dictionary_config, "sparse_dictionaries_type", "SAE")
                    if sparse_dictionary_config
                    else "SAE"
                )
                group_name = f"sae/{dataset_variant}/{sae_type}"

                run_name = (
                    args.run_name if args.run_name else f"{dataset_variant}_{sae_type}"
                )

                base_path = (
                    getattr(
                        sparse_dictionary_config,
                        "base_path",
                        generator.config.base_path,
                    )
                    if sparse_dictionary_config
                    else generator.config.base_path
                )
                os.makedirs(base_path, exist_ok=True)

                # Only a subset of the 40 CelebA attributes is annotated
                # consistently enough for a macro F1 over them to mean anything
                # (arXiv:2210.07356, Table 1) — aggregate over those separately.
                subset_names = [
                    n for n in CELEBA_WELL_DEFINED_20 if n in list(attribute_names)
                ]

                eval_results = run_sae_eval(
                    sae=sd,
                    x=X,
                    y=Y,
                    base_path=base_path,
                    label_names=attribute_names,
                    x_holdout=X_test,
                    y_holdout=Y_test,
                    subset_names=subset_names or None,
                    subset_key="welldef20",
                )

                log_evaluation_to_wandb(
                    results=eval_results,
                    run_name=run_name,
                    group=group_name,
                    project=args.wandb_project,
                )

                log_evaluation_to_tensorboard(
                    results=eval_results,
                    log_dir=os.path.join(base_path, "sae_logs"),
                )


if __name__ == "__main__":
    main()
