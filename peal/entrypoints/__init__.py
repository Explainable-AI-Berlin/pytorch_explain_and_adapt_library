"""Command-line entry points of PEAL.

This package holds the modules behind PEAL's commands: running the adaptors
(``run_cfkd``, ``run_didae``, ``run_adaptor``), the explainers
(``run_explainer``), the training and evaluation scripts
(``train_generator``, ``train_predictor``, ``train_distilled_predictor``,
``evaluate_predictor``), the sparse-dictionary analyses
(``run_sae_analysis``, ``run_component_analysis``), the dataset generator
(``generate_dataset``) and the config check (``preflight``).

Each module keeps a ``main()`` that takes no arguments and parses ``sys.argv``
itself, so it can be exposed as a ``console_scripts`` entry point of an
installed PEAL.

The files of the same name in the repository root are thin wrappers that
import ``main`` from here. They are kept because the reproduction scripts, the
notebooks and the README all invoke the tools as ``python <script>.py ...``.
"""
