"""Distil a trained predictor into a smoother surrogate for counterfactuals.

Thin wrapper. The implementation moved to ``peal.entrypoints.train_distilled_predictor``
so that an installed PEAL exposes it as the ``peal-distill-predictor`` console command;
this file is kept because the reproduction scripts, notebooks and README all invoke
``python train_distilled_predictor.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.train_distilled_predictor import main

if __name__ == "__main__":
    main()
