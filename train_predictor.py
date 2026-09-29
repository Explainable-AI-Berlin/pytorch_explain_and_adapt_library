"""Train a predictor (student or test model) from a ``PredictorConfig`` YAML.

Thin wrapper. The implementation moved to ``peal.entrypoints.train_predictor`` so that
an installed PEAL exposes it as the ``peal-train-predictor`` console command; this file
is kept because the reproduction scripts, notebooks and README all invoke ``python
train_predictor.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.train_predictor import main

if __name__ == "__main__":
    main()
