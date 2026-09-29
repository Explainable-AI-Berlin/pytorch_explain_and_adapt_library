"""Print the accuracy and group accuracies of a trained predictor.

Thin wrapper. The implementation moved to ``peal.entrypoints.evaluate_predictor`` so
that an installed PEAL exposes it as the ``peal-evaluate-predictor`` console command;
this file is kept because the reproduction scripts, notebooks and README all invoke
``python evaluate_predictor.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.evaluate_predictor import main

if __name__ == "__main__":
    main()
