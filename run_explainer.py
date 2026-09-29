"""Explain a trained predictor with a stand-alone explainer run.

Thin wrapper. The implementation moved to ``peal.entrypoints.run_explainer`` so that an
installed PEAL exposes it as the ``peal-explain`` console command; this file is kept
because the reproduction scripts, notebooks and README all invoke ``python
run_explainer.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.run_explainer import main

if __name__ == "__main__":
    main()
