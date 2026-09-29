"""Fit or load a sparse dictionary on a generator's latents and evaluate it.

Thin wrapper. The implementation moved to ``peal.entrypoints.run_sae_analysis`` so that
an installed PEAL exposes it as the ``peal-sae-analysis`` console command; this file is
kept because the reproduction scripts, notebooks and README all invoke ``python
run_sae_analysis.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.run_sae_analysis import main

if __name__ == "__main__":
    main()
