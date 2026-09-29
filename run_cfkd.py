"""Run Counterfactual Knowledge Distillation (CFKD) from a YAML config.

Thin wrapper. The implementation moved to ``peal.entrypoints.run_cfkd`` so that an
installed PEAL exposes it as the ``peal-cfkd`` console command; this file is kept
because the reproduction scripts, notebooks and README all invoke ``python run_cfkd.py
...``. See that module for the full documentation.
"""

from peal.entrypoints.run_cfkd import main

if __name__ == "__main__":
    main()
