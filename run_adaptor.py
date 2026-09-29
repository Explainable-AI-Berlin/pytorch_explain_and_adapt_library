"""Run any PEAL adaptor from its YAML config.

Thin wrapper. The implementation moved to ``peal.entrypoints.run_adaptor`` so that an
installed PEAL exposes it as the ``peal-adapt`` console command; this file is kept
because the reproduction scripts, notebooks and README all invoke ``python
run_adaptor.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.run_adaptor import main

if __name__ == "__main__":
    main()
