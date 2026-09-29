"""Visualise every component of a sparse dictionary fitted on a generator.

Thin wrapper. The implementation moved to ``peal.entrypoints.run_component_analysis`` so
that an installed PEAL exposes it as the ``peal-component-analysis`` console command;
this file is kept because the reproduction scripts, notebooks and README all invoke
``python run_component_analysis.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.run_component_analysis import main

if __name__ == "__main__":
    main()
