"""Generate a synthetic confounded dataset from a ``DataConfig`` YAML.

Thin wrapper. The implementation moved to ``peal.entrypoints.generate_dataset`` so that
an installed PEAL exposes it as the ``peal-generate-dataset`` console command; this file
is kept because the reproduction scripts, notebooks and README all invoke ``python
generate_dataset.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.generate_dataset import main

if __name__ == "__main__":
    main()
