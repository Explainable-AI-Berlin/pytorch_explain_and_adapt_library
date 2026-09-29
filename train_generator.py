"""Train a PEAL generator (DDPM, diffusion autoencoder, RAE, ...) from YAML.

Thin wrapper. The implementation moved to ``peal.entrypoints.train_generator`` so that
an installed PEAL exposes it as the ``peal-train-generator`` console command; this file
is kept because the reproduction scripts, notebooks and README all invoke ``python
train_generator.py ...``. See that module for the full documentation.
"""

from peal.entrypoints.train_generator import main

if __name__ == "__main__":
    main()
