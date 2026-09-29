"""DiDAE: Dictionary-based Interpretable Diffusion Autoencoder Explanations.

Thin wrapper. The implementation moved to ``peal.entrypoints.run_didae`` so that an
installed PEAL exposes it as the ``peal-didae`` console command; this file is kept
because the reproduction scripts, notebooks and README all invoke ``python run_didae.py
...``. See that module for the full documentation.
"""

from peal.entrypoints.run_didae import main

if __name__ == "__main__":
    main()
