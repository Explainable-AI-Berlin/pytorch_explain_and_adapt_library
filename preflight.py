"""Check run configs before spending GPU time on them.

Thin wrapper. The implementation moved to ``peal.entrypoints.preflight`` so that an
installed PEAL exposes it as the ``peal-preflight`` console command; this file is kept
because the reproduction scripts, notebooks and README all invoke ``python preflight.py
...``. See that module for the full documentation.
"""

import sys

from peal.entrypoints.preflight import main

if __name__ == "__main__":
    sys.exit(main())
