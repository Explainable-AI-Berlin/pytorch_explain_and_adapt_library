"""Make `pytest` work from a clone, without installing PEAL first.

The research environments run PEAL from the checkout rather than from an
installed wheel, so `import peal` only resolves when the repository root is on
`sys.path`. pytest puts the rootdir there only under some import modes; doing it
here makes `pytest`, `pytest tests/web`, and `pytest <file>` behave the same.

An installed PEAL is left alone: the entry is appended, not prepended, so a
wheel under test keeps priority over the checkout next to it.
"""

import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.append(_ROOT)
