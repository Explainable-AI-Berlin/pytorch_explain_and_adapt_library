"""Compatibility shim for the vendored PathLDM, loaded via PYTHONPATH.

`peal/dependencies/PathLDM/ldm/models/diffusion/ddpm.py` and `main.py` do

    from pytorch_lightning.utilities.distributed import rank_zero_only

which was valid in PyTorch Lightning 1.x. PL 2.x removed that module and moved
the symbols to `pytorch_lightning.utilities.rank_zero`.

A parallel `pytorch_lightning/utilities/distributed.py` directory on PYTHONPATH
does NOT work: pytorch_lightning is a regular installed package, so Python
resolves the package from site-packages and never consults the shadow tree.
Registering the module in sys.modules before anything imports it does work, and
`site` imports `sitecustomize` automatically at interpreter startup, so this
takes effect for any process run with this directory on PYTHONPATH and leaves
both site-packages and the vendored source untouched.
"""

import sys
import types

try:
    from pytorch_lightning.utilities import rank_zero as _rz
except Exception:  # pragma: no cover - PL absent or restructured again
    _rz = None

if _rz is not None and "pytorch_lightning.utilities.distributed" not in sys.modules:
    _mod = types.ModuleType("pytorch_lightning.utilities.distributed")
    for _name in (
        "rank_zero_debug",
        "rank_zero_deprecation",
        "rank_zero_info",
        "rank_zero_only",
        "rank_zero_warn",
    ):
        if hasattr(_rz, _name):
            setattr(_mod, _name, getattr(_rz, _name))
    sys.modules["pytorch_lightning.utilities.distributed"] = _mod
