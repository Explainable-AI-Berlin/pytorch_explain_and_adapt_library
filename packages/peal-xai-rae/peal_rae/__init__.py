"""peal-xai-rae: the RAEv2 fork behind PEAL's RAE generators. NON-COMMERCIAL.

This package is licensed CC BY-NC 4.0, not LGPL like PEAL itself: it carries a
modified copy of RAEv2 (Singh, Zheng, Wu, Zhang, Shechtman, Xie; *Improved
Baselines with Representation Autoencoders*, https://github.com/nanovisionx/RAEv2),
which its authors released under that licence. See ``RAEv2/LICENSE`` and
``RAEv2/MODIFICATIONS.md``.

PEAL never imports this package's code as ``peal_rae.*``. RAEv2 is organised as
top-level modules (``stage1``, ``stage2``, ``utils``, ``data`` ...) that would
clash with other libraries if installed as such, so they ship as package data
under ``RAEv2/src`` and PEAL's ``RAEDiffusionAutoencoder`` puts that folder on
``sys.path`` when an RAE generator is built. The only thing PEAL asks of this
package is :func:`raev2_dir`.

The ImageNet weights are not in the wheel (6.6 GB); PEAL downloads them from
``hf://sidney1505/peal-rae-clip-imagenet`` (also CC BY-NC 4.0) on first use.
"""

import os

__version__ = "0.1.0"

#: Upstream RAEv2 commit this fork is based on.
UPSTREAM_REPO = "https://github.com/nanovisionx/RAEv2"
UPSTREAM_COMMIT = "8a0d238f8dc3b261aba98b217f6c79c0182e8e94"
LICENCE = "CC BY-NC 4.0"


def raev2_dir():
    """Root of the bundled RAEv2 checkout (holds ``src/``, ``configs/``, ``scripts/``).

    Returns
    -------
    str
        Absolute path to the ``RAEv2`` folder inside this package.
    """
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "RAEv2")
