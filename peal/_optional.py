"""Lazy access to optional third-party dependencies.

PEAL ships a small core dependency set; heavier or harder-to-install
packages (``pygame``, ``flask``, ``wilds``, ``mpi4py``, ``corelay``,
``virelay``, ``crp``, ``h5py``, ``clip``, ...) are declared as installation
*extras* instead. Modules that need such a package import it from inside the
function that uses it via :func:`require`, so importing :mod:`peal` never
fails because an extra is missing. The error is raised only when the
corresponding feature is actually used, and it then names the exact
``pip install`` command that fixes it.
"""

import importlib

__all__ = ["require"]


def require(module, extra, purpose):
    """
    Import an optional third-party module, or explain how to install it.

    Parameters
    ----------
    module : str
        Importable name of the optional module, for example ``"pygame"`` or
        ``"flask_cors"``.
    extra : str
        Name of the PEAL installation extra that provides ``module``, for
        example ``"web"``. It is used to build the install hint
        ``pip install peal-xai[<extra>]``.
    purpose : str
        Short human-readable description of what the module is needed for,
        for example ``"rasterizing the elliptical mask"``. It is embedded in
        the error message.

    Returns
    -------
    module
        The imported module object.

    Raises
    ------
    ImportError
        If ``module`` cannot be imported. The message names the missing
        module, ``purpose`` and the ``pip install peal-xai[<extra>]``
        command that installs it. The original :class:`ImportError` is
        chained as the cause.

    Examples
    --------
    >>> from peal._optional import require
    >>> pygame = require(
    ...     "pygame", "pygame", "rasterizing the circular cut"
    ... )  # doctest: +SKIP
    """
    try:
        return importlib.import_module(module)
    except ImportError as exc:
        raise ImportError(
            f"PEAL needs the optional module '{module}' for {purpose}, "
            f"but it is not installed. Install it with: "
            f"pip install peal-xai[{extra}]"
        ) from exc
