"""Repo root without importing peal.global_utils (which pulls in transformers,
tensorflow and friends): the web process only needs the path."""

import os

import peal


def get_project_resource_dir():
    """Return the absolute path of the repository root.

    The root is resolved as the parent directory of the ``peal`` package, which
    is where ``configs/``, ``peal_runs/`` and the web templates live.

    Returns
    -------
    str
        Absolute path of the directory containing the ``peal`` package.
    """
    return os.path.abspath(os.path.join(os.path.dirname(peal.__file__), ".."))
