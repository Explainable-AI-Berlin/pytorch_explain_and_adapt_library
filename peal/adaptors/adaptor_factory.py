"""Factory that resolves an adaptor specification into an ``Adaptor`` instance.

Adaptors are the repair methods of PEAL (CFKD, DiDAE, ClArC, GroupDRO, ...).
``get_adaptor`` accepts an already-built adaptor, a path to a yaml config or a
plain config dict, discovers every ``Adaptor`` subclass under ``peal/adaptors``
and instantiates the one named by the ``adaptor_type`` config key.
"""

import os

from typing import Union

from peal.adaptors.interfaces import Adaptor
from peal.registry import lookup
from peal.global_utils import (
    load_yaml_config,
    get_project_resource_dir,
)


def get_adaptor(
    adaptor: Union[Adaptor, str, dict],
) -> Adaptor:
    """Return an ``Adaptor`` built from an instance, a yaml path or a config dict.

    Parameters
    ----------
    adaptor : Adaptor or str or dict
        Either a ready ``Adaptor`` (returned unchanged), a path to a yaml file
        or a dict-like config. Configs are normalised with ``load_yaml_config``
        and must carry an ``adaptor_type`` key naming an ``Adaptor`` subclass
        found by scanning ``peal/adaptors``. The class is called as
        ``cls(adaptor_config=config)``.

    Returns
    -------
    Adaptor
        The resolved adaptor.

    Raises
    ------
    peal.registry.UnknownComponentError
        If a config is given whose ``adaptor_type`` is neither registered in
        ``peal.registry.ADAPTORS`` nor discoverable under ``peal/adaptors``.
    """
    if isinstance(adaptor, Adaptor):
        adaptor_out = adaptor

    else:
        adaptor_config = load_yaml_config(adaptor)
        adaptor_class = lookup(
            "adaptors",
            getattr(adaptor_config, "adaptor_type", None),
            base_class=Adaptor,
            scan_dir=os.path.join(get_project_resource_dir(), "peal", "adaptors"),
        )
        adaptor_out = adaptor_class(adaptor_config=adaptor_config)

    return adaptor_out
