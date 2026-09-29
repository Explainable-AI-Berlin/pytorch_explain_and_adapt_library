"""Factory that resolves an explainer specification into an ``ExplainerInterface``.

Explainers produce the counterfactual (or other) explanations that PEAL's
teachers judge and its adaptors learn from. ``get_explainer`` maps the legacy
counterfactual ``explainer_type`` names (``SCE``, ``ACE``, ``TIME``,
``DAEdistill``) onto ``CounterfactualExplainer`` and otherwise looks the type up
among the ``ExplainerInterface`` subclasses found under ``peal/explainers``.
"""

import torch
import os

from typing import Union

from peal.explainers.counterfactual_explainer import CounterfactualExplainer
from peal.explainers.interfaces import (
    ExplainerInterface,
)
from peal.registry import lookup
from peal.global_utils import (
    load_yaml_config,
    get_project_resource_dir,
)


def get_explainer(
    explainer: Union[ExplainerInterface, str, dict],
    device: Union[str, torch.device] = "cuda",
    predictor_datasets=None,
    **kwargs,
) -> ExplainerInterface:
    """Return an explainer built from an instance, a yaml path or a config dict.

    Parameters
    ----------
    explainer : ExplainerInterface or str or dict
        A ready explainer (returned unchanged), a yaml path or a dict-like
        config. The config's ``explainer_type`` selects the class: the values
        ``SCE``, ``ACE``, ``TIME`` and ``DAEdistill`` all instantiate
        ``CounterfactualExplainer(explainer_config=..., **kwargs)``; any other
        value must be the class name of an ``ExplainerInterface`` subclass
        discovered under ``peal/explainers``, which is then called as
        ``cls(config=..., device=..., predictor_dataset=predictor_datasets,
        **kwargs)``.
    device : str or torch.device, optional
        Device handed to non-counterfactual explainers. Default ``"cuda"``.
    predictor_datasets : optional
        Dataset(s) of the predictor, forwarded as ``predictor_dataset``.
    **kwargs
        Extra constructor arguments forwarded to the explainer class.

    Returns
    -------
    ExplainerInterface
        The resolved explainer.

    Raises
    ------
    peal.registry.UnknownComponentError
        If the config's ``explainer_type`` matches neither the legacy names nor
        a registered or discoverable subclass.
    """
    if not isinstance(explainer, ExplainerInterface):
        explainer_config = load_yaml_config(explainer)
        explainer_type = getattr(explainer_config, "explainer_type", None)
        if explainer_type in ["SCE", "ACE", "TIME", "DAEdistill"]:
            explainer_out = CounterfactualExplainer(
                explainer_config=explainer_config, **kwargs
            )

        else:
            explainer_class = lookup(
                "explainers",
                explainer_type,
                base_class=ExplainerInterface,
                scan_dir=os.path.join(get_project_resource_dir(), "peal", "explainers"),
            )
            explainer_out = explainer_class(
                config=explainer_config,
                device=device,
                predictor_dataset=predictor_datasets,
                **kwargs,
            )

    else:
        explainer_out = explainer

    return explainer_out
