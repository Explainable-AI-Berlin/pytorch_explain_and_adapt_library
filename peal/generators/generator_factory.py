"""Resolve a generator specification into a live generator object.

PEAL configs name their generator either as a pickled ``.cpl`` object, as a
path to (or dict of) a generator config, or hand over an already constructed
generator. :func:`get_generator` accepts all of these, instantiates the matching
``Generator`` subclass found under ``peal/generators`` by its ``generator_type``
key, and returns it in eval mode on the requested device.
"""

import torch
import os

from typing import Union

from peal.generators.interfaces import (
    InvertibleGenerator,
    EditCapableGenerator,
    Generator,
)
from peal.registry import lookup
from peal.global_utils import (
    load_yaml_config,
    get_project_resource_dir,
)


def get_generator(
    generator: Union[InvertibleGenerator, str, dict],
    device: Union[str, torch.device] = "cuda",
    predictor_dataset=None,
    timestep_respacing: int = None,
) -> InvertibleGenerator:
    """Build or load a generator from a path, config or existing instance.

    Three input forms are handled. A string ending in ``.cpl`` is loaded with
    ``torch.load`` (falling back to ``weights_only=False``). A config path or
    dict is loaded with ``load_yaml_config``; its ``generator_type`` key selects
    the ``Generator`` subclass found under ``peal/generators``, and when the
    config came from a file its directory (after expanding ``$PEAL_RUNS``,
    ``$PEAL_DATA`` and ``<PEAL_BASE>``) becomes ``config.base_path`` so the
    generator can find its weights. An ``InvertibleGenerator``,
    ``EditCapableGenerator`` or ``None`` is passed through unchanged.

    Parameters
    ----------
    generator : InvertibleGenerator, EditCapableGenerator, str, dict or None
        The generator instance, ``.cpl`` path, config path or config dict.
    device : str or torch.device, optional
        Device the generator is moved to. Default ``"cuda"``.
    predictor_dataset : optional
        Dataset of the predictor being explained; forwarded to the generator
        constructor so it can align normalisation and resolution.
    timestep_respacing : int, optional
        When given and the config has a ``timestep_respacing`` field, it
        overrides the number of diffusion sampling steps.

    Returns
    -------
    InvertibleGenerator or None
        The generator in ``eval()`` mode on ``device``, or ``None`` when
        ``generator`` was ``None``.

    Raises
    ------
    peal.registry.UnknownComponentError
        When a config is given whose ``generator_type`` is neither registered
        nor discoverable under ``peal/generators``.
    """
    if isinstance(generator, str) and generator[-4:] == ".cpl":
        try:
            generator_out = torch.load(generator, map_location=device)
        except Exception:
            generator_out = torch.load(
                generator, map_location=device, weights_only=False
            )

    elif not (
        isinstance(generator, InvertibleGenerator)
        or isinstance(generator, EditCapableGenerator)
        or generator is None
    ):
        generator_config = load_yaml_config(generator)
        if (
            hasattr(generator_config, "timestep_respacing")
            and not timestep_respacing is None
        ):
            generator_config.timestep_respacing = str(timestep_respacing)

        generator_class = lookup(
            "generators",
            getattr(generator_config, "generator_type", None),
            base_class=Generator,
            scan_dir=os.path.join(get_project_resource_dir(), "peal", "generators"),
        )
        if True:  # keeps the indentation of the base_path block below unchanged
            if isinstance(generator, str):
                resolved_generator = generator
                if resolved_generator.startswith("$PEAL_RUNS"):
                    resolved_generator = resolved_generator.replace(
                        "$PEAL_RUNS", os.environ.get("PEAL_RUNS", "peal_runs")
                    )
                if resolved_generator.startswith("$PEAL_DATA"):
                    resolved_generator = resolved_generator.replace(
                        "$PEAL_DATA", os.environ.get("PEAL_DATA", "datasets")
                    )
                if "<PEAL_BASE>" in resolved_generator:
                    resolved_generator = resolved_generator.replace(
                        "<PEAL_BASE>", get_project_resource_dir()
                    )
                generator_config.base_path = os.path.dirname(resolved_generator)

            generator_out = generator_class(
                config=generator_config,
                device=device,
                predictor_dataset=predictor_dataset,
            )

    else:
        generator_out = generator

    if not generator_out is None:
        generator_out.eval().to(device)

    return generator_out
