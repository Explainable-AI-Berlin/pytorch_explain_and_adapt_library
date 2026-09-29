"""Editors: latent-inversion recipes applied to an existing image.

An editor describes how an image is inverted into the noise and latents a
sampler can re-run, so that an edit can be decoded from them.
``interfaces.EditorConfig`` is the pydantic base carrying the ``editor_type``
key the yaml loader dispatches on; ``ddpm_inversion`` configures the
edit-friendly DDPM inversion (model id, CFG scales, skip steps, attention
mixing).
"""
