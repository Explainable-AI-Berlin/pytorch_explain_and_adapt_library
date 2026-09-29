<!-- Provenance record for a vendored third-party component of PEAL.
     Generated 2026-09-22; keep it updated when the copy is refreshed. -->

# PathLDM — Text-conditioned Latent Diffusion for Histopathology

**This directory is not part of PEAL.** It is a vendored copy of third-party
research code, kept in `peal/dependencies/` and separate from PEAL's own
implementation. All credit for the method and the original implementation
belongs to the authors below; please cite their work, not PEAL, when you use it.

| | |
|---|---|
| **Original authors** | Srikar Yellapragada, Alexandros Graikos, Prateek Prasanna, Tahsin Kurc, Joel Saltz, Dimitris Samaras (Stony Brook University) |
| **Reference** | Yellapragada et al., *PathLDM: Text Conditioned Latent Diffusion Model for Histopathology*, WACV 2024 |
| **Upstream repository** | <https://github.com/cvlab-stonybrook/PathLDM> |
| **Compared against** | `a717d0388b4bac624f4d995ab91e7b1752b7c7c5` on branch `main` (2024-07-07) |
| **Upstream license** | MIT |

## Why PEAL vendors it

The Camelyon17 decoder. `peal/generators/pathldm_autoencoder.py` and `peal/generators/ddpm_pathldm.py` import `ldm.util.instantiate_from_config` and `ldm.models.diffusion.ddim.DDIMSampler`; the checkpoint config instantiates the rest of the `ldm` package at runtime.

Upstream publishes research code, not an installable package, and the copy here
needed changes to run inside PEAL's pipeline and to keep baseline comparisons
fair. That is why it is forked rather than declared as a dependency.

## How this copy differs from upstream

Measured against upstream `a717d0388b` (2024-07-07): **18 of 33 Python files are
byte-identical**, 1 differs only in formatting (Black; identical syntax trees),
and 14 carry functional changes.

- 14 files under `ldm/` carry functional changes, chiefly `models/diffusion/ddpm.py`, `models/autoencoder.py` and `modules/diffusionmodules/{model,openaimodel}.py`, to condition the model on a frozen PLIP embedding instead of a text prompt and to expose the sampler to PEAL's inversion code.
- One file differs only in formatting; 18 files are byte-identical.

Any differences listed here are ours, assuming upstream has not changed since
the copy was taken.

## Note

**Licensing: MIT.** Upstream now carries an MIT licence, and the PathLDM authors
additionally granted permission to redistribute this copy as part of PEAL with
attribution.

Upstream's MIT `LICENSE` is now present verbatim in this directory, fetched from
the upstream repository on 2026-09-26.

Pretrained PathLDM weights are not redistributed here.
