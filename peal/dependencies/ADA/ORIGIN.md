# ADA — separate research project, vendored here

`ADA` (*Actionable Data Atlases*) is a research codebase maintained separately
from PEAL and vendored because PEAL's ImageNet and NICO++ generators use the
representation autoencoder that ships with it
(`third_party/RAEv2`, reached from `peal/generators/rae_pipeline.py`,
`rae_diffusion_autoencoder.py`, `global_utils.py` and ~21 configs).

Its own third-party declarations are in `THIRD_PARTY.md`, which records:

> **RAEv2** (Singh, Zheng, Wu, Zhang, Shechtman, Xie; *Improved Baselines
> with Representation Autoencoders*): `https://github.com/nanovisionx/RAEv2`, pinned at
> `8a0d238f8dc3b261aba98b217f6c79c0182e8e94`. Upstream and derivative
> integration code are governed by **CC BY-NC 4.0**.

**ADA itself: CC BY-NC 4.0, written by David Drexlin** (`LICENSE` in this
folder). It is kept out of the installed `peal-xai` package so that the package
stays LGPL-only. See `LICENSING.md` at the repository root.

**`third_party/RAEv2`: the most restrictive case in this tree.**
CC BY-NC 4.0 forbids commercial use and cannot be relicensed under PEAL's
LGPL-3.0, but it allows sharing the work and modified versions of it for
non-commercial purposes, with attribution. `third_party/RAEv2` itself is only a
submodule pointer to upstream (a clone gets an empty folder); ADA's modified
RAEv2, which PEAL's RAE generators were trained with, is tracked since
2026-09-28 as the separate non-commercial package `packages/peal-xai-rae`
(CC BY-NC 4.0, changes listed in `peal_rae/RAEv2/MODIFICATIONS.md`). Point
`$PEAL_RAEV2_DIR` at `packages/peal-xai-rae/peal_rae/RAEv2` to use it from here.

`third_party/RAEv2/pretrained_models/encoders/dino` (~80 MB of weights) should
not be committed either way; it belongs in a model-hub download.

## Local changes

- 2026-09-28: personal absolute paths (colleagues' home directories, cluster scratch) in the ImageNet `meta.bin` defaults of `atlas/predictions`, `atlas/cli` and `interpretability/phrase_bank.py`, plus the provenance comment of `integrations/raev2/overlay/.../DDT.py` replaced by defaults derived from `$PEAL_DATA` / `$PEAL_RUNS`; no functional change otherwise.
