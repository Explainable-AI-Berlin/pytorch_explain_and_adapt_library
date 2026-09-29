# Licensing map

PEAL is **LGPL-3.0-or-later** (`LICENSE.txt`, with the full texts in
`COPYING.LESSER` and `COPYING`). That covers everything PEAL's authors wrote and
permits commercial use.

PEAL also vendors third-party research code under `peal/dependencies/` so that
the baselines and generators it compares against run unmodified. Those keep
their own licences, and **one of them is non-commercial**. This file is the map.
`THIRD_PARTY_NOTICES.md` has the per-component detail.

## The short version

| If you are | You can use |
|---|---|
| An academic or other non-commercial user | All of PEAL, including the RAE generators |
| A commercial user | All of PEAL **except** the RAE generators and anything else marked below |

## The non-commercial part: RAEv2 and the RAE generators

`RAEDiffusionAutoencoder` — the ImageNet and CelebA representation-autoencoder
generators, and the published RAE weights — are built on **RAEv2**
([nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2)), which its authors
license **CC BY-NC 4.0**. That forbids commercial use, is not an open-source
licence, and cannot be sublicensed under PEAL's LGPL-3.0.

CC BY-NC 4.0 does allow sharing the material and modified versions of it for
non-commercial purposes, with attribution. The repository is therefore split:

- The RAE generators need a **modified** RAEv2 (its `clipproj-vit-L` encoder and
  stage-2 latent cache are not upstream). That fork lives in the repository under
  `packages/peal-xai-rae/`, keeps the CC BY-NC 4.0 `LICENSE` and RAEv2's copyright
  notice, and lists every changed file in `peal_rae/RAEv2/MODIFICATIONS.md`. It is
  **never** relicensed as LGPL.
- It ships as its own distribution, **`peal-xai-rae`**, which you opt into with
  `pip install "peal-xai[rae]"` (or `pip install ./packages/peal-xai-rae`, or
  `python tools/install_rae.py`, which shows you the licence first). It is
  deliberately not part of the `full` extra.
- The `peal-xai` package on PyPI contains no RAEv2 and no ADA, so the package
  itself is LGPL-only.
- The pretrained ImageNet weights are not in any wheel (6.6 GB); they are
  downloaded from `hf://sidney1505/peal-rae-clip-imagenet` (CC BY-NC 4.0) on
  first use.
- **Using the RAE generators is strictly non-commercial**, whatever the rest of
  PEAL permits. That restriction comes from RAEv2's authors, not from us, and we
  cannot waive it.

## Commercial users: use a different generator

PEAL selects generators by name through `peal/generators/generator_factory.py`,
and adaptors talk to them through a common interface, so swapping the generator
is a config change rather than a rewrite. The generators that carry no
non-commercial restriction:

| Generator | Vendored code it needs | Licence of that code |
|---|---|---|
| `DiffusionAutoencoder` (the DiffAE-style generator behind most DiDAE results) | `diffusion_regression_counterfactuals` | MIT |
| `DDPMGenerator` | `FastDiME_CelebA`, `ace` | MIT |

Both are full-strength diffusion generators, and the diffusion autoencoder is the
one the Square, CelebA and Camelyon17 experiments use. A commercial user who
wants ImageNet-scale rendering can train or plug in their own generator against
the same interface; nothing in the analysis or correction pipeline is
RAE-specific.

## ADA: non-commercial, by David Drexlin

`peal/dependencies/ADA` (Actionable Data Atlases) is a separate research codebase
written by **David Drexlin** and licensed **CC BY-NC 4.0**
(`peal/dependencies/ADA/LICENSE`). Like the RAEv2 fork it is for non-commercial use
only, requires credit to its author, and is never part of the `peal-xai` package.

## Components with permission rather than a public licence

One vendored folder carries no licence file and is redistributed because its
authors gave permission:

| Folder | Status |
|---|---|
| `PathLDM` | upstream ships no licence file; its authors granted permission to redistribute with attribution |

Permission granted to PEAL is not automatically a licence to you. If you plan to
redistribute PEAL yourself, or to use this component commercially, contact its
authors.

## Previously unresolved

**None as of 2026-09-25.** Every vendored third-party component now has a
confirmed upstream and a licence file. The last two to be resolved were:

- `attacks` — [Hadisalman/smoothing-adversarial](https://github.com/Hadisalman/smoothing-adversarial), MIT
- `smoothdiff_experiments` — [adrhill/smoothdiff-experiments](https://github.com/adrhill/smoothdiff-experiments), MIT

What remains is not a licence question but a permission one: `PathLDM` is
redistributed on its authors' permission rather than under a public licence, as
described above.

## Summary by artifact

| Artifact | Contains RAEv2? | Effective terms |
|---|---|---|
| The `peal-xai` PyPI package | no | LGPL-3.0-or-later |
| The `peal-xai-rae` package (`packages/peal-xai-rae`) | yes, the modified fork | CC BY-NC 4.0 |
| This git repository | yes, under `packages/peal-xai-rae/` (and ADA's RAEv2 integration) | LGPL-3.0-or-later, except `packages/peal-xai-rae/` and `peal/dependencies/ADA/` (both CC BY-NC 4.0) and `PathLDM` (permission) |
| An install with `peal-xai[rae]` | yes | LGPL for PEAL, CC BY-NC 4.0 for anything touching the RAE generators |
| The published RAE weights on Hugging Face | derived from it | CC BY-NC 4.0 |
