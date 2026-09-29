# Third-party notices

PEAL itself is distributed under the GNU Lesser General Public License v3 or
later; see `LICENSE.txt`, with the full licence texts in `COPYING.LESSER` and
`COPYING`.

Everything under `peal/dependencies/` is **not** part of PEAL. Each folder is a
vendored copy of someone else's research code, kept there so that the baselines
and generators PEAL compares against run unmodified inside PEAL's pipeline.
Each folder carries an `UPSTREAM.md` (third-party, provenance confirmed) or an
`ORIGIN.md` (PEAL's own code, or provenance still to be confirmed) recording the
upstream commit and how this copy differs. Please cite the original authors, not
PEAL, when you use their method.

Installing the `peal` package **does** redistribute the folders listed under
"Vendored components with a clear licence" below, because eighteen first-party
modules import them and a wheel without them cannot load most generators, the
editors, either GroupDRO adaptor, the trainers or two of the sparse
dictionaries. Each keeps its own licence, which is unchanged by being packaged
alongside PEAL.

Every component still vendored here carries a licence that permits
redistribution, so as of 2026-09-25 nothing under `peal/dependencies/` is
withheld from the package for licensing reasons.

One component was **removed from the repository** on that date rather than
merely excluded, because it declared no licence and PEAL therefore had no right
to redistribute it at all:

| Removed | Why | Replacement |
|---|---|---|
| `matryoshka_sae` | upstream declares no licence | reimplemented as `peal/sparse_dictionaries/batch_topk_network.py` |

`ADA` (Actionable Data Atlases) stays in this repository. It was written by
David Drexlin and is licensed **CC BY-NC 4.0** (`peal/dependencies/ADA/LICENSE`):
non-commercial use only, with credit to its author. It is kept out of the
installed package: nothing under `peal/` imports it, and its non-commercial terms
must not leak into the LGPL wheel. See `LICENSING.md`.

**`LICENSING.md` is the map of which parts of PEAL are commercially usable.** The
short version: PEAL is LGPL-3.0-or-later, but the RAE generators depend on RAEv2
(CC BY-NC 4.0) and are therefore strictly non-commercial. Every other generator,
including the diffusion autoencoder behind most published results, is free of
that restriction.

## Vendored components with a clear licence

| Folder | Upstream | Licence |
|---|---|---|
| `DiCE` | [interpretml/DiCE](https://github.com/interpretml/DiCE) | MIT |
| `FastDiME_CelebA` | [ppegiosk/FastDiME_CelebA](https://github.com/ppegiosk/FastDiME_CelebA) | MIT |
| `FunnyNodules` | [XRad-Ulm/FunnyNodules](https://github.com/XRad-Ulm/FunnyNodules) | MIT |
| `MSAE` | [WolodjaZ/MSAE](https://github.com/WolodjaZ/MSAE) | MIT |
| `ace` | [guillaumejs2403/ACE](https://github.com/guillaumejs2403/ACE) | MIT |
| `attacks` | [Hadisalman/smoothing-adversarial](https://github.com/Hadisalman/smoothing-adversarial) | MIT |
| `smoothdiff_experiments` | [adrhill/smoothdiff-experiments](https://github.com/adrhill/smoothdiff-experiments) | MIT |
| `ddpm_inversion` | [inbarhub/DDPM_inversion](https://github.com/inbarhub/DDPM_inversion) | MIT |
| `diffusion_regression_counterfactuals` | [DevinTDHa/Diffusion-Counterfactuals-for-Image-Regressors](https://github.com/DevinTDHa/Diffusion-Counterfactuals-for-Image-Regressors) | MIT |
| `glow` | [rosinality/glow-pytorch](https://github.com/rosinality/glow-pytorch) | MIT |
| `group_dro` | [kohpangwei/group_DRO](https://github.com/kohpangwei/group_DRO) | MIT |
| `time` | [guillaumejs2403/TIME](https://github.com/guillaumejs2403/TIME) | MIT |
| `PathLDM` | [cvlab-stonybrook/PathLDM](https://github.com/cvlab-stonybrook/PathLDM) | MIT |
| `SpLiCE` | [AI4LIFE-GROUP/SpLiCE](https://github.com/AI4LIFE-GROUP/SpLiCE) | Apache-2.0 |
| `lora` | [huggingface/diffusers](https://github.com/huggingface/diffusers) (`examples/text_to_image/train_text_to_image_lora.py`) | Apache-2.0 |

Twelve of these ship the upstream licence file verbatim. `group_dro`, `lora` and
`lora` retains the Apache-2.0 header inside the file itself. `PathLDM` now carries
upstream's MIT `LICENSE` verbatim, fetched from
<https://raw.githubusercontent.com/cvlab-stonybrook/PathLDM/main/LICENSE> on
2026-09-26; the PathLDM authors had also granted PEAL written permission to
redistribute, which is now belt and braces rather than the only basis.

## PEAL's own code that happens to live under `dependencies/`

| Folder | Status |
|---|---|
| `edit_friendly_ddpm_inversion` | PEAL's own implementation, and nothing imports it. The implementation used at runtime is the vendored fork in `ddpm_inversion`. Delete it, or wire it up deliberately. |
| `pathldm_shim` | PEAL's own `sitecustomize.py` path shim. Belongs under `peal/` proper. |

## RAEv2: a separate, non-commercial package

The RAE generators (`RAEDiffusionAutoencoder`, the ImageNet and CelebA
representation-autoencoder pipelines) are built on **RAEv2**
([nanovisionx/RAEv2](https://github.com/nanovisionx/RAEv2)), which is licensed
**CC BY-NC 4.0**. That licence forbids commercial use and cannot be relicensed
under PEAL's LGPL-3.0, but it does allow sharing the work and modified versions
of it for non-commercial purposes, with attribution.

The generators need a **modified** RAEv2, not stock upstream: the fork adds the
encoder type `clipproj-vit-L` that both published generators use (upstream
rejects it with `Unknown encoder type: clipproj`), a premixed stage-2 latent
cache and resumable stage-2 state. ADA's share of those changes has been in the
repository since 2026-09-25 as `peal/dependencies/ADA/integrations/raev2/`
(a patch plus an overlay), but applying it needs the RAEv2 submodule, which a
fresh clone cannot initialise; PEAL's own `clipproj` encoder was in no commit at
all, only in an untracked working copy and in the source snapshots RAEv2 writes
into each run directory. So until 2026-09-28 a fresh clone could neither train
nor run the RAE generators.

Since 2026-09-28 the fork is tracked in the repository as its own distribution:

- `packages/peal-xai-rae/` -- package `peal-xai-rae`, licence **CC BY-NC 4.0**
  (`LICENSE`, with RAEv2's copyright notice), base commit
  `8a0d238f8dc3b261aba98b217f6c79c0182e8e94`, every changed file listed in
  `peal_rae/RAEv2/MODIFICATIONS.md`. Its `src/` is byte-identical to the
  snapshots of the ImageNet and CelebA stage-1 and stage-2 training runs.
- It is **not** part of the `peal-xai` wheel or sdist. Install it with
  `pip install "peal-xai[rae]"`, `pip install ./packages/peal-xai-rae`, or

```
python tools/install_rae.py            # shows the licence, then pip-installs the package
python tools/install_rae.py --dir DIR  # or copies the fork to DIR; set PEAL_RAEV2_DIR=DIR
python tools/install_rae.py --check    # which copy PEAL uses, and whether it is the fork
```

From a clone nothing has to be installed: PEAL resolves `<PEAL_RAEV2>` to
`$PEAL_RAEV2_DIR`, else the installed package, else the in-repository copy, else
`external/RAEv2`. Without any of them PEAL raises an error naming the licence
when an RAE generator is constructed, and nothing else is affected. In
particular the ImageNet and CelebA diffusion autoencoders, and PEAL's own
edit-friendly DDPM and DDIM inversion, do not use RAEv2 at all. The ~79 MB of
DINO encoder weights RAEv2 keeps under `pretrained_models/` (training only) are
not in the repository.

`peal/dependencies/ADA/third_party/RAEv2` is still a bare submodule pointer to
upstream, which a clone cannot initialise; ADA's own RAEv2 users should point
`$PEAL_RAEV2_DIR` (or that path) at the fork.

## Provenance questions, all resolved 2026-09-25

This section tracked the components that blocked a public release. All three are
settled; the entries are kept struck through so the history is legible.


1. ~~**`attacks` — provenance unverified.**~~ **Resolved 2026-09-25.** It is
   `code/attacks.py` from
   [Hadisalman/smoothing-adversarial](https://github.com/Hadisalman/smoothing-adversarial)
   (Salman et al., NeurIPS 2019), **MIT**. Identified by comparing syntax trees:
   same three classes and ten methods, nine of them identical modulo Black
   formatting. The licence is now in `peal/dependencies/attacks/LICENSE` and the
   provenance in its `UPSTREAM.md`, which also records the one local change — a
   `pdb.set_trace()` in an exception handler inside `PGD_L2._attack` that should
   be removed.
2. ~~**`smoothdiff_experiments` — provenance unverified.**~~ **Resolved
   2026-09-25.** It is
   [adrhill/smoothdiff-experiments](https://github.com/adrhill/smoothdiff-experiments)
   (Hill, McKee, Maeß, *Smoothed Differentiation Efficiently Mitigates Shattered
   Gradients in Explanations*, NeurIPS 2025), **MIT**, authorship confirmed by
   the PEAL maintainer. Seven of the eight shared top-level definitions are
   identical in their syntax tree. The licence is now in
   `peal/dependencies/smoothdiff_experiments/LICENSE` and the provenance in its
   `UPSTREAM.md`.
3. ~~**`matryoshka_sae` — no licence declared.**~~ **Resolved 2026-09-25: removed.**
   Upstream
   ([bartbussmann/matryoshka_sae](https://github.com/bartbussmann/matryoshka_sae))
   ships no licence file, so every right stayed with its authors and PEAL could
   not redistribute it.

   It backed the batch top-K dictionary of the Sparse Numbers experiments. PEAL
   now implements batch top-K itself, in
   `peal/sparse_dictionaries/batch_topk_network.py`, LGPL-3.0 like the rest of
   PEAL. The method is Bussmann, Leask and Nanda (2024); methods are not
   copyrightable, only a given expression of them. Please keep citing their
   paper.

   The swap was verified bit for bit against the removed copy: identical
   initialisation, identical forward output in training and evaluation mode,
   identical `encode`/`decode`, and identical weights after training. The
   Sparse Numbers checkpoint fitted with the old backend loads unchanged, so
   published results are unaffected.

   The `MatryoshkaSAE` and `SVDFilteredMatryoshkaSAE` dictionaries, which wrapped
   the same upstream's nested-dictionary variant, were removed with it. They
   backed no published result: the papers report four dictionaries, and the
   "Matryoshka SAE" among them is MSAE (Zaigrajew et al., MIT), a different
   project that PEAL still vendors.

   Two MIT reimplementations exist (SAELens `batchtopk_sae.py`,
   dictionary_learning `batch_top_k.py`) and were considered as dependencies
   instead. Both declare `requires-python >= 3.10` while PEAL runs on 3.9, and
   both pull a language-model stack for what is a 512-dimensional
   image-representation dictionary, so neither was adopted.

## Model weights and datasets

No pretrained weights and no datasets are distributed with PEAL. The PathLDM
PLIP checkpoint is expected under
`peal/dependencies/plip_imagenet_finetune/`, overridable with `$PEAL_PLIP_DIR`;
CelebA, CelebA-HQ, Camelyon17, NICO++ and ImageNet are downloaded from their own
sources under `$PEAL_DATA` and keep their own terms, several of which are
research-use only.
