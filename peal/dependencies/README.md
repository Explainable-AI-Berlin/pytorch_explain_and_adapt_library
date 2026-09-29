# `peal/dependencies/` — third-party code

Everything in this directory is **external research code**, kept apart from
PEAL's own implementation under `peal/`. PEAL vendors rather than depends on it
because these projects publish paper repositories rather than installable
packages, and because the baselines had to be adapted to run on identical data,
checkpoints and metrics for the comparisons to be fair.

Each folder carries an `UPSTREAM.md` naming the original authors, the paper, the
upstream repository, the commit this copy is compared against, the upstream
licence, and exactly how our copy differs. Credit for these methods belongs to
their authors; please cite their work.

| Folder | Original authors | Upstream | Licence |
|---|---|---|---|
| `DiCE` | Mothilal, Sharma, Tan | interpretml/DiCE | MIT |
| `FastDiME_CelebA` | Pegios, Weng, Petersen, Feragen, Bigdeli | ppegiosk/FastDiME_CelebA | MIT |
| `FunnyNodules` | Gallée, Xiong, Beer, Götz | XRad-Ulm/FunnyNodules | MIT |
| `MSAE` | Zaigrajew, Baniecki, Biecek | WolodjaZ/MSAE | MIT |
| `PathLDM` | Yellapragada, Graikos, Prasanna, Kurc, Saltz, Samaras | cvlab-stonybrook/PathLDM | none declared; **redistribution permitted by the authors (2026-09-23)** |
| `SpLiCE` | Bhalla, Oesterling, Srinivas, Calmon, Lakkaraju | AI4LIFE-GROUP/SpLiCE | Apache-2.0 |
| `ace` | Jeanneret, Simon, Jurie | guillaumejs2403/ACE | MIT |
| `ddpm_inversion` | Huberman-Spiegelglas, Kulikov, Michaeli | inbarhub/DDPM_inversion | MIT |
| `diffusion_regression_counterfactuals` | Ha, Bender | DevinTDHa/Diffusion-Counterfactuals-for-Image-Regressors | MIT |
| `glow` | Kim Seonghyeon | rosinality/glow-pytorch | MIT |
| `group_dro` | Sagawa, Koh, Hashimoto, Liang | kohpangwei/group_DRO | MIT |
| `lora` | HuggingFace | huggingface/diffusers (example) | Apache-2.0 |
| `matryoshka_sae` | Bussmann, Leask, Nanda | bartbussmann/matryoshka_sae | **none declared** |
| `time` | Jeanneret, Simon, Jurie | guillaumejs2403/TIME | MIT |
| `ADA` (incl. `third_party/RAEv2`) | separate project of the authors; RAEv2 by Singh, Zheng, Wu, Zhang, Shechtman, Xie | nanovisionx/RAEv2 | **CC BY-NC 4.0** for RAEv2 |

Folders documented by `ORIGIN.md` instead of `UPSTREAM.md` are PEAL's own code
or have unconfirmed provenance: `attacks`, `smoothdiff_experiments`,
`pathldm_shim`, `edit_friendly_ddpm_inversion`, `ADA`.

## Before publishing

Two components cannot be redistributed as they stand: **matryoshka_sae**
declares no licence, and **RAEv2** is CC BY-NC 4.0. PathLDM also declares no
licence, but its authors granted redistribution permission on 2026-09-23 (see
`PathLDM/UPSTREAM.md`). Each folder's note states the resolution being pursued.
