---
license: cc-by-nc-4.0
tags:
  - representation-autoencoder
  - diffusion
  - counterfactual-explanations
  - explainable-ai
  - clip
datasets:
  - imagenet-1k
library_name: pytorch
---

# RAE-CLIP ImageNet — representation autoencoder for counterfactual explanation

Two-stage representation autoencoder (RAE) over frozen **OpenAI CLIP ViT-L/14**
features, trained at 256×256 on ImageNet-1k. It is the image decoder behind the
ImageNet experiments of **DiDAE** (Disentangled Diffusion Autoencoders) in
[PEAL](https://github.com/Explainable-AI-Berlin/pytorch_explain_and_adapt_library).

Its purpose is not sampling. It exists so that an **edit made in CLIP embedding
space can be rendered back into an image**: DiDAE deactivates or activates a
sparse-dictionary concept in the 768-d CLIP image embedding, and this model turns
the edited embedding back into a picture, keeping the rest of the input intact.
That makes the counterfactual visible and lets a classifier be queried on it.

## Files

| File | Size | What it is |
|---|---|---|
| `decoder.pt` | 1.7 GB | Stage-1 EMA decoder (ViT-XL) from CLIP patch tokens to pixels |
| `stage2_ema.pt` | 5.0 GB | Stage-2 EMA flow-matching model (DDT), epoch 43, step 107 586 |
| `stats.pt` | 2.1 MB | Latent normalisation statistics for stage 1 |

The **encoder is not included**. OpenAI CLIP ViT-L/14 is downloaded separately at
runtime and is unmodified.

## Architecture

**Stage 1** reconstructs pixels from the CLIP ViT-L/14 patch-token latent
(16×16×1024 at 256×256) with a ViT-XL decoder, trained for 16 epochs with an
adversarial term (DINO ViT-S/8 discriminator, starting at epoch 8) and a
perceptual loss.

**Stage 2** is a DDT flow-matching model (`DiTwDDTHeadIG`, about 1.24 B
parameters, hidden 1440/2048, depth 28/2) over that latent, conditioned through
additive AdaLN on the **768-d CLIP image embedding** — the same vector the
sparse dictionaries are fitted in, which is what makes a dictionary edit
decodable. It was trained with 10 % conditioning dropout against a learned null
embedding, so classifier-free guidance is available at inference.

## Intended use

Research on counterfactual explanation and on spurious correlations in image
classifiers: rendering concept-level edits, auditing what a classifier reacts
to, and generating counterfactual training data.

**Out of scope.** This is not a general-purpose image generator, it is not
trained for prompt-driven synthesis, and it is **non-commercial** (see the
licence section). It was trained on natural images at 256×256; on other domains
such as histopathology or radiology the reconstructions are not meaningful, so
check round-trip fidelity before trusting an edit.

## How to use

The weights are consumed by PEAL's `RAEDiffusionAutoencoder`. Point a generator
config at this repository:

```yaml
generator_type: RAEDiffusionAutoencoder
weights: hf://<org>/peal-rae-clip-imagenet   # this repo
encoder: "clip:ViT-L/14"
image_size: 256
sampler:
  type: ddpm
  num_steps: 100
guidance_scale: 2.0
render_noise: inverted
```

PEAL downloads the three files on first use.

**The model classes are not in upstream RAEv2.** They come from a *modified*
RAEv2 in which the encoder type `clipproj-vit-L` was added: it exposes CLIP
ViT-L/14 patch tokens as the latent and the projected global token as the
conditioning vector. Stock upstream RAEv2 rejects that encoder with
`Unknown encoder type: clipproj`. The fork ships as `peal-xai-rae`
(`packages/peal-xai-rae`, CC BY-NC 4.0): `pip install "peal-xai[rae]"` or
`python tools/install_rae.py`.

The settings above are the ones used for the published ImageNet edits: edit-
friendly stochastic (DDPM) inversion, 100 steps, classifier-free guidance 2.0,
decoding from the inverted noise so the input's structure is preserved.

## Evaluation and known limits

Measured on 16 ImageNet images (2026-09-15), comparing guidance scales at the
published settings:

| Quantity | Value |
|---|---|
| Round-trip reconstruction after inversion | 15.6 dB |
| Fraction of a requested embedding edit realised, guidance 1.0 | 0.07 |
| Fraction of a requested embedding edit realised, guidance 2.0 | 0.21 |

The first row is the honest ceiling: inversion preserves structure at the cost
of fidelity, so reconstructions are recognisable rather than pixel-accurate. The
second and third say that an inverted starting point resists edits, and guidance
is what makes the edit show. Raising guidance further increases the edit and
degrades realism.

For edits that should regenerate detail rather than preserve it, PEAL's
`render_noise: fresh` samples from the edited embedding instead of inverting.

## Training data

ImageNet-1k train split (about 1.28 M images) at 256×256. ImageNet's own terms
apply to the data and are research-oriented; they are not granted by this
repository.

## Licence and attribution

Released under **CC BY-NC 4.0**: non-commercial use with attribution.

These weights are our own training output, but they were produced with **RAEv2**
(Singh, Zheng, Wu, Zhang, Shechtman and Xie, *Improved Baselines with
Representation Autoencoders*), which is itself licensed CC BY-NC 4.0, using a
DINO discriminator checkpoint that project ships, on ImageNet. The most
restrictive term in that chain governs, so these weights carry it forward rather
than PEAL's own LGPL-3.0. Anything built on them, including PEAL's web demo, is
non-commercial.

Please cite RAEv2 as well as the paper below.

## Citation

```bibtex
@article{bender2026visual,
  title={Visual Disentangled Diffusion Autoencoders: Scalable Counterfactual Generation for Foundation Models},
  author={Bender, Sidney and Morik, Marco},
  journal={arXiv preprint arXiv:2601.21851},
  year={2026}
}
```
