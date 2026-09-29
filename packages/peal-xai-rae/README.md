# peal-xai-rae

**Non-commercial use only (CC BY-NC 4.0).**

The optional companion of [`peal-xai`](https://pypi.org/project/peal-xai/) that
enables PEAL's RAE generators (`RAEDiffusionAutoencoder`, the ImageNet and
CelebA representation-autoencoder counterfactual generators). It contains a
modified copy of [RAEv2](https://github.com/nanovisionx/RAEv2) by Singh, Zheng,
Wu, Zhang, Shechtman and Xie, which is licensed CC BY-NC 4.0, so this package is
too. `peal-xai` itself is LGPL-3.0-or-later and does not need this package for
anything else.

## Install

```bash
pip install "peal-xai[rae]"                  # from PyPI
pip install ./packages/peal-xai-rae          # from a clone of the PEAL repository
# the clip:ViT-L/14 encoder needs OpenAI CLIP, which is not on PyPI:
pip install "clip @ git+https://github.com/openai/CLIP.git"
# only for retraining stage 1 / stage 2:
pip install "peal-xai-rae[train]"
```

## Weights

The pretrained ImageNet weights (`decoder.pt` 1.66 GB, `stats.pt`,
`stage2_ema.pt` 4.96 GB) are too large for a wheel. They live at
<https://huggingface.co/sidney1505/peal-rae-clip-imagenet> (also CC BY-NC 4.0)
and are downloaded on first use by any generator config with
`weights: hf://sidney1505/peal-rae-clip-imagenet`, for example
`configs/web_demo/imagenet_rae_clip_hf.yaml`. They are cached under
`$PEAL_HF_WEIGHTS_DIR`, default `$PEAL_RUNS/hf_weights`.

## How PEAL finds it

PEAL looks for RAEv2 in this order: `$PEAL_RAEV2_DIR`, this installed package
(`peal_rae.raev2_dir()`), the in-repository copy under
`packages/peal-xai-rae/peal_rae/RAEv2`, and finally `external/RAEv2`.

## Licence

`LICENSE` (CC BY-NC 4.0 with the RAEv2 copyright notice). What was changed
relative to upstream is listed in `peal_rae/RAEv2/MODIFICATIONS.md`.
