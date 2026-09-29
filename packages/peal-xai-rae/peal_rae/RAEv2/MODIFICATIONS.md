# Modifications to RAEv2

This folder is a **modified** copy of RAEv2,
Copyright (c) 2025-2026 Jaskirat Singh, Boyang Zheng, Zongze Wu, Richard Zhang,
Eli Shechtman, Saining Xie, <https://github.com/nanovisionx/RAEv2>,
licensed under **CC BY-NC 4.0** (<https://creativecommons.org/licenses/by-nc/4.0/>,
full text in `LICENSE`). The modified version is distributed under the same
licence, for non-commercial purposes only. It is not covered by PEAL's LGPL.

Base: upstream commit `8a0d238f8dc3b261aba98b217f6c79c0182e8e94` ("init").

## Where the changes come from

1. **ADA (Actionable Data Atlases), by David Drexlin**: the stage-2 latent cache and sampler,
   resumable stage-2 state, training configs, diagnostics and the broader
   compatibility work behind ADA's generator experiments. Recorded in PEAL's
   repository as `peal/dependencies/ADA/integrations/raev2/`
   (`patches/raev2-ada-development.patch` plus `overlay/`); applying both to the
   base commit reproduces every file here except the three below.
2. **PEAL**: the `clipproj` encoder (`CLIPProjEncoder` in
   `src/encoders/vision_encoder.py`, 34 lines plus its registry entry), whose
   global token is CLIP's projected image embedding so that the stage-2 model can
   be edited with OpenAI-CLIP sparse dictionaries (MSAE); the published ImageNet
   and CelebA RAE generators use it. Plus `from __future__ import annotations`
   in `src/stage1/disc/utils.py` and `src/utils/logging.py` for Python 3.9, and
   `_load_decoder` in `src/stage1/rae.py` fills in `patch_size` before building
   the decoder config (transformers>=5 rejects the configs' placeholder string).

`src/` is byte-identical to the source snapshots of PEAL's ImageNet and CelebA
stage-1 and stage-2 training runs.

## Omitted from upstream

`pyproject.toml`, `uv.lock`, `.pre-commit-config.yaml`, `.gitignore` (upstream
packaging), and `src/stage1/disc/.caches/vgg.pth` (7 KB of LPIPS weights, which
`src/stage1/disc/lpips_utils.py` downloads on first use).

## Files changed relative to the base commit

`M` modified, `A` added:

```
M  src/configs/shared.py
M  src/configs/stage1.py
M  src/configs/stage2.py
M  src/data/__init__.py
M  src/data/imagenet_hf_dataset.py
M  src/data/unified_dataloader.py
M  src/encoders/vision_encoder.py
M  src/eval/__init__.py
M  src/eval/datasets.py
M  src/eval/distributed.py
M  src/eval/generation.py
M  src/eval/reconstruction.py
M  src/offline_eval.py
M  src/offline_eval_stage1.py
M  src/stage1/__init__.py
M  src/stage1/decoders/decoder.py
M  src/stage1/disc/dinodisc.py
M  src/stage1/disc/utils.py
M  src/stage1/engine.py
M  src/stage1/rae.py
M  src/stage1/utils.py
M  src/stage2/engine.py
M  src/stage2/models/DDT.py
M  src/stage2/models/model_utils.py
M  src/stage2/transport/sampler.py
M  src/stage2/transport/transport.py
M  src/stage2/utils.py
M  src/train.py
M  src/train_stage1.py
M  src/utils/checkpoint.py
M  src/utils/logging.py
A  configs/stage2/training/imagenet-dinov2l-k1-patch-class-only-cache-premix64-80ep-4gpu-accum32.yaml
A  configs/stage2/training/imagenet-dinov2l-k1-patch-cls-cache-premix64-80ep-4gpu-accum32.yaml
A  configs/stage2/training/imagenet-dinov2l-k1-patch-cls-only-cache-premix64-80ep-4gpu-accum32.yaml
A  configs/stage2/training/imagenet-dinov2l-k1-patch-cls-only-cache-premix64-80ep-4gpu-ca8-every4.yaml
A  scripts/build_stage2_latent_cache_distributed.py
A  scripts/evaluate_cls_gaussian_perturbations.py
A  scripts/evaluate_cls_only_vs_cls_class_same_noise.py
A  src/data/cache_index_order.py
A  src/data/latent_cache_dataset.py
A  src/stage2/state_utils.py
A  tests/test_cache_index_order.py
A  tests/test_latent_cache_sampler.py
```
