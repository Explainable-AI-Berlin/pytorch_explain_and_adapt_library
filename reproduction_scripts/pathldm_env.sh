#!/bin/bash
# Environment for the Camelyon17 DiDAE row (PathLDM + PLIP generator). Source this before any
# run_component_analysis.py / run_cfkd.py command that uses a PathldmAutoencoder generator:
#     source reproduction_scripts/pathldm_env.sh
#
# 1. Public PathLDM checkpoint (the PLIP-conditioned model of the PathLDM authors, FID 7.64 row
#    of their README). The two files are used as-is; nothing is trained on top of them:
#       pip install gdown
#       gdown --folder https://drive.google.com/drive/folders/1v3SXkA1D94w7Q1XMPSEA1yrSfpwhXzCr -O $PEAL_RUNS/pathldm_plip
#    -> $PEAL_RUNS/pathldm_plip/checkpoints/epoch_3.ckpt and .../configs/08-03T09-35-project.yaml
# 2. Extra packages the vendored PathLDM needs. Install them --no-deps into a side directory so
#    they do not pull a second torch/CUDA stack into the PEAL environment:
#       pip install --no-deps --target $PEAL_RUNS/pyenv_pathldm omegaconf kornia kornia_rs taming-transformers-rom1504 pytorch-fid
# 3. The vendored PathLDM targets PyTorch Lightning 1.x
#    (pytorch_lightning.utilities.distributed.rank_zero_only); the sitecustomize shim below
#    re-registers that module on PL 2.x. It is picked up automatically via PYTHONPATH.
PEAL_BASE=${PEAL_BASE:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export PYTHONPATH=$PEAL_RUNS/pyenv_pathldm:$PEAL_BASE/peal/dependencies/pathldm_shim:$PEAL_BASE/peal/dependencies/PathLDM${PYTHONPATH:+:$PYTHONPATH}
