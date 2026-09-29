# Installation

PEAL runs on Python 3.9 to 3.11. Install it from PyPI to use it as a library,
or clone the repository to reproduce the papers.

## From PyPI

The distribution is called `peal-xai`; the import package is still `peal`,
because the name `peal` on PyPI belongs to an unrelated project.

```bash
pip install peal-xai
```

That gives the core library: datasets, predictors, the diffusion autoencoder
and DDPM generators, the counterfactual explainer, and the CFKD, DiDAE, ClArC
and GroupDRO adaptors. Everything else is an extra, listed below.

:::{warning}
The wheel does not carry the 800 experiment configs under `configs/`, which
live outside the `peal` package. They ship in the source distribution instead.
If you want to run the documented `--config <PEAL_BASE>/configs/...` commands,
clone the repository (or unpack the sdist) and point `$PEAL_BASE` at it.
Loading a config without that raises an error saying so.
:::

## From a clone

### Conda

```bash
git clone https://github.com/Explainable-AI-Berlin/pytorch_explain_and_adapt_library.git
cd pytorch_explain_and_adapt_library
conda env create -f environment.yaml
conda activate peal
```

`requirements.txt` is the exact package set the papers were produced with, and
it is generated from the pip section of `environment.yaml`. It is deliberately
not what `pip install peal-xai` uses: those are pinned versions of a whole
research environment, while the published package declares lower bounds. Use
it when you want to reproduce results, not when you want the library.

### Apptainer

```bash
apptainer build python_container.sif deploy/apptainer/python_container.def
apptainer run --nv python_container.sif
```

## Optional extras

Each extra backs one part of the library. The code imports these packages
lazily, so a missing extra surfaces as an error naming the install command
rather than breaking `import peal`.

| Extra | Installs | Needed for |
| --- | --- | --- |
| `pathldm` | `omegaconf` | the PathLDM generators (Camelyon17, Follicles) |
| `stablediffusion` | `captum` | the Stable Diffusion 3 generator |
| `onnx` | `onnx`, `onnxruntime`, `onnx2torch` | using an ONNX classifier as the predictor |
| `web` | FastAPI and Flask stacks | the web demo and the human feedback pages |
| `xai` | `corelay`, `virelay`, `zennit-crp`, `h5py` | the SpRAy and ViRelAy teachers and the LRP explainer |
| `datasets` | `wilds`, `kaggle`, `kagglehub` | the WILDS datasets (Camelyon17, RxRx1) and Kaggle downloads |
| `clip` | `open-clip-torch` | the Stable Diffusion autoencoder |
| `mpi` | `mpi4py` | distributed sampling in the DDPM generators; needs a system MPI |
| `pygame` | `pygame` | the image transforms that render through pygame |
| `full` | all of the above | everything installable from PyPI |

```bash
pip install "peal-xai[xai,datasets]"
pip install "peal-xai[full]"
```

Two things are not installable from PyPI and have their own commands:

| Command | Gives you |
| --- | --- |
| `pip install "clip @ git+https://github.com/openai/CLIP.git"` | OpenAI CLIP, which is not published on PyPI |
| `pip install "peal-xai[rae]"` or `python tools/install_rae.py` | the modified RAEv2 (`peal-xai-rae`, CC BY-NC 4.0, non-commercial) for the representation-autoencoder generators |

Set `PEAL_RAEV2_DIR` only when the fork lives somewhere PEAL does not look by
itself (it checks the installed `peal-xai-rae` package, then
`packages/peal-xai-rae/peal_rae/RAEv2` in a clone, then `./external/RAEv2`).

## Commands

Installing the package puts twelve commands on your path. Each one is the same
code as the identically named script in a clone, so `peal-cfkd --config x.yaml`
and `python run_cfkd.py --config x.yaml` do the same thing.

| Command | Repository script | Runs |
| --- | --- | --- |
| `peal-cfkd` | `run_cfkd.py` | the CFKD repair loop |
| `peal-didae` | `run_didae.py` | the DiDAE adaptor |
| `peal-explain` | `run_explainer.py` | an explainer on its own |
| `peal-adapt` | `run_adaptor.py` | any adaptor from its config |
| `peal-train-generator` | `train_generator.py` | generator training |
| `peal-train-predictor` | `train_predictor.py` | predictor training |
| `peal-distill-predictor` | `train_distilled_predictor.py` | student distillation |
| `peal-evaluate-predictor` | `evaluate_predictor.py` | predictor evaluation |
| `peal-sae-analysis` | `run_sae_analysis.py` | sparse dictionary evaluation |
| `peal-component-analysis` | `run_component_analysis.py` | generator component analysis |
| `peal-generate-dataset` | `generate_dataset.py` | dataset generation |
| `peal-preflight` | `preflight.py` | the config check |

All of them take a `--config` path, so they need the configs described above.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `PEAL_DATA` | `./datasets` | where datasets such as CelebA live |
| `PEAL_RUNS` | `./peal_runs` | where every run writes logs, checkpoints and collages |
| `PEAL_RAEV2_DIR` | installed `peal-xai-rae`, else the in-repository copy | location of the RAEv2 fork (CC BY-NC 4.0) for the RAE generators |

The placeholder `<PEAL_BASE>` in config files is the repository root and is
resolved by the library itself; it is not an environment variable.
