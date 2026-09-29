# How the pieces fit together

Every PEAL experiment is a composition of a few interchangeable components,
each configured by a YAML file and each replaceable without touching the
others. The packages below map one-to-one onto those roles.

```{list-table}
:header-rows: 1
:widths: 22 78

* - Package
  - Role
* - {mod}`peal.data`
  - Datasets, dataloaders and *dataset generators* that build controlled
    confounder datasets (for example a copyright tag pasted onto one class).
    All datasets derive from the same interface so any explainer or adaptor
    can consume them.
* - {mod}`peal.architectures`
  - Predictor architectures used in the papers plus the ONNX wrapper that lets
    you bring a classifier trained elsewhere.
* - {mod}`peal.generators`
  - Generative models that produce counterfactuals: DDPMs, diffusion
    autoencoders, Stable Diffusion, PathLDM, Flux and the optional RAE
    pipeline. Generators are either *invertible* (encode/decode) or
    *edit-capable* (move an input across a classifier's decision boundary).
* - {mod}`peal.explainers`
  - Explainers that turn a predictor plus a generator into explanations, most
    importantly the counterfactual explainer that implements SCE and the
    baselines it is compared with.
* - {mod}`peal.teachers`
  - Sources of feedback on explanations: a human through the web interface,
    an oracle model, symbolic rules, clustering, segmentation masks, ...
* - {mod}`peal.adaptors`
  - Methods that repair a predictor from that feedback: CFKD, DiDAE, ClArC,
    GroupDRO and projection-based variants.
* - {mod}`peal.sparse_dictionaries`
  - Sparse autoencoders, Matryoshka SAEs, Procrustes and SVD dictionaries
    used by DiDAE to discover directions in a generator's latent space.
* - {mod}`peal.editors`
  - Latent-space editing and inversion routines (DDPM/EDICT inversion) shared
    by the generators.
* - {mod}`peal.training`
  - Trainers, loss criteria and loggers used to train predictors, generators
    and distilled students.
* - {mod}`peal.visualization`
  - Collages, grids and the comparison plots that appear in the papers.
* - {mod}`peal.web`
  - The web demo: upload an ONNX classifier, collect feedback, download a
    corrected model.
```

## The CFKD loop

`run_cfkd.py` drives the central pipeline of the library:

1. A run directory is created under `$PEAL_RUNS` and the original predictor is
   saved there.
2. A generator is trained, or an existing one is loaded from its config.
3. The explainer computes a round of counterfactuals and writes collages.
4. A teacher labels the counterfactuals (a human through the web app, or an
   automatic teacher).
5. The counterfactuals and their feedback-corrected labels become an extra
   dataset.
6. The predictor is fine-tuned on it, and the loop returns to step 3 until
   the configured number of rounds is reached.

The command-line entry points at the repository root (`run_cfkd.py`,
`run_didae.py`, `run_explainer.py`, `train_generator.py`, `train_predictor.py`,
...) each drive one of these stages and are documented in their module
docstrings.
