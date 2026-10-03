# Changelog

## Unreleased

### Notebooks

- The tutorial notebooks are rewritten as working examples that run on Google Colab against the
  PyPI package (`peal-xai` 0.1.0): a shared first cell installs PEAL, fetches the config files and
  the dictionary vocabulary the wheel lacks, and sets `PEAL_BASE` / `PEAL_DATA` / `PEAL_RUNS`.
- `00_tour_of_peal_with_trains.ipynb` replaces the Waterbirds tour: a DINOv3 probe trained on a
  freight-car split where every freight car carries graffiti, the shortcut found by the MSAE
  dictionary and by DiDAE, and corrected with DFR (held-out graffiti counterfactuals no longer fool
  it). It adapts to GPUs under 20 GB (Colab T4).
- `01_explain_and_correct_with_sce.ipynb` (SCE + CFKD with the pretrained ImageNet DDPM),
  `02_bring_your_own_model_and_data.ipynb` (class folders and the ONNX contract) and
  `03_explain_an_onnx_model_with_didae.ipynb` replace the three walkthrough notebooks that needed a
  repository checkout, user data and, for SCE, GPU-days of generator training.

## 0.1.0 (2026-09-29)

First public release of `peal-xai`: the PEAL library (SCE, CFKD, DiDAE), the
paper reproduction scripts, and the DiDAE web demo.

### Web demo (`peal/web`)

- Upload an ONNX image classifier and a zip of class folders; DiDAE ranks the
  sparse-dictionary directions the classifier relies on and asks for a verdict
  per direction.
- The job page shows the classifier's accuracy on the uploaded images first
  (with a warning when it is near chance), a step timeline, the live
  latent-flip ranking while counterfactuals render, and the queue.
- Feedback per direction: bars of latent / ambient / verified flips, the four
  class x concept group accuracies with their average and worst group, and all
  verified factual/counterfactual pairs.
- Correction after the verdicts: **DFR** (refit only the final linear layer,
  about a minute; logistic regression, soft-margin SVM or closed-form ridge,
  chosen by cross-validation), offered when the final layer is found in the
  ONNX graph; full CFKD finetuning stays available as an experimental option.
- Defaults from a hyperparameter search on Waterbirds: bf16 sampling, 25 DDPM
  steps, guidance 3, concept pairs, bounds x3, 10 directions x 30 renders.

### Library

- DiDAE inverts only the images it renders (was: the whole pool), writes live
  progress (`sweep_progress.json`) and an index of every verified pair
  (`successful_flips/index.json`, `max_export_per_direction`).
- `RAEDiffusionAutoencoderConfig`: `sde_autocast` (bf16) and
  `matmul_precision` options.
- `peal.architectures.onnx_predictor`: `find_final_linear`,
  `truncate_to_features`, `write_final_linear`; `select_onnx_outputs` is
  idempotent.
- `WaterbirdsDataset` downloads into the folder named by `x_selection`, so the
  shipped Waterbirds configs find their images.

### Fixes

- RAEv2 fork: decoder configs load with transformers >= 5.
- The final ONNX export of a DiDAE run no longer segfaults the run (an
  unchanged model is copied, a corrected one is exported in a subprocess).
- Importing `peal.data.datasets` no longer forces matplotlib's Agg backend,
  which silenced plots in notebooks.
- DiDAE no longer writes `dino_evaluation_state.pt` into the working directory.
- `deploy/spark/Dockerfile` builds again (installs the package extras instead
  of the research `requirements.txt`).

### Notebooks

- New `notebooks/00_tour_of_peal_with_waterbirds.ipynb`: the whole workflow
  with the library's config classes and factories, from the self-downloading
  Waterbirds dataset and a DINOv3 probe to DiDAE and a DFR correction.

### Known limitations

- The sweep's sampler comes from the explainer config; an explainer without a
  `sampler` resets the step count to 50 whatever the generator config says.
  The web demo passes the generator's sampler explicitly.
- Full CFKD finetuning of the DINOv3 probe is slow and degraded accuracy on
  Waterbirds; use DFR.
- Notebooks 01-03 still drive the command-line scripts.
- The web demo reads its templates from the repository, so it runs from a
  clone or the Docker image (`deploy/spark`), not from the installed wheel.
