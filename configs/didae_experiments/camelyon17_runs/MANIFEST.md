# Camelyon17 poisoned100: paper runs (September 2026)

Student: ResNet18 on Camelyon17 with 100 % hospital poisoning. Seed 0 is the original student
(`$PEAL_RUNS/camelyon17/classifier_poisoned100`), seeds 1-3 are the seed-variability students
(`configs/didae_experiments/predictors/camelyon17_classifier_poisoned100_seed{1,2,3}.yaml`).
Every run: 1 CFKD finetune iteration, 200 validation samples. The results of the paper runs are kept
on the group cluster in `/home/space/datasets/peal_ahmed/camelyon17_paper_results/<method>/seed<s>/`
(with a README); the original locations below are links to it. The configs write a new run under
`$PEAL_RUNS/camelyon17/paper_runs/<method>/seed<s>/` and read every input from `$PEAL_RUNS` / `$PEAL_DATA`,
so reproducing never touches the archived results. Reproduction: the Camelyon17 section of
`reproduction_scripts/reproduce_didae_results.sh`.

| Method | Seeds | Config(s) | Original location (now a link) |
|---|---|---|---|
| DAE | 0-3 | `adaptors/camelyon17_poisoned100_seed{0,1,2,3}_resnet18_dae_baseline_cfkd.yaml` | `peal_ahmed/didae_camelyon17/classifiers/classifier_poisoned100_{original,seed1,seed2,seed3}/dae_baseline_cfkd` |
| DiME | 0 | `camelyon17_runs/baselines/dime_seed0.yaml` | `peal_ahmed/experiments_results_paper/classifier_poisoned_camelyon100/dime_cfkd_peal_runs_sce` |
| ACE | 0 | `camelyon17_runs/baselines/ace_seed0.yaml` | `.../classifier_poisoned_camelyon100/ace_cfkd_peal_runs_sce` |
| FastDiME | 0 | `camelyon17_runs/baselines/fastdime_seed0.yaml` | `.../classifier_poisoned_camelyon100/fastdime_cfkd_peal_runs_sce` |
| FastDiME | 1-3 | `adaptors/camelyon17_poisoned100_seed{1,2,3}_resnet18_fastdime_cfkd.yaml` | `peal_ahmed/didae_camelyon17/classifiers/classifier_poisoned100_seed{1,2,3}/fastdime_cfkd` |
| SCE | 0 | `camelyon17_runs/baselines/sce_seed0.yaml` | `.../classifier_poisoned_camelyon100/sce_cfkd` |
| DiDAE (SAE) | 0-3 | `camelyon17_runs/didae_sae/seed{0,1,2,3}.yaml` | `peal_ahmed/didae_camelyon17/cfkd_signed_seeds/sae_pmdynamic_x2_lambda7_skip5/seed{0,1,2,3}` |

Notes
- The seed-1-3 runs and the DAE and DiDAE runs set `reproducible_sampling: True`, so the samples CFKD
  picks follow the seed. The seed-0 baseline runs (2025) did not have this option.
- DiDAE seed 0 reuses the probe and distilled predictor of the control run
  (`camelyon17_runs/control/newlinesearch_bs4.yaml`); seeds 1-3 train their own. The SAE
  (`sparse_dictionaries/batch_topk_camelyon_pathldm_plip.yaml`) is fitted on the control run's generator
  with `fit_sae_components.py`.
- The DAE runs use the DAE of `generators/camelyon_diffusion_autoencoder_seeds.yaml` (frozen UNI encoder,
  trained with `train_dae_ddim_only.py`) and need lightning 2.1.4 and lmdb.
- ACE seed 0 sampled from a copy of the Camelyon17 DDPM (`peal_ahmed/camelyon17/ddpm`, 50 respaced steps)
  that no longer exists; its config now uses `$PEAL_RUNS/camelyon17/ddpm` like DiME, FastDiME and SCE.
- The DiME/ACE/FastDiME/SCE seed-0 metrics in the paper are the current metric recalculated from these
  runs' stored counterfactuals (25 Aug 2026); the runs themselves are from April-May 2025.
