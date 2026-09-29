# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit TODO


# ---------------------------------------------------------------------------
# Float index of DiDAE_Journal_Paper/main.tex, in caption order. Regenerate it
# whenever a float is added, removed or reordered, because LaTeX numbers floats by
# the order their \caption appears, and a `table`/`figure` holding two minipages
# with two \caption commands produces TWO numbers. That is what made the earlier
# numbering in this script wrong by one: tab:components became Table 1, and the
# two captions sharing one table environment are Tables 6 and 7.
#
#   Table  1  tab:components                                DiDAE components per dataset
#   Table  2  tab:quantitative_results_main                 main results (quality, speed, Gain)
#   Table  3  tab:quantitative_teachers                     teacher choice
#   Table  4  tab:quantitative_base_models                  Gain by student
#   Table  5  tab:quantitative_dictionaries                 Procrustes against SVD
#   Table  6  tab:sota_comparison                           metadata-based baselines
#   Table  7  tab:sota_comparison_foundation_model_fixing   is projection enough?
#   Table  8  tab:hyper_generator                           generator/dictionary hyperparameters
#   Table  9  tab:hyper_editing                             editing hyperparameters
#   Table 10  tab:generator_inversion_ablation              inversion and generator ablation
#
#   Figure  1  fig:teaser                     find and remove a Clever Hans feature
#   Figure  2  fig:overview                   gradient-based / global / component-defined
#   Figure  3  fig:dae_vs_didae               DAE's one direction against DiDAE's components
#   Figure  4  fig:ranking-results            Ranking DiDAE on three probes
#   Figure  5  fig:qualitative_results        four Procrustes directions (Square, CelebA)
#   Figure  6  fig:sae_didae                  batch top-K SAE on Sparse Numbers
#   Figure  7  fig:global_counterfactuals     trajectories on the causal/confounding plane
#   Figure  8  fig:qualitative_random_ddpm    random unselected CelebA, every method, DDPM
#   Figure  9  fig:svd_didae                  Square SVD directions
#   Figure 10  fig:didae_linesearch           line-search factor ablation
#   Figure 11  fig:didae_cfkd_before_after    decision boundary before and after DiDAE-CFKD
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Before spending GPU time, check that every config this script references can
# actually run: that it parses, that its student/teacher/generator exist on this
# machine, and where it would write. This takes about a second.
#
#   python preflight.py --script reproduction_scripts/reproduce_didae_results.sh --quiet
#
# It exits non-zero when something cannot run, so CI can gate on it. Run it
# yourself before a long session; the line below only reports and continues, so
# that a partially reproducible checkout can still run the parts that work.
python preflight.py --script "$0" --quiet || \
    echo "^ these lines will fail; the rest of the script still runs"
# ---------------------------------------------------------------------------

# Table 2  (main results: counterfactual quality, speed and downstream Gain)
# Square
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml"
# train square foundation model
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/square_all_attributes_resnet18.yaml"
# train square diffusion autoencoder
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/square_diffusion_autoencoder.yaml"
# train original DAE
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/square_diffusion_autoencoder_original.yaml"
# train square procrustes component analysis
python run_component_analysis.py --config $PEAL_RUNS/square/diffusion_autoencoder/config.yaml --sd_config configs/didae_experiments/sparse_dictionaries/procrustes_sae_square.yaml
# run CFKD with original DAE
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_dae_original_cfkd.yaml"
# run DiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_dime_cfkd.yaml"
# run ACE-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_ace_cfkd.yaml"
# run FastDiME-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_fastdime_cfkd.yaml"
# run SCE-CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_sce_cfkd.yaml"
# run DiDAE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_didae_procrustes_cfkd.yaml"
# CelebA Blond_Hair confounding Male task
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba_Blond_Hair_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/celeba_latent_oracle.yaml"
# train celeba diffusion autoencoder
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/celeba_diffusion_autoencoder.yaml"
# train original DAE
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/celeba_diffusion_autoencoder_original.yaml"
# train the OpenCLIP ViT-L/14 conditioned diffusion autoencoder (writes
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip_vit_l14). Consumed by the openclip DiDAE row
# below and by the Figure 2 block; it had no training line before.
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/celeba_diffusion_autoencoder_openclip_vit_l14.yaml"
# train celeba component analysis
python run_component_analysis.py --config $PEAL_RUNS/celeba/diffusion_autoencoder/config.yaml --sd_config configs/didae_experiments/sparse_dictionaries/procrustes_sae_celeba.yaml
# run CFKD with original DAE
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_dae_original_cfkd.yaml"
# run DiME CFKDconfigs/didae_experiments/predictors/celeba_latent_oracle.yaml
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_dime_cfkd.yaml"
# run ACE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_ace_cfkd.yaml"
# run FastDiME CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_fastdime_cfkd.yaml"
# run SCE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_sce_cfkd.yaml"
# run DiDAE CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_didae_openclip_cfkd.yaml"
# run DiDAE CFKD with DDPM (edit-friendly) inversion instead of DDIM inversion
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_didae_openclip_ddpm_cfkd.yaml"
# Camelyon
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/camelyon17_1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/camelyon17_latent_oracle.yaml"
# Camelyon (ResNet18, fully poisoned/full-data variant) -- this is the setting of the Camelyon block
# of Table 2: the shared student below is used by the four baselines and by the DiDAE-PathLDM run.
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/camelyon17_classifier_poisoned100.yaml"
# Baselines (DiME, ACE, FastDiME, SCE). NOTE: the published rows were generated with a Camelyon DDPM
# that only existed on a co-author's machine; the configs now point at the DDPM trained two lines
# above ($PEAL_RUNS/camelyon17_1k/ddpm, FID 295). With it, FastDiME seed 0 gives Gain -20.3 instead
# of the published 16.3, so treat the Camelyon baseline rows as re-measurements, not reproductions.
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/camelyon17_poisoned100_resnet18_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/camelyon17_poisoned100_resnet18_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/camelyon17_poisoned100_resnet18_fastdime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/camelyon17_poisoned100_resnet18_sce_cfkd.yaml"
# DiDAE CFKD (PathLDM + PLIP generator, ResNet18 student, fully poisoned data).
# The generator is the public PathLDM release used as-is (no LoRA finetuning; train_generator.py is
# NOT run for it), conditioned on the PLIP image embedding through the inverse text projection.
# Download + environment: see reproduction_scripts/pathldm_env.sh (checkpoint into
# $PEAL_RUNS/pathldm_plip, --no-deps extras, PL 2.x shim). Then:
source reproduction_scripts/pathldm_env.sh
# (a) fit the [tumor, hospital] orthogonal Procrustes dictionary on PLIP embeddings and measure the
#     empirical component bounds (writes $PEAL_RUNS/camelyon17/didae_pathldm/sparse_dictionaries/...)
python run_component_analysis.py --config "<PEAL_BASE>/configs/didae_experiments/generators/camelyon17_pathldm_autoencoder_resnet_didae.yaml" --sd_config configs/didae_experiments/sparse_dictionaries/orthognal_camelyon17.yaml
# (b) CFKD. finetune_iterations must be 1 (the committed value 0 stops before the Gain is computed).
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/camelyon17_poisoned100_resnet18_pathldm_didae_cfkd.yaml"
# REPRODUCIBILITY STATUS (2026-09-21): with the current code and the public checkpoint this run gives
# NAFR 80.0 / Diversity 1.0 / Sparsity 37.7 / NA 81.2 / Unbiasedness 49.3 / Gain 19.4 (seed 0),
# not the published 75.0 / 26.7 / 59.9 / 89.8 / 72.4 / 26.4. Both dictionary directions decode into a
# stain-like change that the oracle reads as "less tumor" (diversity ~1 under every variant tried:
# linesearch [1.1], [1.1, dynamic], dynamic targeting off, and the pre-2026-07-28 per-candidate
# conditioning renormalisation, env PEAL_PATHLDM_ZSEM_REF=0). The published run used a generator
# directory and dictionary weights that no longer exist ($PEAL_RUNS is the only surviving root),
# so the difference could not be pinned to a code change; see .agent/handoff.md.
# evaluation
# to see the results on has to look into the tensorboard files in the PEAL_RUNS folder
python peal/visualization/create_didae_table1.py

# Variance bars: see the seed loop at the bottom of this script, before the evaluation.


# Table 3  (teacher choice)
# Analysis of the influence of the teacher
# For the Square dataset for poisoning 80% (corresponds to 0.6 correlation)
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned080.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1000x080_didae_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1000x080_didae_preclustered_cfkd.yaml"
#python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1000x080_didae_false_cfkd.yaml"
#python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1000x080_didae_mask_cfkd.yaml"
#python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1000x080_didae_human_cfkd.yaml"


# Table 4  (Gain by student: ResNet-18 against the probe)
# Square
# train linear probe from foundation model
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/square1k_foundation_linear_poisoned098.yaml"
# run cfkd on foundation model probe
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1kx098_didae_cfkd.yaml"
# CelebA
# train linear probe from foundation model
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/celeba1k_foundation_linear_poisoned098.yaml"
# run cfkd on foundation model probe
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_openclip_didae_cfkd.yaml"
# evaluation
# to see the results on has to look into the tensorboard files in the PEAL_RUNS folder and add results from previous tables



# Table 5  (supervised Procrustes against unsupervised SVD)
python run_component_analysis.py --config $PEAL_RUNS/square/diffusion_autoencoder/config.yaml --sd_config configs/didae_experiments/sparse_dictionaries/svd_default.yaml
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_didae_svd_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1kx098_didae_svd_cfkd.yaml"
# evaluation
# to see the results on has to look into the tensorboard files in the PEAL_RUNS folder and add results from previous tables

# Table 6  (Gain against the metadata-based baselines)
# Square
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098_dfr.yaml"
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/square_1k_poisoned098_group_dro.yaml"
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_pclarc.yaml"
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_rrclarc.yaml"
# CelebA
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098_dfr.yaml"
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/blond_confounding_male_1k_poisoned098_group_dro.yaml"
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_pclarc.yaml"
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_rrclarc.yaml"
# evaluation
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
cat ${PEAL_RUNS}/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/pclarc/best_model_result.txt
cat ${PEAL_RUNS}/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/rrclarc/best_model_result.txt
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/classifier_poisoned098/pclarc/best_model_result.txt
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/classifier_poisoned098/rrclarc/best_model_result.txt
# the results for CFKD and the baseline can be taken from the previous tables


# Table 7  (is projection enough?)
# Square
# run DiDAE projection
python run_adaptor.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1kx098_didae_projection.yaml"
# CelebA
# run DiDAE projection
python run_adaptor.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_didae_projection.yaml"
# evaluation
# to see the results on has to look into the tensorboard files in the PEAL_RUNS folder and add results from previous tables

# No longer a table in the manuscript.
# The four-component CelebA dictionary survives only as the four-direction variant of
# Figure 5 (see the Components row of Table 8); the table that reported its Gain was
# removed. Kept because the component analysis below still produces Figure 5's CelebA panels.
# CelebA number of components evaluation
python run_component_analysis.py --config $PEAL_RUNS/celeba/diffusion_autoencoder/config.yaml --sd_config configs/didae_experiments/sparse_dictionaries/procrustes_sae_celeba_4comps.yaml
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_didae_4comps_openclip_cfkd.yaml"


# Figure 2  (gradient-based against global against component-defined counterfactuals)
# the SCE run below explains against the unpoisoned CelebA DDPM ($PEAL_RUNS/celeba/ddpm);
# it had no training line before.
python train_generator.py --config "<PEAL_BASE>/configs/sce_experiments/generators/celeba_ddpm.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1k_test_resnet18_dae_original_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1k_test_resnet18_sce_cfkd.yaml"
# images can be found under
# $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/sce_cfkd_test/0/latent_cluster_collages0/0000025_cluster0.png
# $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/sce_cfkd_test/0/latent_cluster_collages1/0000025_cluster1.png
# $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/dae_cfkd_test/0/latent_cluster_collages0/0000025_cluster0.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/0/0000025_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/1/0000025_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/2/0000025_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/3/0000025_collage.png

# Figure 3  (DAE's one direction against DiDAE's components)
# images can be found under
# $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/dae_cfkd_test/0/latent_cluster_collages0/0000033_cluster0.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/0/0000033_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/1/0000033_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/2/0000033_collage.png


# Figure 5  (component-defined counterfactuals, CelebA panels)
# images can be found under
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/0/0000050_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/1/0000050_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/2/0000050_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/3/0000050_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/0/0000034_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/1/0000034_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/2/0000034_collage.png
# $PEAL_RUNS/celeba/diffusion_autoencoder_openclip/OrthogonalProcrustesDictionary/3/0000034_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/0/0000010_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/1/0000010_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/2/0000010_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/3/0000010_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/0/0000059_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/1/0000059_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/2/0000059_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/OrthogonalProcrustesDictionary/3/0000059_collage.png

# Figure 9  (component-defined counterfactuals along the Square SVD directions)
# images can be found under
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/0/0000249_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/1/0000249_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/2/0000249_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/3/0000249_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/0/0000307_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/1/0000307_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/2/0000307_collage.png
# $PEAL_RUNS/square/diffusion_autoencoder/SVDDictionary/3/0000307_collage.png

# =============================================================================================
# Everything below covers the parts of the manuscript that the blocks above do not.
# Every table and figure number in this script is the current manuscript's; see the index at the
# top of the file.
# =============================================================================================


# Sparse Numbers (Section 4.6, Figure 6)
# 1000 three-digit concepts in superposition. The ResNet-18 is the frozen encoder the diffusion
# autoencoder is conditioned on; the batch top-K SAE is the dictionary whose coefficients are
# deactivated one at a time.
python train_predictor.py --config "<PEAL_BASE>/configs/sae_experiments/predictors/only_sparse_numbers.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/sae_experiments/generators/only_sparse_numbers_diffusion_autoencoder.yaml"
python run_sae_analysis.py --config $PEAL_RUNS/only_sparse_numbers/diffusion_autoencoder/config.yaml --sd_config "<PEAL_BASE>/configs/sae_experiments/sparse_dictionaries/batch_topk_only_sparse_numbers.yaml" --run_name "only_sparse_numbers_batch_topk"
# Figure 6 panels:
# $PEAL_RUNS/only_sparse_numbers/diffusion_autoencoder/BatchTopKSAE/<component>/<idx>_collage.png


# Generators for the two natural-image probes (Appendix E)
# Two-stage representation autoencoder over frozen OpenAI CLIP ViT-L/14 patch tokens. Stage 1 is
# the latent decoder, stage 2 the flow-matching transformer; both are trained by this one config.
# The runs below read a decoding preset written next to the checkpoint, e.g.
# $PEAL_RUNS/imagenet/rae_clip/config_ep12_sde_g2.yaml (NICO++) and
# $PEAL_RUNS/imagenet/rae_clip/config_ep43_guided_g2_s100.yaml (ImageNet), which differ only in
# the checkpoint epoch, the sampler and the classifier-free guidance scale.
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/imagenet_rae_clip.yaml"


# Ranking DiDAE on three probes (Section 4.7, Figure 4)
# Each probe is ranked against a dictionary that already exists in its frozen encoder space, and
# nothing is trained per probe. One run per probe performs the whole loop: distil the probe, rank
# the directions by verified ambient flips, show the top ones to the teacher, and repair the ones
# it marks spurious with CFKD (finetune_iterations: 1).

# (1) Sparse Numbers, the synthetic control: Num128 confounded with the distractor Num713.
#     The teacher is the unpoisoned classifier, so the ground-truth directions are known.
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/only_sparse_numbers_classifier_unpoisoned_128_713.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/only_sparse_numbers1k_classifier_poisoned100_128_713.yaml"
python -W ignore run_didae.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/only_sparse_numbers1kx100_128_713_didae.yaml"

# (2) NICO++ crocodile vs lizard, natural context as the cue. ResNet-18 student, MSAE over CLIP
#     ViT-L/14 decoded through the RAE, teacher = the unpoisoned classifier. Concept PAIRS are
#     ranked here, which frees the edit from the trust region of a single coefficient.
#     The dataset subset is extracted from $PEAL_DATA/NICO++.zip on first use.
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_classifier_unpoisoned.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098.yaml"
python -W ignore run_didae.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/nico_crocodile_vs_lizard_500_poisoned098_didae_openai_clip_rae_sde_g2_msae.yaml"

# (3) ImageNet fireboat vs lifeboat, water jet as the cue, DINOv3 ViT-L/16 linear probe. The
#     probed encoder and the dictionary's encoder are different foundation models. SINGLE atoms
#     are ranked here, so each counterfactual changes one named concept.
#     The natural pair is symlinked out of ImageNet:
python tools/generate_imagenet_binary_dataset.py --class1 554 --class2 625 --name fireboat_vs_lifeboat --symlink
#     A probe on the natural pair reaches 99.6-100 % and leaves nothing to repair, so the TRAINING
#     split is filtered to make the shortcut load-bearing: a fireboat is kept when MSAE atom #5717
#     ("squirting") is active on its CLIP embedding and a lifeboat when it is not, leaving 1406 of
#     2600 images; validation and test stay the natural splits. The activations used for the filter
#     are cached next to the dataset as fireboat_vs_lifeboat_curated5717_activations_5717.json and
#     the result is $PEAL_DATA/fireboat_vs_lifeboat_curated5717, which the data config points at.
python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/imagenet_fireboat_vs_lifeboat_curated5717_dinov3_linear.yaml"
python -W ignore run_didae.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/imagenet_fireboat_vs_lifeboat_curated5717_dinov3_ep43s100_msae.yaml"
#     This config's teacher field points at the probe itself, so the run stops after the ranking
#     and the rendered flips. The direction labels and the CFKD repair behind the figure's third
#     panel come from the driver below, which judges every direction, builds the CFKD dataset with
#     a 20 % hold-out, repairs the last layer and evaluates:
python reproduction_scripts/overnight_fig1_pair.py fireboat_vs_lifeboat_curated5717 "<PEAL_BASE>/configs/didae_experiments/adaptors/imagenet_fireboat_vs_lifeboat_curated5717_dinov3_ep43s100_msae.yaml"
#     Figure 1, the teaser, is built from this same fireboat run; it is the only source for it.
#     Group accuracies split by whether the confounder is present on the real image:
python reproduction_scripts/overnight_group_eval.py fireboat_vs_lifeboat_curated5717 "<PEAL_BASE>/configs/didae_experiments/adaptors/imagenet_fireboat_vs_lifeboat_curated5717_dinov3_ep43s100_msae.yaml"

# Figure 4 is assembled from the three run directories (flip counts from sweep_results.pt and
# direction_feedback.txt, accuracies from each repair run):
python docs/paper_figures/make_results.py


# Inversion ablation (Table 10)
# Same generator, dictionary, probe and CFKD budget; only the sampler changes. DDIM is the default
# when a generator config carries no sampler field; edit-friendly DDPM is selected by
#   sampler: {type: ddpm, num_steps: 20, spacing: uniform}
# on the generator. The two CelebA rows are the pair of configs already listed above
# (celeba1kx098_resnet18_didae_openclip_cfkd.yaml and ..._ddpm_cfkd.yaml); the third CelebA row
# swaps the pixel-space decoder for the representation autoencoder:
python train_generator.py --config "<PEAL_BASE>/configs/didae_experiments/generators/celeba_rae_clip.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_didae_rae_ddpm_cfkd.yaml"


# Remaining figures, all read out of run directories produced above
# Figure 7  (counterfactual trajectories on the causal/confounding plane):
#   $PEAL_RUNS/<dataset>/.../<didae run>/0/val_counterfactuals_global.png
# Figure 8  (random unselected CelebA samples, every method, edit-friendly DDPM):
#   the validation counterfactuals of the three CelebA CFKD runs listed above
# Figure 10 (linesearch-factor ablation): the linearsearch panels of the component analysis,
#   $PEAL_RUNS/celeba/diffusion_autoencoder/OrthogonalProcrustesDictionary4Comps/<comp>/<idx>_linearsearch.png
# Figure 11 (decision boundary before and after DiDAE-CFKD):
#   $PEAL_RUNS/<dataset>/.../<didae run>/0/decision_boundary.png and decision_boundary_test.png


# ===========================================================================
# Variance bars: seeds 1-3 (the n column of the main table and the per-seed
# appendix table).
#
# Each seed runs in its own $PEAL_RUNS tree, so the seed-0 runs above are never
# touched and every $PEAL_RUNS/... reference inside the configs resolves within
# that tree. The base models are rebuilt there with the --seed override, so no
# seeded copies of the configs are needed.
#
# The generators and their dictionaries stay at seed 0 and are linked into each
# seed tree rather than retrained: in practice the generator is a large-scale
# pretrained model, the dataset split barely moves it, and rebuilding it per seed
# costs far more compute than the variance it would expose. What is reseeded is
# the dataset split, the student, the unpoisoned teacher, the dataloader order
# and the counterfactual sampling.

PEAL_RUNS_SEED0="${PEAL_RUNS}"     # buffer the caller's root, restored at the end

for SEED in 1 2 3; do
  export PEAL_RUNS="${PEAL_RUNS_SEED0}${SEED}"
  mkdir -p "$PEAL_RUNS"

  # generators + dictionaries: reuse seed 0 (see the note above)
  for GEN in square/diffusion_autoencoder square/diffusion_autoencoder_original square1k/ddpm \
             celeba/diffusion_autoencoder celeba/diffusion_autoencoder_original celeba1k/ddpm; do
    mkdir -p "$PEAL_RUNS/$(dirname "$GEN")"
    ln -sfn "$PEAL_RUNS_SEED0/$GEN" "$PEAL_RUNS/$GEN"
  done

  # -- Square: teacher + student, then every method ---------------------------
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml" --seed $SEED
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml" --seed $SEED
  for METHOD in dae_original dime ace fastdime sce didae_procrustes; do
    python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/square1k_resnet18_poisoned098_${METHOD}_cfkd.yaml" --seed $SEED
  done

  # -- CelebA: the fast methods only; a gradient baseline costs ~1 GPU-day/seed
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba_Blond_Hair_classifier_unpoisoned.yaml" --seed $SEED
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml" --seed $SEED
  python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/celeba_latent_oracle.yaml" --seed $SEED
  for METHOD in dae_original fastdime didae_openclip; do
    python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/celeba1kx098_resnet18_${METHOD}_cfkd.yaml" --seed $SEED
  done

  # -- Camelyon: FastDiME and the PathLDM DiDAE row. The DiDAE run needs the
  #    PathLDM environment, so it is sourced inside the loop.
  # python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_classifier_unpoisoned.yaml" --seed $SEED
  # python train_predictor.py --config "<PEAL_BASE>/configs/didae_experiments/predictors/camelyon17_classifier_poisoned100.yaml" --seed $SEED
  # (camelyon17_1k/ddpm and camelyon17/didae_pathldm are linked from seed 0 the same way)
  # python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/camelyon17_poisoned100_resnet18_fastdime_cfkd.yaml" --seed $SEED
  # source reproduction_scripts/pathldm_env.sh
  # python run_cfkd.py --config "<PEAL_BASE>/configs/didae_experiments/adaptors/camelyon17_poisoned100_resnet18_pathldm_didae_cfkd.yaml" --seed $SEED
done

export PEAL_RUNS="${PEAL_RUNS_SEED0}"


# Evaluation
# Every number in the tables is logged to the tensorboard event files under the run's base_dir.
# The main results table is assembled by:
python peal/visualization/create_didae_table1.py


# ===========================================================================
# Tables from the runs above  (appended 2026-09-22; nothing above was changed)
# ===========================================================================
# Everything below only *reads* finished run directories under $PEAL_RUNS.
# It trains nothing and writes nothing into $PEAL_RUNS. The point is that the
# numbers in the thesis tables are produced mechanically from run artefacts
# rather than transcribed by hand.
#
# Each finished explainer run stores
#   $PEAL_RUNS/<tree>/<method>/0/validation_stats.npz   the desiderata metrics
#   $PEAL_RUNS/<tree>/<method>/logs/events*             the final `gain` scalar
# and the collector turns those into a JSON index plus a LaTeX table body.

# (a) Everything this script produced, discovered automatically:
python reproduction_scripts/collect_results.py \
    --runs "$PEAL_RUNS" --discover \
    --out results/didae_results.json \
    --latex results/didae_main_table.tex

# (b) Or exactly the trees behind the main table, with the labels it uses
#     (seeds of one dataset share a label and are aggregated as mean +- std):
# python reproduction_scripts/collect_results.py --runs "$PEAL_RUNS" \
#     --tree "Square=square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/repro_ddpm_s0" \
#     --tree "CelebA-Blond=celeba1k/Blond_Hair/classifier_poisoned098/repro_ddpm_s0" \
#     --tree "CelebA-Blond=celeba1k/Blond_Hair/classifier_poisoned098_seed1/repro_ddpm" \
#     --tree "CelebA-Blond=celeba1k/Blond_Hair/classifier_poisoned098_seed2/repro_ddpm" \
#     --tree "CelebA-Blond=celeba1k/Blond_Hair/classifier_poisoned098_seed3/repro_ddpm" \
#     --tree "Camelyon=camelyon17/classifier_poisoned100/repro_ddpm_s0" \
#     --tree "Camelyon=camelyon17_1k/classifier_poisoned098/repro_ddpm" \
#     --out results/didae_main_table.json --latex results/didae_main_table.tex
#
# results/didae_main_table.tex is the body the table \input{}s; the column
# order is flip_rate, diversity, sparsity, non-adversarial rate, unbiasedness,
# counterfactuals per second, gain. Note that "NAFR" is defined differently in
# different chapters -- the collector reports the stored keys and the mapping
# is made in the table, not here.

# (c) The Ranking DiDAE funnel figure is still assembled by hand:
#     docs/paper_figures/make_results.py carries its numbers as literals that
#     were transcribed from each ranking run's
#       $PEAL_RUNS/<tree>/<method>/sweep_results.pt        (flip counts)
#       $PEAL_RUNS/<tree>/<method>/direction_feedback.txt  (teacher verdicts)
#     Closing that loop -- reading those two artefacts directly -- is the one
#     step of the results pipeline that is not yet mechanical.
