# This script is meant to be able to reproduce the results of the CFKD paper (ARXIV_LINK),
# submitted to Information Fusion. (It said "PDC paper" before, which is the other
# paper: PDC is reproduce_sce_results.sh.)
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit TODO


# ---------------------------------------------------------------------------
# Float index of the Information Fusion submission, in caption order:
#   dissertation/sidney_papers_latex/Counterfactual_Knowledge_Distillation___Revision/
#     main_inffus.tex        (main paper; main.tex and clean_inffus.tex have the same
#                             float order and the same labels)
#     supplement_inffus.tex  (standalone: its counters restart at 1 and it prefixes
#                             nothing, so cite its float as "Table 1 of the
#                             Supplementary Notes", never "Table S1")
#
#   Table 1  tab:sota_comparison       MAIN RESULTS, AGA per dataset x model-improvement method
#   Table 2  tab:statistical_analysis  three independent CFKD runs, mean and std
#   Table 3  tab:model_comparison      ResNet-18 against UNI-F and UNI-L on Camelyon17
#   Table 4  tab:explainers_merged     ablation over the counterfactual explainer
#
#   Figure 1  fig:intro                    CFKD against subgroup reweighting (schematic)
#   Figure 2  fig:follicle_dataset_intro   the Follicle dataset (schematic)
#   Figure 3  fig:effect                   effect of sample size and correlation level
#   Figure 4  fig:teacher_comparison       teacher comparison on Square at correlation 0.6
#   Figure 5  fig:cfkd_before_after_qualitative  counterfactuals before and after CFKD
#
#   Supplementary Notes, Table 1  tab:tabular_results   the two tabular datasets
#
# COUNTING TRAP: main_inffus.tex holds a sixth `figure*` inside `\if False ... \fi`
# (label fig:cfkd_before_after_qualitative_old). It compiles to nothing, so counting
# \begin{figure} occurrences gives 6 figures and numbers the qualitative collage as
# Figure 6. It is Figure 5. Algorithm 1 (alg:cfkd) uses its own counter and consumes
# no table or figure number.
#
# Figures 1 and 2 are hand-drawn. Figure 5 is data-derived but NO block below
# produces it, so it is a reproduction gap.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Before spending GPU time, check that every config this script references can
# actually run: that it parses, that its student/teacher/generator exist on this
# machine, and where it would write. This takes about a second.
#
#   python preflight.py --script reproduction_scripts/reproduce_cfkd_results.sh --quiet
#
# It exits non-zero when something cannot run, so CI can gate on it. Run it
# yourself before a long session; the line below only reports and continues, so
# that a partially reproducible checkout can still run the parts that work.
python preflight.py --script "$0" --quiet || \
    echo "^ these lines will fail; the rest of the script still runs"
# ---------------------------------------------------------------------------

# Table 1, column "Square"
# Reproduce SOTA results on the square dataset
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/square_1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/square1k/colora_confounding_colorb/torchvision/classifier_poisoned098/rrclarc/best_model_result.txt


# Table 1, column "Smiling"
# Reproduce SOTA results on CelebA copyrighttag dataset (the results over different poisoning levels will be averaged)
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba_copyrighttag_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098_diffusion_augmented.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/smiling_confounding_copyrighttag_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_098_pclarc.yaml"
cat ${PEAL_RUNS}/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/rrclarc/best_model_result.txt


# Table 1, column "Blond"
# Reproduce SOTA results on CelebA Blond_Hair confounding Male task
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba_Blond_Hair_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/celeba1k_Blond_Hair_confounding_Male_poisoned098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/blond_confounding_male_1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k/Blond_Hair/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Blond_Hair_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/blond_confounding_male_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba1k/Blond_Hair/classifier_poisoned098/rrclarc/best_model_result.txt


# Table 1, column "Camelyon"
# Reproduce SOTA results on Camelyon17 task
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/camelyon17_1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_poisoned098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/camelyon17_1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/camelyon17_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/camelyon17_1k/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/camelyon17_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/camelyon17_1k/classifier_poisoned098/rrclarc/best_model_result.txt


# Table 1, column "Follicles"  (the dataset is sketched in Figure 2)
# Reproduce SOTA results on follicle dataset
# If the automatic download fails, you can manually download https://huggingface.co/datasets/janphhe/follicles_true_features and put it into $PEAL_DATA/follicles
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/follicle_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml"
# run CFKD (careful, needs feedback through the human in the loop!!!)
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/follicles_sce_human_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/follicle_cut_classifier_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/follicle_cut_classifier_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/dfr/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/follicles_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/follicles_cut/classifier_natural/group_dro/model.cpl --data_config configs/cfkd_experiments/data/follicles_hints.yaml --model_config configs/cfkd_experiments/predictors/follicle_cut_classifier.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/follicles_pclarc.yaml"
cat ${PEAL_RUNS}/follicles_cut/classifier_natural/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/follicles_rrclarc.yaml"
cat ${PEAL_RUNS}/follicles_cut/classifier_natural/rrclarc/best_model_result.txt


# Table 1, column "FunnyNodules"
# Reproduce SOTA results on FunnyNodules dataset (InternalStructure confounding Roundness)
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/funnynodules_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/funnynodules1k_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/funnynodules1k_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/funnynodules1000x098_sce_cfkd.yaml"
# The four evaluations below use funnynodules_natural.yaml, not funnynodules_unpoisoned.yaml.
# The latter is the config that generates the raw image pool, so it lists all six attributes
# as confounding factors; group accuracies are defined for a label plus one confounder, and
# with more than two the dataset returns has_confounder as a list and the evaluation dies.
python evaluate_predictor.py --model_path $PEAL_RUNS/funnynodules1k/internalstructure_confounding_roundness/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/funnynodules_natural.yaml --model_config configs/cfkd_experiments/predictors/funnynodules1k_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/funnynodules1k_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/funnynodules1k/internalstructure_confounding_roundness/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/funnynodules_natural.yaml --model_config configs/cfkd_experiments/predictors/funnynodules1k_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/funnynodules1k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/funnynodules1k/internalstructure_confounding_roundness/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/funnynodules_natural.yaml --model_config configs/cfkd_experiments/predictors/funnynodules1k_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/funnynodules1k_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/funnynodules1k/internalstructure_confounding_roundness/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/funnynodules_natural.yaml --model_config configs/cfkd_experiments/predictors/funnynodules1k_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/funnynodules1000_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/funnynodules1k/internalstructure_confounding_roundness/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/funnynodules1000_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/funnynodules1k/internalstructure_confounding_roundness/classifier_poisoned098/rrclarc/best_model_result.txt


# Table 1, column "Nico++"
# Reproduce SOTA results on NICO++ dataset (Crocodile/Lizard confounding Grass/Rock)
# Ensure you have manually downloaded NICO++ to $PEAL_DATA/nico_plus_plus according to custom_datasets.py instructions
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_classifier_unpoisoned.yaml"
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/nico_crocodile_vs_lizard_500_ddpm_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/nico_crocodile_vs_lizard_500_poisoned098_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/nico_crocodile_vs_lizard_500/classifier_poisoned098/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/nico_crocodile_vs_lizard_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098.yaml
# run DiffAug
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098_diffusion_augmented.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/nico_crocodile_vs_lizard_500/classifier_poisoned098/diffusion_augmented/model.cpl --data_config configs/cfkd_experiments/data/nico_crocodile_vs_lizard_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/nico_crocodile_vs_lizard_500/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/nico_crocodile_vs_lizard_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098.yaml
# run GroupDRO
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/nico_crocodile_vs_lizard_500_poisoned098_group_dro.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/nico_crocodile_vs_lizard_500/classifier_poisoned098/group_dro/model.cpl --data_config configs/cfkd_experiments/data/nico_crocodile_vs_lizard_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/nico_crocodile_vs_lizard_500_classifier_poisoned098.yaml
# run P-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/nico_crocodile_vs_lizard_500_poisoned098_pclarc.yaml"
cat ${PEAL_RUNS}/nico_crocodile_vs_lizard_500/classifier_poisoned098/pclarc/best_model_result.txt
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/nico_crocodile_vs_lizard_500_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/nico_crocodile_vs_lizard_500/classifier_poisoned098/rrclarc/best_model_result.txt


# Table 1 of the Supplementary Notes (tab:tabular_results), columns "Circle" and "Adult".
# NOTE: the block also runs German Credit and COMPAS, which appear in no float of the paper.
# Experiments on tabular datasets
# Circle dataset
python train_predictor.py --config configs/tabular_experiments/models/symbolic_circle_classifier_unpoisoned.yaml
python train_generator.py --config configs/tabular_experiments/generators/circle_diffusion.yaml
python train_predictor.py --config configs/tabular_experiments/models/symbolic_circle_classifier_poisoned.yaml
python run_cfkd.py --config configs/tabular_experiments/adaptors/circle_cfkd.yaml

# Adult Income Dataset
python train_predictor.py --config configs/tabular_experiments/models/adult_classifier_poisoned.yaml
python run_cfkd.py --config configs/tabular_experiments/adaptors/adult_cfkd_dice.yaml

# German Credit Risk Dataset
python train_predictor.py --config configs/tabular_experiments/models/german_classifier_poisoned.yaml
python run_cfkd.py --config configs/tabular_experiments/adaptors/german_cfkd_dice.yaml

# COMPAS Recidivism Dataset
python train_predictor.py --config configs/tabular_experiments/models/compass_classifier_poisoned.yaml
python run_cfkd.py --config configs/tabular_experiments/adaptors/compass_cfkd_dice.yaml


# Figure 3, the correlation-level panels
# Experiments over different poisoning levels
# on Square dataset
# for 50% poisoning (corresponds to 0.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned050.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned050.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x050_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned050/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned050.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned050_rrclarc.yaml"
# running DFR makes no sense for 0.0 correlation, so we just use the unfixed model results

# for 60% poisoning (corresponds to 0.2 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned060.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned060.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x060_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned060/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned060.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned060_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned060/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned060.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned060_rrclarc.yaml"

# for 70% poisoning (corresponds to 0.4 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned070.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned070.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x070_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned070/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned070.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned070_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned070/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned070.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned070_rrclarc.yaml"

# for 80% poisoning (corresponds to 0.6 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned080.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned080.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned080/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned080.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned080_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned080/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned080.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned080_rrclarc.yaml"

# for 90% poisoning (corresponds to 0.8 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned090.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned090.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x090_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned090/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned090.yaml
# run DFR
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned090_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned090/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned090.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned090_rrclarc.yaml"

# for 100% poisoning (corresponds to 1.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/square1k_ddpm_poisoned100.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned100.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x100_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square1k/colora_confounding_colorb/torchvision/classifier_poisoned100/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square1k_classifier_poisoned100.yaml
# running DFR and RR-ClarC for correlation 1.0 is impossible, so we just use the unfixed model results

# on CelebA copyrighttag dataset
# for 50% poisoning (corresponds to 0.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned050.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned050.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x050_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned050/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned050.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_050_rrclarc.yaml"
# running DFR makes no sense for 0.0 correlation, so we just use the unfixed model results

# for 60% poisoning (corresponds to 0.2 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned060.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned060.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x060_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned060/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned060.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned060_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned060/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned060.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_060_rrclarc.yaml"

# for 70% poisoning (corresponds to 0.4 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned070.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned070.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x070_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned070/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned070.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned070_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned070/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned070.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_070_rrclarc.yaml"

# for 80% poisoning (corresponds to 0.6 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned080.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned080.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x080_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned080/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned080.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned080_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned080/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned080.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_080_rrclarc.yaml"

# for 90% poisoning (corresponds to 0.8 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned090.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned090.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x090_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned090/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned090.yaml
# run DFR
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned090_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned090/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned090.yaml
# run RR-ClarC
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/smiling_confounding_copyrighttag_090_rrclarc.yaml"

# for 100% poisoning (corresponds to 1.0 correlation)
python train_generator.py --config "<PEAL_BASE>/configs/cfkd_experiments/generators/celeba1k_copyrighttag_ddpm_poisoned100.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned100.yaml"
# run CFKD
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_sce_cfkd.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba1k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned100/sce_cfkd/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba1k_Smiling_confounding_copyrighttag_classifier_poisoned100.yaml
# running DFR or RR-ClarC for correlation 1.0 is impossible, so we just use the unfixed model results



# Figure 3, the sample-size panels
# NOTE: this block drives CFKD through the *_pfc_cfkd.yaml configs, whose explainer is
# perfect_false_counterfactuals, not SCE as in the Table 1 blocks. If Figure 3's CFKD curve
# is meant to be the same SCE-based CFKD, these point at the wrong explainer.
# Analysis of influence of sample number
# For the Square dataset
# for 1k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_pfc_cfkd.yaml"

# for 2k samples
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square2k_classifier_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square2kx098_pfc_cfkd.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square2k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square2k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square2k_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square2k_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/square2k/colora_confounding_colorb/torchvision/classifier_poisoned098/rrclarc/best_model_result.txt

# for 4k samples
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square4k_classifier_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square4kx098_pfc_cfkd.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square4k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square4k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square4k_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square4k_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/square4k/colora_confounding_colorb/torchvision/classifier_poisoned098/rrclarc/best_model_result.txt

# for 8k samples
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square8k_classifier_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square8kx098_pfc_cfkd.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square8k_classifier_poisoned098_dfr.yaml"
python evaluate_predictor.py --model_path $PEAL_RUNS/square8k/colora_confounding_colorb/torchvision/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/square_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/square8k_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square8k_poisoned098_rrclarc.yaml"
cat ${PEAL_RUNS}/square8k/colora_confounding_colorb/torchvision/classifier_poisoned098/rrclarc/best_model_result.txt

# For the CelebA copyrighttag dataset
# for 1k samples
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_pfc_cfkd.yaml"

# for 2k samples
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba2k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba2kx098_pfc_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba2k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba2k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba2k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/celeba2k_smiling_confounding_copyrighttag_098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba2k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/rrclarc/best_model_result.txt

# for 4k samples
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba4k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba4kx098_pfc_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba4k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba4k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba4k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/celeba4k_smiling_confounding_copyrighttag_098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba4k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/rrclarc/best_model_result.txt

# for 8k samples
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/celeba8k_Smiling_confounding_copyrighttag_classifier_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba8kx098_pfc_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/celeba8k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/celeba8k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/Smiling_confounding_copyrighttag_celeba.yaml --model_config configs/cfkd_experiments/predictors/celeba8k_Smiling_confounding_copyrighttag_classifier_poisoned098_dfr.yaml
python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/celeba8k_smiling_confounding_copyrighttag_098_rrclarc.yaml"
cat ${PEAL_RUNS}/celeba8k_copyrighttag/Smiling_confounding_copyrighttag/regularized0/classifier_poisoned098/rrclarc/best_model_result.txt



# Table 3 (tab:model_comparison), the UNI-F and UNI-L columns
# Analysis of the influence of the student architecture
# For the Camelyon dataset
# UNI-F, the fully fine-tuned readout (only_last_layer: False).
# The config is named dinov2 but declares architecture: torchvision_UNI, and it writes to
# $PEAL_RUNS/dae/UNI rather than the camelyon17_1k path the evaluate line below reads.
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_finetuned_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_dino_v2_finetuned_poisoned098_osce_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_finetuned_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/dinov2_finetuned_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_finetuned_poisoned098_dfr.yaml

# UNI-L, the linear readout (only_last_layer: True).
# Also named dinov2 while declaring architecture: torchvision_UNI.
# These two comments were swapped before: the labels now follow only_last_layer.
python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_linear_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_dino_v2_linear_poisoned098_osce_cfkd.yaml"
python train_predictor.py --config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_linear_poisoned098_dfr.yaml
python evaluate_predictor.py --model_path $PEAL_RUNS/camelyon17_1k/dinov2_linear_poisoned098/dfr/model.cpl --data_config configs/cfkd_experiments/data/camelyon17_unpoisoned.yaml --model_config configs/cfkd_experiments/predictors/camelyon17_1k_dinov2_linear_poisoned098_dfr.yaml



# Figure 4 (fig:teacher_comparison), Square at 80 % poisoning = correlation 0.6
# Analysis of the influence of the teacher
# For the Square dataset for poisoning 80% (corresponds to 0.6 correlation)
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_false_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_mask_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_human_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x080_sce_noisy_cfkd.yaml"


# Table 4 (tab:explainers_merged), columns Square, Smiling and Camelyon
# Analysis of the influence of the Counterfactual Explainer
# For the Square dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_fastdime_cfkd.yaml"

# For Smiling confounding Copyrighttag
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_fastdime_cfkd.yaml"

# For the Camelyon17 dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_poisoned098_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_poisoned098_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/camelyon17_1k_poisoned098_fastdime_cfkd.yaml"


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

# ===========================================================================
# Table 2 (tab:statistical_analysis), the three independent CFKD runs.
# INCOMPLETE: Table 2 reports Square, Smiling, Blond, Camelyon, FunnyNodules and Nico++
# (Follicles is omitted there because its teacher was human). The loop below covers Square
# only, so it reproduces one column of six.
# Variance bars: seeds 1-3.
#
# Each seed runs in its own $PEAL_RUNS tree, so the seed-0 runs above are never
# touched and every $PEAL_RUNS/... reference inside the configs resolves within
# that tree. The base models are rebuilt there with the --seed override, so no
# seeded copies of the configs are needed. collect_results.py below aggregates
# runs that share a label as mean +- population std across these trees.
#
# The generators stay at seed 0 and are linked into each seed tree rather than
# retrained: the generator stands in for a large-scale pretrained model, the
# dataset split barely moves it, and rebuilding it per seed costs far more
# compute than the variance it would expose.

PEAL_RUNS_SEED0="${PEAL_RUNS}"     # buffer the caller's root, restored at the end

for SEED in 1 2 3; do
  export PEAL_RUNS="${PEAL_RUNS_SEED0}${SEED}"
  mkdir -p "$PEAL_RUNS"

  # generator: reuse seed 0 (see the note above)
  mkdir -p "$PEAL_RUNS/square1k"
  ln -sfn "$PEAL_RUNS_SEED0/square1k/ddpm" "$PEAL_RUNS/square1k/ddpm"

  # -- Square ----------------------------------------------------------------
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square_classifier_unpoisoned.yaml" --seed $SEED
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098.yaml" --seed $SEED
  python run_cfkd.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/square1000x098_sce_cfkd.yaml" --seed $SEED
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098_diffusion_augmented.yaml" --seed $SEED
  python train_predictor.py --config "<PEAL_BASE>/configs/cfkd_experiments/predictors/square1k_classifier_poisoned098_dfr.yaml" --seed $SEED
  python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/group_dro/square_1k_poisoned098_group_dro.yaml" --seed $SEED
  python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_pclarc.yaml" --seed $SEED
  python run_adaptor.py --config "<PEAL_BASE>/configs/cfkd_experiments/adaptors/clarc/square1000_poisoned098_rrclarc.yaml" --seed $SEED
done

export PEAL_RUNS="${PEAL_RUNS_SEED0}"


python reproduction_scripts/collect_results.py \
    --runs "$PEAL_RUNS" --discover \
    --out results/cfkd_results.json \
    --latex results/cfkd_main_table.tex

# To restrict the collection to the trees behind one table, pass them
# explicitly instead of --discover; runs that share a label are aggregated as
# mean +- population std over seeds:
#   python reproduction_scripts/collect_results.py --runs "$PEAL_RUNS" \
#       --tree "LABEL=relative/path/under/PEAL_RUNS" [--tree ...] \
#       --out results/cfkd_main_table.json --latex results/cfkd_main_table.tex
#
# results/cfkd_main_table.tex is the body the table \input{}s; the column order
# is flip_rate, diversity, sparsity, non-adversarial rate, unbiasedness,
# counterfactuals per second, gain.
