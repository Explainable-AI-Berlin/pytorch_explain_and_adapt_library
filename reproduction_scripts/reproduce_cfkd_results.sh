# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit TODO


# Reproduce SOTA results on the square dataset
python train_generator.py --config "<PEAL_BASE>/configs/generators/square_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square_classifier_poisoned098.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square_classifier_unpoisoned.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR


# Reproduce SOTA results on CelebA Blond_Hair confounding Male task
python train_generator.py --config "<PEAL_BASE>/configs/generators/celeba_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_Blond_Hair_classifier_poisoned100.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_confounding_Male_poisoned100_pdc_cluster_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR


# Reproduce SOTA results on camelyon17 dataset
python train_generator.py --config "<PEAL_BASE>/configs/generators/camelyon17_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/camelyon17_classifier_poisoned100.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/camelyon17_classifier_unpoisoned.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/camelyon17_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR


# Reproduce SOTA results on follicle dataset
python train_generator.py --config "<PEAL_BASE>/configs/generators/follicle_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/follicle_cut_classifier.yaml"
python run_cfkd.py --config "<PEAL_BASE>/confiegs/adaptors/follicles_pdc_human_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR


# Reproduce SOTA results on CelebA copyrighttag dataset (the results over different poisoning levels will be averaged)
python train_generator.py --config "<PEAL_BASE>/configs/generators/celeba_copyrighttag_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_copyrighttag_unpoisoned.yaml"
# For 90% poisoning
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_copyrighttag_poisoned090.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x090_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR
# For 92% poisoning
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_copyrighttag_poisoned092.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x092_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR
# For 94% poisoning
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_copyrighttag_poisoned094.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x094_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR
# For 96% poisoning
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_copyrighttag_poisoned096.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x096_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR
# For 98% poisoning
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_copyrighttag_poisoned098.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x098_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR
# For 100% poisoning
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_copyrighttag_poisoned100.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_pdc_cfkd.yaml"
# run GroupDRO
# run JTT
# run DFR


# Analysis of the influence of the teacher
# For the Square dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pdc_false_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pdc_true_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pdc_random_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pdc_mask_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pdc_cluster_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pdc_human_cfkd.yaml"
# For Smiling confounding Copyrighttag
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x094_pdc_false_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x094_pdc_true_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x094_pdc_random_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x094_pdc_mask_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x094_pdc_cluster_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x094_pdc_human_cfkd.yaml"
# For the follicles dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/follicles_pdc_false_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/follicles_pdc_true_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/follicles_pdc_random_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/follicles_pdc_cluster_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/follicles_pdc_mask_cfkd.yaml"


# Analysis of influence of sample number
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square250x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square500x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square2000x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square4000x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba250x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba500x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba2000x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba4000x100_pfc_cfkd.yaml"


# Analysis of the influence of the Counterfactual Explainer
# For the Square dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x098_fastdime_cfkd.yaml"
# For Smiling confounding Copyrighttag
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_pfc_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_fastdime_cfkd.yaml"
# For the Camelyon17 dataset
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/camelyon17_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/camelyon17_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/camelyon17_fastdime_cfkd.yaml"