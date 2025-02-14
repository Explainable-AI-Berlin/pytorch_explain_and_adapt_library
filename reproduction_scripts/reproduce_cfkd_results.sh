# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit TODO


# Reproduce the results on the square dataset

# train the generator that is used for the counterfactual explainer (the dataset will be generated automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/square_ddpm.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square3_classifier_poisoned100.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pdc_unbiased_cfkd.yaml"


# Reproduce results on CelebA copyrighttag dataset

# train the generator
python train_generator.py --config "<PEAL_BASE>/configs/generators/celeba_copyrighttag_ddpm.yaml"

# Run CFKD for 90% poisoning
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba500x090_pdc_cfkd_mask.yaml"

# Run CFKD for 100% poisoning
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/Smiling_confounding_CopyrightTag_celeba500x100_pdc_cfkd_mask.yaml"


# Reproduce the results on the square dataset

# train the generator
python train_generator.py --config "<PEAL_BASE>/configs/generators/square_ddpm.yaml"


# Reproduce results on Waterbirds dataset

# train the generator that is used for the counterfactual explainer
python train_generator.py --config "<PEAL_BASE>/configs/generators/waterbirds_ddpm.yaml"


# Reproduce results on follicle dataset

# train the generator that is used for the counterfactual explainer (the dataset will be generated automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/follicle_ddpm.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/follicle_cut_classifier.yaml"


# Analysis of influence of sample number
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square250x100_perfect_false_counterfactuals_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square500x100_perfect_false_counterfactuals_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_perfect_false_counterfactuals_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square2000x100_perfect_false_counterfactuals_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square4000x100_perfect_false_counterfactuals_cfkd.yaml"