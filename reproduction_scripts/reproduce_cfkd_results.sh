# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit 94c3dc8

# Reproduce the results on the square dataset

# train the generator that is used for the counterfactual explainer (the dataset will be generated automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/square_ddpm.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square3_classifier_poisoned100.yaml"

# train the teacher for CFKD
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square_classifier.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pdc_unbiased_cfkd.yaml"


# train the generator that is used for the counterfactual explainer (the dataset will be generated automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/follicle_ddpm.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/follicle_cut_classifier.yaml"