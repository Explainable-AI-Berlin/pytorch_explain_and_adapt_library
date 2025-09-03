# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit XXX

# Reproduce the results on CelebA

# train the generator that is used for the counterfactual explainer (the dataset will be downloaded automatically)
python train_generator.py --config "<PEAL_BASE>/configs/sce_experiments/generators/celeba_ddpm.yaml"

# the Oracle for estimating the latent space
# could be alternative downloaded from https://huggingface.co/guillaumejs2403/DiME
# and place under "pretrained_models/oracle.pth"
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/celeba_latent_oracle.yaml"

# Smiling
# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/celeba_Smiling_classifier.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Smiling_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/ace_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/ace_cfkd/logs

# get the explanations for dime
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Smiling_natural_dime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/dime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/dime_cfkd/logs

# get the explanations for fastdime
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Smiling_natural_fastdime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/fastdime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/fastdime_cfkd/logs

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Smiling_natural_sce_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/sce_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/sce_cfkd/logs

# Blond_Hair
# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/celeba_Blond_Hair_classifier.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Blond_Hair_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/ace_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/ace_cfkd/logs

# get the explanations for dime
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Blond_Hair_natural_dime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/dime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/dime_cfkd/logs

# get the explanations for fastdime
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Blond_Hair_natural_fastdime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/fastdime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/fastdime_cfkd/logs

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/celeba_Blond_Hair_natural_sce_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/sce_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/sce_cfkd/logs


# Reproduce the results on the square dataset

# train the generator that is used for the counterfactual explainer (the dataset will be generated automatically)
python train_generator.py --config "<PEAL_BASE>/configs/sce_experiments/generators/square_ddpm.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/square_classifier_poisoned100.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/square_classifier_unpoisoned.yaml"

# Run the different counterfactual explainers
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/square1000x100_sce_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/square1000x100_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/square1000x100_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/square1000x100_fastdime_cfkd.yaml"

# For Smiling confounding Copyrighttag
python train_generator.py --config "<PEAL_BASE>/configs/sce_experiments/generators/celeba_copyrighttag_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/celeba_Smiling_confounding_copyrighttag_classifier_poisoned100.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/celeba_copyrighttag_unpoisoned.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_sce_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_sce_no_sparsity_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_sce_no_smoothing_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_sce_no_gradient_filtering_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_sce_no_exploration_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_fastdime_cfkd.yaml"

# vit-b-16 experiments
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/celeba_Smiling_confounding_copyrighttag_vit_b_16_poisoned100.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_vit_b_16_sce_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_vit_b_16_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_vit_b_16_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/Smiling_confounding_CopyrightTag_celeba1000x100_vit_b_16_fastdime_cfkd.yaml"

# For the Camelyon17 dataset
python train_generator.py --config "<PEAL_BASE>/configs/sce_experiments/generators/camelyon17_ddpm.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/camelyon17_classifier_poisoned100.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/camelyon17_classifier_unpoisoned.yaml"
python train_predictor.py --config "<PEAL_BASE>/configs/sce_experiments/predictors/camelyon17_latent_oracle.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/camelyon17_sce_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/camelyon17_ace_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/camelyon17_dime_cfkd.yaml"
python run_cfkd.py --config "<PEAL_BASE>/configs/sce_experiments/adaptors/camelyon17_fastdime_cfkd.yaml"
