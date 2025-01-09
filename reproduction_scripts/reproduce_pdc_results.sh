# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs

# Reproduce the results on the square dataset

# train the generator that is used for the counterfactual explainer (the dataset will be generated automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/square_ddpm.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square3_classifier_poisoned100.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_ace_cfkd.yaml"
# you can the the global visualization in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/ace_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/ace_cfkd/0/validation_collages0

# get the explanations for non-adversarial PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pdc_nonadversarial_cfkd.yaml"
# you can the the global visualization in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_nonadversarial_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_nonadversarial_cfkd/0/validation_collages0

# get the explanations for sensitive PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pdc_sensitive_cfkd.yaml"
# you can the the global visualization in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sensitive_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sensitive_cfkd/0/validation_collages0

# get the explanations for sparse PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pdc_sparse_cfkd.yaml"
# you can the the global visualization in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sparse_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sparse_cfkd/0/validation_collages0

# get the explanations for diverse PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pdc_diverse_cfkd.yaml"
# you can the the global visualization in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_diverse_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_diverse_cfkd/0/validation_collages0

# get the explanations for unbiased PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_pdc_unbiased_cfkd.yaml"
# you can the the global visualization in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_unbiased_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_unbiased_cfkd/0/validation_collages0_0


# Reproduce the results on CelebA

# train the generator that is used for the counterfactual explainer (the dataset will be downloaded automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/celeba_ddpm.yaml"

# Smiling
# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_classifier_smiling.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Smiling_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/ace_cfkd/0/validation_collages0

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Smiling_natural_pdc_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/pdc_cfkd/0/validation_collages0

# Blond_Hair
# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_classifier_blond_hair.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/ace_cfkd/0/validation_collages0

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_natural_pdc_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/pdc_cfkd/0/validation_collages0

# Reproduce the results on Waterbirds

# train the generator that is used for the counterfactual explainer (the dataset will be downloaded automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/waterbirds_ddpm.yaml"

# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/waterbirds_classifier.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/waterbirds_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/waterbirds/classifier_natural/ace_cfkd/0/validation_collages0

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/waterbirds_natural_pdc_cfkd.yaml"
# you can see the collages in $PEAL_BASE/waterbirds/classifier_natural/pdc_cfkd/0/validation_collages0

