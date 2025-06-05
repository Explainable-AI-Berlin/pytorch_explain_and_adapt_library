# This script is meant to be able to reproduce the results of the PDC paper (ARXIV_LINK).
# The results were reproduced with the following software versions: (GIT_HASH)
# the batch sizes are optimized for a GPU with 80gb VRAM, but can be decreased for smaller GPUs
# this script was executed at commit 94c3dc8

# Reproduce the results on the square dataset

# train the generator that is used for the counterfactual explainer (the dataset will be generated automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/square_ddpm.yaml"

# train the predictor that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/square_classifier_poisoned100.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_ace_pe_only_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/ace_pe_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/ace_pe_cfkd/0/validation_collages0

# get the explanations for non-adversarial PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_sce_nonadversarial_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_nonadversarial_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_nonadversarial_cfkd/0/validation_collages0

# get the explanations for sensitive PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_sce_sensitive_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sensitive_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sensitive_cfkd/0/validation_collages0

# get the explanations for sparse PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_sce_sparse_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sparse_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_sparse_cfkd/0/validation_collages0

# get the explanations for diverse PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_sce_diverse_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_diverse_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_diverse_cfkd/0/validation_collages0

# get the explanations for unbiased PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_sce_unbiased_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_unbiased_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_unbiased_cfkd/0/validation_collages0_0
# square unbiased is also used for reporting the metrics from the experiments
# metrics: tensorboard --logdir $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/pdc_unbiased_cfkd/logs

# get the explanations for unbiased PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_ace_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/ace_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/ace_cfkd/0/validation_collages0_0
# square unbiased is also used for reporting the metrics from the experiments
# metrics: tensorboard --logdir $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/ace_cfkd/logs

# get the explanations for unbiased PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_dime_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/dime_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/dime_cfkd/0/validation_collages0_0
# square unbiased is also used for reporting the metrics from the experiments
# metrics: tensorboard --logdir $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/dime_cfkd/logs

# get the explanations for unbiased PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/square1000x100_fastdime_cfkd.yaml"
# you can the the global visualization in
# $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/fastdime_cfkd/0/val_counterfactuals_global.png
# you can see the collages in $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/fastdime_cfkd/0/validation_collages0_0
# square unbiased is also used for reporting the metrics from the experiments
# metrics: tensorboard --logdir $PEAL_BASE/square3/colora_confounding_colorb/torchvision/classifier_poisoned100/fastdime_cfkd/logs



# Reproduce the results on CelebA

# train the generator that is used for the counterfactual explainer (the dataset will be downloaded automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/celeba_ddpm.yaml"

# Smiling
# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_classifier_smiling.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Smiling_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/ace_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/ace_cfkd/logs

# get the explanations for dime
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Smiling_natural_dime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/dime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/dime_cfkd/logs

# get the explanations for fastdime
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Smiling_natural_fastdime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/fastdime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/fastdime_cfkd/logs

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Smiling_natural_sce_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Smiling/classifier_natural/pdc_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Smiling/classifier_natural/pdc_cfkd/logs

# Blond_Hair
# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_classifier_blond_hair.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/ace_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/ace_cfkd/logs

# get the explanations for dime
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_natural_dime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/dime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/dime_cfkd/logs

# get the explanations for fastdime
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_natural_fastdime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/fastdime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/fastdime_cfkd/logs

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_natural_sce_cfkd.yaml"
# you can see the collages in $PEAL_BASE/celeba/Blond_Hair/classifier_natural/pdc_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/celeba/Blond_Hair/classifier_natural/pdc_cfkd/logs

# Reproduce the results on Waterbirds

# train the generator that is used for the counterfactual explainer (the dataset will be downloaded automatically)
python train_generator.py --config "<PEAL_BASE>/configs/generators/waterbirds_ddpm.yaml"

# Train the classifier that shall be analyzed
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/waterbirds_classifier.yaml"

# get the explanations for ACE
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/waterbirds_natural_ace_cfkd.yaml"
# you can see the collages in $PEAL_BASE/waterbirds/classifier_natural/ace_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/waterbirds/classifier_natural/ace_cfkd/logs

# get the explanations for dime
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/waterbirds_natural_dime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/waterbirds/classifier_natural/dime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/waterbirds/classifier_natural/dime_cfkd/logs

# get the explanations for fastdime
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/waterbirds_natural_fastdime_cfkd.yaml"
# you can see the collages in $PEAL_BASE/waterbirds/classifier_natural/fastdime_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/waterbirds/classifier_natural/fastdime_cfkd/logs

# get the explanations for PDC
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/waterbirds_natural_sce_cfkd.yaml"
# you can see the collages in $PEAL_BASE/waterbirds/classifier_natural/pdc_cfkd/0/validation_collages0
# metrics: tensorboard --logdir $PEAL_BASE/waterbirds/classifier_natural/pdc_cfkd/logs


# Reproduction of results on Resnet50
#
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_Blond_Hair_resnet50.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_resnet50_ace_cfkd.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_resnet50_dime_cfkd.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_resnet50_fastdime_cfkd.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_resnet50_sce_cfkd.yaml"


# Reproduction of results on VitB16
#
python train_predictor.py --config "<PEAL_BASE>/configs/predictors/celeba_Blond_Hair_vit_b_16.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_vit_b_16_ace_cfkd.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_vit_b_16_dime_cfkd.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_vit_b_16_fastdime_cfkd.yaml"

#
python run_cfkd.py --config "<PEAL_BASE>/configs/adaptors/celeba_Blond_Hair_vit_b_16_sce_cfkd.yaml"